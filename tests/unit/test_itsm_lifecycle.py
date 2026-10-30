"""In-process ITSM episode lifecycle, scoring retries, budget and isolation (step 5).

Real Bench2 tools, seeds and DB scoring; deterministic user stand-in; faults are
injected with monkeypatch. No LLM is called (the llm judge is a mock).
"""

from __future__ import annotations

import copy
import functools
import json
import threading
import time
from typing import Any

import pytest

from enterprise_worlds.data_model.message import AssistantMessage, ToolCall, ToolMessage
from enterprise_worlds.domains.itsm import environment as itsm_environment
from enterprise_worlds.domains.itsm.tools import ItsmTools
from enterprise_worlds.evaluator import text_match_strategy
from enterprise_worlds.evaluator.text_match_strategy import TextMatchConfig
from enterprise_worlds.orchestrator.orchestrator import Orchestrator
from enterprise_worlds.user.base import STOP
from enterprise_worlds.utils.clock import get_now, reset_now, set_now

from itsm_test_support import ScriptedUser, seed_file_digests

import agent_rl.envs.itsm_env as itsm_env_module
from agent_rl.envs.itsm_env import ItsmAction, ItsmEnv, ItsmEnvConfig
from agent_rl.envs.itsm_scoring import (
    ItsmInfrastructureError,
    build_task_environment,
    load_itsm_tasks,
    task_clock,
)


TASKS = load_itsm_tasks()
FUZZY = TextMatchConfig(strategy="fuzzy")
LLM_JUDGE = TextMatchConfig(strategy="llm", llm="mock/judge", llm_args={"temperature": 0.0})
MSP_021 = "itsmbench_msp_021"
SINGLE_017 = "itsmbench_single_017"
OPENING = "Hi, I need help with a ticket."
CLOSING_REPLY = "Great, thanks."
#: Explicit small test budget; the formal budget is fixed in step 6.
TEST_MAX_STEPS = 40
SENTINEL_TIME = "2030-01-01T00:00:00"
SENTINEL_STAMP = "2030-01-01 00:00:00"


def _env(task_id, replies, *, max_steps=TEST_MAX_STEPS, text_match=FUZZY, episode_id=None, **config):
    user = ScriptedUser(replies)
    cfg = ItsmEnvConfig(task_id=task_id, text_match=text_match, max_steps=max_steps, **config)
    return ItsmEnv(cfg, user=user, episode_id=episode_id or f"{task_id}:life"), user


def _gold(task_id: str) -> list[ItsmAction]:
    return [
        ItsmAction.tool_call(action.name, copy.deepcopy(action.arguments))
        for action in TASKS[task_id].evaluation_criteria.actions
    ]


def _completed(actions: list[ItsmAction]) -> list[ItsmAction]:
    return [*actions, ItsmAction.message("All done, summary sent."), ItsmAction.done()]


def _play(env: ItsmEnv, actions: list[ItsmAction]):
    transitions = []
    for action in actions:
        transitions.append(env.step(action))
        if env.done:
            break
    return transitions


def _fail_next_env_builds(monkeypatch, count: int) -> dict[str, int]:
    """The next ``count`` Bench2 env builds raise; later builds work. Install after reset."""
    original = itsm_environment.get_environment
    state = {"left": count, "calls": 0}

    def get_environment(*args, **kwargs):
        state["calls"] += 1
        if state["left"] > 0:
            state["left"] -= 1
            raise OSError("seed file unreadable")
        return original(*args, **kwargs)

    monkeypatch.setattr(itsm_environment, "get_environment", get_environment)
    return state


def _forbid_execution(monkeypatch, env: ItsmEnv) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("re-scoring must not run tools or the user simulator")

    monkeypatch.setattr(env, "_execute", forbidden)
    monkeypatch.setattr(env, "_user_turn", forbidden)


def _ended_with_scoring_failure(monkeypatch, failures: int):
    env, user = _env(MSP_021, [OPENING, CLOSING_REPLY])
    env.reset()
    for action in _completed(_gold(MSP_021))[:-1]:
        env.step(action)
    builds = _fail_next_env_builds(monkeypatch, failures)
    with pytest.raises(ItsmInfrastructureError) as raised:
        env.step(ItsmAction.done())
    return env, user, builds, raised.value


# -- scoring states -------------------------------------------------------------------------------


def test_scoring_failure_keeps_the_ended_interaction_and_rescores_it(monkeypatch) -> None:
    env, user, _, error = _ended_with_scoring_failure(monkeypatch, 1)

    assert (error.stage, error.kind) == ("score_env_build", "gold")
    assert env.status == "scoring_retry_pending"
    assert env.done and env.failure is None
    saved = (env.trajectory, env.tool_records, list(user.received), env.current_db_hash())
    for call in (
        lambda: env.step(ItsmAction.message("Anything else?")),
        lambda: env.step(ItsmAction.done()),
        lambda: env.truncate("response_length"),
    ):
        with pytest.raises(RuntimeError, match="already finished"):
            call()

    _forbid_execution(monkeypatch, env)
    transition = env.score()

    info = transition.reward_info
    assert (transition.terminated, transition.termination_reason) == (True, "agent_done")
    assert (info.db_match, info.training_reward, info.scoring_attempts) == (True, 1.0, 2)
    assert env.status == "scored"
    assert saved == (env.trajectory, env.tool_records, list(user.received), env.current_db_hash())
    assert [m for m in env.trajectory if getattr(m, "content", None) == STOP] == [
        AssistantMessage(content=STOP)
    ]
    assert [e["stage"] for e in env.scoring_errors] == ["score_env_build"]


@pytest.mark.parametrize(("drop_sla", "expected"), [(False, 1.0), (True, 0.0)], ids=["db1", "db0"])
def test_a_valid_score_is_kept_and_never_recomputed(monkeypatch, drop_sla: bool, expected: float) -> None:
    calls: list[int] = []
    original = itsm_env_module.score_final_db

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(itsm_env_module, "score_final_db", counting)
    actions = _gold(MSP_021)
    if drop_sla:
        del actions[3]
    env, _ = _env(MSP_021, [OPENING, CLOSING_REPLY])
    env.reset()

    final = _play(env, _completed(actions))[-1]
    again = [env.score() for _ in range(3)]

    assert final.reward_info.db_reward == expected
    assert len(calls) == 1
    assert all(t == final and t.reward_info is final.reward_info for t in again)


def test_scoring_retries_are_bounded_to_four_attempts(monkeypatch) -> None:
    env, _, builds, _ = _ended_with_scoring_failure(monkeypatch, 10)
    _forbid_execution(monkeypatch, env)

    for _ in range(3):
        with pytest.raises(ItsmInfrastructureError):
            env.score()

    assert env.status == "failed"
    assert env.failure["stage"] == "score_env_build"
    assert (env.failure["reason"], env.failure["attempts"]) == ("scoring_attempts_exhausted", 4)
    assert builds["calls"] == 4
    with pytest.raises(RuntimeError, match="cannot score"):
        env.score()
    assert builds["calls"] == 4
    trace = env.export_trace()
    assert trace["reward_info"] is None
    assert [e["attempt"] for e in trace["scoring_errors"]] == [1, 2, 3, 4]


def test_judge_protocol_error_is_rescored_on_the_same_final_db(monkeypatch) -> None:
    requests: list[dict[str, Any]] = []

    def generate(**kwargs):
        requests.append(kwargs)
        if len(requests) == 1:
            return AssistantMessage(content="not json")
        pairs = json.loads(kwargs["messages"][1].content.removeprefix("pairs:\n"))
        results = [{"index": pair["index"], "equivalent": True} for pair in pairs]
        return AssistantMessage(content=json.dumps({"results": results}))

    monkeypatch.setattr(text_match_strategy, "generate", generate)
    env, _ = _env(MSP_021, [OPENING, CLOSING_REPLY], text_match=LLM_JUDGE)
    env.reset()
    actions = _completed(_gold(MSP_021))
    for action in actions[:-1]:
        env.step(action)

    with pytest.raises(ItsmInfrastructureError) as raised:
        env.step(actions[-1])
    final = env.score()

    assert (raised.value.stage, raised.value.kind) == ("text_judge_protocol", "unparseable")
    assert final.reward_info.db_match and final.reward_info.scoring_attempts == 2
    assert len(requests) == 2 and requests[0] == requests[1]


def test_score_on_finish_false_separates_ending_from_scoring() -> None:
    env, _ = _env(MSP_021, [OPENING, CLOSING_REPLY], score_on_finish=False)
    env.reset()

    ended = _play(env, _completed(_gold(MSP_021)))[-1]

    assert (ended.terminated, ended.termination_reason, ended.reward_info) == (True, "agent_done", None)
    assert env.status == "interaction_ended"
    scored = env.score()
    assert scored.reward_info.db_match and env.status == "scored"


def test_rescoring_refuses_a_final_db_changed_after_the_end() -> None:
    env, _ = _env(MSP_021, [OPENING, CLOSING_REPLY], score_on_finish=False)
    env.reset()
    _play(env, _completed(_gold(MSP_021)))
    with task_clock(env.task):
        env._environment.make_tool_call("update_incident", incident_id="INC_054", status="in_progress")

    with pytest.raises(RuntimeError, match="final DB changed"):
        env.score()


# -- lifecycle ------------------------------------------------------------------------------------


def test_lifecycle_new_allows_only_reset_and_close() -> None:
    env, _ = _env(SINGLE_017, [OPENING])

    for call in (lambda: env.step(ItsmAction.done()), lambda: env.truncate("response_length")):
        with pytest.raises(RuntimeError, match="reset"):
            call()
    with pytest.raises(RuntimeError, match="cannot score"):
        env.score()
    assert env.status == "new"
    env.close()
    with pytest.raises(RuntimeError, match="closed"):
        env.reset()


def test_lifecycle_active_rejects_a_second_reset_and_scoring() -> None:
    env, _ = _env(SINGLE_017, [OPENING])
    env.reset()

    with pytest.raises(RuntimeError, match="exactly one episode"):
        env.reset()
    with pytest.raises(RuntimeError, match="cannot score"):
        env.score()
    assert env.status == "active"


def test_lifecycle_scored_rejects_business_actions() -> None:
    env, _ = _env(MSP_021, [OPENING, CLOSING_REPLY])
    env.reset()
    final = _play(env, _completed(_gold(MSP_021)))[-1]

    for call in (
        lambda: env.step(ItsmAction.message("One more thing.")),
        lambda: env.step(ItsmAction.done()),
        lambda: env.truncate("invalid_action"),
    ):
        with pytest.raises(RuntimeError, match="already finished"):
            call()
    with pytest.raises(RuntimeError, match="exactly one episode"):
        env.reset()
    assert env.score() == final


def test_lifecycle_failed_rejects_everything_but_export_and_close() -> None:
    env, _ = _env(SINGLE_017, [OPENING])
    env.reset()
    with pytest.raises(ItsmInfrastructureError):
        env.step(ItsmAction.message("Which incident?"))  # no scripted reply left

    assert (env.status, env.failure["stage"]) == ("failed", "user_sim")
    for call in (lambda: env.step(ItsmAction.done()), lambda: env.truncate("response_length")):
        with pytest.raises(RuntimeError, match="infrastructure"):
            call()
    with pytest.raises(RuntimeError, match="cannot score"):
        env.score()
    with pytest.raises(RuntimeError, match="exactly one episode"):
        env.reset()
    assert env.export_trace()["reward_info"] is None
    env.close()
    env.close()


def test_close_is_idempotent_and_blocks_every_call() -> None:
    env, _ = _env(MSP_021, [OPENING, CLOSING_REPLY])
    env.reset()
    _play(env, _completed(_gold(MSP_021)))
    trace = env.export_trace()

    env.close()
    env.close()

    assert env.closed
    for call in (
        env.reset,
        lambda: env.step(ItsmAction.done()),
        lambda: env.truncate("response_length"),
        env.score,
        env.current_db_hash,
    ):
        with pytest.raises(RuntimeError, match="closed"):
            call()
    after = env.export_trace()
    assert (after["messages"], after["reward_info"], after["closed"]) == (
        trace["messages"],
        trace["reward_info"],
        True,
    )


# -- task clock restored on every path --------------------------------------------------------------


def _clock_normal(monkeypatch) -> None:
    env, _ = _env(MSP_021, [OPENING, CLOSING_REPLY])
    env.reset()
    # msp_021 is clock-sensitive: DB=1 needs the task time inside every tool call.
    assert _play(env, _completed(_gold(MSP_021)))[-1].reward_info.db_match


def _clock_domain_error(monkeypatch) -> None:
    env, _ = _env(SINGLE_017, [OPENING])
    env.reset()
    action = ItsmAction.tool_call("find_incident_by_number", {"number": "INC9999999"})
    assert env.step(action).tool_record.error_class == "domain_error"


def _clock_tool_internal(monkeypatch) -> None:
    def broken(self, number: str):
        raise KeyError("index corrupted")

    monkeypatch.setattr(ItsmTools, "find_incident_by_number", broken)
    env, _ = _env(SINGLE_017, [OPENING])
    env.reset()
    with pytest.raises(ItsmInfrastructureError):
        env.step(ItsmAction.tool_call("find_incident_by_number", {"number": "INC0000007"}))


def _clock_scoring_failure(monkeypatch) -> None:
    _ended_with_scoring_failure(monkeypatch, 1)


def _clock_step_budget(monkeypatch) -> None:
    env, _ = _env(MSP_021, [OPENING], max_steps=3)
    env.reset()
    assert _play(env, _gold(MSP_021))[-1].termination_reason == "max_steps"


def _clock_user_failure(monkeypatch) -> None:
    env, _ = _env(SINGLE_017, [OPENING])
    env.reset()
    with pytest.raises(ItsmInfrastructureError):
        env.step(ItsmAction.message("Which incident?"))


CLOCK_SCENARIOS = {
    "normal": _clock_normal,
    "domain_error": _clock_domain_error,
    "tool_internal": _clock_tool_internal,
    "scoring_failure": _clock_scoring_failure,
    "step_budget": _clock_step_budget,
    "user_failure": _clock_user_failure,
}


@pytest.mark.parametrize("scenario", sorted(CLOCK_SCENARIOS))
def test_task_clock_is_restored_on_every_path(monkeypatch, scenario: str) -> None:
    set_now(SENTINEL_TIME)
    try:
        CLOCK_SCENARIOS[scenario](monkeypatch)
        assert get_now() == SENTINEL_STAMP
    finally:
        reset_now()


# -- budget and termination -------------------------------------------------------------------------


class ScriptedOrchestratorAgent:
    def __init__(self, actions: list[ItsmAction]) -> None:
        self.messages = []
        for index, action in enumerate(actions):
            if action.kind == "tool_call":
                call = ToolCall(id=f"orc-{index}", name=action.name, arguments=copy.deepcopy(action.arguments))
                self.messages.append(AssistantMessage(tool_calls=[call]))
            elif action.kind == "message":
                self.messages.append(AssistantMessage(content=action.text))
            else:
                self.messages.append(AssistantMessage(content=STOP))

    def get_init_state(self) -> int:
        return 0

    def generate_next_message(self, message, state: int):
        return self.messages[state], state + 1


def _normalized(trajectory) -> list[tuple[Any, ...]]:
    return [
        (
            message.role,
            message.content,
            [(c.name, c.arguments) for c in message.tool_calls] if getattr(message, "tool_calls", None) else None,
            getattr(message, "error", None),
        )
        for message in trajectory
    ]


def _budget_actions() -> list[ItsmAction]:
    gold = _gold(MSP_021)
    return [gold[0], gold[1], ItsmAction.message("Let me check the SLA."), gold[2], gold[3], ItsmAction.done()]


@pytest.mark.parametrize(
    ("max_steps", "reason", "executed"),
    [(1, "max_steps", 0), (2, "max_steps", 1), (3, "max_steps", 2), (4, "max_steps", 2), (20, "agent_done", 4)],
)
def test_step_budget_matches_bench2_orchestrator(max_steps: int, reason: str, executed: int) -> None:
    actions = _budget_actions()
    replies = [OPENING, "Sure."]
    env, user = _env(MSP_021, replies, max_steps=max_steps)
    env.reset()
    final = _play(env, actions)[-1]

    task = TASKS[MSP_021]
    bench2_user = ScriptedUser(replies)
    with task_clock(task):
        bench2_env = build_task_environment(task)
        run = Orchestrator(ScriptedOrchestratorAgent(actions), bench2_user, bench2_env, max_steps=max_steps).run()

    assert final.termination_reason == reason
    assert (final.terminated, final.truncated) == (reason == "agent_done", reason == "max_steps")
    assert run.stopped == (reason == "agent_done")
    assert _normalized(env.trajectory) == _normalized(run.trajectory)
    assert user.received == bench2_user.received
    assert env.current_db_hash() == bench2_env.get_db_hash()
    assert sum(record.result_received for record in env.tool_records) == executed


def test_tool_call_at_the_step_limit_is_recorded_but_not_executed() -> None:
    env, _ = _env(MSP_021, [OPENING], max_steps=2)
    env.reset()
    gold = _gold(MSP_021)
    env.step(gold[0])
    before = env.current_db_hash()

    final = env.step(gold[1])  # update_incident, a write

    assert (final.terminated, final.truncated, final.termination_reason) == (False, True, "max_steps")
    record = final.tool_record
    assert (record.call_id, record.name, record.result_received, record.result, record.error) == (
        "call_002",
        "update_incident",
        False,
        None,
        None,
    )
    assert record.to_tool_call_record().result_received is False
    assert env.current_db_hash() == before
    assert env.trajectory[-1].tool_calls[0].id == "call_002"
    assert not [m for m in env.trajectory if isinstance(m, ToolMessage) and m.id == "call_002"]
    assert (final.reward_info.db_match, final.reward_info.training_reward) == (False, 0.0)


def test_done_at_the_step_limit_is_agent_done() -> None:
    env, _ = _env(MSP_021, [OPENING], max_steps=1)
    env.reset()

    final = env.step(ItsmAction.done())

    assert (final.terminated, final.truncated, final.termination_reason) == (True, False, "agent_done")


def test_agent_turns_and_underlying_steps_are_counted_separately() -> None:
    env, _ = _env(SINGLE_017, [OPENING, "It is INC0000007."])
    reset = env.reset()

    transitions = [
        env.step(ItsmAction.tool_call("find_incident_by_number", {"number": "INC9999999"})),
        env.step(ItsmAction.message("Which incident?")),
        env.step(ItsmAction.done()),
    ]

    # Steps count user replies and executed tool calls (reset's first user turn is
    # step 1); agent turns count policy actions, including done.
    assert reset.step_count == 1
    assert [(t.agent_turns, t.step_count) for t in transitions] == [(1, 2), (2, 3), (3, 3)]
    trace = env.export_trace()
    assert (trace["agent_turns"], trace["step_count"]) == (3, 3)


@pytest.mark.parametrize(
    ("reason", "terminated", "truncated", "training", "source"),
    [
        ("response_length", False, True, 1.0, "db_only"),
        ("invalid_action", True, False, 0.0, "invalid_action"),
    ],
)
def test_external_end_reasons(reason, terminated, truncated, training, source) -> None:
    env, _ = _env(MSP_021, [OPENING])
    env.reset()
    for action in _gold(MSP_021):
        env.step(action)

    final = env.truncate(reason)

    assert (final.terminated, final.truncated, final.termination_reason) == (terminated, truncated, reason)
    info = final.reward_info
    assert info.db_match
    assert (info.training_reward, info.training_reward_source) == (training, source)
    assert final.agent_turns == 6 + (reason == "invalid_action")
    assert STOP not in [getattr(m, "content", None) for m in env.trajectory]


def test_truncate_rejects_unknown_reasons_and_inactive_episodes() -> None:
    env, _ = _env(MSP_021, [OPENING])
    env.reset()

    for reason in ("max_steps", "agent_done", "timeout"):
        with pytest.raises(ValueError):
            env.truncate(reason)
    env.truncate("response_length")
    with pytest.raises(RuntimeError, match="already finished"):
        env.truncate("response_length")


# -- parallel isolation (threads in one process) ---------------------------------------------------


def _run_lockstep(plans: list[tuple[str, list[ItsmAction]]]):
    """Step every episode in lockstep behind a barrier so tool calls interleave."""
    envs = []
    for index, (task_id, _) in enumerate(plans):
        env, _ = _env(task_id, [OPENING, CLOSING_REPLY], episode_id=f"{task_id}:par{index}")
        env.reset()
        envs.append(env)
    rounds = max(len(actions) for _, actions in plans)
    barrier = threading.Barrier(len(plans), timeout=60)
    results: list[Any] = [None] * len(plans)
    errors: list[BaseException] = []

    def run(index: int) -> None:
        actions = plans[index][1]
        try:
            for round_index in range(rounds):
                barrier.wait()
                if round_index < len(actions):
                    results[index] = envs[index].step(actions[round_index])
        except BaseException as error:
            errors.append(error)
            barrier.abort()

    threads = [threading.Thread(target=run, args=(index,)) for index in range(len(plans))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(120)
    assert not errors, errors
    return envs, results


def _sequential_final_hash(task_id: str, actions: list[ItsmAction]) -> str:
    env, _ = _env(task_id, [OPENING, CLOSING_REPLY])
    env.reset()
    return _play(env, actions)[-1].reward_info.final_db_hash


def test_parallel_episodes_with_different_task_clocks_do_not_interfere() -> None:
    seeds_before = seed_file_digests()
    clock_before = get_now()
    plans = [
        (MSP_021, _completed(_gold(MSP_021))),
        (SINGLE_017, _completed(_gold(SINGLE_017))),
    ] * 2
    reference = {task_id: _sequential_final_hash(task_id, actions) for task_id, actions in plans[:2]}

    envs, results = _run_lockstep(plans)

    for (task_id, _), final in zip(plans, results):
        assert final.reward_info.db_match, (task_id, final.reward_info.db_mismatches)
        assert final.reward_info.final_db_hash == reference[task_id]
    assert len({id(env._environment.tools.db) for env in envs}) == 4
    assert get_now() == clock_before
    assert seed_file_digests() == seeds_before


def test_concurrent_group_on_one_task_keeps_each_db_separate() -> None:
    gold = _gold(MSP_021)
    no_sla_pause = [action for index, action in enumerate(gold) if index != 3]
    duplicate_notification = [*gold, gold[5]]
    plans = [(MSP_021, _completed(p)) for p in (gold, gold, no_sla_pause, duplicate_notification)]

    _, results = _run_lockstep(plans)

    assert [r.reward_info.db_reward for r in results] == [1.0, 1.0, 0.0, 0.0]
    assert any("incident_sla/TSLA_021 stage" in m for m in results[2].reward_info.db_mismatches)
    assert "notification: unexpected ['NOTIF_031']" in results[3].reward_info.db_mismatches


# -- why the in-process env has no timeout ---------------------------------------------------------


def test_in_process_wait_timeout_does_not_stop_the_call(monkeypatch) -> None:
    """Counterexample motivating ItsmEnvWorker: giving up on a thread does not stop it."""
    original = ItsmTools.update_incident
    started = threading.Event()

    @functools.wraps(original)
    def slow(self, *args, **kwargs):
        started.set()
        time.sleep(0.5)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ItsmTools, "update_incident", slow)
    env, _ = _env(MSP_021, [OPENING])
    env.reset()
    gold = _gold(MSP_021)
    env.step(gold[0])
    before = env.current_db_hash()

    caller = threading.Thread(target=env.step, args=(gold[1],))
    caller.start()
    assert started.wait(10)
    caller.join(timeout=0.05)
    assert caller.is_alive()  # the wait gave up ...
    caller.join(10)

    assert not caller.is_alive()
    assert env.current_db_hash() != before  # ... but the write still landed
