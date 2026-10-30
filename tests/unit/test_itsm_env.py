"""Single-episode ITSM adapter (step 4).

Real Bench2 tools, seeds, tasks and DB scoring. The policy is a scripted agent
and the user is a deterministic stand-in (``ScriptedUser``); no LLM is called.
These tests show the adapter's protocol behaviour, not model capability.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from typing import Any

import pytest

from enterprise_worlds.data_model.message import (
    AssistantMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from enterprise_worlds.domains.itsm.tools import ItsmTools
from enterprise_worlds.evaluator.text_match_strategy import TextMatchConfig
from enterprise_worlds.gym.gym_agent import DONE_TOOL_SCHEMA
from enterprise_worlds.orchestrator.orchestrator import Orchestrator
from enterprise_worlds.user.base import STOP
from enterprise_worlds.user.user_simulator import UserSimulator
from enterprise_worlds.utils.clock import get_now

from agent_rl.data.schemas import ToolCallRecord
from agent_rl.envs.action_parser import ActionFormatError, ModelToolCall
from agent_rl.envs.itsm_env import (
    DEFAULT_USER_LLM,
    ItsmAction,
    ItsmEnv,
    ItsmEnvConfig,
    to_itsm_action,
)
from agent_rl.envs.itsm_scoring import (
    ItsmInfrastructureError,
    build_task_environment,
    load_itsm_tasks,
    task_clock,
)


TASKS = load_itsm_tasks()
FUZZY = TextMatchConfig(strategy="fuzzy")
MSP_021 = "itsmbench_msp_021"
SINGLE_017 = "itsmbench_single_017"
OPENING = "Hi, I need help with a ticket."
CLOSING_REPLY = "Great, thanks."


class ScriptedUser:
    """Deterministic user stand-in: replies from a fixed list, records what it was sent."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.received: list[str | None] = []

    def get_init_state(self) -> int:
        return 0

    def generate_next_message(self, agent_message, state: int):
        self.received.append(agent_message.content)
        return UserMessage(content=self.replies[state]), state + 1

    @staticmethod
    def is_stop(message) -> bool:
        return STOP in (message.content or "")


def _gold(task_id: str) -> list[ItsmAction]:
    return [
        ItsmAction.tool_call(action.name, copy.deepcopy(action.arguments))
        for action in TASKS[task_id].evaluation_criteria.actions
    ]


def _env(task_id: str, replies: list[str], **kwargs) -> tuple[ItsmEnv, ScriptedUser]:
    user = ScriptedUser(replies)
    config = ItsmEnvConfig(task_id=task_id, text_match=FUZZY)
    return ItsmEnv(config, user=user, episode_id=f"{task_id}:g0:0:test", **kwargs), user


def _play(task_id: str, actions: list[ItsmAction], replies: list[str]):
    env, user = _env(task_id, replies)
    reset = env.reset()
    transitions = [env.step(action) for action in actions]
    return env, user, reset, transitions


def _completed(tool_actions: list[ItsmAction]) -> list[ItsmAction]:
    return [*tool_actions, ItsmAction.message("All done, summary sent."), ItsmAction.done()]


# -- action conversion and schemas --------------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "calls", "expected"),
    [
        ("  Could you confirm the number? ", None, ItsmAction.message("Could you confirm the number?")),
        (
            None,
            [ModelToolCall("find_incident_by_number", {"number": "INC0000007"}, "call_7")],
            ItsmAction.tool_call("find_incident_by_number", {"number": "INC0000007"}, "call_7"),
        ),
        (None, [ModelToolCall("find_users", {})], ItsmAction.tool_call("find_users", {}, None)),
        ("", [ModelToolCall("done", {})], ItsmAction.done()),
    ],
    ids=["message", "tool_call", "tool_call_without_model_id", "done"],
)
def test_model_output_converts_to_one_action(content, calls, expected) -> None:
    assert to_itsm_action(content=content, tool_calls=calls) == expected


@pytest.mark.parametrize(
    ("content", "calls"),
    [
        ("text", [ModelToolCall("find_users", {})]),
        (None, [ModelToolCall("find_users", {}), ModelToolCall("find_users", {})]),
        (None, [ModelToolCall("done", {"reason": "x"})]),
        (f"bye {STOP}", None),
        ("   ", None),
        (None, None),
    ],
    ids=["text_and_call", "parallel_calls", "done_with_arguments", "reserved_stop", "blank", "empty"],
)
def test_invalid_model_output_is_rejected(content, calls) -> None:
    with pytest.raises(ActionFormatError):
        to_itsm_action(content=content, tool_calls=calls)


def test_done_cannot_be_sent_as_a_tool_call() -> None:
    with pytest.raises(ActionFormatError):
        ItsmAction.tool_call("done", {})


def test_public_tool_schemas_are_bench2_tools_plus_done() -> None:
    env, _ = _env(SINGLE_017, [OPENING])
    tools = env.reset().public_info["tools"]
    names = [tool["function"]["name"] for tool in tools]

    assert names[:-1] == list(ItsmTools._tool_names)
    assert tools[-1] == DONE_TOOL_SCHEMA
    assert len(set(names)) == len(names)
    assert all(tool["type"] == "function" for tool in tools)


def test_public_info_is_identical_for_every_task_and_has_no_private_text() -> None:
    digests = set()
    for task_id, task in TASKS.items():
        reset = _env(task_id, [OPENING])[0].reset()
        assert set(reset.public_info) == {"policy", "tools"}
        public = json.dumps(reset.public_info, sort_keys=True)
        digests.add(hashlib.sha256(public.encode()).hexdigest())
        visible = public + reset.observation
        private = [
            task.id,
            task.scenario.task_description,
            task.scenario.simulator_guidance,
            *task.evaluation_criteria.nl_assertions,
        ]
        assert not [text for text in private if text and text in visible], task_id

    # Task-invariant public info cannot carry any task-specific field.
    assert len(digests) == 1


def test_reset_observation_is_greeting_and_first_user_turn() -> None:
    env, user = _env(SINGLE_017, [OPENING])

    reset = env.reset()

    assert reset.observation == f"assistant: Hi! How can I help you today?\nuser: {OPENING}"
    assert reset.step_count == 1
    assert user.received == ["Hi! How can I help you today?"]


def test_default_user_is_bench2_simulator_with_decision_d5() -> None:
    env = ItsmEnv(ItsmEnvConfig(task_id=SINGLE_017, text_match=FUZZY))

    assert isinstance(env._user, UserSimulator)
    assert (env._user.llm, env._user.llm_args) == (DEFAULT_USER_LLM, {"temperature": 0.0})
    assert DEFAULT_USER_LLM == "deepseek/deepseek-v4-flash"


def test_config_rejects_degrading_text_match() -> None:
    with pytest.raises(ValueError, match="degrade"):
        ItsmEnvConfig(task_id=SINGLE_017, text_match=TextMatchConfig())


# -- normal and error tool calls ----------------------------------------------------------------


def test_normal_tool_call_returns_the_real_tool_result() -> None:
    env, _ = _env(SINGLE_017, [OPENING])
    env.reset()
    before = env.current_db_hash()

    transition = env.step(ItsmAction.tool_call("find_incident_by_number", {"number": "INC0000007"}))

    record = transition.tool_record
    assert transition.observation.startswith(f"tool[{record.call_id}] ok: {{")
    assert json.loads(record.result)["incident_id"] == "INC_020"
    assert (record.call_id, record.result_received, record.error_class) == (
        f"{SINGLE_017}:g0:0:test/c001",
        True,
        None,
    )
    assert not transition.terminated and not transition.truncated
    assert transition.step_count == 2
    assert env.current_db_hash() == before


ERROR_CALLS = {
    "domain_error": (
        ItsmAction.tool_call("find_incident_by_number", {"number": "INC9999999"}),
        "Incident with number 'INC9999999' not found",
        ("VALIDATION_ERROR", "number"),
    ),
    "unknown_tool": (
        ItsmAction.tool_call("find_incident_by_numbr", {}),
        "Tool 'find_incident_by_numbr' not found.",
        (None, None),
    ),
    "bad_arguments": (
        ItsmAction.tool_call("find_incident_by_number", {"number": "INC0000007", "num": "x"}),
        "Invalid arguments for tool 'find_incident_by_number': got an unexpected keyword argument 'num'",
        (None, None),
    ),
    "bad_arguments_missing": (
        ItsmAction.tool_call("find_incident_by_number", {}),
        "Invalid arguments for tool 'find_incident_by_number': missing a required argument: 'number'",
        (None, None),
    ),
}


@pytest.mark.parametrize("case", sorted(ERROR_CALLS))
def test_tool_error_is_classified_and_the_episode_continues(case: str) -> None:
    action, message, (code, error_field) = ERROR_CALLS[case]
    error_class = case.removesuffix("_missing")
    env, _ = _env(SINGLE_017, [OPENING])
    env.reset()
    before = env.current_db_hash()

    transition = env.step(action)

    record = transition.tool_record
    assert transition.observation == f"tool[{record.call_id}] error({error_class}): {message}"
    assert (record.error_class, record.error, record.error_code, record.error_field) == (
        error_class,
        message,
        code,
        error_field,
    )
    assert record.result_received and record.result is None
    assert "Mixin" not in transition.observation
    assert env.trajectory[-1] == ToolMessage(
        id=record.call_id, content=f"Error: {message}", requestor="assistant", error=True
    )
    assert not transition.terminated
    assert env.current_db_hash() == before


def test_unexpected_tool_exception_is_its_own_class(monkeypatch) -> None:
    def broken(self, number: str):
        raise KeyError("index corrupted")

    monkeypatch.setattr(ItsmTools, "find_incident_by_number", broken)
    env, _ = _env(SINGLE_017, [OPENING])
    env.reset()

    record = env.step(ItsmAction.tool_call("find_incident_by_number", {"number": "INC0000007"})).tool_record

    assert (record.error_class, record.error) == ("tool_exception_other", "'index corrupted'")


# -- recovery and done --------------------------------------------------------------------------


def test_recovery_after_errors_reaches_the_gold_state() -> None:
    gold = _gold(SINGLE_017)
    create_problem = gold[3]
    actions = [
        ItsmAction.tool_call("find_incident_by_number", {"number": "INC9999999"}),
        *gold[:3],
        ItsmAction.tool_call("create_problem", {**create_problem.arguments, "impact_level": "medium"}),
        *gold[3:],
    ]

    env, _, _, transitions = _play(SINGLE_017, _completed(actions), [OPENING, CLOSING_REPLY])

    classes = [record.error_class for record in env.tool_records]
    assert classes == ["domain_error", None, None, None, "bad_arguments"] + [None] * 7
    # The rejected create_problem never executed: exactly one problem was created.
    assert transitions[-1].reward_info.db_match, transitions[-1].reward_info.db_mismatches
    assert transitions[-1].termination_reason == "agent_done"


def test_done_scores_the_final_db_db_only() -> None:
    env, user, _, transitions = _play(MSP_021, _completed(_gold(MSP_021)), [OPENING, CLOSING_REPLY])

    final = transitions[-1]
    info = final.reward_info
    assert (final.observation, final.terminated, final.truncated, final.termination_reason) == (
        "",
        True,
        False,
        "agent_done",
    )
    assert (info.db_evaluated, info.db_match, info.db_reward) == (True, True, 1.0)
    assert (info.text_match_requested, info.text_match_effective) == ("fuzzy", "fuzzy")
    assert (info.nl_status, info.nl_reward, info.joint_reward) == ("not_evaluated", None, None)
    assert (info.training_reward, info.training_reward_source) == (1.0, "db_only")
    assert env.trajectory[-1] == AssistantMessage(content=STOP)
    assert user.received == ["Hi! How can I help you today?", "All done, summary sent."]


def test_done_on_a_wrong_path_is_a_valid_zero() -> None:
    actions = _gold(MSP_021)
    del actions[3]  # SLA never paused

    _, _, _, transitions = _play(MSP_021, _completed(actions), [OPENING, CLOSING_REPLY])

    info = transitions[-1].reward_info
    assert (info.db_match, info.db_reward, info.training_reward) == (False, 0.0, 0.0)
    assert "incident_sla/TSLA_021 stage: 'paused' != 'in_progress'" in info.db_mismatches


def test_user_stop_ends_and_scores_the_episode() -> None:
    actions = [*_gold(MSP_021), ItsmAction.message("Done, anything else?")]

    _, _, _, transitions = _play(MSP_021, actions, [OPENING, STOP])

    final = transitions[-1]
    assert (final.observation, final.terminated, final.termination_reason) == ("", True, "user_stop")
    assert final.reward_info.db_match


def test_message_returns_only_the_user_reply() -> None:
    env, user = _env(SINGLE_017, [OPENING, "It is INC0000007."])
    env.reset()

    transition = env.step(ItsmAction.message("Which incident number?"))

    assert transition.observation == "user: It is INC0000007."
    assert user.received[-1] == "Which incident number?"
    assert transition.step_count == 2


# -- call IDs, incremental observations, structured trace -----------------------------------------


def _mixed_single_017_run():
    actions = _completed(
        [
            ItsmAction.tool_call("find_incident_by_number", {"number": "INC9999999"}),
            ItsmAction.tool_call("find_incident_by_numbr", {}),
            *_gold(SINGLE_017),
        ],
    )
    return _play(SINGLE_017, actions, [OPENING, CLOSING_REPLY])


def test_every_call_has_a_unique_id_and_exactly_one_result() -> None:
    env, _, _, _ = _mixed_single_017_run()
    trajectory = env.trajectory

    call_ids = []
    for index, message in enumerate(trajectory):
        if isinstance(message, AssistantMessage) and message.is_tool_call():
            assert len(message.tool_calls) == 1
            call_id = message.tool_calls[0].id
            assert isinstance(trajectory[index + 1], ToolMessage)
            assert trajectory[index + 1].id == call_id
            call_ids.append(call_id)
    tool_message_ids = [m.id for m in trajectory if isinstance(m, ToolMessage)]

    assert call_ids == tool_message_ids == [record.call_id for record in env.tool_records]
    assert len(set(call_ids)) == len(call_ids) == 12
    assert call_ids == [f"{SINGLE_017}:g0:0:test/c{n:03d}" for n in range(1, 13)]


def test_observations_are_incremental() -> None:
    env, _, reset, transitions = _mixed_single_017_run()
    observations = [reset.observation] + [t.observation for t in transitions]
    joined = "\n".join(observations)

    assert joined.count("Hi! How can I help you today?") == 1
    assert joined.count(OPENING) == 1
    for record in env.tool_records:
        payload = record.result if record.error_class is None else record.error
        assert joined.count(f"tool[{record.call_id}]") == 1
        assert payload in observations[env.tool_records.index(record) + 1]
    # Each tool observation is one line: the new result only.
    assert all("\n" not in t.observation for t in transitions)


def test_trace_is_json_serializable_and_converts_to_tool_call_records() -> None:
    env, _, _, transitions = _mixed_single_017_run()

    trace = json.loads(json.dumps(env.export_trace()))

    assert trace["termination_reason"] == "agent_done"
    assert trace["reward_info"]["db_match"] is True
    assert [c["call_id"] for c in trace["tool_calls"]] == [
        m["id"] for m in trace["messages"] if m["role"] == "tool"
    ]
    converted = [record.to_tool_call_record() for record in env.tool_records]
    assert all(isinstance(record, ToolCallRecord) for record in converted)
    assert [(r.error is None, r.result_received) for r in converted][:3] == [
        (False, True),
        (False, True),
        (True, True),
    ]


# -- parity with Bench2's Orchestrator ------------------------------------------------------------


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


@pytest.mark.parametrize("task_id", [MSP_021, SINGLE_017])
def test_adapter_matches_bench2_orchestrator(task_id: str) -> None:
    # Bad-argument text intentionally differs from Bench2 (no class name), so it is excluded.
    actions = _completed(
        [
            ItsmAction.tool_call("find_incident_by_number", {"number": "INC9999999"}),
            ItsmAction.tool_call("find_incident_by_numbr", {}),
            *_gold(task_id),
        ],
    )
    replies = [OPENING, CLOSING_REPLY]
    env, user, _, _ = _play(task_id, actions, replies)

    task = TASKS[task_id]
    bench2_user = ScriptedUser(replies)
    with task_clock(task):
        bench2_env = build_task_environment(task)
        run = Orchestrator(ScriptedOrchestratorAgent(actions), bench2_user, bench2_env, max_steps=100).run()

    assert run.stopped
    assert _normalized(env.trajectory) == _normalized(run.trajectory)
    assert user.received == bench2_user.received
    assert env.current_db_hash() == bench2_env.get_db_hash()


# -- task clock in the executing thread ---------------------------------------------------------


def test_each_step_runs_under_the_task_clock_in_its_own_thread() -> None:
    env, _ = _env(MSP_021, [OPENING, CLOSING_REPLY])
    env.reset()
    results: list[Any] = []
    thread_clocks: list[str] = []

    for action in _completed(_gold(MSP_021)):
        def run(action=action) -> None:
            results.append(env.step(action))
            thread_clocks.append(get_now())

        thread = threading.Thread(target=run)
        thread.start()
        thread.join()

    # msp_021 is clock-sensitive: DB=1 needs the task time in every executing thread.
    assert results[-1].reward_info.db_match, results[-1].reward_info.db_mismatches
    assert set(thread_clocks) == {"2024-06-01 00:00:00"}


# -- single-episode use and infrastructure classification ---------------------------------------


def test_env_runs_exactly_one_episode() -> None:
    env, _ = _env(MSP_021, [OPENING, CLOSING_REPLY])
    with pytest.raises(RuntimeError, match="reset"):
        env.step(ItsmAction.done())
    env.reset()
    with pytest.raises(RuntimeError, match="exactly one episode"):
        env.reset()
    env.step(ItsmAction.done())
    with pytest.raises(RuntimeError, match="already finished"):
        env.step(ItsmAction.done())
    env.close()
    env.close()
    with pytest.raises(RuntimeError, match="closed"):
        env.step(ItsmAction.done())


class BrokenUser(ScriptedUser):
    def generate_next_message(self, agent_message, state):
        if state >= len(self.replies):
            raise TimeoutError("user llm timed out")
        return super().generate_next_message(agent_message, state)


@pytest.mark.parametrize(
    ("replies", "stage", "at_reset"),
    [
        ([], "user_sim", True),
        ([""], "user_sim", True),
        ([STOP], "user_sim_empty_episode", True),
        ([OPENING], "user_sim", False),
    ],
    ids=["raises_at_reset", "empty_reply", "stop_before_first_turn", "raises_mid_episode"],
)
def test_user_failures_are_infrastructure_not_reward(replies, stage, at_reset) -> None:
    env = ItsmEnv(ItsmEnvConfig(task_id=SINGLE_017, text_match=FUZZY), user=BrokenUser(replies))

    with pytest.raises(ItsmInfrastructureError) as raised:
        env.reset()
        assert not at_reset
        env.step(ItsmAction.message("Which incident?"))

    assert raised.value.stage == stage
    with pytest.raises(RuntimeError, match="infrastructure"):
        env.step(ItsmAction.done())
    assert env.export_trace()["reward_info"] is None
