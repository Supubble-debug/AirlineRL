"""Process-isolated ITSM episodes: bounded waits, cancellation, crashes (step 5).

Each test spawns real child processes (a few seconds each). Faults are injected in
the child through the worker ``initializer``. No LLM is called.
"""

from __future__ import annotations

import copy
import functools
import os
import threading
import time

import pytest

from enterprise_worlds.evaluator.text_match_strategy import TextMatchConfig
from enterprise_worlds.utils.clock import get_now

from itsm_test_support import (
    ScriptedUser,
    install_crashing_tool,
    install_slow_tool,
    seed_file_digests,
)

from agent_rl.envs.itsm_env import ItsmAction, ItsmEnv, ItsmEnvConfig
from agent_rl.envs.itsm_scoring import ItsmInfrastructureError, load_itsm_tasks
from agent_rl.envs.itsm_worker import ItsmEnvWorker


TASKS = load_itsm_tasks()
FUZZY = TextMatchConfig(strategy="fuzzy")
MSP_021 = "itsmbench_msp_021"
SINGLE_017 = "itsmbench_single_017"
OPENING = "Hi, I need help with a ticket."
CLOSING_REPLY = "Great, thanks."
TEST_MAX_STEPS = 40
SLOW_TOOL_S = 5.0


def _config(task_id: str) -> ItsmEnvConfig:
    return ItsmEnvConfig(task_id=task_id, text_match=FUZZY, max_steps=TEST_MAX_STEPS)


def _worker(task_id: str, replies=(OPENING, CLOSING_REPLY), **kwargs) -> ItsmEnvWorker:
    kwargs.setdefault("call_timeout_s", 60.0)
    return ItsmEnvWorker(
        _config(task_id),
        user_factory=functools.partial(ScriptedUser, list(replies)),
        episode_id=f"{task_id}:worker",
        **kwargs,
    )


def _gold(task_id: str) -> list[ItsmAction]:
    return [
        ItsmAction.tool_call(action.name, copy.deepcopy(action.arguments))
        for action in TASKS[task_id].evaluation_criteria.actions
    ]


def _completed(actions: list[ItsmAction]) -> list[ItsmAction]:
    return [*actions, ItsmAction.message("All done, summary sent."), ItsmAction.done()]


def _wait_past_slow_tool(started: float) -> None:
    time.sleep(max(0.0, started + SLOW_TOOL_S + 1.0 - time.monotonic()))


DOMAIN_ERROR = ItsmAction.tool_call("find_incident_by_number", {"number": "INC9999999"})


def test_worker_runs_an_episode_like_the_in_process_env() -> None:
    clock_before = get_now()
    actions = [DOMAIN_ERROR, *_completed(_gold(MSP_021))]
    worker = _worker(MSP_021)
    try:
        assert worker.pid != os.getpid() and worker.is_alive
        worker.reset()
        transitions = [worker.step(action) for action in actions]
        final = transitions[-1]
        assert transitions[0].tool_record.error_class == "domain_error"
        assert final.reward_info.db_match and worker.status == "scored"
        with pytest.raises(RuntimeError, match="already finished"):
            worker.step(ItsmAction.done())  # raised in the child, re-raised here
        assert worker.score() == final
        trace = worker.export_trace()
    finally:
        worker.close()
    worker.close()

    assert not worker.is_alive
    assert worker.export_trace() == trace
    with pytest.raises(RuntimeError, match="closed"):
        worker.step(ItsmAction.done())
    reference = ItsmEnv(_config(MSP_021), user=ScriptedUser([OPENING, CLOSING_REPLY]))
    reference.reset()
    for action in actions:
        reference.step(action)
    expected = reference.export_trace()
    assert (trace["messages"], trace["final_db_hash"]) == (expected["messages"], expected["final_db_hash"])
    assert get_now() == clock_before


def test_call_timeout_kills_the_child_and_the_call_never_completes(tmp_path) -> None:
    marker = tmp_path / "slow-call-finished"
    worker = _worker(
        MSP_021,
        call_timeout_s=2.0,
        initializer=functools.partial(install_slow_tool, "update_incident", SLOW_TOOL_S, str(marker)),
    )
    gold = _gold(MSP_021)
    try:
        worker.reset()
        worker.step(gold[0])
        started = time.monotonic()
        with pytest.raises(ItsmInfrastructureError) as raised:
            worker.step(gold[1])
        waited = time.monotonic() - started

        assert (raised.value.stage, raised.value.kind) == ("timeout", "killed")
        assert 1.9 <= waited < SLOW_TOOL_S
        assert worker.failure["execution_stopped"] is True
        assert not worker.is_alive and worker.status == "failed"
        _wait_past_slow_tool(started)
        assert not marker.exists()  # stopped, not merely abandoned
        for call in (lambda: worker.step(gold[2]), worker.score):
            with pytest.raises(RuntimeError, match="worker failed"):
                call()
        with pytest.raises(RuntimeError, match="trace was lost"):
            worker.export_trace()
    finally:
        worker.close()
    worker.close()


def test_cancel_from_another_thread_stops_an_in_flight_call(tmp_path) -> None:
    marker = tmp_path / "slow-call-finished"
    worker = _worker(
        MSP_021,
        initializer=functools.partial(install_slow_tool, "update_incident", SLOW_TOOL_S, str(marker)),
    )
    gold = _gold(MSP_021)
    try:
        worker.reset()
        worker.step(gold[0])
        threading.Timer(0.5, worker.cancel).start()
        started = time.monotonic()
        with pytest.raises(ItsmInfrastructureError) as raised:
            worker.step(gold[1])

        assert (raised.value.stage, raised.value.kind) == ("cancelled", "killed")
        assert time.monotonic() - started < SLOW_TOOL_S
        assert not worker.is_alive
        _wait_past_slow_tool(started)
        assert not marker.exists()
    finally:
        worker.close()


def test_child_side_failure_propagates_and_idle_cancel_reclaims_the_child() -> None:
    worker = _worker(SINGLE_017, replies=[OPENING])
    try:
        worker.reset()
        with pytest.raises(ItsmInfrastructureError) as raised:
            worker.step(ItsmAction.message("Which incident?"))  # no scripted reply left
        assert raised.value.stage == "user_sim"
        # The episode failed inside a healthy child: its trace is still available.
        assert worker.status == "failed" and worker.failure is None and worker.is_alive
        assert worker.export_trace()["failure"]["stage"] == "user_sim"

        worker.cancel()

        assert (worker.failure["stage"], worker.failure["execution_stopped"]) == ("cancelled", True)
        assert not worker.is_alive
        with pytest.raises(RuntimeError, match="worker failed"):
            worker.step(ItsmAction.done())
    finally:
        worker.close()


def test_episode_deadline_bounds_the_whole_episode() -> None:
    worker = _worker(MSP_021, episode_timeout_s=3.0)
    gold = _gold(MSP_021)
    try:
        worker.reset()
        worker.step(gold[0])
        time.sleep(3.2)
        started = time.monotonic()
        with pytest.raises(ItsmInfrastructureError) as raised:
            worker.step(gold[1])

        assert raised.value.stage == "timeout"
        assert time.monotonic() - started < 1.0
        assert worker.failure["execution_stopped"] and not worker.is_alive
    finally:
        worker.close()


def test_child_crash_is_infrastructure_and_reclaimed() -> None:
    worker = _worker(
        SINGLE_017, initializer=functools.partial(install_crashing_tool, "find_incident_by_number", 17)
    )
    try:
        worker.reset()
        with pytest.raises(ItsmInfrastructureError) as raised:
            worker.step(ItsmAction.tool_call("find_incident_by_number", {"number": "INC0000007"}))

        assert raised.value.stage == "worker_crashed"
        assert (worker.failure["exitcode"], worker.failure["execution_stopped"]) == (17, True)
    finally:
        worker.close()


def test_parallel_workers_are_isolated_processes() -> None:
    seeds_before = seed_file_digests()
    workers = [_worker(MSP_021), _worker(SINGLE_017)]
    results = {}

    def run(worker: ItsmEnvWorker) -> None:
        worker.reset()
        for action in _completed(_gold(worker.config.task_id)):
            final = worker.step(action)
        results[worker.config.task_id] = final

    try:
        threads = [threading.Thread(target=run, args=(worker,)) for worker in workers]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(120)
        assert set(results) == {MSP_021, SINGLE_017}
        assert all(result.reward_info.db_match for result in results.values())
        assert len({worker.pid for worker in workers} | {os.getpid()}) == 3
    finally:
        for worker in workers:
            worker.close()
    assert seed_file_digests() == seeds_before
