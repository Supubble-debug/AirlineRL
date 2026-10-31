"""Explicit v1/v2 ITSM task resolution, in process and in the worker (step 6C).

Spawns real worker processes (a few seconds each). No tokenizer, model or LLM.
"""

from __future__ import annotations

import copy
import functools
import json

import pytest

from enterprise_worlds.evaluator.text_match_strategy import TextMatchConfig

from itsm_test_support import ScriptedUser

from agent_rl.data import itsm_task_sets
from agent_rl.data.itsm_eops_pilot import load_pilot, pilot_tasks
from agent_rl.data.itsm_split_v2 import ACCEPTED_DERIVED_TASK_IDS
from agent_rl.data.itsm_task_sets import load_task_set, resolve_task, task_ids
from agent_rl.envs.itsm_env import ItsmAction, ItsmEnv, ItsmEnvConfig
from agent_rl.envs.itsm_scoring import load_itsm_tasks
from agent_rl.envs.itsm_worker import ItsmEnvWorker


FUZZY = TextMatchConfig(strategy="fuzzy")
V2 = "itsm_split_v2"
REPLIES = ["Hi, I need some help.", "Thanks, that's all."]
PILOT = {e["id"]: e for e in load_pilot()["tasks"]}


def _config(task_id: str, split_version: str | None = V2) -> ItsmEnvConfig:
    return ItsmEnvConfig(task_id=task_id, text_match=FUZZY, max_steps=40, split_version=split_version)


def _completed_gold(task) -> list[ItsmAction]:
    gold = [ItsmAction.tool_call(a.name, copy.deepcopy(a.arguments)) for a in task.evaluation_criteria.actions]
    return [*gold, ItsmAction.message("All done."), ItsmAction.done()]


def _private_texts(task_id: str) -> list[str]:
    task, entry = resolve_task(task_id, split_version=V2), PILOT[task_id]
    return [
        task_id,
        task.scenario.task_description,
        task.scenario.simulator_guidance,
        *task.evaluation_criteria.nl_assertions,
        entry["source"]["upstream_task_id"],
        entry["source"]["commit"],
        "EnterpriseOps",
        *entry["conversion_notes"],
        *(v["query"] for v in entry["upstream_verifiers"]),
    ]


def test_every_entry_point_requires_an_explicit_version() -> None:
    with pytest.raises(TypeError):
        load_task_set()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        resolve_task(ACCEPTED_DERIVED_TASK_IDS[0])  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        task_ids("train")  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="unknown split_version"):
        load_task_set(split_version="itsm_split_v4")


def test_v1_and_v2_task_sets() -> None:
    v1, v2 = load_task_set(split_version="itsm_split_v1"), load_task_set(split_version=V2)
    bench = load_itsm_tasks()

    assert set(v1) == set(bench) and len(v1) == 53
    assert set(v2) == set(v1) | set(ACCEPTED_DERIVED_TASK_IDS) and len(v2) == 56
    assert all(v2[t] == v1[t] for t in v1)
    assert {t: v2[t] for t in ACCEPTED_DERIVED_TASK_IDS} == pilot_tasks()
    with pytest.raises(ValueError, match="not in itsm_split_v1"):
        resolve_task(ACCEPTED_DERIVED_TASK_IDS[0], split_version="itsm_split_v1")
    assert len(task_ids("train", split_version=V2)) == 32
    assert task_ids("dev", split_version=V2) == task_ids("dev", split_version="itsm_split_v1")
    with pytest.raises(PermissionError):
        task_ids("heldout", split_version=V2)
    assert task_ids("heldout", split_version=V2, final_evaluation=True) == task_ids(
        "heldout", split_version="itsm_split_v1", final_evaluation=True
    )


def test_derived_task_must_match_the_v2_checksum(monkeypatch) -> None:
    original = itsm_task_sets.derived_raw_tasks()
    edited = copy.deepcopy(original)
    edited[ACCEPTED_DERIVED_TASK_IDS[0]]["current_time"] = "2026-01-14T09:00:00"
    monkeypatch.setattr(itsm_task_sets, "derived_raw_tasks", lambda: edited)

    with pytest.raises(ValueError, match="checksum"):
        itsm_task_sets._tasks.__wrapped__(V2)


def test_env_default_does_not_switch_versions() -> None:
    task_id = ACCEPTED_DERIVED_TASK_IDS[0]

    with pytest.raises(ValueError, match="unknown ITSM task_id"):
        ItsmEnv(_config(task_id, split_version=None), user=ScriptedUser(REPLIES))
    with pytest.raises(ValueError, match="not in itsm_split_v1"):
        ItsmEnv(_config(task_id, split_version="itsm_split_v1"), user=ScriptedUser(REPLIES))
    with pytest.raises(ValueError, match="not both"):
        ItsmEnv(_config(task_id), user=ScriptedUser(REPLIES), task=resolve_task(task_id, split_version=V2))
    # A v1 task resolves to the same Task with or without an explicit version.
    plain = ItsmEnv(_config("itsmbench_msp_001", split_version=None), user=ScriptedUser(REPLIES))
    versioned = ItsmEnv(_config("itsmbench_msp_001", split_version=V2), user=ScriptedUser(REPLIES))
    assert plain.task == versioned.task


@pytest.mark.parametrize("task_id", ACCEPTED_DERIVED_TASK_IDS)
def test_in_process_env_loads_and_solves_derived_task_without_leaks(task_id: str) -> None:
    env = ItsmEnv(_config(task_id), user=ScriptedUser(REPLIES), episode_id="ep")
    reset = env.reset()
    visible = reset.observation + json.dumps(reset.public_info)
    transitions = [env.step(action) for action in _completed_gold(env.task)]
    visible += "".join(t.observation for t in transitions)

    assert transitions[-1].reward_info.db_match
    assert not [text for text in _private_texts(task_id) if text in visible]


@pytest.mark.parametrize("task_id", ACCEPTED_DERIVED_TASK_IDS)
def test_worker_child_loads_and_solves_derived_task(task_id: str) -> None:
    worker = ItsmEnvWorker(
        _config(task_id),
        call_timeout_s=60.0,
        user_factory=functools.partial(ScriptedUser, REPLIES),
        episode_id="ep",
    )
    try:
        reset = worker.reset()
        visible = reset.observation + json.dumps(reset.public_info)
        actions = _completed_gold(resolve_task(task_id, split_version=V2))
        transitions = [worker.step(action) for action in actions]
        visible += "".join(t.observation for t in transitions)
        trace = worker.export_trace()
    finally:
        worker.close()

    assert transitions[-1].reward_info.db_match and trace["task_id"] == task_id
    assert not [text for text in _private_texts(task_id) if text in visible]


def test_worker_without_a_version_refuses_a_derived_task() -> None:
    with pytest.raises(ValueError, match="unknown ITSM task_id"):
        ItsmEnvWorker(
            _config(ACCEPTED_DERIVED_TASK_IDS[0], split_version=None),
            call_timeout_s=60.0,
            user_factory=functools.partial(ScriptedUser, REPLIES),
        )
