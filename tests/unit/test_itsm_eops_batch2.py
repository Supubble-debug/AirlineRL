"""EnterpriseOps-Gym ITSM second migration batch (step 6D-2). Offline; uses the
committed batch file only (no upstream checkout, tokenizer, model or LLM)."""

from __future__ import annotations

import json

import pytest

from enterprise_worlds.data_model.message import UserMessage
from enterprise_worlds.data_model.tasks import Task
from enterprise_worlds.domains.itsm import environment as itsm_environment
from enterprise_worlds.domains.itsm.data_model import ItsmDB
from enterprise_worlds.evaluator.text_match_strategy import TextMatchConfig

from agent_rl.data.itsm_eops_batch2 import DEFAULT_BATCH2_PATH, family_overlap_check_v2
from agent_rl.data.itsm_eops_pilot import edited_actions, load_pilot, pilot_tasks, run_actions, run_upstream_verifiers
from agent_rl.data.itsm_split_v2 import load_manifest_v2
from agent_rl.envs.itsm_env import ItsmEnv, ItsmEnvConfig
from agent_rl.envs.itsm_scoring import build_task_environment, load_itsm_tasks, replay_gold_actions, score_final_db, task_clock

BATCH = load_pilot(DEFAULT_BATCH2_PATH)
TASKS = pilot_tasks(DEFAULT_BATCH2_PATH)
ENTRIES = {e["id"]: e for e in BATCH["tasks"]}
ISOLATED = {i["batch_item"]: i for i in BATCH["isolated"]}
FUZZY = TextMatchConfig(strategy="fuzzy")
IDS = sorted(TASKS)
CASES = [(tid, case["name"]) for tid in IDS for case in ENTRIES[tid]["validation_cases"]]


def test_batch_is_the_approved_scope_and_separate_from_bench2_and_v2() -> None:
    items = sorted([e["batch_item"] for e in BATCH["tasks"]] + list(ISOLATED))
    assert items == [1, 2, 3, 4, 6, 7, 8]  # 6D-1 items approved for 6D-2; 5, 9, 10 deferred
    assert sorted(e["batch_item"] for e in BATCH["tasks"]) == [1, 3, 6, 7]
    assert not set(TASKS) & set(load_itsm_tasks())
    assert not set(TASKS) & set(load_manifest_v2()["tasks"])
    for entry in BATCH["tasks"]:
        assert entry["route"] == "faithful" and entry["id"].startswith("itsmbench_eops_")
        assert entry["org_ids"], "every batch task is org-scoped"
        assert entry["source"]["commit"] == "271f2c357f763376997dfd16807fcde2474ae41b"
        assert entry["source"]["initial_state_vs_upstream"]["delta_size_after_bench2_delta"] == 0


@pytest.mark.parametrize("task_id", IDS)
def test_gold_passes_bench2_and_upstream_verifiers(task_id: str) -> None:
    task, entry = TASKS[task_id], ENTRIES[task_id]
    gold = replay_gold_actions(task)
    with task_clock(task):
        initial = build_task_environment(task)

    assert score_final_db(task, gold, text_match=FUZZY).db_reward == 1.0
    assert replay_gold_actions(task).get_db_hash() == gold.get_db_hash()
    assert all(r["passed"] for r in run_upstream_verifiers(gold.tools.db, entry["upstream_verifiers"]))
    on_initial = run_upstream_verifiers(initial.tools.db, entry["upstream_verifiers"])
    assert not all(r["passed"] for r in on_initial)
    # Only registered absence constraints / preconditions may already hold.
    assert sorted(r["name"] for r in on_initial if r["passed"]) == sorted(entry["initial_pass_allowed"])


@pytest.mark.parametrize(("task_id", "case_name"), CASES)
def test_validation_case_fails_bench2_as_expected(task_id: str, case_name: str) -> None:
    task, entry = TASKS[task_id], ENTRIES[task_id]
    case = next(c for c in entry["validation_cases"] if c["name"] == case_name)
    env = run_actions(task, edited_actions(task, case["edit"]))

    score = score_final_db(task, env, text_match=FUZZY)
    upstream = run_upstream_verifiers(env.tools.db, entry["upstream_verifiers"])

    assert score.db_reward == 0.0
    assert any(case["bench2_mismatch"] in m for m in score.mismatches), score.mismatches
    assert all(r["passed"] for r in upstream) is case["upstream_all_pass"]


def test_family_and_overlap_check_over_v2_keeps_dev_and_heldout() -> None:
    result = family_overlap_check_v2(BATCH["tasks"])
    assert all(result["checks"].values()), result["checks"]


@pytest.mark.parametrize("task_id", IDS)
def test_runs_in_itsm_env_without_leaking_private_fields(task_id: str) -> None:
    task, entry = TASKS[task_id], ENTRIES[task_id]

    class OneReply:
        def get_init_state(self):
            return 0

        def generate_next_message(self, agent_message, state):
            return UserMessage(content="Hi, I need some help."), state + 1

    reset = ItsmEnv(ItsmEnvConfig(task_id=task_id, text_match=FUZZY, max_steps=40), user=OneReply(), task=task).reset()
    visible = reset.observation + json.dumps(reset.public_info)
    private = [
        task.scenario.task_description,
        task.scenario.simulator_guidance,
        task_id,
        entry["source"]["upstream_task_id"],
        "EnterpriseOps",
        *task.evaluation_criteria.nl_assertions,
        *(v["query"] for v in entry["upstream_verifiers"]),
    ]
    assert not [text for text in private if text in visible]


def test_isolation_evidence_still_holds() -> None:
    msp = ItsmDB.load(itsm_environment.ITSM_TASKS_PATH.parent / "msp_db.json")
    assert {i["batch_item"]: i["reason"] for i in BATCH["isolated"]} == {
        2: "ambiguous REQUIRED cascade",
        4: "core requirement not discriminated by Bench2 DB scoring",
        8: "shares 2 same-seed entities with a held-out task",
    }

    # Item 2: INC_061 has no SLA row, client incidents use provider SLA definitions,
    # and high priority has two provider definitions.
    assert not [r for r in msp.incident_sla.values() if r.incident_id == "INC_061"]
    assert msp.incident["INC_049"].org_id == "ORG_013"
    assert any(r.incident_id == "INC_049" and r.sla_def_id == "SLA_002" for r in msp.incident_sla.values())
    assert sorted(s.sla_def_id for s in msp.sla_definition.values() if s.applies_to_priority == "high") == ["SLA_003", "SLA_004"]

    # Item 4: the server can go unrecorded and still score DB = 1.
    entry4 = ISOLATED[4]["attempted_conversion"]
    task4 = Task.model_validate(entry4)
    case = next(c for c in entry4["validation_cases"] if c["name"] == "server_named_in_description_only")
    assert score_final_db(task4, run_actions(task4, edited_actions(task4, case["edit"])), text_match=FUZZY).db_reward == 1.0

    # Item 8: two gold entities shared with held-out itsmbench_single_012.
    entry8 = ISOLATED[8]["attempted_conversion"]
    overlap = family_overlap_check_v2([*BATCH["tasks"], entry8])["tasks"][entry8["id"]]["dev_or_heldout_entity_overlap_ge2"]
    assert overlap == [{"task": "itsmbench_single_012", "split": "heldout", "same_seed_entities": ["GROUP_001", "INC_003"]}]
