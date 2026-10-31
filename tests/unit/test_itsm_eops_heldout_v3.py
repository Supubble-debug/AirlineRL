"""v3 held-out candidates (step 6D-3). Offline; uses the committed candidate file
only (no upstream checkout, tokenizer, model or LLM)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from enterprise_worlds.data_model.message import UserMessage
from enterprise_worlds.data_model.tasks import Task
from enterprise_worlds.domains.itsm import environment as itsm_environment
from enterprise_worlds.domains.itsm.data_model import ItsmDB
from enterprise_worlds.evaluator.text_match_strategy import TextMatchConfig

from agent_rl.data.itsm_eops_heldout_v3 import DEFAULT_HELDOUT_V3_PATH, heldout_gate_check
from agent_rl.data.itsm_eops_pilot import edited_actions, load_pilot, pilot_tasks, run_actions, run_upstream_verifiers
from agent_rl.data.itsm_split_v2 import load_manifest_v2
from agent_rl.envs.itsm_env import ItsmEnv, ItsmEnvConfig
from agent_rl.envs.itsm_scoring import build_task_environment, load_itsm_tasks, replay_gold_actions, score_final_db, task_clock

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = load_pilot(DEFAULT_HELDOUT_V3_PATH)
TASKS = pilot_tasks(DEFAULT_HELDOUT_V3_PATH)
ENTRIES = {e["id"]: e for e in DATA["tasks"]}
ISOLATED = {i["candidate_item"]: i for i in DATA["isolated"]}
FUZZY = TextMatchConfig(strategy="fuzzy")
IDS = sorted(TASKS)
CASES = [(tid, case["name"]) for tid in IDS for case in ENTRIES[tid]["validation_cases"]]
MSP = ItsmDB.load(itsm_environment.ITSM_TASKS_PATH.parent / "msp_db.json")
SINGLE = ItsmDB.load(itsm_environment.ITSM_TASKS_PATH.parent / "single_tenant_db.json")


def _ongoing(db: ItsmDB, service: str) -> list:
    return [i for i in db.incident.values() if i.service == service and i.status not in ("resolved", "closed", "canceled")]


def test_scope_is_the_eight_proposal_candidates_and_nothing_enters_a_split() -> None:
    items = sorted([e["candidate_item"] for e in DATA["tasks"]] + list(ISOLATED))
    assert items == list(range(1, 9))
    assert [e["candidate_item"] for e in DATA["tasks"]] == [2]
    assert not set(TASKS) & set(load_itsm_tasks())
    assert not set(TASKS) & set(load_manifest_v2()["tasks"])
    assert not (REPO_ROOT / "configs" / "data" / "itsm_split_v3.json").exists()
    for entry in DATA["tasks"]:
        assert entry["route"] == "faithful" and entry["intended_split"] == "heldout"
        assert entry["source"]["initial_state_vs_upstream"]["delta_size_after_bench2_delta"] == 0


def test_heldout_candidates_are_not_used_for_token_or_budget_measurements() -> None:
    ids = {e["id"] for e in DATA["tasks"]} | {i["attempted_conversion"]["id"] for i in DATA["isolated"] if "attempted_conversion" in i}
    for path in ("docs/itsm_step6/token_measurements.json", "docs/itsm_step6c/token_measurements_v2.json"):
        text = (REPO_ROOT / path).read_text(encoding="utf-8")
        assert not [i for i in ids if i in text]


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
    assert sorted(r["name"] for r in on_initial if r["passed"]) == sorted(entry["initial_pass_allowed"])
    assert not all(r["passed"] for r in on_initial)


@pytest.mark.parametrize(("task_id", "case_name"), CASES)
def test_validation_case_fails_bench2_as_expected(task_id: str, case_name: str) -> None:
    task, entry = TASKS[task_id], ENTRIES[task_id]
    case = next(c for c in entry["validation_cases"] if c["name"] == case_name)
    env = run_actions(task, edited_actions(task, case["edit"]))
    score = score_final_db(task, env, text_match=FUZZY)

    assert score.db_reward == 0.0
    assert any(case["bench2_mismatch"] in m for m in score.mismatches), score.mismatches
    assert all(r["passed"] for r in run_upstream_verifiers(env.tools.db, entry["upstream_verifiers"])) is case["upstream_all_pass"]


@pytest.mark.parametrize("task_id", IDS)
def test_every_core_requirement_has_a_targeted_negative(task_id: str) -> None:
    entry = ENTRIES[task_id]
    cases = {c["name"]: c for c in entry["validation_cases"]}
    for requirement in entry["core_requirements"]:
        assert requirement["negatives"] and set(requirement["negatives"]) <= set(cases)
        if requirement["free_text"]:
            assert any(cases[n]["kind"] == "key_fact" for n in requirement["negatives"])


def test_sap_erp_decision_needs_the_client_orgs_and_matches_the_unsliced_db() -> None:
    task = TASKS["itsmbench_eops_99ba2325_54f35603"]
    with task_clock(task):
        scoped = build_task_environment(task).tools.db
    assert len(_ongoing(scoped, "SVC_017")) == len(_ongoing(MSP, "SVC_017")) == 4
    # Provider-only scope would show 2 and flip the decision to retirement.
    assert len([i for i in _ongoing(MSP, "SVC_017") if i.org_id == "ORG_001"]) == 2


def test_heldout_gate_for_the_accepted_set() -> None:
    result = heldout_gate_check(DATA["tasks"])
    assert result["reference"]["sizes"] == {"train": 36, "dev": 13, "heldout": 11}
    for tid, row in result["tasks"].items():
        assert row["gate_ok"], (tid, row["checks"])
        assert row["component_reference_members"] == []  # a new held-out family of its own


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
    private = [task.scenario.task_description, task.scenario.simulator_guidance, task_id, entry["source"]["upstream_task_id"],
               "EnterpriseOps", *task.evaluation_criteria.nl_assertions, *(v["query"] for v in entry["upstream_verifiers"])]
    assert not [text for text in private if text in visible]


@pytest.mark.parametrize("item", [1, 5, 7])
def test_isolated_after_conversion_key_fact_negatives_score_db_one(item: int) -> None:
    entry = ISOLATED[item]["attempted_conversion"]
    task = Task.model_validate(entry)
    assert score_final_db(task, replay_gold_actions(task), text_match=FUZZY).db_reward == 1.0
    key_fact = [c for c in entry["validation_cases"] if c["kind"] == "key_fact"]
    assert key_fact
    for case in key_fact:
        env = run_actions(task, edited_actions(task, case["edit"]))
        assert score_final_db(task, env, text_match=FUZZY).db_reward == 1.0, case["name"]
    for case in entry["validation_cases"]:
        if case["kind"] == "core":
            env = run_actions(task, edited_actions(task, case["edit"]))
            assert score_final_db(task, env, text_match=FUZZY).db_reward == 0.0, case["name"]


def test_item_7_shares_two_gold_entities_with_train_tasks() -> None:
    entry = ISOLATED[7]["attempted_conversion"]
    overlap = heldout_gate_check([entry])["tasks"][entry["id"]]["entity_overlap_ge2"]
    assert {(o["task"], tuple(o["same_seed_entities"])) for o in overlap} == {
        ("itsmbench_single_003", ("LOC_003", "USER_002")),
        ("itsmbench_single_005", ("LOC_001", "USER_002")),
    }


def test_isolated_before_conversion_seed_facts_hold() -> None:
    assert {i: ISOLATED[i]["reason"] for i in (3, 4, 6, 8)} == {
        3: "prompt contradicts itself and the expected incident is not unique",
        4: "upstream expected state conflicts with mandatory policy and is not requested",
        6: "prompt and verifier contradict each other",
        8: "ambiguous number in the prompt resolved only by the verifier",
    }
    ongoing = _ongoing(MSP, "SVC_041")
    assert len(ongoing) == 9
    assert sorted(i.incident_id for i in ongoing if i.priority == "high") == ["INC_026", "INC_068"]
    assert {MSP.incident[k].status for k in ("INC_015", "INC_064")} == {"new"}
    assert MSP.users["USER_039"].role == "admin"
    assert SINGLE.incident["INC_021"].assigned_to == "USER_008"
    assert MSP.configuration_item["CI_092"].cost == 800.0
