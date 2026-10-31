"""EnterpriseOps-Gym ITSM migration pilot (step 6B). Offline; uses the committed pilot
file only (no upstream checkout, tokenizer, model or LLM)."""

from __future__ import annotations

import json

import pytest

from enterprise_worlds.data_model.message import UserMessage
from enterprise_worlds.domains.itsm import environment as itsm_environment
from enterprise_worlds.domains.itsm.data_model import ItsmDB
from enterprise_worlds.domains.itsm.tools import ItsmTools
from enterprise_worlds.evaluator.text_match_strategy import TextMatchConfig

from agent_rl.data.itsm_eops_pilot import (
    edited_actions,
    family_overlap_check,
    load_pilot,
    pilot_tasks,
    run_actions,
    run_upstream_verifiers,
)
from agent_rl.data.itsm_split import load_manifest
from agent_rl.envs.itsm_env import ItsmEnv, ItsmEnvConfig
from agent_rl.envs.itsm_scoring import (
    build_task_environment,
    load_itsm_tasks,
    replay_gold_actions,
    score_final_db,
    task_clock,
)


PILOT = load_pilot()
TASKS = pilot_tasks()
ENTRIES = {e["id"]: e for e in PILOT["tasks"]}
FUZZY = TextMatchConfig(strategy="fuzzy")
IDS = sorted(TASKS)
CASES = [(tid, case["name"]) for tid in IDS for case in ENTRIES[tid]["validation_cases"]]


def test_pilot_is_separate_from_bench2_and_split_v1() -> None:
    bench = load_itsm_tasks()
    manifest = load_manifest()

    assert len(TASKS) == 3 and len(PILOT["isolated"]) == 5
    assert not set(TASKS) & set(bench)
    assert len(bench) == 53 and not set(TASKS) & set(manifest["tasks"])
    assert all(e["source"]["commit"] == "271f2c357f763376997dfd16807fcde2474ae41b" for e in PILOT["tasks"])
    assert all(e["source"]["initial_state_vs_upstream"]["delta_size_after_bench2_delta"] == 0 for e in PILOT["tasks"])


@pytest.mark.parametrize("task_id", IDS)
def test_gold_passes_bench2_and_upstream_verifiers(task_id: str) -> None:
    task, entry = TASKS[task_id], ENTRIES[task_id]
    gold = replay_gold_actions(task)
    with task_clock(task):
        initial = build_task_environment(task)

    assert score_final_db(task, gold, text_match=FUZZY).db_reward == 1.0
    assert all(r["passed"] for r in run_upstream_verifiers(gold.tools.db, entry["upstream_verifiers"]))
    # The upstream verifiers discriminate: none passes before any action.
    assert not any(r["passed"] for r in run_upstream_verifiers(initial.tools.db, entry["upstream_verifiers"]))


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


def test_cross_source_family_and_overlap_check_keeps_dev_and_heldout() -> None:
    result = family_overlap_check(PILOT["tasks"])

    assert all(result["checks"].values()), result["checks"]
    assert not result["bridged_v1_families"]


@pytest.mark.parametrize("task_id", IDS)
def test_pilot_runs_in_itsm_env_without_leaking_private_fields(task_id: str) -> None:
    task, entry = TASKS[task_id], ENTRIES[task_id]

    class OneReply:
        def get_init_state(self):
            return 0

        def generate_next_message(self, agent_message, state):
            return UserMessage(content="Hi, I need some help."), state + 1

    config = ItsmEnvConfig(task_id=task_id, text_match=FUZZY, max_steps=40)
    reset = ItsmEnv(config, user=OneReply(), task=task).reset()
    visible = reset.observation + json.dumps(reset.public_info)

    private = [
        task.scenario.task_description,
        task.scenario.simulator_guidance,
        task_id,
        entry["source"]["upstream_task_id"],
        *task.evaluation_criteria.nl_assertions,
        *(v["query"] for v in entry["upstream_verifiers"]),
    ]
    assert not [text for text in private if text in visible]
    with pytest.raises(ValueError, match="does not match"):
        ItsmEnv(ItsmEnvConfig(task_id="other", text_match=FUZZY, max_steps=40), user=OneReply(), task=task)


def test_isolation_evidence_still_holds() -> None:
    """The recorded reasons for isolating five tasks, re-checked on Bench2 data."""
    single = ItsmDB.load(itsm_environment.ITSM_TASKS_PATH.parent / "single_tenant_db.json")
    msp = ItsmDB.load(itsm_environment.ITSM_TASKS_PATH.parent / "msp_db.json")
    isolated = {i["upstream_task_id"][-17:]: i for i in PILOT["isolated"]}

    assert set(isolated) == {"97d7df7d_a52988e5", "66392a82_74b17170", "e6427839_ca9119ec", "46ca2862_796e7685", "66392a82_fd509f97"}
    # a52988e5: a plausible but not certain knowledge match exists.
    assert single.knowledge["KB_002"].title == "Email Client Configuration Guide"
    # ca9119ec: GROUP_001 still handles active incidents, so §12.3 blocks dissolution.
    assert {k for k, i in single.incident.items() if i.assignment_group == "GROUP_001" and i.status not in ("resolved", "closed", "canceled")} == {"INC_003", "INC_004", "INC_014", "INC_016"}
    # 796e7685 / fd509f97: two ORG_001 SLAs per priority in the MSP world.
    for priority in ("moderate", "low"):
        assert len([s for s in msp.sla_definition.values() if s.applies_to_priority == priority and s.org_id == "ORG_001"]) == 2
    # fd509f97: a user_role row to remove, and no tool that removes user_role rows.
    assert "USER_011:ROLE_002:ORG_001" in msp.user_role
    assert not [n for n in ItsmTools._tool_names if "role" in n and ("remove" in n or "delete" in n)]
