"""Step S0: joint SFT/GRPO usage audit (offline; committed files and replays only)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from agent_rl.data.itsm_s0_usage import (
    DERIVED_TRAIN_TASK_IDS,
    S0_CASES,
    SFT_PILOT_CANDIDATES,
    build_s0_audit,
)
from agent_rl.data.itsm_split import DEFAULT_MANIFEST_PATH
from agent_rl.data.itsm_split_v2 import DEFAULT_V2_PATH


REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT = json.loads((REPO_ROOT / "docs" / "itsm_stepS0" / "usage_audit.json").read_text(encoding="utf-8"))
CORE = AUDIT["core_requirements"]
REGISTRY = AUDIT["registry"]
SUSPENDED = {
    "itsmbench_eops_01395ded_eff85dae",
    "itsmbench_eops_46ca2862_b6ef60ca",
    "itsmbench_eops_1d219d93_d1ba6c2d",
    "itsmbench_eops_66392a82_52c8290e",
    "itsmbench_eops_99ba2325_7915481a",
}
SFT_ONLY = SUSPENDED | {"itsmbench_eops_78a5b71e_a4fb07a9"}
R1_PENDING = {f"itsmbench_single_{n:03d}" for n in (5, 6, 7, 11, 15)}


def test_committed_audit_is_reproducible() -> None:
    assert build_s0_audit() == AUDIT


def test_splits_are_not_changed() -> None:
    v2 = json.loads(DEFAULT_V2_PATH.read_text(encoding="utf-8"))
    assert hashlib.sha256(DEFAULT_MANIFEST_PATH.read_bytes()).hexdigest().startswith("77cc5519")
    assert {s: len(v2["splits"][s]) for s in ("train", "dev", "heldout")} == {"train": 32, "dev": 13, "heldout": 11}


def test_key_fact_negatives_score_db_1_and_structured_ones_db_0() -> None:
    for task_id, cases in S0_CASES.items():
        for case in cases:
            result = CORE[task_id]["negatives"][case["name"]]
            if case["kind"] == "key_fact":
                assert result["db_reward"] == 1.0 and result["fuzzy_recall"] >= result["threshold"], case["name"]
            else:
                assert result["db_reward"] == 0.0, case["name"]
    # Every pre-existing validation case still fails Bench2 scoring.
    for entry in CORE.values():
        assert all(c["db_reward"] == 0.0 for c in entry["negatives"].values() if c["source"] == "existing_validation_case")


def test_every_key_negative_is_checked_on_its_own() -> None:
    """S0-R1: a requirement is discriminated only if every listed negative scores DB=0;
    a structured DB=0 never covers another confirmed core error that scores DB=1."""
    assert set(CORE) == set(DERIVED_TRAIN_TASK_IDS)
    for entry in CORE.values():
        for requirement in entry["core_requirements"]:
            rewards = requirement["negative_db_rewards"]
            assert requirement["discriminated"] is all(v == 0.0 for v in rewards.values())
            assert requirement["missed_negatives"] == [n for n in requirement["negatives"] if rewards[n] != 0.0]
            # Every miss is a fuzzy blind spot, never a structured field.
            assert all(entry["negatives"][n]["kind"] == "key_fact" for n in requirement["missed_negatives"])


def test_grpo_is_suspended_exactly_where_a_core_requirement_is_missed() -> None:
    suspended = {t for t, e in CORE.items() if e["grpo"]["proposed"] != "eligible"}
    assert suspended == SUSPENDED
    for task_id, entry in CORE.items():
        missed = [r for r in entry["core_requirements"] if not r["discriminated"]]
        assert bool(missed) is (task_id in SUSPENDED)
        assert entry["sft"]["proposed"] == "eligible_pending_trajectory_review"


def test_b6ef60ca_software_removal_negative_suspends_grpo() -> None:
    entry = CORE["itsmbench_eops_46ca2862_b6ef60ca"]
    removal = entry["negatives"]["s0_subject_says_removal"]
    assert removal["kind"] == "key_fact" and removal["tool_calls_succeeded"]
    assert removal["db_reward"] == 1.0 and removal["fuzzy_recall"] == 0.5
    case = next(c for c in S0_CASES["itsmbench_eops_46ca2862_b6ef60ca"] if c["name"] == "s0_subject_says_removal")
    assert (case["field"], case["value"]) == ("short_description", "Software removal")  # category stays software
    facts = {r["id"]: r for r in entry["core_requirements"]}
    assert facts["fact_installation_not_removal"]["missed_negatives"] == ["s0_subject_says_removal"]
    # The structured category negative still fails, but no longer covers the subject miss.
    software = facts["fact_software_not_hardware"]
    assert software["negative_db_rewards"] == {"s0_category_hardware": 0.0, "s0_subject_says_hardware": 1.0}
    assert not software["discriminated"]
    assert entry["grpo"]["proposed"] == "suspended_reward_blind_spot"


def test_training_union_is_isolated_from_dev_and_heldout() -> None:
    joint = AUDIT["joint_isolation"]
    assert all(joint["checks"].values()), joint["checks"]
    assert joint["training_union"] == {"tasks": 37, "bench2": 29, "derived": 8}
    assert joint["evaluation"] == {"dev": 13, "heldout": 12}
    assert not [p for p in joint["pairs_flagged"] if p["train_origin"] == "derived"]


def test_registry_pools_and_exclusions() -> None:
    rows = {r["id"]: r for r in REGISTRY["tasks"]}
    pools = REGISTRY["pools"]
    assert len(pools["grpo"]) == 31 and len(pools["sft"]) == 37
    assert set(pools["sft"]) - set(pools["grpo"]) == set(pools["sft_only"]) == SFT_ONLY
    assert set(pools["grpo_pending_r1_before_step9"]) == R1_PENDING <= set(pools["grpo"])
    for task_id in R1_PENDING:
        assert rows[task_id]["grpo"] == "eligible_pending_r1_before_step_9" and rows[task_id]["use"] == "both"
    assert rows["itsmbench_eops_78a5b71e_a4fb07a9"]["use"] == "sft_only" and rows["itsmbench_eops_78a5b71e_a4fb07a9"]["grpo"] == "no"
    held = rows["itsmbench_eops_99ba2325_54f35603"]
    assert held["use"] == "heldout_candidate" and held["grpo"] == held["sft"] == "no"
    families = REGISTRY["pool_families"]
    assert (families["grpo"]["tasks"], families["grpo"]["sizes"]) == (31, [29, 1, 1])
    assert (families["sft"]["tasks"], families["sft"]["sizes"]) == (37, [32, 1, 1, 1, 1, 1])
    for task_id in pools["grpo"] + pools["sft"]:
        assert rows[task_id]["use"] in ("both", "sft_only")
    assert not {t for t, r in rows.items() if r["use"] in ("dev", "heldout", "heldout_candidate")} & set(pools["sft"])
    excluded = {e["upstream"] for e in REGISTRY["excluded_converted_or_reviewed"]}
    assert len(excluded) == 14
    assert all(e["sft"] == e["grpo"] == "no" for e in REGISTRY["excluded_converted_or_reviewed"])


def test_pilot_candidates_are_audited_training_sources() -> None:
    pilot = AUDIT["sft_pilot_candidates"]
    required = [c for c in pilot if not c["optional"]]
    assert 4 <= len(required) <= 6 and len(pilot) <= 6
    assert all(c["max_candidate_trajectories"] == 2 for c in pilot)
    assert all(c["use"] in ("both", "sft_only") for c in required)
    assert [(c["task_id"], c["use"]) for c in pilot if c["optional"]] == [("itsmbench_eops_78a5b71e_a4fb07a9", "sft_only")]
    assert "itsmbench_eops_99ba2325_54f35603" not in {c["task_id"] for c in pilot}
    assert "itsmbench_eops_1d219d93_d1ba6c2d" not in {c["task_id"] for c in SFT_PILOT_CANDIDATES}  # 30k-token unfiltered reads
