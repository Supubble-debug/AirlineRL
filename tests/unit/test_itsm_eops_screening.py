"""Step 6D-1 candidate screening: invariants of the committed screening result.

Reads docs/itsm_step6d/candidate_screening.json only (regenerating it needs the
pinned tokenizer: scripts/data/screen_itsm_eops_candidates.py --check).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agent_rl.data.itsm_split import write_tool_names

REPO_ROOT = Path(__file__).resolve().parents[2]
SCREENING = json.loads((REPO_ROOT / "docs" / "itsm_step6d" / "candidate_screening.json").read_text(encoding="utf-8"))
AUDIT = json.loads((REPO_ROOT / "docs" / "itsm_step6a" / "eopsgym_itsm_audit.json").read_text(encoding="utf-8"))
BATCH = SCREENING["batch"]
CANDIDATES = SCREENING["candidates"]


@pytest.mark.parametrize("name", ["audit", "split_v1", "split_v2", "pilot"])
def test_screening_was_built_from_the_current_frozen_inputs(name: str) -> None:
    entry = SCREENING["inputs"][name]
    assert hashlib.sha256((REPO_ROOT / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]


def test_candidates_are_the_audit_tasks_minus_the_6b_pilot() -> None:
    assert set(CANDIDATES) == set(AUDIT["tasks"]) - set(AUDIT["pilot"])
    assert len(CANDIDATES) == 95


def test_batch_size_and_routes() -> None:
    assert 8 <= len(BATCH) <= 12
    for entry in BATCH:
        suffix = "_".join(entry["upstream_task_id"].split("_")[-2:])
        if entry["route"] == "faithful":
            assert entry["task_id"] == f"itsmbench_eops_{suffix}"
            assert "adaptations" not in entry
        else:
            assert entry["route"] == "adapted"
            assert entry["task_id"] == f"itsmbench_eopsadapt_{suffix}"
            assert entry["adaptations"] and all(a["change"] and a["reason"] for a in entry["adaptations"])
            assert "no equivalence claimed" in " ".join(entry["acceptance"])


def test_batch_never_reuses_a_pilot_task() -> None:
    assert not {entry["upstream_task_id"] for entry in BATCH} & set(AUDIT["pilot"])


def test_batch_tasks_pass_the_train_gate_with_their_planned_writes() -> None:
    writes = write_tool_names()
    for entry in BATCH:
        assert not CANDIDATES[entry["upstream_task_id"]]["train_gate_failures"], entry["task_id"]
        assert set(entry["planned_write_tools"]) <= writes
        assert entry["planned_gate_ok"], entry["task_id"]
        relations = entry["planned_relations_v2"]
        assert set(relations["linked_splits"]) <= {"train"}
        assert all(relations["max_shared_write_tools"].get(s, {"count": 0})["count"] <= 1 for s in ("dev", "heldout"))
        assert not set(relations["entity_overlap_ge2"]) & {"dev", "heldout"}


def test_batch_tasks_are_not_linked_to_each_other() -> None:
    assert SCREENING["batch_internal_links"] == {}


def test_every_batch_task_has_negatives_policy_notes_and_bounded_reads() -> None:
    for entry in BATCH:
        assert len(entry["negatives"]) >= 3, entry["task_id"]
        assert entry["policy"], entry["task_id"]
        reads = entry["length"]["unfiltered_read_under_scope"]
        assert reads["over_max_response_length"] == [], entry["task_id"]
        assert entry["scope"]["org_ids"], "every batch task is org-scoped"


def test_every_train_gate_passer_and_only_those_carry_a_manual_verdict() -> None:
    for task_id, row in CANDIDATES.items():
        if row["train_gate_failures"]:
            assert row["manual"] is None, task_id
        else:
            assert row["manual"] and row["manual"]["verdict"], task_id
    verdicts = [row["manual"]["verdict"] for row in CANDIDATES.values() if row["manual"]]
    assert verdicts.count("selected") == len(BATCH)


def test_msp_incident_creation_is_not_selected_before_the_sla_decision() -> None:
    for entry in BATCH:
        flags = CANDIDATES[entry["upstream_task_id"]]["policy_flags"]
        if "incident_creation" in flags:
            assert flags["incident_creation"]["sla_status"] == "unique"
            assert entry["scope"]["seed_db"] == "single_tenant_db.json"
    assert {d["id"] for d in SCREENING["pending_decisions"]} == {"D6D-1", "D6D-2"}


def test_v3_heldout_pool_excludes_the_batch() -> None:
    batch_suffixes = {"_".join(e["upstream_task_id"].split("_")[-2:]) for e in BATCH}
    assert not set(SCREENING["v3_inputs"]["heldout_gate_passers"]) & batch_suffixes
