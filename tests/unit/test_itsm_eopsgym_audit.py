"""Committed EnterpriseOps-Gym ITSM audit (step 6A). Reads the committed result only;
the upstream reference data is not needed and nothing upstream is imported."""

from __future__ import annotations

import json
from pathlib import Path

from agent_rl.data.itsm_split import DEFAULT_MANIFEST_PATH


REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT = json.loads((REPO_ROOT / "docs" / "itsm_step6a" / "eopsgym_itsm_audit.json").read_text(encoding="utf-8"))
TASKS = AUDIT["tasks"]
RULES = AUDIT["classification_rules"]


def test_provenance_is_pinned() -> None:
    provenance = AUDIT["provenance"]

    assert provenance["commit"] == "271f2c357f763376997dfd16807fcde2474ae41b"
    assert provenance["hf_revision"] == "c8e538eae8a6205294f0a86675fefdc1fac408f6"
    assert set(provenance["parquet_sha256"]) == {"oracle", "plus_5_tools", "plus_10_tools", "plus_15_tools"}
    assert len(provenance["itsm_seed_sha256"]) == 9


def test_tool_set_modes_are_one_task_set_of_103() -> None:
    modes = AUDIT["modes"]

    assert modes["same_task_ids_in_all_modes"] and modes["verifier_multisets_equal_across_modes"]
    assert all(v["oracle_tools_always_included"] for v in modes["tool_sets"].values())
    assert AUDIT["dedup"]["independent_task_ids"] == len(TASKS) == 103
    assert not AUDIT["dedup"]["exact_content_duplicate_groups"]


def test_categories_follow_the_issue_rules() -> None:
    for task_id, task in TASKS.items():
        issues = set(task["issues"])
        if issues & set(RULES["blocking_not_suitable_now"]):
            expected = "not_suitable_now"
        elif issues & set(RULES["minor_adaptation"]):
            expected = "minor_adaptation"
        else:
            expected = "direct"
        assert task["category"] == expected, task_id
    assert sum(AUDIT["summary"]["categories"].values()) == 103


def test_pilot_is_training_side_only_and_meets_its_checks() -> None:
    manifest = json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
    pilot = AUDIT["pilot"]

    assert 5 <= len(pilot) <= 10
    for task_id, entry in pilot.items():
        task = TASKS[task_id]
        assert all(entry["checks"].values()), task_id
        assert task["category"] != "not_suitable_now"
        assert not set(task["linked_v1_family_splits"]) & {"dev", "heldout"}
        assert task["max_shared_write_tools_with_dev_or_heldout_task"]["count"] <= 1
        for link in task["links_to_bench2"]:
            if link["family_link"]:
                assert manifest["tasks"][link["bench2_task"]]["split"] == "train"


def test_known_verifier_defects_are_classified_not_suitable() -> None:
    for suffix in ("391f1eff_ad65c160", "e6427839_6413728a"):
        task = next(t for k, t in TASKS.items() if k.endswith(suffix))
        assert task["verifier_audit"]["conflicts"] and task["category"] == "not_suitable_now"
    broken = next(t for k, t in TASKS.items() if k.endswith("e6427839_47076c15"))
    assert "verifier_does_not_execute" in broken["issues"]


def test_upstream_prompts_are_stored_as_excerpts_only() -> None:
    assert all("user_prompt" not in t and len(t["user_prompt_excerpt"]) <= 200 for t in TASKS.values())
