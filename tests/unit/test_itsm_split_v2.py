"""ITSM split v2: v1 plus three accepted EnterpriseOps-Gym-derived train tasks (step 6C).

Offline: committed files only (no upstream checkout, tokenizer, model or LLM).
"""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from agent_rl.data.itsm_eops_pilot import load_pilot
from agent_rl.data.itsm_split import DEFAULT_MANIFEST_PATH, SPLITS, load_manifest, load_raw_tasks
from agent_rl.data.itsm_split_v2 import (
    ACCEPTED_DERIVED_TASK_IDS,
    DEFAULT_V2_PATH,
    SOURCE_DERIVED,
    build_manifest_v2,
    load_manifest_v2,
    task_content_sha256,
    validate_manifest_v2,
)


V1 = load_manifest()
V2 = json.loads(DEFAULT_V2_PATH.read_text(encoding="utf-8"))
#: sha256 of configs/data/itsm_split_v1.json as committed in step 6 (c757c01).
V1_MANIFEST_SHA256 = "77cc55196b01f8429d17b1b89c8fcf61c4eae4757db0103511941a0c88631beb"
MERGED = "onboarding_catalog_and_change_problem_kb_merged"
NEW = "incident_intake_with_ci_update"


def test_committed_v2_manifest_is_reproducible_and_valid() -> None:
    assert V2 == build_manifest_v2()
    assert load_manifest_v2() == V2


def test_v1_manifest_is_kept_unchanged() -> None:
    assert hashlib.sha256(DEFAULT_MANIFEST_PATH.read_bytes()).hexdigest() == V1_MANIFEST_SHA256
    assert V2["data"]["base_manifest"]["sha256"] == V1_MANIFEST_SHA256
    assert V1["version"] == "itsm_split_v1" and V2["version"] == "itsm_split_v2"


def test_sizes_and_unchanged_dev_and_heldout() -> None:
    assert {s: len(V2["splits"][s]) for s in SPLITS} == V2["sizes"] == {"train": 32, "dev": 13, "heldout": 11}
    assert len(V2["tasks"]) == 56
    assert V2["splits"]["dev"] == V1["splits"]["dev"]
    assert V2["splits"]["heldout"] == V1["splits"]["heldout"]
    assert set(V2["splits"]["train"]) == set(V1["splits"]["train"]) | set(ACCEPTED_DERIVED_TASK_IDS)


def test_original_53_task_definitions_are_unchanged() -> None:
    raw = {task["id"]: task for task in load_raw_tasks()}

    assert V2["data"]["bench2"] == V1["data"]
    assert {t for t, e in V2["tasks"].items() if e["source"] == "bench2"} == set(V1["tasks"]) == set(raw)
    for task_id, entry in V1["tasks"].items():
        v2 = V2["tasks"][task_id]
        assert v2["content_sha256"] == task_content_sha256(raw[task_id])
        assert (v2["split"], v2["business_type"], v2["tenancy"], v2["seed_db"]) == (
            entry["split"], entry["business_type"], entry["tenancy"], entry["seed_db"]
        )
        assert v2["org_slicing"] is True  # every Bench2 task is org-sliced


def test_derived_tasks_are_labelled_with_provenance_and_differences() -> None:
    pilot = {e["id"]: e for e in load_pilot()["tasks"]}

    for task_id in ACCEPTED_DERIVED_TASK_IDS:
        entry = V2["tasks"][task_id]
        derived = entry["derived_from"]
        source = pilot[task_id]["source"]
        assert entry["source"] == SOURCE_DERIVED and entry["split"] == "train"
        assert entry["content_sha256"] == task_content_sha256(pilot[task_id])
        assert entry["org_slicing"] is False
        assert derived["dataset"] == "EnterpriseOps-Gym"
        assert derived["upstream_task_id"] == source["upstream_task_id"]
        assert (derived["commit"], derived["hf_revision"], derived["license"]) == (
            "271f2c357f763376997dfd16807fcde2474ae41b",
            "c8e538eae8a6205294f0a86675fefdc1fac408f6",
            "Apache-2.0",
        )
        assert derived["multi_turn_rewrite"] is True
        assert derived["equivalent_to_upstream_evaluation"] is False
        assert derived["conversion_notes"] == pilot[task_id]["conversion_notes"]
        assert "Not equivalent" in derived["scoring_differences"]
        assert derived["accepted"]["date"] == "2026-10-30"
    # Only the accepted three: the isolated five are not in v2.
    isolated = {i["upstream_task_id"] for i in load_pilot()["isolated"]}
    assert not isolated & {e.get("derived_from", {}).get("upstream_task_id") for e in V2["tasks"].values()}


def test_families_are_recomputed_over_56_tasks() -> None:
    v1_members = {name: set(e["tasks"]) for name, e in V1["families"].items()}
    families = V2["families"]
    changes = V2["family_changes"]

    merged = set(families[MERGED]["tasks"])
    assert merged == (
        v1_members["onboarding_and_catalog_admin"]
        | v1_members["change_problem_advance_with_kb"]
        | {"itsmbench_eops_1d219d93_d1ba6c2d", "itsmbench_eops_46ca2862_b6ef60ca"}
    )
    assert families[MERGED]["origin"] == "merged_v1_families" and families[MERGED]["split"] == "train"
    assert families[NEW]["tasks"] == ["itsmbench_eops_66392a82_52c8290e"]
    assert families[NEW]["origin"] == "new_in_v2" and families[NEW]["v1_families"] == []
    # The two merged v1 names are gone; every kept v1 name has exactly its v1 members.
    assert set(changes["v1_families_no_longer_present"]) == {"onboarding_and_catalog_admin", "change_problem_advance_with_kb"}
    assert not set(changes["v1_families_no_longer_present"]) & set(families)
    for name in changes["unchanged_from_v1"]:
        assert set(families[name]["tasks"]) == v1_members[name]
        assert families[name]["split"] == V1["families"][name]["split"]
    assert (changes["v1_family_count"], changes["v2_family_count"]) == (11, 11)
    assert V2["distribution"]["train"]["families"] == 3
    assert V2["distribution"]["train"]["family"][MERGED] == 30


def test_distribution_matches_the_task_map() -> None:
    for split in SPLITS:
        ids = V2["splits"][split]
        dist = V2["distribution"][split]
        assert dist["tasks"] == len(ids)
        assert sum(dist["source"].values()) == sum(dist["family"].values()) == len(ids)
        assert dist["gold_actions_total"] == sum(V2["tasks"][t]["gold_actions"] for t in ids)
    assert V2["distribution"]["train"]["source"] == {"bench2": 29, SOURCE_DERIVED: 3}


def _swap_lists(m):
    train, dev = m["splits"]["train"], m["splits"]["dev"]
    i, j = 0, 0
    train[i], dev[j] = dev[j], train[i]


def _move_new_family_to_dev(m):
    task = "itsmbench_eops_66392a82_52c8290e"
    m["splits"]["train"].remove(task)
    m["splits"]["dev"] = sorted([*m["splits"]["dev"], task])
    m["tasks"][task]["split"] = "dev"
    m["families"][NEW]["split"] = "dev"


TAMPERS = {
    "swap_ids_between_lists_only": (_swap_lists, "map says"),
    "duplicate_id_in_one_list": (lambda m: m["splits"]["train"].append(m["splits"]["train"][0]), "more than once"),
    "same_id_in_two_lists": (lambda m: m["splits"]["heldout"].append(m["splits"]["train"][0]), "is in splits"),
    "id_omitted_from_lists": (lambda m: m["splits"]["train"].remove("itsmbench_eops_46ca2862_b6ef60ca"), "omit"),
    "unknown_id_in_list": (lambda m: m["splits"]["train"].append("itsmbench_eops_97d7df7d_a52988e5"), "unknown"),
    "map_split_changed_only": (lambda m: m["tasks"][m["splits"]["dev"][0]].__setitem__("split", "train"), "map says"),
    "task_missing_from_map": (lambda m: m["tasks"].pop("itsmbench_eops_46ca2862_b6ef60ca"), "task map"),
    "family_member_duplicated": (lambda m: m["families"][NEW]["tasks"].append("itsmbench_eops_66392a82_52c8290e"), "more than once"),
    "task_in_two_families": (lambda m: m["families"][NEW]["tasks"].append("itsmbench_eops_46ca2862_b6ef60ca"), "families"),
    "family_member_dropped": (lambda m: m["families"][MERGED]["tasks"].remove("itsmbench_eops_46ca2862_b6ef60ca"), "no family"),
    "task_family_field_wrong": (lambda m: m["tasks"]["itsmbench_eops_46ca2862_b6ef60ca"].__setitem__("family", NEW), "map says family"),
    "family_split_wrong": (lambda m: m["families"][NEW].__setitem__("split", "dev"), "straddles"),
    "derived_moved_to_dev_everywhere": (_move_new_family_to_dev, "sizes"),
    "declared_sizes_wrong": (lambda m: m["sizes"].__setitem__("train", 29), "sizes"),
    "dev_reordered": (lambda m: m["splits"]["dev"].reverse(), "differs from v1"),
    "derived_relabelled_bench2": (lambda m: m["tasks"]["itsmbench_eops_46ca2862_b6ef60ca"].__setitem__("source", "bench2"), "derived tasks"),
    "content_checksum": (lambda m: m["tasks"]["itsmbench_msp_001"].__setitem__("content_sha256", "0" * 64), "checksum"),
    "pilot_file_checksum": (lambda m: m["data"]["derived_source"].__setitem__("pilot_file_sha256", "0" * 64), "changed"),
    "family_label_stale": (lambda m: m["families"].__setitem__("onboarding_and_catalog_admin", m["families"].pop(MERGED)), "map says family"),
    "distribution_edited": (lambda m: m["distribution"]["train"].__setitem__("families", 4), "rebuild"),
    "wrong_version": (lambda m: m.__setitem__("version", "itsm_split_v1"), "expected"),
}


@pytest.mark.parametrize("tamper", sorted(TAMPERS))
def test_validation_rejects_an_inconsistent_v2_manifest(tamper: str) -> None:
    edit, reason = TAMPERS[tamper]
    manifest = copy.deepcopy(V2)
    edit(manifest)

    with pytest.raises(ValueError, match=reason):
        validate_manifest_v2(manifest)


def test_builder_refuses_a_stale_v1_name_or_an_unlabelled_family(monkeypatch) -> None:
    from agent_rl.data import itsm_split_v2

    labels = dict(itsm_split_v2.V2_FAMILY_LABELS)
    anchor = "itsmbench_eops_1d219d93_d1ba6c2d"
    monkeypatch.setattr(itsm_split_v2, "V2_FAMILY_LABELS", {**labels, anchor: ("onboarding_and_catalog_admin", "")})
    with pytest.raises(ValueError, match="reuses a v1 family name"):
        build_manifest_v2()

    monkeypatch.setattr(itsm_split_v2, "V2_FAMILY_LABELS", {k: v for k, v in labels.items() if k != anchor})
    with pytest.raises(ValueError, match="no curated label"):
        build_manifest_v2()
