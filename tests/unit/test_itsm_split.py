"""Frozen ITSM train/dev/held-out split and task-family audit (step 6). Offline."""

from __future__ import annotations

import copy
import itertools
import json

import pytest

from agent_rl.data.itsm_split import (
    DEFAULT_MANIFEST_PATH,
    SPLITS,
    build_families,
    build_manifest,
    link_reason,
    load_manifest,
    load_raw_tasks,
    split_task_ids,
    task_features,
    validate_manifest,
)


MANIFEST = json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
FEATURES = task_features(load_raw_tasks())


def test_committed_manifest_is_reproducible() -> None:
    assert build_manifest() == MANIFEST


def test_splits_are_disjoint_cover_all_tasks_and_keep_families_whole() -> None:
    validate_manifest(MANIFEST)
    sets = {split: set(MANIFEST["splits"][split]) for split in SPLITS}

    for a, b in itertools.combinations(SPLITS, 2):
        assert not sets[a] & sets[b]
    assert set().union(*sets.values()) == set(FEATURES)
    assert sum(len(s) for s in sets.values()) == 53
    for family, entry in MANIFEST["families"].items():
        assert {MANIFEST["tasks"][t]["split"] for t in entry["tasks"]} == {entry["split"]}, family


def test_every_linked_pair_shares_a_split() -> None:
    linked = 0
    for a, b in itertools.combinations(sorted(FEATURES), 2):
        if link_reason(FEATURES[a], FEATURES[b]) is not None:
            linked += 1
            assert MANIFEST["tasks"][a]["split"] == MANIFEST["tasks"][b]["split"], (a, b)
    assert linked > 0


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("itsmbench_msp_021", "itsmbench_msp_022"),
        ("itsmbench_msp_025", "itsmbench_msp_026"),
        ("itsmbench_msp_027", "itsmbench_msp_028"),
        ("itsmbench_msp_021", "itsmbench_single_019"),  # cross-tenancy twin
        ("itsmbench_msp_027", "itsmbench_single_017"),  # cross-tenancy, shared creation tool
    ],
)
def test_near_duplicates_share_a_family(a: str, b: str) -> None:
    assert MANIFEST["tasks"][a]["family"] == MANIFEST["tasks"][b]["family"]
    assert MANIFEST["tasks"][a]["split"] == MANIFEST["tasks"][b]["split"]


def test_families_match_the_curated_labels() -> None:
    families = build_families(FEATURES)

    assert {name: len(tasks) for name, tasks in families.items()} == {
        name: len(entry["tasks"]) for name, entry in MANIFEST["families"].items()
    }
    assert sorted(len(t) for t in families.values()) == [1, 1, 1, 1, 1, 1, 2, 5, 11, 12, 17]


def test_heldout_meets_the_diversity_constraints() -> None:
    heldout = MANIFEST["distribution"]["heldout"]

    assert len(heldout["family"]) >= 5
    assert len(heldout["business_type"]) >= 3
    assert max(heldout["family"].values()) <= 0.5 * heldout["tasks"]
    for split in ("dev", "heldout"):
        assert min(MANIFEST["distribution"][split]["tenancy"].values()) >= 2


def test_manifest_records_data_checksums() -> None:
    files = MANIFEST["data"]["files_sha256"]

    assert {"tasks.json", "msp_db.json", "single_tenant_db.json", "policy.md"} <= set(files)
    assert files["tasks.json"].startswith("e82c54a25edf334e")


@pytest.mark.parametrize("tamper", ["overlap", "missing", "family_split", "data"])
def test_validation_rejects_a_tampered_manifest(tamper: str) -> None:
    manifest = copy.deepcopy(MANIFEST)
    task = MANIFEST["families"]["incident_hold_and_notify"]["tasks"][0]
    if tamper == "overlap":
        manifest["splits"]["train"].append(task)
    elif tamper == "missing":
        manifest["splits"]["dev"].remove(task)
    elif tamper == "family_split":
        manifest["tasks"][task]["split"] = "train"
    else:
        manifest["data"]["files_sha256"]["tasks.json"] = "0" * 64

    with pytest.raises(ValueError):
        validate_manifest(manifest)


def test_heldout_requires_an_explicit_final_evaluation_purpose() -> None:
    with pytest.raises(PermissionError):
        split_task_ids("heldout")

    assert split_task_ids("heldout", final_evaluation=True) == MANIFEST["splits"]["heldout"]
    assert split_task_ids("train") == MANIFEST["splits"]["train"]
    assert load_manifest() == MANIFEST
