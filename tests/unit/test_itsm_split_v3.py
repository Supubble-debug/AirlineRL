"""ITSM split v3 manifest, usage registry v1 and purpose-gated task selection.
Offline: committed files, gold replays only (no tokenizer, model or LLM)."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from enterprise_worlds.evaluator.text_match_strategy import TextMatchConfig

from agent_rl.data import itsm_task_sets as ts
from agent_rl.data.itsm_split_v3 import (
    DEFAULT_V3_PATH,
    HELDOUT_ADDED_TASK_IDS,
    V3_TRAIN_ADDITIONS,
    build_manifest_v3,
    load_manifest_v3,
    validate_manifest_v3,
)
from agent_rl.data.itsm_split_v2 import load_manifest_v2
from agent_rl.data.itsm_usage import (
    DEFAULT_USAGE_PATH,
    GRPO_PENDING_R1_TASK_IDS,
    GRPO_SUSPENDED_TASK_IDS,
    build_usage_registry,
    load_usage_registry,
    validate_usage_registry,
)
from agent_rl.envs.itsm_env import ItsmEnvConfig
from agent_rl.envs.itsm_scoring import replay_gold_actions, score_final_db

REPO_ROOT = Path(__file__).resolve().parents[2]
V3 = load_manifest_v3()
V2 = load_manifest_v2()
USAGE = load_usage_registry()
SFT_ONLY = set(GRPO_SUSPENDED_TASK_IDS) | {"itsmbench_eops_78a5b71e_a4fb07a9"}
USAGE_VERSION = "itsm_usage_v1"


# -- manifest ------------------------------------------------------------------------------------


def test_v1_and_v2_are_unchanged() -> None:
    for name, prefix in (("itsm_split_v1.json", "77cc55196b01f842"), ("itsm_split_v2.json", "f5ba77a58184aa49")):
        assert hashlib.sha256((REPO_ROOT / "configs" / "data" / name).read_bytes()).hexdigest().startswith(prefix)


def test_v3_sizes_and_additions() -> None:
    assert V3["sizes"] == {"train": 37, "dev": 13, "heldout": 12}
    assert V3["splits"]["dev"] == V2["splits"]["dev"]
    assert set(V3["splits"]["heldout"]) == set(V2["splits"]["heldout"]) | set(HELDOUT_ADDED_TASK_IDS)
    assert set(V3["splits"]["train"]) == set(V2["splits"]["train"]) | set(V3_TRAIN_ADDITIONS)
    assert all(V3["tasks"][t]["split"] == e["split"] for t, e in V2["tasks"].items())
    assert V3["splits"]["heldout"].count("itsmbench_eops_99ba2325_54f35603") == 1


def test_families_isolation_and_checksums() -> None:
    assert {s: d["families"] for s, d in V3["distribution"].items()} == {"train": 6, "dev": 2, "heldout": 7}
    assert all(V3["isolation"]["checks"].values())
    assert not V3["isolation"]["flagged_pairs"] and not V3["isolation"]["near_duplicates"]
    for task_id, entry in V3["tasks"].items():
        assert len(entry["content_sha256"]) == len(entry["source_sha256"]) == 64
        if entry["source"] != "bench2":
            assert entry["derived_from"]["equivalent_to_upstream_evaluation"] is False
            assert entry["derived_from"]["upstream_user_prompt_sha256"]


def test_committed_manifest_is_reproducible() -> None:
    text = DEFAULT_V3_PATH.read_text(encoding="utf-8")
    assert json.loads(text) == build_manifest_v3()


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        (lambda m: m["splits"]["train"].remove("itsmbench_eops_78a5b71e_a4fb07a9") or m["splits"]["dev"].append("itsmbench_eops_78a5b71e_a4fb07a9"), "listed in dev"),
        (lambda m: m["tasks"]["itsmbench_eops_99ba2325_54f35603"].update(content_sha256="0" * 64), "content checksum"),
        (lambda m: m["tasks"]["itsmbench_single_011"].update(source_sha256="0" * 64), "source checksum"),
        (lambda m: m.update(sizes={"train": 38, "dev": 13, "heldout": 11}), "sizes"),
    ],
)
def test_tampered_manifest_is_rejected(tamper, message: str) -> None:
    manifest = copy.deepcopy(V3)
    tamper(manifest)
    with pytest.raises(ValueError, match=message):
        validate_manifest_v3(manifest)


# -- usage registry ------------------------------------------------------------------------------


def test_usage_counts_and_pools() -> None:
    assert USAGE["counts"] == {
        "sft_sources": 37, "grpo_candidates": 31, "grpo_trainable": 26, "grpo_pending_r1": 5,
        "sft_only": 6, "dev": 13, "heldout": 12,
    }
    pools = {k: set(v) for k, v in USAGE["pools"].items()}
    assert pools["sft_only"] == SFT_ONLY
    assert pools["grpo_pending_r1"] == set(GRPO_PENDING_R1_TASK_IDS)
    assert pools["sft_sources"] == set(V3["splits"]["train"])
    assert not pools["grpo_trainable"] & (pools["sft_only"] | pools["grpo_pending_r1"])


def test_source_approval_is_separate_from_trajectory_acceptance() -> None:
    assert USAGE["trajectory_acceptance"]["accepted"] == [] and USAGE["trajectory_acceptance"]["rejected"] == []
    for task_id, row in USAGE["tasks"].items():
        assert "trajectories" not in row
        if row["split"] == "train":
            assert row["source_approval"]["sft"] == "approved"
        else:
            assert row["source_approval"] == {"sft": "not_a_training_source", "grpo": "not_a_training_source"}
    for task_id in GRPO_PENDING_R1_TASK_IDS:
        assert USAGE["tasks"][task_id]["grpo_status"] == "pending_r1_before_step_9"
        assert USAGE["tasks"][task_id]["source_approval"]["grpo"] != "approved"


def test_committed_registry_is_reproducible_and_tampering_is_rejected() -> None:
    assert json.loads(DEFAULT_USAGE_PATH.read_text(encoding="utf-8")) == build_usage_registry()
    bad = copy.deepcopy(USAGE)
    bad["pools"]["grpo_trainable"].append("itsmbench_single_011")
    with pytest.raises(ValueError, match="GRPO-trainable"):
        validate_usage_registry(bad)
    bad = copy.deepcopy(USAGE)
    bad["pools"]["sft_sources"].append("itsmbench_eops_99ba2325_54f35603")
    with pytest.raises(ValueError):
        validate_usage_registry(bad)


# -- purpose-gated selection (data entry) --------------------------------------------------------


def test_v3_train_cannot_be_listed_as_a_training_set() -> None:
    with pytest.raises(PermissionError, match="training_task_ids"):
        ts.task_ids("train", split_version="itsm_split_v3")
    assert ts.task_ids("dev", split_version="itsm_split_v3") == V3["splits"]["dev"]
    with pytest.raises(PermissionError, match="final frozen evaluation"):
        ts.task_ids("heldout", split_version="itsm_split_v3")
    assert "itsmbench_eops_99ba2325_54f35603" in ts.task_ids("heldout", split_version="itsm_split_v3", final_evaluation=True)


def test_grpo_entry_selects_only_trainable_tasks() -> None:
    grpo = set(ts.training_task_ids(purpose="grpo", usage_version=USAGE_VERSION))
    assert len(grpo) == 26
    assert not grpo & SFT_ONLY
    assert not grpo & set(GRPO_PENDING_R1_TASK_IDS)
    assert not grpo & (set(V3["splits"]["dev"]) | set(V3["splits"]["heldout"]))
    sft = set(ts.training_task_ids(purpose="sft", usage_version=USAGE_VERSION))
    assert sft == set(V3["splits"]["train"]) and SFT_ONLY <= sft
    with pytest.raises(ValueError, match="purpose"):
        ts.training_task_ids(purpose="train", usage_version=USAGE_VERSION)
    with pytest.raises(ValueError, match="usage_version"):
        ts.training_task_ids(purpose="grpo", usage_version="itsm_usage_v0")


def test_every_v3_task_resolves_with_its_checksum() -> None:
    tasks = ts.load_task_set(split_version="itsm_split_v3")
    assert set(tasks) == set(V3["tasks"])


@pytest.mark.parametrize("task_id", ["itsmbench_eops_78a5b71e_a4fb07a9", "itsmbench_eops_99ba2325_54f35603", "itsmbench_eops_01395ded_eff85dae"])
def test_added_tasks_resolve_and_replay_to_db_1(task_id: str) -> None:
    ItsmEnvConfig(task_id=task_id, text_match=TextMatchConfig(strategy="fuzzy"), max_steps=40, split_version="itsm_split_v3")
    task = ts.resolve_task(task_id, split_version="itsm_split_v3")
    assert score_final_db(task, replay_gold_actions(task), text_match=TextMatchConfig(strategy="fuzzy")).db_reward == 1.0
