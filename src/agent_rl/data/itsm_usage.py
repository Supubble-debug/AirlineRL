"""ITSM usage registry v1: which v3 task may serve SFT and/or GRPO.

The v3 manifest says which split a task is in; this registry says what a train
task may be used for. Training entry points must select tasks here (through
``itsm_task_sets.training_task_ids``), never from v3's train list directly:

* ``sft_sources`` (37): every v3 train task; the source is approved, which is
  not the same as a trajectory being accepted (see below);
* ``grpo_candidates`` (31): train tasks not suspended for GRPO;
* ``grpo_trainable`` (26): candidates whose R1 is not pending -- the only set a
  GRPO training entry may use;
* ``grpo_pending_r1`` (5): Bench2 train tasks whose gold writes fuzzy-compared
  free text; key-fact R1 is pending and must be completed before step 9;
* ``sft_only`` (6): five derived tasks suspended for GRPO (a core requirement
  scores DB=1 under a key-fact negative) and 78a5b71e_a4fb07a9, admitted as an
  SFT-only source.

Source approval and trajectory acceptance are recorded separately: approving a
source says a task may be used to make demonstrations; a trajectory enters SFT
only after its own review (S1 onwards), recorded in ``trajectory_acceptance``.
Dev and held-out are never training sources; held-out stays behind the final
evaluation permission in ``itsm_task_sets``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from agent_rl.data.itsm_split_v3 import DEFAULT_V3_PATH, SFT_ONLY_ADMITTED_TASK_IDS, SPLIT_VERSION_V3, load_manifest_v3

USAGE_VERSION = "itsm_usage_v1"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_USAGE_PATH = REPO_ROOT / "configs" / "data" / "itsm_usage_v1.json"
S0_AUDIT_PATH = REPO_ROOT / "docs" / "itsm_stepS0" / "usage_audit.json"
#: GRPO suspended, SFT kept (S0 / S0-R1, confirmed 2026-10-31).
GRPO_SUSPENDED_TASK_IDS = (
    "itsmbench_eops_01395ded_eff85dae",
    "itsmbench_eops_1d219d93_d1ba6c2d",
    "itsmbench_eops_46ca2862_b6ef60ca",
    "itsmbench_eops_66392a82_52c8290e",
    "itsmbench_eops_99ba2325_7915481a",
)
#: Key-fact R1 pending, to be completed before step 9; not GRPO-trainable until then.
GRPO_PENDING_R1_TASK_IDS = (
    "itsmbench_single_005",
    "itsmbench_single_006",
    "itsmbench_single_007",
    "itsmbench_single_011",
    "itsmbench_single_015",
)
PURPOSES = ("sft", "grpo")
DECISIONS = {
    "date": "2026-10-31",
    "user": (
        "S0-R1 验收通过。明确确认本次回复中的各项决定：五道派生题暂停 GRPO；78a5 保留为 SFT-only 来源；"
        "6D-3 验收通过，54f35603 纳入 held-out。……单独登记 SFT 37、GRPO 候选 31、SFT-only 6。"
        "原有 5 题明确标为 R1 pending，不得进入 GRPO 训练。……将“来源获准”和“轨迹经审核接纳”分别登记。"
    ),
}
TRAJECTORY_RECORD_FIELDS = (
    "trajectory_id",
    "task_id",
    "source_content_sha256",
    "teacher_provider",
    "teacher_model_id_returned",
    "sampling_parameters",
    "final_db_hash",
    "hard_checks",
    "review_decision",
    "reviewer",
    "review_date",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _s0_audit() -> dict[str, Any]:
    return json.loads(S0_AUDIT_PATH.read_text(encoding="utf-8"))


def _grpo_basis(task_id: str, s0_rows: dict[str, dict[str, Any]]) -> str:
    return s0_rows.get(task_id, {}).get("basis", "")


def build_usage_registry() -> dict[str, Any]:
    manifest = load_manifest_v3()
    s0 = _s0_audit()
    s0_rows = {r["id"]: r for r in s0["registry"]["tasks"]}
    suspended, pending, sft_only_admitted = set(GRPO_SUSPENDED_TASK_IDS), set(GRPO_PENDING_R1_TASK_IDS), set(SFT_ONLY_ADMITTED_TASK_IDS)
    train = set(manifest["splits"]["train"])
    if not (suspended | pending | sft_only_admitted) <= train:
        raise ValueError("every suspended, pending or SFT-only task must be in v3 train")
    if suspended & pending or suspended & sft_only_admitted or pending & sft_only_admitted:
        raise ValueError("suspended, pending and SFT-only admitted sets must be disjoint")

    tasks = {}
    for task_id, entry in sorted(manifest["tasks"].items()):
        split = entry["split"]
        if split != "train":
            tasks[task_id] = {
                "split": split,
                "use": split,
                "source_approval": {"sft": "not_a_training_source", "grpo": "not_a_training_source"},
                "grpo_status": "not_a_training_source",
                "content_sha256": entry["content_sha256"],
                "basis": f"v3 {split}" + ("; final frozen evaluation only" if split == "heldout" else "; debugging and length checks only"),
            }
            continue
        if task_id in sft_only_admitted:
            grpo, basis = "never_grpo", "admitted as an SFT-only source; its core requirement is invisible to the DB reward (D8)"
        elif task_id in suspended:
            grpo, basis = "suspended_reward_blind_spot", _grpo_basis(task_id, s0_rows)
        elif task_id in pending:
            grpo, basis = "pending_r1_before_step_9", _grpo_basis(task_id, s0_rows)
        else:
            grpo, basis = "trainable", _grpo_basis(task_id, s0_rows) or "v1/v2 train"
        tasks[task_id] = {
            "split": "train",
            "use": "both" if grpo in ("trainable", "pending_r1_before_step_9") else "sft_only",
            "source_approval": {"sft": "approved", "grpo": "approved" if grpo == "trainable" else grpo},
            "grpo_status": grpo,
            "content_sha256": entry["content_sha256"],
            "basis": basis,
        }

    def pool(predicate) -> list[str]:
        return sorted(t for t, r in tasks.items() if predicate(r))

    pools = {
        "sft_sources": pool(lambda r: r["source_approval"]["sft"] == "approved"),
        "grpo_candidates": pool(lambda r: r["grpo_status"] in ("trainable", "pending_r1_before_step_9")),
        "grpo_trainable": pool(lambda r: r["grpo_status"] == "trainable"),
        "grpo_pending_r1": pool(lambda r: r["grpo_status"] == "pending_r1_before_step_9"),
        "sft_only": pool(lambda r: r["use"] == "sft_only"),
        "dev": pool(lambda r: r["use"] == "dev"),
        "heldout": pool(lambda r: r["use"] == "heldout"),
    }
    return {
        "version": USAGE_VERSION,
        "split_version": SPLIT_VERSION_V3,
        "status": "candidate for freeze acceptance",
        "base": {
            "manifest": {"path": _rel(DEFAULT_V3_PATH), "sha256": _sha256_file(DEFAULT_V3_PATH)},
            "s0_audit": {"path": _rel(S0_AUDIT_PATH), "sha256": _sha256_file(S0_AUDIT_PATH), "version": s0["version"]},
        },
        "rules": {
            "selection": (
                "training entry points select tasks only through this registry: purpose 'grpo' -> grpo_trainable, "
                "purpose 'sft' -> sft_sources; v3 train is never used directly as a GRPO set"
            ),
            "grpo_pending_r1": "not GRPO-trainable until its key-fact R1 is completed and accepted, before step 9",
            "grpo_suspended": "a core requirement scores Bench2 DB=1 under a key-fact negative (S0/S0-R1)",
            "source_vs_trajectory": (
                "source_approval says a task may be used; a trajectory enters SFT only after its own review, "
                "recorded in trajectory_acceptance. Approving a source accepts no trajectory"
            ),
            "evaluation": "dev and held-out are never training sources; held-out needs final_evaluation=True",
        },
        "decisions": DECISIONS,
        "counts": {name: len(ids) for name, ids in pools.items()},
        "pools": pools,
        "tasks": tasks,
        "trajectory_acceptance": {
            "status": "no trajectory generated or reviewed",
            "accepted": [],
            "rejected": [],
            "record_fields": list(TRAJECTORY_RECORD_FIELDS),
        },
    }


def validate_usage_registry(registry: dict[str, Any]) -> None:
    """Raise unless ``registry`` is consistent with v3 and the S0 audit and equals a rebuild."""
    if registry.get("version") != USAGE_VERSION or registry.get("split_version") != SPLIT_VERSION_V3:
        raise ValueError("not an itsm_usage_v1 registry for itsm_split_v3")
    manifest = load_manifest_v3()
    pools = {name: set(ids) for name, ids in registry["pools"].items()}
    train = set(manifest["splits"]["train"])
    if pools["sft_sources"] != train:
        raise ValueError("SFT sources must be exactly v3 train")
    if pools["grpo_trainable"] & (set(GRPO_PENDING_R1_TASK_IDS) | set(GRPO_SUSPENDED_TASK_IDS) | set(SFT_ONLY_ADMITTED_TASK_IDS)):
        raise ValueError("a pending, suspended or SFT-only task is GRPO-trainable")
    if pools["grpo_candidates"] != pools["grpo_trainable"] | pools["grpo_pending_r1"]:
        raise ValueError("GRPO candidates must be trainable plus pending R1")
    if pools["sft_only"] != train - pools["grpo_candidates"]:
        raise ValueError("SFT-only must be the train tasks outside the GRPO candidates")
    for name in ("sft_sources", "grpo_candidates", "grpo_trainable", "sft_only"):
        if pools[name] & (set(manifest["splits"]["dev"]) | set(manifest["splits"]["heldout"])):
            raise ValueError(f"{name} contains a dev or held-out task")
    s0_pools = _s0_audit()["registry"]["pools"]
    if set(s0_pools["sft_only"]) != pools["sft_only"] or set(s0_pools["grpo_pending_r1_before_step9"]) != pools["grpo_pending_r1"]:
        raise ValueError("registry disagrees with the accepted S0 audit")
    if set(s0_pools["grpo"]) != pools["grpo_candidates"] or set(s0_pools["sft"]) != pools["sft_sources"]:
        raise ValueError("registry pools disagree with the accepted S0 audit")
    for task_id, row in registry["tasks"].items():
        if row["content_sha256"] != manifest["tasks"][task_id]["content_sha256"]:
            raise ValueError(f"{task_id}: content checksum differs from v3")
    if registry != build_usage_registry():
        raise ValueError("usage registry differs from a fresh rebuild")


def load_usage_registry(path: Path = DEFAULT_USAGE_PATH) -> dict[str, Any]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    validate_usage_registry(registry)
    return registry
