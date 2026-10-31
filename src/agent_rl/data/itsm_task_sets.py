"""Explicit, versioned ITSM task resolution (step 6C).

Every entry point takes ``split_version`` as a required keyword; nothing here
defaults to a version, so moving from v1 to v2 is always a visible change at the
call site. ``itsm_split_v1`` is the 53 Bench2 tasks of the v1 manifest;
``itsm_split_v2`` adds the three accepted EnterpriseOps-Gym-derived train tasks
from the pilot file, each checked against the v2 manifest's content checksum.
``itsm_split_v3`` adds the 6D-2 train tasks, the SFT-only source 78a5 and the
6D-3 held-out task, each checked against the v3 manifest's content checksum.

v3 ``train`` is the training-source union, not a training set: ``task_ids``
refuses to list it, and training entry points must use ``training_task_ids``,
which selects through the usage registry (``purpose="grpo"`` gives only the
GRPO-trainable tasks; SFT-only and R1-pending tasks are never included).

Resolved values are Bench2 ``Task`` objects only: provenance, upstream
verifiers, conversion notes and the manifests stay out of the env. ``ItsmEnv``
uses this module only when ``ItsmEnvConfig.split_version`` is set; without it the
env keeps its original Bench2-only lookup.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from enterprise_worlds.data_model.tasks import Task

from agent_rl.data.itsm_split import SPLIT_VERSION, SPLITS, load_manifest
from agent_rl.data.itsm_split_v2 import (
    SPLIT_VERSION_V2,
    derived_raw_tasks,
    load_manifest_v2,
    task_content_sha256,
)
from agent_rl.data.itsm_split_v3 import SPLIT_VERSION_V3, all_raw_tasks, load_manifest_v3
from agent_rl.data.itsm_usage import PURPOSES, USAGE_VERSION, load_usage_registry
from agent_rl.envs.itsm_scoring import load_itsm_tasks


SPLIT_VERSIONS = (SPLIT_VERSION, SPLIT_VERSION_V2, SPLIT_VERSION_V3)
USAGE_VERSIONS = (USAGE_VERSION,)


def _check_version(split_version: str) -> None:
    if split_version not in SPLIT_VERSIONS:
        raise ValueError(f"unknown split_version {split_version!r}; expected one of {list(SPLIT_VERSIONS)}")


@lru_cache(maxsize=None)
def _manifest(split_version: str) -> dict[str, Any]:
    if split_version == SPLIT_VERSION:
        return load_manifest()
    return load_manifest_v2() if split_version == SPLIT_VERSION_V2 else load_manifest_v3()


@lru_cache(maxsize=None)
def _tasks(split_version: str) -> dict[str, Task]:
    manifest = _manifest(split_version)
    bench = load_itsm_tasks()
    tasks: dict[str, Task] = {}
    for task_id, entry in manifest["tasks"].items():
        if entry.get("source", "bench2") == "bench2":
            tasks[task_id] = bench[task_id]
    if split_version == SPLIT_VERSION_V2:
        for task_id, raw in derived_raw_tasks().items():
            if task_content_sha256(raw) != manifest["tasks"][task_id]["content_sha256"]:
                raise ValueError(f"derived task {task_id} differs from the v2 manifest checksum")
            tasks[task_id] = Task.model_validate(raw)
    if split_version == SPLIT_VERSION_V3:
        raw_tasks, _ = all_raw_tasks()
        for task_id, entry in manifest["tasks"].items():
            if entry["source"] == "bench2":
                continue
            if task_content_sha256(raw_tasks[task_id]) != entry["content_sha256"]:
                raise ValueError(f"derived task {task_id} differs from the v3 manifest checksum")
            tasks[task_id] = Task.model_validate(raw_tasks[task_id])
    if set(tasks) != set(manifest["tasks"]):
        raise ValueError(f"{split_version}: resolved tasks differ from the manifest")
    return tasks


def load_split_manifest(*, split_version: str) -> dict[str, Any]:
    """The validated manifest of ``split_version`` (a copy is not made: do not mutate)."""
    _check_version(split_version)
    return _manifest(split_version)


def load_task_set(*, split_version: str) -> dict[str, Task]:
    """Every task of ``split_version`` (all splits), keyed by task ID."""
    _check_version(split_version)
    return dict(_tasks(split_version))


def resolve_task(task_id: str, *, split_version: str) -> Task:
    _check_version(split_version)
    tasks = _tasks(split_version)
    if task_id not in tasks:
        raise ValueError(f"task {task_id!r} is not in {split_version}")
    return tasks[task_id]


def task_ids(split: str, *, split_version: str, final_evaluation: bool = False) -> list[str]:
    """Task IDs of one split; held-out only for the final frozen evaluation.

    v3 train is refused: it mixes SFT-only and R1-pending sources with GRPO-trainable
    ones, so training sets come only from ``training_task_ids``."""
    _check_version(split_version)
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    if split == "train" and split_version == SPLIT_VERSION_V3:
        raise PermissionError(
            "itsm_split_v3 train is the training-source union, not a training set; "
            "use training_task_ids(purpose=..., usage_version=...)"
        )
    if split == "heldout" and not final_evaluation:
        raise PermissionError(
            "the held-out split is reserved for the final frozen evaluation; "
            "pass final_evaluation=True only there"
        )
    return list(_manifest(split_version)["splits"][split])


@lru_cache(maxsize=None)
def _usage(usage_version: str) -> dict[str, Any]:
    if usage_version not in USAGE_VERSIONS:
        raise ValueError(f"unknown usage_version {usage_version!r}; expected one of {list(USAGE_VERSIONS)}")
    return load_usage_registry()


def load_usage(*, usage_version: str) -> dict[str, Any]:
    """The validated usage registry (a copy is not made: do not mutate)."""
    return _usage(usage_version)


def training_task_ids(*, purpose: str, usage_version: str) -> list[str]:
    """The tasks a training entry may use: ``"grpo"`` -> GRPO-trainable only (no
    SFT-only, suspended or R1-pending task); ``"sft"`` -> approved SFT sources.
    Neither ever contains a dev or held-out task."""
    if purpose not in PURPOSES:
        raise ValueError(f"unknown purpose {purpose!r}; expected one of {list(PURPOSES)}")
    pools = _usage(usage_version)["pools"]
    return list(pools["grpo_trainable"] if purpose == "grpo" else pools["sft_sources"])

