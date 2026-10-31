"""Explicit, versioned ITSM task resolution (step 6C).

Every entry point takes ``split_version`` as a required keyword; nothing here
defaults to a version, so moving from v1 to v2 is always a visible change at the
call site. ``itsm_split_v1`` is the 53 Bench2 tasks of the v1 manifest;
``itsm_split_v2`` adds the three accepted EnterpriseOps-Gym-derived train tasks
from the pilot file, each checked against the v2 manifest's content checksum.

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
from agent_rl.envs.itsm_scoring import load_itsm_tasks


SPLIT_VERSIONS = (SPLIT_VERSION, SPLIT_VERSION_V2)


def _check_version(split_version: str) -> None:
    if split_version not in SPLIT_VERSIONS:
        raise ValueError(f"unknown split_version {split_version!r}; expected one of {list(SPLIT_VERSIONS)}")


@lru_cache(maxsize=None)
def _manifest(split_version: str) -> dict[str, Any]:
    return load_manifest() if split_version == SPLIT_VERSION else load_manifest_v2()


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
    """Task IDs of one split; held-out only for the final frozen evaluation."""
    _check_version(split_version)
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    if split == "heldout" and not final_evaluation:
        raise PermissionError(
            "the held-out split is reserved for the final frozen evaluation; "
            "pass final_evaluation=True only there"
        )
    return list(_manifest(split_version)["splits"][split])
