"""ITSM task families and the frozen train/dev/held-out split (step 6).

Families are computed, not hand-picked: two tasks are linked when the sets of
write tools their gold actions use have Jaccard similarity >= 0.6, or when both
create the same kind of object (``CREATION_TOOLS``); families are the connected
components. Only the family names and business types are curated, keyed by
each family's first task ID. The split assigns whole families, so similar
workflows never straddle splits regardless of task ID or tenancy.

The held-out split is for the final frozen evaluation only: no debugging,
tuning or checkpoint selection. ``split_task_ids`` refuses it unless the caller
states that purpose.
"""

from __future__ import annotations

import collections
import hashlib
import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from enterprise_worlds.domains.itsm import environment as itsm_environment
from enterprise_worlds.domains.itsm.tools import ItsmTools
from enterprise_worlds.environment.toolkit import ToolType


SPLIT_VERSION = "itsm_split_v1"
SPLIT_SEED = 20261030
SPLITS = ("train", "dev", "heldout")
WRITE_JACCARD_THRESHOLD = 0.6
#: Write tools that create a new object; sharing one links two tasks.
CREATION_TOOLS = frozenset(
    {
        "add_child_incident",
        "add_location",
        "add_new_service",
        "add_new_sla_definition",
        "add_new_user",
        "add_new_user_group",
        "create_knowledge_article",
        "create_problem",
        "register_configuration_item",
        "register_new_service_offering",
    }
)
#: Three families hold 40 of 53 tasks, so a family-level split must trade a
#: diverse held-out against keeping every large workflow in train. v1 favours
#: the held-out: >= 5 families, >= 3 business types, no family above half of it.
TARGET_SIZES = {"train": 29, "dev": 12, "heldout": 12}
SIZE_RANGES = {"train": (26, 34), "dev": (9, 14), "heldout": (10, 14)}
MIN_TASKS_PER_TENANCY = 2  # in dev and held-out
MIN_FAMILIES = {"dev": 2, "heldout": 5}
MIN_HELDOUT_BUSINESS_TYPES = 3
MAX_HELDOUT_FAMILY_SHARE = 0.5
#: Curated (family name, business type), keyed by the family's first task ID.
FAMILY_LABELS = {
    "itsmbench_msp_001": ("onboarding_and_catalog_admin", "user_and_catalog_admin"),
    "itsmbench_msp_005": ("change_problem_advance_with_kb", "problem_change_lifecycle"),
    "itsmbench_msp_019": ("incident_hold_and_notify", "incident_lifecycle"),
    "itsmbench_msp_020": ("service_retirement", "service_catalog"),
    "itsmbench_msp_023": ("incident_resolve", "incident_lifecycle"),
    "itsmbench_msp_027": ("problem_from_incident_and_change_mapping", "problem_change_lifecycle"),
    "itsmbench_msp_030": ("change_with_ci_and_kb_links", "problem_change_lifecycle"),
    "itsmbench_single_009": ("incident_reassign_with_ci", "incident_lifecycle"),
    "itsmbench_single_012": ("incident_handoff_with_child", "incident_lifecycle"),
    "itsmbench_single_016": ("knowledge_update_and_notify", "knowledge"),
    "itsmbench_single_018": ("ci_maintenance", "cmdb"),
}
#: Substring -> business object touched by a write tool (first match wins).
_OBJECT_RULES = (
    ("sla", "sla"),
    ("knowledge", "knowledge"),
    ("configuration_item", "cmdb"),
    ("affected_ci", "cmdb"),
    ("notification", "notification"),
    ("incident_template", "template"),
    ("incident", "incident"),
    ("problem", "problem"),
    ("change", "change"),
    ("service", "service"),
    ("location", "location"),
    ("user", "user"),
    ("group", "user"),
)
DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parents[3] / "configs" / "data" / "itsm_split_v1.json"


@dataclass(frozen=True, slots=True)
class TaskFeatures:
    task_id: str
    tenancy: str
    seed_db: str
    write_tools: frozenset[str]
    objects: tuple[str, ...]
    gold_action_count: int


def write_tool_names() -> frozenset[str]:
    return frozenset(
        name
        for name in ItsmTools._tool_names
        if getattr(ItsmTools, name).__tool_type__ == ToolType.WRITE
    )


def object_of(tool_name: str) -> str:
    for needle, business_object in _OBJECT_RULES:
        if needle in tool_name:
            return business_object
    raise ValueError(f"no business object for tool {tool_name!r}")


def load_raw_tasks() -> list[dict[str, Any]]:
    # ``Task`` drops ``tenancy``, so the raw file is the source here.
    return json.loads(itsm_environment.ITSM_TASKS_PATH.read_text(encoding="utf-8"))


def task_features(raw_tasks: list[dict[str, Any]]) -> dict[str, TaskFeatures]:
    writes = write_tool_names()
    features = {}
    for task in raw_tasks:
        names = [a["name"] for a in task["evaluation_criteria"]["actions"]]
        task_writes = frozenset(name for name in names if name in writes)
        features[task["id"]] = TaskFeatures(
            task_id=task["id"],
            tenancy=task["tenancy"]["model"],
            seed_db=task["seed_db"],
            write_tools=task_writes,
            objects=tuple(sorted({object_of(name) for name in task_writes})),
            gold_action_count=len(names),
        )
    return features


def write_jaccard(a: TaskFeatures, b: TaskFeatures) -> float:
    union = a.write_tools | b.write_tools
    return len(a.write_tools & b.write_tools) / len(union) if union else 0.0


def link_reason(a: TaskFeatures, b: TaskFeatures) -> str | None:
    """Why two tasks must share a family, or ``None``."""
    jaccard = write_jaccard(a, b)
    if jaccard >= WRITE_JACCARD_THRESHOLD:
        return f"write_jaccard={jaccard:.2f}"
    shared = sorted(a.write_tools & b.write_tools & CREATION_TOOLS)
    if shared:
        return f"shared_creation={','.join(shared)}"
    return None


def build_families(features: dict[str, TaskFeatures]) -> dict[str, tuple[str, ...]]:
    """Connected components of the link graph, named via ``FAMILY_LABELS``."""
    parent = {task_id: task_id for task_id in features}

    def root(task_id: str) -> str:
        while parent[task_id] != task_id:
            parent[task_id] = parent[parent[task_id]]
            task_id = parent[task_id]
        return task_id

    for a, b in itertools.combinations(sorted(features), 2):
        if link_reason(features[a], features[b]) is not None:
            parent[root(a)] = root(b)
    components: dict[str, list[str]] = collections.defaultdict(list)
    for task_id in sorted(features):
        components[root(task_id)].append(task_id)
    families = {}
    for members in components.values():
        anchor = members[0]
        if anchor not in FAMILY_LABELS:
            raise ValueError(f"family anchored at {anchor} has no curated label: {members}")
        families[FAMILY_LABELS[anchor][0]] = tuple(members)
    if set(FAMILY_LABELS) != {members[0] for members in families.values()}:
        raise ValueError("FAMILY_LABELS no longer matches the computed families")
    return dict(sorted(families.items()))


def business_type(family: str) -> str:
    return {name: kind for name, kind in FAMILY_LABELS.values()}[family]


def _assignment_score(assignment, families, features, msp_share) -> float | None:
    sizes = {split: 0 for split in SPLITS}
    msp = {split: 0 for split in SPLITS}
    members = {split: [] for split in SPLITS}
    for family, split in assignment.items():
        sizes[split] += len(families[family])
        msp[split] += sum(features[t].tenancy == "provider_client" for t in families[family])
        members[split].append(family)
    for split, (low, high) in SIZE_RANGES.items():
        if not low <= sizes[split] <= high:
            return None
    for split in ("dev", "heldout"):
        single = sizes[split] - msp[split]
        if min(msp[split], single) < MIN_TASKS_PER_TENANCY:
            return None
        if len(members[split]) < MIN_FAMILIES[split]:
            return None
    heldout = members["heldout"]
    if len({business_type(family) for family in heldout}) < MIN_HELDOUT_BUSINESS_TYPES:
        return None
    if max(len(families[f]) for f in heldout) > MAX_HELDOUT_FAMILY_SHARE * sizes["heldout"]:
        return None
    size_penalty = sum(abs(sizes[s] - TARGET_SIZES[s]) for s in SPLITS)
    tenancy_penalty = sum(abs(msp[s] / sizes[s] - msp_share) for s in ("dev", "heldout"))
    return round(size_penalty + 10 * tenancy_penalty, 9)


def assign_splits(
    families: dict[str, tuple[str, ...]], features: dict[str, TaskFeatures], seed: int = SPLIT_SEED
) -> tuple[dict[str, str], dict[str, Any]]:
    """Enumerate every family -> split assignment, keep those meeting the
    constraints, take the lowest score, and break ties with ``seed``."""
    names = sorted(families)
    msp_share = sum(f.tenancy == "provider_client" for f in features.values()) / len(features)
    scored = []
    for choice in itertools.product(SPLITS, repeat=len(names)):
        assignment = dict(zip(names, choice))
        score = _assignment_score(assignment, families, features, msp_share)
        if score is not None:
            scored.append((score, choice))
    if not scored:
        raise ValueError("no family assignment satisfies the split constraints")
    best = min(score for score, _ in scored)
    tied = sorted(choice for score, choice in scored if score == best)
    chosen = random.Random(seed).choice(tied)
    stats = {"feasible_assignments": len(scored), "best_score": best, "tied_at_best": len(tied)}
    return dict(zip(names, chosen)), stats


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data_version() -> dict[str, Any]:
    root = itsm_environment.ITSM_TASKS_PATH.parent
    return {
        "bench2_source": "Enterprise-Worlds 37175af590d3b9c7d2288501b604020e547e3783",
        "files_sha256": {
            path.name: _sha256(path) for path in sorted(root.iterdir()) if path.is_file()
        },
    }


def _distribution(task_ids: list[str], features, family_of) -> dict[str, Any]:
    def count(values) -> dict[str, int]:
        return dict(sorted(collections.Counter(values).items()))

    return {
        "tasks": len(task_ids),
        "tenancy": count(features[t].tenancy for t in task_ids),
        "business_type": count(business_type(family_of[t]) for t in task_ids),
        "family": count(family_of[t] for t in task_ids),
        "objects_written": count(o for t in task_ids for o in features[t].objects),
        "gold_actions_total": sum(features[t].gold_action_count for t in task_ids),
    }


def build_manifest(seed: int = SPLIT_SEED) -> dict[str, Any]:
    raw_tasks = load_raw_tasks()
    features = task_features(raw_tasks)
    families = build_families(features)
    family_split, stats = assign_splits(families, features, seed)
    family_of = {t: family for family, members in families.items() for t in members}
    split_tasks = {
        split: sorted(t for t, family in family_of.items() if family_split[family] == split)
        for split in SPLITS
    }
    return {
        "version": SPLIT_VERSION,
        "usage": (
            "train: optimisation; dev: debugging, budget checks and checkpoint selection; "
            "heldout: final frozen evaluation only, never for debugging, tuning or "
            "checkpoint selection"
        ),
        "method": {
            "family_rule": (
                f"link two tasks if the Jaccard similarity of their gold write-tool sets "
                f">= {WRITE_JACCARD_THRESHOLD} or they share a creation tool; families are "
                "connected components; names/business types curated by first task ID"
            ),
            "creation_tools": sorted(CREATION_TOOLS),
            "split_rule": (
                "enumerate all family->split assignments; keep sizes within size_ranges, "
                f">= {MIN_TASKS_PER_TENANCY} tasks of each tenancy in dev and heldout, "
                f">= {MIN_FAMILIES['dev']} families in dev, >= {MIN_FAMILIES['heldout']} "
                f"families and >= {MIN_HELDOUT_BUSINESS_TYPES} business types in heldout, "
                f"no family above {MAX_HELDOUT_FAMILY_SHARE:.0%} of heldout; "
                "score = sum |size - target| + 10 * sum over "
                "dev/heldout |msp share - overall msp share|; lowest score wins, ties broken "
                "by random.Random(seed).choice over the sorted tied assignments"
            ),
            "seed": seed,
            "target_sizes": TARGET_SIZES,
            "size_ranges": {k: list(v) for k, v in SIZE_RANGES.items()},
            **stats,
        },
        "data": data_version(),
        "families": {
            family: {
                "business_type": business_type(family),
                "split": family_split[family],
                "tasks": list(members),
            }
            for family, members in families.items()
        },
        "splits": {split: split_tasks[split] for split in SPLITS},
        "tasks": {
            t: {
                "split": family_split[family_of[t]],
                "family": family_of[t],
                "business_type": business_type(family_of[t]),
                "tenancy": features[t].tenancy,
                "seed_db": features[t].seed_db,
                "objects_written": list(features[t].objects),
                "gold_actions": features[t].gold_action_count,
            }
            for t in sorted(features)
        },
        "distribution": {
            split: _distribution(split_tasks[split], features, family_of) for split in SPLITS
        },
    }


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Raise if the splits overlap, miss a task, split a family, or the data changed."""
    all_ids = {task["id"] for task in load_raw_tasks()}
    split_sets = {split: set(manifest["splits"][split]) for split in SPLITS}
    for a, b in itertools.combinations(SPLITS, 2):
        if split_sets[a] & split_sets[b]:
            raise ValueError(f"splits {a} and {b} overlap")
    if set().union(*split_sets.values()) != all_ids:
        raise ValueError("splits do not cover exactly the task suite")
    for family, entry in manifest["families"].items():
        homes = {manifest["tasks"][t]["split"] for t in entry["tasks"]}
        if homes != {entry["split"]}:
            raise ValueError(f"family {family} straddles splits {sorted(homes)}")
    if manifest["data"] != data_version():
        raise ValueError("task data changed since the manifest was frozen")


def load_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    return manifest


def split_task_ids(
    split: str, *, final_evaluation: bool = False, path: Path = DEFAULT_MANIFEST_PATH
) -> list[str]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    if split == "heldout" and not final_evaluation:
        raise PermissionError(
            "the held-out split is reserved for the final frozen evaluation; "
            "pass final_evaluation=True only there"
        )
    return list(load_manifest(path)["splits"][split])
