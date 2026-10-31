"""ITSM split v2 (step 6C): split v1 plus three accepted EnterpriseOps-Gym-derived tasks.

Split v1 (``configs/data/itsm_split_v1.json``) stays frozen and unchanged. v2 keeps
every one of the 53 Bench2 tasks in its v1 split, adds the three pilot tasks the
user accepted on 2026-10-30 to train, and leaves dev and held-out identical to v1.
The family/split assignment is not re-optimised.

Families are recomputed over all 56 tasks with the v1 rule. A v1 family name is
kept only when the recomputed family has exactly the same members as in v1;
every other family (merged, extended or new) gets a v2 label, so no stale v1
name or count is carried over. The derived tasks keep their provenance and
conversion notes here; the policy never sees this file (``itsm_task_sets``
resolves Bench2 ``Task`` objects only).
"""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
from typing import Any

from agent_rl.data.itsm_eops_pilot import DEFAULT_PILOT_PATH, load_pilot
from agent_rl.data.itsm_split import (
    CREATION_TOOLS,
    DEFAULT_MANIFEST_PATH,
    SPLITS,
    WRITE_JACCARD_THRESHOLD,
    connected_components,
    data_version,
    load_manifest,
    load_raw_tasks,
    task_features,
    validate_split_structure,
)


SPLIT_VERSION_V2 = "itsm_split_v2"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_V2_PATH = REPO_ROOT / "configs" / "data" / "itsm_split_v2.json"
#: Accepted into train by the user on 2026-10-30 (step 6B acceptance); the other
#: five pilot candidates stay isolated.
ACCEPTED_DERIVED_TASK_IDS = (
    "itsmbench_eops_1d219d93_d1ba6c2d",
    "itsmbench_eops_46ca2862_b6ef60ca",
    "itsmbench_eops_66392a82_52c8290e",
)
ACCEPTANCE = {
    "date": "2026-10-30",
    "decision": (
        "第 6B 步验收通过。批准将以下三题纳入训练集 v2：itsmbench_eops_46ca2862_b6ef60ca、"
        "itsmbench_eops_66392a82_52c8290e、itsmbench_eops_1d219d93_d1ba6c2d。其余五题继续隔离"
    ),
}
EXPECTED_SIZES = {"train": 32, "dev": 13, "heldout": 11}
#: Fields that define a task for the env and for scoring (plus tenancy); the
#: checksum covers exactly these, so provenance edits do not change it.
TASK_CONTENT_KEYS = (
    "id",
    "scenario",
    "evaluation_criteria",
    "org_ids",
    "org_id",
    "seed_db",
    "initial_state_delta",
    "current_time",
    "tenancy",
)
#: Curated (family name, note), keyed by the first task ID, for v2 families that
#: are not a v1 family with unchanged members.
V2_FAMILY_LABELS = {
    "itsmbench_eops_1d219d93_d1ba6c2d": (
        "onboarding_catalog_and_change_problem_kb_merged",
        "v1 onboarding_and_catalog_admin and change_problem_advance_with_kb joined by 1d219d93 "
        "(creates CIs and a knowledge article); b6ef60ca joins via register_configuration_item",
    ),
    "itsmbench_eops_66392a82_52c8290e": (
        "incident_intake_with_ci_update",
        "update a CI, then create an incident and link its SLA; no Bench2 task uses this write-tool set",
    ),
}
#: Business type of each derived task, curated by its primary request (other
#: objects it writes are in ``objects_written``). Bench2 tasks keep their v1 type.
DERIVED_BUSINESS_TYPES = {
    "itsmbench_eops_1d219d93_d1ba6c2d": "cmdb",
    "itsmbench_eops_46ca2862_b6ef60ca": "incident_lifecycle",
    "itsmbench_eops_66392a82_52c8290e": "incident_lifecycle",
}
SOURCE_BENCH2 = "bench2"
SOURCE_DERIVED = "enterpriseops_gym_derived"
SCORING_DIFFERENCES = (
    "Not equivalent to the EnterpriseOps-Gym evaluation: the single upstream instruction is "
    "rewritten as a multi-turn conversation driven by the user simulator; the agent sees Bench2's "
    "task-invariant policy.md (not the upstream per-task system prompt), so gold adds the policy's "
    "REQUIRED cascades (SLA, knowledge search, default channel); scoring is Bench2's full-DB "
    "comparison with fuzzy free text, stricter than the upstream row-level SQL verifiers; "
    "supplemental facts (asset names, categories, declined notifications) were added for "
    "Bench2. See docs/itsm_step6b/README.md section 3."
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def task_content_sha256(raw_task: dict[str, Any]) -> str:
    """sha256 of the task's canonical JSON restricted to ``TASK_CONTENT_KEYS``."""
    content = {key: raw_task[key] for key in TASK_CONTENT_KEYS if key in raw_task}
    return _sha256_bytes(
        json.dumps(content, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )


def derived_raw_tasks(pilot_path: Path = DEFAULT_PILOT_PATH) -> dict[str, dict[str, Any]]:
    """The accepted pilot entries, as raw task dicts (with provenance keys)."""
    entries = {entry["id"]: entry for entry in load_pilot(pilot_path)["tasks"]}
    missing = sorted(set(ACCEPTED_DERIVED_TASK_IDS) - set(entries))
    if missing:
        raise ValueError(f"accepted derived tasks missing from the pilot file: {missing}")
    return {task_id: entries[task_id] for task_id in ACCEPTED_DERIVED_TASK_IDS}


def _data_block(pilot_path: Path) -> dict[str, Any]:
    pilot = load_pilot(pilot_path)
    return {
        "bench2": data_version(),
        "derived_source": {
            "pilot_file": _rel(pilot_path),
            "pilot_version": pilot["version"],
            "pilot_file_sha256": _sha256_bytes(pilot_path.read_bytes()),
        },
        "base_manifest": {
            "version": "itsm_split_v1",
            "path": _rel(DEFAULT_MANIFEST_PATH),
            "sha256": _sha256_bytes(DEFAULT_MANIFEST_PATH.read_bytes()),
        },
    }


def _derived_record(entry: dict[str, Any], pilot_version: str, pilot_path: Path) -> dict[str, Any]:
    source = entry["source"]
    return {
        "dataset": "EnterpriseOps-Gym",
        "repo": source["repo"],
        "commit": source["commit"],
        "hf_dataset": source["hf_dataset"],
        "hf_revision": source["hf_revision"],
        "license": source["license"],
        "attribution": source["attribution"],
        "mode_used": source["mode_used"],
        "upstream_task_id": source["upstream_task_id"],
        "upstream_user_prompt_sha256": source["upstream_user_prompt_sha256"],
        "upstream_seed": source["upstream_seed"],
        "upstream_seed_sha256": source["upstream_seed_sha256"],
        "pilot_version": pilot_version,
        "pilot_file": _rel(pilot_path),
        "multi_turn_rewrite": True,
        "org_slicing": "none (org_ids null): the whole Bench2 seed is loaded, as upstream does",
        "conversion_notes": list(entry["conversion_notes"]),
        "scoring_differences": SCORING_DIFFERENCES,
        "equivalent_to_upstream_evaluation": False,
        "accepted": ACCEPTANCE,
    }


def build_manifest_v2(pilot_path: Path = DEFAULT_PILOT_PATH) -> dict[str, Any]:
    base = load_manifest()
    bench = {task["id"]: task for task in load_raw_tasks()}
    derived = derived_raw_tasks(pilot_path)
    if set(derived) & set(bench):
        raise ValueError("derived task IDs collide with Bench2 task IDs")
    raw = {**bench, **derived}
    features = task_features(list(raw.values()))

    split_of = {t: base["tasks"][t]["split"] for t in bench}
    split_of.update({t: "train" for t in derived})
    business_type = {t: base["tasks"][t]["business_type"] for t in bench}
    business_type.update(DERIVED_BUSINESS_TYPES)

    v1_members = {name: set(entry["tasks"]) for name, entry in base["families"].items()}
    families: dict[str, dict[str, Any]] = {}
    used_anchors = set()
    for members in connected_components(features):
        v1_families = sorted({base["tasks"][t]["family"] for t in members if t in bench})
        derived_members = [t for t in members if t in derived]
        if not derived_members and len(v1_families) == 1 and v1_members[v1_families[0]] == set(members):
            name, origin, note = v1_families[0], "unchanged_from_v1", None
        else:
            anchor = members[0]
            if anchor not in V2_FAMILY_LABELS:
                raise ValueError(f"v2 family anchored at {anchor} has no curated label: {list(members)}")
            used_anchors.add(anchor)
            name, note = V2_FAMILY_LABELS[anchor]
            origin = "merged_v1_families" if len(v1_families) > 1 else (
                "extended_v1_family" if v1_families else "new_in_v2"
            )
            if name in v1_members:
                raise ValueError(f"v2 family {name} reuses a v1 family name with different members")
        homes = {split_of[t] for t in members}
        if len(homes) != 1:
            raise ValueError(f"v2 family {name} straddles splits {sorted(homes)}: {list(members)}")
        families[name] = {
            "split": homes.pop(),
            "origin": origin,
            "v1_families": v1_families,
            "derived_tasks": derived_members,
            "business_types": sorted({business_type[t] for t in members}),
            **({"note": note} if note else {}),
            "tasks": list(members),
        }
    if used_anchors != set(V2_FAMILY_LABELS):
        raise ValueError("V2_FAMILY_LABELS no longer matches the computed families")
    families = dict(sorted(families.items()))
    family_of = {t: name for name, entry in families.items() for t in entry["tasks"]}
    splits = {split: sorted(t for t in raw if split_of[t] == split) for split in SPLITS}

    pilot_version = load_pilot(pilot_path)["version"]
    tasks = {}
    for t in sorted(raw):
        tasks[t] = {
            "split": split_of[t],
            "family": family_of[t],
            "business_type": business_type[t],
            "source": SOURCE_DERIVED if t in derived else SOURCE_BENCH2,
            "tenancy": features[t].tenancy,
            "seed_db": features[t].seed_db,
            "org_slicing": raw[t].get("org_ids") is not None,
            "objects_written": list(features[t].objects),
            "gold_actions": features[t].gold_action_count,
            "content_sha256": task_content_sha256(raw[t]),
        }
        if t in derived:
            tasks[t]["derived_from"] = _derived_record(raw[t], pilot_version, pilot_path)

    def count(values) -> dict[str, int]:
        return dict(sorted(collections.Counter(values).items()))

    v1_names = set(base["families"])
    return {
        "version": SPLIT_VERSION_V2,
        "positioning": base["positioning"],
        "usage": base["usage"],
        "method": {
            "base": (
                "split v1 kept: the 53 Bench2 tasks keep their v1 split, dev and heldout are "
                "identical to v1; the three EnterpriseOps-Gym-derived tasks accepted on "
                "2026-10-30 are added to train; the assignment is not re-optimised"
            ),
            "family_rule": (
                f"as v1: link two tasks if the Jaccard similarity of their gold write-tool sets "
                f">= {WRITE_JACCARD_THRESHOLD} or they share a creation tool; families are the "
                "connected components over all 56 tasks; a v1 family name is kept only for a "
                "family whose members are exactly its v1 members, other families get a v2 label"
            ),
            "creation_tools": sorted(CREATION_TOOLS),
            "business_type_rule": (
                "per task: Bench2 tasks keep their v1 business type, derived tasks are curated "
                "by their primary request; a family lists its members' types"
            ),
            "task_content_keys": list(TASK_CONTENT_KEYS),
            "accepted": ACCEPTANCE,
        },
        "data": _data_block(pilot_path),
        "sizes": {split: len(splits[split]) for split in SPLITS},
        "family_changes": {
            "v1_family_count": len(v1_names),
            "v2_family_count": len(families),
            "unchanged_from_v1": sorted(n for n, e in families.items() if e["origin"] == "unchanged_from_v1"),
            "v1_families_no_longer_present": sorted(v1_names - set(families)),
            "changed": {
                n: {k: e[k] for k in ("origin", "v1_families", "derived_tasks", "split")}
                for n, e in families.items()
                if e["origin"] != "unchanged_from_v1"
            },
        },
        "families": families,
        "splits": splits,
        "tasks": tasks,
        "distribution": {
            split: {
                "tasks": len(ids),
                "source": count(tasks[t]["source"] for t in ids),
                "tenancy": count(tasks[t]["tenancy"] for t in ids),
                "business_type": count(tasks[t]["business_type"] for t in ids),
                "family": count(tasks[t]["family"] for t in ids),
                "families": len({tasks[t]["family"] for t in ids}),
                "objects_written": count(o for t in ids for o in tasks[t]["objects_written"]),
                "gold_actions_total": sum(tasks[t]["gold_actions"] for t in ids),
            }
            for split, ids in splits.items()
        },
    }


def validate_manifest_v2(manifest: dict[str, Any], pilot_path: Path = DEFAULT_PILOT_PATH) -> None:
    """Raise unless ``manifest`` is a consistent v2 manifest equal to a fresh rebuild.

    The targeted checks come first so a tampered manifest fails with a specific
    reason; the final rebuild comparison catches anything else (family members,
    labels, checksums, distribution).
    """
    if manifest.get("version") != SPLIT_VERSION_V2:
        raise ValueError(f"expected {SPLIT_VERSION_V2}, got {manifest.get('version')!r}")
    base = load_manifest()
    expected_ids = set(base["tasks"]) | set(ACCEPTED_DERIVED_TASK_IDS)
    validate_split_structure(manifest, expected_ids)
    sizes = {split: len(manifest["splits"][split]) for split in SPLITS}
    if sizes != EXPECTED_SIZES or manifest.get("sizes") != EXPECTED_SIZES:
        raise ValueError(f"v2 sizes must be {EXPECTED_SIZES}, got lists {sizes} / declared {manifest.get('sizes')}")
    for split in ("dev", "heldout"):
        if manifest["splits"][split] != base["splits"][split]:
            raise ValueError(f"v2 {split} differs from v1")
    for task_id, entry in base["tasks"].items():
        if manifest["tasks"][task_id]["split"] != entry["split"]:
            raise ValueError(f"task {task_id} moved from its v1 split {entry['split']}")
    derived = {t for t, e in manifest["tasks"].items() if e["source"] != SOURCE_BENCH2}
    if derived != set(ACCEPTED_DERIVED_TASK_IDS):
        raise ValueError(f"derived tasks must be exactly the accepted ones, got {sorted(derived)}")
    for task_id in ACCEPTED_DERIVED_TASK_IDS:
        entry = manifest["tasks"][task_id]
        if entry["source"] != SOURCE_DERIVED or entry["split"] != "train" or "derived_from" not in entry:
            raise ValueError(f"derived task {task_id} must be labelled {SOURCE_DERIVED} and in train")
    if manifest["data"] != _data_block(pilot_path):
        raise ValueError("Bench2 data, the pilot file or the v1 manifest changed since v2 was frozen")
    raw = {task["id"]: task for task in load_raw_tasks()} | derived_raw_tasks(pilot_path)
    for task_id, task in raw.items():
        if manifest["tasks"][task_id]["content_sha256"] != task_content_sha256(task):
            raise ValueError(f"task {task_id}: content checksum mismatch")
    if manifest != build_manifest_v2(pilot_path):
        raise ValueError("v2 manifest differs from a fresh rebuild")


def load_manifest_v2(path: Path = DEFAULT_V2_PATH) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest_v2(manifest)
    return manifest
