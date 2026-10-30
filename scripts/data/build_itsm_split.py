"""Build the frozen ITSM split manifest and the task-family audit (step 6).

Usage (from the repo root):
    python scripts/data/build_itsm_split.py [--check]

Writes configs/data/itsm_split_v1.json and docs/itsm_step6/family_audit.json.
``--check`` rebuilds both in memory and fails if the committed files differ.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

from agent_rl.data.itsm_split import (
    DEFAULT_MANIFEST_PATH,
    build_manifest,
    link_reason,
    load_raw_tasks,
    task_features,
    validate_manifest,
    write_jaccard,
    write_tool_names,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = REPO_ROOT / "docs" / "itsm_step6" / "family_audit.json"
#: Pairs the review asked about explicitly (MSP-021/022, 025/026, 027/028), plus
#: the cross-tenancy twins found by the audit.
WATCHED_PAIRS = (
    ("itsmbench_msp_021", "itsmbench_msp_022"),
    ("itsmbench_msp_025", "itsmbench_msp_026"),
    ("itsmbench_msp_027", "itsmbench_msp_028"),
    ("itsmbench_msp_021", "itsmbench_single_019"),
    ("itsmbench_msp_027", "itsmbench_single_017"),
    ("itsmbench_msp_001", "itsmbench_single_002"),
)
_RECORD = re.compile(r"[A-Z]+_\d{3,}|[A-Z]{3}-?\d{6,7}")


def _description_jaccard(a: str, b: str) -> float:
    ta, tb = (set(re.findall(r"[a-z]{4,}", s.lower())) for s in (a, b))
    return len(ta & tb) / len(ta | tb) if ta | tb else 0.0


_SEED_IDS: dict[str, set[str]] = {}


def _seed_ids(seed_db: str) -> set[str]:
    """Record IDs present in a seed file before any task delta."""
    if seed_db not in _SEED_IDS:
        from enterprise_worlds.domains.itsm import environment as itsm_environment

        seed = json.loads((itsm_environment.ITSM_TASKS_PATH.parent / seed_db).read_text(encoding="utf-8"))
        _SEED_IDS[seed_db] = {key for value in seed.values() if isinstance(value, dict) for key in value}
    return _SEED_IDS[seed_db]


def _delta_created(task: dict) -> set[str]:
    delta = task.get("initial_state_delta") or {}
    return {rid for ops in delta.values() for rid, op in ops.items() if "create" in op}


def _records(task: dict) -> set[tuple[str, str]]:
    found = set()
    for action in task["evaluation_criteria"]["actions"]:
        for value in action["arguments"].values():
            if isinstance(value, str) and _RECORD.fullmatch(value):
                found.add((task["seed_db"], value))
    return found


def _same_entities(a: dict, b: dict) -> list[str]:
    """Shared IDs that name the same seed record in both tasks (not a delta-created or
    task-created object that merely reuses the next sequential ID)."""
    if a["seed_db"] != b["seed_db"]:
        return []
    seed = _seed_ids(a["seed_db"])
    shared = {v for _, v in _records(a) & _records(b) if "_" in v}
    return sorted(
        rid for rid in shared if rid in seed and rid not in _delta_created(a) | _delta_created(b)
    )


def build_audit(manifest: dict) -> dict:
    raw = {task["id"]: task for task in load_raw_tasks()}
    features = task_features(list(raw.values()))
    split_of = {t: entry["split"] for t, entry in manifest["tasks"].items()}
    family_of = {t: entry["family"] for t, entry in manifest["tasks"].items()}

    links, signals = [], []
    for a, b in itertools.combinations(sorted(raw), 2):
        reason = link_reason(features[a], features[b])
        if reason is not None:
            links.append({"a": a, "b": b, "reason": reason, "same_family": family_of[a] == family_of[b]})
        shared = sorted(v for _, v in _records(raw[a]) & _records(raw[b]))
        desc = _description_jaccard(
            raw[a]["scenario"]["task_description"], raw[b]["scenario"]["task_description"]
        )
        if reason is None and (desc >= 0.35 or len(shared) >= 3):
            signals.append(
                {
                    "a": a,
                    "b": b,
                    "write_jaccard": round(write_jaccard(features[a], features[b]), 3),
                    "description_jaccard": round(desc, 3),
                    "shared_gold_ids": shared,
                    "same_seed_entities": _same_entities(raw[a], raw[b]),
                    "splits": [split_of[a], split_of[b]],
                    "involves_train_and_heldout": {split_of[a], split_of[b]} == {"train", "heldout"},
                }
            )

    watched = []
    for a, b in WATCHED_PAIRS:
        watched.append(
            {
                "a": a,
                "b": b,
                "tenancy": [features[a].tenancy, features[b].tenancy],
                "write_jaccard": round(write_jaccard(features[a], features[b]), 3),
                "description_jaccard": round(
                    _description_jaccard(
                        raw[a]["scenario"]["task_description"], raw[b]["scenario"]["task_description"]
                    ),
                    3,
                ),
                "family": [family_of[a], family_of[b]],
                "split": [split_of[a], split_of[b]],
            }
        )

    train = [t for t, s in split_of.items() if s == "train"]
    heldout = [t for t, s in split_of.items() if s == "heldout"]
    writes = write_tool_names()

    def write_sequence(task_id: str) -> list[str]:
        return [a["name"] for a in raw[task_id]["evaluation_criteria"]["actions"] if a["name"] in writes]

    shared_subworkflows = []
    for a, b in itertools.product(sorted(train), sorted(heldout)):
        shared = features[a].write_tools & features[b].write_tools
        if len(shared) < 2:
            continue
        shared_subworkflows.append(
            {
                "train_task": a,
                "heldout_task": b,
                "write_jaccard": round(write_jaccard(features[a], features[b]), 3),
                "shared_write_tools": sorted(shared),
                "only_in_train_task": sorted(features[a].write_tools - shared),
                "only_in_heldout_task": sorted(features[b].write_tools - shared),
                "train_write_sequence": write_sequence(a),
                "heldout_write_sequence": write_sequence(b),
                "same_seed_entities": _same_entities(raw[a], raw[b]),
            }
        )

    residual = {}
    for task_id, split in sorted(split_of.items()):
        if split == "train":
            continue
        nearest = max(train, key=lambda t: (write_jaccard(features[task_id], features[t]), t))
        residual[task_id] = {
            "split": split,
            "nearest_train_task": nearest,
            "write_jaccard": round(write_jaccard(features[task_id], features[nearest]), 3),
            "shared_write_tools": sorted(features[task_id].write_tools & features[nearest].write_tools),
        }

    return {
        "manifest_version": manifest["version"],
        "note": (
            "links: every pair joined by the family rule (all same_family by construction). "
            "unlinked_similarity_signals: pairs below the rule but with similar descriptions "
            "(token Jaccard >= 0.35) or >= 3 shared gold IDs. A shared ID is often not a shared "
            "object: deltas and creation tools reuse the next sequential ID for different "
            "objects (e.g. CHG_004 in single_007/015/022 are three different delta-created "
            "changes). same_seed_entities keeps only IDs that are the same pre-existing seed "
            "record in both tasks; these are reported for review, not used as a rule. "
            "residual_to_train: highest write-tool "
            "overlap of each dev/heldout task with any train task (sub-workflows such as "
            "update_incident remain shared across splits). cross_split_shared_subworkflows: "
            "every train/heldout pair sharing >= 2 write tools, with the shared sub-workflow "
            "and what each task combines it with; these pairs are deliberately kept across "
            "splits (workflow-composition generalisation), and a Jaccard below the family "
            "threshold is not treated as proof that they are not near-duplicates."
        ),
        "cross_split_shared_subworkflows": shared_subworkflows,
        "watched_pairs": watched,
        "links": links,
        "unlinked_similarity_signals": signals,
        "residual_to_train": residual,
    }


def _dump(data: dict) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, sort_keys=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if committed files differ")
    args = parser.parse_args()

    manifest = build_manifest()
    validate_manifest(manifest)
    outputs = {DEFAULT_MANIFEST_PATH: _dump(manifest), AUDIT_PATH: _dump(build_audit(manifest))}
    if args.check:
        stale = [str(p) for p, text in outputs.items() if not p.exists() or p.read_text(encoding="utf-8") != text]
        if stale:
            print("out of date:", *stale, sep="\n  ")
            return 1
        print("split manifest and family audit are up to date")
        return 0
    for path, text in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print("wrote", path.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
