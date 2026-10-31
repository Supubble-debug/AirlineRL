"""EnterpriseOps-Gym ITSM migration, second batch (step 6D-2).

Same entry format as the step 6B pilot (``itsm_eops_pilot``): each entry is a
Bench2 ``Task`` plus provenance, the deduplicated upstream verifiers and
validation cases. Unlike the pilot, these tasks are org-scoped (``org_ids``) as
Bench2 tasks are. The batch is not part of Bench2's ``tasks.json`` or of any
split; inclusion in a split needs the user's acceptance.

``family_overlap_check_v2`` re-runs step 6's family rule over split v2's 56
tasks plus the batch, with the 6A train gate (<= 1 write tool shared with any
dev/held-out task, < 2 same-seed entities with any dev/held-out task).
"""

from __future__ import annotations

import collections
import itertools
import json
from pathlib import Path
from typing import Any

from agent_rl.data.itsm_eops_pilot import _delta_created, _gold_ids, _seed_ids, _write_set
from agent_rl.data.itsm_split import CREATION_TOOLS, WRITE_JACCARD_THRESHOLD, load_raw_tasks, write_tool_names
from agent_rl.data.itsm_split_v2 import DEFAULT_V2_PATH, derived_raw_tasks

BATCH2_VERSION = "itsm_eops_batch2_v1"
DEFAULT_BATCH2_PATH = Path(__file__).resolve().parents[3] / "configs" / "data" / "itsm_eops_batch2_v1.json"
MAX_SHARED_WRITES_WITH_EVAL_SPLITS = 1


def family_overlap_check_v2(batch_entries: list[dict[str, Any]]) -> dict[str, Any]:
    writes = write_tool_names()
    manifest = json.loads(DEFAULT_V2_PATH.read_text(encoding="utf-8"))
    reference = {t["id"]: t for t in load_raw_tasks()}
    reference.update(derived_raw_tasks())
    batch = {t["id"]: t for t in batch_entries}
    if set(batch) & set(reference):
        raise ValueError("batch task IDs collide with split v2 task IDs")
    combined = {**reference, **batch}
    sets = {tid: _write_set(t, writes) for tid, t in combined.items()}
    split_of = {tid: manifest["tasks"][tid]["split"] for tid in reference}

    parent = {tid: tid for tid in combined}

    def root(tid: str) -> str:
        while parent[tid] != tid:
            parent[tid] = parent[parent[tid]]
            tid = parent[tid]
        return tid

    links = []
    for a, b in itertools.combinations(sorted(combined), 2):
        union = sets[a] | sets[b]
        jaccard = len(sets[a] & sets[b]) / len(union) if union else 0.0
        shared_creation = sorted(sets[a] & sets[b] & CREATION_TOOLS)
        if jaccard >= WRITE_JACCARD_THRESHOLD or shared_creation:
            parent[root(a)] = root(b)
            if a in batch or b in batch:
                links.append({"a": a, "b": b, "write_jaccard": round(jaccard, 3), "shared_creation_tools": shared_creation})
    components = collections.defaultdict(list)
    for tid in sorted(combined):
        components[root(tid)].append(tid)

    seed_ids = {seed: _seed_ids(seed) for seed in {t["seed_db"] for t in combined.values()}}
    per_task = {}
    for bid, entry in sorted(batch.items()):
        component = next(c for c in components.values() if bid in c)
        members = [t for t in component if t in reference]
        eval_writes, eval_entities = (0, None), []
        for rid, other in reference.items():
            if split_of[rid] == "train":
                continue
            eval_writes = max(eval_writes, (len(sets[bid] & sets[rid]), rid), key=lambda x: x[0])
            if other["seed_db"] == entry["seed_db"]:
                shared = (_gold_ids(entry) & _gold_ids(other) & seed_ids[entry["seed_db"]]) - _delta_created(entry) - _delta_created(other)
                if len(shared) >= 2:
                    eval_entities.append({"task": rid, "split": split_of[rid], "same_seed_entities": sorted(shared)})
        per_task[bid] = {
            "write_tools": sorted(sets[bid]),
            "component_v2_members": members,
            "component_v2_splits": sorted({split_of[t] for t in members}),
            "component_v2_families": sorted({manifest["tasks"][t]["family"] for t in members}),
            "component_batch_members": [t for t in component if t in batch],
            "max_shared_write_tools_with_dev_or_heldout": {"count": eval_writes[0], "task": eval_writes[1]},
            "dev_or_heldout_entity_overlap_ge2": eval_entities,
        }

    bridged = [
        sorted({manifest["tasks"][t]["family"] for t in c if t in reference})
        for c in components.values()
        if len({split_of[t] for t in c if t in reference}) > 1
    ]
    eval_families_unchanged = all(
        sorted(t for t in c if t in reference) == sorted(manifest["families"][manifest["tasks"][c[0]]["family"]]["tasks"])
        and not any(t in batch for t in c)
        for c in components.values()
        if any(t in reference and split_of[t] != "train" for t in c)
    )
    checks = {
        "batch_components_touch_train_only": all(set(r["component_v2_splits"]) <= {"train"} for r in per_task.values()),
        "no_component_bridges_v2_splits": not bridged,
        "batch_shares_le_1_write_tool_with_dev_or_heldout": all(
            r["max_shared_write_tools_with_dev_or_heldout"]["count"] <= MAX_SHARED_WRITES_WITH_EVAL_SPLITS for r in per_task.values()
        ),
        "no_batch_task_shares_2plus_entities_with_dev_or_heldout": not any(r["dev_or_heldout_entity_overlap_ge2"] for r in per_task.values()),
        "dev_and_heldout_families_unchanged": eval_families_unchanged,
    }
    return {"checks": checks, "batch_links": links, "bridged_v2_families": bridged, "tasks": per_task}
