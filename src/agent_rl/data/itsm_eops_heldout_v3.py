"""EnterpriseOps-Gym ITSM held-out candidates for split v3 (step 6D-3).

Same entry format as the 6B pilot and the 6D-2 batch. The candidates file is not
part of Bench2's ``tasks.json`` or of any split; no v3 manifest exists yet.

``heldout_gate_check`` applies the admission rules fixed in
docs/itsm_step6d3/README.md §1 against the reference set: split v2's 56 tasks
plus the four 6D-2 tasks the user accepted as v3 train additions (train 36,
dev 13, held-out 11). Everything is computed from the candidates' gold actions.
"""

from __future__ import annotations

import collections
import itertools
import json
from pathlib import Path
from typing import Any

from agent_rl.data.itsm_eops_batch2 import DEFAULT_BATCH2_PATH
from agent_rl.data.itsm_eops_pilot import _delta_created, _gold_ids, _seed_ids, _write_set, load_pilot
from agent_rl.data.itsm_split import CREATION_TOOLS, WRITE_JACCARD_THRESHOLD, load_raw_tasks, write_tool_names
from agent_rl.data.itsm_split_v2 import DEFAULT_V2_PATH, derived_raw_tasks

HELDOUT_V3_VERSION = "itsm_eops_heldout_v3_candidates_v1"
DEFAULT_HELDOUT_V3_PATH = Path(__file__).resolve().parents[3] / "configs" / "data" / "itsm_eops_heldout_v3_candidates.json"
#: Accepted by the user on 2026-10-30 (6D-2 acceptance) as v3 train additions.
ACCEPTED_V3_TRAIN_TASK_IDS = (
    "itsmbench_eops_99ba2325_887f5580",
    "itsmbench_eops_01395ded_eff85dae",
    "itsmbench_eops_46ca2862_56a4f834",
    "itsmbench_eops_99ba2325_7915481a",
)
MAX_SHARED_WRITES = 1
MIN_SHARED_ENTITIES_BLOCKING = 2


def reference_set() -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, str]]:
    """Raw tasks, split and family of v2 plus the accepted v3 train additions."""
    manifest = json.loads(DEFAULT_V2_PATH.read_text(encoding="utf-8"))
    raw = {t["id"]: t for t in load_raw_tasks()}
    raw.update(derived_raw_tasks())
    split = {tid: manifest["tasks"][tid]["split"] for tid in raw}
    family = {tid: manifest["tasks"][tid]["family"] for tid in raw}
    batch = {e["id"]: e for e in load_pilot(DEFAULT_BATCH2_PATH)["tasks"]}
    if set(ACCEPTED_V3_TRAIN_TASK_IDS) - set(batch):
        raise ValueError("accepted v3 train tasks missing from the 6D-2 batch file")
    for tid in ACCEPTED_V3_TRAIN_TASK_IDS:
        raw[tid] = batch[tid]
        split[tid] = "train"
        family[tid] = "v3_train_addition"
    return raw, split, family


def heldout_gate_check(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    writes = write_tool_names()
    raw, split, family = reference_set()
    cand = {c["id"]: c for c in candidates}
    if set(cand) & set(raw):
        raise ValueError("candidate IDs collide with reference task IDs")
    combined = {**raw, **cand}
    sets = {tid: _write_set(t, writes) for tid, t in combined.items()}
    seed_ids = {seed: _seed_ids(seed) for seed in {t["seed_db"] for t in combined.values()}}

    def entities(tid: str) -> set[str]:
        task = combined[tid]
        return (_gold_ids(task) & seed_ids[task["seed_db"]]) - _delta_created(task)

    ents = {tid: entities(tid) for tid in combined}

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
            if a in cand or b in cand:
                links.append({"a": a, "b": b, "write_jaccard": round(jaccard, 3), "shared_creation_tools": shared_creation})
    components = collections.defaultdict(list)
    for tid in sorted(combined):
        components[root(tid)].append(tid)

    per_task = {}
    for cid in sorted(cand):
        component = next(c for c in components.values() if cid in c)
        members = [t for t in component if t in raw]
        shared_writes = {"train": (0, None), "dev": (0, None)}
        entity_overlap, near_duplicates = [], []
        for rid in sorted(raw):
            s = split[rid]
            if s in shared_writes:
                shared_writes[s] = max(shared_writes[s], (len(sets[cid] & sets[rid]), rid), key=lambda x: x[0])
            same_seed = raw[rid]["seed_db"] == cand[cid]["seed_db"]
            shared = sorted(ents[cid] & ents[rid]) if same_seed else []
            if len(shared) >= MIN_SHARED_ENTITIES_BLOCKING:
                entity_overlap.append({"task": rid, "split": s, "same_seed_entities": shared})
                if sets[cid] == sets[rid]:
                    near_duplicates.append({"task": rid, "split": s})
        for other in sorted(cand):
            if other == cid or cand[other]["seed_db"] != cand[cid]["seed_db"]:
                continue
            shared = ents[cid] & ents[other]
            if len(shared) >= MIN_SHARED_ENTITIES_BLOCKING and sets[cid] == sets[other]:
                near_duplicates.append({"task": other, "split": "candidate"})
        component_splits = sorted({split[t] for t in members})
        checks = {
            "component_has_no_train_or_dev_member": set(component_splits) <= {"heldout"},
            "shares_le_1_write_tool_with_train_and_dev": all(v[0] <= MAX_SHARED_WRITES for v in shared_writes.values()),
            "shares_lt_2_entities_with_train_and_dev": not [o for o in entity_overlap if o["split"] in ("train", "dev")],
            "not_a_near_duplicate": not near_duplicates,
        }
        per_task[cid] = {
            "write_tools": sorted(sets[cid]),
            "gold_entities": sorted(ents[cid]),
            "component_reference_members": members,
            "component_splits": component_splits,
            "component_heldout_families": sorted({family[t] for t in members}),
            "component_candidates": [t for t in component if t in cand],
            "merges_heldout_families": len({family[t] for t in members}) > 1,
            "max_shared_write_tools": {s: {"count": c, "task": t} for s, (c, t) in shared_writes.items()},
            "entity_overlap_ge2": entity_overlap,
            "near_duplicates": near_duplicates,
            "checks": checks,
            "gate_ok": all(checks.values()),
        }
    return {
        "reference": {
            "split_v2": str(DEFAULT_V2_PATH.name),
            "accepted_v3_train_additions": list(ACCEPTED_V3_TRAIN_TASK_IDS),
            "sizes": dict(collections.Counter(split.values())),
        },
        "candidate_links": links,
        "tasks": per_task,
    }
