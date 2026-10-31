"""ITSM split v3: split v2 plus the accepted 6D-2 and 6D-3 tasks.

v1 (``itsm_split_v1.json``) and v2 (``itsm_split_v2.json``) stay frozen and
unchanged. v3 keeps every one of v2's 56 tasks in its v2 split and only adds:

* train: the four 6D-2 tasks accepted as v3 train additions (2026-10-30) and
  ``78a5b71e_a4fb07a9``, admitted as an SFT-only source (2026-10-31);
* held-out: ``99ba2325_54f35603``, accepted in 6D-3 (2026-10-31).

**v3 ``train`` is the training-source union, not the GRPO set.** Which train
task may be used by SFT or GRPO is recorded only in the usage registry
(``itsm_usage``); ``itsm_task_sets`` refuses to list v3 train without a purpose.

Families are recomputed over all 62 tasks with the step-6 rule. A v2 family
name is kept only when its members are unchanged; any other family gets a v3
label. Each task records a content checksum (the fields that define it for the
env and scoring) and a source checksum (its full entry, provenance and
validation cases included, as stored in its source file).
"""

from __future__ import annotations

import collections
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

from agent_rl.data.itsm_eops_batch2 import DEFAULT_BATCH2_PATH
from agent_rl.data.itsm_eops_heldout_v3 import ACCEPTED_V3_TRAIN_TASK_IDS, DEFAULT_HELDOUT_V3_PATH
from agent_rl.data.itsm_eops_pilot import DEFAULT_PILOT_PATH, _delta_created, _gold_ids, _seed_ids, load_pilot
from agent_rl.data.itsm_split import (
    CREATION_TOOLS,
    DEFAULT_MANIFEST_PATH,
    SPLITS,
    WRITE_JACCARD_THRESHOLD,
    connected_components,
    data_version,
    load_raw_tasks,
    task_features,
    validate_split_structure,
)
from agent_rl.data.itsm_split_v2 import (
    DEFAULT_V2_PATH,
    SOURCE_BENCH2,
    SOURCE_DERIVED,
    derived_raw_tasks,
    load_manifest_v2,
    task_content_sha256,
)

SPLIT_VERSION_V3 = "itsm_split_v3"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_V3_PATH = REPO_ROOT / "configs" / "data" / "itsm_split_v3.json"
#: Admitted by the user as an SFT-only source (S0-R1, confirmed 2026-10-31).
SFT_ONLY_ADMITTED_TASK_IDS = ("itsmbench_eops_78a5b71e_a4fb07a9",)
#: 6D-3 accepted held-out addition (2026-10-31).
HELDOUT_ADDED_TASK_IDS = ("itsmbench_eops_99ba2325_54f35603",)
V3_TRAIN_ADDITIONS = tuple(sorted((*ACCEPTED_V3_TRAIN_TASK_IDS, *SFT_ONLY_ADMITTED_TASK_IDS)))
EXPECTED_SIZES = {"train": 37, "dev": 13, "heldout": 12}
ACCEPTANCE = {
    "date": "2026-10-31",
    "decision": (
        "S0-R1 验收通过。明确确认本次回复中的各项决定：五道派生题暂停 GRPO；78a5 保留为 SFT-only 来源；"
        "6D-3 验收通过，54f35603 纳入 held-out。现在只执行：生成 v3 manifest 与用途登记，提交冻结验收。"
        "1. v1/v2 不改。v3 训练来源并集为 37，dev 13、held-out 12。"
    ),
    "earlier": [
        {"date": "2026-10-30", "decision": "第 6D-2 验收通过。……批准第 1、3、6、7 题作为 v3 新增训练题"},
    ],
}
#: Curated (family name, note), keyed by the first task ID, for v3 families whose
#: members differ from every v2 family.
V3_FAMILY_LABELS: dict[str, tuple[str, str]] = {
    "itsmbench_eops_1d219d93_d1ba6c2d": (
        "onboarding_catalog_change_problem_kb_v3",
        "v2 onboarding_catalog_and_change_problem_kb_merged extended by 46ca2862_56a4f834 (register_new_service_offering) "
        "and 99ba2325_7915481a (create_knowledge_article); train only",
    ),
    "itsmbench_eops_01395ded_eff85dae": ("notification_queue_repair", "update a stuck notification and send role-specific notifications; train"),
    "itsmbench_eops_78a5b71e_a4fb07a9": ("change_post_implementation_review", "advance a change to review (update_change only); SFT-only source in train"),
    "itsmbench_eops_99ba2325_54f35603": ("service_hold_instead_of_retirement", "set a service and its offerings non-operational instead of retiring it; held-out"),
    "itsmbench_eops_99ba2325_887f5580": ("ci_retire_and_affected_link", "retire a CI, link an active CI as affected, refuse the retired one; train"),
}
#: Business type of each v3 addition, curated by its primary request.
ADDED_BUSINESS_TYPES = {
    "itsmbench_eops_99ba2325_887f5580": "cmdb",
    "itsmbench_eops_01395ded_eff85dae": "incident_lifecycle",
    "itsmbench_eops_46ca2862_56a4f834": "service_catalog",
    "itsmbench_eops_99ba2325_7915481a": "knowledge",
    "itsmbench_eops_78a5b71e_a4fb07a9": "problem_change_lifecycle",
    "itsmbench_eops_99ba2325_54f35603": "service_catalog",
}
V3_USAGE_NOTE = (
    "train: training-source union (SFT and GRPO candidates); a task's SFT/GRPO use comes only from the usage "
    "registry configs/data/itsm_usage_v1.json, never from this split (SFT-only and R1-pending tasks are in train "
    "but not GRPO-trainable); dev: debugging and budget/length checks only, not checkpoint selection; heldout: "
    "final frozen evaluation only, never for debugging, tuning, length/budget work, Teacher demonstrations or "
    "checkpoint selection"
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _source_files() -> dict[str, Path]:
    return {"pilot": DEFAULT_PILOT_PATH, "batch2": DEFAULT_BATCH2_PATH, "heldout_candidates": DEFAULT_HELDOUT_V3_PATH}


def added_raw_tasks() -> dict[str, tuple[str, dict[str, Any]]]:
    """The six v3 additions: task ID -> (source file key, full raw entry)."""
    batch = load_pilot(DEFAULT_BATCH2_PATH)
    accepted = {e["id"]: e for e in batch["tasks"]}
    isolated = {i["attempted_conversion"]["id"]: i["attempted_conversion"] for i in batch["isolated"] if "attempted_conversion" in i}
    heldout = {e["id"]: e for e in load_pilot(DEFAULT_HELDOUT_V3_PATH)["tasks"]}
    added: dict[str, tuple[str, dict[str, Any]]] = {}
    for tid in ACCEPTED_V3_TRAIN_TASK_IDS:
        added[tid] = ("batch2", accepted[tid])
    for tid in SFT_ONLY_ADMITTED_TASK_IDS:
        added[tid] = ("batch2", isolated[tid])
    for tid in HELDOUT_ADDED_TASK_IDS:
        added[tid] = ("heldout_candidates", heldout[tid])
    return added


def all_raw_tasks() -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Raw entries of all 62 v3 tasks and the source key of each."""
    raw = {t["id"]: t for t in load_raw_tasks()}
    source_of = {tid: "bench2_tasks_json" for tid in raw}
    for tid, entry in derived_raw_tasks().items():
        raw[tid], source_of[tid] = entry, "pilot"
    for tid, (key, entry) in added_raw_tasks().items():
        if tid in raw:
            raise ValueError(f"v3 addition {tid} collides with a v2 task")
        raw[tid], source_of[tid] = entry, key
    return raw, source_of


def _data_block() -> dict[str, Any]:
    return {
        "bench2": data_version(),
        "source_files": {
            key: {
                "path": _rel(path),
                "version": load_pilot(path)["version"],
                "sha256": _sha256_bytes(path.read_bytes()),
            }
            for key, path in _source_files().items()
        },
        "base_manifests": {
            "itsm_split_v1": {"path": _rel(DEFAULT_MANIFEST_PATH), "sha256": _sha256_bytes(DEFAULT_MANIFEST_PATH.read_bytes())},
            "itsm_split_v2": {"path": _rel(DEFAULT_V2_PATH), "sha256": _sha256_bytes(DEFAULT_V2_PATH.read_bytes())},
        },
    }


def _derived_record(entry: dict[str, Any], source_key: str) -> dict[str, Any]:
    source = entry["source"]
    return {
        "dataset": "EnterpriseOps-Gym",
        "repo": source["repo"],
        "commit": source["commit"],
        "hf_dataset": source["hf_dataset"],
        "hf_revision": source["hf_revision"],
        "license": source["license"],
        "attribution": source["attribution"],
        "upstream_task_id": source["upstream_task_id"],
        "upstream_user_prompt_sha256": source["upstream_user_prompt_sha256"],
        "upstream_seed": source["upstream_seed"],
        "upstream_seed_sha256": source["upstream_seed_sha256"],
        "source_file": source_key,
        "route": entry.get("route", "faithful"),
        "org_ids": entry.get("org_ids"),
        "equivalent_to_upstream_evaluation": False,
    }


def isolation_audit(raw: dict[str, dict[str, Any]], split_of: dict[str, str], features) -> dict[str, Any]:
    """Training (v3 train) against evaluation (dev + held-out): shared write tools,
    gold entities, near duplicates and upstream sources. Families are checked by
    ``validate_split_structure`` (a family may not straddle splits)."""
    seed_ids = {seed: _seed_ids(seed) for seed in {t["seed_db"] for t in raw.values()}}
    ents = {tid: (_gold_ids(t) & seed_ids[t["seed_db"]]) - _delta_created(t) for tid, t in raw.items()}
    writes = {tid: f.write_tools for tid, f in features.items()}
    training = sorted(t for t in raw if split_of[t] == "train")
    evaluation = sorted(t for t in raw if split_of[t] in ("dev", "heldout"))
    derived = {t for t in raw if t.startswith("itsmbench_eops_")}
    flagged = []
    for t, e in itertools.product(training, evaluation):
        shared_writes = sorted(writes[t] & writes[e])
        shared_entities = sorted(ents[t] & ents[e]) if raw[t]["seed_db"] == raw[e]["seed_db"] else []
        near_duplicate = len(shared_entities) >= 2 and writes[t] == writes[e]
        if (t in derived or e in derived) and (len(shared_writes) >= 2 or len(shared_entities) >= 2 or near_duplicate):
            flagged.append({"train": t, "eval": e, "shared_write_tools": shared_writes, "shared_gold_entities": shared_entities, "near_duplicate": near_duplicate})
    bench_pairs = sum(
        1
        for t, e in itertools.product(training, evaluation)
        if t not in derived and e not in derived and len(writes[t] & writes[e]) >= 2
    )
    bench_near_duplicates = [
        {"train": t, "eval": e}
        for t, e in itertools.product(training, evaluation)
        if raw[t]["seed_db"] == raw[e]["seed_db"] and writes[t] == writes[e] and len(ents[t] & ents[e]) >= 2
    ]
    upstream = {tid: (raw[tid].get("source") or {}).get("upstream_task_id") for tid in derived}
    upstream_both = sorted({upstream[t] for t in training if t in derived} & {upstream[e] for e in evaluation if e in derived})
    checks = {
        "derived_pairs_share_le_1_write_tool_lt_2_entities_no_near_duplicate": not flagged,
        "no_near_duplicate_across_training_and_evaluation": not bench_near_duplicates,
        "no_upstream_task_in_training_and_evaluation": not upstream_both,
    }
    return {
        "rule": (
            "pairs involving a derived task: <= 1 shared write tool and < 2 shared same-seed gold entities with any "
            "dev/held-out task; no near duplicate (same seed, same write-tool set, >= 2 shared gold entities) between any "
            "training and evaluation task; no upstream task on both sides. Bench2 train x eval pairs sharing >= 2 write "
            "tools are the sub-workflow recombination accepted since v1 and are counted, not blocked"
        ),
        "training": len(training),
        "evaluation": len(evaluation),
        "flagged_pairs": flagged,
        "bench2_train_eval_pairs_sharing_ge_2_write_tools": bench_pairs,
        "near_duplicates": bench_near_duplicates,
        "upstream_in_both": upstream_both,
        "checks": checks,
    }


def build_manifest_v3() -> dict[str, Any]:
    v2 = load_manifest_v2()
    raw, source_of = all_raw_tasks()
    features = task_features(list(raw.values()))
    added = added_raw_tasks()

    split_of = {t: v2["tasks"][t]["split"] for t in v2["tasks"]}
    split_of.update({t: "train" for t in V3_TRAIN_ADDITIONS})
    split_of.update({t: "heldout" for t in HELDOUT_ADDED_TASK_IDS})
    business_type = {t: v2["tasks"][t]["business_type"] for t in v2["tasks"]}
    business_type.update(ADDED_BUSINESS_TYPES)

    v2_members = {name: set(entry["tasks"]) for name, entry in v2["families"].items()}
    families: dict[str, dict[str, Any]] = {}
    used_anchors = set()
    for members in connected_components(features):
        v2_families = sorted({v2["tasks"][t]["family"] for t in members if t in v2["tasks"]})
        new_members = [t for t in members if t in added]
        if not new_members and len(v2_families) == 1 and v2_members[v2_families[0]] == set(members):
            name, origin, note = v2_families[0], "unchanged_from_v2", None
        else:
            anchor = members[0]
            if anchor not in V3_FAMILY_LABELS:
                raise ValueError(f"v3 family anchored at {anchor} has no curated label: {list(members)}")
            used_anchors.add(anchor)
            name, note = V3_FAMILY_LABELS[anchor]
            origin = "merged_v2_families" if len(v2_families) > 1 else ("extended_v2_family" if v2_families else "new_in_v3")
            if name in v2_members:
                raise ValueError(f"v3 family {name} reuses a v2 family name with different members")
        homes = {split_of[t] for t in members}
        if len(homes) != 1:
            raise ValueError(f"v3 family {name} straddles splits {sorted(homes)}: {list(members)}")
        families[name] = {
            "split": homes.pop(),
            "origin": origin,
            "v2_families": v2_families,
            "added_in_v3": new_members,
            "business_types": sorted({business_type[t] for t in members}),
            **({"note": note} if note else {}),
            "tasks": list(members),
        }
    if used_anchors != set(V3_FAMILY_LABELS):
        raise ValueError("V3_FAMILY_LABELS no longer matches the computed families")
    families = dict(sorted(families.items()))
    family_of = {t: name for name, entry in families.items() for t in entry["tasks"]}
    splits = {split: sorted(t for t in raw if split_of[t] == split) for split in SPLITS}

    tasks = {}
    for t in sorted(raw):
        tasks[t] = {
            "split": split_of[t],
            "family": family_of[t],
            "business_type": business_type[t],
            "source": SOURCE_BENCH2 if source_of[t] == "bench2_tasks_json" else SOURCE_DERIVED,
            "source_file": source_of[t],
            "added_in": "v3" if t in added else ("v2" if source_of[t] == "pilot" else "v1"),
            "tenancy": features[t].tenancy,
            "seed_db": features[t].seed_db,
            "org_slicing": raw[t].get("org_ids") is not None,
            "objects_written": list(features[t].objects),
            "gold_actions": features[t].gold_action_count,
            "content_sha256": task_content_sha256(raw[t]),
            "source_sha256": _sha256_json(raw[t]),
        }
        if tasks[t]["source"] == SOURCE_DERIVED:
            tasks[t]["derived_from"] = _derived_record(raw[t], source_of[t])

    def count(values) -> dict[str, int]:
        return dict(sorted(collections.Counter(values).items()))

    return {
        "version": SPLIT_VERSION_V3,
        "positioning": v2["positioning"],
        "usage": V3_USAGE_NOTE,
        "method": {
            "base": (
                "split v2 kept: its 56 tasks keep their v2 split; dev is identical to v2; v3 adds four 6D-2 train tasks "
                "and the SFT-only source 78a5b71e_a4fb07a9 to train, and the 6D-3 task 99ba2325_54f35603 to heldout; "
                "the assignment is not re-optimised"
            ),
            "family_rule": (
                f"as v1/v2: link two tasks if the Jaccard similarity of their gold write-tool sets >= "
                f"{WRITE_JACCARD_THRESHOLD} or they share a creation tool; families are the connected components over "
                "all 62 tasks; a v2 family name is kept only for a family whose members are exactly its v2 members"
            ),
            "creation_tools": sorted(CREATION_TOOLS),
            "checksums": (
                "content_sha256: canonical JSON of the fields that define the task for env and scoring (as v2); "
                "source_sha256: canonical JSON of the task's full entry in its source file (provenance, upstream "
                "verifiers and validation cases included)"
            ),
            "accepted": ACCEPTANCE,
        },
        "data": _data_block(),
        "sizes": {split: len(splits[split]) for split in SPLITS},
        "family_changes": {
            "v2_family_count": len(v2_members),
            "v3_family_count": len(families),
            "unchanged_from_v2": sorted(n for n, e in families.items() if e["origin"] == "unchanged_from_v2"),
            "v2_families_no_longer_present": sorted(set(v2_members) - set(families)),
            "changed": {
                n: {k: e[k] for k in ("origin", "v2_families", "added_in_v3", "split")}
                for n, e in families.items()
                if e["origin"] != "unchanged_from_v2"
            },
        },
        "isolation": isolation_audit(raw, split_of, features),
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
            }
            for split, ids in splits.items()
        },
    }


def validate_manifest_v3(manifest: dict[str, Any]) -> None:
    """Raise unless ``manifest`` is a consistent v3 manifest equal to a fresh rebuild."""
    if manifest.get("version") != SPLIT_VERSION_V3:
        raise ValueError(f"expected {SPLIT_VERSION_V3}, got {manifest.get('version')!r}")
    v2 = load_manifest_v2()
    added = set(V3_TRAIN_ADDITIONS) | set(HELDOUT_ADDED_TASK_IDS)
    validate_split_structure(manifest, set(v2["tasks"]) | added)
    sizes = {split: len(manifest["splits"][split]) for split in SPLITS}
    if sizes != EXPECTED_SIZES or manifest.get("sizes") != EXPECTED_SIZES:
        raise ValueError(f"v3 sizes must be {EXPECTED_SIZES}, got lists {sizes} / declared {manifest.get('sizes')}")
    if manifest["splits"]["dev"] != v2["splits"]["dev"]:
        raise ValueError("v3 dev differs from v2")
    for task_id, entry in v2["tasks"].items():
        if manifest["tasks"][task_id]["split"] != entry["split"]:
            raise ValueError(f"task {task_id} moved from its v2 split {entry['split']}")
    for task_id in V3_TRAIN_ADDITIONS:
        if manifest["tasks"][task_id]["split"] != "train":
            raise ValueError(f"v3 train addition {task_id} must be in train")
    for task_id in HELDOUT_ADDED_TASK_IDS:
        if manifest["tasks"][task_id]["split"] != "heldout":
            raise ValueError(f"v3 held-out addition {task_id} must be in heldout")
    if set(manifest["tasks"]) - set(v2["tasks"]) != added:
        raise ValueError("v3 must add exactly the accepted tasks")
    if manifest["data"] != _data_block():
        raise ValueError("Bench2 data, a source file or a base manifest changed since v3 was frozen")
    raw, _ = all_raw_tasks()
    for task_id, task in raw.items():
        entry = manifest["tasks"][task_id]
        if entry["content_sha256"] != task_content_sha256(task):
            raise ValueError(f"task {task_id}: content checksum mismatch")
        if entry["source_sha256"] != _sha256_json(task):
            raise ValueError(f"task {task_id}: source checksum mismatch")
    if not all(manifest["isolation"]["checks"].values()):
        raise ValueError(f"v3 isolation checks fail: {manifest['isolation']['checks']}")
    if manifest != build_manifest_v3():
        raise ValueError("v3 manifest differs from a fresh rebuild")


def load_manifest_v3(path: Path = DEFAULT_V3_PATH) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest_v3(manifest)
    return manifest
