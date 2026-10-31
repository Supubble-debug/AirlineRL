"""EnterpriseOps-Gym ITSM migration pilot (step 6B).

The pilot tasks live in their own file (``configs/data/itsm_eops_pilot_v1.json``):
they are not part of Bench2's ``tasks.json`` and not part of the v1 split. Each
entry is a Bench2 ``Task`` plus provenance, the upstream verifiers (Apache-2.0,
attributed) used as an independent second check, and validation cases.

Nothing here reads the upstream reference checkout; the upstream verifiers are
run against a Bench2 DB exported to in-memory SQLite.
"""

from __future__ import annotations

import collections
import copy
import itertools
import json
import re
import sqlite3
import typing
from pathlib import Path
from typing import Any

from enterprise_worlds.data_model.tasks import Task
from enterprise_worlds.domains.itsm.data_model import ItsmDB
from enterprise_worlds.environment.environment import Environment

from agent_rl.data.itsm_split import (
    CREATION_TOOLS,
    WRITE_JACCARD_THRESHOLD,
    load_manifest,
    load_raw_tasks,
    write_tool_names,
)
from agent_rl.envs.itsm_scoring import build_task_environment, task_clock


PILOT_VERSION = "itsm_eops_pilot_v1"
DEFAULT_PILOT_PATH = Path(__file__).resolve().parents[3] / "configs" / "data" / "itsm_eops_pilot_v1.json"
_RECORD_ID = re.compile(r"\b[A-Z]{2,7}_\d{3,}\b")


def load_pilot(path: Path = DEFAULT_PILOT_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pilot_tasks(path: Path = DEFAULT_PILOT_PATH) -> dict[str, Task]:
    """Validated Bench2 tasks of the pilot (extra provenance keys are ignored)."""
    return {entry["id"]: Task.model_validate(entry) for entry in load_pilot(path)["tasks"]}


# -- upstream verifiers on a Bench2 DB ------------------------------------------------------------


def _deny_attach(action: int, *args: Any) -> int:
    return sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_ATTACH else sqlite3.SQLITE_OK


def _sql_value(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def db_to_sqlite(db: ItsmDB) -> sqlite3.Connection:
    """Every Bench2 collection as a table of its model fields (booleans as 0/1)."""
    connection = sqlite3.connect(":memory:")
    connection.set_authorizer(_deny_attach)
    for name, field in type(db).model_fields.items():
        args = typing.get_args(field.annotation)
        if len(args) != 2 or not hasattr(args[1], "model_fields"):
            continue
        columns = list(args[1].model_fields)
        connection.execute(f'CREATE TABLE "{name}" ({", ".join(repr(c) for c in columns)})')
        rows = [[_sql_value(record.model_dump()[c]) for c in columns] for record in getattr(db, name).values()]
        if rows:
            connection.executemany(
                f'INSERT INTO "{name}" VALUES ({", ".join("?" * len(columns))})', rows
            )
    return connection


def run_upstream_verifiers(db: ItsmDB, verifiers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run upstream ``database_state`` verifiers (COUNT queries, ``equals``)."""
    connection = db_to_sqlite(db)
    results = []
    for verifier in verifiers:
        try:
            row = connection.execute(verifier["query"]).fetchone()
            value, error = (row[0] if row else None), None
        except sqlite3.Error as exc:
            value, error = None, str(exc)
        results.append(
            {
                "name": verifier["name"],
                "expected": verifier["expected_value"],
                "actual": value,
                "error": error,
                "passed": error is None and value == verifier["expected_value"],
            }
        )
    connection.close()
    return results


def run_actions(task: Task, actions: list[dict[str, Any]]) -> Environment:
    """Execute actions on a fresh env under the task clock; every action must succeed."""
    with task_clock(task):
        environment = build_task_environment(task)
        for action in actions:
            environment.make_tool_call(action["name"], **copy.deepcopy(action["arguments"]))
    return environment


def edited_actions(task: Task, edit: dict[str, Any]) -> list[dict[str, Any]]:
    """Gold actions with ``set`` (index -> arguments), ``drop`` (indices) and ``extra`` applied."""
    actions = [{"name": a.name, "arguments": copy.deepcopy(a.arguments)} for a in task.evaluation_criteria.actions]
    for index, arguments in edit.get("set", {}).items():
        actions[int(index)]["arguments"].update(arguments)
    for index in sorted((int(i) for i in edit.get("drop", ())), reverse=True):
        del actions[index]
    return actions + copy.deepcopy(edit.get("extra", []))


# -- cross-source family and overlap re-check ------------------------------------------------------


def _write_set(task: dict[str, Any], writes: frozenset[str]) -> frozenset[str]:
    return frozenset(a["name"] for a in task["evaluation_criteria"]["actions"] if a["name"] in writes)


def _gold_ids(task: dict[str, Any]) -> set[str]:
    return {
        v
        for a in task["evaluation_criteria"]["actions"]
        for v in a["arguments"].values()
        if isinstance(v, str) and _RECORD_ID.fullmatch(v)
    }


def _delta_created(task: dict[str, Any]) -> set[str]:
    delta = task.get("initial_state_delta") or {}
    return {rid for ops in delta.values() for rid, op in ops.items() if "create" in op}


def _seed_ids(seed_db: str) -> set[str]:
    from enterprise_worlds.domains.itsm import environment as itsm_environment

    seed = json.loads((itsm_environment.ITSM_TASKS_PATH.parent / seed_db).read_text(encoding="utf-8"))
    return {key for value in seed.values() if isinstance(value, dict) for key in value}


def family_overlap_check(pilot_entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute step 6's family rule over the 53 Bench2 tasks plus the pilot tasks.

    The v1 split stays valid only if every family component that contains a pilot
    task has v1 members in train only (or none), and no component joins v1 tasks
    from different splits. Also reports same-seed entity overlap and the 6A gate
    (<= 1 write tool shared with any dev/held-out task).
    """
    writes = write_tool_names()
    manifest = load_manifest()
    bench = {t["id"]: t for t in load_raw_tasks()}
    pilot = {t["id"]: t for t in pilot_entries}
    combined = {**bench, **pilot}
    sets = {tid: _write_set(t, writes) for tid, t in combined.items()}

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
            if a in pilot or b in pilot:
                links.append({"a": a, "b": b, "write_jaccard": round(jaccard, 3), "shared_creation_tools": shared_creation})
    components = collections.defaultdict(list)
    for tid in sorted(combined):
        components[root(tid)].append(tid)

    seed_ids = {seed: _seed_ids(seed) for seed in {t["seed_db"] for t in combined.values()}}
    per_task = {}
    for pid, entry in sorted(pilot.items()):
        component = next(c for c in components.values() if pid in c)
        v1_members = [t for t in component if t in bench]
        entity_overlap, eval_overlap = [], (0, None)
        for bid, btask in bench.items():
            split = manifest["tasks"][bid]["split"]
            if split != "train":
                eval_overlap = max(eval_overlap, (len(sets[pid] & sets[bid]), bid), key=lambda x: x[0])
            if btask["seed_db"] != entry["seed_db"]:
                continue
            shared = (_gold_ids(entry) & _gold_ids(btask) & seed_ids[entry["seed_db"]]) - _delta_created(btask) - _delta_created(entry)
            if shared:
                entity_overlap.append({"bench2_task": bid, "split": split, "same_seed_entities": sorted(shared)})
        per_task[pid] = {
            "component_v1_members": v1_members,
            "component_v1_splits": sorted({manifest["tasks"][t]["split"] for t in v1_members}),
            "component_v1_families": sorted({manifest["tasks"][t]["family"] for t in v1_members}),
            "component_pilot_members": [t for t in component if t in pilot],
            "same_seed_entity_overlap": entity_overlap,
            "max_shared_write_tools_with_dev_or_heldout": {"count": eval_overlap[0], "task": eval_overlap[1]},
        }

    v1_family_of = {t: manifest["tasks"][t]["family"] for t in bench}
    bridged = [
        sorted({v1_family_of[t] for t in c if t in bench})
        for c in components.values()
        if len({manifest["tasks"][t]["split"] for t in c if t in bench}) > 1
    ]
    eval_entity = [
        (pid, o["bench2_task"])
        for pid, r in per_task.items()
        for o in r["same_seed_entity_overlap"]
        if o["split"] != "train" and len(o["same_seed_entities"]) >= 2
    ]
    checks = {
        "pilot_components_touch_train_only": all(set(r["component_v1_splits"]) <= {"train"} for r in per_task.values()),
        "no_component_bridges_v1_splits": not bridged,
        "no_pilot_shares_2plus_entities_with_dev_or_heldout": not eval_entity,
        "pilot_shares_le_1_write_tool_with_dev_or_heldout": all(
            r["max_shared_write_tools_with_dev_or_heldout"]["count"] <= 1 for r in per_task.values()
        ),
        "dev_and_heldout_unchanged": all(
            sorted(t for t in c if t in bench) == sorted(manifest["families"][v1_family_of[c[0]]]["tasks"])
            for c in components.values()
            if c[0] in bench and manifest["tasks"][c[0]]["split"] != "train"
        ),
    }
    return {"checks": checks, "pilot_links": links, "bridged_v1_families": bridged, "tasks": per_task}
