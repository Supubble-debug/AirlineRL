"""Feasibility audit of EnterpriseOps-Gym ITSM tasks for the Bench2 ITSM setup (step 6A).

Usage (from the repo root):
    python scripts/data/audit_eopsgym_itsm.py --ref-dir DIR [--out FILE]

DIR is the isolated reference directory holding the pinned upstream checkout
(``repo/``), the downloaded ITSM parquet files (``hf-dataset/``) and the ITSM SQL
seeds extracted from ``gym_dbs.zip`` (``itsm-seeds/``). Nothing in DIR is
imported or executed as code: parquet and JSON are parsed, and the SQL seeds and
verifier queries run in in-memory SQLite with ATTACH denied. Nothing is written
to Bench2 or to the project task set; no model or LLM is called.

The upstream SQL seeds contain INSERT statements only (the table definitions
live in the closed MCP server image), so each table is created with the union
of the columns its INSERTs name and the fields of the matching Bench2 model.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import re
import sqlite3
import sys
import typing
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from enterprise_worlds.domains.itsm import environment as itsm_environment
from enterprise_worlds.domains.itsm.data_model import _COMPOSITE_KEYS, _ID_SPEC, ItsmDB
from enterprise_worlds.domains.itsm.tools import ItsmTools

from agent_rl.data.itsm_split import (
    CREATION_TOOLS,
    WRITE_JACCARD_THRESHOLD,
    load_manifest,
    load_raw_tasks,
    write_tool_names,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "docs" / "itsm_step6a" / "eopsgym_itsm_audit.json"
UPSTREAM_REPO = "https://github.com/ServiceNow/EnterpriseOps-Gym"
UPSTREAM_COMMIT = "271f2c357f763376997dfd16807fcde2474ae41b"
HF_DATASET = "ServiceNow-AI/EnterpriseOps-Gym"
HF_REVISION = "c8e538eae8a6205294f0a86675fefdc1fac408f6"
MODES = ("oracle", "plus_5_tools", "plus_10_tools", "plus_15_tools")
SEED_DIR = "Domain Wise DBs and Task-DB Mappings/itsm/dbs"
SEED_MATCH_THRESHOLD = 0.9
#: Issue -> category effect. Blocking: unusable without changing the task's meaning,
#: Bench2 tools, or the frozen v1 split. Minor: mechanical work (seed delta, verifier
#: clean-up or ID re-pinning). Informational only: weaker validation, no effect on reward.
BLOCKING_ISSUES = {
    "tools_missing_in_bench2": "an oracle tool does not exist in Bench2",
    "non_db_verifier": "a response_check/tool_execution verifier has no DB-only equivalent",
    "verifier_does_not_execute": "a verifier references a table/column that exists nowhere",
    "verifier_reads_column_absent_in_bench2": "a verifier reads a column Bench2 does not model",
    "conflicting_verifiers": "two verifiers require different values on the same record (all must pass)",
    "no_write_tool": "no write tool: nothing for DB scoring to check",
    "links_dev_or_heldout_family": "shares a v1 dev/held-out workflow family; importing it would change those splits",
    "links_families_in_several_v1_splits": "bridges families from different v1 splits",
    "near_duplicate_of_bench2_task": "same seed entities and workflow, or near-identical text, as an existing task",
    "shares_entities_with_dev_or_heldout_task": ">= 2 same seed records as a v1 dev/held-out task (would put them into training)",
    "seed_not_in_bench2": "seed matches neither Bench2 seed",
}
MINOR_ISSUES = {
    "seed_variant_needs_delta": "upstream seed differs from the Bench2 seed; needs an initial_state_delta",
    "verifier_pins_created_record_id": "a verifier hard-codes the next sequential ID of a created record",
    "pinned_created_id_exists_in_bench2_seed": "that pinned ID already exists in the Bench2 seed",
    "verifier_id_absent_from_bench2_seed": "a verifier names a seed record the Bench2 seed lacks",
    "duplicate_verifiers": "identical verifiers repeated",
}
#: Proposed 6B pilot (training-side only); rationale in docs/itsm_step6a/README.md.
PILOT_TASKS = (
    "task_20251219_144431_663_97d7df7d_a52988e5",
    "task_20260113_182602_664_46ca2862_b6ef60ca",
    "task_20251218_104259_233_66392a82_74b17170",
    "task_20260114_085016_654_e6427839_ca9119ec",
    "task_20251218_124419_383_66392a82_52c8290e",
    "task_20260114_094327_107_46ca2862_796e7685",
    "task_20260116_081645_096_66392a82_fd509f97",
    "task_20251223_101136_983_1d219d93_d1ba6c2d",
)
MAX_SHARED_WRITES_WITH_EVAL_SPLITS = 1
INFORMATIONAL_ISSUES = {
    "verifier_passes_on_initial_seed": "a verifier already passes before any action (no discrimination)",
    "weak_verifier_predicate": "LIKE / IS NOT NULL / != predicates accept loose matches",
    "shares_entities_with_train_task": ">= 2 same seed records as a v1 train task (overlap stays inside train)",
}
PROMPT_JACCARD_FLAG = 0.35
_INSERT = re.compile(r'INSERT\s+INTO\s+"?(\w+)"?\s*\(([^)]*)\)\s*VALUES', re.IGNORECASE)
_EQ = re.compile(r"""\b(\w+)\s*=\s*(?:'([^']*)'|"([^"]*)"|(-?\d+(?:\.\d+)?|true|false)\b)""", re.IGNORECASE)
_RECORD_ID = re.compile(r"\b[A-Z]{2,7}_\d{3,}\b")
_WEAK = (
    ("is_not_null", re.compile(r"\bIS\s+NOT\s+NULL\b", re.IGNORECASE)),
    ("like_pattern", re.compile(r"\bLIKE\b", re.IGNORECASE)),
    ("or_tautology", re.compile(r"\bOR\s+\w+\s+IS\s+NOT\s+NULL\b", re.IGNORECASE)),
    ("not_equal", re.compile(r"!=|<>")),
)


def _json(value: Any) -> Any:
    while isinstance(value, str):
        value = json.loads(value)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{4,}", text.lower()))


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def _verifier_key(verifier: dict) -> str:
    return json.dumps(verifier, sort_keys=True)


# -- Bench2 side -------------------------------------------------------------------------------


def bench2_model_fields() -> dict[str, set[str]]:
    fields = {}
    for name, field in ItsmDB.model_fields.items():
        args = typing.get_args(field.annotation)
        if len(args) == 2 and hasattr(args[1], "model_fields"):
            fields[name] = set(args[1].model_fields)
    return fields


def bench2_seed(path: Path) -> dict[str, dict[str, dict]]:
    db = ItsmDB.load(path).model_dump()
    return {table: records for table, records in db.items() if isinstance(records, dict)}


def _norm(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        return value.replace("T", " ") if re.fullmatch(r"\d{4}-\d\d-\d\dT[\d:.]+", value) else value
    return value


# -- upstream side -----------------------------------------------------------------------------


def _deny_attach(action: int, *args: Any) -> int:
    return sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_ATTACH else sqlite3.SQLITE_OK


class Seed:
    """An in-memory seed DB plus the columns its INSERT statements name."""

    def __init__(self, connection: Seed, inserted_columns: dict[str, set[str]]) -> None:
        self.connection = connection
        self.inserted_columns = inserted_columns

    def execute(self, query: str):
        return self.connection.execute(query)

    def set_authorizer(self, callback) -> None:
        self.connection.set_authorizer(callback)


def load_seed(path: Path, model_fields: dict[str, set[str]]) -> Seed:
    sql = path.read_text(encoding="utf-8")
    columns: dict[str, list[str]] = collections.defaultdict(list)
    for table, names in _INSERT.findall(sql):
        for column in (c.strip().strip('"') for c in names.split(",")):
            if column not in columns[table]:
                columns[table].append(column)
    connection = sqlite3.connect(":memory:")
    connection.set_authorizer(_deny_attach)
    for table in sorted(set(columns) | set(model_fields)):
        names = list(columns.get(table, [])) + sorted(model_fields.get(table, set()) - set(columns.get(table, [])))
        connection.execute(f'CREATE TABLE "{table}" ({", ".join(f"{c!r}" for c in names)})')
    connection.executescript(sql)
    connection.set_authorizer(None)
    return Seed(connection, {t: set(c) for t, c in columns.items()})


def _row_key(table: str, row: dict) -> str:
    """Bench2's record key: the id field, or ':'-joined segments for composite tables."""
    if table in _COMPOSITE_KEYS:
        return ":".join(str(row[field]) for _, _, field in _COMPOSITE_KEYS[table])
    return str(row[_ID_SPEC[table][1]])


def seed_tables(connection: Seed) -> dict[str, dict[str, dict]]:
    """table -> Bench2 record key -> row, for tables that have rows."""
    out = {}
    for table in connection.inserted_columns:
        cursor = connection.execute(f'SELECT * FROM "{table}"')
        names = [d[0] for d in cursor.description]
        rows = [dict(zip(names, row)) for row in cursor.fetchall()]
        if rows and (table in _ID_SPEC or table in _COMPOSITE_KEYS):
            out[table] = {_row_key(table, row): row for row in rows}
    return out


def seed_diff(upstream: dict[str, dict[str, dict]], bench: dict[str, dict[str, dict]]) -> dict[str, Any]:
    """Records that differ between an upstream seed and a Bench2 seed (timestamps ignored)."""
    volatile = {"created_at", "updated_at", "created_on", "updated_on"}
    only_upstream, only_bench, changed, renamed = [], [], [], []
    for table in sorted(set(upstream) | set(bench)):
        u, b = upstream.get(table, {}), bench.get(table, {})
        id_field = _ID_SPEC[table][1] if table in _ID_SPEC else None

        def content(row: dict) -> str:
            return json.dumps({c: _norm(v) for c, v in row.items() if c not in volatile and c != id_field}, sort_keys=True, default=str)

        b_by_content = {content(b[k]): k for k in set(b) - set(u)}
        for key in sorted(set(u) - set(b)):
            match = b_by_content.pop(content(u[key]), None)
            if match is not None:
                renamed.append(f"{table}/{key}->{match}")
            else:
                only_upstream.append(f"{table}/{key}")
        only_bench += [f"{table}/{k}" for k in sorted(b_by_content.values())]
        for key in sorted(set(u) & set(b)):
            fields = sorted(
                c for c in (set(u[key]) & set(b[key])) - volatile if _norm(u[key][c]) != _norm(b[key][c])
            )
            if fields:
                changed.append({"record": f"{table}/{key}", "fields": {c: [u[key][c], b[key][c]] for c in fields}})
    total = sum(len(rows) for rows in upstream.values())
    same = total - len(only_upstream) - len(changed) - len(renamed)
    return {
        "records": total,
        "same_records": same,
        "same_fraction": round((same + len(renamed)) / total, 4) if total else 0.0,
        "renamed_ids_same_content": renamed,
        "delta_size": len(only_upstream) + len(only_bench) + len(changed),
        "only_in_upstream": only_upstream,
        "only_in_bench2": only_bench,
        "changed": changed,
    }


def run_verifier(connection: Seed, query: str) -> dict[str, Any]:
    reads: set[tuple[str, str]] = set()

    def authorizer(action, arg1, arg2, *rest):
        if action == sqlite3.SQLITE_ATTACH:
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_READ and arg1 and arg2:
            reads.add((arg1, arg2))
        return sqlite3.SQLITE_OK

    connection.set_authorizer(authorizer)
    try:
        row = connection.execute(query).fetchone()
        return {"ok": True, "value": row[0] if row else None, "reads": sorted(reads)}
    except sqlite3.Error as error:
        return {"ok": False, "error": str(error), "reads": sorted(reads)}
    finally:
        connection.set_authorizer(None)


def _compare(actual: Any, expected: Any, comparison: str) -> bool | None:
    try:
        if comparison == "equals":
            return actual == expected
        if comparison == "greater_than":
            return actual > expected
        if comparison == "less_than":
            return actual < expected
        if comparison == "contains":
            return str(expected) in str(actual)
    except TypeError:
        return None
    return None


def _predicates(query: str) -> dict[str, set[str]]:
    found: dict[str, set[str]] = collections.defaultdict(set)
    parts = re.split(r"\bWHERE\b", query, maxsplit=1, flags=re.IGNORECASE)
    where = parts[1] if len(parts) == 2 else ""
    for column, single, double, bare in _EQ.findall(where):
        found[column.lower()].add((single or double or bare).lower())
    return found


def _main_table(query: str) -> str | None:
    match = re.search(r"\bFROM\s+\"?(\w+)", query, re.IGNORECASE)
    return match.group(1) if match else None


def audit_verifiers(task: dict, connection: Seed, model_fields) -> dict[str, Any]:
    verifiers = _json(task["verifiers"])
    keys = [_verifier_key(v) for v in verifiers]
    queries = [" ".join(v.get("validation_config", {}).get("query", "").split()).lower() for v in verifiers]
    results = []
    for verifier in verifiers:
        kind = verifier.get("verifier_type")
        config = verifier.get("validation_config", {})
        entry: dict[str, Any] = {"name": verifier.get("name"), "type": kind}
        if kind != "database_state":
            entry["note"] = "not a database_state verifier"
            results.append(entry)
            continue
        query = config.get("query", "")
        run = run_verifier(connection, query)
        expected, comparison = config.get("expected_value"), config.get("comparison_type", "equals")
        unknown_columns = sorted(
            f"{t}.{c}"
            for t, c in run["reads"]
            if c not in model_fields.get(t, set())
            and c not in connection.inserted_columns.get(t, set())
        )
        not_in_bench2 = sorted(f"{t}.{c}" for t, c in run["reads"] if t not in model_fields or c not in model_fields[t])
        entry.update(
            {
                "query": query,
                "expected": expected,
                "comparison": comparison,
                "executes_on_seed": run["ok"],
                "error": run.get("error"),
                "initial_value": run.get("value"),
                "passes_on_initial_seed": _compare(run.get("value"), expected, comparison) if run["ok"] else None,
                "columns_read": [f"{t}.{c}" for t, c in run["reads"]],
                "columns_not_in_bench2_model": not_in_bench2,
                "columns_unknown_anywhere": unknown_columns,
                "weak_predicates": [name for name, pattern in _WEAK if pattern.search(query)],
                "expects_absence": comparison == "equals" and expected == 0,
            }
        )
        results.append(entry)

    duplicates = [
        [i, j] for i, j in itertools.combinations(range(len(verifiers)), 2) if keys[i] == keys[j] or (queries[i] and queries[i] == queries[j])
    ]
    conflicts = []
    db_entries = [(i, v) for i, v in enumerate(verifiers) if v.get("verifier_type") == "database_state"]
    for (i, a), (j, b) in itertools.combinations(db_entries, 2):
        qa, qb = a["validation_config"].get("query", ""), b["validation_config"].get("query", "")
        if _main_table(qa) != _main_table(qb):
            continue
        ea, eb = a["validation_config"].get("expected_value"), b["validation_config"].get("expected_value")
        if not (isinstance(ea, int) and isinstance(eb, int) and ea >= 1 and eb >= 1):
            continue
        table = _main_table(qa)
        if table not in _ID_SPEC:
            continue
        pk = _ID_SPEC[table][1].lower()
        pa, pb = _predicates(qa), _predicates(qb)
        if not (pk in pa and pk in pb and pa[pk] == pb[pk] and len(pa[pk]) == 1):
            continue
        ids = {pk}
        clash = {c: [sorted(pa[c]), sorted(pb[c])] for c in pa if c in pb and c not in ids and pa[c] != pb[c] and len(pa[c]) == len(pb[c]) == 1}
        if clash:
            conflicts.append({"verifiers": [i, j], "same_record": sorted(ids), "contradicting_columns": clash})
    return {"verifiers": results, "duplicate_pairs": duplicates, "conflicts": conflicts}


# -- main audit ----------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    ref = args.ref_dir.resolve()

    head = (ref / "repo" / ".git" / "HEAD").read_text(encoding="utf-8").strip()
    if head != UPSTREAM_COMMIT:
        raise SystemExit(f"reference checkout is at {head}, expected {UPSTREAM_COMMIT}")
    parquet = {m: ref / "hf-dataset" / m / "itsm-00000-of-00001.parquet" for m in MODES}
    seed_paths = sorted((ref / "itsm-seeds" / SEED_DIR).glob("*.sql"))
    data = {m: {r["task_id"]: r for r in pq.read_table(p).to_pylist()} for m, p in parquet.items()}
    oracle = data["oracle"]

    # 1. modes vs independent tasks
    same_ids = all(set(data[m]) == set(oracle) for m in MODES)
    field_diffs = collections.Counter()
    order_only = collections.Counter()
    extra_tools = {}
    for mode in MODES[1:]:
        for tid, row in oracle.items():
            other = data[mode][tid]
            for field in row:
                if row[field] != other[field]:
                    field_diffs[f"{mode}.{field}"] += 1
            a, b = _json(row["verifiers"]), _json(other["verifiers"])
            if a != b and sorted(map(_verifier_key, a)) == sorted(map(_verifier_key, b)):
                order_only[mode] += 1
        extra = [len(set(data[mode][t]["selected_tools"]) - set(oracle[t]["selected_tools"])) for t in oracle]
        extra_tools[mode] = {
            "oracle_tools_always_included": all(set(oracle[t]["selected_tools"]) <= set(data[mode][t]["selected_tools"]) for t in oracle),
            "extra_tools_min": min(extra),
            "extra_tools_max": max(extra),
        }
    verifier_multiset_equal = all(
        sorted(map(_verifier_key, _json(oracle[t]["verifiers"]))) == sorted(map(_verifier_key, _json(data[m][t]["verifiers"])))
        for m in MODES
        for t in oracle
    )

    def content_key(row: dict) -> str:
        return json.dumps(
            [" ".join(row["user_prompt"].lower().split()), _json(row["gym_servers_config"]), sorted(map(_verifier_key, _json(row["verifiers"])))],
            sort_keys=True,
        )

    groups = collections.defaultdict(list)
    for tid, row in oracle.items():
        groups[content_key(row)].append(tid)
    exact_duplicates = [sorted(v) for v in groups.values() if len(v) > 1]
    prompt_groups = collections.defaultdict(list)
    for tid, row in oracle.items():
        prompt_groups[" ".join(row["user_prompt"].lower().split())].append(tid)

    # upstream task features
    bench_writes = write_tool_names()
    bench_tools = set(ItsmTools._tool_names)
    model_fields = bench2_model_fields()
    seed_of, user_of = {}, {}
    for tid, row in oracle.items():
        server = _json(_json(row["gym_servers_config"])[0])
        seed_of[tid] = server["seed_database_file"].split("/")[-1]
        user = _json(server.get("user_info") or {})
        user_of[tid] = {"user_id": user.get("user_id"), "name": user.get("name")}

    # 2. seeds: schema and initial state vs Bench2
    connections = {p.name: load_seed(p, model_fields) for p in seed_paths}
    bench_seed_paths = {
        name: itsm_environment.ITSM_TASKS_PATH.parent / name for name in ("msp_db.json", "single_tenant_db.json")
    }
    bench_seeds = {name: bench2_seed(path) for name, path in bench_seed_paths.items()}
    seed_report = {}
    for name, connection in connections.items():
        tables = seed_tables(connection)
        inserted = connection.inserted_columns
        match = {name: seed_diff(tables, bench) for name, bench in bench_seeds.items()}
        best = max(match, key=lambda k: match[k]["same_fraction"])
        seed_report[name] = {
            "sha256": _sha256(next(p for p in seed_paths if p.name == name)),
            "tasks_using": sum(1 for s in seed_of.values() if s == name),
            "record_counts": {t: len(r) for t, r in sorted(tables.items())},
            "tables_not_in_bench2": sorted(set(inserted) - set(model_fields)),
            "columns_not_in_bench2_model": {t: sorted(c - model_fields[t]) for t, c in inserted.items() if t in model_fields and c - model_fields[t]},
            "bench2_fields_not_seeded": {t: sorted(model_fields[t] - c) for t, c in inserted.items() if t in model_fields and model_fields[t] - c},
            "match_vs_bench2": {
                name: {k: v for k, v in m.items() if k in ("records", "same_records", "same_fraction")} for name, m in match.items()
            },
            "closest_bench2_seed": best if match[best]["same_fraction"] >= SEED_MATCH_THRESHOLD else None,
            "diff_vs_closest": match[best],
        }

    # Bench2 tasks for cross-source checks
    bench_raw = {t["id"]: t for t in load_raw_tasks()}
    manifest = load_manifest()
    bench_writes_of = {
        tid: {a["name"] for a in t["evaluation_criteria"]["actions"] if a["name"] in bench_writes} for tid, t in bench_raw.items()
    }
    bench_ids_of = {
        tid: {
            v
            for a in t["evaluation_criteria"]["actions"]
            for v in a["arguments"].values()
            if isinstance(v, str) and _RECORD_ID.fullmatch(v)
        }
        for tid, t in bench_raw.items()
    }
    bench_created = {
        tid: {rid for ops in (t.get("initial_state_delta") or {}).values() for rid, op in ops.items() if "create" in op}
        for tid, t in bench_raw.items()
    }
    bench_desc = {tid: _tokens(t["scenario"]["task_description"]) for tid, t in bench_raw.items()}

    # 3-5. per-task audit
    tasks = {}
    for tid, row in sorted(oracle.items()):
        tools = list(row["selected_tools"])
        writes = {t for t in tools if t in bench_writes}
        missing = sorted(set(tools) - bench_tools)
        connection = connections[seed_of[tid]]
        verifier_audit = audit_verifiers(row, connection, model_fields)
        db_verifiers = [v for v in verifier_audit["verifiers"] if v["type"] == "database_state"]
        mapped = seed_report[seed_of[tid]]["closest_bench2_seed"]
        upstream_ids = set(_RECORD_ID.findall(row["user_prompt"])) | {
            i for v in db_verifiers for i in _RECORD_ID.findall(v.get("query", ""))
        }
        upstream_tables = seed_tables(connection)
        seed_ids = {rid for rows in upstream_tables.values() for rid in rows}
        verifier_ids = {i for v in db_verifiers for i in _RECORD_ID.findall(v.get("query", "").upper())}
        bench_seed_ids = (
            {rid for rows in bench_seeds[mapped].values() for rid in rows} if mapped else set()
        )
        # IDs a verifier names that the upstream seed lacks: records the task must create,
        # pinned to the upstream seed's next sequential ID.
        created_ids = sorted(verifier_ids - seed_ids)
        created_id_collisions = sorted(set(created_ids) & bench_seed_ids)
        absent_ids = sorted(i for i in verifier_ids & seed_ids if mapped and i not in bench_seed_ids)
        seed_delta = seed_report[seed_of[tid]]["diff_vs_closest"]
        seed_variant = seed_delta["delta_size"] > 0

        links = []
        for bid, bw in bench_writes_of.items():
            union = writes | bw
            jaccard = len(writes & bw) / len(union) if union else 0.0
            shared_creation = sorted(writes & bw & CREATION_TOOLS)
            same_entities = []
            if mapped and bench_raw[bid]["seed_db"] == mapped:
                same_entities = sorted((upstream_ids & bench_ids_of[bid] & seed_ids) - bench_created[bid])
            prompt_j = _jaccard(_tokens(row["user_prompt"]), bench_desc[bid])
            if jaccard >= WRITE_JACCARD_THRESHOLD or shared_creation or len(same_entities) >= 2 or prompt_j >= PROMPT_JACCARD_FLAG:
                entry = manifest["tasks"][bid]
                links.append(
                    {
                        "bench2_task": bid,
                        "family": entry["family"],
                        "split": entry["split"],
                        "write_jaccard": round(jaccard, 3),
                        "shared_creation_tools": shared_creation,
                        "same_seed_entities": same_entities,
                        "prompt_description_jaccard": round(prompt_j, 3),
                        "family_link": jaccard >= WRITE_JACCARD_THRESHOLD or bool(shared_creation),
                    }
                )
        family_splits = sorted({l["split"] for l in links if l["family_link"]})
        near_dup = [
            l["bench2_task"]
            for l in links
            if (len(l["same_seed_entities"]) >= 2 and l["write_jaccard"] >= 0.5) or l["prompt_description_jaccard"] >= PROMPT_JACCARD_FLAG
        ]

        issues = []
        if missing:
            issues.append("tools_missing_in_bench2")
        if any(v["type"] != "database_state" for v in verifier_audit["verifiers"]):
            issues.append("non_db_verifier")
        if any(v.get("executes_on_seed") is False for v in db_verifiers):
            issues.append("verifier_does_not_execute")
        if any(v.get("columns_not_in_bench2_model") for v in db_verifiers):
            issues.append("verifier_reads_column_absent_in_bench2")
        if verifier_audit["duplicate_pairs"]:
            issues.append("duplicate_verifiers")
        if verifier_audit["conflicts"]:
            issues.append("conflicting_verifiers")
        if any(v.get("passes_on_initial_seed") for v in db_verifiers):
            issues.append("verifier_passes_on_initial_seed")
        if any(v.get("weak_predicates") for v in db_verifiers):
            issues.append("weak_verifier_predicate")
        if not mapped:
            issues.append("seed_not_in_bench2")
        if seed_variant:
            issues.append("seed_variant_needs_delta")
        if created_ids:
            issues.append("verifier_pins_created_record_id")
        if created_id_collisions:
            issues.append("pinned_created_id_exists_in_bench2_seed")
        if absent_ids:
            issues.append("verifier_id_absent_from_bench2_seed")
        if not writes:
            issues.append("no_write_tool")
        if len(family_splits) > 1:
            issues.append("links_families_in_several_v1_splits")
        if set(family_splits) & {"dev", "heldout"}:
            issues.append("links_dev_or_heldout_family")
        if near_dup:
            issues.append("near_duplicate_of_bench2_task")
        entity_splits = {l["split"] for l in links if len(l["same_seed_entities"]) >= 2}
        if entity_splits & {"dev", "heldout"}:
            issues.append("shares_entities_with_dev_or_heldout_task")
        if "train" in entity_splits:
            issues.append("shares_entities_with_train_task")

        blocking = set(BLOCKING_ISSUES)
        minor = set(MINOR_ISSUES)
        if set(issues) & blocking:
            category = "not_suitable_now"
        elif set(issues) & minor:
            category = "minor_adaptation"
        else:
            category = "direct"

        eval_overlap = max(
            ((len(writes & bw), bid) for bid, bw in bench_writes_of.items() if manifest["tasks"][bid]["split"] != "train"),
            default=(0, None),
        )
        tasks[tid] = {
            # Excerpt only: the upstream data is referenced, not copied into the project.
            "user_prompt_excerpt": row["user_prompt"][:200],
            "user_prompt_sha256": hashlib.sha256(row["user_prompt"].encode("utf-8")).hexdigest(),
            "seed": seed_of[tid],
            "mapped_bench2_seed": mapped,
            "acting_user": user_of[tid],
            "system_prompt_sha256": hashlib.sha256(row["system_prompt"].encode("utf-8")).hexdigest(),
            "oracle_tools": tools,
            "oracle_write_tools": sorted(writes),
            "tools_missing_in_bench2": missing,
            "verifier_created_ids": created_ids,
            "created_id_collisions_in_bench2_seed": created_id_collisions,
            "verifier_ids_absent_from_bench2_seed": absent_ids,
            "verifier_count": len(verifier_audit["verifiers"]),
            "verifier_audit": verifier_audit,
            "links_to_bench2": links,
            "linked_v1_family_splits": family_splits,
            "near_duplicate_of": near_dup,
            "max_shared_write_tools_with_dev_or_heldout_task": {"count": eval_overlap[0], "task": eval_overlap[1]},
            "write_tools_new_to_bench2_tasks": sorted(writes - set().union(*bench_writes_of.values())),
            "seed_delta_size": seed_delta["delta_size"],
            "issues": issues,
            "category": category,
        }

    pilot = {}
    for tid in PILOT_TASKS:
        task = tasks[tid]
        checks = {
            "not_blocked": task["category"] != "not_suitable_now",
            "no_dev_or_heldout_family_link": not (set(task["linked_v1_family_splits"]) & {"dev", "heldout"}),
            "shared_writes_with_dev_or_heldout_le_1": task["max_shared_write_tools_with_dev_or_heldout_task"]["count"]
            <= MAX_SHARED_WRITES_WITH_EVAL_SPLITS,
            "no_shared_entities_with_dev_or_heldout": "shares_entities_with_dev_or_heldout_task" not in task["issues"],
        }
        if not all(checks.values()):
            raise SystemExit(f"pilot task {tid} fails {checks}")
        pilot[tid] = {
            "category": task["category"],
            "seed": task["seed"],
            "mapped_bench2_seed": task["mapped_bench2_seed"],
            "seed_delta_size": task["seed_delta_size"],
            "oracle_tools": task["oracle_tools"],
            "oracle_write_tools": task["oracle_write_tools"],
            "write_tools_new_to_bench2_tasks": task["write_tools_new_to_bench2_tasks"],
            "verifier_count": task["verifier_count"],
            "issues": task["issues"],
            "linked_v1_families": sorted({l["family"] for l in task["links_to_bench2"] if l["family_link"]}),
            "checks": checks,
        }

    categories = collections.Counter(t["category"] for t in tasks.values())
    issue_counts = collections.Counter(i for t in tasks.values() for i in t["issues"])
    result = {
        "provenance": {
            "repo": UPSTREAM_REPO,
            "commit": UPSTREAM_COMMIT,
            "license": "Apache-2.0",
            "hf_dataset": HF_DATASET,
            "hf_revision": HF_REVISION,
            "parquet_sha256": {m: _sha256(p) for m, p in parquet.items()},
            "gym_dbs_zip_sha256": _sha256(ref / "repo" / "gym_dbs.zip"),
            "itsm_seed_sha256": {p.name: _sha256(p) for p in seed_paths},
            "bench2_relation": "Bench2/data/itsmbench/README.md: seed data and action space adapted from EnterpriseOps-Gym; tasks original",
        },
        "modes": {
            "same_task_ids_in_all_modes": same_ids,
            "rows_per_mode": {m: len(data[m]) for m in MODES},
            "fields_differing_from_oracle": dict(field_diffs),
            "verifier_lists_differing_only_in_order": dict(order_only),
            "verifier_multisets_equal_across_modes": verifier_multiset_equal,
            "tool_sets": extra_tools,
            "conclusion": (
                "the four modes are one task set: identical task IDs, prompts, seeds and verifier "
                "multisets; they differ only by distractor tools added to selected_tools (and verifier "
                "order). 103 independent task IDs, not 412."
            ),
        },
        "dedup": {
            "independent_task_ids": len(oracle),
            "exact_content_duplicate_groups": exact_duplicates,
            "identical_prompt_groups": [sorted(v) for v in prompt_groups.values() if len(v) > 1],
            "distinct_system_prompts": len({r["system_prompt"] for r in oracle.values()}),
        },
        "bench2_tools": {
            "upstream_oracle_tools_used": len({t for r in oracle.values() for t in r["selected_tools"]}),
            "missing_in_bench2": dict(
                collections.Counter(t for r in oracle.values() for t in r["selected_tools"] if t not in bench_tools)
            ),
            "distractor_tools_missing_in_bench2": sorted(
                {t for m in MODES for r in data[m].values() for t in r["selected_tools"]} - bench_tools
            ),
        },
        "seeds": seed_report,
        "classification_rules": {
            "blocking_not_suitable_now": BLOCKING_ISSUES,
            "minor_adaptation": MINOR_ISSUES,
            "informational": INFORMATIONAL_ISSUES,
            "direct": "no blocking or minor issue; still needs the per-task conversion every task needs (multi-turn scenario, persona/known_info, gold actions, NL assertions)",
        },
        "summary": {"categories": dict(categories), "issues": dict(issue_counts)},
        "pilot": pilot,
        "tasks": tasks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("modes", "dedup", "bench2_tools", "summary")}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
