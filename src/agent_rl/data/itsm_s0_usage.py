"""Joint SFT/GRPO data usage audit, offline only (step S0).

Nothing here samples, trains or calls an LLM, and no split file is changed: the
v1/v2 manifests, the pilot/batch/candidate files, D8, the fuzzy threshold,
policy, gold and user texts are read as they are.

Three parts:

* ``core_requirement_audit``: the core requirements of the seven accepted
  derived train tasks (6B: 3, 6D-2: 4), each with targeted negatives. The
  existing validation cases are re-run; the key-fact negatives added here
  (6D-3-R1: "everything else correct, only a key fact wrong") cover the core
  requirements carried by fuzzy-compared free text. GRPO eligibility needs a
  DB = 0 negative for every core requirement; SFT eligibility is judged
  separately (a fuzzy blind spot does not make a correct demonstration wrong).
* ``joint_isolation_audit``: the training union SFT u GRPO against dev and
  held-out (family, shared write tools, gold entities, near duplicates, upstream
  source).
* ``source_registry``: every task or upstream candidate seen so far, with its
  proposed GRPO/SFT use or the reason it stays excluded.
"""

from __future__ import annotations

import collections
import itertools
import json
from pathlib import Path
from typing import Any

from enterprise_worlds.data_model.tasks import Task
from enterprise_worlds.domains.itsm.data_model import _FREETEXT_FIELDS
from enterprise_worlds.evaluator.text_match_strategy import TextMatchConfig
from enterprise_worlds.utils.text_match import DB_FUZZY_THRESHOLD, text_overlap

from agent_rl.data.itsm_eops_batch2 import DEFAULT_BATCH2_PATH
from agent_rl.data.itsm_eops_heldout_v3 import ACCEPTED_V3_TRAIN_TASK_IDS, DEFAULT_HELDOUT_V3_PATH
from agent_rl.data.itsm_eops_pilot import (
    DEFAULT_PILOT_PATH,
    _delta_created,
    _gold_ids,
    _seed_ids,
    _write_set,
    edited_actions,
    load_pilot,
    run_actions,
)
from agent_rl.data.itsm_split import CREATION_TOOLS, WRITE_JACCARD_THRESHOLD, load_raw_tasks, write_tool_names
from agent_rl.data.itsm_split_v2 import ACCEPTED_DERIVED_TASK_IDS, DEFAULT_V2_PATH, derived_raw_tasks
from agent_rl.envs.itsm_scoring import build_task_environment, replay_gold_actions, score_final_db, task_clock

S0_VERSION = "itsm_usage_s0_candidate_v1"
REPO_ROOT = Path(__file__).resolve().parents[3]
SCREENING_PATH = REPO_ROOT / "docs" / "itsm_step6d" / "candidate_screening.json"
FUZZY = TextMatchConfig(strategy="fuzzy")
DERIVED_TRAIN_TASK_IDS = tuple(sorted((*ACCEPTED_DERIVED_TASK_IDS, *ACCEPTED_V3_TRAIN_TASK_IDS)))
#: Isolated in 6D-2 only because its core requirement is invisible to the DB
#: reward; no ambiguity, missing tool or policy conflict. Proposed as an SFT-only
#: source; admitting it needs the user's approval.
SFT_ONLY_CANDIDATE_IDS = ("itsmbench_eops_78a5b71e_a4fb07a9",)
#: 6D-3 passing held-out candidate (6D-3 not yet accepted): reserved for evaluation either way.
HELDOUT_CANDIDATE_IDS = ("itsmbench_eops_99ba2325_54f35603",)

_CAPTION = "Status update on the SAP database outage (INC-000001): the team is still working on it and resolution progress is being tracked."
_REMINDER = "Reminder: please keep the status of the SAP database outage (INC-000001) current and report resolution progress."

#: Negatives added in S0. ``kind`` "key_fact": the rest of the text is the gold
#: text, only a key fact changes. ``kind`` "structured": a non-text field.
#: ``value`` may be a ``(old, new)`` pair applied to the gold text.
S0_CASES: dict[str, list[dict[str, Any]]] = {
    "itsmbench_eops_46ca2862_b6ef60ca": [
        {"name": "s0_subject_says_hardware", "kind": "key_fact", "index": 5, "field": "short_description", "value": "Hardware installation", "reason": "incident subject names hardware instead of software installation"},
        {"name": "s0_category_hardware", "kind": "structured", "index": 5, "field": "category", "value": "hardware", "reason": "incident category hardware instead of software"},
    ],
    "itsmbench_eops_66392a82_52c8290e": [
        {"name": "s0_symptom_replaced", "kind": "key_fact", "index": 5, "field": "short_description", "value": "Video conferencing hub overheating and rebooting", "reason": "incident describes overheating/rebooting, not freezing and dropped calls"},
        {"name": "s0_symptom_unrelated", "kind": "key_fact", "index": 5, "field": "short_description", "value": "Video conferencing hub needs a firmware update", "reason": "incident describes no fault at all"},
    ],
    "itsmbench_eops_1d219d93_d1ba6c2d": [
        {"name": "s0_location_wrong_city", "kind": "structured", "index": 2, "field": "city", "value": "Mumbai", "reason": "Pune HQ created in the wrong city"},
        {"name": "s0_kb_wrong_serial", "kind": "key_fact", "index": 9, "field": "body", "value": "SN-MER-888, SN-DELL-998", "reason": "article lists a wrong serial for the Dell"},
        {"name": "s0_kb_missing_serial", "kind": "key_fact", "index": 9, "field": "body", "value": "SN-MER-888", "reason": "article omits the Dell serial"},
    ],
    "itsmbench_eops_01395ded_eff85dae": [
        {"name": "s0_update_says_work_finished", "kind": "key_fact", "index": 8, "field": "message", "value": (_CAPTION, _CAPTION.replace("the team is still working on it and resolution progress is being tracked", "the team has finished working on it and resolution progress is no longer being tracked")), "reason": "caller told the work is finished, the opposite of the operator's text"},
        {"name": "s0_update_wrong_incident_number", "kind": "key_fact", "index": 8, "field": "message", "value": (_CAPTION, _CAPTION.replace("INC-000001", "INC-000002")), "reason": "caller update cites another incident number"},
        {"name": "s0_reminder_wrong_service", "kind": "key_fact", "index": 9, "field": "message", "value": (_REMINDER, _REMINDER.replace("SAP database outage", "email service outage")), "reason": "reminder names the wrong outage"},
    ],
    "itsmbench_eops_99ba2325_7915481a": [
        {"name": "s0_kb_old_serial", "kind": "key_fact", "index": 5, "field": "body", "value": ("DB-2025-PROD-001", "SRV-ORC-9981"), "reason": "article records the replaced serial instead of the new one"},
        {"name": "s0_kb_wrong_criticality", "kind": "key_fact", "index": 5, "field": "body", "value": ("is a business-critical service", "is a low-priority internal service"), "reason": "article misstates the service's criticality"},
    ],
}

#: Core requirements per derived train task -> negatives (existing validation
#: cases or S0 cases). ``carrier``: "structured", "free_text" or
#: "free_text+structured" (the same fact is also held by a structured field).
CORE_REQUIREMENTS: dict[str, list[dict[str, Any]]] = {
    "itsmbench_eops_46ca2862_b6ef60ca": [
        {"id": "laptop_registered_as_specified", "carrier": "structured", "negatives": ["laptop_registered_in_london"]},
        {"id": "incident_fields_and_priority", "carrier": "structured", "negatives": ["wrong_priority"]},
        {"id": "incident_about_software_installation", "carrier": "free_text+structured", "negatives": ["s0_category_hardware", "s0_subject_says_hardware"]},
        {"id": "required_sla_record", "carrier": "structured", "negatives": ["sla_not_spawned"]},
        {"id": "declined_notification_not_sent", "carrier": "structured", "negatives": ["unconfirmed_report_notification"]},
    ],
    "itsmbench_eops_66392a82_52c8290e": [
        {"id": "cost_corrected", "carrier": "structured", "negatives": ["cost_not_corrected"]},
        {"id": "caller_is_elena", "carrier": "structured", "negatives": ["operator_as_caller"]},
        {"id": "sla_matches_critical", "carrier": "structured", "negatives": ["sla_for_wrong_priority"]},
        {"id": "incident_describes_freezing_and_dropped_calls", "carrier": "free_text", "negatives": ["s0_symptom_replaced", "s0_symptom_unrelated"]},
    ],
    "itsmbench_eops_1d219d93_d1ba6c2d": [
        {"id": "pune_hq_created", "carrier": "structured", "negatives": ["s0_location_wrong_city"]},
        {"id": "only_in_stock_items_moved", "carrier": "structured", "negatives": ["in_stock_item_not_moved", "in_use_item_also_moved"]},
        {"id": "new_assets_at_pune", "carrier": "structured", "negatives": ["dell_left_in_bangalore"]},
        {"id": "elizabeth_promoted_not_relocated", "carrier": "structured", "negatives": ["role_not_changed", "elizabeth_relocated"]},
        {"id": "kb_lists_only_the_two_serials", "carrier": "free_text", "negatives": ["s0_kb_wrong_serial", "s0_kb_missing_serial"]},
    ],
    "itsmbench_eops_99ba2325_887f5580": [
        {"id": "spare_ci_retired", "carrier": "structured", "negatives": ["ci_set_to_disposed"]},
        {"id": "printer_linked_as_affected_ci", "carrier": "structured", "negatives": ["printer_not_linked", "printer_set_as_primary_ci"]},
        {"id": "retired_ci_not_linked", "carrier": "structured", "negatives": ["retired_ci_also_linked"]},
    ],
    "itsmbench_eops_01395ded_eff85dae": [
        {"id": "only_queued_notification_failed", "carrier": "structured", "negatives": ["queued_left_queued", "delivered_also_failed"]},
        {"id": "recipients_and_types", "carrier": "structured", "negatives": ["caller_sent_reminder", "group_manager_skipped"]},
        {"id": "operator_texts_status_and_progress", "carrier": "free_text", "negatives": ["s0_update_says_work_finished", "s0_update_wrong_incident_number", "s0_reminder_wrong_service"]},
    ],
    "itsmbench_eops_46ca2862_56a4f834": [
        {"id": "offerings_under_parent_with_inherited_attributes", "carrier": "structured", "negatives": ["offering_under_wrong_service", "criticality_not_inherited", "one_offering_missing"]},
        {"id": "template_uses_mobile_offering", "carrier": "structured", "negatives": ["template_on_archiving_offering"]},
    ],
    "itsmbench_eops_99ba2325_7915481a": [
        {"id": "serial_updated_on_existing_ci", "carrier": "structured", "negatives": ["new_ci_registered", "serial_set_on_wrong_ci"]},
        {"id": "service_operational", "carrier": "structured", "negatives": ["service_left_in_repair"]},
        {"id": "article_published_external", "carrier": "structured", "negatives": ["article_left_draft_internal"]},
        {"id": "article_body_operator_facts", "carrier": "free_text", "negatives": ["s0_kb_old_serial", "s0_kb_wrong_criticality"]},
    ],
}

#: SFT acceptance checks that the DB reward cannot make: exact facts a correct
#: demonstration must carry (checked on the trajectory's final DB, then read by a human).
SFT_FACT_CHECKS: dict[str, list[str]] = {
    "itsmbench_eops_46ca2862_b6ef60ca": ["incident short_description names software installation"],
    "itsmbench_eops_66392a82_52c8290e": ["incident short_description or description states freezing and dropped calls"],
    "itsmbench_eops_1d219d93_d1ba6c2d": ["article body contains SN-MER-888 and SN-DELL-999 and no other serial"],
    "itsmbench_eops_01395ded_eff85dae": ["each notification subject/message equals the operator's text (whitespace-normalised)"],
    "itsmbench_eops_99ba2325_7915481a": ["article body equals the operator's body (whitespace-normalised); it carries DB-2025-PROD-001"],
    "itsmbench_eops_99ba2325_887f5580": [],
    "itsmbench_eops_46ca2862_56a4f834": [],
    "itsmbench_eops_78a5b71e_a4fb07a9": ["change short_description carries the operator's text naming SRV-LNX-4421"],
}

#: Proposed S1 pilot (4-6 sources, at most 2 candidate trajectories each). The
#: optional entry needs the SFT-only admission decision first.
SFT_PILOT_CANDIDATES: tuple[dict[str, Any], ...] = (
    {"task_id": "itsmbench_msp_018", "optional": False, "why": "Bench2 original, MSP (provider + client), change and problem lifecycle steps with a KB link; no free text"},
    {"task_id": "itsmbench_single_011", "optional": False, "why": "Bench2 original, single tenant, problem -> change mapping with a new KB article; writes free text not yet key-fact audited, so the fact checks are exercised"},
    {"task_id": "itsmbench_eops_99ba2325_887f5580", "optional": False, "why": "derived, use both; the correct behaviour includes refusing to link a retired CI (§5.1)"},
    {"task_id": "itsmbench_eops_01395ded_eff85dae", "optional": False, "why": "derived, SFT-only after R1; operator-dictated texts must pass through the simulator verbatim; inactive caller decision"},
    {"task_id": "itsmbench_eops_66392a82_52c8290e", "optional": False, "why": "derived, SFT-only after R1; single tenant; REQUIRED SLA and knowledge search cascades; declined PROPOSED notification"},
    {"task_id": "itsmbench_eops_78a5b71e_a4fb07a9", "optional": True, "why": "SFT-only candidate (needs admission); change lifecycle with an operator text naming the extra CI"},
)
MAX_PILOT_TRAJECTORIES_PER_TASK = 2


# -- loading ---------------------------------------------------------------------------------------


def _batch2() -> dict[str, Any]:
    return load_pilot(DEFAULT_BATCH2_PATH)


def derived_entries() -> dict[str, dict[str, Any]]:
    """Raw entries (with provenance and validation cases) of the 7 derived train tasks."""
    entries = {e["id"]: e for e in load_pilot(DEFAULT_PILOT_PATH)["tasks"] + _batch2()["tasks"]}
    missing = sorted(set(DERIVED_TRAIN_TASK_IDS) - set(entries))
    if missing:
        raise ValueError(f"derived train tasks missing: {missing}")
    return {tid: entries[tid] for tid in DERIVED_TRAIN_TASK_IDS}


def sft_only_candidate_entries() -> dict[str, dict[str, Any]]:
    conversions = {
        i["attempted_conversion"]["id"]: i["attempted_conversion"]
        for i in _batch2()["isolated"]
        if "attempted_conversion" in i
    }
    return {tid: conversions[tid] for tid in SFT_ONLY_CANDIDATE_IDS}


def heldout_candidate_entries() -> dict[str, dict[str, Any]]:
    tasks = {t["id"]: t for t in load_pilot(DEFAULT_HELDOUT_V3_PATH)["tasks"]}
    return {tid: tasks[tid] for tid in HELDOUT_CANDIDATE_IDS}


# -- core requirements and key-fact negatives (6D-3-R1) -----------------------------------------


def _s0_edit(task: Task, case: dict[str, Any]) -> tuple[dict[str, Any], str | None, str | None]:
    gold = task.evaluation_criteria.actions[case["index"]].arguments.get(case["field"])
    value = case["value"]
    if isinstance(value, tuple | list):
        old, new = value
        if old not in gold:
            raise ValueError(f"{task.id} {case['name']}: {old!r} not in the gold text")
        value = gold.replace(old, new)
    return {"set": {str(case["index"]): {case["field"]: value}}}, gold, value


def run_negative(task: Task, edit: dict[str, Any]) -> dict[str, Any]:
    environment = run_actions(task, edited_actions(task, edit))
    score = score_final_db(task, environment, text_match=FUZZY)
    return {"db_reward": score.db_reward, "mismatches": list(score.mismatches)}


def free_text_exposure(task: Task) -> list[str]:
    """Fuzzy-compared fields (``_FREETEXT_FIELDS``) that the gold sets to non-empty text."""
    with task_clock(task):
        initial = build_task_environment(task).tools.db.model_dump()
    final = replay_gold_actions(task).tools.db.model_dump()
    exposed = []
    for collection, fields in sorted(_FREETEXT_FIELDS.items()):
        before = initial.get(collection, {})
        for record_id, record in sorted(final.get(collection, {}).items()):
            for field in fields:
                value = record.get(field)
                if value and value != before.get(record_id, {}).get(field):
                    exposed.append(f"{collection}/{record_id} {field}")
    return exposed


def core_requirement_audit() -> dict[str, Any]:
    entries = derived_entries()
    result = {}
    for task_id, entry in entries.items():
        task = Task.model_validate(entry)
        cases: dict[str, dict[str, Any]] = {}
        for case in entry["validation_cases"]:
            cases[case["name"]] = {"source": "existing_validation_case", "kind": case.get("kind", "structured"), **run_negative(task, case["edit"])}
        for case in S0_CASES.get(task_id, []):
            edit, gold, value = _s0_edit(task, case)
            record = {"source": "s0", "kind": case["kind"], "reason": case["reason"], **run_negative(task, edit)}
            if case["kind"] == "key_fact":
                record["fuzzy_recall"] = round(text_overlap(gold, value), 3)
                record["threshold"] = DB_FUZZY_THRESHOLD
            cases[case["name"]] = record
        requirements = []
        for requirement in CORE_REQUIREMENTS[task_id]:
            rewards = {name: cases[name]["db_reward"] for name in requirement["negatives"]}
            structured = [n for n in requirement["negatives"] if cases[n]["kind"] != "key_fact"]
            key_facts = [n for n in requirement["negatives"] if cases[n]["kind"] == "key_fact"]
            requirements.append(
                {
                    **requirement,
                    "negative_db_rewards": rewards,
                    # Discriminated when a structured negative carrying the fact fails, or,
                    # for text-only requirements, every key-fact negative fails.
                    "discriminated": (
                        any(rewards[n] == 0.0 for n in structured)
                        if structured
                        else all(rewards[n] == 0.0 for n in key_facts)
                    ),
                    "free_text_key_fact_missed": [n for n in key_facts if rewards[n] == 1.0],
                }
            )
        grpo_ok = all(r["discriminated"] for r in requirements)
        result[task_id] = {
            "source_file": "pilot_v1" if task_id in ACCEPTED_DERIVED_TASK_IDS else "batch2_v1",
            "free_text_written_by_gold": free_text_exposure(task),
            "core_requirements": requirements,
            "negatives": cases,
            "grpo": {
                "proposed": "eligible" if grpo_ok else "suspended_reward_blind_spot",
                "basis": (
                    "every core requirement has a DB=0 negative"
                    if grpo_ok
                    else "a core requirement carried only by fuzzy-compared free text is not discriminated: "
                    + ", ".join(r["id"] for r in requirements if not r["discriminated"])
                ),
            },
            "sft": {
                "proposed": "eligible_pending_trajectory_review",
                "basis": (
                    "unambiguous user goal, tools exist, no policy conflict, gold DB=1 and valid upstream "
                    "verifiers pass; the reward blind spot is covered by explicit fact checks in review"
                ),
                "fact_checks": SFT_FACT_CHECKS[task_id],
            },
        }
    return result


# -- joint isolation: SFT u GRPO against dev / held-out -----------------------------------------


def _upstream_id(entry: dict[str, Any]) -> str | None:
    return (entry.get("source") or {}).get("upstream_task_id")


def joint_isolation_audit() -> dict[str, Any]:
    writes = write_tool_names()
    manifest = json.loads(DEFAULT_V2_PATH.read_text(encoding="utf-8"))
    raw = {t["id"]: t for t in load_raw_tasks()}
    raw.update(derived_raw_tasks())
    use = {tid: manifest["tasks"][tid]["split"] for tid in raw}
    raw.update({tid: e for tid, e in derived_entries().items()})
    raw.update(sft_only_candidate_entries())
    raw.update(heldout_candidate_entries())
    for tid in ACCEPTED_V3_TRAIN_TASK_IDS + SFT_ONLY_CANDIDATE_IDS:
        use[tid] = "train"
    for tid in HELDOUT_CANDIDATE_IDS:
        use[tid] = "heldout"
    origin = {
        tid: "bench2" if tid.startswith("itsmbench_msp_") or tid.startswith("itsmbench_single_") else "derived"
        for tid in raw
    }
    training = sorted(t for t in raw if use[t] == "train")
    evaluation = sorted(t for t in raw if use[t] in ("dev", "heldout"))

    sets = {tid: _write_set(t, writes) for tid, t in raw.items()}
    seed_ids = {seed: _seed_ids(seed) for seed in {t["seed_db"] for t in raw.values()}}
    ents = {tid: (_gold_ids(t) & seed_ids[t["seed_db"]]) - _delta_created(t) for tid, t in raw.items()}

    parent = {tid: tid for tid in raw}

    def root(tid: str) -> str:
        while parent[tid] != tid:
            parent[tid] = parent[parent[tid]]
            tid = parent[tid]
        return tid

    for a, b in itertools.combinations(sorted(raw), 2):
        union = sets[a] | sets[b]
        jaccard = len(sets[a] & sets[b]) / len(union) if union else 0.0
        if jaccard >= WRITE_JACCARD_THRESHOLD or sets[a] & sets[b] & CREATION_TOOLS:
            parent[root(a)] = root(b)
    components = collections.defaultdict(list)
    for tid in sorted(raw):
        components[root(tid)].append(tid)
    straddling = [
        {"members": len(c), "uses": sorted({use[t] for t in c})}
        for c in components.values()
        if {use[t] for t in c} & {"train"} and {use[t] for t in c} & {"dev", "heldout"}
    ]

    pairs = []
    for t, e in itertools.product(training, evaluation):
        shared_writes = sorted(sets[t] & sets[e])
        shared_entities = sorted(ents[t] & ents[e]) if raw[t]["seed_db"] == raw[e]["seed_db"] else []
        near_duplicate = len(shared_entities) >= 2 and sets[t] == sets[e]
        if len(shared_writes) >= 2 or len(shared_entities) >= 2 or near_duplicate:
            pairs.append(
                {
                    "train": t,
                    "train_origin": origin[t],
                    "eval": e,
                    "eval_use": use[e],
                    "shared_write_tools": shared_writes,
                    "shared_gold_entities": shared_entities,
                    "near_duplicate": near_duplicate,
                }
            )
    new_entrants = {t for t in training if origin[t] == "derived"}
    upstream = {tid: _upstream_id(raw[tid]) for tid in raw if _upstream_id(raw[tid])}
    upstream_overlap = sorted(
        {upstream[t] for t in training if t in upstream} & {upstream[e] for e in evaluation if e in upstream}
    )
    annotators = collections.defaultdict(lambda: collections.defaultdict(list))
    for tid, up in upstream.items():
        annotators[up.split("_")[-2]][use[tid]].append(tid)
    return {
        "training_union": {"tasks": len(training), "bench2": sum(origin[t] == "bench2" for t in training), "derived": len(new_entrants)},
        "evaluation": {"dev": sum(use[e] == "dev" for e in evaluation), "heldout": sum(use[e] == "heldout" for e in evaluation)},
        "families_straddling_training_and_evaluation": straddling,
        "pairs_flagged": pairs,
        "checks": {
            "no_family_spans_training_and_evaluation": not straddling,
            "derived_training_shares_le_1_write_tool_with_eval": not [p for p in pairs if p["train"] in new_entrants and len(p["shared_write_tools"]) >= 2],
            "derived_training_shares_lt_2_entities_with_eval": not [p for p in pairs if p["train"] in new_entrants and len(p["shared_gold_entities"]) >= 2],
            "no_near_duplicate_across_training_and_evaluation": not [p for p in pairs if p["near_duplicate"]],
            "no_upstream_task_in_both": not upstream_overlap,
        },
        "bench2_cross_split_pairs_known_since_v1": len([p for p in pairs if p["train_origin"] == "bench2"]),
        "upstream_annotators_shared_across_uses": {
            a: {u: sorted(ids) for u, ids in by_use.items()}
            for a, by_use in sorted(annotators.items())
            if "train" in by_use and ({"dev", "heldout"} & set(by_use))
        },
    }


# -- source registry -------------------------------------------------------------------------------

_BARRED_REASON_CLASS = {
    "ambiguous REQUIRED cascade": "required_cascade_ambiguous_or_sla_rule_undecided",
    "upstream prompt contradicts its verifier": "prompt_verifier_conflict",
    "upstream expected state conflicts with a REQUIRED Bench2 rule": "policy_conflict",
    "REQUIRED cascade impossible with Bench2 tools, plus ambiguous": "missing_tool",
    "shares 2 same-seed entities with a held-out task": "heldout_entity_leak",
}


def source_registry(core: dict[str, Any]) -> dict[str, Any]:
    manifest = json.loads(DEFAULT_V2_PATH.read_text(encoding="utf-8"))
    rows = []
    for tid, entry in sorted(manifest["tasks"].items()):
        split = entry["split"]
        if split != "train":
            rows.append({"id": tid, "origin": entry["source"], "use": split, "grpo": "no", "sft": "no", "basis": f"v2 {split}"})
        elif tid in core:
            rows.append({"id": tid, "origin": "derived_6B", "use": None, "grpo": core[tid]["grpo"]["proposed"], "sft": core[tid]["sft"]["proposed"], "basis": core[tid]["grpo"]["basis"]})
        else:
            rows.append({"id": tid, "origin": "bench2", "use": None, "grpo": "eligible_as_accepted_in_steps_3_and_6", "sft": "eligible_pending_trajectory_review", "basis": "v1/v2 train; free-text key facts not audited (open item)"})
    for tid in ACCEPTED_V3_TRAIN_TASK_IDS:
        rows.append({"id": tid, "origin": "derived_6D-2", "use": None, "grpo": core[tid]["grpo"]["proposed"], "sft": core[tid]["sft"]["proposed"], "basis": core[tid]["grpo"]["basis"]})
    rows.append({"id": SFT_ONLY_CANDIDATE_IDS[0], "origin": "derived_6D-2_isolated", "use": None, "grpo": "no", "sft": "candidate_needs_approval", "basis": "isolated for GRPO only: core requirement invisible to the DB reward (D8 skips description); no ambiguity, tool gap or policy conflict"})
    rows.append({"id": HELDOUT_CANDIDATE_IDS[0], "origin": "derived_6D-3", "use": "heldout_candidate", "grpo": "no", "sft": "no", "basis": "6D-3 passing held-out candidate (6D-3 not yet accepted)"})
    for row in rows:
        if row["use"] is None and row["sft"] == "candidate_needs_approval":
            row["use"] = "sft_only_candidate"
        elif row["use"] is None:
            row["use"] = {
                (True, True): "both",
                (True, False): "grpo_only",
                (False, True): "sft_only",
                (False, False): "quarantined",
            }[(row["grpo"].startswith("eligible"), row["sft"].startswith("eligible"))]

    excluded = []
    for item in load_pilot(DEFAULT_PILOT_PATH)["isolated"]:
        reason = next((v for k, v in _BARRED_REASON_CLASS.items() if item["reason"].startswith(k)), item["reason"])
        excluded.append({"upstream": item["upstream_task_id"][-17:], "from": "6B", "reason": reason, "sft": "no", "grpo": "no"})
    for item in _batch2()["isolated"]:
        if item["upstream_task_id"].endswith(SFT_ONLY_CANDIDATE_IDS[0][-17:]):
            continue
        reason = next((v for k, v in _BARRED_REASON_CLASS.items() if item["reason"].startswith(k)), item["reason"])
        excluded.append({"upstream": item["upstream_task_id"][-17:], "from": "6D-2", "reason": reason, "sft": "no", "grpo": "no"})
    for item in load_pilot(DEFAULT_HELDOUT_V3_PATH)["isolated"]:
        excluded.append({"upstream": item["upstream_task_id"][-17:], "from": "6D-3", "reason": "heldout_family_candidate: " + item["reason"], "sft": "no", "grpo": "no"})

    screening = json.loads(SCREENING_PATH.read_text(encoding="utf-8"))["candidates"]
    processed = {r["upstream"] for r in excluded} | {
        e["source"]["upstream_task_id"][-17:]
        for e in load_pilot(DEFAULT_PILOT_PATH)["tasks"] + _batch2()["tasks"] + load_pilot(DEFAULT_HELDOUT_V3_PATH)["tasks"]
    } | {SFT_ONLY_CANDIDATE_IDS[0][-17:]}
    screening_rows = collections.Counter()
    deferred = []
    for candidate in sorted(screening.values(), key=lambda c: c["suffix"]):
        suffix = candidate["suffix"]
        if suffix in processed:
            continue
        if {"dev", "heldout"} & set(candidate["linked_splits"]) or candidate["train_gate_failures"]:
            screening_rows["failed_train_gate_or_links_eval_family"] += 1
            continue
        verdict = (candidate["manual"] or {}).get("verdict")
        screening_rows[f"manual_{verdict}"] += 1
        if verdict in ("deferred", "reserve", "selected"):
            deferred.append({"upstream": suffix, "manual": verdict, "reason": candidate["manual"].get("reason")})

    counts = collections.Counter(r["use"] for r in rows)
    return {
        "tasks": rows,
        "use_counts": dict(sorted(counts.items())),
        "pools": {
            "grpo": sorted(r["id"] for r in rows if r["use"] in ("both", "grpo_only")),
            "sft": sorted(r["id"] for r in rows if r["use"] in ("both", "sft_only")),
            "sft_only_candidates": sorted(r["id"] for r in rows if r["use"] == "sft_only_candidate"),
        },
        "excluded_converted_or_reviewed": excluded,
        "screening_6D-1_unprocessed": dict(sorted(screening_rows.items())),
        "deferred_by_user_not_converted": deferred,
    }


def bench2_train_free_text_exposure() -> dict[str, list[str]]:
    """Fuzzy-compared fields each Bench2 v2-train task's gold writes. These tasks
    had no key-fact negatives (open item); listing them is not a reclassification."""
    from agent_rl.data.itsm_task_sets import load_task_set, task_ids

    tasks = load_task_set(split_version="itsm_split_v2")
    return {
        tid: free_text_exposure(tasks[tid])
        for tid in task_ids("train", split_version="itsm_split_v2")
        if tid.startswith(("itsmbench_msp_", "itsmbench_single_"))
    }


def build_s0_audit() -> dict[str, Any]:
    core = core_requirement_audit()
    registry = source_registry(core)
    exposure = bench2_train_free_text_exposure()
    return {
        "version": S0_VERSION,
        "status": "candidate for S0 acceptance; v1/v2 manifests unchanged; no usage version frozen",
        "scope": (
            "offline: gold replays and edited replays on real Bench2 tools under the task clock; "
            "no tokenizer, model, LLM or GPU; held-out only read for the isolation audit"
        ),
        "text_match": {"strategy": "fuzzy", "threshold": DB_FUZZY_THRESHOLD, "d8": "unchanged"},
        "core_requirements": core,
        "joint_isolation": joint_isolation_audit(),
        "registry": registry,
        "sft_pilot_candidates": [
            {
                **c,
                "max_candidate_trajectories": MAX_PILOT_TRAJECTORIES_PER_TASK,
                "use": next(r["use"] for r in registry["tasks"] if r["id"] == c["task_id"]),
                "fact_checks": SFT_FACT_CHECKS.get(c["task_id"], [])
                or (["no free text written by gold"] if not exposure.get(c["task_id"]) else ["free-text facts read against the operator's known_info: " + ", ".join(exposure[c["task_id"]])]),
            }
            for c in SFT_PILOT_CANDIDATES
        ],
        "bench2_train_free_text_exposure": {
            "tasks_writing_free_text": sum(bool(v) for v in exposure.values()),
            "tasks": len(exposure),
            "by_task": exposure,
        },
    }
