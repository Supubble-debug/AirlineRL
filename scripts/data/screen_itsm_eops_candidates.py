"""Screen the remaining EnterpriseOps-Gym ITSM tasks for the next batch (step 6D-1).

Usage (from the repo root):
    python scripts/data/screen_itsm_eops_candidates.py --tokenizer-dir DIR [--out FILE] [--check]

Reads only committed inputs: the 6A audit (docs/itsm_step6a/eopsgym_itsm_audit.json),
split v2, the Bench2 seeds and policy-relevant seed records. No task is converted,
no split or policy is changed, and nothing runs a model. DIR holds the pinned
Qwen3-8B tokenizer (no weights); it is used only to measure the largest unfiltered
read observation under each proposed org scope.

Three layers, kept apart in the output:
  1. ``automatic``: v2 family links, shared write tools and entities per split, and
     conservative policy flags derived from verifiers and seed data;
  2. ``manual``: the reviewer's verdict for every candidate that passes the train
     gate, with the reason (hand-written below, not computed);
  3. ``batch``: the proposed 6D-2 batch, each task with its route (faithful
     migration vs. Bench2-adapted derived task), scope, planned writes, policy
     cascades, ambiguity handling, negative cases and acceptance method.
The v3 section only counts what a re-split could use; it builds no manifest.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import inspect
import itertools
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from enterprise_worlds.domains.itsm import environment as itsm_environment
from enterprise_worlds.environment.environment import _stringify

from agent_rl.data.itsm_eops_pilot import DEFAULT_PILOT_PATH, load_pilot, pilot_tasks
from agent_rl.data.itsm_split import (
    CREATION_TOOLS,
    DEFAULT_MANIFEST_PATH,
    WRITE_JACCARD_THRESHOLD,
    load_raw_tasks,
    task_features,
    write_tool_names,
)
from agent_rl.data.itsm_split_v2 import DEFAULT_V2_PATH, derived_raw_tasks

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = REPO_ROOT / "docs" / "itsm_step6a" / "eopsgym_itsm_audit.json"
TOKENS_V2_PATH = REPO_ROOT / "docs" / "itsm_step6c" / "token_measurements_v2.json"
DEFAULT_OUT = REPO_ROOT / "docs" / "itsm_step6d" / "candidate_screening.json"
TOKENIZER_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
MAX_SHARED_WRITES = 1  # 6A train gate, reused unchanged
MAX_RESPONSE_LENGTH = 12288  # configs/itsm/budget_v1.yaml
FAITHFUL_PREFIX = "itsmbench_eops_"
ADAPTED_PREFIX = "itsmbench_eopsadapt_"
MSP_MAIN_DELTA_SOURCE = "itsmbench_eops_1d219d93_d1ba6c2d"  # its delta restores the upstream MSP main seed

PRIORITY_MATRIX = {
    ("low", "low"): "planning", ("low", "medium"): "low", ("low", "high"): "moderate",
    ("medium", "low"): "low", ("medium", "medium"): "moderate", ("medium", "high"): "high",
    ("high", "low"): "moderate", ("high", "medium"): "high", ("high", "high"): "critical",
}
_RECORD_ID = re.compile(r"\b[A-Z]{2,7}_\d{3,}\b")
_EQ = re.compile(r"""(?:\b\w+\.)?\b(\w+)\s*=\s*(?:'([^']*)'|"([^"]*)"|\b(true|false|0|1)\b)""", re.IGNORECASE)
_FROM = re.compile(r"\bFROM\s+\"?(\w+)", re.IGNORECASE)


# -- manual review (hand-written; the screening never computes these) --------------------------------

#: Proposed 6D-2 batch. ``route``: ``faithful`` keeps the 6B standard (Bench2 DB=1 on
#: gold and every upstream verifier passing on gold, failing on the initial state);
#: ``adapted`` is a Bench2-adapted derived task with a new ID that is not required to
#: pass the upstream verifiers and is never presented as equivalent to them.
BATCH: dict[str, dict[str, Any]] = {
    "99ba2325_887f5580": {
        "route": "faithful",
        "workflow": "retire a spare CI, link the active printer to an open incident, refuse to link the retired CI",
        "scope": {"seed_db": "msp_db.json", "org_ids": ["ORG_001"], "tenancy": "msp_provider_internal"},
        "delta": "msp_main",
        "planned_write_tools": ["update_configuration_item", "link_affected_ci_to_incident"],
        "estimated_tool_calls": 6,
        "policy": [
            "§5.1: CIs in state retired/disposed may not be linked to active incidents; the gold links only CI_096 and the agent must decline the CI_041 link with the reason (upstream verifier 3 expects that link to be absent).",
            "§5.1 retire cascades (PROPOSED): no active incident references CI_041 (checked), so none apply.",
            "in_stock -> retired skips states; §5.1 allows skipped steps depending on the procurement path.",
        ],
        "ambiguity": "none blocking; 'MacBook Kernel Panic' resolves to INC_026 only.",
        "conversion": "multi-turn rewrite only; the simulator accepts the agent's refusal of the retired-CI link and does not insist.",
        "negatives": ["retired_ci_also_linked", "ci_set_to_disposed", "printer_not_linked", "printer_set_as_primary_ci_instead_of_affected"],
        "risk": "low",
    },
    "c5ca08eb_67009578": {
        "route": "faithful",
        "workflow": "escalate a problem's impact and propagate it to the linked incident with matrix-derived priorities",
        "scope": {"seed_db": "msp_db.json", "org_ids": ["ORG_001", "ORG_006", "ORG_013"], "tenancy": "msp_multi_client"},
        "delta": "audit_seed_variant",
        "planned_write_tools": ["update_problem", "update_incident"],
        "estimated_tool_calls": 5,
        "policy": [
            "§3.1/§4.3 priority matrix: tools do not derive priority, so the gold sets problem high x high = critical and incident high x medium = high explicitly.",
            "§6 priority change -> reset/swap/cancel the incident_sla row: INC_061 has no incident_sla row and ORG_013 has no SLA definition, so no SLA action exists; recorded as a gold note.",
            "The only linked incident is PRB_009.original_task = INC_061 (no other incident references PRB_009).",
        ],
        "ambiguity": "none blocking. Scope is nonstandard: PRB_009 lives in ORG_006 and INC_061 in ORG_013, so a provider/client pair would drop one of them.",
        "conversion": "multi-turn rewrite only.",
        "negatives": ["problem_priority_not_recomputed", "incident_untouched", "incident_priority_critical", "sla_linked_from_provider_definition"],
        "risk": "low",
    },
    "01395ded_eff85dae": {
        "route": "faithful",
        "workflow": "repair a stuck notification queue and send role-specific status notifications",
        "scope": {"seed_db": "msp_db.json", "org_ids": ["ORG_001"], "tenancy": "msp_provider_internal"},
        "delta": "msp_main",
        "planned_write_tools": ["update_notification", "send_notification"],
        "estimated_tool_calls": 9,
        "policy": [
            "§1.2 ASK: message wording is the operator's call, so the operator supplies each subject/message; every text contains 'status' and 'progress' (upstream LIKE predicates).",
            "Recipients come from data: caller USER_001 (type update), assignee USER_059 and GROUP_008 manager USER_002 (type reminder).",
        ],
        "ambiguity": "'major SAP outage' resolves to INC_001 (the only MAJOR incident in scope); NOTIF_003 is its only queued notification.",
        "conversion": "multi-turn rewrite; notification texts given by the operator.",
        "negatives": ["queued_left_queued", "delivered_notification_also_failed", "caller_sent_reminder", "group_manager_skipped"],
        "risk": "low",
    },
    "78a5b71e_a4fb07a9": {
        "route": "faithful",
        "workflow": "move a completed change through implement to post-implementation review and record an extra affected CI",
        "scope": {"seed_db": "msp_db.json", "org_ids": ["ORG_001"], "tenancy": "msp_provider_internal"},
        "delta": "msp_main",
        "planned_write_tools": ["update_change"],
        "estimated_tool_calls": 5,
        "policy": [
            "§4.2 transitions: CHG_001 is scheduled, so the gold goes scheduled -> implement -> review; no close_code/close_notes because it is not closed.",
            "§4.2 'implement within the planned window': the change model has no window fields; the operator states it completed inside the approved window and current_time is set inside that narrative.",
        ],
        "ambiguity": "Bench2 changes hold one configuration_item; the second CI (SRV-LNX-4421 = CI_002) cannot be linked, so the operator asks for an exact short_description wording that names it (upstream checks short_description LIKE '%SRV-LNX-4421%'). configuration_item stays CI_001.",
        "conversion": "multi-turn rewrite; exact short_description text given by the operator; no tool gap is papered over (no CI link is claimed).",
        "negatives": ["change_closed", "left_in_implement", "new_change_created", "configuration_item_overwritten"],
        "risk": "low-medium (a model may jump scheduled -> review in one call; the final DB is the same, which is acceptable)",
    },
    "66392a82_514cea3c": {
        "route": "faithful",
        "workflow": "restructure a frontline group: new hub group, activate its manager, move all members out of the old group",
        "scope": {"seed_db": "msp_db.json", "org_ids": ["ORG_001"], "tenancy": "msp_provider_internal"},
        "delta": "msp_main",
        "planned_write_tools": ["add_new_user_group", "update_user_details", "add_new_group_member", "remove_group_membership"],
        "estimated_tool_calls": 15,
        "policy": [
            "§8.1: Leah Lewis (USER_068) is inactive and must be activated before being set as manager (upstream verifier expects active).",
            "§12.3 remove-member / merge cascades (PROPOSED: reassign active incidents first): in the provider-internal scope GROUP_003 has no active incident, so none apply.",
        ],
        "ambiguity": "none blocking. Scope trade-off: with client ORG_005 in scope, INC_004/INC_073 (assigned to moved members) would trigger the PROPOSED incident move; that adds update_incident, which shares 2 write tools with dev task itsmbench_msp_019 and breaks the train gate. The provider-internal scope keeps the upstream request intact and passes the gate.",
        "conversion": "multi-turn rewrite only.",
        "negatives": ["leah_left_inactive", "members_copied_not_moved", "manager_not_set", "old_group_deactivated"],
        "risk": "low-medium (10 membership writes)",
    },
    "46ca2862_56a4f834": {
        "route": "faithful",
        "workflow": "expand a service's offering portfolio and re-point an incident template to a new offering",
        "scope": {"seed_db": "msp_db.json", "org_ids": ["ORG_001"], "tenancy": "msp_provider_internal"},
        "delta": "msp_main",
        "planned_write_tools": ["register_new_service_offering", "update_incident_template"],
        "estimated_tool_calls": 8,
        "policy": [
            "§5.2 REQUIRED: offerings inherit business_criticality/environment from the parent (Enterprise Email SVC_001: critical, production), consistent with the request.",
            "Owner = the parent service owner USER_004 (data); classification technology-management is an enum value.",
        ],
        "ambiguity": "offering short descriptions are free text: the operator supplies them. TMPL_004 keeps service SVC_001 (already correct) and gets the new Mobile Email Access offering.",
        "conversion": "multi-turn rewrite; short descriptions given by the operator.",
        "negatives": ["offering_under_wrong_service", "template_service_changed", "criticality_not_inherited", "one_offering_missing"],
        "risk": "low-medium",
    },
    "99ba2325_7915481a": {
        "route": "faithful",
        "workflow": "document a service restoration: publish a KB article, correct the replaced CI's serial, restore service status",
        "scope": {"seed_db": "msp_db.json", "org_ids": ["ORG_001"], "tenancy": "msp_provider_internal"},
        "delta": "msp_main",
        "planned_write_tools": ["create_knowledge_article", "update_configuration_item", "update_service"],
        "estimated_tool_calls": 7,
        "policy": [
            "§7 PROPOSED draft-and-review does not bind a Manager: Joanne Simpson (manager) may publish directly; the request asks for public visibility (external).",
        ],
        "ambiguity": "'someone swapped out the database CI … not totally sure' could mean registering a new CI; the simulator clarifies: update the serial on the existing CI_001 record, do not register a new CI. 'I believe it's back now': the simulator confirms the service is restored (SVC_007 repair_in_progress -> operational).",
        "conversion": "multi-turn rewrite; two operator clarifications that the upstream verifiers do not contradict; title/body verbatim.",
        "negatives": ["new_ci_registered", "article_left_draft_internal", "service_left_in_repair", "serial_set_on_wrong_ci"],
        "risk": "low-medium (long verbatim body ~450 tokens in one action)",
    },
    "97d7df7d_8fd400d1": {
        "route": "faithful",
        "workflow": "select aging open incidents of a group by date and alert their assignees",
        "scope": {"seed_db": "single_tenant_db.json", "org_ids": ["ORG_001"], "tenancy": "single_tenant"},
        "delta": None,
        "planned_write_tools": ["send_notification"],
        "estimated_tool_calls": 6,
        "policy": [
            "'treat them as top-priority work' is not a priority change: the simulator says not to edit the incidents (a priority change would also trigger the §6 SLA swap).",
        ],
        "ambiguity": "data gives a unique set: open GROUP_001 incidents created before 2024-02-20 are INC_003 (USER_003) and INC_004 (USER_006).",
        "conversion": "multi-turn rewrite; the operator gives the subject and a message template the agent fills with each incident's summary and creation date.",
        "negatives": ["alert_for_incident_after_cutoff", "priority_raised", "caller_alerted_instead_of_assignee", "notification_type_report"],
        "risk": "low",
    },
    "0be70e35_6bb33392": {
        "route": "adapted",
        "workflow": "log an incident from an emailed hardware report, attach the matching SLA, send the proposed report notification",
        "scope": {"seed_db": "single_tenant_db.json", "org_ids": ["ORG_001"], "tenancy": "single_tenant"},
        "delta": None,
        "planned_write_tools": ["create_incident", "link_new_incident_sla", "send_notification"],
        "estimated_tool_calls": 8,
        "adaptations": [
            {"change": "urgency high -> medium (impact stays high)", "reason": "upstream verifier requires priority=high, which contradicts the §3.1 matrix for high x high (critical); medium keeps the intended high priority and its unique ORG_001 SLA_002"},
            {"change": "channel phone -> email", "reason": "the prompt says the report arrived by email; upstream verifier says phone"},
            {"change": "notification type alert -> report, sent after the agent proposes it and the operator confirms", "reason": "§3.1 PROPOSED report notification to the caller; operator supplies the text"},
            {"change": "scenario states the laptop boots normally and shuts down under load", "reason": "makes the REQUIRED knowledge search unambiguous: no ORG_001 article (password reset, email client, desktop boot failure) matches, so the gold links none"},
        ],
        "policy": [
            "§3.1 REQUIRED knowledge search: no match (see adaptation 4).",
            "§3.1/§6 REQUIRED SLA: ORG_001 has exactly one high-priority SLA (SLA_002).",
            "§3.1 PROPOSED CI: CI_001 'Carlos Laptop' set as configuration_item after confirmation.",
        ],
        "ambiguity": "residual: a model may still link KB_003 (desktop boot failure); judged a wrong match given the adapted scenario.",
        "conversion": "redesigned scenario/known_info/gold/NL assertions; upstream task and verifiers kept for provenance. Expected after adaptation: upstream incident and notification verifiers fail (urgency, channel, type), the SLA verifier passes.",
        "negatives": ["priority_critical", "kb_003_linked", "sla_missing", "notification_type_alert"],
        "risk": "medium",
    },
    "66392a82_a172d90d": {
        "route": "adapted",
        "workflow": "audit one client's password-reset incidents and apply the existing reset article where missing",
        "scope": {"seed_db": "msp_db.json", "org_ids": ["ORG_001", "ORG_016"], "tenancy": "provider_client"},
        "delta": "msp_main",
        "planned_write_tools": ["link_knowledge_to_incident"],
        "estimated_tool_calls": 5,
        "adaptations": [
            {"change": "scope: all orgs (10 incidents across 10 client orgs) -> one client org ORG_016 (INC_071, INC_088)", "reason": "MSP tasks are provider/client scoped in Bench2; unsliced, one unfiltered list_incidents call is ~30k tokens (step 6C)"},
            {"change": "kb_use resolution -> applied", "reason": "§7: 'resolution' is for an article authored to document that incident; KB_001 is a pre-existing published article, i.e. 'applied'"},
            {"change": "drop 'if the article helped solve the issue'", "reason": "not decidable from the records; the operator states the article was used for every password-reset incident"},
        ],
        "policy": ["§7 use tracking (applied)."],
        "ambiguity": "none after adaptation: ORG_016 has two password-reset incidents since 2025-11-01, both without a KB link.",
        "conversion": "redesigned scope/scenario/gold; upstream task and verifiers kept for provenance. Expected after adaptation: upstream verifiers fail (used_as and out-of-scope incidents).",
        "negatives": ["linked_as_resolution", "sap_incident_linked", "closed_incident_missed", "linked_as_suggested"],
        "risk": "low",
    },
}

#: Verdict for every other candidate that passes the automatic train gate.
OTHER_GATE_PASSERS: dict[str, tuple[str, str]] = {
    "391f1eff_27934d6b": ("reserve", "KB authored for a resolved incident (§7 REQUIRED, consistent); upstream pins KB_026/IKB_046, so org slicing must be checked first; overlaps the KB-authoring workflows already in the train merged family"),
    "97d7df7d_be7218cc": ("reserve", "same workflow as the selected 97d7df7d_8fd400d1 (query-selected incident, notify participants); held back to avoid an entity-swap duplicate"),
    "a8eea1c0_73a73ce7": ("excluded", "same write signature and pattern as the selected 99ba2325_887f5580 (CI state change + affected-CI link)"),
    "99ba2325_51726f9c": ("excluded", "workflow near-duplicate of accepted pilot 46ca2862_b6ef60ca (register CI, then log an incident); also low priority -> two ORG_001 SLAs"),
    "1d219d93_76cd5b4d": ("excluded", "workflow near-duplicate of accepted pilot 46ca2862_b6ef60ca (register a new device, log an incident on it)"),
    "764e5ead_f8619b02": ("excluded", "workflow near-duplicate of accepted pilot 1d219d93_d1ba6c2d (new location, move a CI, register a CI)"),
    "66392a82_1c69521d": ("excluded", "onboarding: §12.1 REQUIRED user_role rows have no tool (Bench2 originals share this gap); onboarding already in train"),
    "66392a82_c1a599bb": ("excluded", "onboarding: §12.1 user_role tool gap; onboarding already in train"),
    "764e5ead_20429ef8": ("excluded", "onboarding: §12.1 user_role tool gap; near-duplicate of accepted 1d219d93_d1ba6c2d (new office set-up)"),
    "764e5ead_48850a98": ("deferred", "'validate active and legitimate CIs' leaves the CI set undefined; would need a full rewrite"),
    "accab84d_d7ec710e": ("deferred", "REQUIRED knowledge match ambiguous in ORG_002 (KB_005 'Corporate Email Access Guide' vs an email-server outage); assigned_to at creation vs the §3.2 work-note rule is unsettled"),
    "391f1eff_c23d76df": ("deferred", "REQUIRED knowledge match ambiguous in ORG_002 (KB_005 vs an unresponsive email server)"),
    "99ba2325_f931fd42": ("deferred", "names single-tenant records (Elena Petrov, 'Desktop Boot Failure Troubleshooting', INC_024) but maps to the MSP seed, where Elena does not exist; seed mapping must be investigated"),
    "c5ca08eb_db61cf35": ("excluded", "the only verifier passes when the new incident does NOT carry the reference ticket's service/offering, contradicting the prompt; MSP incident creation (SLA)"),
}
#: MSP incident creation: blocked by decision D6D-1 (SLA rule), not by the task itself.
SLA_DECISION_TAG = "deferred_pending_D6D-1"

PENDING_DECISIONS = [
    {
        "id": "D6D-1",
        "question": "MSP incident creation: which incident_sla rows does the §3.1/§6 REQUIRED 'spawn an incident_sla row matched to priority' produce?",
        "evidence": "ORG_001 in msp_db.json has a response and a resolution SLA for every priority; seed incidents with SLA rows: 18 resolution-only, 3 response-only, 2 both; upstream 391f1eff_8cb006f5 expects both rows; other MSP orgs have no SLA definitions.",
        "options": ["keep excluding MSP incident creation (default)", "one row per active matching definition (response + resolution), as upstream 391f1eff_8cb006f5", "resolution only (seed majority)"],
        "effect": "a fixed rule would unlock the 6 train-gate candidates whose caller is in ORG_001 (incl. the policy-refusal tasks c5ca08eb_3ce3c52f and d4a463c3_3f20bdca); the 4 whose org or priority has no SLA definition stay excluded either way",
    },
    {
        "id": "D6D-2",
        "question": "Onboarding with add_new_user: §12.1 REQUIRED user_role rows cannot be written by any tool. Accept derived onboarding tasks under the same convention as the 9 Bench2 originals (gold without user_role rows)?",
        "evidence": "No Bench2 tool writes user_role; 9 Bench2 tasks add users without it; 6B isolated offboarding (66392a82_fd509f97) partly because user_role removal is impossible.",
        "options": ["no (default; this batch selects no onboarding task)", "yes, with the gap recorded per task"],
        "effect": "would unlock 3 onboarding candidates, all close to train workflows",
    },
]


# -- helpers -------------------------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def _suffix(task_id: str) -> str:
    return "_".join(task_id.split("_")[-2:])


def _seed(seed_db: str) -> dict[str, dict[str, dict]]:
    return json.loads((itsm_environment.ITSM_TASKS_PATH.parent / seed_db).read_text(encoding="utf-8"))


def _jaccard(a: frozenset, b: frozenset) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _link_reason(a: frozenset, b: frozenset) -> str | None:
    jaccard = _jaccard(a, b)
    if jaccard >= WRITE_JACCARD_THRESHOLD:
        return f"write_jaccard={jaccard:.2f}"
    shared = sorted(a & b & CREATION_TOOLS)
    return f"shared_creation={','.join(shared)}" if shared else None


def _predicates(query: str) -> dict[str, set[str]]:
    found: dict[str, set[str]] = collections.defaultdict(set)
    for column, single, double, bare in _EQ.findall(query):
        found[column.lower()].add((single or double or bare).lower() if (single or double or bare) else "")
    return found


def _main_table(query: str) -> str | None:
    match = _FROM.search(query)
    return match.group(1).lower() if match else None


def _dedup_verifiers(task: dict[str, Any]) -> list[dict[str, Any]]:
    seen, unique = set(), []
    for verifier in task["verifier_audit"]["verifiers"]:
        key = " ".join(verifier["query"].split())
        if key not in seen:
            seen.add(key)
            unique.append(verifier)
    return unique


# -- v2 reference set ----------------------------------------------------------------------------------


def v2_reference() -> dict[str, Any]:
    manifest = json.loads(DEFAULT_V2_PATH.read_text(encoding="utf-8"))
    raw = {t["id"]: t for t in load_raw_tasks()}
    raw.update(derived_raw_tasks())
    features = task_features(list(raw.values()))
    gold_ids = {
        tid: {
            v
            for a in t["evaluation_criteria"]["actions"]
            for v in a["arguments"].values()
            if isinstance(v, str) and _RECORD_ID.fullmatch(v)
        }
        for tid, t in raw.items()
    }
    return {
        "manifest": manifest,
        "writes": {tid: f.write_tools for tid, f in features.items()},
        "seed": {tid: t["seed_db"] for tid, t in raw.items()},
        "gold_ids": gold_ids,
        "split": {tid: manifest["tasks"][tid]["split"] for tid in raw},
        "family": {tid: manifest["tasks"][tid]["family"] for tid in raw},
    }


def split_relations(writes: frozenset, entities: set[str], seed_db: str, ref: dict[str, Any]) -> dict[str, Any]:
    links, shared_writes, entity_overlap = [], {}, collections.defaultdict(list)
    max_jaccard_train = (0.0, None)
    for tid, other in ref["writes"].items():
        split = ref["split"][tid]
        reason = _link_reason(writes, other)
        if reason:
            links.append({"task": tid, "split": split, "family": ref["family"][tid], "reason": reason})
        count = len(writes & other)
        if count > shared_writes.get(split, (0, None))[0]:
            shared_writes[split] = (count, tid)
        if split == "train":
            max_jaccard_train = max(max_jaccard_train, (round(_jaccard(writes, other), 3), tid), key=lambda x: x[0])
        if ref["seed"][tid] == seed_db:
            shared = sorted(entities & ref["gold_ids"][tid])
            if len(shared) >= 2:
                entity_overlap[split].append({"task": tid, "entities": shared})
    return {
        "family_links": links,
        "linked_splits": sorted({link["split"] for link in links}),
        "linked_families": sorted({link["family"] for link in links}),
        "max_shared_write_tools": {s: {"count": c, "task": t} for s, (c, t) in sorted(shared_writes.items())},
        "max_write_jaccard_with_train_task": {"jaccard": max_jaccard_train[0], "task": max_jaccard_train[1]},
        "entity_overlap_ge2": dict(sorted(entity_overlap.items())),
    }


# -- policy flags --------------------------------------------------------------------------------------


def policy_flags(task: dict[str, Any], seeds: dict[str, dict]) -> dict[str, Any]:
    seed = seeds[task["mapped_bench2_seed"]]
    users = seed["users"]
    writes = set(task["oracle_write_tools"])
    flags: dict[str, Any] = {}
    verifiers = _dedup_verifiers(task)

    if "create_incident" in writes:
        incident = collections.defaultdict(set)
        for verifier in verifiers:
            if _main_table(verifier["query"]) == "incident" or " from incident" in verifier["query"].lower():
                for column, values in _predicates(verifier["query"]).items():
                    incident[column] |= values
        caller = next(iter(sorted(incident.get("caller_id", set()))), None)
        org = users.get(caller.upper(), {}).get("org_id") if caller else None
        if org is None and task["acting_user"]["user_id"]:
            org = users.get(task["acting_user"]["user_id"], {}).get("org_id")
        impact = next(iter(incident.get("impact", set())), None)
        urgency = next(iter(incident.get("urgency", set())), None)
        verifier_priority = next(iter(incident.get("priority", set())), None)
        matrix_priority = PRIORITY_MATRIX.get((impact, urgency)) if impact and urgency else None
        priority = matrix_priority or verifier_priority
        slas = [
            s["sla_def_id"]
            for s in seed["sla_definition"].values()
            if s["org_id"] == org and s["active"] and s["applies_to_priority"] == priority
        ]
        kbs = [k["knowledge_id"] for k in seed["knowledge"].values() if k["org_id"] == org and k["state"] == "published"]
        flags["incident_creation"] = {
            "incident_org": org,
            "impact": impact,
            "urgency": urgency,
            "verifier_priority": verifier_priority,
            "matrix_priority": matrix_priority,
            "priority_matrix_conflict": bool(matrix_priority and verifier_priority and matrix_priority != verifier_priority),
            "sla_definitions_for_priority": slas,
            "sla_status": "unique" if len(slas) == 1 else ("none" if not slas else "ambiguous"),
            "published_kb_in_org": kbs,
        }
    seed_users = set(users)
    for verifier in verifiers:
        query = verifier["query"]
        table = _main_table(query)
        preds = _predicates(query)
        inactive = preds.get("active", set()) & {"false", "0"}
        if table == "users" and inactive and set(_RECORD_ID.findall(query)) & seed_users and "add_new_user" not in writes:
            flags["deactivates_existing_user"] = True
        if table == "user_group" and inactive:
            flags["deactivates_group"] = True
        if table == "incident" and "resolved" in preds.get("status", set()):
            flags["resolves_incident"] = True
    if "add_new_user" in writes:
        flags["user_role_tool_gap"] = "§12.1 REQUIRED user_role rows: no Bench2 tool writes user_role"
    if flags.get("deactivates_existing_user"):
        flags["user_role_tool_gap"] = "§12.2 REQUIRED user_role removal: no Bench2 tool writes user_role"
    return flags


def automatic_policy_blockers(flags: dict[str, Any]) -> list[str]:
    blockers = []
    incident = flags.get("incident_creation")
    if incident:
        if incident["sla_status"] != "unique":
            blockers.append(f"incident_sla_{incident['sla_status']}")
        if incident["priority_matrix_conflict"]:
            blockers.append("verifier_priority_contradicts_matrix")
    if "user_role_tool_gap" in flags:
        blockers.append("user_role_tool_gap")
    if flags.get("deactivates_group"):
        blockers.append("group_dissolution_check_needed")
    return blockers


# -- lengths -------------------------------------------------------------------------------------------


def unfiltered_read_max(tokenizer, seed_db: str, org_ids: list[str] | None, delta) -> dict[str, Any]:
    environment = itsm_environment.get_environment(db_delta=delta, org_ids=org_ids, seed_db=seed_db)
    best = (0, None)
    over = []
    for name, tool in environment.tools.tools.items():
        if environment.tools.tool_type(name).value != "read":
            continue
        if any(p.default is inspect.Parameter.empty for p in inspect.signature(tool).parameters.values()):
            continue
        try:
            text = _stringify(environment.make_tool_call(name)) or "null"
        except Exception:
            continue
        tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        best = max(best, (tokens, name), key=lambda x: x[0])
        if tokens > MAX_RESPONSE_LENGTH:
            over.append({"tool": name, "raw_tokens": tokens})
    return {"max_raw_observation_tokens": best[0], "tool": best[1], "over_max_response_length": over}


def tokens_per_step() -> dict[str, Any]:
    episodes = json.loads(TOKENS_V2_PATH.read_text(encoding="utf-8"))["episodes"]
    ratios = [e["response_tokens"] / e["step_count"] for e in episodes if e["step_count"]]
    return {"median": round(statistics.median(ratios), 1), "max": round(max(ratios), 1), "episodes": len(ratios), "source": _rel(TOKENS_V2_PATH)}


# -- build ---------------------------------------------------------------------------------------------


def build(tokenizer_dir: Path) -> dict[str, Any]:
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    if len(load_pilot()["isolated"]) + 3 != len(audit["pilot"]):
        raise ValueError("pilot file and 6A pilot list disagree")
    ref = v2_reference()
    seeds = {name: _seed(name) for name in ("msp_db.json", "single_tenant_db.json")}
    seed_ids = {name: {k for v in seed.values() if isinstance(v, dict) for k in v} for name, seed in seeds.items()}
    bench_writes = write_tool_names()
    used_in_v2 = set().union(*ref["writes"].values())

    candidates = {tid: t for tid, t in audit["tasks"].items() if tid not in audit["pilot"]}
    by_suffix = {_suffix(tid): tid for tid in candidates}
    unknown = (set(BATCH) | set(OTHER_GATE_PASSERS)) - set(by_suffix)
    if unknown:
        raise ValueError(f"manual review names unknown candidates: {sorted(unknown)}")

    rows = {}
    for tid, task in sorted(candidates.items()):
        writes = frozenset(task["oracle_write_tools"])
        entities = {
            rid
            for v in _dedup_verifiers(task)
            for rid in _RECORD_ID.findall(v["query"])
        } & seed_ids[task["mapped_bench2_seed"]]
        entities -= set(task["verifier_created_ids"])
        relations = split_relations(writes, entities, task["mapped_bench2_seed"], ref)
        flags = policy_flags(task, seeds)
        blocking_verifier = sorted(set(task["issues"]) & {"conflicting_verifiers", "verifier_does_not_execute"})
        near_dup = bool(task["near_duplicate_of"])
        eval_shared = max(
            (relations["max_shared_write_tools"].get(s, {"count": 0})["count"] for s in ("dev", "heldout")), default=0
        )
        train_gate = []
        if set(relations["linked_splits"]) - {"train"}:
            train_gate.append("links_dev_or_heldout_family_v2")
        if eval_shared > MAX_SHARED_WRITES:
            train_gate.append(f"shares_{eval_shared}_write_tools_with_dev_or_heldout_task")
        if any(s in relations["entity_overlap_ge2"] for s in ("dev", "heldout")):
            train_gate.append("shares_2plus_entities_with_dev_or_heldout_task")
        if near_dup:
            train_gate.append("near_duplicate_of_bench2_task")
        train_gate += blocking_verifier
        train_shared = relations["max_shared_write_tools"].get("train", {"count": 0})["count"]
        heldout_gate = []
        if set(relations["linked_splits"]) & {"train", "dev"}:
            heldout_gate.append("links_train_or_dev_family_v2")
        if train_shared > MAX_SHARED_WRITES:
            heldout_gate.append(f"shares_{train_shared}_write_tools_with_train_task")
        if "train" in relations["entity_overlap_ge2"]:
            heldout_gate.append("shares_2plus_entities_with_train_task")
        if near_dup:
            heldout_gate.append("near_duplicate_of_bench2_task")
        heldout_gate += blocking_verifier
        suffix = _suffix(tid)
        if suffix in BATCH:
            manual = {"verdict": "selected", "route": BATCH[suffix]["route"]}
        elif suffix in OTHER_GATE_PASSERS:
            verdict, reason = OTHER_GATE_PASSERS[suffix]
            manual = {"verdict": verdict, "reason": reason}
        elif not train_gate and flags.get("incident_creation", {}).get("sla_status") == "ambiguous" and task["mapped_bench2_seed"] == "msp_db.json":
            manual = {"verdict": SLA_DECISION_TAG, "reason": "MSP incident creation with two matching SLA definitions; see D6D-1"}
        elif not train_gate and flags.get("incident_creation", {}).get("sla_status") == "none":
            manual = {
                "verdict": "excluded",
                "reason": "REQUIRED SLA spawn on creation is impossible: no SLA definition for the incident's org/priority (same treatment as 6B's impossible REQUIRED steps)",
            }
        else:
            manual = None
        rows[tid] = {
            "suffix": suffix,
            "annotator": tid.split("_")[-2],
            "seed": task["seed"],
            "mapped_bench2_seed": task["mapped_bench2_seed"],
            "audit_category": task["category"],
            "audit_issues": task["issues"],
            "oracle_write_tools": sorted(writes),
            "write_tools_unused_in_v2": sorted(writes - used_in_v2),
            "verifier_entities": sorted(entities),
            **relations,
            "policy_flags": flags,
            "automatic_policy_blockers": automatic_policy_blockers(flags),
            "train_gate_failures": train_gate,
            "heldout_gate_failures": heldout_gate,
            "manual": manual,
        }

    gate_passers = sorted(tid for tid, r in rows.items() if not r["train_gate_failures"])
    unreviewed = [tid for tid in gate_passers if rows[tid]["manual"] is None]
    if unreviewed:
        raise ValueError(f"train-gate passers without a manual verdict: {[_suffix(t) for t in unreviewed]}")
    reviewed_failing = [rows[by_suffix[s]]["suffix"] for s in [*BATCH, *OTHER_GATE_PASSERS] if rows[by_suffix[s]]["train_gate_failures"]]
    if reviewed_failing:
        raise ValueError(f"manual verdicts on candidates that fail the train gate: {reviewed_failing}")

    # -- the proposed batch: re-check planned writes against v2 ------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), trust_remote_code=False)
    msp_delta = pilot_tasks()[MSP_MAIN_DELTA_SOURCE].initial_state_delta
    per_step = tokens_per_step()
    length_cache: dict[tuple, dict[str, Any]] = {}
    batch, batch_writes = [], {}
    for suffix, record in BATCH.items():
        tid = by_suffix[suffix]
        row = rows[tid]
        writes = frozenset(record["planned_write_tools"])
        if not writes <= bench_writes:
            raise ValueError(f"{suffix}: planned writes are not Bench2 write tools")
        relations = split_relations(writes, set(row["verifier_entities"]), record["scope"]["seed_db"], ref)
        eval_shared = max((relations["max_shared_write_tools"].get(s, {"count": 0})["count"] for s in ("dev", "heldout")), default=0)
        scope = record["scope"]
        key = (scope["seed_db"], tuple(scope["org_ids"] or ()))
        if key not in length_cache:
            delta = msp_delta if scope["seed_db"] == "msp_db.json" else None
            length_cache[key] = unfiltered_read_max(tokenizer, scope["seed_db"], scope["org_ids"], delta)
        new_id = (ADAPTED_PREFIX if record["route"] == "adapted" else FAITHFUL_PREFIX) + suffix
        batch_writes[new_id] = writes
        batch.append(
            {
                "task_id": new_id,
                "upstream_task_id": tid,
                **{k: v for k, v in record.items()},
                "audit_seed_delta_records": candidates[tid]["seed_delta_size"],
                "planned_relations_v2": {
                    "linked_splits": relations["linked_splits"],
                    "linked_families": relations["linked_families"],
                    "max_shared_write_tools": relations["max_shared_write_tools"],
                    "entity_overlap_ge2": relations["entity_overlap_ge2"],
                    "max_write_jaccard_with_train_task": relations["max_write_jaccard_with_train_task"],
                },
                "planned_gate_ok": not (set(relations["linked_splits"]) - {"train"}) and eval_shared <= MAX_SHARED_WRITES
                and not any(s in relations["entity_overlap_ge2"] for s in ("dev", "heldout")),
                "write_tools_unused_in_v2": sorted(writes - used_in_v2),
                "length": {
                    "unfiltered_read_under_scope": length_cache[key],
                    "estimated_response_tokens": round(record["estimated_tool_calls"] * per_step["median"]),
                    "estimate_basis": f"estimated_tool_calls x median response tokens per step ({per_step['median']}) over the v2 gold replays",
                },
                "acceptance": [
                    "Bench2 Task with scenario/persona/known_info/simulator guidance, current_time, org_ids and (when needed) initial_state_delta; new ID as given",
                    "strict gold replay under the task clock: Bench2 DB (fuzzy) = 1, replayed twice with identical final DB hash",
                    "faithful: every deduplicated upstream verifier passes on the gold final DB and fails on the initial state; sliced seed + delta equals the same slice of the upstream seed"
                    if record["route"] == "faithful"
                    else "adapted: upstream verifiers run on the gold final DB and the expected pass/fail per verifier matches the recorded adaptations; no equivalence claimed",
                    "each negative case: Bench2 DB = 0 with the expected mismatch; upstream verdict recorded",
                    "information boundary: reset observation and public info carry no task/provenance text",
                    "family re-check over v2 + batch: every batch task's family has train members only; dev/held-out unchanged",
                    "token measurement of the scripted gold episode (prompt/response/full sequence)",
                ],
            }
        )
    batch_component = {}
    for a, b in itertools.combinations(sorted(batch_writes), 2):
        reason = _link_reason(batch_writes[a], batch_writes[b])
        if reason:
            batch_component.setdefault(a, []).append({"task": b, "reason": reason})

    # -- v3 inputs: what a re-split could use (counts only; no manifest) ---------------------------
    def policy_clean(r: dict[str, Any]) -> bool:
        return not r["automatic_policy_blockers"] and not set(r["audit_issues"]) & {"conflicting_verifiers", "verifier_does_not_execute"}

    selected_ids = {by_suffix[s] for s in BATCH}
    # The proposed batch counts as train here: a held-out task must not sit next to it either.
    ref_with_batch = {key: dict(value) for key, value in ref.items() if key != "manifest"}
    for entry in batch:
        new_id = entry["task_id"]
        ref_with_batch["writes"][new_id] = batch_writes[new_id]
        ref_with_batch["seed"][new_id] = entry["scope"]["seed_db"]
        ref_with_batch["gold_ids"][new_id] = set(rows[entry["upstream_task_id"]]["verifier_entities"])
        ref_with_batch["split"][new_id] = "train"
        ref_with_batch["family"][new_id] = "proposed_6d_batch"

    def heldout_failures_with_batch(tid: str) -> list[str]:
        r = rows[tid]
        relations = split_relations(
            frozenset(r["oracle_write_tools"]), set(r["verifier_entities"]), r["mapped_bench2_seed"], ref_with_batch
        )
        failures = [f for f in r["heldout_gate_failures"] if not f.startswith(("links_train", "shares_"))]
        if set(relations["linked_splits"]) & {"train", "dev"}:
            failures.append("links_train_or_dev_family_incl_batch")
        train_shared = relations["max_shared_write_tools"].get("train", {"count": 0})["count"]
        if train_shared > MAX_SHARED_WRITES:
            failures.append(f"shares_{train_shared}_write_tools_with_train_task_incl_batch")
        if "train" in relations["entity_overlap_ge2"]:
            failures.append("shares_2plus_entities_with_train_task_incl_batch")
        return sorted(set(failures))

    heldout_pool = sorted(tid for tid in rows if tid not in selected_ids and not heldout_failures_with_batch(tid))
    heldout_clean = [tid for tid in heldout_pool if policy_clean(rows[tid])]
    # components among the held-out pool itself (new families vs. joining existing held-out ones)
    parent = {tid: tid for tid in heldout_clean}

    def root(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in itertools.combinations(heldout_clean, 2):
        if _link_reason(frozenset(rows[a]["oracle_write_tools"]), frozenset(rows[b]["oracle_write_tools"])):
            parent[root(a)] = root(b)
    groups = collections.defaultdict(list)
    for tid in heldout_clean:
        groups[root(tid)].append(rows[tid]["suffix"])
    dev_only = sorted(
        tid for tid, r in rows.items() if r["linked_splits"] == ["dev"] and policy_clean(r) and tid not in selected_ids
    )

    summary = {
        "candidates": len(rows),
        "train_gate_passers": len(gate_passers),
        "manual_verdicts": dict(collections.Counter(rows[t]["manual"]["verdict"] for t in gate_passers)),
        "train_gate_failure_reasons": dict(
            collections.Counter(reason.split("_with_")[0] if reason.startswith("shares_") else reason for r in rows.values() for reason in r["train_gate_failures"])
        ),
        "automatic_policy_blockers": dict(collections.Counter(b for r in rows.values() for b in r["automatic_policy_blockers"])),
        "batch_size": len(batch),
        "batch_routes": dict(collections.Counter(b["route"] for b in batch)),
        "batch_seeds": dict(collections.Counter(b["scope"]["seed_db"] for b in batch)),
    }
    v3_inputs = {
        "heldout_gate": "against v2 train plus the proposed batch: no train/dev family link; <= 1 write tool shared with any train task; < 2 same-seed entities with any train task; not a near-duplicate; no conflicting/non-executable verifier",
        "heldout_gate_passers": [rows[t]["suffix"] for t in heldout_pool],
        "heldout_gate_passers_policy_clean": [rows[t]["suffix"] for t in heldout_clean],
        "heldout_policy_clean_groups": sorted(sorted(g) for g in groups.values()),
        "heldout_policy_clean_joining_existing_heldout_families": sorted(
            rows[t]["suffix"] for t in heldout_clean if rows[t]["linked_splits"] == ["heldout"]
        ),
        "dev_only_linked_policy_clean": [rows[t]["suffix"] for t in dev_only],
        "note": "automatic flags only: every v3 task still needs the manual review and conversion this batch gets; MSP incident creation stays blocked until D6D-1",
    }
    return {
        "version": "itsm_eops_screening_v1",
        "step": "6D-1",
        "scope_of_this_file": "candidate screening and batch proposal only; no task converted, no split, policy or scoring change",
        "inputs": {
            "audit": {"path": _rel(AUDIT_PATH), "sha256": _sha256(AUDIT_PATH)},
            "split_v1": {"path": _rel(DEFAULT_MANIFEST_PATH), "sha256": _sha256(DEFAULT_MANIFEST_PATH)},
            "split_v2": {"path": _rel(DEFAULT_V2_PATH), "sha256": _sha256(DEFAULT_V2_PATH)},
            "pilot": {"path": _rel(DEFAULT_PILOT_PATH), "sha256": _sha256(DEFAULT_PILOT_PATH)},
            "tokenizer": {"repo": "Qwen/Qwen3-8B", "revision": TOKENIZER_REVISION},
        },
        "rules": {
            "train_gate": "6A gate re-run against split v2: no dev/held-out family link (v2 families, oracle write tools), <= 1 write tool shared with any dev/held-out task, < 2 same-seed verifier entities with any dev/held-out task, not a near-duplicate, no conflicting/non-executable verifier",
            "excluded_upfront": "the 8 6B pilot tasks (3 accepted, 5 isolated); isolated tasks are not reconsidered",
            "routes": {
                "faithful": f"ID {FAITHFUL_PREFIX}<suffix>; 6B standard",
                "adapted": f"ID {ADAPTED_PREFIX}<suffix>; Bench2-adapted derived task; upstream task and verifiers kept for provenance only; every change recorded with its reason; never claimed equivalent to EnterpriseOps-Gym",
            },
            "policy_handling": "Bench2 policy.md unchanged; REQUIRED cascades whose result is not unique, missing tools for REQUIRED steps and requests violating mandatory policy are not released by adaptation",
        },
        "summary": summary,
        "pending_decisions": PENDING_DECISIONS,
        "tokens_per_step": per_step,
        "batch": batch,
        "batch_internal_links": batch_component,
        "v3_inputs": v3_inputs,
        "candidates": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true", help="fail if the committed file differs")
    args = parser.parse_args()
    result = build(args.tokenizer_dir)
    text = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=False) + "\n"
    if args.check:
        if not args.out.exists() or args.out.read_text(encoding="utf-8") != text:
            print(f"{_rel(args.out)} is stale; rerun without --check", file=sys.stderr)
            return 1
        print("candidate screening is up to date")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(json.dumps(result["summary"], indent=1, ensure_ascii=False))
    for entry in result["batch"]:
        print(entry["task_id"], entry["route"], "gate_ok" if entry["planned_gate_ok"] else "GATE_FAIL", entry["planned_relations_v2"]["linked_families"], entry["length"]["unfiltered_read_under_scope"]["max_raw_observation_tokens"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
