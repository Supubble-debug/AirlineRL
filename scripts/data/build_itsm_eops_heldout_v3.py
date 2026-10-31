"""Build the v3 held-out candidate file (step 6D-3).

Usage (from the repo root):
    python scripts/data/build_itsm_eops_heldout_v3.py --ref-dir DIR [--check]

Scope: the eight candidates listed in docs/itsm_step6d/v3_proposal.md, admitted or
isolated under the rules fixed in docs/itsm_step6d3/README.md §1 before any
conversion. Faithful route only. ``CONVERTED`` holds the hand-written conversions
(each with its core requirements, written before validation); a converted task
that fails validation carries an ``isolate`` record and is written to
``isolated`` with its attempted conversion. ``ISOLATED`` holds candidates
isolated before conversion; each records the upstream phrases (checked against
the upstream prompt/verifiers in DIR) and the seed facts behind the decision.

Writes configs/data/itsm_eops_heldout_v3_candidates.json; ``--check`` fails if the
committed file differs. No token measurement: held-out candidates are not used
for length or budget work.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

from enterprise_worlds.data_model.tasks import Task

from agent_rl.data.itsm_eops_heldout_v3 import DEFAULT_HELDOUT_V3_PATH, HELDOUT_V3_VERSION
from agent_rl.envs.itsm_scoring import build_task_environment, task_clock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_eopsgym_itsm as audit  # noqa: E402  (step 6A)
from build_itsm_eops_batch2 import MSP_MAIN_DELTA  # noqa: E402  (step 6D-2)
from build_itsm_eops_pilot import INFORMATION_BOUNDARY, KAREN, MARCUS, SOURCE, _a  # noqa: E402  (step 6B)

RAJ = {"user_id": "USER_007", "first_name": "Raj", "last_name": "Patel", "email": "raj.patel@acme.com", "role": "admin"}
APPROVAL = {
    "date": "2026-10-30",
    "decision": "二、只执行第 6D-3：v3 held-out 候选审读与迁移验证。范围限定为 docs/itsm_step6d/v3_proposal.md 已列出的 8 道候选。……不通过的候选隔离并记录理由，不另找题替补，不放宽门槛。",
}
COMMON_NOTES = [
    "Held-out candidate for split v3; faithful route only. Not part of any split until the user accepts v3.",
    "Upstream single-turn instruction rewritten as a multi-turn conversation: the operator states the request and reveals facts from known_info when asked; simulator_guidance fixes the answers to clarifications and cascades.",
    "The agent sees Bench2's task-invariant policy.md, not the upstream per-task system prompt (sha256 recorded in source).",
    "current_time is 09:00 on the date in the upstream task_id.",
    "Free text the policy leaves to the operator is given verbatim by the operator in known_info.",
    "Scoring is Bench2's full-DB comparison; free-text fields use the fuzzy rule (gold content-token recall >= 0.5) and D8. Not equivalent to the EnterpriseOps-Gym evaluation.",
]

WORKNOTE = (
    "Issue confirmed as service-wide and recurring: multiple users report email delivery delays and shared mailbox "
    "failures. Root cause investigation is active and the service is under monitoring."
)
WORKNOTE_WRONG_SCOPE = WORKNOTE.replace("service-wide and recurring", "isolated and one-off")
WORKNOTE_NO_INVESTIGATION = "Issue confirmed as service-wide and recurring: multiple users report email delivery delays and shared mailbox failures."
DOC_MESSAGE = "Please ask your team to try the steps in '{title}' for resolving the issue"
SERVICE_DESCRIPTION = "Primary interface through which users authenticate and access core business functionality."
PROBLEM_STATEMENT = "User authentication through User App is fully blocked; no operational work can continue until restoration is achieved."

CONVERTED = [
    {
        "id": "itsmbench_eops_0be70e35_0916fba8",
        "candidate_item": 1,
        "isolate": {
            "reason": "core free-text requirement not discriminated by Bench2 DB scoring",
            "detail": (
                "Core requirement: the note on INC_006 states the issue is service-wide, recurring and under investigation. "
                "Both key-fact negatives score Bench2 DB = 1: the same note calling the issue 'isolated and one-off' (fuzzy "
                "recall 0.90) and the note without the investigation sentence (recall 0.70); the threshold is 0.5. The other "
                "three core requirements (service and offering status, which incident) are discriminated. D8, the fuzzy "
                "threshold and scoring are not changed (user decision)."
            ),
        },
        "upstream_task_id": "task_20251222_195853_743_0be70e35_0916fba8",
        "upstream_seed": "db_1765301900121_3mwjj54xy.sql",
        "seed_db": "single_tenant_db.json",
        "org_ids": ["ORG_001", "ORG_002"],
        "tenancy": {"model": "single_tenant"},
        "current_time": "2025-12-22T09:00:00",
        "initial_state_delta": None,
        "scenario": {
            "persona": {
                "identity": RAJ,
                "personality": "Calm, gives facts when asked, wants clear visibility for support teams.",
                "role_description": "Administrator (admin role), single-tenant ITSM world",
                "known_info": {
                    "requester_name": "Raj Patel",
                    "service": "the Email Service",
                    "symptom": "intermittent email delivery delays and failures when accessing shared mailboxes, reported by multiple users",
                    "service_status": "non-operational",
                    "offering_status": "non-operational (that is what 'reduced availability' means here)",
                    "incident_to_note": "the existing 'Critical system outage' incident on the Email Service",
                    "worknote": WORKNOTE,
                },
            },
            "task_description": (
                "You are Raj Patel, an administrator. Users report intermittent email delivery delays and failures when "
                "accessing shared mailboxes; it is a service-level issue. You want the Email Service's state checked, "
                "the impacted offering identified, the Email Service set to non-operational, the impacted offering set "
                "to non-operational (reduced availability), and the existing 'Critical system outage' incident on the "
                "Email Service updated with your work note."
            ),
            "simulator_guidance": (
                "Open with the email delivery problem and ask for the Email Service and its impacted offering to be updated "
                "and an existing incident noted. If asked which status 'reduced availability' means, say non-operational. "
                "If asked which incident, say the 'Critical system outage' incident on the Email Service. Give the work note "
                "verbatim from known_info when asked. Do not ask for other changes. Confirm the agent's plan otherwise. End "
                "once the changes are confirmed."
            ),
        },
        "gold": [
            _a("get_user_using_name", first_name="Raj", last_name="Patel"),
            _a("find_service_by_name", name="Email Service"),
            _a("find_service_offerings"),
            _a("list_incidents", service="SVC_001"),
            _a("update_service", service_id="SVC_001", status="non-operational"),
            _a("update_service_offering", service_offering_id="SVCOFF_001", status="non-operational"),
            _a("update_incident", incident_id="INC_006", worknotes=WORKNOTE),
        ],
        "nl_assertions": [
            "The assistant verifies the operator's identity before making any change.",
            "The Email Service (SVC_001) is set to non-operational.",
            "Its offering Outlook Email Access (SVCOFF_001) is set to non-operational.",
            "The 'Critical system outage' incident (INC_006) gets the operator's work note stating the issue is service-wide, recurring and under investigation.",
        ],
        "notes": [
            "Scope ORG_001 + ORG_002 (the whole single-tenant world): the upstream operator Raj Patel is an admin of ORG_002 while every record touched is in ORG_001; §2 gives admins CRUD on all tables.",
            "Operator clarifications, neither contradicted by the upstream verifiers: 'reduced availability' = non-operational; the incident to note is 'Critical system outage' (INC_006), one of five active Email Service incidents.",
            "No policy cascade applies to non-operational service/offering states. INC_006 is already in_progress (upstream 'Verify Incident Status' passes before any action).",
        ],
        "core_requirements": [
            {"id": "service_non_operational", "free_text": False, "negatives": ["service_left_operational"]},
            {"id": "offering_non_operational", "free_text": False, "negatives": ["offering_repair_in_progress"]},
            {"id": "note_on_outage_incident", "free_text": False, "negatives": ["note_on_wrong_incident"]},
            {"id": "note_states_service_wide_recurring_investigating", "free_text": True, "negatives": ["note_scope_wrong", "note_omits_investigation"]},
        ],
        "initial_pass_allowed": {
            "Verify Incident Status": "precondition already true in the seed (INC_006 is in_progress); the work-note verifier discriminates",
        },
        "validation_cases": [
            {"name": "service_left_operational", "kind": "core", "edit": {"drop": [4]}, "bench2_mismatch": "service/SVC_001 status", "upstream_all_pass": False, "reason": "the Email Service must become non-operational"},
            {"name": "offering_repair_in_progress", "kind": "core", "edit": {"set": {"5": {"status": "repair_in_progress"}}}, "bench2_mismatch": "service_offering/SVCOFF_001 status", "upstream_all_pass": False, "reason": "the operator defined reduced availability as non-operational"},
            {"name": "note_on_wrong_incident", "kind": "core", "edit": {"set": {"6": {"incident_id": "INC_004"}}}, "bench2_mismatch": "incident/INC_00", "upstream_all_pass": False, "reason": "the note belongs on the outage incident INC_006"},
            {"name": "note_scope_wrong", "kind": "key_fact", "edit": {"set": {"6": {"worknotes": WORKNOTE_WRONG_SCOPE}}}, "bench2_mismatch": "incident/INC_006 worknotes", "upstream_all_pass": False, "reason": "rest of the note correct, but the issue is called isolated and one-off instead of service-wide and recurring"},
            {"name": "note_omits_investigation", "kind": "key_fact", "edit": {"set": {"6": {"worknotes": WORKNOTE_NO_INVESTIGATION}}}, "bench2_mismatch": "incident/INC_006 worknotes", "upstream_all_pass": False, "reason": "rest of the note correct, but it no longer says the issue is under investigation"},
        ],
    },
    {
        "id": "itsmbench_eops_99ba2325_54f35603",
        "candidate_item": 2,
        "upstream_task_id": "task_20260113_031046_703_99ba2325_54f35603",
        "upstream_seed": "db_1765893060767_9rwtfbbp7.sql",
        "seed_db": "msp_db.json",
        "org_ids": ["ORG_001", "ORG_007", "ORG_016"],
        "tenancy": {"model": "msp_multi_client"},
        "current_time": "2026-01-13T09:00:00",
        "initial_state_delta": MSP_MAIN_DELTA,
        "decision_check": {"kind": "active_incidents_for_service", "service": "SVC_017", "expected": 4, "threshold_more_than": 3},
        "scenario": {
            "persona": {
                "identity": KAREN,
                "personality": "Careful about retirements; wants the condition checked before anything changes.",
                "role_description": "ITSM administrator, ServiceNow (provider, ORG_001)",
                "known_info": {
                    "requester_name": "Karen Watkins",
                    "service": "SAP ERP",
                    "plan": "retire SAP ERP next quarter",
                    "condition": "if more than three incidents on SAP ERP are still ongoing (not resolved or closed), do not retire; set SAP ERP to non-operational instead",
                    "offerings": "all offerings of SAP ERP get the same status as the service",
                },
            },
            "task_description": (
                "You are Karen Watkins, ITSM administrator. You plan to retire the SAP ERP service next quarter. First check "
                "whether incidents associated with it are still active. If more than three are ongoing, do not retire it; set "
                "the service to non-operational instead and give all of its offerings the same status."
            ),
            "simulator_guidance": (
                "Open with the retirement plan and the rule: more than three ongoing incidents means no retirement, set "
                "non-operational instead, and the same status for all SAP ERP offerings. If asked what counts as ongoing, say "
                "any incident that is not resolved or closed, including on-hold ones. Do not ask to touch the incidents. "
                "Confirm the agent's plan otherwise. End once the changes are confirmed."
            ),
        },
        "gold": [
            _a("get_user_using_name", first_name="Karen", last_name="Watkins"),
            _a("find_service_by_name", name="SAP ERP"),
            _a("list_incidents", service="SVC_017"),
            _a("find_service_offerings", name="SAP ERP"),
            _a("update_service", service_id="SVC_017", status="non-operational"),
            _a("update_service_offering", service_offering_id="SVCOFF_021", status="non-operational"),
            _a("update_service_offering", service_offering_id="SVCOFF_022", status="non-operational"),
            _a("update_service_offering", service_offering_id="SVCOFF_038", status="non-operational"),
        ],
        "nl_assertions": [
            "The assistant verifies the operator's identity before making any change.",
            "The assistant finds four ongoing SAP ERP incidents (INC_001 in progress; INC_044, INC_069, INC_082 on hold) and therefore does not retire the service.",
            "SAP ERP (SVC_017) is set to non-operational.",
            "All three SAP ERP offerings (Financials, HR, Logistics) are set to non-operational.",
            "No incident is changed.",
        ],
        "notes": [
            "Scope ORG_001 + ORG_007 + ORG_016: the decisive count needs the client incidents INC_044 (ORG_007) and INC_082 (ORG_016). Provider-only scope would show 2 ongoing incidents and flip the decision; the validator checks the count is 4 both in this scope and in the unsliced DB.",
            "'Ongoing' = state not in {resolved, closed}, as the policy uses 'active' throughout (§3.2, §5.1); on_hold incidents count. The operator confirms this when asked.",
            "No policy cascade applies to non-operational states; the §5.2 retirement cascades do not apply because the service is not retired.",
        ],
        "core_requirements": [
            {"id": "no_retirement_when_more_than_three_ongoing", "free_text": False, "negatives": ["retired_instead", "on_hold_not_counted"]},
            {"id": "service_non_operational", "free_text": False, "negatives": ["service_not_updated"]},
            {"id": "every_offering_same_status", "free_text": False, "negatives": ["one_offering_missed", "offerings_wrong_status"]},
        ],
        "initial_pass_allowed": {},
        "validation_cases": [
            {"name": "retired_instead", "kind": "core", "edit": {"set": {"4": {"status": "retired"}, "5": {"status": "retired"}, "6": {"status": "retired"}, "7": {"status": "retired"}}}, "bench2_mismatch": "service/SVC_017 status", "upstream_all_pass": False, "reason": "four ongoing incidents block the retirement"},
            {"name": "on_hold_not_counted", "kind": "core", "edit": {"set": {"4": {"status": "retired"}}, "drop": [5, 6, 7]}, "bench2_mismatch": "service/SVC_017 status", "upstream_all_pass": False, "reason": "counting only in-progress incidents (1) would wrongly allow retiring the service"},
            {"name": "service_not_updated", "kind": "core", "edit": {"drop": [4]}, "bench2_mismatch": "service/SVC_017 status", "upstream_all_pass": False, "reason": "the service must become non-operational"},
            {"name": "one_offering_missed", "kind": "core", "edit": {"drop": [7]}, "bench2_mismatch": "service_offering/SVCOFF_038 status", "upstream_all_pass": False, "reason": "all three offerings follow the service"},
            {"name": "offerings_wrong_status", "kind": "core", "edit": {"set": {"5": {"status": "repair_in_progress"}}}, "bench2_mismatch": "service_offering/SVCOFF_021 status", "upstream_all_pass": False, "reason": "offerings take the same status as the service"},
        ],
    },
    {
        "id": "itsmbench_eops_66392a82_ccb66a9a",
        "candidate_item": 5,
        "isolate": {
            "reason": "core free-text requirements not discriminated by Bench2 DB scoring",
            "detail": (
                "The agent's key decision, which article to point the team to, exists only in the notification text. The "
                "message naming the wrong article ('Desktop Boot Failure Troubleshooting') scores Bench2 DB = 1 (fuzzy recall "
                "0.64), and so does the problem statement 'Outlook sync failure' instead of 'Outlook connectivity failure' "
                "(recall 0.67). Severity, offering, recipient/type and the no-new-article rule are discriminated."
            ),
        },
        "upstream_task_id": "task_20260116_235649_372_66392a82_ccb66a9a",
        "upstream_seed": "db_1765301900121_3mwjj54xy.sql",
        "seed_db": "single_tenant_db.json",
        "org_ids": ["ORG_001"],
        "tenancy": {"model": "single_tenant"},
        "current_time": "2026-01-16T09:00:00",
        "initial_state_delta": None,
        "scenario": {
            "persona": {
                "identity": MARCUS,
                "personality": "Wants the problem logged and the team pointed at existing documentation.",
                "role_description": "IT administrator, TechCorp (ORG_001)",
                "known_info": {
                    "requester_name": "Marcus Thompson",
                    "reports": "Outlook email access failures disrupting development",
                    "existing_incident": "Cannot access email account",
                    "problem": "high impact, high urgency, statement 'Outlook connectivity failure', on the Email Service and its Outlook Email Access offering, raised from that incident",
                    "documentation_check": "any article about troubleshooting or configuring Outlook email",
                    "if_found": "alert Priya Sharma (priya.sharma@techcorp.com) on that incident with: " + DOC_MESSAGE.format(title="<title of the article found>"),
                    "if_not_found": "create an internal knowledge article titled 'Outlook Access Troubleshooting'",
                },
            },
            "task_description": (
                "You are Marcus Thompson, IT administrator. You receive reports of Outlook email access failures; an incident "
                "'Cannot access email account' already exists. You want a high-impact, high-urgency problem 'Outlook connectivity "
                "failure' on the Email Service's Outlook Email Access offering, raised from that incident. Then check for an "
                "article about troubleshooting or configuring Outlook email: if there is one, alert Priya Sharma on that incident "
                "with your message naming the article; if not, create an internal article 'Outlook Access Troubleshooting'."
            ),
            "simulator_guidance": (
                "Open with the Outlook reports and the existing incident. Give the problem details from known_info when asked. "
                "Give the message template; the agent fills in the article it found. Do not name the article yourself. Do not "
                "ask for other changes. Confirm the agent's plan otherwise. End once the work is confirmed."
            ),
        },
        "gold": [
            _a("get_user_using_name", first_name="Marcus", last_name="Thompson"),
            _a("list_incidents", short_description="Cannot access email account"),
            _a("find_service_offering_by_name", name="Outlook Email Access"),
            _a("create_problem", problem_statement="Outlook connectivity failure", status="new", impact="high", urgency="high", priority="critical", service="SVC_001", service_offering="SVCOFF_001", original_task="INC_001"),
            _a("retrieve_knowledge_articles", body="Outlook"),
            _a("send_notification", incident_id="INC_001", email="priya.sharma@techcorp.com", type="alert", message=DOC_MESSAGE.format(title="Email Client Configuration Guide")),
        ],
        "nl_assertions": [
            "The assistant verifies the operator's identity before making any change.",
            "A problem 'Outlook connectivity failure' is created: high impact, high urgency, critical priority, state new, Email Service / Outlook Email Access, raised from INC_001.",
            "The assistant finds 'Email Client Configuration Guide' as the Outlook troubleshooting/configuration article.",
            "Priya Sharma receives an alert on INC_001 naming 'Email Client Configuration Guide'.",
            "No new knowledge article is created.",
        ],
        "notes": [
            "Problem linked to its incident via original_task, as Bench2's own problem tasks do (msp_027, msp_028, single_017); INC_001 is also PRB_001's original_task, and the operator explicitly asks for a new problem.",
            "KB_002 'Email Client Configuration Guide' (how to configure Outlook, includes troubleshooting) is the only ORG_001 article about Outlook; KB_003 is about desktop boot failures.",
        ],
        "core_requirements": [
            {"id": "problem_severity_and_offering", "free_text": False, "negatives": ["problem_priority_high", "problem_on_wrong_offering"]},
            {"id": "problem_statement_connectivity_failure", "free_text": True, "negatives": ["statement_key_fact_wrong"]},
            {"id": "message_names_the_found_article", "free_text": True, "negatives": ["wrong_article_named"]},
            {"id": "alert_to_priya_on_incident", "free_text": False, "negatives": ["notification_type_update"]},
            {"id": "no_article_created_when_one_exists", "free_text": False, "negatives": ["article_created_anyway"]},
        ],
        "initial_pass_allowed": {},
        "validation_cases": [
            {"name": "problem_priority_high", "kind": "core", "edit": {"set": {"3": {"priority": "high"}}}, "bench2_mismatch": "priority", "upstream_all_pass": False, "reason": "high x high is critical"},
            {"name": "problem_on_wrong_offering", "kind": "core", "edit": {"set": {"3": {"service_offering": "SVCOFF_002", "service": "SVC_002"}}}, "bench2_mismatch": "service_offering", "upstream_all_pass": False, "reason": "the problem belongs to Outlook Email Access"},
            {"name": "statement_key_fact_wrong", "kind": "key_fact", "edit": {"set": {"3": {"problem_statement": "Outlook sync failure"}}}, "bench2_mismatch": "problem_statement", "upstream_all_pass": False, "reason": "rest correct, but the statement says sync instead of connectivity"},
            {"name": "wrong_article_named", "kind": "key_fact", "edit": {"set": {"5": {"message": DOC_MESSAGE.format(title="Desktop Boot Failure Troubleshooting")}}}, "bench2_mismatch": "message", "upstream_all_pass": False, "reason": "rest of the message correct, but it names the wrong article"},
            {"name": "notification_type_update", "kind": "core", "edit": {"set": {"5": {"type": "update"}}}, "bench2_mismatch": "type", "upstream_all_pass": False, "reason": "the operator asked for an alert"},
            {"name": "article_created_anyway", "kind": "core", "edit": {"extra": [_a("create_knowledge_article", title="Outlook Access Troubleshooting", owner_id="USER_001", visibility="internal")]}, "bench2_mismatch": "knowledge", "upstream_all_pass": True, "reason": "an article exists, so none is created"},
        ],
    },
    {
        "id": "itsmbench_eops_97d7df7d_ece487bd",
        "candidate_item": 7,
        "isolate": {
            "reason": "core free-text requirements not discriminated, and 2 gold entities shared with train tasks",
            "detail": (
                "The service description no longer calling it the primary authentication interface (fuzzy recall 0.80) and the "
                "problem statement calling authentication 'intermittently slow' instead of 'fully blocked' (recall 0.86) both "
                "score Bench2 DB = 1. Independently, the gold shares two same-seed entities with train tasks: LOC_003 and "
                "USER_002 with itsmbench_single_003, LOC_001 and USER_002 with itsmbench_single_005 (rule 3: < 2). It would "
                "also have merged the two held-out families."
            ),
        },
        "upstream_task_id": "task_20251218_143555_095_97d7df7d_ece487bd",
        "upstream_seed": "db_1765301900121_3mwjj54xy.sql",
        "seed_db": "single_tenant_db.json",
        "org_ids": None,
        "tenancy": {"model": "single_tenant"},
        "current_time": "2025-12-18T09:00:00",
        "initial_state_delta": None,
        "scenario": {
            "persona": {
                "identity": MARCUS,
                "personality": "Precise about the new service's attributes; urgent about the outage.",
                "role_description": "IT administrator, TechCorp (ORG_001)",
                "known_info": {
                    "requester_name": "Marcus Thompson",
                    "service": "User App: production, operational, application service, critical, owned by Priya Sharma",
                    "service_description": SERVICE_DESCRIPTION,
                    "problem_item": "the configuration item located in the USA with a cost below 1000",
                    "problem_statement": PROBLEM_STATEMENT,
                    "problem_severity": "highest impact, urgency and priority",
                },
            },
            "task_description": (
                "You are Marcus Thompson, IT administrator. You want a new application service 'User App' (production, "
                "operational, critical, owned by Priya Sharma) with your description, and a problem logged against it on the "
                "configuration item located in the USA that costs less than 1000, where user authentication is fully blocked. "
                "The problem gets the highest impact, urgency and priority and your problem statement."
            ),
            "simulator_guidance": (
                "Open with the new service and the problem. Give the description and the problem statement verbatim from "
                "known_info when asked. Describe the item only as located in the USA with a cost below 1000; do not name it. "
                "Do not ask for other changes. Confirm the agent's plan otherwise. End once both records are confirmed."
            ),
        },
        "gold": [
            _a("get_user_using_name", first_name="Marcus", last_name="Thompson"),
            _a("get_user_using_name", first_name="Priya", last_name="Sharma"),
            _a("add_new_service", name="User App", owned_by="USER_002", used_for="production", status="operational", service_classification="application", business_criticality="critical", description=SERVICE_DESCRIPTION),
            _a("find_locations", country="USA"),
            _a("find_configuration_items", location_id="LOC_001"),
            _a("find_configuration_items", location_id="LOC_003"),
            _a("create_problem", problem_statement=PROBLEM_STATEMENT, status="new", impact="high", urgency="high", priority="critical", service="SVC_005", configuration_item="CI_002"),
        ],
        "nl_assertions": [
            "The assistant verifies the operator's identity before making any change.",
            "Service 'User App' (SVC_005) is created: owned by Priya Sharma, production, operational, application, critical, with the operator's description.",
            "The assistant identifies Aisha Desktop (CI_002, USA, cost 800) as the only USA configuration item costing less than 1000.",
            "A problem with the operator's statement is created on SVC_005 and CI_002 with high impact, high urgency, critical priority, state new.",
        ],
        "notes": [
            "Not org-sliced: the upstream verifier pins the new service ID SVC_005, which only results from the whole single-tenant service table (SVC_001-004).",
            "USA locations: LOC_001 (ORG_001) and LOC_003 (ORG_002); CIs costing < 1000 there: CI_002 only (800).",
            "§4.3 'link all affected incidents': no existing incident is about authentication being blocked, so none is linked.",
        ],
        "core_requirements": [
            {"id": "service_attributes", "free_text": False, "negatives": ["service_owner_wrong"]},
            {"id": "service_description_primary_auth_interface", "free_text": True, "negatives": ["description_key_fact_wrong"]},
            {"id": "problem_on_usa_ci_below_1000", "free_text": False, "negatives": ["problem_on_wrong_ci"]},
            {"id": "problem_severity_and_service", "free_text": False, "negatives": ["problem_priority_high"]},
            {"id": "statement_authentication_fully_blocked", "free_text": True, "negatives": ["statement_key_fact_wrong"]},
        ],
        "initial_pass_allowed": {},
        "validation_cases": [
            {"name": "service_owner_wrong", "kind": "core", "edit": {"set": {"2": {"owned_by": "USER_001"}}}, "bench2_mismatch": "owned_by", "upstream_all_pass": False, "reason": "Priya owns the service"},
            {"name": "description_key_fact_wrong", "kind": "key_fact", "edit": {"set": {"2": {"description": SERVICE_DESCRIPTION.replace("Primary interface through which users authenticate and", "Secondary reporting interface through which users")}}}, "bench2_mismatch": "description", "upstream_all_pass": False, "reason": "rest of the description correct, but it is no longer the primary authentication interface"},
            {"name": "problem_on_wrong_ci", "kind": "core", "edit": {"set": {"6": {"configuration_item": "CI_001"}}}, "bench2_mismatch": "configuration_item", "upstream_all_pass": False, "reason": "CI_001 costs 1200"},
            {"name": "problem_priority_high", "kind": "core", "edit": {"set": {"6": {"priority": "high"}}}, "bench2_mismatch": "priority", "upstream_all_pass": False, "reason": "highest levels: critical"},
            {"name": "statement_key_fact_wrong", "kind": "key_fact", "edit": {"set": {"6": {"problem_statement": PROBLEM_STATEMENT.replace("fully blocked", "intermittently slow")}}}, "bench2_mismatch": "problem_statement", "upstream_all_pass": False, "reason": "rest of the statement correct, but authentication is called intermittently slow instead of fully blocked"},
        ],
    },
]

#: Isolated before conversion. ``prompt_phrases`` / ``verifier_phrases`` are checked
#: against the upstream prompt and verifier queries when building.
ISOLATED = [
    {
        "upstream_task_id": "task_20251222_072805_599_99ba2325_ae6b0778",
        "candidate_item": 3,
        "reason": "prompt contradicts itself and the expected incident is not unique",
        "detail": (
            "The prompt asks to set Office Printing to repair in progress when more than three incidents are ongoing, and then "
            "to 'treat this situation as resolved from a service perspective ... even if some incident details are still pending'. "
            "The upstream verifier keeps repair_in_progress, i.e. picks one reading; adopting it means using the verifier to "
            "rewrite the prompt (6B rule), and resolving incidents with details pending would breach §3.3 (resolution_code and "
            "notes required). The owner alert must be attached to one incident, 'referencing the high priority nature': the count "
            "needs all orgs (9 ongoing Office Printing incidents), where two ongoing high-priority incidents exist (INC_026 in "
            "ORG_001, INC_068 in ORG_007); the verifier expects INC_026 without a stated criterion. The required reference to "
            "the high priority is free text, which Bench2's fuzzy rule would not discriminate."
        ),
        "prompt_phrases": ["update the service status to repair in progress", "treat this situation as resolved from a service perspective", "even if some incident details are still pending", "referencing the high priority nature"],
        "verifier_phrases": ['incident_id = "INC_026"', 'status = "repair_in_progress"'],
        "seed_facts": {"service": "SVC_041", "ongoing_incident_count_all_orgs": 9, "ongoing_high_priority_incidents": ["INC_026", "INC_068"]},
    },
    {
        "upstream_task_id": "task_20251222_140957_029_c5ca08eb_3c4271f0",
        "candidate_item": 4,
        "reason": "upstream expected state conflicts with mandatory policy and is not requested",
        "detail": (
            "The prompt only says Office Printing is being retired and its offerings are no longer supported. The upstream "
            "verifiers additionally require INC_015 and INC_064 (both in state new) to be closed with close notes. That is not "
            "asked for; new -> closed is not a valid transition (§3.3: new -> in_progress/on_hold/resolved; closed only from "
            "resolved), manual closure is permitted only to a Manager (Karen Watkins is admin), and the other seven ongoing "
            "Office Printing incidents are left untouched without a stated criterion. The §5.2 retirement cascades are PROPOSED "
            "and there is no successor service record."
        ),
        "prompt_phrases": ["Office Printing service is being retired", "no longer supported"],
        "verifier_phrases": ['incident_id="INC_015" AND status="closed"', 'incident_id="INC_064" AND status="closed"'],
        "seed_facts": {"closed_incidents_expected": {"INC_015": "new", "INC_064": "new"}, "operator_role": "admin"},
    },
    {
        "upstream_task_id": "task_20260114_073658_459_78a5b71e_cab5cc9f",
        "candidate_item": 6,
        "reason": "prompt and verifier contradict each other",
        "detail": (
            "The verbatim message in the prompt says 'customer-facing systems'; the upstream verifier compares the message for "
            "exact equality with 'customer facing systems'. Passing it means changing the operator's text to the verifier's "
            "(same class as 6B's '$64,00' vs 6400). The verifier also requires status = 'sent' for a notification that "
            "send_notification creates as 'queued' (6D-2 probe); writing 'sent' would assert a delivery outcome the system did "
            "not report (§1.3)."
        ),
        "prompt_phrases": ["customer-facing systems"],
        "verifier_phrases": ["customer facing systems", "status='sent'"],
        "seed_facts": {"incident": "INC_021", "assignee": "USER_008"},
    },
    {
        "upstream_task_id": "task_20251218_183819_272_99ba2325_dd71c332",
        "candidate_item": 8,
        "reason": "ambiguous number in the prompt resolved only by the verifier",
        "detail": (
            "The prompt sets the device cost to '$9,50', readable as 9.50 or 950; the upstream verifier requires 950 (seed value "
            "800). Choosing 950 uses the verifier to rewrite the prompt (same class as 6B's isolated 66392a82_74b17170, '$64,00' "
            "vs 6400). The problem record must also 'clearly document' the call drops and audio issues, a free-text requirement "
            "Bench2's fuzzy rule would not discriminate."
        ),
        "prompt_phrases": ["$9,50"],
        "verifier_phrases": ["cost = 950"],
        "seed_facts": {"configuration_item": "CI_092", "seed_cost": 800.0},
    },
]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _verifiers(row: dict, task_id: str) -> list[dict]:
    verifiers, seen = [], set()
    for v in audit._json(row["verifiers"]):
        config = v["validation_config"]
        if v["verifier_type"] != "database_state" or config.get("comparison_type", "equals") != "equals":
            raise SystemExit(f"{task_id}: unsupported verifier {v['name']}")
        query = config["query"].strip()
        if query in seen:
            continue
        seen.add(query)
        verifiers.append({"name": v["name"], "query": query, "expected_value": config["expected_value"]})
    return verifiers


def build(ref: Path) -> dict:
    oracle = {r["task_id"]: r for r in pq.read_table(ref / "hf-dataset" / "oracle" / "itsm-00000-of-00001.parquet").to_pylist()}
    seed_dir = ref / "itsm-seeds" / audit.SEED_DIR
    fields = audit.bench2_model_fields()
    tasks, isolated = [], []
    for spec in CONVERTED:
        row = oracle[spec["upstream_task_id"]]
        server = audit._json(audit._json(row["gym_servers_config"])[0])
        if server["seed_database_file"].split("/")[-1] != spec["upstream_seed"]:
            raise SystemExit(f"{spec['id']}: upstream seed mismatch")
        verifiers = _verifiers(row, spec["id"])
        unknown = set(spec["initial_pass_allowed"]) - {v["name"] for v in verifiers}
        if unknown:
            raise SystemExit(f"{spec['id']}: initial_pass_allowed names unknown verifiers {sorted(unknown)}")
        cases = {c["name"] for c in spec["validation_cases"]}
        for requirement in spec["core_requirements"]:
            if not requirement["negatives"] or set(requirement["negatives"]) - cases:
                raise SystemExit(f"{spec['id']}: core requirement {requirement['id']} lacks its negatives")
            if requirement["free_text"] and not any(
                c["kind"] == "key_fact" for c in spec["validation_cases"] if c["name"] in requirement["negatives"]
            ):
                raise SystemExit(f"{spec['id']}: free-text requirement {requirement['id']} lacks a key-fact negative")
        seed_path = seed_dir / spec["upstream_seed"]
        entry = {
            "id": spec["id"],
            "scenario": spec["scenario"],
            "evaluation_criteria": {"actions": spec["gold"], "nl_assertions": spec["nl_assertions"]},
            "initial_state_delta": spec["initial_state_delta"],
            "current_time": spec["current_time"],
            "org_ids": spec["org_ids"],
            "seed_db": spec["seed_db"],
            "tenancy": spec["tenancy"],
            "route": "faithful",
            "intended_split": "heldout",
            "candidate_item": spec["candidate_item"],
            "source": {
                **SOURCE,
                "upstream_task_id": spec["upstream_task_id"],
                "upstream_user_prompt_sha256": _sha256(row["user_prompt"].encode("utf-8")),
                "upstream_system_prompt_sha256": _sha256(row["system_prompt"].encode("utf-8")),
                "upstream_seed": spec["upstream_seed"],
                "upstream_seed_sha256": _sha256(seed_path.read_bytes()),
                "upstream_acting_user": audit._json(server.get("user_info") or {}),
                "upstream_verifiers_raw": len(audit._json(row["verifiers"])),
            },
            "upstream_verifiers": verifiers,
            "initial_pass_allowed": spec["initial_pass_allowed"],
            "core_requirements": spec["core_requirements"],
            "decision_check": spec.get("decision_check"),
            "conversion_notes": COMMON_NOTES + spec["notes"],
            "information_boundary": INFORMATION_BOUNDARY,
            "validation_cases": spec["validation_cases"],
            "approval": APPROVAL,
        }
        Task.model_validate(entry)
        unsliced = Task.model_validate({**copy.deepcopy(entry), "org_ids": None})
        with task_clock(unsliced):
            bench_tables = {
                t: dict(rows)
                for t, rows in build_task_environment(unsliced).tools.db.model_dump().items()
                if isinstance(rows, dict)
            }
        diff = audit.seed_diff(audit.seed_tables(audit.load_seed(seed_path, fields)), bench_tables)
        entry["source"]["initial_state_vs_upstream"] = {
            "compared": "unsliced Bench2 seed + delta vs whole upstream seed (timestamps ignored)",
            "delta_size_after_bench2_delta": diff["delta_size"],
            "renamed_ids_same_content": diff["renamed_ids_same_content"],
        }
        if diff["delta_size"]:
            raise SystemExit(f"{spec['id']}: Bench2 seed + delta differs from upstream: {diff}")
        if "isolate" in spec:
            isolated.append(
                {
                    "upstream_task_id": spec["upstream_task_id"],
                    "candidate_item": spec["candidate_item"],
                    **spec["isolate"],
                    "upstream_user_prompt_sha256": entry["source"]["upstream_user_prompt_sha256"],
                    "attempted_conversion": entry,
                }
            )
        else:
            tasks.append(entry)
    for item in ISOLATED:
        row = oracle[item["upstream_task_id"]]
        prompt = row["user_prompt"]
        queries = " ".join(v["query"] for v in _verifiers(row, item["upstream_task_id"]))
        missing = [p for p in item["prompt_phrases"] if p not in prompt] + [p for p in item["verifier_phrases"] if p not in queries]
        if missing:
            raise SystemExit(f"{item['upstream_task_id']}: evidence phrases not found upstream: {missing}")
        isolated.append({**item, "upstream_user_prompt_sha256": _sha256(prompt.encode("utf-8"))})
    return {
        "version": HELDOUT_V3_VERSION,
        "status": (
            "step 6D-3 held-out candidates for split v3; not part of Bench2 tasks.json or of any split; no v3 manifest "
            "exists; inclusion needs the user's acceptance"
        ),
        "source": SOURCE,
        "approval": APPROVAL,
        "tasks": tasks,
        "isolated": sorted(isolated, key=lambda item: item["candidate_item"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref-dir", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = json.dumps(build(args.ref_dir.resolve()), indent=2, ensure_ascii=False) + "\n"
    if args.check:
        same = DEFAULT_HELDOUT_V3_PATH.exists() and DEFAULT_HELDOUT_V3_PATH.read_text(encoding="utf-8") == text
        print("held-out candidate file is up to date" if same else f"out of date: {DEFAULT_HELDOUT_V3_PATH}")
        return 0 if same else 1
    DEFAULT_HELDOUT_V3_PATH.write_text(text, encoding="utf-8")
    print("wrote", DEFAULT_HELDOUT_V3_PATH.relative_to(Path(__file__).resolve().parents[2]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
