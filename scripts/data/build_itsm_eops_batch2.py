"""Build the EnterpriseOps-Gym ITSM second migration batch (step 6D-2).

Usage (from the repo root):
    python scripts/data/build_itsm_eops_batch2.py --ref-dir DIR [--check]

Scope: the seven tasks the user approved for 6D-2 validation on 2026-10-30
(6D-1 batch items 1-4 and 6-8), all on the faithful route. The hand-written
conversion of each task lives in ``CONVERTED``; a task isolated before
conversion is in ``ISOLATED``, and a converted task that failed validation
carries an ``isolate`` record and is written to ``isolated`` together with its
attempted conversion, so validate_itsm_eops_batch2.py can re-check the evidence. As in step 6B, the
upstream verifiers, prompt/seed checksums and source version are read from the
isolated reference directory DIR (parsed, never executed), and the Bench2 seed +
delta must reproduce the upstream initial state (timestamps ignored).

These tasks are org-scoped (``org_ids``). The initial-state check compares the
UNSLICED Bench2 seed + delta with the whole upstream seed; slicing is a
deterministic function of the DB, so equal inputs give equal slices. None of
the upstream verifiers here pins the ID of a record the task creates, so the
slice does not change what they check.

Writes configs/data/itsm_eops_batch2_v1.json; ``--check`` fails if the committed
file differs.
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

from agent_rl.data.itsm_eops_batch2 import BATCH2_VERSION, DEFAULT_BATCH2_PATH
from agent_rl.envs.itsm_scoring import build_task_environment, task_clock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_eopsgym_itsm as audit  # noqa: E402  (our own step 6A script)
from build_itsm_eops_pilot import INFORMATION_BOUNDARY, KAREN, MARCUS, SOURCE, _a  # noqa: E402  (step 6B)

JOANNE = {"user_id": "USER_004", "first_name": "Joanne", "last_name": "Simpson", "email": "joanne.simpson@servicenow.com", "role": "manager"}
#: Restores the upstream MSP seed db_1765893060767 on top of Bench2 msp_db.json (as in step 6B).
MSP_MAIN_DELTA = {
    "change_request_mapping": {"CRM_017": {"delete": True}},
    "problem": {"PRB_011": {"delete": True}},
    "change": {"CHG_017": {"set": {"status": "scheduled"}}},
}
APPROVAL = {
    "date": "2026-10-30",
    "decision": "本轮仅批准第1–4、6–8题进入6D-2迁移验证，不保证最终全部纳入。……按既定逐题标准提交结果，失败则隔离。不为凑数放宽标准",
}

COMMON_NOTES = [
    "Upstream single-turn instruction rewritten as a multi-turn conversation: the operator states the request and reveals facts from known_info when asked; simulator_guidance fixes the answers to clarifications and cascades.",
    "The agent sees Bench2's task-invariant policy.md, not the upstream per-task system prompt (sha256 recorded in source).",
    "Org-scoped like Bench2 tasks (org_ids): MSP tasks here are provider-internal (ORG_001; foreign-key closure pulls in referenced rows of other orgs), the single-tenant task is ORG_001. The initial-state check is done on the unsliced DB; no upstream verifier of these tasks pins a created record's ID.",
    "current_time is 09:00 on the date in the upstream task_id.",
    "Free-text values the policy leaves to the operator (notification subject/message, offering short descriptions, record wording) are given verbatim by the operator in known_info.",
    "Scoring is Bench2's full-DB comparison (stricter than upstream): any unrequested write, or a REQUIRED policy cascade left out, gives DB=0 even where every upstream verifier passes. NL assertions are new and are not used for the training reward (decision D1). Not equivalent to the EnterpriseOps-Gym evaluation.",
]

KB_BODY = (
    "E-Commerce Website Service Restoration Procedure\n"
    "Document Version: 1.0 | Effective Date: December 2025\n"
    "Author: Joanne Simpson (USER_004)\n"
    "The E-Commerce Website service (SVC_007) is a business-critical service that depends on the primary database server (CI_001) for transaction processing and data availability, along with redundant web application servers to ensure continuous access.\n"
    "During the database upgrade, the environment was enhanced to improve stability and reliability, the database server serial number was updated to DB-2025-PROD-001, and post-upgrade validation confirmed data integrity and system readiness.\n"
    "For database-related service disruptions, restoration involves validating database availability, confirming application connectivity, reviewing system logs, verifying dependent configuration items, and assessing performance indicators. If required, a controlled failover to a standby database is performed following approved recovery procedures.\n"
    "Restoration activities included system validation, database migration and optimization, CMDB updates, service status restoration, and stakeholder notification. Preventive measures include automated health monitoring, periodic capacity reviews, maintained recovery runbooks, scheduled failover drills, and proactive hardware lifecycle management.\n"
    "Escalations are handled by the service owner (User_001) and the Infrastructure Team, with 24/7 support available at servicedesk@techcorp.com."
)
ESCALATION_TEMPLATE = (
    "Incident {number} ('{summary}'), created on {date}, has been escalated as top-priority work because it "
    "has remained unresolved for too long. Please handle it ahead of other tasks and drive it to resolution "
    "as quickly as possible."
)
CALLER_UPDATE = "Status update on the SAP database outage (INC-000001): the team is still working on it and resolution progress is being tracked."
REMINDER = "Reminder: please keep the status of the SAP database outage (INC-000001) current and report resolution progress."

CONVERTED = [
    {
        "id": "itsmbench_eops_99ba2325_887f5580",
        "batch_item": 1,
        "upstream_task_id": "task_20251219_081545_939_99ba2325_887f5580",
        "upstream_seed": "db_1765893060767_9rwtfbbp7.sql",
        "seed_db": "msp_db.json",
        "org_ids": ["ORG_001"],
        "tenancy": {"model": "msp_provider_internal"},
        "current_time": "2025-12-19T09:00:00",
        "initial_state_delta": MSP_MAIN_DELTA,
        "scenario": {
            "persona": {
                "identity": KAREN,
                "personality": "Efficient and audit-minded; accepts a clear policy explanation.",
                "role_description": "ITSM administrator, ServiceNow (provider, ORG_001)",
                "known_info": {
                    "requester_name": "Karen Watkins",
                    "context": "wrapping up a device refresh for the ACME headquarters team",
                    "retire_serial": "SPR-AGT-01 (no longer planned for reuse)",
                    "incident": "the ongoing 'MacBook Kernel Panic' incident",
                    "link_as_affected_cis": "SPR-AGT-01 and the currently active PRT-ACME-01",
                    "purpose": "track the hardware replacement history on the incident for audit and follow-up",
                },
            },
            "task_description": (
                "You are Karen Watkins, ITSM administrator. You are wrapping up a device refresh for the ACME "
                "headquarters team. You want the spare laptop with serial SPR-AGT-01 marked as retired in the "
                "CMDB because it will not be reused, and then both SPR-AGT-01 and the currently active "
                "PRT-ACME-01 linked to the ongoing 'MacBook Kernel Panic' incident as affected configuration "
                "items, so the hardware replacement history is tracked for audit."
            ),
            "simulator_guidance": (
                "Open by asking to retire SPR-AGT-01 and to link SPR-AGT-01 and PRT-ACME-01 to the 'MacBook Kernel "
                "Panic' incident as affected CIs. Give the serials and the incident name when asked. If the agent "
                "explains that a retired CI cannot be linked to an active incident, accept that and ask it to link "
                "only the printer; do not insist and do not suggest workarounds (no linking before retiring, no "
                "un-retiring). Do not ask for any other change. Confirm the agent's plan otherwise. End once the "
                "changes are confirmed."
            ),
        },
        "gold": [
            _a("get_user_using_name", first_name="Karen", last_name="Watkins"),
            _a("find_configuration_item_by_serial_number", serial_number="SPR-AGT-01"),
            _a("find_configuration_item_by_serial_number", serial_number="PRT-ACME-01"),
            _a("list_incidents", short_description="MacBook Kernel Panic"),
            _a("update_configuration_item", configuration_item_id="CI_041", status="retired"),
            _a("link_affected_ci_to_incident", configuration_item="CI_096", incident_id="INC_026"),
        ],
        "nl_assertions": [
            "The assistant verifies the operator's identity before making any change.",
            "The configuration item with serial SPR-AGT-01 (CI_041) is set to retired.",
            "PRT-ACME-01 (CI_096) is linked to the 'MacBook Kernel Panic' incident (INC_026) as an affected configuration item.",
            "The assistant does not link SPR-AGT-01 to the incident and explains that retired configuration items may not be linked to active incidents.",
            "The incident's own configuration item and the other records are left unchanged.",
        ],
        "notes": [
            "Policy §5.1: only CIs in state in_use or maintenance may be linked to active incidents; retired/disposed CIs may not. The upstream verifier 'Avoid Linking Asset 1 to incident' expects that link to be absent, which agrees with the policy.",
            "§5.1 retire cascades (PROPOSED, find active incidents / affected-CI rows referencing the CI): no incident or incident_affected_cis row references CI_041, so none apply.",
            "CI_041 goes in_stock -> retired; §5.1 notes that steps may be skipped depending on the procurement path. CI_041 is not linkable in in_stock either, so the order of operations does not create a second valid outcome.",
        ],
        "initial_pass_allowed": {
            "Avoid Linking Asset 1 to incident": "absence constraint (CI_041 not linked to INC_026); true before any action and made discriminative by validation case retired_ci_also_linked",
        },
        "validation_cases": [
            {"name": "retired_ci_also_linked", "edit": {"extra": [_a("link_affected_ci_to_incident", configuration_item="CI_041", incident_id="INC_026")]}, "bench2_mismatch": "incident_affected_cis", "upstream_all_pass": False, "reason": "§5.1 forbids linking a retired CI to an active incident"},
            {"name": "ci_set_to_disposed", "edit": {"set": {"4": {"status": "disposed"}}}, "bench2_mismatch": "configuration_item/CI_041 status", "upstream_all_pass": False, "reason": "the operator asked for retired"},
            {"name": "printer_not_linked", "edit": {"drop": [5]}, "bench2_mismatch": "incident_affected_cis", "upstream_all_pass": False, "reason": "the printer link is requested"},
            {"name": "printer_set_as_primary_ci", "edit": {"drop": [5], "extra": [_a("update_incident", incident_id="INC_026", configuration_item="CI_096")]}, "bench2_mismatch": "incident/INC_026 configuration_item", "upstream_all_pass": False, "reason": "an affected-CI link is requested, not a change of the incident's own CI"},
        ],
    },
    {
        "id": "itsmbench_eops_01395ded_eff85dae",
        "batch_item": 3,
        "upstream_task_id": "task_20251223_064932_736_01395ded_eff85dae",
        "upstream_seed": "db_1765893060767_9rwtfbbp7.sql",
        "seed_db": "msp_db.json",
        "org_ids": ["ORG_001"],
        "tenancy": {"model": "msp_provider_internal"},
        "current_time": "2025-12-23T09:00:00",
        "initial_state_delta": MSP_MAIN_DELTA,
        "scenario": {
            "persona": {
                "identity": KAREN,
                "personality": "Concerned about communication gaps during a major outage; precise about wording.",
                "role_description": "ITSM administrator, ServiceNow (provider, ORG_001)",
                "known_info": {
                    "requester_name": "Karen Watkins",
                    "incident": "the major SAP outage incident (MAJOR:SAP Database Unreachable)",
                    "queued_rule": "any notification for that incident still in status queued was never sent: set it to failed; leave the others as they are",
                    "caller_notification": "type update",
                    "caller_subject": "INC-000001 status update",
                    "caller_message": CALLER_UPDATE,
                    "reminder_recipients": "the person the incident is assigned to and the manager of its assignment group",
                    "reminder_type": "reminder",
                    "reminder_subject": "Reminder: INC-000001 status and resolution progress",
                    "reminder_message": REMINDER,
                },
            },
            "task_description": (
                "You are Karen Watkins, ITSM administrator. You suspect notifications for the major SAP outage "
                "incident are stuck in the queue. Any notification for it that is still queued was never sent and "
                "should be set to failed. Then the incident's caller should get an update notification, and the "
                "assigned person and the manager of the incident's assignment group should each get a reminder "
                "about the current status and resolution progress, using the exact texts you have."
            ),
            "simulator_guidance": (
                "Open by saying some notifications for the major SAP outage incident may be stuck in the queue: "
                "queued ones were never sent and should be marked failed; then the caller should get an update and "
                "the assigned person and the group's manager a reminder about status and resolution progress. Give "
                "the exact subjects and messages from known_info when asked (the same reminder text for both "
                "reminder recipients). If the agent points out that the caller's account is inactive, say to send "
                "the update anyway. Notifications that were delivered or opened stay unchanged. Confirm the agent's "
                "plan otherwise. End once everything is confirmed."
            ),
        },
        "gold": [
            _a("get_user_using_name", first_name="Karen", last_name="Watkins"),
            _a("list_incidents", short_description="MAJOR:SAP Database Unreachable"),
            _a("find_notifications", incident_id="INC_001", status="queued"),
            _a("update_notification", notification_id="NOTIF_003", status="failed"),
            _a("get_user", user_id="USER_001"),
            _a("get_user", user_id="USER_059"),
            _a("list_user_groups", group_id="GROUP_008"),
            _a("get_user", user_id="USER_002"),
            _a("send_notification", incident_id="INC_001", email="thomas.green@servicenow.com", type="update", subject="INC-000001 status update", message=CALLER_UPDATE),
            _a("send_notification", incident_id="INC_001", email="gregory.richards36@servicenow.com", type="reminder", subject="Reminder: INC-000001 status and resolution progress", message=REMINDER),
            _a("send_notification", incident_id="INC_001", email="christina.oliver@carsonllc.com", type="reminder", subject="Reminder: INC-000001 status and resolution progress", message=REMINDER),
        ],
        "nl_assertions": [
            "The assistant verifies the operator's identity before making any change.",
            "The only queued notification of the major SAP outage incident (NOTIF_003) is set to failed; delivered or opened notifications are not changed.",
            "The incident's caller (Thomas Green) receives an update notification with the operator's subject and message.",
            "The assigned person (Gregory Richards) and the assignment group's manager (Christina Oliver) each receive a reminder notification with the operator's subject and message.",
        ],
        "notes": [
            "'major SAP outage' resolves to INC_001 (MAJOR:SAP Database Unreachable), the only incident marked MAJOR in scope; NOTIF_003 is its only queued notification.",
            "Recipients come from data: caller USER_001, assignee USER_059, GROUP_008 manager USER_002 (Christina Oliver, ORG_002, pulled into the provider scope through the group's manager reference).",
            "The caller account USER_001 is inactive in both the upstream and the Bench2 seed. §8.1 requires active users for assigned_to/caller_id and live actors; a notification recipient is not one, and the operator explicitly asks for the update (simulator guidance covers the case).",
            "Message texts are the operator's (§1.2 ASK); they contain 'status' and 'progress' as the upstream LIKE predicates require.",
        ],
        "initial_pass_allowed": {},
        "validation_cases": [
            {"name": "queued_left_queued", "edit": {"drop": [3]}, "bench2_mismatch": "notification/NOTIF_003 status", "upstream_all_pass": False, "reason": "the queued notification must be marked failed"},
            {"name": "delivered_also_failed", "edit": {"extra": [_a("update_notification", notification_id="NOTIF_002", status="failed")]}, "bench2_mismatch": "notification/NOTIF_002 status", "upstream_all_pass": True, "reason": "only never-sent (queued) notifications change; upstream does not check NOTIF_002"},
            {"name": "caller_sent_reminder", "edit": {"set": {"8": {"type": "reminder"}}}, "bench2_mismatch": "type", "upstream_all_pass": False, "reason": "the caller gets an update notification"},
            {"name": "group_manager_skipped", "edit": {"drop": [10]}, "bench2_mismatch": "notification", "upstream_all_pass": False, "reason": "the group's manager must be reminded"},
        ],
    },
    {
        "id": "itsmbench_eops_78a5b71e_a4fb07a9",
        "batch_item": 4,
        "isolate": {
            "reason": "core requirement not discriminated by Bench2 DB scoring",
            "detail": (
                "Validation case server_named_in_description_only (status review, short description unchanged, server "
                "named in description) scores Bench2 DB = 1: description is a free-text field the gold does not change, "
                "so D8 skips it, and the unchanged short description still passes the fuzzy (token-recall) comparison "
                "against the gold text, which only appends a clause. Recording SRV-LNX-4421, the point of the task, is "
                "therefore not enforced by the training reward. Not fixed by re-wording the operator's text after the "
                "failure (that would tune the task to the scorer); any redesign needs the user's decision."
            ),
        },
        "upstream_task_id": "task_20260114_081229_036_78a5b71e_a4fb07a9",
        "upstream_seed": "db_1765893060767_9rwtfbbp7.sql",
        "seed_db": "msp_db.json",
        "org_ids": ["ORG_001"],
        "tenancy": {"model": "msp_provider_internal"},
        "current_time": "2026-01-14T09:00:00",
        "initial_state_delta": MSP_MAIN_DELTA,
        "scenario": {
            "persona": {
                "identity": KAREN,
                "personality": "Wants the record finalised without delay; firm that no new change is created.",
                "role_description": "ITSM administrator, ServiceNow (provider, ORG_001)",
                "known_info": {
                    "requester_name": "Karen Watkins",
                    "change_number": "CHG0000001",
                    "outcome": "implementation completed successfully today, within the approved maintenance window",
                    "extra_ci": "the same remediation was also applied to Application Server SRV-LNX-4421, which was not originally listed",
                    "short_description": "Apply Oracle Patch 19.4.2 to SAP DB (also applied to SRV-LNX-4421)",
                    "target_state": "post-implementation review (not closed yet)",
                    "constraint": "do not create any new change records; keep the change's configuration item as it is",
                },
            },
            "task_description": (
                "You are Karen Watkins, ITSM administrator. The implementation under change CHG0000001 completed "
                "successfully within the approved maintenance window; the same remediation was also applied to "
                "Application Server SRV-LNX-4421, which was not originally listed. You want the existing change "
                "record to reflect that server and moved to the post-implementation state so it can be finalised, "
                "without creating any new change record."
            ),
            "simulator_guidance": (
                "Open with the CHG0000001 situation. If asked how to record the extra server, give the exact short "
                "description from known_info and say the change's configuration item stays as it is (a change holds "
                "only one). If asked whether to close the change, say no, it should go to review. If asked about the "
                "maintenance window, say the work finished inside the approved window today. Do not ask for new "
                "change records or other edits. Confirm the agent's plan otherwise. End once the change is updated."
            ),
        },
        "gold": [
            _a("get_user_using_name", first_name="Karen", last_name="Watkins"),
            _a("find_change_by_number", number="CHG0000001"),
            _a("find_configuration_item_by_serial_number", serial_number="SRV-LNX-4421"),
            _a("update_change", change_id="CHG_001", status="implement"),
            _a("update_change", change_id="CHG_001", status="review", short_description="Apply Oracle Patch 19.4.2 to SAP DB (also applied to SRV-LNX-4421)"),
        ],
        "nl_assertions": [
            "The assistant verifies the operator's identity before making any change.",
            "CHG0000001 (CHG_001) moves from scheduled through implement to review and is not closed.",
            "The change's short description becomes the operator's text naming SRV-LNX-4421; its configuration item stays CI_001.",
            "No new change record is created.",
        ],
        "notes": [
            "§4.2 transitions: scheduled -> implement -> review (implement -> review is the post-implementation step). No close_code/close_notes because the change is not closed.",
            "§4.2 'implement only within the planned window': the Bench2 change has no window fields; the operator states the work finished inside the approved window and current_time is that day.",
            "Tool limit, stated rather than worked around: a Bench2 change holds a single configuration_item, so SRV-LNX-4421 (CI_002) cannot be linked; the operator's short description names it (upstream checks short_description LIKE '%SRV-LNX-4421%'). No CI link is claimed.",
            "update_change does not enforce transitions; a single scheduled -> review update reaches the same final DB and scores the same.",
        ],
        "initial_pass_allowed": {},
        "validation_cases": [
            {"name": "change_closed", "edit": {"set": {"4": {"status": "closed"}}}, "bench2_mismatch": "change/CHG_001 status", "upstream_all_pass": False, "reason": "the change goes to review, not closed"},
            {"name": "left_in_implement", "edit": {"drop": [4]}, "bench2_mismatch": "change/CHG_001", "upstream_all_pass": False, "reason": "the change must reach review with the server named"},
            {"name": "configuration_item_overwritten", "edit": {"set": {"4": {"configuration_item": "CI_002"}}}, "bench2_mismatch": "change/CHG_001 configuration_item", "upstream_all_pass": True, "reason": "the change's CI stays CI_001; upstream does not check it"},
            {"name": "server_named_in_description_only", "edit": {"drop": [4], "extra": [_a("update_change", change_id="CHG_001", status="review", description="Applying critical vendor patch to fix memory leak ORA-04030. Also applied to SRV-LNX-4421.")]}, "bench2_mismatch": "change/CHG_001", "upstream_all_pass": False, "reason": "the operator's wording goes into the short description"},
        ],
    },
    {
        "id": "itsmbench_eops_46ca2862_56a4f834",
        "batch_item": 6,
        "upstream_task_id": "task_20260113_143610_437_46ca2862_56a4f834",
        "upstream_seed": "db_1765893060767_9rwtfbbp7.sql",
        "seed_db": "msp_db.json",
        "org_ids": ["ORG_001"],
        "tenancy": {"model": "msp_provider_internal"},
        "current_time": "2026-01-13T09:00:00",
        "initial_state_delta": MSP_MAIN_DELTA,
        "scenario": {
            "persona": {
                "identity": KAREN,
                "personality": "Portfolio-minded; gives exact names and descriptions when asked.",
                "role_description": "ITSM administrator, ServiceNow (provider, ORG_001)",
                "known_info": {
                    "requester_name": "Karen Watkins",
                    "service": "Enterprise Email",
                    "new_offerings": "Mobile Email Access; Email Archiving and Compliance; Advanced Email Security",
                    "short_description_mobile_email_access": "Email access from mobile devices",
                    "short_description_email_archiving_and_compliance": "Mailbox archiving and compliance retention",
                    "short_description_advanced_email_security": "Advanced threat protection for email",
                    "attributes": "production, ready, technology-management, business-critical",
                    "owner": "the owner of the Enterprise Email service",
                    "template": "Outlook Not Syncing: service Enterprise Email, offering Mobile Email Access",
                },
            },
            "task_description": (
                "You are Karen Watkins, ITSM administrator. You want the Enterprise Email offering portfolio "
                "expanded with three production-ready, business-critical technology-management offerings, each "
                "owned by the Enterprise Email service owner: Mobile Email Access, Email Archiving and Compliance, "
                "and Advanced Email Security. The 'Outlook Not Syncing' incident template should be associated with "
                "the Enterprise Email service and the new Mobile Email Access offering."
            ),
            "simulator_guidance": (
                "Open by asking to review the Enterprise Email offerings and add the three new ones, and to point the "
                "'Outlook Not Syncing' template at Enterprise Email and the new Mobile Email Access offering. The "
                "owner is whoever owns Enterprise Email (confirm if the agent names Joanne Simpson). Give the short "
                "descriptions from known_info when asked; there is no longer description. Existing offerings stay "
                "unchanged. Confirm the agent's plan otherwise. End once everything is confirmed."
            ),
        },
        "gold": [
            _a("get_user_using_name", first_name="Karen", last_name="Watkins"),
            _a("find_service_by_name", name="Enterprise Email"),
            _a("find_service_offerings", owned_by="USER_004"),
            _a("get_incident_template_by_name", name="Outlook Not Syncing"),
            _a("register_new_service_offering", name="Mobile Email Access", owned_by="USER_004", parent="SVC_001", short_description="Email access from mobile devices", used_for="production", status="ready", service_classification="technology-management", business_criticality="critical"),
            _a("register_new_service_offering", name="Email Archiving and Compliance", owned_by="USER_004", parent="SVC_001", short_description="Mailbox archiving and compliance retention", used_for="production", status="ready", service_classification="technology-management", business_criticality="critical"),
            _a("register_new_service_offering", name="Advanced Email Security", owned_by="USER_004", parent="SVC_001", short_description="Advanced threat protection for email", used_for="production", status="ready", service_classification="technology-management", business_criticality="critical"),
            _a("update_incident_template", incident_template_id="TMPL_004", change_request_values={"service_offering": "SVCOFF_061"}),
        ],
        "nl_assertions": [
            "The assistant verifies the operator's identity before making any change.",
            "Three offerings are registered under Enterprise Email (SVC_001), owned by its owner Joanne Simpson: Mobile Email Access, Email Archiving and Compliance, Advanced Email Security; each production, ready, technology-management, critical.",
            "The 'Outlook Not Syncing' template keeps service Enterprise Email and now uses the Mobile Email Access offering.",
            "Existing offerings (Enterprise Email - Gold, Email - Silver) are not changed.",
        ],
        "notes": [
            "§5.2 REQUIRED: offerings inherit business_criticality and environment from the parent unless overridden; Enterprise Email is critical/production, which matches the request. Classification is not inherited; the operator asks for technology-management.",
            "Owner from data: Enterprise Email is owned by USER_004 (Joanne Simpson). Existing offerings of SVC_001: SVCOFF_001 (Gold), SVCOFF_018 (Silver); none for mobile access.",
            "The template already references SVC_001, so only service_offering changes (via change_request_values). The upstream verifier on the template's service passes before any action (precondition).",
            "New offering IDs in this scope: SVCOFF_061/062/063 in registration order.",
        ],
        "initial_pass_allowed": {
            "verify_outlook_template_service_updated": "precondition already true in the seed (TMPL_004.service = SVC_001); the template's offering verifier discriminates",
        },
        "validation_cases": [
            {"name": "offering_under_wrong_service", "edit": {"set": {"5": {"parent": "SVC_002"}}}, "bench2_mismatch": "service_offering/SVCOFF_062 business_service", "upstream_all_pass": True, "reason": "all three belong to Enterprise Email; upstream does not check the parent"},
            {"name": "criticality_not_inherited", "edit": {"set": {"6": {"business_criticality": "somewhat-critical"}}}, "bench2_mismatch": "service_offering/SVCOFF_063 business_criticality", "upstream_all_pass": False, "reason": "§5.2 inherit the parent's criticality (critical), as requested"},
            {"name": "template_on_archiving_offering", "edit": {"set": {"7": {"change_request_values": {"service_offering": "SVCOFF_062"}}}}, "bench2_mismatch": "incident_template/TMPL_004 service_offering", "upstream_all_pass": False, "reason": "the template goes to Mobile Email Access"},
            {"name": "one_offering_missing", "edit": {"drop": [6]}, "bench2_mismatch": "service_offering", "upstream_all_pass": False, "reason": "all three offerings are requested"},
        ],
    },
    {
        "id": "itsmbench_eops_99ba2325_7915481a",
        "batch_item": 7,
        "upstream_task_id": "task_20260114_174427_042_99ba2325_7915481a",
        "upstream_seed": "db_1765893060767_9rwtfbbp7.sql",
        "seed_db": "msp_db.json",
        "org_ids": ["ORG_001"],
        "tenancy": {"model": "msp_provider_internal"},
        "current_time": "2026-01-14T09:00:00",
        "initial_state_delta": MSP_MAIN_DELTA,
        "scenario": {
            "persona": {
                "identity": JOANNE,
                "personality": "Casual and a little unsure of the technical details; answers clarifying questions plainly.",
                "role_description": "Manager, ServiceNow (provider, ORG_001)",
                "known_info": {
                    "requester_name": "Joanne Simpson",
                    "service": "E-Commerce Website (it went down and is back now)",
                    "service_status": "restored: mark it operational",
                    "old_serial": "SRV-ORC-9981",
                    "new_serial": "DB-2025-PROD-001",
                    "ci_handling": "update the serial number on the existing database CI record; do not register a new CI",
                    "kb_title": "E-Commerce Service Restoration Procedure",
                    "kb_body": KB_BODY,
                    "kb_visibility": "visible to the public (external)",
                    "kb_state": "published right away",
                    "kb_owner": "Joanne Simpson",
                },
            },
            "task_description": (
                "You are Joanne Simpson, a manager. The 'E-Commerce Website' service went down and is back now. "
                "The database configuration item with serial SRV-ORC-9981 now carries the new hardware serial "
                "DB-2025-PROD-001. You want a knowledge article titled 'E-Commerce Service Restoration Procedure' "
                "with the body you have, owned by you, published and visible to the public; the database CI's "
                "serial updated; and the service marked as operational."
            ),
            "simulator_guidance": (
                "Open casually: ask for a write-up about the E-Commerce Website service that went down and seems to be "
                "back, mention the database CI with serial SRV-ORC-9981 was swapped for hardware with serial "
                "DB-2025-PROD-001, and that you are not sure of the details. If asked whether to register a new CI or "
                "update the existing record, say update the serial on the existing record. If asked whether the "
                "service is restored, say yes, mark it operational. Give the article title and full body from "
                "known_info when asked; it should be published and publicly visible (external), owned by you. Confirm "
                "the agent's plan otherwise. End once everything is confirmed."
            ),
        },
        "gold": [
            _a("get_user_using_name", first_name="Joanne", last_name="Simpson"),
            _a("find_service_by_name", name="E-Commerce Website"),
            _a("find_configuration_item_by_serial_number", serial_number="SRV-ORC-9981"),
            _a("update_configuration_item", configuration_item_id="CI_001", serial_number="DB-2025-PROD-001"),
            _a("update_service", service_id="SVC_007", status="operational"),
            _a("create_knowledge_article", title="E-Commerce Service Restoration Procedure", owner_id="USER_004", body=KB_BODY, state="published", visibility="external"),
        ],
        "nl_assertions": [
            "The assistant verifies the operator's identity before making any change.",
            "The serial number of the existing database CI (CI_001) changes from SRV-ORC-9981 to DB-2025-PROD-001; no new CI is registered.",
            "The E-Commerce Website service (SVC_007) is set to operational.",
            "A knowledge article 'E-Commerce Service Restoration Procedure' with the operator's body is created, owned by Joanne Simpson, published and external.",
        ],
        "notes": [
            "Two operator clarifications, neither contradicted by the upstream verifiers: update the existing CI's serial (not a new CI), and the service is restored (repair_in_progress -> operational).",
            "§7: the PROPOSED draft-and-review routing does not bind a Manager; Joanne Simpson (manager) publishes directly, external as requested ('visible to the public').",
            "The upstream prompt writes the body with literal '\\n' sequences; they are real line breaks here. The upstream verifiers do not check the body.",
        ],
        "initial_pass_allowed": {},
        "validation_cases": [
            {"name": "new_ci_registered", "edit": {"drop": [3], "extra": [_a("register_configuration_item", name="SAP-DB-01 (Replacement Hardware)", owner_id="USER_001", location_id="LOC_003", serial_number="DB-2025-PROD-001", status="in_use", cost=15000.0)]}, "bench2_mismatch": "configuration_item", "upstream_all_pass": False, "reason": "the operator asked to update the existing record"},
            {"name": "article_left_draft_internal", "edit": {"set": {"5": {"state": "draft", "visibility": "internal"}}}, "bench2_mismatch": "knowledge/KB_026", "upstream_all_pass": False, "reason": "the article is published and public"},
            {"name": "service_left_in_repair", "edit": {"drop": [4]}, "bench2_mismatch": "service/SVC_007 status", "upstream_all_pass": False, "reason": "the service is restored"},
            {"name": "serial_set_on_wrong_ci", "edit": {"set": {"3": {"configuration_item_id": "CI_002"}}}, "bench2_mismatch": "serial_number", "upstream_all_pass": False, "reason": "the swapped hardware is the database CI (CI_001)"},
        ],
    },
    {
        "id": "itsmbench_eops_97d7df7d_8fd400d1",
        "batch_item": 8,
        "isolate": {
            "reason": "shares 2 same-seed entities with a held-out task",
            "detail": (
                "The gold uses GROUP_001 (Level 1 Support Team) and INC_003, which also appear in the gold of held-out "
                "task itsmbench_single_012 (incident_handoff_with_child). The 6A train gate forbids >= 2 shared same-seed "
                "entities with a dev/held-out task, because training on them puts held-out records into training. "
                "The 6D-1 screening checked only the entities named in the upstream verifiers and missed this; the "
                "6D-2 check uses the gold arguments, as step 6B did."
            ),
        },
        "upstream_task_id": "task_20251218_063226_501_97d7df7d_8fd400d1",
        "upstream_seed": "db_1765301900121_3mwjj54xy.sql",
        "seed_db": "single_tenant_db.json",
        "org_ids": ["ORG_001"],
        "tenancy": {"model": "single_tenant"},
        "current_time": "2025-12-18T09:00:00",
        "initial_state_delta": None,
        "scenario": {
            "persona": {
                "identity": MARCUS,
                "personality": "Wants old tickets moving; clear that this is about alerts, not editing tickets.",
                "role_description": "IT administrator, TechCorp (ORG_001)",
                "known_info": {
                    "requester_name": "Marcus Thompson",
                    "group": "Level 1 Support Team",
                    "selection": "incidents of that group that are still open and were created before 20 February 2024",
                    "recipients": "the users those incidents are currently assigned to",
                    "notification_type": "alert",
                    "subject": "Priority Escalation for Aging Incidents",
                    "message_template": ESCALATION_TEMPLATE,
                    "template_fields": "number = the incident number, summary = its short description, date = its creation date as YYYY-MM-DD",
                    "incident_records": "do not edit the incidents themselves (no priority changes); alerts only",
                },
            },
            "task_description": (
                "You are Marcus Thompson, IT administrator at TechCorp. Open incidents of the 'Level 1 Support Team' "
                "created before 20 February 2024 should be treated with top-priority attention. You want an alert sent "
                "to each user those incidents are currently assigned to, with your subject and message template "
                "(incident number, summary and creation date filled in). The incidents themselves are not edited."
            ),
            "simulator_guidance": (
                "Open by asking to review the open incidents of the Level 1 Support Team created before 20 February "
                "2024 and alert their current assignees that these are now top-priority work. Give the subject and the "
                "message template from known_info when asked, and explain the fields. If the agent asks whether to "
                "change the incidents' priority or other fields, say no, only send the alerts. Confirm the agent's plan "
                "otherwise. End once the alerts are sent."
            ),
        },
        "gold": [
            _a("get_user_using_name", first_name="Marcus", last_name="Thompson"),
            _a("find_group_by_name", name="Level 1 Support Team"),
            _a("get_incidents_assigned_to", assignment_group="GROUP_001"),
            _a("get_user", user_id="USER_003"),
            _a("get_user", user_id="USER_006"),
            _a("send_notification", incident_id="INC_003", email="carlos.rodriguez@techcorp.com", type="alert", subject="Priority Escalation for Aging Incidents", message=ESCALATION_TEMPLATE.format(number="INC0000003", summary="Printer connectivity issue", date="2024-02-19")),
            _a("send_notification", incident_id="INC_004", email="elena.petrov@techcorp.com", type="alert", subject="Priority Escalation for Aging Incidents", message=ESCALATION_TEMPLATE.format(number="INC0000004", summary="Network connectivity issues", date="2024-02-18")),
        ],
        "nl_assertions": [
            "The assistant verifies the operator's identity before making any change.",
            "Alerts go only to the assignees of the Level 1 Support Team incidents that are open and were created before 2024-02-20: INC_003 (Carlos Rodriguez) and INC_004 (Elena Petrov).",
            "Each alert uses the operator's subject and the template filled with that incident's number, summary and creation date.",
            "No incident record is modified.",
        ],
        "notes": [
            "Data gives a unique set: GROUP_001's open incidents are INC_003 (2024-02-19), INC_004 (2024-02-18), INC_014 (2024-02-23) and INC_016 (2024-02-25); only the first two precede the cutoff.",
            "'treat them as top-priority work' is not a priority change: the operator says not to edit the incidents (a priority change would also trigger the §6 SLA reset).",
            "The template contains every term the upstream LIKE predicates require (escalated, unresolved, summary, ahead of other tasks, quick, top, priority, possible, day of month).",
        ],
        "initial_pass_allowed": {},
        "validation_cases": [
            {"name": "alert_for_incident_after_cutoff", "edit": {"extra": [_a("send_notification", incident_id="INC_014", email="carlos.rodriguez@techcorp.com", type="alert", subject="Priority Escalation for Aging Incidents", message=ESCALATION_TEMPLATE.format(number="INC0000014", summary="Monitor display flickering", date="2024-02-23"))]}, "bench2_mismatch": "notification", "upstream_all_pass": True, "reason": "INC_014 was created after the cutoff; upstream does not penalise extra notifications"},
            {"name": "priority_raised", "edit": {"extra": [_a("update_incident", incident_id="INC_003", priority="critical")]}, "bench2_mismatch": "incident/INC_003 priority", "upstream_all_pass": True, "reason": "the operator said not to edit the incidents"},
            {"name": "caller_alerted_instead_of_assignee", "edit": {"set": {"5": {"email": "benjamin.chen@techcorp.com"}}}, "bench2_mismatch": "email", "upstream_all_pass": False, "reason": "alerts go to the current assignees"},
            {"name": "notification_type_report", "edit": {"set": {"6": {"type": "report"}}}, "bench2_mismatch": "type", "upstream_all_pass": False, "reason": "the operator asked for alerts"},
        ],
    },
]

ISOLATED = [
    {
        "upstream_task_id": "task_20251223_135402_072_c5ca08eb_67009578",
        "batch_item": 2,
        "reason": "ambiguous REQUIRED cascade",
        "detail": (
            "Raising PRB_009's impact to high also raises the linked incident INC_061 (ORG_013) from moderate to high "
            "priority. Policy §6 REQUIRES 'Incident priority changes -> reset/swap/cancel the incident_sla row per the "
            "change'. INC_061 has no incident_sla row. In the MSP seed, client incidents use the provider's (ORG_001) "
            "SLA definitions (e.g. INC_049 in ORG_013 carries SLA_002), so two readings are defensible: do nothing "
            "because there is no row to reset, or start an SLA for the new priority, where high has two ORG_001 "
            "definitions (SLA_003 response, SLA_004 resolution). Full-DB scoring would give one reasonable behaviour 0. "
            "Releasing it needs a fixed MSP SLA rule, which the user declined to set (D6D-1). The 6D-1 screening's "
            "statement that ORG_013 'has no SLA definition, so no SLA action' was wrong and is corrected."
        ),
    },
]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build(ref: Path) -> dict:
    oracle = {r["task_id"]: r for r in pq.read_table(ref / "hf-dataset" / "oracle" / "itsm-00000-of-00001.parquet").to_pylist()}
    seed_dir = ref / "itsm-seeds" / audit.SEED_DIR
    fields = audit.bench2_model_fields()
    tasks, isolated_after_validation = [], []
    for spec in CONVERTED:
        row = oracle[spec["upstream_task_id"]]
        server = audit._json(audit._json(row["gym_servers_config"])[0])
        if server["seed_database_file"].split("/")[-1] != spec["upstream_seed"]:
            raise SystemExit(f"{spec['id']}: upstream seed mismatch")
        verifiers, seen = [], set()
        for v in audit._json(row["verifiers"]):
            config = v["validation_config"]
            if v["verifier_type"] != "database_state" or config.get("comparison_type", "equals") != "equals":
                raise SystemExit(f"{spec['id']}: unsupported verifier {v['name']}")
            query = config["query"].strip()
            if query in seen:
                continue
            seen.add(query)
            verifiers.append({"name": v["name"], "query": query, "expected_value": config["expected_value"]})
        unknown = set(spec["initial_pass_allowed"]) - {v["name"] for v in verifiers}
        if unknown:
            raise SystemExit(f"{spec['id']}: initial_pass_allowed names unknown verifiers {sorted(unknown)}")
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
            "batch_item": spec["batch_item"],
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
        upstream_tables = audit.seed_tables(audit.load_seed(seed_path, fields))
        diff = audit.seed_diff(upstream_tables, bench_tables)
        entry["source"]["initial_state_vs_upstream"] = {
            "compared": "unsliced Bench2 seed + delta vs whole upstream seed (timestamps ignored)",
            "delta_size_after_bench2_delta": diff["delta_size"],
            "renamed_ids_same_content": diff["renamed_ids_same_content"],
        }
        if diff["delta_size"]:
            raise SystemExit(f"{spec['id']}: Bench2 seed + delta differs from upstream: {diff}")
        if "isolate" in spec:
            # Converted, then isolated by validation; kept so the evidence can be re-checked.
            isolated_after_validation.append(
                {
                    "upstream_task_id": spec["upstream_task_id"],
                    "batch_item": spec["batch_item"],
                    **spec["isolate"],
                    "upstream_user_prompt_sha256": entry["source"]["upstream_user_prompt_sha256"],
                    "attempted_conversion": entry,
                }
            )
        else:
            tasks.append(entry)
    isolated = []
    for item in ISOLATED:
        row = oracle[item["upstream_task_id"]]
        isolated.append({**item, "upstream_user_prompt_sha256": _sha256(row["user_prompt"].encode("utf-8"))})
    return {
        "version": BATCH2_VERSION,
        "status": (
            "step 6D-2 validation batch; not part of Bench2 tasks.json or of any split; inclusion needs the "
            "user's acceptance"
        ),
        "source": SOURCE,
        "approval": APPROVAL,
        "tasks": tasks,
        "isolated": sorted(isolated + isolated_after_validation, key=lambda item: item["batch_item"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref-dir", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = json.dumps(build(args.ref_dir.resolve()), indent=2, ensure_ascii=False) + "\n"
    if args.check:
        same = DEFAULT_BATCH2_PATH.exists() and DEFAULT_BATCH2_PATH.read_text(encoding="utf-8") == text
        print("batch file is up to date" if same else f"out of date: {DEFAULT_BATCH2_PATH}")
        return 0 if same else 1
    DEFAULT_BATCH2_PATH.write_text(text, encoding="utf-8")
    print("wrote", DEFAULT_BATCH2_PATH.relative_to(Path(__file__).resolve().parents[2]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
