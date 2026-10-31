"""Build the EnterpriseOps-Gym ITSM migration pilot file (step 6B).

Usage (from the repo root):
    python scripts/data/build_itsm_eops_pilot.py --ref-dir DIR [--check]

DIR is the isolated reference directory used in step 6A. The hand-written
conversion of each pilot task lives in ``CONVERTED`` below; the upstream
verifiers, prompt/seed checksums and source version are read from DIR (parsed,
never executed). The script also checks that each Bench2 seed + delta reproduces
the upstream task's initial state (timestamps ignored). Writes
configs/data/itsm_eops_pilot_v1.json; ``--check`` fails if the committed file differs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

from enterprise_worlds.data_model.tasks import Task
from enterprise_worlds.domains.itsm.data_model import ItsmDB

from agent_rl.data.itsm_eops_pilot import DEFAULT_PILOT_PATH, PILOT_VERSION
from agent_rl.envs.itsm_scoring import build_task_environment, task_clock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_eopsgym_itsm as audit  # noqa: E402  (our own step 6A script)

SOURCE = {
    "repo": audit.UPSTREAM_REPO,
    "commit": audit.UPSTREAM_COMMIT,
    "hf_dataset": audit.HF_DATASET,
    "hf_revision": audit.HF_REVISION,
    "license": "Apache-2.0",
    "attribution": "Adapted from EnterpriseOps-Gym (ServiceNow), Apache-2.0; see Bench2/NOTICE",
    "mode_used": "oracle",
}

#: Visible to the policy / private to the environment and scoring (protocol §3).
INFORMATION_BOUNDARY = {
    "policy_visible": [
        "Bench2 policy.md and the 94 tool schemas (task-invariant)",
        "fixed greeting",
        "user simulator messages (known_info facts are revealed only when the user states them)",
        "real tool results and errors",
    ],
    "private": [
        "scenario.task_description, simulator_guidance, persona.known_info (until stated)",
        "evaluation_criteria.actions (gold) and nl_assertions",
        "initial_state_delta, seed_db, current_time, org scope",
        "upstream task_id, prompt, system prompt and verifiers",
        "validation_cases",
    ],
}

COMMON_NOTES = [
    "Upstream single-turn instruction rewritten as a multi-turn conversation: the operator states the request and reveals facts from known_info when asked; simulator_guidance fixes the answers to PROPOSED/ASK cascades.",
    "The agent sees Bench2's task-invariant policy.md, not the upstream per-task system prompt (sha256 recorded in source).",
    "No org slicing (org_ids null): the whole seed is loaded as upstream does, so sequential IDs of created records match the upstream verifiers.",
    "current_time is 09:00 on the date in the upstream task_id.",
    "Scoring is Bench2's full-DB comparison (stricter than upstream): any unrequested write, or a REQUIRED policy cascade left out, gives DB=0 even where every upstream verifier passes. NL assertions are new and are not used for the training reward (decision D1).",
]

MARCUS = {"user_id": "USER_001", "first_name": "Marcus", "last_name": "Thompson", "email": "marcus.thompson@techcorp.com", "role": "admin"}
KAREN = {"user_id": "USER_039", "first_name": "Karen", "last_name": "Watkins", "email": "karen.watkins@servicenow.com", "role": "admin"}


def _a(tool: str, /, **arguments) -> dict:
    return {"name": tool, "arguments": arguments}


CONVERTED = [
    {
        "id": "itsmbench_eops_46ca2862_b6ef60ca",
        "upstream_task_id": "task_20260113_182602_664_46ca2862_b6ef60ca",
        "upstream_seed": "db_1765301900121_3mwjj54xy.sql",
        "seed_db": "single_tenant_db.json",
        "tenancy": {"model": "single_tenant"},
        "current_time": "2026-01-13T09:00:00",
        "initial_state_delta": None,
        "scenario": {
            "persona": {
                "identity": MARCUS,
                "personality": "Brisk and practical. Answers questions directly and does not volunteer extra detail.",
                "role_description": "IT administrator, TechCorp (ORG_001)",
                "known_info": {
                    "requester_name": "Marcus Thompson",
                    "asset_owner": "Benjamin Chen",
                    "asset_name": "Benjamin Chen Laptop",
                    "serial_number": "LAPTOP-UK-229",
                    "cost": "92500",
                    "location": "TechCorp NYC Headquarters (the New York office)",
                    "asset_status": "in_use",
                    "hardware": "Intel i7, 16GB RAM, 512GB SSD, upgraded wireless",
                    "incident_caller": "Benjamin Chen",
                    "incident_short_description": "Software installation",
                    "incident_category": "software",
                    "incident_impact": "medium",
                    "incident_urgency": "medium",
                },
            },
            "task_description": (
                "You are Marcus Thompson, IT administrator at TechCorp. Benjamin Chen was issued a new "
                "corporate laptop (serial LAPTOP-UK-229, about 92,500 rupees, Intel i7 / 16GB RAM / 512GB "
                "SSD) in the Q1 hardware refresh. He uses it at the New York office (TechCorp NYC "
                "Headquarters) and it is in use. You want it registered in the asset records under the "
                "name 'Benjamin Chen Laptop', owned by Benjamin, and an incident opened for Benjamin about "
                "software installation on that laptop with medium impact and medium urgency."
            ),
            "simulator_guidance": (
                "Open by asking to add Benjamin's new laptop to the asset records and to open an incident "
                "for him. Give serial number, cost (92500), location and status when asked; if asked for an "
                "asset name, say 'Benjamin Chen Laptop'. For the incident: Benjamin is the caller, it is "
                "about software installation on that laptop, category software, medium impact, medium "
                "urgency. If asked for a support group, say there is none to record. If the agent proposes "
                "sending Benjamin a report notification about the new incident, decline (you will tell him "
                "yourself). Confirm the agent's plan otherwise. End once both records are confirmed."
            ),
        },
        "gold": [
            _a("get_user_using_name", first_name="Marcus", last_name="Thompson"),
            _a("get_user_using_name", first_name="Benjamin", last_name="Chen"),
            _a("find_locations", name="TechCorp NYC Headquarters"),
            _a("register_configuration_item", name="Benjamin Chen Laptop", owner_id="USER_005", location_id="LOC_001", serial_number="LAPTOP-UK-229", status="in_use", cost=92500.0),
            _a("retrieve_knowledge_articles", body="install"),
            _a("create_incident", caller_id="USER_005", short_description="Software installation", configuration_item="CI_005", channel="chat", category="software", impact="medium", urgency="medium", priority="moderate"),
            _a("find_sla_definitions", applies_to_priority="moderate"),
            _a("link_new_incident_sla", incident_id="INC_024", sla_def_id="SLA_004"),
        ],
        "nl_assertions": [
            "The assistant verifies the operator's identity before making any change.",
            "A configuration item with serial LAPTOP-UK-229 is registered, owned by Benjamin Chen, at TechCorp NYC Headquarters, in_use, cost 92500.",
            "A new incident is opened with Benjamin Chen as caller, linked to that laptop, medium impact and medium urgency, priority moderate, state new.",
            "The new incident receives an SLA record matching its moderate priority.",
            "No report notification is sent to Benjamin, because the operator declined it.",
        ],
        "notes": [
            "Added for Bench2: asset name (required by register_configuration_item, absent upstream) and incident category 'software' (from 'an incident for Software installation'); both are unconstrained by the upstream verifiers.",
            "Policy cascades in gold: channel defaults to chat (§3.1); SLA_004 is the only moderate-priority SLA of ORG_001 (§6, REQUIRED); the REQUIRED knowledge search finds no article about software installation, so no link (§7); the PROPOSED report notification is declined.",
            "'around ₹92,500' is recorded as 92500, as the upstream verifier requires; the cost field has no currency.",
        ],
        "validation_cases": [
            {"name": "wrong_priority", "edit": {"set": {"5": {"priority": "high"}}}, "bench2_mismatch": "incident/INC_024 priority", "upstream_all_pass": False, "reason": "priority must follow the medium x medium matrix cell (moderate)"},
            {"name": "sla_not_spawned", "edit": {"drop": [7]}, "bench2_mismatch": "incident_sla: missing", "upstream_all_pass": True, "reason": "Bench2 policy REQUIRES an SLA on creation; the upstream verifiers do not check it"},
            {"name": "laptop_registered_in_london", "edit": {"set": {"3": {"location_id": "LOC_002"}}}, "bench2_mismatch": "configuration_item/CI_005 location_id", "upstream_all_pass": False, "reason": "the laptop is used at the New York office, not Benjamin's home location"},
            {"name": "unconfirmed_report_notification", "edit": {"extra": [_a("send_notification", incident_id="INC_024", email="benjamin.chen@techcorp.com", type="report", subject="Incident INC-000024 opened", message="Your software installation incident has been opened.")]}, "bench2_mismatch": "notification: unexpected", "upstream_all_pass": True, "reason": "the operator declined the PROPOSED notification; upstream does not penalise extra writes"},
        ],
    },
    {
        "id": "itsmbench_eops_66392a82_52c8290e",
        "upstream_task_id": "task_20251218_124419_383_66392a82_52c8290e",
        "upstream_seed": "db_1765984530018_6f72jkf3p.sql",
        "seed_db": "single_tenant_db.json",
        "tenancy": {"model": "single_tenant"},
        "current_time": "2025-12-18T09:00:00",
        "initial_state_delta": {
            "configuration_item": {
                "CI_005": {
                    "create": {
                        "configuration_item_id": "CI_005",
                        "name": "Boardroom Video Hub",
                        "serial_number": "VC-HUB-LON-014",
                        "owner_id": "USER_006",
                        "location_id": "LOC_003",
                        "org_id": "ORG_002",
                        "status": "in_use",
                        "cost": 6000.0,
                        "created_on": "2024-01-10 14:00:00",
                        "updated_on": "2024-01-10 14:00:00",
                    }
                }
            }
        },
        "scenario": {
            "persona": {
                "identity": MARCUS,
                "personality": "Concerned but polite; gives details when asked.",
                "role_description": "IT administrator, TechCorp (ORG_001)",
                "known_info": {
                    "requester_name": "Marcus Thompson",
                    "device": "4K video conferencing hub",
                    "serial_number": "VC-HUB-LON-014",
                    "device_contact": "Elena Petrov",
                    "correct_cost": "4000",
                    "problem": "freezing and dropping video calls during critical client sessions in San Francisco",
                    "incident_caller": "Elena Petrov",
                    "incident_category": "hardware",
                    "incident_impact": "high",
                    "incident_urgency": "high",
                },
            },
            "task_description": (
                "You are Marcus Thompson, IT administrator at TechCorp. A 4K video conferencing hub (serial "
                "VC-HUB-LON-014) whose contact is Elena Petrov has been freezing and dropping video calls "
                "during critical client sessions in San Francisco. Its recorded cost is wrong: the correct "
                "cost is 4000 (pounds). You want the cost fixed and a high-impact, high-urgency incident "
                "opened describing the freezing and dropped calls, with Elena as the caller."
            ),
            "simulator_guidance": (
                "Open by reporting that the video conferencing hub keeps freezing and dropping calls and that "
                "its recorded cost is wrong. Give the serial number and the correct cost (4000) when asked. "
                "Elena Petrov is the caller. Impact high, urgency high, category hardware. If the agent "
                "proposes sending Elena a report notification, decline. Do not ask for any other change to "
                "the device. Confirm the agent's plan otherwise. End once both changes are confirmed."
            ),
        },
        "gold": [
            _a("get_user_using_name", first_name="Marcus", last_name="Thompson"),
            _a("find_configuration_item_by_serial_number", serial_number="VC-HUB-LON-014"),
            _a("update_configuration_item", configuration_item_id="CI_005", cost=4000.0),
            _a("get_user_using_name", first_name="Elena", last_name="Petrov"),
            _a("retrieve_knowledge_articles", body="video"),
            _a("create_incident", caller_id="USER_006", short_description="Video conferencing hub freezing and dropping calls", configuration_item="CI_005", channel="chat", category="hardware", impact="high", urgency="high", priority="critical"),
            _a("find_sla_definitions", applies_to_priority="critical"),
            _a("link_new_incident_sla", incident_id="INC_024", sla_def_id="SLA_001"),
        ],
        "nl_assertions": [
            "The assistant verifies the operator's identity before making any change.",
            "The cost of the hub with serial VC-HUB-LON-014 is corrected to 4000 and nothing else on that configuration item is changed.",
            "A new incident is opened with Elena Petrov as caller, linked to the hub, high impact, high urgency, priority critical, state new, describing the freezing and dropped calls.",
            "The new incident receives an SLA record matching its critical priority.",
            "No report notification is sent to Elena, because the operator declined it.",
        ],
        "notes": [
            "The upstream seed adds CI_005 (Boardroom Video Hub) to the single-tenant world; the delta creates that exact row. Upstream data places it in ORG_002 (Acme Corp HQ) while its owner/contact Elena is in ORG_001; kept as is.",
            "Added for Bench2: category 'hardware'. Policy cascades in gold: channel chat (§3.1); SLA_001 is the only critical SLA of ORG_001 (§6); the REQUIRED knowledge search finds no article about the hub (§7); the PROPOSED report notification is declined.",
            "'£4,000' is recorded as 4000, as the upstream verifier requires; the cost field has no currency.",
        ],
        "validation_cases": [
            {"name": "cost_not_corrected", "edit": {"drop": [2]}, "bench2_mismatch": "configuration_item/CI_005 cost", "upstream_all_pass": False, "reason": "the requested cost correction is missing"},
            {"name": "operator_as_caller", "edit": {"set": {"5": {"caller_id": "USER_001"}}}, "bench2_mismatch": "incident/INC_024 caller_id", "upstream_all_pass": False, "reason": "Elena, the device contact, is the caller"},
            {"name": "sla_for_wrong_priority", "edit": {"set": {"7": {"sla_def_id": "SLA_002"}}}, "bench2_mismatch": "sla_def_id", "upstream_all_pass": True, "reason": "the SLA must match the critical priority; upstream does not check SLAs"},
        ],
    },
    {
        "id": "itsmbench_eops_1d219d93_d1ba6c2d",
        "upstream_task_id": "task_20251223_101136_983_1d219d93_d1ba6c2d",
        "upstream_seed": "db_1765893060767_9rwtfbbp7.sql",
        "seed_db": "msp_db.json",
        "tenancy": {"model": "msp_provider_internal"},
        "current_time": "2025-12-23T09:00:00",
        "initial_state_delta": {
            "change_request_mapping": {"CRM_017": {"delete": True}},
            "problem": {"PRB_011": {"delete": True}},
            "change": {"CHG_017": {"set": {"status": "scheduled"}}},
        },
        "scenario": {
            "persona": {
                "identity": KAREN,
                "personality": "Organised and to the point; expects the agent to check before creating things.",
                "role_description": "ITSM administrator, ServiceNow (provider, ORG_001)",
                "known_info": {
                    "requester_name": "Karen Watkins",
                    "new_location_name": "Pune HQ",
                    "new_location_street": "Tech Park, Building 4",
                    "new_location_city": "Pune",
                    "new_location_country": "India",
                    "transfer_rule": "every in_stock configuration item at the other locations in India moves to Pune HQ",
                    "asset_1": "Cisco Meraki MX85, serial SN-MER-888, cost 2000, in_use",
                    "asset_2": "Dell Precision 5000, serial SN-DELL-999, cost 3500, in_stock",
                    "asset_owner": "Elizabeth Gregory",
                    "elizabeth_change": "promote to manager; her location record stays as it is",
                    "kb_title": "Pune HQ IT Setup",
                    "kb_body": "only the serial numbers of the two new assets: SN-MER-888, SN-DELL-999",
                    "kb_owner": "Karen Watkins",
                },
            },
            "task_description": (
                "You are Karen Watkins, ITSM administrator. You are opening 'Pune HQ' at Tech Park, Building 4 "
                "(street), Pune, India; check whether the location exists and create it if missing. Move every "
                "in_stock configuration item at the other locations in India to Pune HQ. Register two new "
                "assets at Pune HQ owned by Elizabeth Gregory, who becomes the manager for the site: a Cisco "
                "Meraki MX85 (SN-MER-888, 2000, in_use) and a Dell Precision 5000 (SN-DELL-999, 3500, "
                "in_stock). Promote Elizabeth to manager. Finally create a knowledge article titled 'Pune HQ "
                "IT Setup', owned by you, that lists only the serial numbers of the two new assets."
            ),
            "simulator_guidance": (
                "Open with the Pune HQ setup request and give details when asked. If asked whether Elizabeth's "
                "own location should change, say no, only her role changes. If asked who owns the article, say "
                "yourself. Only in_stock items move; items in use stay where they are. Confirm the agent's plan "
                "otherwise. End once everything is confirmed."
            ),
        },
        "gold": [
            _a("get_user_using_name", first_name="Karen", last_name="Watkins"),
            _a("find_locations", country="India"),
            _a("add_location", name="Pune HQ", city="Pune", country="India", street="Tech Park, Building 4"),
            _a("find_configuration_items", location_id="LOC_004", status="in_stock"),
            _a("update_configuration_item", configuration_item_id="CI_035", location_id="LOC_024"),
            _a("get_user_using_name", first_name="Elizabeth", last_name="Gregory"),
            _a("update_user_details", user_id="USER_042", role="manager"),
            _a("register_configuration_item", name="Cisco Meraki MX85", owner_id="USER_042", location_id="LOC_024", serial_number="SN-MER-888", status="in_use", cost=2000.0),
            _a("register_configuration_item", name="Dell Precision 5000", owner_id="USER_042", location_id="LOC_024", serial_number="SN-DELL-999", status="in_stock", cost=3500.0),
            _a("create_knowledge_article", title="Pune HQ IT Setup", owner_id="USER_039", body="SN-MER-888, SN-DELL-999"),
        ],
        "nl_assertions": [
            "The assistant checks whether Pune HQ already exists before creating it.",
            "Pune HQ is created at Tech Park, Building 4, Pune, India.",
            "The only in_stock configuration item at the other Indian location (Dell Latitude - Christine Torres, CI_035) is moved to Pune HQ; in-use items in India are not moved.",
            "Two configuration items are registered at Pune HQ owned by Elizabeth Gregory: Cisco Meraki MX85 (SN-MER-888, 2000, in_use) and Dell Precision 5000 (SN-DELL-999, 3500, in_stock).",
            "Elizabeth Gregory's role is changed to manager and her own location is not changed.",
            "A knowledge article 'Pune HQ IT Setup' is created listing only SN-MER-888 and SN-DELL-999.",
        ],
        "notes": [
            "Delta restores the upstream seed db_1765893060767 on top of Bench2 msp_db.json (= upstream canonical db_1766420030407): delete CRM_017 and PRB_011, CHG_017 status back to 'scheduled'. None of these records is touched by the task.",
            "Tenancy label 'msp_provider_internal': MSP world, but every record involved is in the provider org ORG_001 (Bench2's MSP tasks are provider/client).",
            "Added for Bench2: 'Tech Park, Building 4' is the street field; Elizabeth's location stays unchanged; the article owner is Karen. All three are unconstrained by the upstream verifiers.",
        ],
        "validation_cases": [
            {"name": "dell_left_in_bangalore", "edit": {"set": {"8": {"location_id": "LOC_004"}}}, "bench2_mismatch": "location_id", "upstream_all_pass": False, "reason": "both new assets belong at Pune HQ"},
            {"name": "role_not_changed", "edit": {"drop": [6]}, "bench2_mismatch": "users/USER_042 role", "upstream_all_pass": False, "reason": "Elizabeth must become manager"},
            {"name": "in_stock_item_not_moved", "edit": {"drop": [4]}, "bench2_mismatch": "configuration_item/CI_035 location_id", "upstream_all_pass": False, "reason": "CI_035 is in_stock in India and must move"},
            {"name": "in_use_item_also_moved", "edit": {"extra": [_a("update_configuration_item", configuration_item_id="CI_023", location_id="LOC_024")]}, "bench2_mismatch": "configuration_item/CI_023 location_id", "upstream_all_pass": True, "reason": "only in_stock items move; upstream does not check untouched rows"},
            {"name": "elizabeth_relocated", "edit": {"extra": [_a("update_user_details", user_id="USER_042", location_id="LOC_024")]}, "bench2_mismatch": "users/USER_042 location_id", "upstream_all_pass": True, "reason": "the operator said only her role changes; upstream does not check her location"},
        ],
    },
]

ISOLATED = [
    {
        "upstream_task_id": "task_20251219_144431_663_97d7df7d_a52988e5",
        "reason": "ambiguous REQUIRED cascade",
        "detail": "Bench2 policy §3.1/§7 REQUIRE a knowledge search at incident creation and linking 'any matching' article. For an Email Service slowdown, KB_002 'Email Client Configuration Guide' (published, ORG_001) can defensibly be linked or not; full-DB scoring would make one of two reasonable behaviours score 0. Not resolvable by user-provided information because the cascade is REQUIRED.",
    },
    {
        "upstream_task_id": "task_20251218_104259_233_66392a82_74b17170",
        "reason": "upstream prompt contradicts its verifier",
        "detail": "The prompt gives the storage node cost as '$64,00'; the verifier requires 6400. Choosing 6400 for the multi-turn user would adopt the verifier's reading over the prompt text.",
    },
    {
        "upstream_task_id": "task_20260114_085016_654_e6427839_ca9119ec",
        "reason": "upstream expected state conflicts with a REQUIRED Bench2 rule",
        "detail": "The verifier requires GROUP_001 'Level 1 Support Team' to be dissolved (active=0), but GROUP_001 has active incidents (INC_003, INC_004, INC_014, INC_016) and Bench2 policy §12.3 REQUIRES blocking the dissolution until they are reassigned; the upstream task asks for no reassignment.",
    },
    {
        "upstream_task_id": "task_20260114_094327_107_46ca2862_796e7685",
        "reason": "ambiguous REQUIRED cascade",
        "detail": "Creating the child incident (priority moderate) REQUIRES spawning a matching SLA (§6). The MSP world has two moderate SLAs for ORG_001 (SLA_005 'Priority 3 - Response', SLA_006 'Priority 3 - Resolution') and the seed's existing incidents follow no single convention (mostly one row, either kind). The REQUIRED knowledge search may also match KB_006 'SAP Login \"Session Locked\"'.",
    },
    {
        "upstream_task_id": "task_20260116_081645_096_66392a82_fd509f97",
        "reason": "REQUIRED cascade impossible with Bench2 tools, plus ambiguous SLA",
        "detail": "Deactivating Travis Wood REQUIRES removing his user_role row (USER_011:ROLE_002:ORG_001) (§12.2); Bench2 has no tool that removes user_role rows. The new low-priority incident also has two candidate SLAs (SLA_007/SLA_008), and four PROPOSED offboarding cascades (INC_002, INC_093, CHG_010, PRB_006) would need scripted answers.",
    },
]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build(ref: Path) -> dict:
    oracle = {r["task_id"]: r for r in pq.read_table(ref / "hf-dataset" / "oracle" / "itsm-00000-of-00001.parquet").to_pylist()}
    seed_dir = ref / "itsm-seeds" / audit.SEED_DIR
    fields = audit.bench2_model_fields()
    tasks = []
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
        seed_path = seed_dir / spec["upstream_seed"]
        entry = {
            "id": spec["id"],
            "scenario": spec["scenario"],
            "evaluation_criteria": {"actions": spec["gold"], "nl_assertions": spec["nl_assertions"]},
            "initial_state_delta": spec["initial_state_delta"],
            "current_time": spec["current_time"],
            "org_ids": None,
            "seed_db": spec["seed_db"],
            "tenancy": spec["tenancy"],
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
            "conversion_notes": COMMON_NOTES + spec["notes"],
            "information_boundary": INFORMATION_BOUNDARY,
            "validation_cases": spec["validation_cases"],
        }
        task = Task.model_validate(entry)
        with task_clock(task):
            bench_tables = {
                t: {k: r for k, r in rows.items()}
                for t, rows in build_task_environment(task).tools.db.model_dump().items()
                if isinstance(rows, dict)
            }
        upstream_tables = audit.seed_tables(audit.load_seed(seed_path, fields))
        diff = audit.seed_diff(upstream_tables, bench_tables)
        entry["source"]["initial_state_vs_upstream"] = {
            "delta_size_after_bench2_delta": diff["delta_size"],
            "renamed_ids_same_content": diff["renamed_ids_same_content"],
        }
        if diff["delta_size"]:
            raise SystemExit(f"{spec['id']}: Bench2 seed + delta differs from upstream: {diff}")
        tasks.append(entry)
    isolated = []
    for item in ISOLATED:
        row = oracle[item["upstream_task_id"]]
        isolated.append({**item, "upstream_user_prompt_sha256": _sha256(row["user_prompt"].encode("utf-8"))})
    return {
        "version": PILOT_VERSION,
        "status": "pilot; not part of Bench2 tasks.json or of split v1; inclusion in a train split v2 awaits acceptance",
        "source": SOURCE,
        "tasks": tasks,
        "isolated": isolated,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref-dir", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = json.dumps(build(args.ref_dir.resolve()), indent=2, ensure_ascii=False) + "\n"
    if args.check:
        same = DEFAULT_PILOT_PATH.exists() and DEFAULT_PILOT_PATH.read_text(encoding="utf-8") == text
        print("pilot file is up to date" if same else f"out of date: {DEFAULT_PILOT_PATH}")
        return 0 if same else 1
    DEFAULT_PILOT_PATH.write_text(text, encoding="utf-8")
    print("wrote", DEFAULT_PILOT_PATH.relative_to(Path(__file__).resolve().parents[2]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
