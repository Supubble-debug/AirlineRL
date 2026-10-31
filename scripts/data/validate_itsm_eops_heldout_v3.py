"""Validate the v3 held-out candidates (step 6D-3).

Usage (from the repo root):
    python scripts/data/validate_itsm_eops_heldout_v3.py [--out FILE]

Applies the rules fixed in docs/itsm_step6d3/README.md §1 to every converted
candidate: gold DB = 1 and a repeat replay with the same final DB hash; upstream
verifiers pass on gold and only the registered ones pass on the initial state;
every validation case scores Bench2 DB = 0 with the expected mismatch (key-fact
cases included); every core requirement has a negative and every free-text one a
key-fact negative; the cross-org decision quantity is the same in the task's
scope and in the unsliced DB; no private text at reset; a scripted episode ends
with agent_done and DB match. Plus the held-out gate over v2 and the accepted v3
train additions, and a re-check of the evidence behind every isolation.

No tokenizer, no token measurement: held-out candidates are not used for length,
budget or model configuration. No model or LLM is called.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

from enterprise_worlds.data_model.message import UserMessage
from enterprise_worlds.data_model.tasks import Task
from enterprise_worlds.domains.itsm import environment as itsm_environment
from enterprise_worlds.domains.itsm.data_model import ItsmDB
from enterprise_worlds.evaluator.text_match_strategy import TextMatchConfig

from agent_rl.data.itsm_eops_heldout_v3 import DEFAULT_HELDOUT_V3_PATH, heldout_gate_check
from agent_rl.data.itsm_eops_pilot import edited_actions, load_pilot, run_actions, run_upstream_verifiers
from agent_rl.envs.itsm_env import ItsmAction, ItsmEnv, ItsmEnvConfig
from agent_rl.envs.itsm_scoring import build_task_environment, replay_gold_actions, score_final_db, task_clock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_itsm_eops_batch2 import information_boundary  # noqa: E402  (step 6D-2)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "docs" / "itsm_step6d3" / "heldout_validation.json"
FUZZY = TextMatchConfig(strategy="fuzzy")
MAX_STEPS = 40
ACTIVE_EXCLUDED = ("resolved", "closed", "canceled")


class _ScriptedUser:
    replies = ("Hi, I need some changes made.", "Yes, go ahead.", "Thanks, that's everything.")

    def get_init_state(self) -> int:
        return 0

    def generate_next_message(self, agent_message, state: int):
        return UserMessage(content=self.replies[min(state, len(self.replies) - 1)]), state + 1


def scripted_episode(task) -> dict:
    with task_clock(task):
        environment = build_task_environment(task)
    gold = [ItsmAction.tool_call(a.name, copy.deepcopy(a.arguments)) for a in task.evaluation_criteria.actions]
    first_write = next(i for i, a in enumerate(gold) if environment.tools.tool_type(a.name).value != "read")
    actions = [*gold[:first_write], ItsmAction.message("Here is my plan. Shall I proceed?"), *gold[first_write:],
               ItsmAction.message("Done. Anything else?"), ItsmAction.done()]
    env = ItsmEnv(ItsmEnvConfig(task_id=task.id, text_match=FUZZY, max_steps=MAX_STEPS), user=_ScriptedUser(), task=task)
    env.reset()
    for action in actions:
        transition = env.step(action)
        if transition.terminated or transition.truncated:
            break
    return {"termination_reason": transition.termination_reason, "db_match": transition.reward_info.db_match, "step_count": transition.step_count}


def decision_quantity(task, check: dict) -> dict:
    def count(environment) -> int:
        return sum(
            1 for i in environment.tools.db.incident.values()
            if i.service == check["service"] and i.status not in ACTIVE_EXCLUDED
        )

    with task_clock(task):
        scoped = count(build_task_environment(task))
        unsliced = count(build_task_environment(task.model_copy(update={"org_ids": None})))
    return {
        **check,
        "in_scope": scoped,
        "unsliced": unsliced,
        "ok": scoped == unsliced == check["expected"] and (scoped > check["threshold_more_than"]),
    }


def validate_entry(entry: dict) -> dict:
    task = Task.model_validate(entry)
    gold_env = replay_gold_actions(task)
    score = score_final_db(task, gold_env, text_match=FUZZY)
    repeat_hash = replay_gold_actions(task).get_db_hash()
    with task_clock(task):
        initial_env = build_task_environment(task)
    cases = []
    for case in entry["validation_cases"]:
        env = run_actions(task, edited_actions(task, case["edit"]))
        case_score = score_final_db(task, env, text_match=FUZZY)
        upstream = run_upstream_verifiers(env.tools.db, entry["upstream_verifiers"])
        upstream_all = all(r["passed"] for r in upstream)
        found = any(case["bench2_mismatch"] in m for m in case_score.mismatches)
        cases.append(
            {
                "name": case["name"],
                "kind": case["kind"],
                "reason": case["reason"],
                "bench2_db_reward": case_score.db_reward,
                "bench2_mismatches": list(case_score.mismatches),
                "expected_mismatch_found": found,
                "upstream_all_pass": upstream_all,
                "upstream_failed": [r["name"] for r in upstream if not r["passed"]],
                "as_expected": case_score.db_reward == 0.0 and found and upstream_all == case["upstream_all_pass"],
            }
        )
    by_name = {c["name"]: c for c in cases}
    requirements = []
    for requirement in entry["core_requirements"]:
        negatives = [by_name[n] for n in requirement["negatives"]]
        requirements.append(
            {
                **requirement,
                "discriminated": all(n["bench2_db_reward"] == 0.0 for n in negatives),
                "undiscriminated_negatives": [n["name"] for n in negatives if n["bench2_db_reward"] != 0.0],
            }
        )
    upstream_gold = run_upstream_verifiers(gold_env.tools.db, entry["upstream_verifiers"])
    upstream_initial = run_upstream_verifiers(initial_env.tools.db, entry["upstream_verifiers"])
    decision = decision_quantity(task, entry["decision_check"]) if entry.get("decision_check") else None
    episode = scripted_episode(task)
    checks = {
        "bench2_gold_db_1": score.db_reward == 1.0,
        "repeat_replay_same_final_db": repeat_hash == gold_env.get_db_hash(),
        "upstream_all_pass_on_gold": all(r["passed"] for r in upstream_gold),
        "upstream_not_all_pass_on_initial": not all(r["passed"] for r in upstream_initial),
        "initial_passes_exactly_the_registered_ones": sorted(r["name"] for r in upstream_initial if r["passed"]) == sorted(entry["initial_pass_allowed"]),
        "every_core_requirement_discriminated": all(r["discriminated"] for r in requirements),
        "validation_cases_as_expected": all(c["as_expected"] for c in cases),
        "decision_quantity_same_in_scope_and_unsliced": decision is None or decision["ok"],
        "no_private_text_visible_at_reset": not information_boundary(task, entry)["leaked"],
        "scripted_episode_ends_with_db_match": episode["db_match"] and episode["termination_reason"] == "agent_done",
    }
    return {
        "candidate_item": entry["candidate_item"],
        "upstream_task_id": entry["source"]["upstream_task_id"],
        "org_ids": entry["org_ids"],
        "gold_actions": len(task.evaluation_criteria.actions),
        "bench2_gold_db_reward": score.db_reward,
        "upstream_verifiers_on_gold": upstream_gold,
        "upstream_verifiers_on_initial_state": upstream_initial,
        "initial_pass_allowed": entry["initial_pass_allowed"],
        "core_requirements": requirements,
        "validation_cases": cases,
        "decision_check": decision,
        "scripted_episode": episode,
        "checks": checks,
        "accepted": all(checks.values()),
    }


def isolated_seed_facts(item: dict) -> dict:
    """Re-derive the seed facts recorded for candidates isolated before conversion."""
    msp = ItsmDB.load(itsm_environment.ITSM_TASKS_PATH.parent / "msp_db.json")
    facts, item_no = item["seed_facts"], item["candidate_item"]
    if item_no == 3:
        ongoing = [i for i in msp.incident.values() if i.service == facts["service"] and i.status not in ACTIVE_EXCLUDED]
        observed = {"ongoing_incident_count_all_orgs": len(ongoing), "ongoing_high_priority_incidents": sorted(i.incident_id for i in ongoing if i.priority == "high")}
    elif item_no == 4:
        observed = {
            "closed_incidents_expected": {k: msp.incident[k].status for k in facts["closed_incidents_expected"]},
            "operator_role": msp.users["USER_039"].role,
        }
    elif item_no == 6:
        single = ItsmDB.load(itsm_environment.ITSM_TASKS_PATH.parent / "single_tenant_db.json")
        observed = {"incident": facts["incident"], "assignee": single.incident[facts["incident"]].assigned_to}
    elif item_no == 8:
        observed = {"configuration_item": facts["configuration_item"], "seed_cost": msp.configuration_item[facts["configuration_item"]].cost}
    else:
        raise ValueError(f"no seed-fact check for candidate {item_no}")
    return {"recorded": facts, "observed": {k: observed[k] for k in facts if k in observed} | {k: facts[k] for k in facts if k not in observed}, "holds": all(observed.get(k, facts[k]) == facts[k] for k in facts)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    data = load_pilot(DEFAULT_HELDOUT_V3_PATH)
    results = {entry["id"]: validate_entry(entry) for entry in data["tasks"]}
    attempted = {item["attempted_conversion"]["id"]: validate_entry(item["attempted_conversion"]) for item in data["isolated"] if "attempted_conversion" in item}
    seed_facts = {item["candidate_item"]: isolated_seed_facts(item) for item in data["isolated"] if "attempted_conversion" not in item}
    # Audit over every converted candidate, then the gate as v3 would hold it (accepted only).
    gate_all = heldout_gate_check([*data["tasks"], *(i["attempted_conversion"] for i in data["isolated"] if "attempted_conversion" in i)])
    gate = heldout_gate_check(data["tasks"])

    for tid, result in results.items():
        result["checks"]["heldout_gate"] = gate["tasks"][tid]["gate_ok"]
        result["accepted"] = all(result["checks"].values())
    result = {
        "version": data["version"],
        "caveat": (
            "Bench2 scoring is full-DB fuzzy (gold content-token recall >= 0.5 for free text) with D8; upstream verifiers run "
            "on the same final DB exported to SQLite. Scripted episodes replay gold calls with fixed user replies; no token "
            "measurement is made on held-out candidates. Passing does not make a task equivalent to EnterpriseOps-Gym."
        ),
        "tasks": results,
        "attempted_conversions_isolated": attempted,
        "isolated_before_conversion_seed_facts": seed_facts,
        "heldout_gate_accepted_only": gate,
        "heldout_gate_all_converted": gate_all,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {**{k: {"accepted": v["accepted"], "failed": [c for c, ok in v["checks"].items() if not ok]} for k, v in results.items()},
               **{k: {"isolated_after_conversion": True, "failed": [c for c, ok in v["checks"].items() if not ok], "undiscriminated": [r["id"] for r in v["core_requirements"] if not r["discriminated"]]} for k, v in attempted.items()}}
    print(json.dumps(summary, indent=1))
    print(json.dumps({t: r["checks"] for t, r in gate_all["tasks"].items()}, indent=1))
    print(json.dumps({k: v["holds"] for k, v in seed_facts.items()}))
    return 0 if all(v["accepted"] for v in results.values()) and all(v["holds"] for v in seed_facts.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
