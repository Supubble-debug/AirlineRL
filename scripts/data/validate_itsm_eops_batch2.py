"""Validate and measure the EnterpriseOps-Gym ITSM second batch (step 6D-2).

Usage (from the repo root):
    python scripts/data/validate_itsm_eops_batch2.py --tokenizer-dir DIR [--out FILE]

Per task, with the standard fixed in 6D-1 (docs/itsm_step6d/README.md §2.3):
  1. strict gold replay under the task clock, Bench2 DB (fuzzy) = 1, and a second
     replay with the same final DB hash;
  2. every deduplicated upstream verifier passes on the gold final DB; on the
     untouched initial state not all pass, and the ones that do are exactly those
     registered in ``initial_pass_allowed`` with a reason;
  3. every validation case: Bench2 DB = 0 with the expected mismatch, and the
     recorded upstream all-pass verdict;
  4. information boundary: the reset observation and public info contain no
     private text;
  5. a scripted ItsmEnv episode (gold tool calls, one plan and one recap message,
     fixed user replies; not a policy distribution, no user simulator) ends with
     agent_done and DB match, measured with the pinned Qwen3-8B tokenizer.
Plus the family/overlap re-check over split v2's 56 tasks and the batch.
A task that fails any check is reported as not accepted; nothing is relaxed.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

from transformers import AutoTokenizer

from enterprise_worlds.data_model.message import UserMessage
from enterprise_worlds.evaluator.text_match_strategy import TextMatchConfig
from enterprise_worlds.gym.gym_agent import DONE_TOOL_SCHEMA

from agent_rl.data.itsm_eops_batch2 import DEFAULT_BATCH2_PATH, family_overlap_check_v2
from enterprise_worlds.data_model.tasks import Task

from agent_rl.data.itsm_eops_pilot import edited_actions, load_pilot, pilot_tasks, run_actions, run_upstream_verifiers
from agent_rl.envs.itsm_env import ItsmAction, ItsmEnv, ItsmEnvConfig
from agent_rl.envs.itsm_scoring import build_task_environment, replay_gold_actions, score_final_db, task_clock
from agent_rl.prompts.action_prompt import OBSERVATION_PROMPT, build_action_messages

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_itsm_tokens as measure  # noqa: E402  (our own step 6 script)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "docs" / "itsm_step6d2" / "batch_validation.json"
FUZZY = TextMatchConfig(strategy="fuzzy")
MAX_STEPS = 40  # configs/itsm/budget_v1.yaml
OPENINGS = {
    "itsmbench_eops_99ba2325_887f5580": "Hi, Karen Watkins here. Please retire SPR-AGT-01 and link it and PRT-ACME-01 to the MacBook Kernel Panic incident as affected CIs.",
    "itsmbench_eops_01395ded_eff85dae": "Hi, Karen Watkins here. I think notifications for the major SAP outage are stuck in the queue; mark the queued ones failed and send the status notifications.",
    "itsmbench_eops_78a5b71e_a4fb07a9": "Hi, Karen Watkins here. CHG0000001 was implemented successfully and also covered SRV-LNX-4421; please update the change and move it on.",
    "itsmbench_eops_46ca2862_56a4f834": "Hi, Karen Watkins here. Please add three new offerings to Enterprise Email and point the Outlook Not Syncing template at Mobile Email Access.",
    "itsmbench_eops_99ba2325_7915481a": "Hey, Joanne here. Can you write up the E-Commerce Website outage? The database CI SRV-ORC-9981 was swapped for DB-2025-PROD-001, I think.",
    "itsmbench_eops_97d7df7d_8fd400d1": "Hi, Marcus Thompson here. Please alert the assignees of the Level 1 Support Team's open incidents created before 20 February 2024.",
}
CONFIRM = "Yes, go ahead."
CLOSING = "Thanks, that's everything."


def pilot_tasks_from_entries(entries: list[dict]) -> dict[str, Task]:
    return {entry["id"]: Task.model_validate(entry) for entry in entries}


class _ScriptedUser:
    def __init__(self, opening: str) -> None:
        self.replies = [opening, CONFIRM, CLOSING]

    def get_init_state(self) -> int:
        return 0

    def generate_next_message(self, agent_message, state: int):
        return UserMessage(content=self.replies[min(state, len(self.replies) - 1)]), state + 1


def _scripted_actions(task, environment) -> list[ItsmAction]:
    gold = [ItsmAction.tool_call(a.name, copy.deepcopy(a.arguments)) for a in task.evaluation_criteria.actions]
    first_write = next(i for i, a in enumerate(gold) if environment.tools.tool_type(a.name).value != "read")
    plan = ItsmAction.message(
        "Here is my plan: "
        + "; ".join(f"{a.name}({', '.join(f'{k}={v}' for k, v in a.arguments.items())})" for a in gold[first_write:])
        + ". Shall I proceed?"
    )
    recap = ItsmAction.message("Done. Summary: " + ", ".join(a.name for a in gold[first_write:]) + ". Anything else?")
    return [*gold[:first_write], plan, *gold[first_write:], recap, ItsmAction.done()]


def measure_episode(renderer, task, policy: str) -> dict:
    config = ItsmEnvConfig(task_id=task.id, text_match=FUZZY, max_steps=MAX_STEPS)
    started = time.perf_counter()
    env = ItsmEnv(config, user=_ScriptedUser(OPENINGS[task.id]), episode_id=f"{task.id}:measure", task=task)
    reset = env.reset()
    reset_s = time.perf_counter() - started
    initial = renderer.count(renderer.text(build_action_messages(domain_policy=policy, observation=reset.observation), generation=True))
    with task_clock(task):
        actions = _scripted_actions(task, build_task_environment(task))
    generated, environment, tool_seconds = [], [], []
    for action in actions:
        if action.kind == "tool_call":
            assistant = measure._assistant_tool_call(action.name, action.arguments)
        elif action.kind == "message":
            assistant = {"role": "assistant", "content": action.text}
        else:
            assistant = measure._assistant_tool_call("done", {})
        tick = time.perf_counter()
        transition = env.step(action)
        if action.kind == "tool_call":
            tool_seconds.append(time.perf_counter() - tick)
        generated.append(renderer.generated(assistant))
        if transition.terminated or transition.truncated:
            break
        environment.append(renderer.environment_turn(OBSERVATION_PROMPT.format(observation=transition.observation)))
    response = sum(generated) + sum(environment)
    return {
        "termination_reason": transition.termination_reason,
        "db_match": transition.reward_info.db_match,
        "agent_turns": transition.agent_turns,
        "step_count": transition.step_count,
        "tool_calls": len(tool_seconds),
        "initial_prompt_tokens": initial,
        "generated_tokens": sum(generated),
        "generated_tokens_max_action": max(generated),
        "environment_tokens": sum(environment),
        "environment_tokens_max_turn": max(environment),
        "response_tokens": response,
        "full_sequence_tokens": initial + response,
        "seconds": {"env_build_and_reset": round(reset_s, 4), "tool_calls_total": round(sum(tool_seconds), 4)},
    }


def information_boundary(task, entry) -> dict:
    class OneReply:
        def get_init_state(self):
            return 0

        def generate_next_message(self, agent_message, state):
            return UserMessage(content="Hi, I need some help."), state + 1

    reset = ItsmEnv(ItsmEnvConfig(task_id=task.id, text_match=FUZZY, max_steps=MAX_STEPS), user=OneReply(), task=task).reset()
    visible = reset.observation + json.dumps(reset.public_info)
    private = [
        task.scenario.task_description,
        task.scenario.simulator_guidance,
        task.id,
        entry["source"]["upstream_task_id"],
        "EnterpriseOps",
        *task.evaluation_criteria.nl_assertions,
        *(v["query"] for v in entry["upstream_verifiers"]),
        *entry["conversion_notes"],
    ]
    leaked = [text[:80] for text in private if text in visible]
    return {"private_texts_checked": len(private), "leaked": leaked}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    batch = load_pilot(DEFAULT_BATCH2_PATH)
    tasks = pilot_tasks(DEFAULT_BATCH2_PATH)
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_dir), trust_remote_code=False)
    first = next(iter(tasks.values()))
    with task_clock(first):
        environment = build_task_environment(first)
    renderer = measure.Renderer(tokenizer, environment.get_tool_schemas() + [copy.deepcopy(DONE_TOOL_SCHEMA)])
    policy = environment.get_policy()

    results = {}
    for entry in batch["tasks"]:
        task = tasks[entry["id"]]
        tick = time.perf_counter()
        gold_env = replay_gold_actions(task)
        score = score_final_db(task, gold_env, text_match=FUZZY)
        scoring_s = time.perf_counter() - tick
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
                    "reason": case["reason"],
                    "bench2_db_reward": case_score.db_reward,
                    "bench2_mismatches": list(case_score.mismatches),
                    "expected_mismatch_found": found,
                    "upstream_all_pass": upstream_all,
                    "upstream_failed": [r["name"] for r in upstream if not r["passed"]],
                    "as_expected": case_score.db_reward == 0.0 and found and upstream_all == case["upstream_all_pass"],
                }
            )
        upstream_gold = run_upstream_verifiers(gold_env.tools.db, entry["upstream_verifiers"])
        upstream_initial = run_upstream_verifiers(initial_env.tools.db, entry["upstream_verifiers"])
        initial_passing = sorted(r["name"] for r in upstream_initial if r["passed"])
        boundary = information_boundary(task, entry)
        episode = measure_episode(renderer, task, policy)
        episode["seconds"]["gold_replay_and_db_scoring"] = round(scoring_s, 4)
        checks = {
            "bench2_gold_db_1": score.db_reward == 1.0,
            "repeat_replay_same_final_db": repeat_hash == gold_env.get_db_hash(),
            "upstream_all_pass_on_gold": all(r["passed"] for r in upstream_gold),
            "upstream_not_all_pass_on_initial": not all(r["passed"] for r in upstream_initial),
            "initial_passes_exactly_the_registered_ones": initial_passing == sorted(entry["initial_pass_allowed"]),
            "validation_cases_as_expected": all(c["as_expected"] for c in cases),
            "no_private_text_visible_at_reset": not boundary["leaked"],
            "scripted_episode_ends_with_db_match": episode["db_match"] and episode["termination_reason"] == "agent_done",
        }
        results[entry["id"]] = {
            "batch_item": entry["batch_item"],
            "upstream_task_id": entry["source"]["upstream_task_id"],
            "org_ids": entry["org_ids"],
            "gold_actions": len(task.evaluation_criteria.actions),
            "bench2_gold_db_reward": score.db_reward,
            "upstream_verifiers_on_gold": upstream_gold,
            "upstream_verifiers_on_initial_state": upstream_initial,
            "initial_pass_allowed": entry["initial_pass_allowed"],
            "validation_cases": cases,
            "information_boundary": boundary,
            "measured_scripted_episode": episode,
            "checks": checks,
            "accepted": all(checks.values()),
        }

    family = family_overlap_check_v2(batch["tasks"])

    # Re-check the evidence behind every task isolated after conversion.
    evidence = {}
    for item in batch["isolated"]:
        entry = item.get("attempted_conversion")
        if entry is None:
            continue
        task = pilot_tasks_from_entries([entry])[entry["id"]]
        gold_db = score_final_db(task, replay_gold_actions(task), text_match=FUZZY).db_reward
        cases = {}
        for case in entry["validation_cases"]:
            env = run_actions(task, edited_actions(task, case["edit"]))
            cases[case["name"]] = score_final_db(task, env, text_match=FUZZY).db_reward
        overlap = family_overlap_check_v2([*batch["tasks"], entry])["tasks"][entry["id"]]
        evidence[entry["id"]] = {
            "batch_item": item["batch_item"],
            "reason": item["reason"],
            "gold_db_reward": gold_db,
            "validation_case_db_rewards": cases,
            "undiscriminated_cases": sorted(name for name, reward in cases.items() if reward != 0.0),
            "dev_or_heldout_entity_overlap_ge2": overlap["dev_or_heldout_entity_overlap_ge2"],
        }

    result = {
        "batch_version": batch["version"],
        "caveat": (
            "Scripted gold episodes: tool calls are the gold actions, user turns are fixed text, one plan and one "
            "recap message are templated. Token figures are not a policy output distribution; timings are local CPU "
            "(WSL2). Bench2 scoring is full-DB fuzzy; upstream verifiers run on the same final DB exported to SQLite. "
            "Passing both does not make a converted task equivalent to the EnterpriseOps-Gym evaluation."
        ),
        "tokenizer": {"revision": "b968826d9c46dd6066d109eabc6255188de91218", "enable_thinking": False},
        "max_steps": MAX_STEPS,
        "tasks": results,
        "isolated": [{"upstream_task_id": i["upstream_task_id"], "batch_item": i["batch_item"], "reason": i["reason"]} for i in batch["isolated"]],
        "isolation_evidence": evidence,
        "family_overlap_v2": family,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: {"accepted": v["accepted"], "failed": [c for c, ok in v["checks"].items() if not ok], **{m: v["measured_scripted_episode"][m] for m in ("step_count", "response_tokens", "full_sequence_tokens")}} for k, v in results.items()}, indent=1))
    print(json.dumps(family["checks"], indent=1))
    return 0 if all(v["accepted"] for v in results.values()) and all(family["checks"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
