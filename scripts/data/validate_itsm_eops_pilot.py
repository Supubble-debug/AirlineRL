"""Validate and measure the EnterpriseOps-Gym ITSM migration pilot (step 6B).

Usage (from the repo root):
    python scripts/data/validate_itsm_eops_pilot.py --tokenizer-dir DIR [--out FILE]

For every pilot task: strict gold replay and Bench2 DB scoring (fuzzy), the
upstream verifiers on the gold final DB and on the untouched initial state, every
validation case (Bench2 DB and upstream verifiers), and a scripted ItsmEnv
episode measured with the pinned Qwen3-8B tokenizer (same counting as
measure_itsm_tokens.py). The scripted episode replays gold tool calls with one
planning message, one recap message and fixed user replies; it is not a policy
output distribution and the user simulator is not called. Also re-runs the
cross-source family/overlap check over the 53 Bench2 tasks plus the pilot.
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

from agent_rl.data.itsm_eops_pilot import (
    edited_actions,
    family_overlap_check,
    load_pilot,
    pilot_tasks,
    run_actions,
    run_upstream_verifiers,
)
from agent_rl.envs.itsm_env import ItsmAction, ItsmEnv, ItsmEnvConfig
from agent_rl.envs.itsm_scoring import build_task_environment, replay_gold_actions, score_final_db, task_clock
from agent_rl.prompts.action_prompt import OBSERVATION_PROMPT, build_action_messages

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_itsm_tokens as measure  # noqa: E402  (our own step 6 script)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "docs" / "itsm_step6b" / "pilot_validation.json"
FUZZY = TextMatchConfig(strategy="fuzzy")
MAX_STEPS = 40  # configs/itsm/budget_v1.yaml
#: Scripted (not simulated) user turns for the measured episode: opening, plan confirmation, closing.
OPENINGS = {
    "itsmbench_eops_46ca2862_b6ef60ca": "Hi, this is Marcus Thompson. Benjamin Chen got a new laptop in the Q1 refresh; please add it to the asset records and open an incident for him about software installation.",
    "itsmbench_eops_66392a82_52c8290e": "Hi, Marcus Thompson here. The 4K video conferencing hub keeps freezing and dropping calls, and its recorded cost is wrong. Can you fix the cost and log an incident?",
    "itsmbench_eops_1d219d93_d1ba6c2d": "Hi, Karen Watkins here. We're opening Pune HQ and I need the location, asset moves, two new assets, a role change and a short KB article set up.",
}
CONFIRM = "Yes, go ahead. No notification needed."
CLOSING = "Thanks, that's everything."


class _ScriptedUser:
    def __init__(self, opening: str) -> None:
        self.replies = [opening, CONFIRM, CLOSING]

    def get_init_state(self) -> int:
        return 0

    def generate_next_message(self, agent_message, state: int):
        return UserMessage(content=self.replies[min(state, len(self.replies) - 1)]), state + 1


def _scripted_actions(task) -> list[ItsmAction]:
    gold = [ItsmAction.tool_call(a.name, copy.deepcopy(a.arguments)) for a in task.evaluation_criteria.actions]
    read_names = {"get_user_using_name", "find_locations", "find_configuration_items", "find_configuration_item_by_serial_number", "retrieve_knowledge_articles", "find_sla_definitions"}
    first_write = next(i for i, a in enumerate(gold) if a.name not in read_names)
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
    initial = renderer.count(
        renderer.text(build_action_messages(domain_policy=policy, observation=reset.observation), generation=True)
    )
    generated, environment, tool_seconds = [], [], []
    for action in _scripted_actions(task):
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
        "seconds": {
            "env_build_and_reset": round(reset_s, 4),
            "tool_calls_total": round(sum(tool_seconds), 4),
            "tool_call_max": round(max(tool_seconds), 4),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    pilot = load_pilot()
    tasks = pilot_tasks()
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_dir), trust_remote_code=False)
    first = next(iter(tasks.values()))
    environment = build_task_environment(first)
    renderer = measure.Renderer(tokenizer, environment.get_tool_schemas() + [copy.deepcopy(DONE_TOOL_SCHEMA)])
    policy = environment.get_policy()

    results = {}
    for entry in pilot["tasks"]:
        task = tasks[entry["id"]]
        tick = time.perf_counter()
        gold_env = replay_gold_actions(task)
        score = score_final_db(task, gold_env, text_match=FUZZY)
        scoring_s = time.perf_counter() - tick
        with task_clock(task):
            initial_env = build_task_environment(task)
        cases = []
        for case in entry["validation_cases"]:
            env = run_actions(task, edited_actions(task, case["edit"]))
            case_score = score_final_db(task, env, text_match=FUZZY)
            upstream = run_upstream_verifiers(env.tools.db, entry["upstream_verifiers"])
            upstream_all = all(r["passed"] for r in upstream)
            cases.append(
                {
                    "name": case["name"],
                    "reason": case["reason"],
                    "bench2_db_reward": case_score.db_reward,
                    "bench2_mismatches": list(case_score.mismatches),
                    "expected_mismatch_found": any(case["bench2_mismatch"] in m for m in case_score.mismatches),
                    "upstream_all_pass": upstream_all,
                    "upstream_failed": [r["name"] for r in upstream if not r["passed"]],
                    "as_expected": case_score.db_reward == 0.0
                    and any(case["bench2_mismatch"] in m for m in case_score.mismatches)
                    and upstream_all == case["upstream_all_pass"],
                }
            )
        upstream_gold = run_upstream_verifiers(gold_env.tools.db, entry["upstream_verifiers"])
        upstream_initial = run_upstream_verifiers(initial_env.tools.db, entry["upstream_verifiers"])
        episode = measure_episode(renderer, task, policy)
        episode["seconds"]["gold_replay_and_db_scoring"] = round(scoring_s, 4)
        results[entry["id"]] = {
            "upstream_task_id": entry["source"]["upstream_task_id"],
            "gold_actions": len(task.evaluation_criteria.actions),
            "bench2_gold_db_reward": score.db_reward,
            "upstream_verifiers_on_gold": upstream_gold,
            "upstream_verifiers_on_initial_state": upstream_initial,
            "validation_cases": cases,
            "measured_scripted_episode": episode,
            "accepted": score.db_reward == 1.0
            and all(r["passed"] for r in upstream_gold)
            and not any(r["passed"] for r in upstream_initial)
            and all(c["as_expected"] for c in cases)
            and episode["db_match"]
            and episode["termination_reason"] == "agent_done",
        }

    result = {
        "pilot_version": pilot["version"],
        "caveat": (
            "Scripted gold episodes: tool calls are the gold actions, user turns are fixed text, "
            "one plan and one recap message are templated. Token figures are not a policy output "
            "distribution; timings are local CPU (WSL2). Bench2 scoring is full-DB fuzzy; upstream "
            "verifiers are run on the same final DB exported to SQLite. Passing both does not make "
            "the converted task equivalent to the EnterpriseOps-Gym evaluation."
        ),
        "tokenizer": {"revision": "b968826d9c46dd6066d109eabc6255188de91218", "enable_thinking": False},
        "max_steps": MAX_STEPS,
        "tasks": results,
        "isolated": [{"upstream_task_id": i["upstream_task_id"], "reason": i["reason"]} for i in pilot["isolated"]],
        "family_overlap": family_overlap_check(pilot["tasks"]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: {"accepted": v["accepted"], **{m: v["measured_scripted_episode"][m] for m in ("agent_turns", "step_count", "initial_prompt_tokens", "response_tokens", "full_sequence_tokens")}} for k, v in results.items()}, indent=1))
    print(json.dumps(result["family_overlap"]["checks"], indent=1))
    return 0 if all(v["accepted"] for v in results.values()) and all(result["family_overlap"]["checks"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
