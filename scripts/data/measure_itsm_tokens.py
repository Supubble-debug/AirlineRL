"""Measure ITSM prompt, schema, observation and action token costs on CPU (step 6).

Usage (from the repo root):
    python scripts/data/measure_itsm_tokens.py --tokenizer-dir DIR [--out FILE]

DIR holds the pinned Qwen3-8B tokenizer files only (no weights). Episodes are
scripted gold replays of train/dev tasks through ``ItsmEnv``: they measure the
environment's side and the length of *gold* tool calls, NOT the length
distribution of real policy outputs, and user turns are fixed placeholders
(the user simulator is not called). The held-out split is never read.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import transformers
from transformers import AutoTokenizer

from enterprise_worlds.data_model.message import UserMessage
from enterprise_worlds.environment.environment import _stringify
from enterprise_worlds.evaluator.text_match_strategy import TextMatchConfig
from enterprise_worlds.gym.gym_agent import DONE_TOOL_SCHEMA

from agent_rl.data.itsm_split import load_manifest, split_task_ids
from agent_rl.envs.itsm_env import ItsmAction, ItsmEnv, ItsmEnvConfig
from agent_rl.envs.itsm_scoring import (
    build_task_environment,
    load_itsm_tasks,
    score_final_db,
    task_clock,
)
from agent_rl.prompts.action_prompt import OBSERVATION_PROMPT, build_action_messages

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "docs" / "itsm_step6" / "token_measurements.json"
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "config.json",
    "generation_config.json",
)
PLACEHOLDER_REPLIES = ("Hi, I need help with a ticket.", "Thanks, that is all.")
CLOSING_MESSAGE = "All done; the requested updates are recorded."
MEASUREMENT_MAX_STEPS = 1000  # measurement only: never truncate a gold replay


class _PlaceholderUser:
    def get_init_state(self) -> int:
        return 0

    def generate_next_message(self, agent_message, state: int):
        return UserMessage(content=PLACEHOLDER_REPLIES[min(state, 1)]), state + 1


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(values: list[int | float]) -> dict[str, float]:
    ordered = sorted(values)

    def pct(q: float) -> float:
        return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]

    return {
        "n": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p90": pct(0.9),
        "max": ordered[-1],
        "mean": round(statistics.fmean(ordered), 2),
    }


class Renderer:
    def __init__(self, tokenizer, tools: list[dict[str, Any]]) -> None:
        self.tokenizer = tokenizer
        self.tools = tools

    def text(self, messages, *, generation: bool, tools: bool = True) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tools=self.tools if tools else None,
            tokenize=False,
            add_generation_prompt=generation,
            enable_thinking=False,
        )

    def count(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def generated(self, assistant: dict[str, Any]) -> int:
        """Tokens the policy emits for one action: the text after the generation
        prompt up to and including <|im_end|> (the assistant segment does not depend
        on earlier context)."""
        context = [{"role": "user", "content": "x"}]
        before = self.text(context, generation=True, tools=False)
        after = self.text([*context, assistant], generation=False, tools=False)
        if not after.startswith(before):
            raise RuntimeError("chat template is not prefix-stable for one assistant turn")
        return self.count(after[len(before):].rstrip("\n"))

    def environment_turn(self, content: str) -> int:
        """Tokens the loop appends after an action, as ``TauAgentLoop`` builds them:
        a newline separator after <|im_end|> plus one user turn rendered on its own
        with the generation prompt (``<think>\\n\\n</think>\\n\\n`` when thinking is off)."""
        turn = self.text([{"role": "user", "content": content}], generation=True, tools=False)
        return self.count("\n" + turn)


def _assistant_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}],
    }


def measure_static(renderer: Renderer, policy: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
    system_only = build_action_messages(domain_policy=policy, observation="x")[:1]
    with_tools = renderer.count(renderer.text(system_only, generation=False))
    without_tools = renderer.count(renderer.text(system_only, generation=False, tools=False))
    per_tool = sorted(
        ((renderer.count(json.dumps(tool, ensure_ascii=False)), tool["function"]["name"]) for tool in tools),
        reverse=True,
    )
    return {
        "policy_text_tokens": renderer.count(policy),
        "policy_chars": len(policy),
        "system_message_without_tools": without_tools,
        "system_message_with_tools": with_tools,
        "tool_schema_block_tokens": with_tools - without_tools,
        "tool_count_including_done": len(tools),
        "largest_tool_schemas": [{"tool": n, "json_tokens": t} for t, n in per_tool[:10]],
        "tool_schema_json_tokens_total": sum(t for t, _ in per_tool),
    }


def measure_episode(renderer: Renderer, task_id: str, policy: str) -> dict[str, Any]:
    task = load_itsm_tasks()[task_id]
    config = ItsmEnvConfig(task_id=task_id, text_match=TextMatchConfig(strategy="fuzzy"), max_steps=MEASUREMENT_MAX_STEPS)
    started = time.perf_counter()
    env = ItsmEnv(config, user=_PlaceholderUser(), episode_id=f"{task_id}:measure")
    reset = env.reset()
    reset_s = time.perf_counter() - started

    messages = build_action_messages(domain_policy=policy, observation=reset.observation)
    initial_prompt = renderer.count(renderer.text(messages, generation=True))
    actions, observations, tool_seconds = [], [], []
    gold = [
        ItsmAction.tool_call(a.name, copy.deepcopy(a.arguments))
        for a in task.evaluation_criteria.actions
    ]
    for action in [*gold, ItsmAction.message(CLOSING_MESSAGE), ItsmAction.done()]:
        if action.kind == "tool_call":
            assistant = _assistant_tool_call(action.name, action.arguments)
        elif action.kind == "message":
            assistant = {"role": "assistant", "content": action.text}
        else:
            assistant = _assistant_tool_call("done", {})
        tick = time.perf_counter()
        transition = env.step(action)
        if action.kind == "tool_call":
            tool_seconds.append(time.perf_counter() - tick)
        actions.append(
            {"kind": action.kind, "tool": action.name, "generated_tokens": renderer.generated(assistant)}
        )
        if transition.terminated or transition.truncated:
            break
        content = OBSERVATION_PROMPT.format(observation=transition.observation)
        observations.append(
            {
                "kind": "user" if action.kind == "message" else "tool",
                "raw_observation_tokens": renderer.count(transition.observation),
                "turn_tokens_incl_template_and_generation_prompt": renderer.environment_turn(content),
            }
        )

    generated_total = sum(a["generated_tokens"] for a in actions)
    environment_total = sum(o["turn_tokens_incl_template_and_generation_prompt"] for o in observations)
    tick = time.perf_counter()
    with task_clock(task):
        score_final_db(task, env._environment, text_match=TextMatchConfig(strategy="fuzzy"))
    scoring_s = time.perf_counter() - tick
    return {
        "task_id": task_id,
        "gold_tool_calls": len(gold),
        "agent_turns": transition.agent_turns,
        "step_count": transition.step_count,
        "db_match": transition.reward_info.db_match,
        "initial_prompt_tokens": initial_prompt,
        "response_tokens": generated_total + environment_total,
        "full_sequence_tokens": initial_prompt + generated_total + environment_total,
        "generated_tokens": generated_total,
        "environment_tokens": environment_total,
        "actions": actions,
        "observations": observations,
        "seconds": {
            "env_build_and_reset": round(reset_s, 4),
            "tool_calls_total": round(sum(tool_seconds), 4),
            "tool_call_max": round(max(tool_seconds), 4),
            "db_scoring_fuzzy": round(scoring_s, 4),
        },
    }


def measure_unfiltered_reads(renderer: Renderer, task_ids: list[str]) -> list[dict[str, Any]]:
    """Worst-case observations: every read tool that needs no argument, on each
    train/dev task's sliced DB (largest per tool kept)."""
    tasks = load_itsm_tasks()
    worst: dict[str, tuple[int, str]] = {}
    for task_id in task_ids:
        task = tasks[task_id]
        with task_clock(task):
            environment = build_task_environment(task)
            for name, tool in environment.tools.tools.items():
                if environment.tools.tool_type(name).value != "read":
                    continue
                required = [
                    p for p in inspect.signature(tool).parameters.values()
                    if p.default is inspect.Parameter.empty
                ]
                if required:
                    continue
                try:
                    text = _stringify(environment.make_tool_call(name)) or "null"
                except Exception:
                    continue
                tokens = renderer.count(text)
                if tokens > worst.get(name, (0, ""))[0]:
                    worst[name] = (tokens, task_id)
    return [
        {"tool": name, "max_raw_observation_tokens": tokens, "task_id": task_id}
        for name, (tokens, task_id) in sorted(worst.items(), key=lambda item: -item[1][0])
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_dir), trust_remote_code=False)
    task_ids = split_task_ids("train") + split_task_ids("dev")
    environment = build_task_environment(load_itsm_tasks()[task_ids[0]])
    policy = environment.get_policy()
    tools = environment.get_tool_schemas() + [copy.deepcopy(DONE_TOOL_SCHEMA)]
    renderer = Renderer(tokenizer, tools)
    manifest = load_manifest()

    episodes = [measure_episode(renderer, task_id, policy) for task_id in task_ids]
    tool_actions = [
        {**a, "task_id": e["task_id"]} for e in episodes for a in e["actions"] if a["kind"] == "tool_call"
    ]
    tool_observations = [o for e in episodes for o in e["observations"] if o["kind"] == "tool"]
    tokenizer_config = json.loads((args.tokenizer_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    model_config = json.loads((args.tokenizer_dir / "config.json").read_text(encoding="utf-8"))

    result = {
        "caveat": (
            "Scripted gold replays of train/dev tasks. Generated-token figures are the "
            "lengths of the gold tool calls rendered by the chat template, not a policy "
            "output distribution; user turns are fixed placeholders; held-out is not read."
        ),
        "tokenizer": {
            "repo": "Qwen/Qwen3-8B",
            "revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "files_sha256": {f: _sha256_file(args.tokenizer_dir / f) for f in TOKENIZER_FILES},
            "chat_template_sha256": hashlib.sha256(tokenizer_config["chat_template"].encode("utf-8")).hexdigest(),
            "transformers_version": transformers.__version__,
            "enable_thinking": False,
            "max_position_embeddings": model_config["max_position_embeddings"],
        },
        "prompt_construction": (
            "initial prompt = chat template of build_action_messages(policy, reset observation) "
            "with tools=<93 ITSM schemas + done> and the generation prompt; then, as "
            "TauAgentLoop concatenates ids, each action adds its generated tokens and each "
            "observation adds a newline separator plus one user turn (OBSERVATION_PROMPT) "
            "rendered alone with the generation prompt. Re-rendering the whole conversation "
            "differs because Qwen3's template drops earlier empty <think> blocks."
        ),
        "observation_rule": "full_tool_results_no_truncation",
        "observation_rule_note": (
            "Every tool result is counted in full, as configs/itsm/budget_v1.yaml requires. "
            "unfiltered_read_observations shows single results (up to ~10k tokens) that can "
            "exhaust the response budget; under this rule such an episode ends with "
            "truncate('response_length') and keeps the full observation in its trace."
        ),
        "data": {
            "split_version": manifest["version"],
            "tasks_measured": task_ids,
            "splits_measured": ["train", "dev"],
        },
        "machine": {"platform": platform.platform(), "python": sys.version.split()[0], "cpu_count": os.cpu_count()},
        "static": measure_static(renderer, policy, tools),
        "summary": {
            "initial_prompt_tokens": _summary([e["initial_prompt_tokens"] for e in episodes]),
            "response_tokens_gold_replay": _summary([e["response_tokens"] for e in episodes]),
            "full_sequence_tokens": _summary([e["full_sequence_tokens"] for e in episodes]),
            "generated_tokens_per_tool_call": _summary([a["generated_tokens"] for a in tool_actions]),
            "tool_observation_turn_tokens": _summary(
                [o["turn_tokens_incl_template_and_generation_prompt"] for o in tool_observations]
            ),
            "tool_observation_raw_tokens": _summary([o["raw_observation_tokens"] for o in tool_observations]),
            "gold_tool_calls_per_task": _summary([e["gold_tool_calls"] for e in episodes]),
            "step_count_gold_replay": _summary([e["step_count"] for e in episodes]),
            "agent_turns_gold_replay": _summary([e["agent_turns"] for e in episodes]),
            "seconds_tool_call_max": _summary([e["seconds"]["tool_call_max"] for e in episodes]),
            "seconds_env_build_and_reset": _summary([e["seconds"]["env_build_and_reset"] for e in episodes]),
            "seconds_db_scoring_fuzzy": _summary([e["seconds"]["db_scoring_fuzzy"] for e in episodes]),
        },
        "longest_tool_calls": sorted(tool_actions, key=lambda a: -a["generated_tokens"])[:10],
        "unfiltered_read_observations": measure_unfiltered_reads(renderer, task_ids),
        "episodes": episodes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["static"] | result["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
