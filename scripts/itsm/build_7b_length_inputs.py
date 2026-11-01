"""Step 7B: build the three length-test inputs with ItsmAgentLoop's own token construction.

Usage (from the repo root; CPU only, deterministic):
    python scripts/itsm/build_7b_length_inputs.py --tokenizer-dir DIR --out FILE [--check META]

The sequences are produced by the real ``ItsmAgentLoop`` (prompt rendering, one
turn per action, ``turn_separator`` + observation turn) driven by a sampler that
replays fixed assistant segments, against an in-process ``ItsmEnv`` of the
GRPO-trainable train task ``itsmbench_msp_013`` (longest gold replay, step 6C).
The user is step 6's placeholder (first turn "Hi, I need help with a ticket."),
so tier a equals the measured 28,104-token initial prompt. No held-out task, no
LLM call.

Tiers (prompt = the generation input; target = the next gold action, used only
for the teacher-forced diagnostic):
  a  initial prompt (~28.1k), no target
  b  gold replay cut at the last turn boundary whose prompt <= 32768 - 512; target = next gold action
  c  b's prefix + extra real read-only tool calls and their real observations of the same
     env until the prompt is as close to 40960 - 512 as turn boundaries allow; same target.
     SPLICED: real content but not a natural episode (engineering stress only).

Determinism: Bench2's ``slice_db_to_orgs`` builds the sliced DB by iterating a
``set`` of record ids, so the ORDER of unfiltered list results (e.g. tier c's
``list_incidents``) depends on Python's hash seed. The builder therefore always
runs with ``PYTHONHASHSEED=0`` (it re-executes itself if needed).

``--check META`` compares the built token sequences' SHA256 with a committed meta
file (configs/itsm/step7b/length_inputs_meta.json), e.g. on the remote with the
model directory's tokenizer.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__" and os.environ.get("PYTHONHASHSEED") != "0":
    os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, "PYTHONHASHSEED": "0"})
sys.path.insert(0, str(REPO_ROOT / "src"))

from transformers import AutoTokenizer  # noqa: E402

from enterprise_worlds.data_model.message import UserMessage  # noqa: E402
from enterprise_worlds.evaluator.text_match_strategy import TextMatchConfig  # noqa: E402
from verl.utils.tokenizer.chat_template import initialize_system_prompt, initialize_turn_separator  # noqa: E402

from agent_rl.data.itsm_task_sets import resolve_task  # noqa: E402
from agent_rl.envs.itsm_env import ItsmEnv, ItsmEnvConfig  # noqa: E402
from agent_rl.rollout.itsm_agent_loop import ItsmAgentLoop, ItsmAgentLoopSettings  # noqa: E402

TASK_ID = "itsmbench_msp_013"
SPLIT = "itsm_split_v3"
LIMIT_B = 32768 - 512
LIMIT_C = 40960 - 512
PLACEHOLDER_REPLIES = ("Hi, I need help with a ticket.", "Thanks, that is all.")  # step 6 measurement user
#: read-only tools (no required arguments) used to extend tier c, in this order
EXTRA_READS = (
    "list_users", "list_incidents", "list_problems", "list_changes", "find_configuration_items",
    "find_services", "find_service_offerings", "find_locations", "list_user_groups",
    "find_sla_definitions", "find_notifications", "retrieve_knowledge_articles",
)


class _PlaceholderUser:
    def get_init_state(self) -> int:
        return 0

    def generate_next_message(self, agent_message, state: int):
        return UserMessage(content=PLACEHOLDER_REPLIES[min(state, 1)]), state + 1

    @staticmethod
    def is_stop(message) -> bool:
        return False


def segment_ids(tokenizer, name: str, arguments: dict) -> list[int]:
    """The tokens a policy emits for one tool call: text after the generation prompt, through <|im_end|>."""
    context = [{"role": "user", "content": "x"}]
    assistant = {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}]}
    before = tokenizer.apply_chat_template(context, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    after = tokenizer.apply_chat_template([*context, assistant], tokenize=False, add_generation_prompt=False, enable_thinking=False)
    if not after.startswith(before):
        raise RuntimeError("chat template is not prefix-stable for one assistant turn")
    return tokenizer.encode(after[len(before):].rstrip("\n"), add_special_tokens=False)


class ReplaySampler:
    def __init__(self, segments: list[list[int]]) -> None:
        self.segments = segments
        self.calls = 0

    async def generate(self, *, request_id, prompt_ids, sampling_params):
        ids = self.segments[self.calls]
        self.calls += 1
        return SimpleNamespace(token_ids=list(ids), log_probs=[-1.0] * len(ids), stop_reason="completed")


def replay(tokenizer, actions: list[tuple[str, dict]], *, response_length: int):
    """Run ItsmAgentLoop over ``actions`` + a final done; return prompt ids and the response with spans."""
    loop = object.__new__(ItsmAgentLoop)
    loop.settings = ItsmAgentLoopSettings(purpose="grpo")
    loop.tokenizer = tokenizer
    loop.processor = None
    loop.apply_chat_template_kwargs = {"enable_thinking": False}
    loop.system_prompt = initialize_system_prompt(tokenizer, enable_thinking=False)
    loop.turn_separator = initialize_turn_separator(tokenizer, enable_thinking=False)
    loop.rollout_config = SimpleNamespace(prompt_length=28672, response_length=response_length)
    loop.server_manager = ReplaySampler([segment_ids(tokenizer, n, a) for n, a in [*actions, ("done", {})]])
    loop._make_env = lambda task_id, episode_id: ItsmEnv(
        ItsmEnvConfig(task_id=task_id, text_match=TextMatchConfig(strategy="fuzzy"), max_steps=40, split_version=SPLIT),
        user=_PlaceholderUser(), episode_id=episode_id,
    )
    loop._check_template()

    async def go():
        loop.loop = asyncio.get_running_loop()
        return await loop.run({"temperature": 1.0}, task_id=TASK_ID, uid="7b-length", rollout_n=0)

    return asyncio.run(go())


def prefix_before_turn(output, turn: int) -> list[int]:
    """prompt + response up to the start of policy turn ``turn`` (ends with an observation turn)."""
    start = output.extra_fields["itsm_turn_spans"][turn]["start"]
    return list(output.prompt_ids) + list(output.response_ids[:start])


def turn_tokens(output, turn: int) -> list[int]:
    span = output.extra_fields["itsm_turn_spans"][turn]
    return list(output.response_ids[span["start"]:span["end"]])


def digest(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def build(tokenizer) -> dict:
    task = resolve_task(TASK_ID, split_version=SPLIT)
    gold = [(a.name, copy.deepcopy(a.arguments)) for a in task.evaluation_criteria.actions]
    full = replay(tokenizer, gold, response_length=65536)
    tier_a = list(full.prompt_ids)
    k = max(t for t in range(len(gold)) if len(prefix_before_turn(full, t)) <= LIMIT_B)
    tier_b = prefix_before_turn(full, k)
    target = turn_tokens(full, k)

    # tier c: same first k gold actions, then extra reads, then gold action k; add reads while the
    # prefix before gold action k stays within LIMIT_C.
    chosen: list[str] = []
    best = full
    for tool in EXTRA_READS:
        trial = replay(tokenizer, [*gold[:k], *[(t, {}) for t in [*chosen, tool]], *gold[k:]], response_length=65536)
        turn = k + len(chosen) + 1
        if turn < len(trial.extra_fields["itsm_turn_spans"]) and len(prefix_before_turn(trial, turn)) <= LIMIT_C:
            chosen.append(tool)
            best = trial
    tier_c = prefix_before_turn(best, k + len(chosen))
    target_c = turn_tokens(best, k + len(chosen))
    if target_c != target:
        raise RuntimeError("tier c target differs from tier b target")
    obs_tokens = []
    env_after = {span["after_turn"]: span for span in best.extra_fields["itsm_env_spans"]}
    for i, tool in enumerate(chosen):
        span = env_after[k + i]
        obs_tokens.append({"tool": tool, "observation_turn_tokens": span["end"] - span["start"]})
    tiers = {
        "a": {"prompt_token_ids": tier_a, "target_token_ids": None, "natural": True,
              "description": "initial prompt (step 6 construction)"},
        "b": {"prompt_token_ids": tier_b, "target_token_ids": target, "natural": True,
              "description": f"gold replay cut before gold action {k} ({gold[k][0]})", "cut_before_gold_action": k},
        "c": {"prompt_token_ids": tier_c, "target_token_ids": target, "natural": False,
              "description": f"SPLICED: gold actions 0..{k - 1}, then {len(chosen)} extra read-only calls, cut before gold action {k}",
              "extra_reads": obs_tokens},
    }
    for tier in tiers.values():
        tier["prompt_tokens"] = len(tier["prompt_token_ids"])
        tier["prompt_sha256"] = digest(tier["prompt_token_ids"])
        tier["target_tokens"] = None if tier["target_token_ids"] is None else len(tier["target_token_ids"])
        tier["target_sha256"] = None if tier["target_token_ids"] is None else digest(tier["target_token_ids"])
    return {"task_id": TASK_ID, "split_version": SPLIT, "limits": {"b": LIMIT_B, "c": LIMIT_C},
            "placeholder_user": list(PLACEHOLDER_REPLIES), "tiers": tiers}


def meta_of(built: dict) -> dict:
    return {**{k: v for k, v in built.items() if k != "tiers"},
            "tiers": {name: {k: v for k, v in tier.items() if not k.endswith("_token_ids")} for name, tier in built["tiers"].items()}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--meta-out", type=Path)
    parser.add_argument("--check", type=Path, help="committed meta file to compare against")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    built = build(tokenizer)
    meta = meta_of(built)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(built) + "\n", encoding="utf-8")
    if args.meta_out:
        args.meta_out.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({n: {k: t[k] for k in ("prompt_tokens", "target_tokens", "prompt_sha256")} for n, t in meta["tiers"].items()}, indent=2))
    if args.check:
        expected = json.loads(args.check.read_text(encoding="utf-8"))
        if expected != meta:
            print("MISMATCH with", args.check, file=sys.stderr)
            return 1
        print("matches", args.check)
    return 0


if __name__ == "__main__":
    sys.exit(main())
