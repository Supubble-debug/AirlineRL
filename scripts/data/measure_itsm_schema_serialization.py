"""Offline token comparison of lossless tool-schema serializations (step 6).

Usage (from the repo root):
    python scripts/data/measure_itsm_schema_serialization.py --tokenizer-dir DIR [--out FILE]

Renders the initial ITSM prompt with Qwen3's chat template, then swaps only the
text inside <tools>...</tools> for each variant in
``agent_rl.prompts.tool_serialization``. A variant counts only if parsing its
block returns schemas equal to the originals (deep equality and canonical JSON).
Nothing outside the tools block changes. CPU only; no model weights.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import transformers
from transformers import AutoTokenizer

from enterprise_worlds.data_model.message import UserMessage
from enterprise_worlds.evaluator.text_match_strategy import TextMatchConfig

from agent_rl.data.itsm_split import split_task_ids
from agent_rl.envs.itsm_env import ItsmEnv, ItsmEnvConfig
from agent_rl.prompts.action_prompt import build_action_messages
from agent_rl.prompts.tool_serialization import SERIALIZATIONS

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "docs" / "itsm_step6" / "schema_serialization.json"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
NATIVE_CONTEXT = 32768
CANDIDATE_CONTEXT = 40960


class _OneReplyUser:
    def get_init_state(self) -> int:
        return 0

    def generate_next_message(self, agent_message, state: int):
        return UserMessage(content="Hi, I need help with a ticket."), state + 1


def _canonical(tools) -> str:
    return json.dumps(tools, ensure_ascii=False, sort_keys=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_dir), trust_remote_code=False)
    task_id = split_task_ids("train")[0]
    env = ItsmEnv(
        ItsmEnvConfig(task_id=task_id, text_match=TextMatchConfig(strategy="fuzzy"), max_steps=40),
        user=_OneReplyUser(),
    )
    reset = env.reset()
    tools = copy.deepcopy(reset.public_info["tools"])
    messages = build_action_messages(domain_policy=reset.public_info["policy"], observation=reset.observation)
    baseline = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

    def count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    default_block = SERIALIZATIONS["template_default"].render(tools)
    wrapped_default = f"<tools>\n{default_block}\n</tools>"
    if baseline.count(wrapped_default) != 1:
        raise RuntimeError("template_default does not reproduce the chat template's tools block")
    head, tail = baseline.split(wrapped_default)
    baseline_tokens = count(baseline)
    baseline_block_tokens = count(default_block)

    variants = {}
    for name, serialization in SERIALIZATIONS.items():
        block = serialization.render(tools)
        parsed = serialization.parse(block)
        prompt = f"{head}<tools>\n{block}\n</tools>{tail}"
        prompt_tokens = count(prompt)
        variants[name] = {
            "description": serialization.description,
            "parse_equal": parsed == tools,
            "parse_equal_canonical_json": _canonical(parsed) == _canonical(tools),
            "outside_tools_block_unchanged": prompt.startswith(head) and prompt.endswith(tail),
            "tools_block_sha256": hashlib.sha256(block.encode("utf-8")).hexdigest(),
            "tools_block_chars": len(block),
            "tools_block_tokens": count(block),
            "initial_prompt_tokens": prompt_tokens,
            "saved_tokens_vs_template_default": baseline_tokens - prompt_tokens,
            "saved_fraction_of_tools_block": round((baseline_tokens - prompt_tokens) / baseline_block_tokens, 4),
            "room_left_at_native_32768": NATIVE_CONTEXT - prompt_tokens,
            "room_left_at_candidate_40960": CANDIDATE_CONTEXT - prompt_tokens,
        }

    tokenizer_config = json.loads((args.tokenizer_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    result = {
        "caveat": (
            "Offline token count only; no variant is used by the rollout. Formats other than "
            "template_default differ from the JSON tool block Qwen3 was trained with, so a "
            "behavioural check would be needed before adopting any of them."
        ),
        "tokenizer": {
            "repo": "Qwen/Qwen3-8B",
            "revision": REVISION,
            "chat_template_sha256": hashlib.sha256(tokenizer_config["chat_template"].encode("utf-8")).hexdigest(),
            "transformers_version": transformers.__version__,
            "enable_thinking": False,
        },
        "prompt": {
            "task_used_for_reset": task_id,
            "tool_count_including_done": len(tools),
            "policy_unchanged": True,
            "template_default_initial_prompt_tokens": baseline_tokens,
        },
        "variants": variants,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(variants, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
