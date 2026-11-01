"""Step 7B GPU phases: length/memory tests, real episodes, training-side log-prob recompute.

Run on the cloud instance only (see scripts/itsm/run_7b.sh, which records the
resource timeline). Subcommands:

  length     --model DIR --max-model-len N --inputs F --out DIR [--tiers a,b,c] [--episodes]
             vLLM engine built as verl builds it (agent_rl.rollout.vllm_token_server);
             greedy generation (temperature 0, <= 512 tokens, top-5 log-probs) for each
             tier, prefix cache reset before each tier. With --episodes, the same
             engine then runs the real ItsmAgentLoop episodes (EPISODE_TASKS) at the
             GRPO sampling parameters (verl defaults: temperature 1, top_p 1, top_k -1).
  recompute  --model DIR --trajectories F --length-inputs F --out DIR
             HF transformers forward of the identical token sequences (same weights,
             bf16, sdpa), verl's log-prob function on logits / temperature, at mask=1
             positions only; teacher-forced gold-action log-probs for tiers b and c.

Never calls an external LLM: the episode user is PersonaScriptedUser and
``litellm`` is replaced by a raising stub in this process and every worker child.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agent_rl.envs.itsm_scripted_user import SCRIPT_VERSION, PersonaScriptedUser, forbid_external_llm  # noqa: E402

forbid_external_llm()  # before anything can import the real litellm

SPLIT = "itsm_split_v3"
#: fixed in advance: GRPO-trainable = the length-test task and the task of the 7A offline tests;
#: dev = the first msp and the first single task by ID.
EPISODE_TASKS = (
    ("grpo", "itsmbench_msp_013"),
    ("grpo", "itsmbench_eops_99ba2325_887f5580"),
    ("dev", "itsmbench_msp_019"),
    ("dev", "itsmbench_single_008"),
)
#: verl rollout defaults (rollout.yaml); the agent loop adds max_tokens and stop ids.
GRPO_SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0, "logprobs": True}
LENGTH_SAMPLING = {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0, "logprobs": True, "max_tokens": 512}
TOP_LOGPROBS = 5


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def finite_report(values) -> dict:
    if values is None:
        return {"present": False}
    bad = [i for i, v in enumerate(values) if not isinstance(v, (int, float)) or not math.isfinite(v)]
    return {"present": True, "count": len(values), "non_finite_positions": bad[:16], "all_finite": not bad}


# -- length + episodes (vLLM) ------------------------------------------------------------------


def make_loop(server, tokenizer, purpose: str, response_length: int):
    from verl.utils.tokenizer.chat_template import initialize_system_prompt, initialize_turn_separator

    from agent_rl.rollout.itsm_agent_loop import ItsmAgentLoop, ItsmAgentLoopSettings

    class Step7BLoop(ItsmAgentLoop):
        worker_initializer = staticmethod(forbid_external_llm)

        def _make_env(self, task_id, episode_id):
            self.user_factory = functools.partial(PersonaScriptedUser, task_id, SPLIT)
            return super()._make_env(task_id, episode_id)

    loop = object.__new__(Step7BLoop)
    loop.settings = ItsmAgentLoopSettings(purpose=purpose)
    loop.tokenizer = tokenizer
    loop.processor = None
    loop.apply_chat_template_kwargs = {"enable_thinking": False}
    loop.system_prompt = initialize_system_prompt(tokenizer, enable_thinking=False)
    loop.turn_separator = initialize_turn_separator(tokenizer, enable_thinking=False)
    loop.rollout_config = SimpleNamespace(prompt_length=28672, response_length=response_length)
    loop.server_manager = server
    loop._check_template()
    return loop


async def run_length(args, server, tokenizer, out: Path) -> dict:
    from agent_rl.rollout.itsm_agent_loop import parse_action

    inputs = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
    stop_ids = sorted({tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id} - {None})
    results = {}
    for name in args.tiers.split(","):
        tier = inputs["tiers"][name]
        prompt = tier["prompt_token_ids"]
        reset = server.engine.reset_prefix_cache()  # measure a cold prefill for every tier
        if asyncio.iscoroutine(reset):
            await reset
        started = time.monotonic()
        entry = {"tier": name, "natural": tier["natural"], "description": tier["description"], "prompt_tokens": len(prompt),
                 "prompt_sha256": tier["prompt_sha256"], "started_utc": utc()}
        try:
            output = await server.generate(request_id=f"length-{name}", prompt_ids=prompt,
                                           sampling_params={**LENGTH_SAMPLING, "stop_token_ids": stop_ids})
            record = server.records[-1]
            ids = output.token_ids
            entry |= {
                "ok": True,
                "output_tokens": len(ids),
                "total_tokens": len(prompt) + len(ids),
                "finish_reason": output.finish_reason,
                "stop_reason": output.stop_reason,
                "ended_with_stop_token": bool(ids) and ids[-1] in stop_ids,
                "first_output_s": record.first_output_s,
                "total_s": record.total_s,
                "output_token_ids": ids,
                "output_logprobs": output.log_probs,
                "output_top_logprobs": [{str(k): v for k, v in d.items()} for d in (output.top_logprobs or [])],
                "logprob_check": finite_report(output.log_probs),
                "logprob_count_matches": output.log_probs is not None and len(output.log_probs) == len(ids),
                "output_text": tokenizer.decode(ids, skip_special_tokens=False),
            }
            parsed = parse_action(tokenizer.decode(ids, skip_special_tokens=False), tokenizer.decode(ids, skip_special_tokens=True),
                                  stopped=bool(ids) and ids[-1] in stop_ids)
            entry["parse_action"] = {"kind": parsed.action.kind if parsed.action else None, "error": parsed.error,
                                     "tool": getattr(parsed.action, "name", None) if parsed.action else None}
        except BaseException as error:  # noqa: BLE001 - recorded; the tier failed
            entry |= {"ok": False, "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()[-4000:]}
        entry["wall_s"] = round(time.monotonic() - started, 3)
        results[name] = entry
        write_json(out / "length_results.json", results)
        print(f"[7b] tier {name}: ok={entry.get('ok')} prompt={entry['prompt_tokens']} out={entry.get('output_tokens')} "
              f"finish={entry.get('finish_reason')} ttft={entry.get('first_output_s')} total={entry.get('total_s')}", flush=True)
    return results


async def run_episodes(server, tokenizer, out: Path) -> list:
    server.keep_top_logprobs = None  # verl semantics: the sampled token's log-prob only
    episodes = []
    for purpose, task_id in EPISODE_TASKS:
        loop = make_loop(server, tokenizer, purpose, response_length=12288)
        loop.loop = asyncio.get_running_loop()
        first_record = len(server.records)
        started = time.monotonic()
        entry = {"purpose": purpose, "task_id": task_id, "user": SCRIPT_VERSION, "sampling": GRPO_SAMPLING, "started_utc": utc()}
        try:
            output = await loop.run(dict(GRPO_SAMPLING), task_id=task_id, uid=f"7b-{task_id}", rollout_n=0)
            turns = output.extra_fields["itsm_turn_spans"]
            entry |= {
                "ok": True,
                "prompt_ids": list(output.prompt_ids),
                "response_ids": list(output.response_ids),
                "response_mask": list(output.response_mask),
                "response_logprobs": list(output.response_logprobs),
                "reward_score": output.reward_score,
                "num_turns": output.num_turns,
                "extra_fields": output.extra_fields,
                "turn_texts": [tokenizer.decode(output.response_ids[t["start"]:t["end"]], skip_special_tokens=False) for t in turns],
                "env_texts": [tokenizer.decode(output.response_ids[s["start"]:s["end"]], skip_special_tokens=False)
                              for s in output.extra_fields["itsm_env_spans"]],
                "logprob_check": finite_report([lp for lp, m in zip(output.response_logprobs, output.response_mask) if m == 1]),
            }
        except BaseException as error:  # noqa: BLE001 - recorded; the episode failed (no sample)
            entry |= {"ok": False, "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()[-4000:]}
        entry["wall_s"] = round(time.monotonic() - started, 3)
        entry["requests"] = [vars(r) for r in server.records[first_record:]]
        episodes.append(entry)
        write_json(out / "episodes.json", episodes)
        info = entry.get("extra_fields", {}).get("reward_extra_info", {})
        print(f"[7b] episode {task_id}: ok={entry['ok']} turns={entry.get('num_turns')} end={info.get('itsm_termination_reason')} "
              f"reward={entry.get('reward_score')} wall={entry['wall_s']} err={entry.get('error')}", flush=True)
    return episodes


def cmd_length(args) -> int:
    from transformers import AutoTokenizer

    from agent_rl.rollout.vllm_token_server import VllmTokenServer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    meta = {"max_model_len": args.max_model_len, "started_utc": utc(), "pythonhashseed": os.environ.get("PYTHONHASHSEED")}

    async def go() -> int:
        status = 0
        t0 = time.monotonic()
        server = VllmTokenServer.build(args.model, max_model_len=args.max_model_len, keep_top_logprobs=TOP_LOGPROBS)  # inside the loop, as verl
        try:
            meta["engine_init_s"] = round(time.monotonic() - t0, 3)
            meta["cli_args"] = server.cli_args
            cache = server.vllm_config.cache_config
            meta["cache_config"] = {"num_gpu_blocks": getattr(cache, "num_gpu_blocks", None), "block_size": getattr(cache, "block_size", None),
                                    "gpu_memory_utilization": getattr(cache, "gpu_memory_utilization", None)}
            meta["model_config"] = {"max_model_len": server.vllm_config.model_config.max_model_len,
                                    "dtype": str(server.vllm_config.model_config.dtype)}
            write_json(out / "engine.json", meta)
            print(f"[7b] engine ready max_model_len={args.max_model_len} init={meta['engine_init_s']}s cache={meta['cache_config']}", flush=True)
            results = await run_length(args, server, tokenizer, out)
            if not all(r.get("ok") for r in results.values()):
                status = 2
            if args.episodes:
                episodes = await run_episodes(server, tokenizer, out)
                if not all(e["ok"] for e in episodes):
                    status = max(status, 3)
        finally:
            meta["finished_utc"] = utc()
            write_json(out / "engine.json", meta)
            server.shutdown()
        return status

    return asyncio.run(go())


# -- recompute (HF, training side) --------------------------------------------------------------


def recompute_token_logprobs(model, ids: list[int], positions: list[int], temperature: float, *, device: str, chunk: int = 2048):
    """log p(ids[p] | ids[:p]) for each p in ``positions``: (verl function on bf16 logits / T, fp32 log-softmax)."""
    import torch

    from verl.utils.torch_functional import logprobs_from_logits

    if any(p < 1 or p >= len(ids) for p in positions):
        raise ValueError("positions must index target tokens after the first token")
    with torch.inference_mode():
        input_ids = torch.tensor([ids], device=device)
        hidden = model.model(input_ids=input_ids, use_cache=False).last_hidden_state[0]
        verl_values, fp32_values = [], []
        for start in range(0, len(positions), chunk):
            part = positions[start:start + chunk]
            rows = torch.tensor([p - 1 for p in part], device=device)  # the logits at p-1 predict token p
            logits = model.lm_head(hidden[rows])  # model dtype (bf16), as the actor's forward returns
            labels = torch.tensor([ids[p] for p in part], device=device)
            scaled = logits / temperature
            verl_values += logprobs_from_logits(scaled, labels).float().tolist()
            fp32_values += torch.log_softmax(scaled.float(), dim=-1).gather(-1, labels[:, None])[:, 0].tolist()
        del hidden
    if device == "cuda":
        torch.cuda.empty_cache()
    return verl_values, fp32_values


def cmd_recompute(args) -> int:
    import torch
    from transformers import AutoModelForCausalLM

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    meta = {"started_utc": utc(), "load_s": round(time.monotonic() - t0, 3), "dtype": "bfloat16", "attn_implementation": "sdpa",
            "torch": torch.__version__, "logprob_function": "verl.utils.torch_functional.logprobs_from_logits (on bf16 logits / T)",
            "also": "float32 log_softmax of the same bf16 logits"}

    def token_logprobs(ids, positions, temperature):
        return recompute_token_logprobs(model, ids, positions, temperature, device="cuda")

    results = {"meta": meta, "episodes": [], "gold_targets": {}}
    for episode in json.loads(Path(args.trajectories).read_text(encoding="utf-8")):
        if not episode.get("ok"):
            continue
        prompt, response, mask = episode["prompt_ids"], episode["response_ids"], episode["response_mask"]
        ids = prompt + response
        positions = [len(prompt) + i for i, m in enumerate(mask) if m == 1]
        started = time.monotonic()
        verl_lp, fp32_lp = token_logprobs(ids, positions, GRPO_SAMPLING["temperature"])
        results["episodes"].append({"task_id": episode["task_id"], "sequence_tokens": len(ids), "mask1_positions": positions,
                                    "recompute_logprobs": verl_lp, "recompute_logprobs_fp32": fp32_lp,
                                    "seconds": round(time.monotonic() - started, 3)})
        print(f"[7b] recompute {episode['task_id']}: {len(positions)} mask-1 tokens of {len(ids)}", flush=True)
        write_json(out / "recompute.json", results)
    inputs = json.loads(Path(args.length_inputs).read_text(encoding="utf-8"))
    for name in ("b", "c"):
        tier = inputs["tiers"][name]
        ids = tier["prompt_token_ids"] + tier["target_token_ids"]
        positions = list(range(len(tier["prompt_token_ids"]), len(ids)))
        verl_lp, fp32_lp = token_logprobs(ids, positions, 1.0)
        results["gold_targets"][name] = {"prompt_tokens": len(tier["prompt_token_ids"]), "target_token_ids": tier["target_token_ids"],
                                         "logprobs": verl_lp, "logprobs_fp32": fp32_lp}
    results["meta"]["finished_utc"] = utc()
    results["meta"]["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    write_json(out / "recompute.json", results)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    length = sub.add_parser("length")
    length.add_argument("--model", required=True)
    length.add_argument("--max-model-len", type=int, required=True)
    length.add_argument("--inputs", required=True)
    length.add_argument("--out", required=True)
    length.add_argument("--tiers", default="a,b,c")
    length.add_argument("--episodes", action="store_true")
    recompute = sub.add_parser("recompute")
    recompute.add_argument("--model", required=True)
    recompute.add_argument("--trajectories", required=True)
    recompute.add_argument("--length-inputs", required=True)
    recompute.add_argument("--out", required=True)
    args = parser.parse_args()
    from agent_rl.utils.hash_seed import require_fixed_hash_seed

    require_fixed_hash_seed("run_7b_gpu.py")  # protocol §13.1 (run_7b.sh exports it before Python starts)
    for key in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "MICU_API_KEY", "ANTHROPIC_API_KEY"):
        os.environ.pop(key, None)
    return cmd_length(args) if args.cmd == "length" else cmd_recompute(args)


if __name__ == "__main__":
    sys.exit(main())
