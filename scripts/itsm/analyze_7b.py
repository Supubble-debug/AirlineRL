"""Step 7B: summarize the GPU run locally (CPU only, no GPU, no LLM).

Usage:
    python scripts/itsm/analyze_7b.py --run-dir DIR --out docs/itsm_step7b/summary.json

DIR is the synced run's ``step7b`` directory (phases.jsonl, gpu_*.csv, *.log,
len32k/, len40k/, recompute/). Log-prob comparisons use only mask=1 positions of
the IDENTICAL token sequence (rollout log-prob from vLLM vs the training-side
recompute of the same prefix and target token); environment placeholders are
never compared. Cross-setting generation agreement is a diagnostic: positions
are compared only up to the first divergence (same prefix), and at the
divergence only the two candidates' log-probs as reported in each setting's
top-5 are listed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path


def stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)

    def q(p: float) -> float:
        return ordered[min(len(ordered) - 1, max(0, math.ceil(p * len(ordered)) - 1))]

    return {"n": len(ordered), "mean": statistics.fmean(ordered), "median": statistics.median(ordered), "p90": q(0.9), "p99": q(0.99),
            "max": ordered[-1], "min": ordered[0]}


def gpu_peaks(run: Path) -> dict:
    peaks = {}
    for path in sorted(run.glob("gpu_*.csv")):
        used = []
        with path.open() as handle:
            for row in csv.reader(handle):
                try:
                    used.append((float(row[1]), float(row[2]), float(row[3])))
                except (IndexError, ValueError):
                    continue
        if used:
            peaks[path.stem[4:]] = {"peak_used_mib": max(u for u, _, _ in used), "total_mib": used[0][1],
                                    "peak_util_pct": max(x for _, _, x in used), "samples": len(used)}
    return peaks


def kv_lines(run: Path) -> dict:
    found = {}
    for name in ("len32k", "len40k"):
        path = run / f"{name}.log"
        if not path.exists():
            continue
        text = path.read_text(errors="replace")
        size = re.findall(r"GPU KV cache size: ([\d,]+) tokens", text)
        conc = re.findall(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", text)
        mem = re.findall(r"Available KV cache memory: ([\d.]+) GiB", text)
        weights = re.findall(r"Model loading took ([\d.]+) Gi?B", text)
        graph = re.findall(r"Graph capturing finished in ([\d.]+) secs?", text)
        compile_ = re.findall(r"torch\.compile takes ([\d.]+) s in total", text)
        found[name] = {"kv_cache_tokens": size[-1] if size else None, "max_concurrency": conc[-1] if conc else None,
                       "available_kv_gib": mem[-1] if mem else None, "model_loading_gib": weights[-1] if weights else None,
                       "graph_capture_s": graph[-1] if graph else None, "torch_compile_s": compile_[-1] if compile_ else None}
    return found


def length_summary(run: Path) -> dict:
    out = {}
    for name in ("len32k", "len40k"):
        path = run / name / "length_results.json"
        if not path.exists():
            continue
        engine = json.loads((run / name / "engine.json").read_text()) if (run / name / "engine.json").exists() else {}
        rows = {}
        for tier, r in json.loads(path.read_text()).items():
            rows[tier] = {k: r.get(k) for k in ("ok", "natural", "prompt_tokens", "output_tokens", "total_tokens", "finish_reason", "stop_reason",
                                                 "ended_with_stop_token", "first_output_s", "total_s", "wall_s", "logprob_check",
                                                 "logprob_count_matches", "parse_action", "error")}
            rows[tier]["output_text_head"] = (r.get("output_text") or "")[:400]
        out[name] = {"engine_init_s": engine.get("engine_init_s"), "cache_config": engine.get("cache_config"), "tiers": rows}
    return out


def cross_setting(run: Path) -> dict:
    a_path, b_path = run / "len32k" / "length_results.json", run / "len40k" / "length_results.json"
    if not (a_path.exists() and b_path.exists()):
        return {}
    r32, r40 = json.loads(a_path.read_text()), json.loads(b_path.read_text())
    result = {}
    for tier in sorted(set(r32) & set(r40)):
        x, y = r32[tier], r40[tier]
        if not (x.get("ok") and y.get("ok")):
            result[tier] = {"comparable": False}
            continue
        tx, ty = x["output_token_ids"], y["output_token_ids"]
        n = min(len(tx), len(ty))
        first = next((i for i in range(n) if tx[i] != ty[i]), None)
        same_prefix = n if first is None else first
        diffs = [abs(x["output_logprobs"][i] - y["output_logprobs"][i]) for i in range(same_prefix)]
        entry = {"identical": tx == ty, "tokens_32k": len(tx), "tokens_40k": len(ty), "first_divergence": first,
                 "abs_logprob_diff_before_divergence": stats(diffs)}
        if first is not None:
            top32, top40 = x["output_top_logprobs"][first], y["output_top_logprobs"][first]
            entry["at_divergence"] = {
                "token_32k": tx[first], "token_40k": ty[first],
                "logprob_in_32k": {"own": top32.get(str(tx[first])), "other": top32.get(str(ty[first]))},
                "logprob_in_40k": {"own": top40.get(str(ty[first])), "other": top40.get(str(tx[first]))},
            }
        result[tier] = entry
    return result


def episodes_summary(run: Path) -> list:
    path = run / "len40k" / "episodes.json"
    if not path.exists():
        return []
    rows = []
    for e in json.loads(path.read_text()):
        row = {"purpose": e["purpose"], "task_id": e["task_id"], "ok": e["ok"], "wall_s": e.get("wall_s"), "error": e.get("error")}
        if e["ok"]:
            info = e["extra_fields"]["reward_extra_info"]
            turns = e["extra_fields"]["itsm_turn_spans"]
            row |= {
                "termination": info["itsm_termination_reason"], "reward": e["reward_score"], "db_reward": info["itsm_db_reward"],
                "agent_turns": info["itsm_agent_turns"], "step_count": info["itsm_step_count"], "policy_turns": len(turns),
                "prompt_tokens": info["itsm_prompt_tokens"], "policy_tokens": info["itsm_policy_tokens"], "response_tokens": info["itsm_response_tokens"],
                "invalid_actions": [t["invalid_action"] for t in turns if "invalid_action" in t],
                "action_kinds": [t.get("action_kind") for t in turns],
                "tools_called": [re.search(r'"name":\s*"([^"]+)"', txt).group(1) if re.search(r'"name":\s*"([^"]+)"', txt) else None
                                 for txt in e["turn_texts"]],
                "max_turn_tokens": max(t["end"] - t["start"] for t in turns),
                "logprob_check": e["logprob_check"],
                "requests": len(e["requests"]),
                "request_total_s": round(sum(r["total_s"] or 0 for r in e["requests"]), 2),
                "request_max_prompt_tokens": max(r["prompt_tokens"] for r in e["requests"]),
                "finish_reasons": sorted({r["finish_reason"] for r in e["requests"]}),
            }
        rows.append(row)
    return rows


def logprob_comparison(run: Path) -> dict:
    episodes_path, recompute_path = run / "len40k" / "episodes.json", run / "recompute" / "recompute.json"
    if not (episodes_path.exists() and recompute_path.exists()):
        return {}
    episodes = {e["task_id"]: e for e in json.loads(episodes_path.read_text()) if e["ok"]}
    recompute = json.loads(recompute_path.read_text())
    per_episode, all_diffs, all_abs = [], [], []
    for r in recompute["episodes"]:
        e = episodes[r["task_id"]]
        prompt_len = len(e["prompt_ids"])
        rollout = [e["response_logprobs"][p - prompt_len] for p in r["mask1_positions"]]
        assert all(e["response_mask"][p - prompt_len] == 1 for p in r["mask1_positions"])
        train = r["recompute_logprobs"]
        diffs = [t - s for t, s in zip(train, rollout)]  # log pi_train - log pi_rollout (log importance ratio)
        abs_d = [abs(d) for d in diffs]
        all_diffs += diffs
        all_abs += abs_d
        spans = e["extra_fields"]["itsm_turn_spans"]
        position_index = {p - prompt_len: i for i, p in enumerate(r["mask1_positions"])}
        per_turn = []
        for t in spans:
            idx = [position_index[k] for k in range(t["start"], t["end"]) if k in position_index]
            per_turn.append({"turn": t["turn_index"], "tokens": len(idx), "sum_rollout": sum(rollout[i] for i in idx),
                             "sum_train": sum(train[i] for i in idx), "sum_diff": sum(diffs[i] for i in idx)})
        fp32_vs_verl = [abs(a - b) for a, b in zip(r["recompute_logprobs_fp32"], train)]
        per_episode.append({
            "task_id": r["task_id"], "mask1_tokens": len(diffs), "sequence_tokens": r["sequence_tokens"],
            "abs_diff": stats(abs_d), "signed_diff_mean": statistics.fmean(diffs) if diffs else None,
            "sequence_sum_rollout": sum(rollout), "sequence_sum_train": sum(train), "sequence_sum_diff": sum(diffs),
            "token_ratio_exp_diff": stats([math.exp(d) for d in diffs]),
            "share_abs_gt": {str(k): sum(a > k for a in abs_d) / len(abs_d) for k in (0.01, 0.1, 0.5, 1.0)} if abs_d else {},
            "per_turn": per_turn,
            "verl_fn_vs_fp32_logsoftmax_abs": stats(fp32_vs_verl),
        })
    gold = {}
    for name, g in recompute.get("gold_targets", {}).items():
        gold[name] = {"prompt_tokens": g["prompt_tokens"], "target_tokens": len(g["logprobs"]), "logprobs": g["logprobs"],
                      "sum": sum(g["logprobs"]), "mean": statistics.fmean(g["logprobs"]) if g["logprobs"] else None}
    return {"meta": recompute["meta"], "definition": "diff = log p_train(token | prefix) - log p_rollout(token | prefix), mask=1 positions only, same sequence",
            "pooled_abs_diff": stats(all_abs), "pooled_signed_diff_mean": statistics.fmean(all_diffs) if all_diffs else None,
            "pooled_share_abs_gt": {str(k): sum(a > k for a in all_abs) / len(all_abs) for k in (0.01, 0.1, 0.5, 1.0)} if all_abs else {},
            "episodes": per_episode, "gold_target_diagnostic": gold}


def phases(run: Path) -> list:
    path = run / "phases.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir
    summary = {"phases": phases(run), "gpu_peaks": gpu_peaks(run), "kv_and_engine_logs": kv_lines(run), "length": length_summary(run),
               "cross_setting_diagnostic": cross_setting(run), "episodes": episodes_summary(run), "logprobs": logprob_comparison(run)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: (v if k in ("phases", "gpu_peaks", "kv_and_engine_logs", "cross_setting_diagnostic") else "...") for k, v in summary.items()}, indent=1)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
