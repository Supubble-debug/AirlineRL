"""Summarise a step-8 run directory locally (CPU only, no LLM, no GPU).

    python scripts/itsm/summarize_8.py --run-dir DIR --tokenizer-dir TOK --out FILE

Reports kept apart, as the protocol requires:
* A (E0-base dev diagnostic): per task c/n under the fuzzy DB training reward; separately the
  LLM-DB result and the NL result, each with its own "evaluated" count (``not_evaluated`` is
  never counted as 0); the joint DB+NL result only where both were evaluated.
* B (GRPO train diagnostic, not a dev score): per task successes in its G=4 group, the share
  of groups whose rewards are not all equal, all-0 and all-1 counts.
* Only complete groups enter A/B; discarded / cancelled groups are listed with reasons.
* Termination, truncation, invalid actions, tool business errors, lengths.
* Cost: the ledger's estimate (reservation prices) -- the provider's actual bill is NOT in the
  ledger and must be read from DeepSeek separately.
* 32k: counterfactual budget analysis along the 40960 trajectories (agent_rl.eval.itsm_length_counterfactual).
pass@k is given only as the unbiased estimator 1 - C(n-c,k)/C(n,k) with n, k stated.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agent_rl.eval.itsm_length_counterfactual import analyse  # noqa: E402
from agent_rl.utils.spend_ledger import SpendLedger  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def pass_at_k(n: int, c: int, k: int) -> float | None:
    if k > n:
        return None
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k) if n - c >= k else 0.0)


def summarise(run_dir: Path, tokenizer=None, budget_id: str = "step8_diagnostics") -> dict:
    groups = read_jsonl(run_dir / "groups.jsonl")
    trajectories = read_jsonl(run_dir / "trajectories.jsonl")
    attempts = read_jsonl(run_dir / "attempts.jsonl")
    complete = {g["group_id"] for g in groups if g["status"] == "complete"}
    kept = [t for t in trajectories if t["group_id"] in complete]
    by_group = collections.defaultdict(list)
    for t in kept:
        by_group[t["group_id"]].append(t)

    part_a, part_b = [], []
    for group in groups:
        if group["status"] != "complete":
            continue
        rows = sorted(by_group[group["group_id"]], key=lambda t: t["sample_index"])
        rewards = [t["reward_score"] for t in rows]
        n, c = len(rewards), sum(r == 1.0 for r in rewards)
        base = {"task_id": group["task_id"], "n": n, "successes_fuzzy": c}
        if group["part"] == "A":
            aux = [t["extra_fields"].get("itsm_aux_eval") or {} for t in rows]
            db = [a.get("db_llm", {}) for a in aux]
            nl = [a.get("nl", {}) for a in aux]
            db_eval = [d for d in db if d.get("status") == "evaluated"]
            nl_eval = [x for x in nl if x.get("status") == "evaluated"]
            joint = [(d.get("db_match"), x.get("nl_met")) for d, x in zip(db, nl) if d.get("status") == "evaluated" and x.get("status") == "evaluated"]
            part_a.append({**base, "pass_at_k_fuzzy": {k: pass_at_k(n, c, k) for k in (1, 2, 4)},
                           "llm_db": {"evaluated": len(db_eval), "not_evaluated": sum(d.get("status") == "not_evaluated" for d in db),
                                      "matches": sum(bool(d.get("db_match")) for d in db_eval)},
                           "nl": {"evaluated": len(nl_eval), "not_evaluated": sum(x.get("status") == "not_evaluated" for x in nl),
                                  "no_assertions": sum(x.get("status") == "no_assertions" for x in nl), "met": sum(bool(x.get("nl_met")) for x in nl_eval)},
                           "joint_db_llm_and_nl": {"evaluated": len(joint), "successes": sum(bool(d) and bool(m) for d, m in joint)}})
        else:
            part_b.append({**base, "rewards": rewards, "mixed": len(set(rewards)) > 1})

    terminations = collections.Counter(t["extra_fields"]["reward_extra_info"]["itsm_termination_reason"] for t in kept)
    invalid = collections.Counter(span.get("invalid_action", "").split(":")[0] for t in kept for span in t["extra_fields"]["itsm_turn_spans"] if "invalid_action" in span)
    errors = collections.Counter()
    if tokenizer is not None:
        for t in kept:
            for span in t["extra_fields"]["itsm_env_spans"]:
                text = tokenizer.decode(t["response_ids"][span["start"]:span["end"]], skip_special_tokens=False)
                for kind in ("domain_error", "unknown_tool", "invalid_arguments"):
                    errors[kind] += text.count(f"error({kind})")
    ledger = run_dir / "ledger.jsonl"
    b_groups = len(part_b)
    return {
        "A_e0_base_dev_diagnostic": {"tasks": part_a, "episodes": sum(r["n"] for r in part_a)},
        "B_grpo_train_diagnostic_not_a_dev_score": {
            "tasks": part_b, "groups": b_groups, "mixed_groups": sum(r["mixed"] for r in part_b),
            "mixed_share": (sum(r["mixed"] for r in part_b) / b_groups) if b_groups else None,
            "all_zero": sum(set(r["rewards"]) == {0.0} for r in part_b), "all_one": sum(set(r["rewards"]) == {1.0} for r in part_b)},
        "groups_not_complete": [{k: g[k] for k in ("group_id", "status", "sample_status")} for g in groups if g["status"] != "complete"],
        "attempt_status": dict(collections.Counter(a["status"] for a in attempts)),
        "infra_failures_by_stage": dict(collections.Counter(a.get("stage") for a in attempts if a["status"] == "infra_failure")),
        "terminations": dict(terminations), "invalid_actions": dict(invalid), "tool_errors_in_observations": dict(errors),
        "lengths": {"max_total_tokens": max((len(t["prompt_ids"]) + len(t["response_ids"]) for t in kept), default=None),
                    "max_response_tokens": max((len(t["response_ids"]) for t in kept), default=None)},
        "cost": {"ledger_estimate": SpendLedger.read_summary(ledger, budget_id) if ledger.exists() else None,
                 "actual_billing": "not in the ledger: read the DeepSeek bill for the run window separately"},
        "counterfactual_32k": analyse(kept, max_model_len=32768),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = None
    if args.tokenizer_dir:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    summary = summarise(args.run_dir, tokenizer)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("A_e0_base_dev_diagnostic", "counterfactual_32k")}, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
