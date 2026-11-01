"""Step 8 minimal API review (D5 simulator, D7 judges): <= 20 requests, <= 1 USD. NOT approved yet.

    scripts/itsm/itsm_python.sh scripts/itsm/review_8_api.py --out DIR              # dry run: plan + reservations, nothing sent
    scripts/itsm/itsm_python.sh scripts/itsm/review_8_api.py --out DIR --confirm-billing

Same clients as the diagnostics (DeepSeekChat: one ledger reservation per HTTP request, no SDK
retries; LedgeredUserSimulator; the D7 request builders and strict parsers), own budget id and
ledger (configs/itsm/step8/plan.json "review"). Checks per request: returned model, thinking off
(no reasoning content/tokens), finish_reason, output length; repeats measure agreement (not
assumed deterministic). Stops at the first budget refusal; never retries automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agent_rl.utils.hash_seed import require_fixed_hash_seed  # noqa: E402

INPUTS = REPO_ROOT / "configs" / "itsm" / "step8" / "review_inputs.json"


def planned_requests(inputs: dict) -> int:
    r = inputs["repeats"]
    return 3 * r["simulator"] + r["text_judge"] + r["nl_judge"]


def run(out: Path, *, billing: bool, http=None) -> dict:
    from enterprise_worlds.data_model.message import AssistantMessage, UserMessage
    from enterprise_worlds.evaluator.text_match_strategy import TextMatchConfig
    from enterprise_worlds.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE
    from enterprise_worlds.utils.llm_utils import to_litellm_messages

    from agent_rl.data import itsm_task_sets
    from agent_rl.envs.itsm_ledgered_user import LedgeredUserSimulator, UserSimContractError
    from agent_rl.envs.itsm_scoring import build_text_judge_request, parse_text_judge_response
    from agent_rl.eval.itsm_aux_eval import JUDGE_ARGS, JUDGE_MODEL, check_prompts, nl_request_messages, parse_nl_response
    from agent_rl.llm.deepseek_chat import DeepSeekChat, LlmCallFailed
    from agent_rl.rollout.itsm_step8 import load_plan, load_prices
    from agent_rl.utils.budget_manager import BudgetManager
    from agent_rl.utils.spend_ledger import BudgetExceeded, SpendLedger

    check_prompts()
    plan, prices, inputs = load_plan(), load_prices(), json.loads(INPUTS.read_text(encoding="utf-8"))
    review = plan["review"]
    n = planned_requests(inputs)
    if n > review["max_billed_requests"]:
        raise SystemExit(f"{n} planned requests exceed the review limit")
    tasks = sorted(itsm_task_sets.training_task_ids(purpose="grpo", usage_version=plan["usage_version"]))[:3]
    report = {"planned_requests": n, "tasks": tasks, "results": [], "stop_reason": None}
    if not billing:
        report["stop_reason"] = "dry run: nothing sent without --confirm-billing"
        return report
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    out.mkdir(parents=True, exist_ok=True)
    with SpendLedger(out / "ledger.jsonl", cap_usd=review["cap_usd"], max_billed_requests=review["max_billed_requests"], budget_id=review["budget_id"]) as ledger:
        manager = BudgetManager(ledger, prices)
        manager.open_group("review", review["cap_usd"], {"purpose": "step8 API review"})

        def chat(role):
            return DeepSeekChat(api_key=key, budget=manager, group_id="review", role=role, http=http, tag=f"review:{role}")

        try:
            for task_id in tasks:
                scenario = itsm_task_sets.resolve_task(task_id, split_version=plan["split_version"]).scenario
                replies = []
                for _ in range(inputs["repeats"]["simulator"]):
                    user = LedgeredUserSimulator(scenario, chat=chat("simulator"))
                    try:
                        message, _ = user.generate_next_message(AssistantMessage(content=DEFAULT_FIRST_AGENT_MESSAGE), user.get_init_state())
                        replies.append(message.content)
                        report["results"].append({"check": "simulator", "task_id": task_id, "ok": True, "call": user.chat.calls[-1]})
                    except (UserSimContractError, LlmCallFailed) as error:
                        report["results"].append({"check": "simulator", "task_id": task_id, "ok": False, "error": str(error)[:300], "call": user.chat.calls[-1]})
                report["results"].append({"check": "simulator_agreement", "task_id": task_id, "replies": len(replies), "distinct": len(set(replies))})
            pairs = inputs["text_pairs"]
            request = build_text_judge_request(pairs, TextMatchConfig(strategy="llm", llm=JUDGE_MODEL, llm_args=dict(JUDGE_ARGS)))
            for _ in range(inputs["repeats"]["text_judge"]):
                client = chat("judge")
                try:
                    result = client.complete(JUDGE_MODEL, to_litellm_messages(request["messages"]), temperature=0.0, max_tokens=JUDGE_ARGS["max_tokens"], extra_body=JUDGE_ARGS["extra_body"])
                    verdicts = [e for e, _ in parse_text_judge_response(result.content, len(pairs))]
                    report["results"].append({"check": "text_judge", "ok": result.finish_reason == "stop" and not result.reasoning_present, "verdicts": verdicts,
                                              "as_expected": verdicts == [p["expected_equivalent"] for p in pairs], "call": client.calls[-1]})
                except Exception as error:  # noqa: BLE001 - recorded, review continues
                    if isinstance(error, BudgetExceeded):
                        raise
                    report["results"].append({"check": "text_judge", "ok": False, "error": f"{type(error).__name__}: {error}"[:300], "call": client.calls[-1] if client.calls else None})
            nl = inputs["nl"]
            conversation = [AssistantMessage(content=m["content"]) if m["role"] == "assistant" else UserMessage(content=m["content"]) for m in nl["conversation"]]
            for _ in range(inputs["repeats"]["nl_judge"]):
                client = chat("judge")
                try:
                    result = client.complete(JUDGE_MODEL, nl_request_messages(conversation, nl["assertions"]), temperature=0.0, max_tokens=JUDGE_ARGS["max_tokens"], extra_body=JUDGE_ARGS["extra_body"])
                    met = [c["met"] for c in parse_nl_response(result.content, nl["assertions"])]
                    report["results"].append({"check": "nl_judge", "ok": result.finish_reason == "stop" and not result.reasoning_present, "met": met,
                                              "as_expected": met == nl["expected_met"], "call": client.calls[-1]})
                except Exception as error:  # noqa: BLE001
                    if isinstance(error, BudgetExceeded):
                        raise
                    report["results"].append({"check": "nl_judge", "ok": False, "error": f"{type(error).__name__}: {error}"[:300], "call": client.calls[-1] if client.calls else None})
        except BudgetExceeded as error:
            report["stop_reason"] = f"BudgetExceeded: {error}"
        finally:
            manager.close_group("review", reason="review finished")
            report["budget"] = manager.summary()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--confirm-billing", action="store_true")
    args = parser.parse_args()
    require_fixed_hash_seed("review_8_api.py")
    if args.confirm_billing and not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("DEEPSEEK_API_KEY is not set")
    report = run(args.out, billing=args.confirm_billing)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "review_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("planned_requests", "tasks", "stop_reason")}, indent=1))
    return 0 if report["stop_reason"] in (None,) or not args.confirm_billing else 1


if __name__ == "__main__":
    sys.exit(main())
