"""Step 8 minimal API review (D5 simulator, D7 judges): <= 20 requests, <= 1 USD. NOT approved yet.

    scripts/itsm/itsm_python.sh scripts/itsm/review_8_api.py --out DIR              # dry run: plan + reservations, nothing sent
    scripts/itsm/itsm_python.sh scripts/itsm/review_8_api.py --out DIR --confirm-billing

Same clients as the diagnostics (DeepSeekChat: one ledger reservation per HTTP request, no SDK
retries; LedgeredUserSimulator; the D7 request builders and strict parsers), own budget id and
ledger (configs/itsm/step8/plan.json "review"). Checks per request: returned model, thinking off
(no reasoning content/tokens), finish_reason, parseable reply, the fixed cases judged as
expected -- the first failure stops the review at once, saves the reason and exits non-zero.
Repeats measure agreement (a diagnostic only, not assumed deterministic). Never retries.
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


class ReviewStop(RuntimeError):
    """A required check failed: stop at once (nothing further is sent)."""


def exit_code(report: dict, *, billing: bool) -> int:
    """0 only for a dry run or a review whose required checks all passed."""
    if not billing:
        return 0
    return 0 if report.get("stop_reason") is None else 1


def run(out: Path, *, billing: bool, http=None) -> dict:
    """Required checks (any failure -> stop now, reason saved, non-zero exit):
    returned model = requested; thinking off (no reasoning content/tokens); finish_reason stop;
    reply/verdict parseable (strict parsers); the fixed cases judged as expected.
    Diagnostic only: agreement between repeated simulator replies."""
    from enterprise_worlds.data_model.message import AssistantMessage, UserMessage
    from enterprise_worlds.evaluator.text_match_strategy import TextMatchConfig
    from enterprise_worlds.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE
    from enterprise_worlds.utils.llm_utils import to_litellm_messages

    from agent_rl.data import itsm_task_sets
    from agent_rl.envs.itsm_ledgered_user import LedgeredUserSimulator, UserSimContractError
    from agent_rl.envs.itsm_scoring import JudgeProtocolError, build_text_judge_request, parse_text_judge_response
    from agent_rl.eval.itsm_aux_eval import JUDGE_ARGS, JUDGE_MODEL, NlProtocolError, check_prompts, nl_request_messages, parse_nl_response
    from agent_rl.llm.deepseek_chat import DeepSeekChat, LlmCallFailed
    from agent_rl.rollout.itsm_step8 import load_plan, load_prices
    from agent_rl.utils.budget_manager import BudgetManager, BudgetUnavailable
    from agent_rl.utils.spend_ledger import BudgetExceeded, ReservationTooSmall, SpendLedger

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

    def fail(row: dict, check: str, detail: str) -> None:
        row |= {"ok": False, "failed_check": check, "detail": detail[:500]}
        report["results"].append(row)
        raise ReviewStop(f"{row['check']}: {check}: {detail[:300]}")

    def judge_checks(row: dict, result) -> None:
        if result.finish_reason != "stop":
            fail(row, "finish_reason", f"finish_reason={result.finish_reason!r}")
        if result.reasoning_present or result.reasoning_tokens:
            fail(row, "thinking_not_disabled", f"reasoning returned ({result.reasoning_tokens} tokens)")

    with SpendLedger(out / "ledger.jsonl", cap_usd=review["cap_usd"], max_billed_requests=review["max_billed_requests"], budget_id=review["budget_id"]) as ledger:
        manager = BudgetManager(ledger, prices)
        manager.open_group("review", review["cap_usd"], {"purpose": "step8 API review"})

        clients: list = []

        def chat(role):
            client = DeepSeekChat(api_key=key, budget=manager, group_id="review", role=role, http=http, tag=f"review:{role}")
            clients.append(client)
            return client

        try:
            for task_id in tasks:
                scenario = itsm_task_sets.resolve_task(task_id, split_version=plan["split_version"]).scenario
                replies = []
                for _ in range(inputs["repeats"]["simulator"]):
                    user = LedgeredUserSimulator(scenario, chat=chat("simulator"))
                    row = {"check": "simulator", "task_id": task_id}
                    try:
                        message, _ = user.generate_next_message(AssistantMessage(content=DEFAULT_FIRST_AGENT_MESSAGE), user.get_init_state())
                    except UserSimContractError as error:  # length cut, finish reason, thinking, empty
                        row["call"] = user.chat.calls[-1] if user.chat.calls else None
                        fail(row, error.kind, str(error))
                    except LlmCallFailed as error:  # model mismatch, HTTP/transport, missing choices
                        row["call"] = user.chat.calls[-1] if user.chat.calls else None
                        fail(row, error.kind, str(error))
                    replies.append(message.content)
                    report["results"].append({**row, "ok": True, "call": user.chat.calls[-1],
                                              "checks": {"returned_model_matches": True, "finish_reason_stop": True, "thinking_off": True, "non_empty_reply": True}})
                # diagnostic only: temperature 0 is not assumed to be deterministic
                report["results"].append({"check": "simulator_agreement", "task_id": task_id, "replies": len(replies),
                                          "distinct": len(set(replies)), "diagnostic_only": True})
            pairs = inputs["text_pairs"]
            request = build_text_judge_request(pairs, TextMatchConfig(strategy="llm", llm=JUDGE_MODEL, llm_args=dict(JUDGE_ARGS)))
            for _ in range(inputs["repeats"]["text_judge"]):
                client, row = chat("judge"), {"check": "text_judge"}
                try:
                    result = client.complete(JUDGE_MODEL, to_litellm_messages(request["messages"]), temperature=0.0, max_tokens=JUDGE_ARGS["max_tokens"], extra_body=JUDGE_ARGS["extra_body"])
                except LlmCallFailed as error:
                    row["call"] = client.calls[-1] if client.calls else None
                    fail(row, error.kind, str(error))
                row["call"] = client.calls[-1]
                judge_checks(row, result)
                try:
                    verdicts = [e for e, _ in parse_text_judge_response(result.content, len(pairs))]
                except JudgeProtocolError as error:
                    fail(row, "protocol", str(error))
                expected = [p["expected_equivalent"] for p in pairs]
                if verdicts != expected:
                    fail(row | {"verdicts": verdicts}, "fixed_case_verdict", f"got {verdicts}, expected {expected}")
                report["results"].append({**row, "ok": True, "verdicts": verdicts,
                                          "checks": {"returned_model_matches": True, "finish_reason_stop": True, "thinking_off": True, "parseable": True, "fixed_cases_as_expected": True}})
            nl = inputs["nl"]
            conversation = [AssistantMessage(content=m["content"]) if m["role"] == "assistant" else UserMessage(content=m["content"]) for m in nl["conversation"]]
            for _ in range(inputs["repeats"]["nl_judge"]):
                client, row = chat("judge"), {"check": "nl_judge"}
                try:
                    result = client.complete(JUDGE_MODEL, nl_request_messages(conversation, nl["assertions"]), temperature=0.0, max_tokens=JUDGE_ARGS["max_tokens"], extra_body=JUDGE_ARGS["extra_body"])
                except LlmCallFailed as error:
                    row["call"] = client.calls[-1] if client.calls else None
                    fail(row, error.kind, str(error))
                row["call"] = client.calls[-1]
                judge_checks(row, result)
                try:
                    met = [c["met"] for c in parse_nl_response(result.content, nl["assertions"])]
                except NlProtocolError as error:
                    fail(row, "protocol", str(error))
                if met != nl["expected_met"]:
                    fail(row | {"met": met}, "fixed_case_verdict", f"got {met}, expected {nl['expected_met']}")
                report["results"].append({**row, "ok": True, "met": met,
                                          "checks": {"returned_model_matches": True, "finish_reason_stop": True, "thinking_off": True, "parseable": True, "fixed_cases_as_expected": True}})
        except ReviewStop as stop:
            report["stop_reason"] = str(stop)
        except (BudgetExceeded, BudgetUnavailable, ReservationTooSmall) as error:
            report["stop_reason"] = f"{type(error).__name__}: {error}"
        finally:
            manager.close_group("review", reason="review finished" if report["stop_reason"] is None else "review stopped")
            report["budget"] = manager.summary()
            calls = [record for client in clients for record in client.calls]
            report["requests"] = calls
            report["http_requests"] = len(calls)  # every reserved request was handed to the HTTP client
            report["ledger_events"] = [json.loads(line) for line in (out / "ledger.jsonl").read_text().splitlines()]
            report["unverified"] = unverified_items(calls)
    return report


def unverified_items(calls: list[dict]) -> list[str]:
    """What this review does NOT establish (listed in the report, not inferred away)."""
    sim_out = [((c.get("usage") or {}).get("completion_tokens")) for c in calls if c.get("model") == "deepseek/deepseek-flash"]
    judge_out = [((c.get("usage") or {}).get("completion_tokens")) for c in calls if c.get("model") == "deepseek/deepseek-v4-pro"]
    items = []
    if not any(c.get("finish_reason") == "length" for c in calls if c.get("model") == "deepseek/deepseek-flash"):
        top = max([x for x in sim_out if isinstance(x, int)], default=None)
        items.append(f"D5 max_tokens=1024 truncation behaviour not observed (longest simulator output {top} tokens); short outputs do not show the cap is enforced")
    top = max([x for x in judge_out if isinstance(x, int)], default=None)
    items.append(f"D7 max_tokens=4096 not approached (longest judge output {top} tokens)")
    items += [
        "thinking off is shown only for these inputs; long multi-turn simulator histories were not sent",
        "simulator behaviour inside real episodes (multi-turn, real agent messages, STOP timing) is not reviewed here",
        "judge agreement with humans is not measured (2 fixed text pairs and 1 NL case only)",
        "latency/rate limits under concurrency 8 not tested (requests were sequential)",
        "the ledger estimate is not the DeepSeek bill; the bill must be read separately",
    ]
    return items


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
    return exit_code(report, billing=args.confirm_billing)


if __name__ == "__main__":
    sys.exit(main())
