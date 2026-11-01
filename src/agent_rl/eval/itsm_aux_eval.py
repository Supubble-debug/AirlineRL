"""Auxiliary dev evaluation inside the env worker: LLM DB text match and NL assertions (D7).

Runs in the ``ItsmEnvWorker`` child after the episode was scored with the training reward
(fuzzy DB, D1b), on the SAME final DB and the SAME trajectory, before the child (and its DB)
goes away. It never replaces the fuzzy result; results are stored separately:

* ``db_llm``: Bench2's DB comparison with ``strategy="llm"``; the free-text pairs that are not
  exactly equal go to one D7 judge call (``build_text_judge_request`` /
  ``parse_text_judge_response``, the strict parser of step 3);
* ``nl``: Bench2's NL-assertion prompt (``evaluate_nl_assertions`` format), parsed strictly:
  one result per assertion, in order, ``expectedOutcome`` equal to the assertion and
  ``metExpectation`` a boolean. (Bench2's own parser turns an unparseable reply into "not
  met"; here that is a protocol failure, never a verdict.)

Failures (call error, non-``stop`` finish, reasoning returned, protocol error) are retried by
re-judging the same inputs, at most ``max_attempts`` (4) in total; the policy and the user
are never called again. Valid negative verdicts are final and never retried. When attempts
run out the part is ``not_evaluated`` -- never scored 0. Budget refusals propagate (the
runner stops the group and opens no new one).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any

from enterprise_worlds.data_model.message import SystemMessage, UserMessage, render_trajectory
from enterprise_worlds.evaluator import evaluator_nl, text_match_strategy
from enterprise_worlds.evaluator.text_match_strategy import TextMatchConfig
from enterprise_worlds.utils.llm_utils import to_litellm_messages

from agent_rl.envs.itsm_scoring import (
    JudgeProtocolError,
    build_scoring_environment,
    build_text_judge_request,
    collect_db_differences,
    parse_text_judge_response,
    replay_gold_actions,
    task_clock,
)
from agent_rl.llm.deepseek_chat import DEEPSEEK_BASE_URL, DeepSeekChat, LlmCallFailed

JUDGE_MODEL = "deepseek/deepseek-v4-pro"  # D7
JUDGE_ARGS = {"temperature": 0.0, "max_tokens": 4096, "extra_body": {"thinking": {"type": "disabled"}}}  # D7 (protocol §14.3)
MAX_ATTEMPTS = 4  # protocol §12: first attempt + 3 retries
TEXT_PROMPT_SHA256 = "2e2350908ca2b54271cc96dab31c414a14413427af177a2fbcbcd1df272761d4"  # protocol §14.3
NL_PROMPT_SHA256 = "295597be7c47347429e8adf6f236e62eae335d7ba587cdc221e866f4dac7927f"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def check_prompts() -> None:
    if _sha(text_match_strategy.JUDGE_SYSTEM_PROMPT) != TEXT_PROMPT_SHA256 or _sha(evaluator_nl.JUDGE_SYSTEM_PROMPT) != NL_PROMPT_SHA256:
        raise RuntimeError("Bench2 judge prompts differ from the D7 pins (protocol §14.3)")


class NlProtocolError(ValueError):
    def __init__(self, kind: str, detail: str) -> None:
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail


def nl_request_messages(trajectory: list, assertions: list[str]) -> list[dict[str, Any]]:
    """Exactly the messages Bench2's ``evaluate_nl_assertions`` sends."""
    user_prompt = f"conversation:\n{render_trajectory(trajectory)}\n\nexpectedOutcomes:\n{json.dumps(assertions, indent=2)}"
    return to_litellm_messages([SystemMessage(content=evaluator_nl.JUDGE_SYSTEM_PROMPT), UserMessage(content=user_prompt)])


def parse_nl_response(content: str | None, assertions: list[str]) -> list[dict[str, Any]]:
    try:
        data = json.loads(evaluator_nl._extract_json(content or ""))
    except (json.JSONDecodeError, TypeError) as error:
        raise NlProtocolError("unparseable", str(error)) from error
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list):
        raise NlProtocolError("unparseable", "no 'results' list")
    if len(results) != len(assertions):
        raise NlProtocolError("count", f"{len(results)} results for {len(assertions)} assertions")
    checks = []
    for index, (item, assertion) in enumerate(zip(results, assertions)):
        if not isinstance(item, dict):
            raise NlProtocolError("invalid_item", f"result {index} is not an object")
        if " ".join(str(item.get("expectedOutcome", "")).split()) != " ".join(assertion.split()):
            raise NlProtocolError("outcome_mismatch", f"result {index} does not restate assertion {index}")
        met = item.get("metExpectation")
        if not isinstance(met, bool):
            raise NlProtocolError("invalid_met", f"result {index} metExpectation={met!r}")
        checks.append({"assertion": assertion, "met": met, "reasoning": item.get("reasoning") if isinstance(item.get("reasoning"), str) else None})
    return checks


@dataclass
class AuxEvaluator:
    """Called in the child as ``evaluator(env)`` (``ItsmEnvWorker.aux_evaluate``)."""

    chat: DeepSeekChat
    max_attempts: int = MAX_ATTEMPTS
    log: list[dict[str, Any]] = field(default_factory=list)

    def _judge(self, part: str, messages: list[dict[str, Any]], parse) -> tuple[Any, list[dict[str, Any]]]:
        attempts = []
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = self.chat.complete(JUDGE_MODEL, messages, temperature=JUDGE_ARGS["temperature"], max_tokens=JUDGE_ARGS["max_tokens"],
                                            extra_body=JUDGE_ARGS["extra_body"])
                row = {"attempt": attempt, "request_id": result.request_id, "finish_reason": result.finish_reason, "charged_usd": result.charged_usd}
                if result.finish_reason != "stop":
                    raise JudgeProtocolError("finish_reason", f"finish_reason={result.finish_reason!r}")
                if result.reasoning_present or result.reasoning_tokens:
                    raise JudgeProtocolError("thinking_not_disabled", f"reasoning returned ({result.reasoning_tokens} tokens)")
                parsed = parse(result.content)
            except LlmCallFailed as error:
                if error.kind == "model_mismatch":
                    raise  # non-retryable: the manager is halted, the run stops
                attempts.append({"attempt": attempt, "stage": f"{part}_judge_call", "kind": error.kind, "detail": error.detail[:300]})
                continue
            except (JudgeProtocolError, NlProtocolError) as error:
                attempts.append({**row, "stage": f"{part}_judge_protocol", "kind": error.kind, "detail": str(error)[:300]})
                continue
            attempts.append({**row, "outcome": "valid"})
            return parsed, attempts  # a valid verdict (positive or negative) is final
        return None, attempts

    def __call__(self, env) -> dict[str, Any]:
        check_prompts()
        task, final_environment, trajectory = env.auxiliary_inputs()
        text_match = TextMatchConfig(strategy="llm", llm=JUDGE_MODEL, llm_args=dict(JUDGE_ARGS))
        gold = replay_gold_actions(task)
        with task_clock(task):
            baseline = build_scoring_environment(task, "baseline").tools.db
            mismatches, pending = collect_db_differences(gold.tools.db, final_environment.tools.db, baseline, text_match)
        db: dict[str, Any] = {"judge_model": JUDGE_MODEL, "prompt_sha256": TEXT_PROMPT_SHA256, "structural_mismatches": mismatches[:20],
                              "pending_pairs": len(pending), "final_db_hash": final_environment.get_db_hash()}
        if not pending:
            db |= {"status": "evaluated", "judge_calls": 0, "db_match": not mismatches, "attempts": []}
        else:
            request = build_text_judge_request(pending, text_match)
            messages = to_litellm_messages(request["messages"])
            verdicts, attempts = self._judge("db_text", messages, lambda content: parse_text_judge_response(content, len(pending)))
            db["attempts"] = attempts
            if verdicts is None:
                db |= {"status": "not_evaluated", "db_match": None}
            else:
                rejected = [pair["key"] for pair, (equivalent, _) in zip(pending, verdicts) if not equivalent]
                db |= {"status": "evaluated", "db_match": not mismatches and not rejected, "judge_rejected": rejected,
                       "verdicts": [{"key": p["key"], "equivalent": e} for p, (e, _) in zip(pending, verdicts)]}
        assertions = list(task.evaluation_criteria.nl_assertions)
        nl: dict[str, Any] = {"judge_model": JUDGE_MODEL, "prompt_sha256": NL_PROMPT_SHA256, "assertions": len(assertions)}
        if not assertions:
            nl |= {"status": "no_assertions", "nl_met": None, "attempts": []}
        else:
            checks, attempts = self._judge("nl", nl_request_messages(trajectory, assertions), lambda content: parse_nl_response(content, assertions))
            nl["attempts"] = attempts
            nl |= ({"status": "not_evaluated", "nl_met": None} if checks is None
                   else {"status": "evaluated", "nl_met": all(c["met"] for c in checks), "checks": checks})
        return {"db_llm": db, "nl": nl, "requests": list(self.chat.calls)}


def make_aux_evaluator(budget_address: str, budget_authkey: bytes, group_id: str, timeout_s: float = 120.0) -> AuxEvaluator:
    """Picklable worker ``aux_evaluator_factory`` (via functools.partial); runs in the env child."""
    from agent_rl.utils.budget_manager import BudgetClient

    chat = DeepSeekChat(api_key=os.environ.get("DEEPSEEK_API_KEY", ""), budget=BudgetClient(budget_address, budget_authkey),
                        group_id=group_id, role="judge", timeout_s=timeout_s, tag="aux_judge",
                        base_url=os.environ.get("ITSM_DEEPSEEK_BASE_URL", DEEPSEEK_BASE_URL))
    return AuxEvaluator(chat=chat)
