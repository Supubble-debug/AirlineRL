"""DeepSeek chat client for D5/D7 with one ledger reservation per HTTP request (step 8).

Why not litellm: litellm (1.82.6) wraps the call in its own retry/fallback layer
(``num_retries`` / ``max_retries`` / module-level settings), so a request could be resent
outside our ledger. This client sends exactly one HTTP POST per ``complete`` call through a
plain ``httpx.Client`` (no retries), after reserving it. The JSON body equals what litellm
1.82.6 sends for ``deepseek/<model>`` with ``extra_body`` (pinned by
``tests/unit/test_itsm_step8_budget.py``): ``{"model", "messages", "temperature",
"max_tokens", **extra_body}`` to ``https://api.deepseek.com/beta/chat/completions``.

Accounting: input bound = UTF-8 bytes of the body + 64 (byte-level BPE never yields more
tokens than bytes), output bound = ``max_tokens``. A timeout, transport error, non-200
status, unparseable body or missing usage books the FULL reservation (the request may have
been billed); otherwise the reported usage is booked. A returned model that differs from the
requested one is settled and then HALTS the budget manager (non-retryable); an actual cost
above the reservation halts it in the ledger. Any retry is a new ``complete`` call,
so it is reserved again. Secrets never appear in errors.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

DEEPSEEK_BASE_URL = "https://api.deepseek.com/beta"  # litellm's default for the deepseek provider
INPUT_OVERHEAD_TOKENS = 64


class Budget(Protocol):
    def reserve(self, *, group_id: str, role: str, input_bound_tokens: int, max_output_tokens: int, detail: dict) -> dict: ...
    def settle(self, *, request_id: str, input_tokens: int | None, output_tokens: int | None, usage: Any) -> float: ...
    def settle_unknown(self, *, request_id: str, note: str) -> float: ...
    def halt(self, *, reason: str) -> None: ...


class LlmCallFailed(RuntimeError):
    """The request did not yield a usable response (it was booked; it may have been billed)."""

    def __init__(self, kind: str, detail: str) -> None:
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail


@dataclass
class ChatResult:
    content: str | None
    finish_reason: str | None
    returned_model: str | None
    usage: dict[str, Any] | None
    reasoning_present: bool
    reasoning_tokens: int | None
    request_id: str
    reserved_usd: float
    charged_usd: float
    latency_s: float
    raw_choice: dict[str, Any] = field(repr=False, default_factory=dict)


def body_for(model: str, messages: list[dict[str, Any]], *, temperature: float, max_tokens: int, extra_body: dict[str, Any] | None) -> dict[str, Any]:
    if not model.startswith("deepseek/"):
        raise ValueError(f"expected a deepseek/<model> route, got {model!r}")
    return {"model": model.split("/", 1)[1], "messages": messages, "temperature": temperature, "max_tokens": max_tokens, **(extra_body or {})}


def input_bound_tokens(body: dict[str, Any]) -> int:
    return len(json.dumps(body, ensure_ascii=False).encode("utf-8")) + INPUT_OVERHEAD_TOKENS


class DeepSeekChat:
    def __init__(self, *, api_key: str, budget: Budget, group_id: str, role: str, http: httpx.Client | None = None,
                 base_url: str = DEEPSEEK_BASE_URL, timeout_s: float = 60.0, tag: str = "") -> None:
        if not api_key:
            raise ValueError("no DeepSeek API key")
        self._key = api_key
        self.budget = budget
        self.group_id = group_id
        self.role = role
        self.http = http or httpx.Client()  # httpx: no automatic retries
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.timeout_s = timeout_s
        self.tag = tag
        self.calls: list[dict[str, Any]] = []  # per-request record (no content, no secrets)

    def _redact(self, text: str) -> str:
        return text.replace(self._key, "<redacted>")

    def complete(self, model: str, messages: list[dict[str, Any]], *, temperature: float, max_tokens: int,
                 extra_body: dict[str, Any] | None = None) -> ChatResult:
        body = body_for(model, messages, temperature=temperature, max_tokens=max_tokens, extra_body=extra_body)
        bound = input_bound_tokens(body)
        # Reserve first; BudgetExceeded / BudgetUnavailable propagate and nothing is sent.
        reservation = self.budget.reserve(group_id=self.group_id, role=self.role, input_bound_tokens=bound, max_output_tokens=max_tokens,
                                          detail={"model": model, "tag": self.tag})
        request_id = reservation["request_id"]
        record: dict[str, Any] = {"request_id": request_id, "model": model, "input_bound_tokens": bound, "max_tokens": max_tokens,
                                  "reserved_usd": reservation["reserved_usd"], "tag": self.tag}
        self.calls.append(record)
        started = time.monotonic()

        def fail(kind: str, detail: str) -> LlmCallFailed:
            record["charged_usd"] = self.budget.settle_unknown(request_id=request_id, note=f"{kind}: {detail[:200]}")
            record["outcome"] = kind
            return LlmCallFailed(kind, detail)

        try:
            response = self.http.post(self.url, json=body, headers={"Authorization": f"Bearer {self._key}"}, timeout=self.timeout_s)
        except httpx.TimeoutException as error:
            raise fail("timeout", self._redact(f"{type(error).__name__}: {error}")) from None
        except httpx.HTTPError as error:
            raise fail("transport", self._redact(f"{type(error).__name__}: {error}")) from None
        except BaseException:  # cancellation / interpreter exit mid-request: book it in full, then propagate
            record["charged_usd"] = self.budget.settle_unknown(request_id=request_id, note="interrupted in flight")
            raise
        record["latency_s"] = round(time.monotonic() - started, 3)
        record["http_status"] = response.status_code
        if response.status_code != 200:
            raise fail("http_status", f"HTTP {response.status_code}: {self._redact(response.text[:300])}")
        try:
            data = response.json()
        except ValueError:
            raise fail("unparseable_body", self._redact(response.text[:300])) from None
        usage = data.get("usage") if isinstance(data, dict) else None
        tokens_in = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        tokens_out = usage.get("completion_tokens") if isinstance(usage, dict) else None
        if isinstance(tokens_in, int) and isinstance(tokens_out, int):
            charged = self.budget.settle(request_id=request_id, input_tokens=tokens_in, output_tokens=tokens_out, usage=usage)
            record["usage_reported"] = True
        else:
            charged = self.budget.settle_unknown(request_id=request_id, note="usage missing: charged in full")
            record["usage_reported"] = False
        record["charged_usd"] = charged
        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            record["outcome"] = "missing_choices"
            raise LlmCallFailed("missing_choices", "response has no choices")
        choice = choices[0]
        message = choice.get("message") or {}
        details = (usage or {}).get("completion_tokens_details") or {} if isinstance(usage, dict) else {}
        result = ChatResult(
            content=message.get("content"), finish_reason=choice.get("finish_reason"), returned_model=data.get("model"), usage=usage,
            reasoning_present=bool(message.get("reasoning_content")), reasoning_tokens=details.get("reasoning_tokens"),
            request_id=request_id, reserved_usd=reservation["reserved_usd"], charged_usd=charged, latency_s=record["latency_s"], raw_choice=choice,
        )
        record |= {"outcome": "ok", "finish_reason": result.finish_reason, "returned_model": result.returned_model,
                   "reasoning_present": result.reasoning_present, "reasoning_tokens": result.reasoning_tokens}
        expected = body["model"]
        if result.returned_model != expected:
            record["outcome"] = "model_mismatch"
            detail = f"returned {result.returned_model!r}, requested {expected!r}"
            # Non-retryable: no silent model change. Halt the single manager (already settled above).
            self.budget.halt(reason=f"model_mismatch ({self.role}): {detail}")
            raise LlmCallFailed("model_mismatch", detail)
        return result
