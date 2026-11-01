"""Step 7A: minimal paid API compatibility probe (run in the cloud, never locally).

Usage (from the repo root, on the cloud instance):
    python scripts/itsm/probe_api_7a.py --prices PRICES.json --teacher-model ID \
        --out-dir DIR [--confirm-billing]

Environment (values are never printed or written): MICU_BASE_URL (OpenAI-style
base, e.g. ending in /v1), MICU_API_KEY, DEEPSEEK_API_KEY.

Budget (step 7A, separate from S1's 45 USD): at most 6 possibly billed requests
and 1 USD in total, retries included. Every billed request is reserved in the
append-only ledger before it is sent; no automatic retries. Without
``--confirm-billing`` only the free model listing runs and the reservations are
printed (dry run).

PRICES.json (USD per million tokens; ``source`` is required and must name the
verified price page or console screenshot):
    {"teacher":   {"input_per_mtok": .., "output_per_mtok": .., "per_request_fee": .., "source": ".."},
     "simulator": {...}, "judge": {...}}

Probes:
  F1 (free)  GET  {base}/models: does the account expose ``--teacher-model``?
  B1 teacher POST {base}/responses, plain text, max_output_tokens cap.
  B2 teacher POST {base}/responses, one Bench2 function tool, tool_choice required,
             parallel_tool_calls false: exactly one function_call with JSON args?
  B3 teacher POST {base}/chat/completions, same tool: the chat tool-call path.
  B4 simulator deepseek/deepseek-v4-flash via litellm (D5 path), temperature 0.
  B5 judge    deepseek/deepseek-v4-pro via litellm, D7 parameters (thinking disabled).
Each records the model identifier the provider returns, usage, finish reason and
the check results. Stops at the first: model mismatch / unconfirmable model,
output above the cap (cap not enforced), actual cost above its reservation,
or an exhausted budget.

Input-token bound: the UTF-8 byte length of the request body plus 64. Byte-level
BPE tokenizers (OpenAI o200k, DeepSeek) never produce more tokens than bytes,
so this bounds billed input tokens without knowing the tokenizer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from agent_rl.utils.spend_ledger import BudgetExceeded, Price, ReservationTooSmall, SpendLedger  # noqa: E402

BUDGET_ID = "step7a_api_probe"
CAP_USD = 1.0
MAX_BILLED_REQUESTS = 6
TEACHER_MAX_OUTPUT = 256
SIM_MAX_OUTPUT = 32
JUDGE_MAX_OUTPUT = 32
INPUT_OVERHEAD_TOKENS = 64
SIMULATOR_MODEL = "deepseek/deepseek-v4-flash"  # D5
JUDGE_MODEL = "deepseek/deepseek-v4-pro"  # D7
JUDGE_EXTRA_BODY = {"thinking": {"type": "disabled"}}  # D7
PROBE_TOOL = {
    "type": "function",
    "name": "find_incident_by_number",
    "description": "Find an incident by its number (e.g. INC0000007).",
    "parameters": {
        "type": "object",
        "properties": {"number": {"type": "string", "description": "Incident number"}},
        "required": ["number"],
        "additionalProperties": False,
    },
}
PROBE_INSTRUCTION = "You are an ITSM assistant. Look up incident INC0000007 with the available tool. Call exactly one tool."


class StopProbe(RuntimeError):
    pass


def b1_body(model: str) -> dict[str, Any]:
    return {"model": model, "input": "Reply with the single word OK.", "max_output_tokens": TEACHER_MAX_OUTPUT, "store": False}


def b2_body(model: str) -> dict[str, Any]:
    return {"model": model, "input": PROBE_INSTRUCTION, "tools": [PROBE_TOOL], "tool_choice": "required",
            "parallel_tool_calls": False, "max_output_tokens": TEACHER_MAX_OUTPUT, "store": False}


def b3_body(model: str) -> dict[str, Any]:
    tool = {"type": "function", "function": {k: PROBE_TOOL[k] for k in ("name", "description", "parameters")}}
    return {"model": model, "messages": [{"role": "user", "content": PROBE_INSTRUCTION}], "tools": [tool], "tool_choice": "required",
            "parallel_tool_calls": False, "max_completion_tokens": TEACHER_MAX_OUTPUT}


DEEPSEEK_MESSAGES = [{"role": "user", "content": "Reply with the single word OK."}]


def deepseek_body(model: str, max_output: int, **extra: Any) -> dict[str, Any]:
    return {"model": model, "messages": DEEPSEEK_MESSAGES, "max_tokens": max_output, "temperature": 0, **extra}


def planned_reservations(prices: dict[str, "Price"], teacher_model: str) -> list[dict[str, Any]]:
    """Worst-case reservation of every billed probe, computed before anything is sent."""
    plan = [
        ("B1_responses_text", "teacher", b1_body(teacher_model), TEACHER_MAX_OUTPUT),
        ("B2_responses_tool", "teacher", b2_body(teacher_model), TEACHER_MAX_OUTPUT),
        ("B3_chat_tool", "teacher", b3_body(teacher_model), TEACHER_MAX_OUTPUT),
        ("B5_judge", "judge", deepseek_body(JUDGE_MODEL, JUDGE_MAX_OUTPUT, extra_body=JUDGE_EXTRA_BODY), JUDGE_MAX_OUTPUT),
        ("B4_simulator", "simulator", deepseek_body(SIMULATOR_MODEL, SIM_MAX_OUTPUT), SIM_MAX_OUTPUT),
    ]
    return [{"probe": name, "role": role, "input_bound_tokens": input_bound(body), "max_output_tokens": out,
             "reservation_usd": round(prices[role].cost(input_bound(body), out), 6)} for name, role, body, out in plan]


def input_bound(body: dict[str, Any]) -> int:
    return len(json.dumps(body, ensure_ascii=False).encode("utf-8")) + INPUT_OVERHEAD_TOKENS


def _redact(text: str, secrets: list[str]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "<redacted>")
    return text


class Probe:
    def __init__(self, *, base_url: str, api_key: str, prices: dict[str, Price], teacher_model: str, ledger: SpendLedger,
                 billing: bool, http: httpx.Client, deepseek_call: Callable[..., Any], secrets: list[str]) -> None:
        self.base = base_url.rstrip("/")
        self.key = api_key
        self.prices = prices
        self.teacher_model = teacher_model
        self.ledger = ledger
        self.billing = billing
        self.http = http
        self.deepseek_call = deepseek_call
        self.secrets = secrets
        self.results: list[dict[str, Any]] = []

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}

    # -- free ------------------------------------------------------------------------------------

    def list_models(self) -> dict[str, Any]:
        response = self.http.get(f"{self.base}/models", headers=self._headers(), timeout=30)
        result: dict[str, Any] = {"probe": "F1_models", "billed": False, "http_status": response.status_code}
        if response.status_code == 200:
            ids = sorted(m.get("id") for m in response.json().get("data", []) if isinstance(m, dict))
            result |= {"model_ids_matching": [i for i in ids if "6.1" in i or "sol" in i.lower()], "teacher_model_listed": self.teacher_model in ids, "models_listed": len(ids)}
        else:
            result["error"] = _redact(response.text[:300], self.secrets)
        self.results.append(result)
        return result

    # -- billed ----------------------------------------------------------------------------------

    def _reserve(self, probe: str, role: str, body: dict[str, Any], max_output: int) -> tuple[str, float]:
        price = self.prices[role]
        amount = price.cost(input_bound(body), max_output)
        request_id = f"{probe}:{uuid.uuid4().hex[:8]}"
        if not self.billing:
            raise StopProbe("dry run: billed probes are not sent without --confirm-billing")
        self.ledger.reserve(request_id, amount, {"probe": probe, "role": role, "input_bound_tokens": input_bound(body), "max_output_tokens": max_output, "price_source": price.source})
        return request_id, amount

    def _settle(self, request_id: str, role: str, usage_in: int | None, usage_out: int | None, usage: Any) -> float:
        actual = None if usage_in is None or usage_out is None else self.prices[role].cost(usage_in, usage_out)
        return self.ledger.settle(request_id, actual_usd=actual, usage=usage)

    def _post(self, probe: str, path: str, body: dict[str, Any], max_output: int) -> dict[str, Any]:
        request_id, reserved = self._reserve(probe, "teacher", body, max_output)
        result: dict[str, Any] = {"probe": probe, "billed": True, "endpoint": path, "requested_model": body["model"], "reserved_usd": round(reserved, 6), "request_params": sorted(k for k in body if k not in ("input", "messages", "tools"))}
        started = time.monotonic()
        try:
            response = self.http.post(f"{self.base}{path}", headers=self._headers(), json=body, timeout=120)
        except httpx.HTTPError as error:
            result["error"] = _redact(f"{type(error).__name__}: {error}", self.secrets)
            result["charged_usd"] = self._settle(request_id, "teacher", None, None, None)
            self.results.append(result)
            raise StopProbe(f"{probe}: transport error") from error
        result["latency_s"] = round(time.monotonic() - started, 3)
        result["http_status"] = response.status_code
        payload = response.json() if response.headers.get("content-type", "").startswith("application/json") else None
        usage = (payload or {}).get("usage") if isinstance(payload, dict) else None
        if path == "/responses" and usage:
            usage_in, usage_out = usage.get("input_tokens"), usage.get("output_tokens")
            reasoning = (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
        elif usage:
            usage_in, usage_out = usage.get("prompt_tokens"), usage.get("completion_tokens")
            reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        else:
            usage_in = usage_out = reasoning = None
        result["usage"] = usage
        result["reasoning_tokens"] = reasoning
        if isinstance(payload, dict):
            result["returned_model"] = payload.get("model")
            result["response_id"] = payload.get("id")
        try:
            result["charged_usd"] = round(self._settle(request_id, "teacher", usage_in, usage_out, usage), 6)
        except ReservationTooSmall:
            # Record what came back before stopping: the model ID and usage are evidence.
            result["charged_usd"] = round(self.prices["teacher"].cost(usage_in, usage_out), 6)
            self.results.append(result)
            raise
        if response.status_code != 200 or not isinstance(payload, dict):
            result["error"] = _redact(response.text[:500], self.secrets)
            self.results.append(result)
            return result
        result["returned_model"] = payload.get("model")
        result["response_id"] = payload.get("id")
        result["system_fingerprint"] = payload.get("system_fingerprint")
        result["output_within_cap"] = usage_out is not None and usage_out <= max_output
        result["payload"] = payload
        self.results.append(result)
        if usage_out is not None and usage_out > max_output:
            raise StopProbe(f"{probe}: {usage_out} output tokens > cap {max_output}: the output cap is not enforced")
        if result["returned_model"] != body["model"]:
            raise StopProbe(f"{probe}: returned model {result['returned_model']!r} != requested {body['model']!r}")
        return result

    def teacher_text(self) -> None:
        result = self._post("B1_responses_text", "/responses", b1_body(self.teacher_model), TEACHER_MAX_OUTPUT)
        texts = [c.get("text") for item in (result.get("payload") or {}).get("output", []) if item.get("type") == "message" for c in item.get("content", [])]
        result["checks"] = {"has_text_output": any(texts)}

    def teacher_tool_responses(self) -> None:
        result = self._post("B2_responses_tool", "/responses", b2_body(self.teacher_model), TEACHER_MAX_OUTPUT)
        calls = [item for item in (result.get("payload") or {}).get("output", []) if item.get("type") == "function_call"]
        args_ok = False
        if len(calls) == 1:
            try:
                args_ok = isinstance(json.loads(calls[0].get("arguments", "")), dict)
            except ValueError:
                pass
        result["checks"] = {"exactly_one_function_call": len(calls) == 1, "tool_name_matches": bool(calls) and calls[0].get("name") == PROBE_TOOL["name"], "arguments_are_json_object": args_ok}

    def teacher_tool_chat(self) -> None:
        result = self._post("B3_chat_tool", "/chat/completions", b3_body(self.teacher_model), TEACHER_MAX_OUTPUT)
        choice = ((result.get("payload") or {}).get("choices") or [{}])[0]
        calls = (choice.get("message") or {}).get("tool_calls") or []
        result["checks"] = {"exactly_one_tool_call": len(calls) == 1, "finish_reason": choice.get("finish_reason")}

    def _deepseek(self, probe: str, role: str, model: str, max_output: int, **extra: Any) -> dict[str, Any]:
        messages = DEEPSEEK_MESSAGES
        body = deepseek_body(model, max_output, **extra)
        request_id, reserved = self._reserve(probe, role, body, max_output)
        result: dict[str, Any] = {"probe": probe, "billed": True, "requested_model": model, "reserved_usd": round(reserved, 6), "request_params": sorted(k for k in body if k != "messages")}
        try:
            response = self.deepseek_call(model=model, messages=messages, max_tokens=max_output, temperature=0, num_retries=0, **extra)
        except Exception as error:  # noqa: BLE001 - any failure is recorded and stops the probe
            result["error"] = _redact(f"{type(error).__name__}: {error}", self.secrets)[:500]
            result["charged_usd"] = self._settle(request_id, role, None, None, None)
            self.results.append(result)
            raise StopProbe(f"{probe}: call failed") from error
        data = response if isinstance(response, dict) else response.model_dump()
        usage = data.get("usage") or {}
        result["charged_usd"] = round(self._settle(request_id, role, usage.get("prompt_tokens"), usage.get("completion_tokens"), usage), 6)
        choice = data["choices"][0]
        message = choice.get("message") or {}
        result |= {
            "returned_model": data.get("model"),
            "response_id": data.get("id"),
            "system_fingerprint": data.get("system_fingerprint"),
            "usage": usage,
            "finish_reason": choice.get("finish_reason"),
            "reasoning_content_present": bool(message.get("reasoning_content")),
        }
        self.results.append(result)
        expected = model.split("/", 1)[1]
        if result["returned_model"] not in (model, expected):
            raise StopProbe(f"{probe}: returned model {result['returned_model']!r} != requested {expected!r}")
        return result

    def simulator(self) -> None:
        result = self._deepseek("B4_simulator", "simulator", SIMULATOR_MODEL, SIM_MAX_OUTPUT)
        result["checks"] = {"finish_reason_stop": result["finish_reason"] == "stop"}

    def judge(self) -> None:
        result = self._deepseek("B5_judge", "judge", JUDGE_MODEL, JUDGE_MAX_OUTPUT, extra_body=JUDGE_EXTRA_BODY)
        result["checks"] = {"finish_reason_stop": result["finish_reason"] == "stop", "no_reasoning_content": not result["reasoning_content_present"]}


def load_prices(path: Path) -> dict[str, Price]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    missing = {"teacher", "simulator", "judge"} - set(raw)
    if missing:
        raise SystemExit(f"prices file lacks {sorted(missing)}; verify prices before any paid probe")
    return {role: Price(**raw[role]) for role in ("teacher", "simulator", "judge")}


def run_probe(probe: Probe, *, skip_deepseek: bool = False) -> dict[str, Any]:
    plan = planned_reservations(probe.prices, probe.teacher_model)
    planned_total = sum(p["reservation_usd"] for p in plan)
    if len(plan) > MAX_BILLED_REQUESTS or planned_total > CAP_USD:
        raise SystemExit(f"planned reservations {planned_total:.6f} USD / {len(plan)} requests exceed the step-7A budget")
    stop_reason = None
    models = probe.list_models()
    if models.get("http_status") != 200 or not models.get("teacher_model_listed"):
        stop_reason = "teacher model not listed for this key (or listing failed): version cannot be confirmed"
    else:
        # Judge before simulator: the D5 simulator alias is reported retired upstream, and a
        # simulator model mismatch (a stop condition) must not prevent the judge check.
        steps = [probe.teacher_text, probe.teacher_tool_responses, probe.teacher_tool_chat]
        if skip_deepseek:
            probe.results += [{"probe": name, "billed": False, "not_run": "no DeepSeek credential"} for name in ("B5_judge", "B4_simulator")]
        else:
            steps += [probe.judge, probe.simulator]
        for step in steps:
            try:
                step()
            except StopProbe as stop:
                stop_reason = str(stop)
                break
            except (BudgetExceeded, ReservationTooSmall) as stop:
                stop_reason = f"{type(stop).__name__}: {stop}"
                break
    for result in probe.results:
        result.pop("payload", None)
    return {"budget": probe.ledger.summary(), "planned_reservations": plan, "planned_total_usd": round(planned_total, 6),
            "stop_reason": stop_reason, "results": probe.results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--teacher-model", required=True, help="model ID exactly as listed by the provider")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--confirm-billing", action="store_true")
    parser.add_argument("--skip-deepseek", action="store_true", help="run only the teacher probes (no DeepSeek credential)")
    args = parser.parse_args()
    base, key, ds_key = os.environ.get("MICU_BASE_URL"), os.environ.get("MICU_API_KEY"), os.environ.get("DEEPSEEK_API_KEY")
    if not base or not key or (not ds_key and not args.skip_deepseek):
        raise SystemExit("MICU_BASE_URL, MICU_API_KEY and DEEPSEEK_API_KEY (unless --skip-deepseek) must be set")
    ds_key = ds_key or ""
    import litellm

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ledger = SpendLedger(args.out_dir / "ledger.jsonl", cap_usd=CAP_USD, max_billed_requests=MAX_BILLED_REQUESTS, budget_id=BUDGET_ID)
    with httpx.Client() as http:
        probe = Probe(base_url=base, api_key=key, prices=load_prices(args.prices), teacher_model=args.teacher_model, ledger=ledger,
                      billing=args.confirm_billing, http=http, deepseek_call=litellm.completion, secrets=[key, ds_key])
        report = run_probe(probe, skip_deepseek=args.skip_deepseek)
    report["litellm_version"] = getattr(litellm, "version", None) or __import__("importlib.metadata").metadata.version("litellm")
    report["prices_file_sha256"] = __import__("hashlib").sha256(args.prices.read_bytes()).hexdigest()
    text = _redact(json.dumps(report, indent=2, ensure_ascii=False), [key, ds_key])
    (args.out_dir / "probe_report.json").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["stop_reason"] is None else 1


if __name__ == "__main__":
    sys.exit(main())
