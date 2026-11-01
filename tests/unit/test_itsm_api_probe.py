"""Step 7A spend ledger and API probe, offline (httpx MockTransport, fake DeepSeek call).

No network: these tests only check that reservations happen before sending,
that the budget and stop conditions hold and that secrets never reach a report.
They say nothing about the real provider.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import httpx
import pytest

from agent_rl.utils.spend_ledger import BudgetExceeded, Price, ReservationTooSmall, SpendLedger

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("probe_api_7a", REPO_ROOT / "scripts" / "itsm" / "probe_api_7a.py")
probe_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe_module)

KEY = "sk-test-SECRET-123"
DS_KEY = "ds-test-SECRET-456"
MODEL = "gpt-6.1-sol"
# Deliberately high test prices (not real): the plan must still fit 1 USD / 6 requests.
PRICES = {role: Price(input_per_mtok=20.0, output_per_mtok=100.0, per_request_fee=0.001, source="test") for role in ("teacher", "simulator", "judge")}


def responses_payload(model=MODEL, output_tokens=12, function_call=False):
    output = [{"type": "function_call", "name": "find_incident_by_number", "arguments": '{"number": "INC0000007"}'}] if function_call else [{"type": "message", "content": [{"type": "output_text", "text": "OK"}]}]
    return {"id": "resp_1", "model": model, "output": output, "usage": {"input_tokens": 40, "output_tokens": output_tokens, "output_tokens_details": {"reasoning_tokens": 4}}}


def make_transport(*, model_listed=True, returned_model=MODEL, output_tokens=12, echo_key=False):
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append((request.method, request.url.path))
        assert request.headers["Authorization"] == f"Bearer {KEY}"
        if request.url.path.endswith("/models"):
            ids = [MODEL, "gpt-4o"] if model_listed else ["gpt-4o"]
            return httpx.Response(200, json={"data": [{"id": i} for i in ids]})
        body = json.loads(request.content)
        if echo_key:
            return httpx.Response(400, json={"error": f"bad key {KEY}"})
        if request.url.path.endswith("/responses"):
            return httpx.Response(200, json=responses_payload(returned_model, output_tokens, function_call="tools" in body))
        return httpx.Response(200, json={"id": "c1", "model": returned_model, "choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [{"id": "t1"}]}}], "usage": {"prompt_tokens": 40, "completion_tokens": 9}})

    return httpx.MockTransport(handler), sent


def fake_deepseek(**kwargs):
    model = kwargs["model"].split("/", 1)[1]
    return {"id": "d1", "model": model, "choices": [{"finish_reason": "stop", "message": {"content": "OK"}}], "usage": {"prompt_tokens": 12, "completion_tokens": 2}}


def make_probe(tmp_path, *, billing=True, **transport_kwargs):
    transport, sent = make_transport(**transport_kwargs)
    ledger = SpendLedger(tmp_path / "ledger.jsonl", cap_usd=1.0, max_billed_requests=6, budget_id="step7a_api_probe")
    probe = probe_module.Probe(base_url="https://example.invalid/v1", api_key=KEY, prices=PRICES, teacher_model=MODEL, ledger=ledger,
                               billing=billing, http=httpx.Client(transport=transport), deepseek_call=fake_deepseek, secrets=[KEY, DS_KEY])
    return probe, sent, ledger


# -- ledger --------------------------------------------------------------------------------------


def test_ledger_reserves_before_and_refuses_over_cap_or_count(tmp_path):
    ledger = SpendLedger(tmp_path / "l.jsonl", cap_usd=1.0, max_billed_requests=2, budget_id="b")
    ledger.reserve("a", 0.6, {})
    with pytest.raises(BudgetExceeded, match="cap"):
        ledger.reserve("b", 0.5, {})
    ledger.reserve("b", 0.3, {})
    with pytest.raises(BudgetExceeded, match="request limit"):
        ledger.reserve("c", 0.01, {})


def test_ledger_charges_full_reservation_without_usage_and_replays_open_ones(tmp_path):
    path = tmp_path / "l.jsonl"
    ledger = SpendLedger(path, cap_usd=1.0, max_billed_requests=6, budget_id="b")
    ledger.reserve("a", 0.4, {})
    assert ledger.settle("a", actual_usd=None, usage=None) == 0.4
    ledger.reserve("b", 0.3, {})  # crash before settling
    reloaded = SpendLedger(path, cap_usd=1.0, max_billed_requests=6, budget_id="b")
    assert reloaded.committed_usd == pytest.approx(0.7) and reloaded.billed_requests == 2
    with pytest.raises(ValueError, match="belongs to budget"):
        SpendLedger(path, cap_usd=1.0, max_billed_requests=6, budget_id="other")


def test_ledger_stops_when_actual_exceeds_reservation(tmp_path):
    ledger = SpendLedger(tmp_path / "l.jsonl", cap_usd=1.0, max_billed_requests=6, budget_id="b")
    ledger.reserve("a", 0.01, {})
    with pytest.raises(ReservationTooSmall):
        ledger.settle("a", actual_usd=0.02, usage={})


def test_price_requires_a_source():
    with pytest.raises(ValueError, match="source"):
        Price(input_per_mtok=1.0, output_per_mtok=1.0)


# -- probe ----------------------------------------------------------------------------------------


def test_plan_fits_the_step7a_budget_even_at_high_test_prices():
    plan = probe_module.planned_reservations(PRICES, MODEL)
    assert len(plan) == 5 <= 6
    assert sum(p["reservation_usd"] for p in plan) <= 1.0
    for p in plan:
        assert p["input_bound_tokens"] > 64


def test_happy_path_runs_five_billed_probes_within_budget(tmp_path):
    probe, sent, ledger = make_probe(tmp_path)
    report = probe_module.run_probe(probe)
    assert report["stop_reason"] is None
    assert ledger.billed_requests == 5 and ledger.committed_usd <= 1.0
    checks = {r["probe"]: r.get("checks") for r in report["results"]}
    assert checks["B2_responses_tool"] == {"exactly_one_function_call": True, "tool_name_matches": True, "arguments_are_json_object": True}
    assert checks["B5_judge"] == {"finish_reason_stop": True, "no_reasoning_content": True}
    events = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    order = [e["event"] for e in events]
    assert order == ["reserve", "settle"] * 5  # every request reserved before it was sent and settled after


def test_unlisted_model_sends_no_billed_request(tmp_path):
    probe, sent, ledger = make_probe(tmp_path, model_listed=False)
    report = probe_module.run_probe(probe)
    assert "cannot be confirmed" in report["stop_reason"]
    assert ledger.billed_requests == 0 and sent == [("GET", "/v1/models")]


def test_model_mismatch_stops_before_further_requests(tmp_path):
    probe, sent, ledger = make_probe(tmp_path, returned_model="gpt-5-mini")
    report = probe_module.run_probe(probe)
    assert "returned model" in report["stop_reason"]
    assert ledger.billed_requests == 1 and len(sent) == 2


def test_output_above_cap_stops(tmp_path):
    probe, sent, ledger = make_probe(tmp_path, output_tokens=probe_module.TEACHER_MAX_OUTPUT + 1)
    report = probe_module.run_probe(probe)
    assert "not enforced" in report["stop_reason"] and ledger.billed_requests == 1


def test_dry_run_sends_only_the_free_listing(tmp_path):
    probe, sent, ledger = make_probe(tmp_path, billing=False)
    report = probe_module.run_probe(probe)
    assert "dry run" in report["stop_reason"] and ledger.billed_requests == 0
    assert sent == [("GET", "/v1/models")] and report["planned_total_usd"] <= 1.0


def test_secrets_never_reach_the_report(tmp_path):
    probe, sent, ledger = make_probe(tmp_path, echo_key=True)
    report = probe_module.run_probe(probe)
    text = json.dumps(report)
    assert KEY not in text and DS_KEY not in text and "<redacted>" in text
    assert KEY not in (tmp_path / "ledger.jsonl").read_text()
