"""Step 7A spend ledger and API probe, offline (httpx MockTransport, fake DeepSeek call).

No network: these tests only check that reservations happen before sending,
that the budget and stop conditions hold and that secrets never reach a report.
They say nothing about the real provider.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import httpx
import pytest

import multiprocessing

from agent_rl.utils.spend_ledger import BudgetExceeded, LedgerBusy, Price, ReservationTooSmall, SpendLedger

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


CHAT_TOOL_CALL = {"id": "t1", "type": "function", "function": {"name": "find_incident_by_number", "arguments": '{"number": "INC0000007"}'}}


def make_transport(*, model_listed=True, returned_model=MODEL, output_tokens=12, echo_key=False, override=None):
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
        if override is not None and (custom := override(request.url.path, body)) is not None:
            return custom
        if request.url.path.endswith("/responses"):
            return httpx.Response(200, json=responses_payload(returned_model, output_tokens, function_call="tools" in body))
        return httpx.Response(200, json={"id": "c1", "model": returned_model, "choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [CHAT_TOOL_CALL]}}], "usage": {"prompt_tokens": 40, "completion_tokens": 9}})

    return httpx.MockTransport(handler), sent


DEEPSEEK_CALLS = []


def fake_deepseek(**kwargs):
    DEEPSEEK_CALLS.append(kwargs)
    model = kwargs["model"].split("/", 1)[1]
    return {"id": "d1", "model": model, "choices": [{"finish_reason": "stop", "message": {"content": "OK"}}], "usage": {"prompt_tokens": 12, "completion_tokens": 2}}


def thinking_deepseek(**kwargs):
    """A provider that ignores thinking=disabled: reasoning comes back anyway."""
    data = fake_deepseek(**kwargs)
    data["choices"][0]["message"]["reasoning_content"] = "let me think"
    data["usage"]["completion_tokens_details"] = {"reasoning_tokens": 10}
    return data


def make_probe(tmp_path, *, billing=True, deepseek_call=fake_deepseek, **transport_kwargs):
    transport, sent = make_transport(**transport_kwargs)
    ledger = SpendLedger(tmp_path / "ledger.jsonl", cap_usd=1.0, max_billed_requests=6, budget_id="step7a_api_probe")
    probe = probe_module.Probe(base_url="https://example.invalid/v1", api_key=KEY, prices=PRICES, teacher_model=MODEL, ledger=ledger,
                               billing=billing, http=httpx.Client(transport=transport), deepseek_call=deepseek_call, secrets=[KEY, DS_KEY])
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
    ledger.close()
    reloaded = SpendLedger(path, cap_usd=1.0, max_billed_requests=6, budget_id="b")
    assert reloaded.committed_usd == pytest.approx(0.7) and reloaded.billed_requests == 2
    reloaded.close()
    with pytest.raises(ValueError, match="belongs to budget"):
        SpendLedger(path, cap_usd=1.0, max_billed_requests=6, budget_id="other")


def test_two_ledger_instances_on_one_path_cannot_both_reserve_075_under_cap_1(tmp_path):
    """7A-R1 regression: a second writer must not replay a stale snapshot and over-reserve."""
    path = tmp_path / "l.jsonl"
    first = SpendLedger(path, cap_usd=1.0, max_billed_requests=6, budget_id="b")
    with pytest.raises(LedgerBusy, match="already has a writer"):
        SpendLedger(path, cap_usd=1.0, max_billed_requests=6, budget_id="b")
    first.reserve("a", 0.75, {})
    first.close()
    with pytest.raises(LedgerBusy, match="closed"):
        first.reserve("late", 0.01, {})
    # the next owner re-reads the file and sees the open 0.75 reservation
    with SpendLedger(path, cap_usd=1.0, max_billed_requests=6, budget_id="b") as second:
        with pytest.raises(BudgetExceeded, match="cap"):
            second.reserve("b", 0.75, {})
        assert second.committed_usd == pytest.approx(0.75)
    total = SpendLedger.read_summary(path, "b")
    assert total["billed_requests"] == 1 and total["open_reservations_usd"] == 0.75


def _reserve_in_child(path, amount, results):
    try:
        with SpendLedger(path, cap_usd=1.0, max_billed_requests=6, budget_id="b") as ledger:
            ledger.reserve(f"child-{os.getpid()}", amount, {})
        results.put("reserved")
    except (LedgerBusy, BudgetExceeded) as error:
        results.put(type(error).__name__)


def test_a_second_process_is_refused_while_the_owner_holds_the_ledger(tmp_path):
    path = tmp_path / "l.jsonl"
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    with SpendLedger(path, cap_usd=1.0, max_billed_requests=6, budget_id="b") as owner:
        owner.reserve("a", 0.75, {})
        child = context.Process(target=_reserve_in_child, args=(path, 0.75, results))
        child.start(); child.join(60)
        assert results.get(timeout=5) == "LedgerBusy"
    child = context.Process(target=_reserve_in_child, args=(path, 0.75, results))
    child.start(); child.join(60)
    assert results.get(timeout=5) == "BudgetExceeded"  # re-read: 0.75 + 0.75 > 1
    assert SpendLedger.read_summary(path, "b")["billed_requests"] == 1


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
    DEEPSEEK_CALLS.clear()
    report = probe_module.run_probe(probe)
    assert report["stop_reason"] is None and report["all_required_checks_passed"] is True
    assert ledger.billed_requests == 5 and ledger.committed_usd <= 1.0
    checks = {r["probe"]: r.get("checks") for r in report["results"]}
    tool_checks = {"exactly_one_function_call": True, "tool_name_matches": True, "arguments_are_json_object": True, "arguments_match": True}
    assert checks["B2_responses_tool"] == {"output_present": True, **tool_checks}
    assert all(checks["B3_chat_tool"].values()) and checks["B3_chat_tool"]["arguments_match"] is True
    no_thinking = {"has_text_output": True, "finish_reason_stop": True, "no_reasoning_content": True, "no_reasoning_tokens": True}
    assert checks["B5_judge"] == no_thinking and checks["B4_simulator"] == no_thinking
    b2 = next(r for r in report["results"] if r["probe"] == "B2_responses_tool")
    assert b2["tool_calls"] == [{"name": "find_incident_by_number", "arguments": {"number": "INC0000007"}}]
    # D5 (7A-R1): the simulator probe sends thinking disabled, like the D7 judge
    sent_extra = {call["model"]: call.get("extra_body") for call in DEEPSEEK_CALLS}
    assert sent_extra == {"deepseek/deepseek-v4-pro": {"thinking": {"type": "disabled"}}, "deepseek/deepseek-flash": {"thinking": {"type": "disabled"}}}
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


def test_skip_deepseek_sends_only_teacher_probes(tmp_path):
    probe, sent, ledger = make_probe(tmp_path)
    report = probe_module.run_probe(probe, skip_deepseek=True)
    assert report["stop_reason"] is None and ledger.billed_requests == 3
    assert {r["probe"]: r.get("not_run") for r in report["results"] if r.get("not_run")} == {"B5_judge": "no DeepSeek credential", "B4_simulator": "no DeepSeek credential"}


def test_reservation_overrun_keeps_the_returned_model_and_usage(tmp_path):
    """A provider that bills hidden instructions (input far above the body bound)."""
    def handler(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": MODEL}]})
        payload = responses_payload()
        payload["usage"]["input_tokens"] = 50_000
        return httpx.Response(200, json=payload)

    ledger = SpendLedger(tmp_path / "ledger.jsonl", cap_usd=1.0, max_billed_requests=6, budget_id="step7a_api_probe")
    probe = probe_module.Probe(base_url="https://example.invalid/v1", api_key=KEY, prices=PRICES, teacher_model=MODEL, ledger=ledger,
                               billing=True, http=httpx.Client(transport=httpx.MockTransport(handler)), deepseek_call=fake_deepseek, secrets=[KEY])
    report = probe_module.run_probe(probe)
    assert report["stop_reason"].startswith("ReservationTooSmall") and ledger.billed_requests == 1
    b1 = next(r for r in report["results"] if r["probe"] == "B1_responses_text")
    assert b1["returned_model"] == MODEL and b1["usage"]["input_tokens"] == 50_000


# -- pass conditions (7A-R1): every failure stops the run and gives a non-zero exit ---------------


def _responses_with(output):
    def override(path, body):
        if path.endswith("/responses") and "tools" in body:
            payload = responses_payload(function_call=True)
            payload["output"] = output
            return httpx.Response(200, json=payload)
        return None
    return override


def _chat_with(message, finish_reason="tool_calls"):
    def override(path, body):
        if path.endswith("/chat/completions"):
            return httpx.Response(200, json={"id": "c1", "model": MODEL, "choices": [{"finish_reason": finish_reason, "message": message}], "usage": {"prompt_tokens": 40, "completion_tokens": 9}})
        return None
    return override


def _fc(name="find_incident_by_number", arguments='{"number": "INC0000007"}'):
    return {"type": "function_call", "name": name, "arguments": arguments}


@pytest.mark.parametrize(
    ("override", "stopped_at", "reason"),
    [
        (lambda path, body: httpx.Response(500, json={"error": "upstream"}) if path.endswith("/responses") else None, "B1_responses_text", "HTTP 500"),
        (lambda path, body: httpx.Response(429, text="slow down", headers={"content-type": "text/plain"}) if path.endswith("/chat/completions") else None, "B3_chat_tool", "HTTP 429"),
        (lambda path, body: httpx.Response(200, text="<html>", headers={"content-type": "text/html"}) if path.endswith("/responses") else None, "B1_responses_text", "not a JSON object"),
        (lambda path, body: httpx.Response(200, json={"id": "r", "model": MODEL, "output": []}) if path.endswith("/responses") else None, "B1_responses_text", "no usage"),
        (_responses_with([]), "B2_responses_tool", "output_present"),
        (_responses_with([_fc(name="list_incidents")]), "B2_responses_tool", "tool_name_matches"),
        (_responses_with([_fc(arguments='{"number": "INC0000008"}')]), "B2_responses_tool", "arguments_match"),
        (_responses_with([_fc(arguments='{"number": "INC0000007", "limit": 5}')]), "B2_responses_tool", "arguments_match"),
        (_responses_with([_fc(arguments="not json")]), "B2_responses_tool", "arguments_are_json_object"),
        (_responses_with([_fc(), _fc()]), "B2_responses_tool", "exactly_one_function_call"),
        (_chat_with({"content": "I looked it up."}, finish_reason="stop"), "B3_chat_tool", "exactly_one_tool_call"),
        (_chat_with({"tool_calls": [{"id": "t", "type": "function", "function": {"name": "get_incident", "arguments": '{"number": "INC0000007"}'}}]}), "B3_chat_tool", "tool_name_matches"),
        (_chat_with({"tool_calls": [{"id": "t", "type": "function", "function": {"name": "find_incident_by_number", "arguments": "{}"}}]}), "B3_chat_tool", "arguments_match"),
        (_chat_with({"tool_calls": [CHAT_TOOL_CALL]}, finish_reason="length"), "B3_chat_tool", "finish_reason_tool_calls"),
        (lambda path, body: httpx.Response(200, json={"id": "c", "model": MODEL, "choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}) if path.endswith("/chat/completions") else None, "B3_chat_tool", "output_present"),
    ],
)
def test_teacher_failures_stop_the_run(tmp_path, override, stopped_at, reason):
    probe, sent, ledger = make_probe(tmp_path, override=override)
    report = probe_module.run_probe(probe)
    assert report["stop_reason"] is not None and reason in report["stop_reason"] and stopped_at in report["stop_reason"]
    assert report["all_required_checks_passed"] is False
    probed = [r["probe"] for r in report["results"] if r.get("billed")]
    assert probed[-1] == stopped_at  # nothing was sent after the failing probe
    assert ledger.billed_requests == len(probed)


@pytest.mark.parametrize(
    ("deepseek_call", "stopped_at", "reason"),
    [
        (thinking_deepseek, "B5_judge", "no_reasoning_content"),
        (lambda **kw: {**fake_deepseek(**kw), "choices": []}, "B5_judge", "missing result"),
        (lambda **kw: {**fake_deepseek(**kw), "usage": None}, "B5_judge", "missing result"),
        (lambda **kw: {**fake_deepseek(**kw), "choices": [{"finish_reason": "length", "message": {"content": "O"}}]}, "B5_judge", "finish_reason_stop"),
        (lambda **kw: {**fake_deepseek(**kw), "choices": [{"finish_reason": "stop", "message": {"content": ""}}]}, "B5_judge", "has_text_output"),
        (lambda **kw: thinking_deepseek(**kw) if "flash" in kw["model"] else fake_deepseek(**kw), "B4_simulator", "no_reasoning"),
        (lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")), "B5_judge", "call failed"),
    ],
)
def test_deepseek_failures_stop_the_run(tmp_path, deepseek_call, stopped_at, reason):
    probe, sent, ledger = make_probe(tmp_path, deepseek_call=deepseek_call)
    report = probe_module.run_probe(probe)
    assert reason in report["stop_reason"] and stopped_at in report["stop_reason"]
    assert report["all_required_checks_passed"] is False


def _run_main(tmp_path, monkeypatch, *, override=None, deepseek_call=fake_deepseek):
    import sys
    import types

    transport, _ = make_transport(override=override)
    real_client = httpx.Client
    monkeypatch.setattr(probe_module.httpx, "Client", lambda *a, **k: real_client(transport=transport))
    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=deepseek_call, version="test"))
    monkeypatch.setenv("MICU_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("MICU_API_KEY", KEY)
    monkeypatch.setenv("DEEPSEEK_API_KEY", DS_KEY)
    prices = tmp_path / "prices.json"
    prices.write_text(json.dumps({role: {"input_per_mtok": 20.0, "output_per_mtok": 100.0, "per_request_fee": 0.001, "source": "test"} for role in ("teacher", "simulator", "judge")}))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["probe", "--prices", str(prices), "--teacher-model", MODEL, "--out-dir", str(out), "--confirm-billing"])
    code = probe_module.main()
    return code, json.loads((out / "probe_report.json").read_text())


def test_main_exits_zero_only_when_every_required_check_passes(tmp_path, monkeypatch, capsys):
    code, report = _run_main(tmp_path, monkeypatch)
    assert code == 0 and report["all_required_checks_passed"] is True
    assert KEY not in capsys.readouterr().out


def test_main_exits_non_zero_on_a_wrong_tool_argument(tmp_path, monkeypatch, capsys):
    code, report = _run_main(tmp_path, monkeypatch, override=_responses_with([_fc(arguments='{"number": "INC1"}')]))
    assert code == 1 and "arguments_match" in report["stop_reason"]


def test_main_exits_non_zero_on_http_error(tmp_path, monkeypatch, capsys):
    code, report = _run_main(tmp_path, monkeypatch, override=lambda path, body: httpx.Response(503, json={"error": "x"}) if path.endswith("/responses") else None)
    assert code == 1 and "HTTP 503" in report["stop_reason"]


def test_main_exits_non_zero_when_the_simulator_still_thinks(tmp_path, monkeypatch, capsys):
    code, report = _run_main(tmp_path, monkeypatch, deepseek_call=lambda **kw: thinking_deepseek(**kw) if "flash" in kw["model"] else fake_deepseek(**kw))
    assert code == 1 and "B4_simulator" in report["stop_reason"]


def test_main_refuses_to_reuse_an_exhausted_ledger(tmp_path, monkeypatch, capsys):
    """The step-7A ledger keeps its count: a rerun on the same out-dir cannot add requests."""
    out = tmp_path / "out"
    with SpendLedger(out / "ledger.jsonl", cap_usd=1.0, max_billed_requests=6, budget_id="step7a_api_probe") as used:
        for i in range(6):
            used.reserve(f"old{i}", 0.001, {})
            used.settle(f"old{i}", actual_usd=0.001, usage={})
    code, report = _run_main(tmp_path, monkeypatch)
    assert code == 1 and "request limit" in report["stop_reason"] and report["budget"]["billed_requests"] == 6

