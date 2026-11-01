"""Step 8 single budget manager, group envelopes, worker IPC and the DeepSeek client (offline).

No network: httpx MockTransport or nothing. These tests say nothing about the provider.
"""

from __future__ import annotations

import json
import multiprocessing
import threading
import time
from pathlib import Path

import httpx
import pytest

from agent_rl.llm.deepseek_chat import DeepSeekChat, LlmCallFailed, body_for, input_bound_tokens
from agent_rl.utils.budget_manager import BudgetClient, BudgetManager, BudgetServer, BudgetUnavailable
from agent_rl.utils.spend_ledger import BudgetExceeded, Price, ReservationTooSmall, SpendLedger

KEY = "ds-test-SECRET-789"
PRICES = {"simulator": Price(input_per_mtok=0.30, output_per_mtok=1.20, source="test"),
          "judge": Price(input_per_mtok=1.32, output_per_mtok=3.96, source="test"),
          "flat": Price(input_per_mtok=0.0, output_per_mtok=0.0, per_request_fee=0.2, source="test: 0.2 USD per request")}


def ledger(tmp_path, cap=1.0, budget_id="b", requests=1000, name="l.jsonl"):
    return SpendLedger(tmp_path / name, cap_usd=cap, max_billed_requests=requests, budget_id=budget_id)


def events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


# -- envelopes ---------------------------------------------------------------------------------


def test_envelope_reserve_settle_top_up_and_release(tmp_path):
    with ledger(tmp_path) as led:
        led.open_envelope("g1", 0.5, {})
        assert led.committed_usd == pytest.approx(0.5)
        led.reserve("r1", 0.3, {}, envelope="g1")
        assert led.committed_usd == pytest.approx(0.5)  # inside the envelope, no new commitment
        led.settle("r1", actual_usd=0.1, usage={})
        assert led.envelope_remaining("g1") == pytest.approx(0.4)  # the unused 0.2 returned to the envelope
        with pytest.raises(BudgetExceeded, match="left in envelope"):
            led.reserve("r2", 0.45, {}, envelope="g1")
        led.extend_envelope("g1", 0.05, {})
        led.reserve("r2", 0.45, {}, envelope="g1")
        with pytest.raises(BudgetExceeded, match="cap"):
            led.open_envelope("g2", 0.5, {})  # 0.55 committed + 0.5 > 1
        closed = led.close_envelope("g1", reason="done")  # r2 still open: charged in full
        assert closed["abandoned_requests"] == ["r2"]
        assert led.committed_usd == pytest.approx(0.55)
        led.open_envelope("g2", 0.45, {})
    assert SpendLedger.read_summary(tmp_path / "l.jsonl", "b")["open_envelopes_unused_usd"] == pytest.approx(0.45)


def test_restart_recovery_charges_open_requests_and_closes_envelopes(tmp_path):
    led = ledger(tmp_path)
    led.open_envelope("g1", 0.6, {})
    led.reserve("r1", 0.25, {}, envelope="g1")
    led.reserve("r0", 0.1, {})
    led.close()  # crash: nothing settled, envelope open
    with ledger(tmp_path) as again:
        assert again.committed_usd == pytest.approx(0.7)  # replay: open envelope (0.6) + open request outside it (0.1)
        manager = BudgetManager(again, PRICES)  # the manager recovers on start
        assert sorted(manager.recovery["charged_requests"]) == ["r0", "r1"] and manager.recovery["closed_envelopes"] == ["g1"]
        assert again.committed_usd == pytest.approx(0.35)  # r1 and r0 charged in full, rest of g1 released
        assert again.open_envelopes() == []


# -- manager + IPC -------------------------------------------------------------------------------


def _grab(address, authkey, results, n):
    client = BudgetClient(address, authkey)
    for _ in range(n):
        try:
            results.put(client.reserve(group_id="g", role="flat", input_bound_tokens=0, max_output_tokens=0, detail={})["request_id"])
        except BudgetExceeded:
            results.put("refused")


def test_concurrent_workers_cannot_overspend_through_the_single_manager(tmp_path):
    """8 processes x 3 requests of 0.2 USD race for a 1.0 USD cap: exactly 5 are reserved."""
    with ledger(tmp_path, cap=1.0) as led:
        manager = BudgetManager(led, PRICES)
        manager.open_group("g", 0.2, {})  # small envelope: every further request needs a top-up from the cap
        server = BudgetServer(manager, tmp_path / "ipc")
        try:
            context = multiprocessing.get_context("spawn")
            results = context.Queue()
            procs = [context.Process(target=_grab, args=(*server.client_config(), results, 3)) for _ in range(8)]
            for p in procs:
                p.start()
            got = [results.get(timeout=60) for _ in range(24)]
            for p in procs:
                p.join(30)
        finally:
            server.close()
        assert sum(g != "refused" for g in got) == 5 and led.committed_usd <= 1.0 + 1e-9
        assert len(manager.refused("g")) == 19
        assert led.billed_requests == 5


def test_child_gets_budget_unavailable_when_the_manager_is_gone(tmp_path):
    with ledger(tmp_path) as led:
        server = BudgetServer(BudgetManager(led, PRICES), tmp_path / "ipc")
        address, authkey = server.client_config()
        server.close()
    with pytest.raises(BudgetUnavailable):
        BudgetClient(address, authkey).reserve(group_id="g", role="flat", input_bound_tokens=0, max_output_tokens=0, detail={})


def test_wrong_key_and_forbidden_operations_are_refused(tmp_path):
    with ledger(tmp_path) as led:
        manager = BudgetManager(led, PRICES)
        manager.open_group("g", 0.5, {})
        server = BudgetServer(manager, tmp_path / "ipc")
        try:
            address, authkey = server.client_config()
            with pytest.raises(BudgetUnavailable):
                BudgetClient(address, b"wrong" * 6).reserve(group_id="g", role="flat", input_bound_tokens=0, max_output_tokens=0, detail={})
            with pytest.raises(BudgetUnavailable, match="not available to workers"):
                BudgetClient(address, authkey)._call("open_group", group_id="x", envelope_usd=0.1, detail={})
        finally:
            server.close()


# -- DeepSeek client ---------------------------------------------------------------------------------


def ok_payload(model="deepseek-flash", content="Sure.", finish="stop", usage=True, reasoning=None):
    message = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    data = {"id": "c1", "model": model, "choices": [{"index": 0, "finish_reason": finish, "message": message}]}
    if usage:
        data["usage"] = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
    return data


def make_chat(tmp_path, handler, cap=1.0):
    led = ledger(tmp_path, cap=cap)
    manager = BudgetManager(led, PRICES)
    manager.open_group("g", 0.05, {})
    sent = []

    def counted(request):
        sent.append(json.loads(request.content))
        return handler(request)

    chat = DeepSeekChat(api_key=KEY, budget=manager, group_id="g", role="simulator", http=httpx.Client(transport=httpx.MockTransport(counted)))
    return chat, manager, led, sent


MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


def call(chat):
    return chat.complete("deepseek/deepseek-flash", MESSAGES, temperature=0.0, max_tokens=1024, extra_body={"thinking": {"type": "disabled"}})


def test_one_reservation_per_request_and_usage_settlement(tmp_path):
    chat, manager, led, sent = make_chat(tmp_path, lambda r: httpx.Response(200, json=ok_payload()))
    result = call(chat)
    assert result.content == "Sure." and result.finish_reason == "stop" and len(sent) == 1
    reserves = [e for e in events(led.path) if e["event"] == "reserve"]
    settles = [e for e in events(led.path) if e["event"] == "settle"]
    assert len(reserves) == len(settles) == 1 and settles[0]["usage_reported"] is True
    assert settles[0]["amount_usd"] == pytest.approx(PRICES["simulator"].cost(100, 20))
    assert reserves[0]["detail"]["input_bound_tokens"] == input_bound_tokens(sent[0]) >= len(json.dumps(sent[0]).encode())


@pytest.mark.parametrize("case", ["timeout", "transport", "http_500", "http_429", "html", "no_usage", "interrupted"])
def test_failures_book_the_full_reservation_and_never_resend(tmp_path, case):
    def handler(request):
        if case == "timeout":
            raise httpx.ReadTimeout("slow", request=request)
        if case == "transport":
            raise httpx.ConnectError("refused", request=request)
        if case == "interrupted":
            raise KeyboardInterrupt
        if case.startswith("http_"):
            return httpx.Response(int(case[5:]), json={"error": f"bad key {KEY}"})
        if case == "html":
            return httpx.Response(200, text="<html>", headers={"content-type": "text/html"})
        return httpx.Response(200, json=ok_payload(usage=False))

    chat, manager, led, sent = make_chat(tmp_path, handler)
    if case == "no_usage":
        assert call(chat).content == "Sure."  # the reply is usable, its cost is unknown -> charged in full
    elif case == "interrupted":
        with pytest.raises(KeyboardInterrupt):
            call(chat)
    else:
        with pytest.raises(LlmCallFailed) as raised:
            call(chat)
        assert KEY not in str(raised.value)
    assert len(sent) == 1  # no automatic retry of any kind
    settle = [e for e in events(led.path) if e["event"] == "settle"]
    assert len(settle) == 1 and settle[0]["usage_reported"] is False and settle[0]["amount_usd"] == settle[0]["reserved_usd"]
    assert KEY not in led.path.read_text()


def test_model_mismatch_is_a_failure_after_settling(tmp_path):
    chat, manager, led, sent = make_chat(tmp_path, lambda r: httpx.Response(200, json=ok_payload(model="deepseek-v4-flash")))
    with pytest.raises(LlmCallFailed, match="model_mismatch"):
        call(chat)
    assert [e["usage_reported"] for e in events(led.path) if e["event"] == "settle"] == [True]


def test_budget_refusal_sends_nothing(tmp_path):
    chat, manager, led, sent = make_chat(tmp_path, lambda r: httpx.Response(200, json=ok_payload()), cap=0.05)
    with pytest.raises(BudgetExceeded):  # 0.05 envelope, a request reserves ~0.0012 + bytes: fine; then make it too big
        chat.complete("deepseek/deepseek-flash", [{"role": "user", "content": "x" * 200_000}], temperature=0.0, max_tokens=1024)
    assert sent == [] and manager.refused("g")


def test_actual_above_reservation_stops(tmp_path):
    big = ok_payload()
    big["usage"] = {"prompt_tokens": 10_000_000, "completion_tokens": 1}
    chat, manager, led, sent = make_chat(tmp_path, lambda r: httpx.Response(200, json=big), cap=100.0)
    with pytest.raises(ReservationTooSmall):
        call(chat)


def test_body_matches_what_litellm_would_send_and_litellm_does_not_retry_here(monkeypatch):
    """Pins the request equivalence with litellm 1.82.6 for D5/D7 (captured offline)."""
    import sys

    stub = sys.modules.get("litellm")
    if stub is not None and getattr(stub, "__file__", None) is None:  # another test installed the raising stub
        monkeypatch.delitem(sys.modules, "litellm")
    litellm = pytest.importorskip("litellm")
    assert getattr(litellm, "__file__", None), "need the real litellm here"
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    captured = []

    def handler(request):
        captured.append((str(request.url), json.loads(request.content)))
        return httpx.Response(500, json={"error": {"message": "x", "type": "server_error"}})

    with pytest.raises(Exception):
        litellm.completion(model="deepseek/deepseek-flash", messages=MESSAGES, temperature=0.0, max_tokens=1024,
                           extra_body={"thinking": {"type": "disabled"}}, api_key="k", timeout=5,
                           client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(handler))))
    assert len(captured) == 1  # one request on a 500 in this configuration
    url, body = captured[0]
    assert url == "https://api.deepseek.com/beta/chat/completions"
    assert body == body_for("deepseek/deepseek-flash", MESSAGES, temperature=0.0, max_tokens=1024, extra_body={"thinking": {"type": "disabled"}})
