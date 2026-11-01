"""Step 8 review fixes (R1), written to reproduce each problem before the fix (offline).

1. API review: a failed required check stops at once with the reason and a non-zero exit;
   inconsistent repeated replies are only a diagnostic.
2. Auxiliary evaluation failures (budget refusal, timeout, worker crash, evaluator error) keep
   the fuzzy result and trajectory and never re-sample the policy or the simulator.
3. Cancelling queued group members does not corrupt the queue count; the next group runs.
4. Model mismatch and actual-above-reservation halt the single manager: no new reservation,
   request or group afterwards; in-flight costs stay recorded; never retried.
"""

from __future__ import annotations

import asyncio
import functools
import importlib.util
import json
import multiprocessing
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from itsm_test_support import crashing_aux_factory, raising_aux_factory  # noqa: E402
from test_itsm_agent_loop import IM_END, TOKENIZER  # noqa: E402

from enterprise_worlds.evaluator import evaluator_nl, text_match_strategy  # noqa: E402

from agent_rl.data.itsm_task_sets import resolve_task  # noqa: E402
from agent_rl.llm.deepseek_chat import DeepSeekChat, LlmCallFailed  # noqa: E402
from agent_rl.rollout.itsm_diagnostic_runner import ConcurrencyRamp, DiagnosticRunner, GroupPlan  # noqa: E402
from agent_rl.rollout.itsm_step8 import load_plan, make_step8_loop  # noqa: E402
from agent_rl.utils.budget_manager import BudgetManager, BudgetServer  # noqa: E402
from agent_rl.utils.spend_ledger import BudgetExceeded, Price, ReservationTooSmall, SpendLedger  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
B_TASK, B_TASK2, A_TASK = "itsmbench_eops_99ba2325_887f5580", "itsmbench_msp_013", "itsmbench_msp_019"
DONE = '<tool_call>\n{"name": "done", "arguments": {}}\n</tool_call>'
SCRIPT = ["Hello, which incident should I look at?", DONE]
CHEAP = {"simulator": Price(input_per_mtok=0.30, output_per_mtok=1.20, source="test"), "judge": Price(input_per_mtok=1.32, output_per_mtok=3.96, source="test")}


def kind_of(body):
    if body["model"] == "deepseek-flash":
        return "sim"
    return "text" if body["messages"][0]["content"] == text_match_strategy.JUDGE_SYSTEM_PROMPT else "nl"


def default_answer(body):
    kind, user = kind_of(body), body["messages"][-1]["content"]
    if kind == "sim":
        return {"content": "Please look at INC-000008."}
    if kind == "text":
        pairs = json.loads(user.split("pairs:\n", 1)[1])
        return {"content": json.dumps({"results": [{"index": p["index"], "equivalent": True} for p in pairs]})}
    outcomes = json.loads(user.split("expectedOutcomes:\n", 1)[1])
    return {"content": json.dumps({"results": [{"expectedOutcome": o, "metExpectation": True} for o in outcomes]})}


def chat_payload(body, answer):
    message = {"role": "assistant", "content": answer["content"], **({"reasoning_content": answer["reasoning"]} if answer.get("reasoning") else {})}
    return {"id": "c", "model": answer.get("model", body["model"]), "choices": [{"index": 0, "finish_reason": answer.get("finish", "stop"), "message": message}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 20}}


class FakeDeepSeek:
    """Loopback /chat/completions; ``hook(body, n)`` may override the answer (n = request number)."""

    def __init__(self, hook=None) -> None:
        self.requests: list[dict] = []
        self.hook = hook or (lambda body, n: None)
        self.lock = threading.Lock()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with fake.lock:
                    fake.requests.append(body)
                    n = len(fake.requests)
                answer = {**default_answer(body), **(fake.hook(body, n) or {})}
                if answer.get("delay"):
                    time.sleep(answer["delay"])
                data = json.dumps(chat_payload(body, answer)).encode()
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"


class StubServer:
    def __init__(self):
        self.episodes: dict[str, int] = {}

    async def generate(self, *, request_id, prompt_ids, sampling_params):
        index = self.episodes.get(request_id, 0)
        self.episodes[request_id] = index + 1
        ids = TOKENIZER.encode(SCRIPT[min(index, len(SCRIPT) - 1)], add_special_tokens=False) + [IM_END]
        return SimpleNamespace(token_ids=ids, log_probs=[-0.5] * len(ids), stop_reason="completed")


@pytest.fixture
def fake_factory(monkeypatch):
    servers = []

    def make(hook=None):
        server = FakeDeepSeek(hook)
        servers.append(server)
        monkeypatch.setenv("ITSM_DEEPSEEK_BASE_URL", server.url)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        return server

    yield make
    for server in servers:
        server.server.shutdown()


def run_runner(tmp_path, groups, *, prices=CHEAP, cap=5.0, ramp=None, max_attempts=4, settings=None, aux_factory=None, timeout_s=240):
    plan = load_plan()
    stub = StubServer()
    with SpendLedger(tmp_path / "ledger.jsonl", cap_usd=cap, max_billed_requests=1000, budget_id="t") as ledger:
        manager = BudgetManager(ledger, prices)
        ipc = BudgetServer(manager, tmp_path / "ipc")
        address, key = ipc.client_config()

        def make_loop(g):
            loop = make_step8_loop(g, server=stub, tokenizer=TOKENIZER, plan=plan, budget_address=address, budget_authkey=key, settings_overrides=settings)
            if aux_factory is not None:  # a partial, so it stays a plain picklable callable (not a bound method)
                type(loop).aux_evaluator_factory = functools.partial(aux_factory)
            return loop

        runner = DiagnosticRunner(plans=groups, budget=manager, sampling_params=plan["sampling"], out_dir=tmp_path / "run", make_loop=make_loop,
                                  ramp=ramp or ConcurrencyRamp(levels=(1, 2), promote_after=1), max_attempts=max_attempts)
        try:
            summary = asyncio.run(asyncio.wait_for(runner.run(), timeout_s))
        finally:
            ipc.close()
        events = [json.loads(l) for l in (tmp_path / "ledger.jsonl").read_text().splitlines()]
        return summary, runner, events, manager


def read(tmp_path, name):
    path = tmp_path / "run" / name
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


def assert_books_closed(events, fake):
    reserves = [e for e in events if e["event"] == "reserve"]
    assert len(reserves) == len(fake.requests)
    assert {e["request_id"] for e in reserves} == {e["request_id"] for e in events if e["event"] == "settle"}
    assert {e["envelope"] for e in events if e["event"] == "envelope_open"} == {e["envelope"] for e in events if e["event"] == "envelope_close"}


# -- 1. API review -------------------------------------------------------------------------------


def load_review():
    spec = importlib.util.spec_from_file_location("review_8_api", REPO_ROOT / "scripts/itsm/review_8_api.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def review_http(case):
    sent = []

    def handler(request):
        body = json.loads(request.content)
        sent.append(body)
        kind = kind_of(body)
        answer = default_answer(body)
        if kind == "text":  # the fixed cases: pair 0 equivalent, pair 1 not
            pairs = json.loads(body["messages"][-1]["content"].split("pairs:\n", 1)[1])
            answer = {"content": json.dumps({"results": [{"index": p["index"], "equivalent": p["index"] == 0} for p in pairs]})}
        if kind == "nl":
            outcomes = json.loads(body["messages"][-1]["content"].split("expectedOutcomes:\n", 1)[1])
            answer = {"content": json.dumps({"results": [{"expectedOutcome": o, "metExpectation": i == 0} for i, o in enumerate(outcomes)]})}
        n_kind = sum(kind_of(b) == kind for b in sent)
        if case == "sim_model" and kind == "sim":
            answer["model"] = "deepseek-v4-flash"
        if case == "sim_thinking" and kind == "sim":
            answer["reasoning"] = "thinking..."
        if case == "text_protocol" and kind == "text":
            answer = {"content": "not json"}
        if case == "text_wrong" and kind == "text":
            answer = {"content": json.dumps({"results": [{"index": 0, "equivalent": False}, {"index": 1, "equivalent": False}]})}
        if case == "nl_wrong" and kind == "nl":
            outcomes = json.loads(body["messages"][-1]["content"].split("expectedOutcomes:\n", 1)[1])
            answer = {"content": json.dumps({"results": [{"expectedOutcome": o, "metExpectation": True} for o in outcomes]})}
        if case == "nl_model" and kind == "nl":
            answer["model"] = "deepseek-v4-flash"
        if case == "sim_inconsistent" and kind == "sim":
            answer = {"content": f"Reply variant {n_kind}"}
        return httpx.Response(200, json=chat_payload(body, answer))

    return httpx.Client(transport=httpx.MockTransport(handler)), sent


@pytest.mark.parametrize(("case", "requests_sent"), [("sim_model", 1), ("sim_thinking", 1), ("text_protocol", 10), ("text_wrong", 10),
                                                     ("nl_wrong", 12), ("nl_model", 12)])
def test_review_stops_at_the_first_failed_required_check(tmp_path, monkeypatch, case, requests_sent):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    review = load_review()
    http, sent = review_http(case)
    report = review.run(tmp_path / "out", billing=True, http=http)
    assert report["stop_reason"] is not None, report["results"][-1:]
    assert len(sent) == requests_sent  # stopped at once, nothing sent after the failure
    assert report["results"][-1]["ok"] is False and report["results"][-1].get("failed_check")
    assert review.exit_code(report, billing=True) != 0


def test_review_inconsistent_repeats_are_only_a_diagnostic(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    review = load_review()
    http, sent = review_http("sim_inconsistent")
    report = review.run(tmp_path / "out", billing=True, http=http)
    assert report["stop_reason"] is None and len(sent) == 13
    agreement = [r for r in report["results"] if r["check"] == "simulator_agreement"]
    assert agreement and all(r["distinct"] == 3 for r in agreement)
    assert review.exit_code(report, billing=True) == 0


def test_review_main_writes_the_report_and_exits_non_zero(tmp_path, monkeypatch):
    review = load_review()
    monkeypatch.setattr(review, "run", lambda out, billing, http=None: {"planned_requests": 13, "tasks": [], "results": [], "stop_reason": "x: failed"})
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(sys, "argv", ["review", "--out", str(tmp_path / "o"), "--confirm-billing"])
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setattr(review, "require_fixed_hash_seed", lambda where: None)
    assert review.main() == 1
    assert json.loads((tmp_path / "o" / "review_report.json").read_text())["stop_reason"] == "x: failed"


# -- 2. auxiliary failures never re-sample ------------------------------------------------------


@pytest.mark.parametrize("failure", ["budget", "timeout", "evaluator_error", "worker_crash"])
def test_aux_failure_keeps_the_fuzzy_result_and_never_resamples(tmp_path, fake_factory, failure):
    hook = (lambda body, n: {"delay": 20} if kind_of(body) != "sim" else None) if failure == "timeout" else None
    fake = fake_factory(hook)
    prices = CHEAP
    if failure == "budget":
        prices = {"simulator": CHEAP["simulator"], "judge": Price(input_per_mtok=0.0, output_per_mtok=0.0, per_request_fee=0.5, source="test")}
    aux_factory = {"evaluator_error": raising_aux_factory, "worker_crash": crashing_aux_factory}.get(failure)
    groups = [GroupPlan("A-g1", "A", "dev", A_TASK, 1, 0.1)]
    summary, runner, events, manager = run_runner(tmp_path, groups, prices=prices, cap=0.3, settings={"aux_timeout_s": 5.0}, aux_factory=aux_factory)
    attempts = read(tmp_path, "attempts.jsonl")
    assert [a["status"] for a in attempts] == ["valid"], attempts  # one attempt, kept: no re-sampling
    trajectory = read(tmp_path, "trajectories.jsonl")
    assert len(trajectory) == 1 and trajectory[0]["reward_score"] in (0.0, 1.0)
    aux = trajectory[0]["extra_fields"]["itsm_aux_eval"]
    assert aux["status"] == "not_evaluated" and aux["reason"] == {"budget": "budget", "timeout": "timeout", "evaluator_error": "evaluator_error",
                                                                 "worker_crash": "worker_crashed"}[failure]
    sims = [b for b in fake.requests if kind_of(b) == "sim"]
    assert len(sims) == 2  # reset + one agent message: the simulator ran for one episode only
    assert multiprocessing.active_children() == []
    assert read(tmp_path, "groups.jsonl")[0]["status"] == "complete"
    if failure == "budget":
        assert summary["stop_reason"] and summary["stop_reason"].startswith("budget")
    assert_books_closed(events, fake)


# -- 3. queue count after cancelling queued members -------------------------------------------


def test_cancelling_queued_members_keeps_the_count_and_the_next_group_runs(tmp_path, fake_factory):
    persona = resolve_task(B_TASK, split_version="itsm_split_v3").scenario.persona.identity
    marker = f"{persona.first_name} {persona.last_name}"
    fake = fake_factory(lambda body, n: {"finish": "length", "content": "cut"} if kind_of(body) == "sim" and marker in body["messages"][0]["content"] else None)
    groups = [GroupPlan("B-g1", "B", "grpo", B_TASK, 3, 0.2), GroupPlan("B-g2", "B", "grpo", B_TASK2, 1, 0.2)]
    summary, runner, events, manager = run_runner(tmp_path, groups, ramp=ConcurrencyRamp(levels=(1,), promote_after=10), max_attempts=2, timeout_s=180)
    results = {g["group_id"]: g for g in read(tmp_path, "groups.jsonl")}
    assert results["B-g1"]["status"] == "discarded" and "cancelled" in results["B-g1"]["sample_status"]
    assert results["B-g2"]["status"] == "complete"
    assert (runner._unqueued, runner._waiting, runner._running) == (0, 0, 0)
    assert multiprocessing.active_children() == []
    assert_books_closed(events, fake)


# -- 4. non-retryable faults halt the manager ---------------------------------------------------


def test_actual_above_reservation_halts_the_ledger_and_survives_restart(tmp_path):
    with SpendLedger(tmp_path / "l.jsonl", cap_usd=1.0, max_billed_requests=100, budget_id="b") as ledger:
        ledger.open_envelope("g", 0.5, {})
        ledger.reserve("r1", 0.01, {}, envelope="g")
        ledger.reserve("r2", 0.01, {}, envelope="g")  # in flight
        with pytest.raises(ReservationTooSmall):
            ledger.settle("r1", actual_usd=0.05, usage={})
        with pytest.raises(BudgetExceeded, match="halted"):
            ledger.reserve("r3", 0.001, {}, envelope="g")
        with pytest.raises(BudgetExceeded, match="halted"):
            ledger.open_envelope("g2", 0.1, {})
        ledger.settle("r2", actual_usd=0.002, usage={})  # in-flight cost still recorded
    with SpendLedger(tmp_path / "l.jsonl", cap_usd=1.0, max_billed_requests=100, budget_id="b") as again:
        with pytest.raises(BudgetExceeded, match="halted"):
            again.reserve("r4", 0.001, {})


def test_model_mismatch_halts_the_manager(tmp_path):
    with SpendLedger(tmp_path / "l.jsonl", cap_usd=1.0, max_billed_requests=100, budget_id="b") as ledger:
        manager = BudgetManager(ledger, CHEAP)
        manager.open_group("g", 0.5, {})
        handler = lambda r: httpx.Response(200, json=chat_payload(json.loads(r.content), {"content": "x", "model": "deepseek-v4-flash"}))
        chat = DeepSeekChat(api_key="k", budget=manager, group_id="g", role="simulator", http=httpx.Client(transport=httpx.MockTransport(handler)))
        with pytest.raises(LlmCallFailed, match="model_mismatch"):
            chat.complete("deepseek/deepseek-flash", [{"role": "user", "content": "u"}], temperature=0.0, max_tokens=1024)
        with pytest.raises(BudgetExceeded, match="halted"):
            chat.complete("deepseek/deepseek-flash", [{"role": "user", "content": "u"}], temperature=0.0, max_tokens=1024)
        assert manager.halt_reason and "model_mismatch" in manager.halt_reason


def test_model_mismatch_under_concurrency_stops_requests_and_groups(tmp_path, fake_factory):
    # the first request returns another model while the second sample's first request is in flight
    fake = fake_factory(lambda body, n: {"model": "deepseek-v4-flash"} if n == 1 else None)
    groups = [GroupPlan("B-g1", "B", "grpo", B_TASK, 2, 0.2), GroupPlan("B-g2", "B", "grpo", B_TASK2, 2, 0.2)]
    summary, runner, events, manager = run_runner(tmp_path, groups, ramp=ConcurrencyRamp(levels=(2,), promote_after=10))
    assert manager.halt_reason and "model_mismatch" in manager.halt_reason
    assert "model_mismatch" in (summary["stop_reason"] or "") and summary["not_started"] == ["B-g2"]
    halt_index = next(i for i, e in enumerate(events) if e["event"] == "halt")
    assert not [e for e in events[halt_index:] if e["event"] in ("reserve", "envelope_open", "envelope_extend")]  # nothing new after the halt
    attempts = read(tmp_path, "attempts.jsonl")
    assert all(a["attempt"] == 1 for a in attempts)  # never retried
    assert_books_closed(events, fake)  # in-flight costs settled and kept
    assert multiprocessing.active_children() == []


def test_review_report_keeps_every_required_record(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    review = load_review()
    http, sent = review_http("ok")
    report = review.run(tmp_path / "out", billing=True, http=http)
    assert report["stop_reason"] is None and report["http_requests"] == len(sent) == 13
    events = report["ledger_events"]
    assert sum(e["event"] == "reserve" for e in events) == sum(e["event"] == "settle" for e in events) == 13
    for call in report["requests"]:
        assert {"returned_model", "finish_reason", "usage", "reasoning_present", "reasoning_tokens", "reserved_usd", "charged_usd", "http_status"} <= set(call)
        assert call["usage"]["completion_tokens"] == 20
    checked = [r for r in report["results"] if r["check"] in ("simulator", "text_judge", "nl_judge")]
    assert len(checked) == 13 and all(r["ok"] and all(r["checks"].values()) for r in checked)
    assert any("1024 truncation behaviour not observed" in item for item in report["unverified"])
    assert "test-key" not in json.dumps(report)
