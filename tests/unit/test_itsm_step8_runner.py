"""Step 8 runner end to end through real ItsmEnvWorker children (offline).

A fake DeepSeek HTTP server on 127.0.0.1 stands in for the provider (children reach it through
``ITSM_DEEPSEEK_BASE_URL``); a stub sampler stands in for vLLM. Checks the single ledger
(every HTTP request was reserved first, everything settled), group envelopes, budget stops,
truncated D5 replies, rollout-engine errors, cancellation and worker reclamation.
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_itsm_agent_loop import IM_END, TOKENIZER  # noqa: E402

from enterprise_worlds.evaluator import evaluator_nl, text_match_strategy  # noqa: E402

from agent_rl.rollout.itsm_diagnostic_runner import ConcurrencyRamp, DiagnosticRunner, GroupPlan  # noqa: E402
from agent_rl.rollout.itsm_step8 import load_plan, make_step8_loop  # noqa: E402
from agent_rl.utils.budget_manager import BudgetManager, BudgetServer  # noqa: E402
from agent_rl.utils.spend_ledger import Price, SpendLedger  # noqa: E402

B_TASK, A_TASK = "itsmbench_eops_99ba2325_887f5580", "itsmbench_msp_019"
DONE = '<tool_call>\n{"name": "done", "arguments": {}}\n</tool_call>'
SCRIPT = ["Hello, which incident should I look at?", DONE]


class FakeDeepSeek:
    """Minimal /chat/completions; behaviour switches set per test."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.simulator_length_once = False
        self.lock = threading.Lock()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with fake.lock:
                    fake.requests.append(body)
                    length = fake.simulator_length_once and body["model"] == "deepseek-flash"
                    if length:
                        fake.simulator_length_once = False
                payload = fake.answer(body, length)
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    @staticmethod
    def answer(body, length):
        system, user = body["messages"][0]["content"], body["messages"][-1]["content"]
        if body["model"] == "deepseek-flash":
            content, finish = ("cut mid-sent", "length") if length else ("Please look at INC-000008.", "stop")
        elif system == text_match_strategy.JUDGE_SYSTEM_PROMPT:
            pairs = json.loads(user.split("pairs:\n", 1)[1])
            content, finish = json.dumps({"results": [{"index": p["index"], "equivalent": True} for p in pairs]}), "stop"
        else:
            assert system == evaluator_nl.JUDGE_SYSTEM_PROMPT
            outcomes = json.loads(user.split("expectedOutcomes:\n", 1)[1])
            content, finish = json.dumps({"results": [{"expectedOutcome": o, "metExpectation": False} for o in outcomes]}), "stop"
        return {"id": "c", "model": body["model"], "choices": [{"index": 0, "finish_reason": finish, "message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": 200, "completion_tokens": 20}}


class StubServer:
    """Replays SCRIPT per episode (keyed by the loop's request id); optional failures/delays."""

    def __init__(self, *, fail_first=0, delay_s=0.0):
        self.episodes: dict[str, int] = {}
        self.fail_first, self.delay_s, self.calls = fail_first, delay_s, 0

    async def generate(self, *, request_id, prompt_ids, sampling_params):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise RuntimeError("CUDA out of memory (stub)")
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        index = self.episodes.get(request_id, 0)
        self.episodes[request_id] = index + 1
        ids = TOKENIZER.encode(SCRIPT[min(index, len(SCRIPT) - 1)], add_special_tokens=False) + [IM_END]
        return SimpleNamespace(token_ids=ids, log_probs=[-0.5] * len(ids), stop_reason="completed")


@pytest.fixture
def fake(monkeypatch):
    server = FakeDeepSeek()
    monkeypatch.setenv("ITSM_DEEPSEEK_BASE_URL", server.url)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    yield server
    server.server.shutdown()


def run_runner(tmp_path, groups, *, prices=None, cap=5.0, stub=None, ramp=None, cancel_after_s=None):
    plan = load_plan()
    prices = prices or {"simulator": Price(input_per_mtok=0.30, output_per_mtok=1.20, source="test"),
                        "judge": Price(input_per_mtok=1.32, output_per_mtok=3.96, source="test")}
    stub = stub or StubServer()
    with SpendLedger(tmp_path / "ledger.jsonl", cap_usd=cap, max_billed_requests=1000, budget_id="t") as ledger:
        manager = BudgetManager(ledger, prices)
        ipc = BudgetServer(manager, tmp_path / "ipc")
        address, key = ipc.client_config()
        runner = DiagnosticRunner(plans=groups, budget=manager, sampling_params=plan["sampling"], out_dir=tmp_path / "run",
                                  make_loop=lambda g: make_step8_loop(g, server=stub, tokenizer=TOKENIZER, plan=plan, budget_address=address, budget_authkey=key),
                                  ramp=ramp or ConcurrencyRamp(levels=(1, 2), promote_after=1))

        async def go():
            task = asyncio.ensure_future(runner.run())
            if cancel_after_s is not None:
                await asyncio.sleep(cancel_after_s)
                task.cancel()
            try:
                return await task
            except asyncio.CancelledError:
                return "cancelled"

        try:
            summary = asyncio.run(go())
        finally:
            ipc.close()
        events = [json.loads(l) for l in (tmp_path / "ledger.jsonl").read_text().splitlines()] if (tmp_path / "ledger.jsonl").exists() else []
        return summary, runner, events, ledger.summary()


def read(tmp_path, name):
    path = tmp_path / "run" / name
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


def assert_books_closed(events, fake):
    reserves = [e for e in events if e["event"] == "reserve"]
    settled = {e["request_id"] for e in events if e["event"] == "settle"}
    assert len(reserves) == len(fake.requests)  # every HTTP request was reserved first; none outside the ledger
    assert {e["request_id"] for e in reserves} == settled  # and every reservation was settled
    opened = {e["envelope"] for e in events if e["event"] == "envelope_open"}
    assert opened == {e["envelope"] for e in events if e["event"] == "envelope_close"}


def test_end_to_end_two_parts_ledger_matches_requests(tmp_path, fake):
    groups = [GroupPlan("A-g1", "A", "dev", A_TASK, 2, 0.5), GroupPlan("B-g1", "B", "grpo", B_TASK, 2, 0.2)]
    summary, runner, events, budget = run_runner(tmp_path, groups)
    assert summary["stop_reason"] is None and summary["groups"]["complete"] == 2, summary
    assert summary["leftover_child_processes"] == 0 and multiprocessing.active_children() == []
    trajectories = read(tmp_path, "trajectories.jsonl")
    assert len(trajectories) == 4
    for t in trajectories:
        aux = t["extra_fields"]["itsm_aux_eval"]
        if t["part"] == "A":  # judges ran in the worker; stored apart from the fuzzy reward
            assert aux["nl"]["status"] == "evaluated" and aux["nl"]["nl_met"] is False and aux["db_llm"]["status"] == "evaluated"
        else:
            assert aux is None
        assert t["extra_fields"]["reward_extra_info"]["itsm_termination_reason"] == "agent_done"
    assert_books_closed(events, fake)
    assert any(h["event"] == "promote" for h in summary["ramp"])


def test_no_new_group_when_its_envelope_cannot_be_reserved(tmp_path, fake):
    expensive = {"simulator": Price(input_per_mtok=0.0, output_per_mtok=0.0, per_request_fee=0.05, source="test"),
                 "judge": Price(input_per_mtok=0.0, output_per_mtok=0.0, per_request_fee=0.05, source="test")}
    groups = [GroupPlan("B-g1", "B", "grpo", B_TASK, 1, 0.95), GroupPlan("B-g2", "B", "grpo", B_TASK, 1, 0.95)]  # 0.10 spent + 0.95 > 1.0
    summary, runner, events, budget = run_runner(tmp_path, groups, prices=expensive, cap=1.0)
    assert summary["groups"]["complete"] == 1 and summary["not_started"] == ["B-g2"]
    assert "envelope not reservable" in summary["stop_reason"]
    assert_books_closed(events, fake)


def test_a_refused_request_is_never_sent_and_fails_the_group(tmp_path, fake):
    pricey = {"simulator": Price(input_per_mtok=0.0, output_per_mtok=0.0, per_request_fee=0.05, source="test"),
              "judge": Price(input_per_mtok=0.0, output_per_mtok=0.0, per_request_fee=0.05, source="test")}
    groups = [GroupPlan("B-g1", "B", "grpo", B_TASK, 1, 0.01)]
    summary, runner, events, budget = run_runner(tmp_path, groups, prices=pricey, cap=0.03)
    assert fake.requests == []  # the first simulator request needed a top-up beyond the cap: not sent
    assert summary["groups"]["discarded"] == 1 and summary["stop_reason"].startswith("budget")
    assert [a["status"] for a in read(tmp_path, "attempts.jsonl")] == ["budget"]
    assert multiprocessing.active_children() == []


def test_truncated_d5_reply_fails_the_attempt_and_its_cost_stays(tmp_path, fake):
    fake.simulator_length_once = True
    summary, runner, events, budget = run_runner(tmp_path, [GroupPlan("B-g1", "B", "grpo", B_TASK, 1, 0.2)])
    attempts = read(tmp_path, "attempts.jsonl")
    assert [a["status"] for a in attempts] == ["infra_failure", "valid"] and attempts[0]["stage"] == "user_sim"
    assert "finish_reason_length" in attempts[0]["error"]
    trajectory = read(tmp_path, "trajectories.jsonl")[0]
    assert "cut mid-sent" not in TOKENIZER.decode(trajectory["response_ids"])  # the cut reply was never used
    first = next(e for e in events if e["event"] == "settle")
    assert first["usage_reported"] is True and first["amount_usd"] > 0  # its cost is booked
    assert_books_closed(events, fake)


def test_engine_error_is_retried_and_workers_are_reclaimed(tmp_path, fake):
    summary, runner, events, budget = run_runner(tmp_path, [GroupPlan("B-g1", "B", "grpo", B_TASK, 1, 0.2)], stub=StubServer(fail_first=1),
                                                 ramp=ConcurrencyRamp(levels=(1, 2), promote_after=10))
    assert [a["status"] for a in read(tmp_path, "attempts.jsonl")] == ["engine_error", "valid"]
    assert "out of memory" in read(tmp_path, "attempts.jsonl")[0]["error"]
    assert summary["groups"]["complete"] == 1 and multiprocessing.active_children() == []
    assert_books_closed(events, fake)


def test_cancellation_reclaims_workers_and_closes_the_books(tmp_path, fake):
    groups = [GroupPlan("B-g1", "B", "grpo", B_TASK, 2, 0.2)]
    summary, runner, events, budget = run_runner(tmp_path, groups, stub=StubServer(delay_s=30.0), cancel_after_s=12.0)
    assert summary == "cancelled"
    assert multiprocessing.active_children() == []
    group = read(tmp_path, "groups.jsonl")[0]
    assert group["status"] == "cancelled"
    assert_books_closed(events, fake)
    assert budget["open_reservations_usd"] == 0 and budget["open_envelopes_unused_usd"] == 0


def test_a_refused_envelope_waits_for_running_groups_then_opens(tmp_path, fake):
    """Unused envelope money returns when a group closes: the next group is deferred, not dropped."""
    groups = [GroupPlan("B-g1", "B", "grpo", B_TASK, 1, 0.6), GroupPlan("B-g2", "B", "grpo", B_TASK, 1, 0.6)]
    summary, runner, events, budget = run_runner(tmp_path, groups, cap=1.0, ramp=ConcurrencyRamp(levels=(2,), promote_after=10))
    assert summary["stop_reason"] is None and summary["groups"]["complete"] == 2
    assert any(e["event"] == "open_deferred" and e["group_id"] == "B-g2" for e in read(tmp_path, "events.jsonl"))
    assert_books_closed(events, fake)


def test_summary_keeps_fuzzy_llm_db_and_nl_apart(tmp_path, fake):
    import importlib.util

    groups = [GroupPlan("A-g1", "A", "dev", A_TASK, 2, 0.5), GroupPlan("B-g1", "B", "grpo", B_TASK, 2, 0.2)]
    run_runner(tmp_path, groups)
    spec = importlib.util.spec_from_file_location("summarize_8", Path(__file__).resolve().parents[2] / "scripts/itsm/summarize_8.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summary = module.summarise(tmp_path / "run", TOKENIZER, budget_id="t")
    a = summary["A_e0_base_dev_diagnostic"]["tasks"][0]
    assert a["n"] == 2 and a["llm_db"]["evaluated"] == 2 and a["nl"]["evaluated"] == 2 and a["nl"]["met"] == 0
    assert a["joint_db_llm_and_nl"]["evaluated"] == 2
    b = summary["B_grpo_train_diagnostic_not_a_dev_score"]
    assert b["groups"] == 1 and b["all_zero"] + b["all_one"] + b["mixed_groups"] == 1
    assert summary["counterfactual_32k"]["label"].startswith("counterfactual") and summary["counterfactual_32k"]["episodes"] == 4
    assert summary["cost"]["actual_billing"].startswith("not in the ledger")
