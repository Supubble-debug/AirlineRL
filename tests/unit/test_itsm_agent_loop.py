"""ItsmAgentLoop token/span/mask/log-prob contract (step 7A, offline).

A deterministic sampler stub stands in for verl's rollout server; the tokenizer
is the pinned Qwen3-8B tokenizer, the env is the real ItsmEnv on real Bench2
tools with a scripted user. The stub's log-probs are synthetic: these tests
check that the loop copies them position by position and never invents them;
they are NOT an acceptance of real sampled log-probs (step 7B).
"""

from __future__ import annotations

import asyncio
import functools
import json
import math
import multiprocessing
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("verl.experimental.agent_loop.agent_loop")
TOKENIZER_DIR = Path(os.environ.get("ITSM_TOKENIZER_DIR", Path.home() / ".cache/entergym/tokenizers/Qwen3-8B-b968826d"))
if not (TOKENIZER_DIR / "tokenizer_config.json").exists():
    pytest.skip("pinned Qwen3-8B tokenizer not available", allow_module_level=True)

from transformers import AutoTokenizer

from enterprise_worlds.evaluator.text_match_strategy import TextMatchConfig
from verl.utils.tokenizer.chat_template import initialize_system_prompt, initialize_turn_separator

from agent_rl.data import itsm_task_sets
from agent_rl.envs.itsm_env import DEFAULT_USER_LLM_ARGS, ItsmEnv, ItsmEnvConfig, to_itsm_action
from agent_rl.envs.itsm_scoring import ItsmInfrastructureError
from agent_rl.envs.itsm_worker import ItsmEnvWorker
from agent_rl.prompts.action_prompt import OBSERVATION_PROMPT, build_action_messages
from agent_rl.rollout.itsm_agent_loop import (
    ItsmAgentLoop,
    ItsmAgentLoopSettings,
    parse_action,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from itsm_test_support import ScriptedUser, failing_startup, install_slow_tool, slow_startup  # noqa: E402

TOKENIZER = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), trust_remote_code=False)
TEMPLATE_KWARGS = {"enable_thinking": False}
IM_END = TOKENIZER.convert_tokens_to_ids("<|im_end|>")
TASK = "itsmbench_eops_99ba2325_887f5580"  # GRPO-trainable, MSP, includes a policy refusal
GOLD = [a.arguments | {"__name": a.name} for a in itsm_task_sets.resolve_task(TASK, split_version="itsm_split_v3").evaluation_criteria.actions]
REPLIES = [
    "Hi, Karen Watkins here. Please retire SPR-AGT-01 and link it and PRT-ACME-01 to the MacBook Kernel Panic incident as affected CIs.",
    "Understood, link only the printer then. Go ahead.",
    "Thanks, that's everything.",
]


def tool_call_text(name: str, arguments: dict) -> str:
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}) + "\n</tool_call>"


GOLD_SCRIPT = [
    *(tool_call_text(g["__name"], {k: v for k, v in g.items() if k != "__name"}) for g in GOLD[:4]),
    "Plan: retire CI_041 and link CI_096 to INC_026. A retired CI may not be linked to an active incident, so I will not link CI_041. Shall I proceed?",
    *(tool_call_text(g["__name"], {k: v for k, v in g.items() if k != "__name"}) for g in GOLD[4:]),
    "Done: CI_041 retired, CI_096 linked to INC_026 as an affected CI. Anything else?",
    tool_call_text("done", {}),
]


class StubSampler:
    """Deterministic stand-in for verl's server manager: replays scripted texts."""

    def __init__(self, texts, *, logprobs="ok", stop=True):
        self.texts = list(texts)
        self.logprobs = logprobs
        self.stop = stop
        self.calls = []
        self.emitted = []

    async def generate(self, *, request_id, prompt_ids, sampling_params):
        index = len(self.calls)
        self.calls.append({"request_id": request_id, "prompt_ids": list(prompt_ids), "sampling_params": dict(sampling_params)})
        ids = TOKENIZER.encode(self.texts[index], add_special_tokens=False) + ([IM_END] if self.stop else [])
        if self.logprobs == "ok":
            lps = [-(index + 1) - 0.001 * k for k in range(len(ids))]
        elif self.logprobs == "missing":
            lps = None
        elif self.logprobs == "misaligned":
            lps = [-1.0] * (len(ids) - 1)
        elif self.logprobs == "empty_tokens":
            ids, lps = [], []
        else:  # a non-finite value at position 1
            lps = [-1.0] * len(ids)
            lps[1] = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}[self.logprobs]
        self.emitted.append((ids, lps))
        return SimpleNamespace(token_ids=ids, log_probs=lps, stop_reason="completed")


def make_loop(sampler, *, purpose="grpo", response_length=12288, replies=REPLIES, enable_thinking=False, worker=False, **settings):
    loop = object.__new__(ItsmAgentLoop)
    loop.settings = ItsmAgentLoopSettings(purpose=purpose, **settings)
    loop.tokenizer = TOKENIZER
    loop.processor = None
    loop.apply_chat_template_kwargs = {"enable_thinking": enable_thinking}
    loop.system_prompt = initialize_system_prompt(TOKENIZER, **TEMPLATE_KWARGS)
    loop.turn_separator = initialize_turn_separator(TOKENIZER, **TEMPLATE_KWARGS)
    loop.rollout_config = SimpleNamespace(prompt_length=28672, response_length=response_length)
    loop.server_manager = sampler
    loop.envs = []

    def make_env(task_id, episode_id):
        config = ItsmEnvConfig(task_id=task_id, text_match=TextMatchConfig(strategy="fuzzy"), max_steps=40, split_version="itsm_split_v3")
        env = ItsmEnv(config, user=ScriptedUser(list(replies)), episode_id=episode_id)
        loop.envs.append(env)
        return env

    if worker:
        # The real default path: ItsmEnvWorker child process with a picklable scripted user.
        loop.user_factory = functools.partial(ScriptedUser, list(replies))
        original = ItsmAgentLoop._make_env.__get__(loop)

        def make_worker(task_id, episode_id):
            created = original(task_id, episode_id)
            loop.envs.append(created)
            return created

        loop._make_env = make_worker
    else:
        loop._make_env = make_env
    loop._check_template()
    return loop


def run(loop, task_id=TASK):
    async def go():
        loop.loop = asyncio.get_running_loop()
        return await loop.run({"temperature": 1.0}, task_id=task_id, uid="g0", rollout_n=0)

    return asyncio.run(go())


@pytest.fixture(scope="module")
def happy():
    sampler = StubSampler(GOLD_SCRIPT)
    loop = make_loop(sampler)
    output = run(loop)
    return loop, sampler, output, loop.envs[0].export_trace()


# -- happy path: rendering, spans, mask, log-probs ------------------------------------------------


def test_episode_reaches_db_1_with_agent_done(happy):
    _, sampler, output, trace = happy
    info = output.extra_fields["reward_extra_info"]
    assert info["itsm_termination_reason"] == "agent_done"
    assert output.reward_score == 1.0 and info["itsm_db_reward"] == 1.0
    assert output.num_turns == len(GOLD_SCRIPT) == len(sampler.calls)


def test_prompt_is_the_qwen3_template_of_public_info_only(happy):
    loop, _, output, trace = happy
    env = ItsmEnv(ItsmEnvConfig(task_id=TASK, text_match=TextMatchConfig(strategy="fuzzy"), max_steps=40, split_version="itsm_split_v3"), user=ScriptedUser(list(REPLIES)))
    reset = env.reset()
    messages = build_action_messages(domain_policy=reset.public_info["policy"], observation=reset.observation)
    expected = TOKENIZER.apply_chat_template(messages, tools=reset.public_info["tools"], add_generation_prompt=True, tokenize=True, enable_thinking=False)
    expected = list(expected["input_ids"]) if hasattr(expected, "keys") else list(expected)
    assert output.prompt_ids == expected
    text = TOKENIZER.decode(output.prompt_ids)
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    task = itsm_task_sets.resolve_task(TASK, split_version="itsm_split_v3")
    full = TOKENIZER.decode(output.prompt_ids + output.response_ids)
    for private in (task.scenario.task_description, task.scenario.simulator_guidance, TASK, "EnterpriseOps", *task.evaluation_criteria.nl_assertions):
        assert private not in full


def test_policy_spans_carry_sampler_tokens_mask_1_and_sampler_logprobs(happy):
    _, sampler, output, _ = happy
    spans = output.extra_fields["itsm_turn_spans"]
    assert len(spans) == len(sampler.emitted)
    for span, (ids, lps) in zip(spans, sampler.emitted):
        s, e = span["start"], span["end"]
        assert output.response_ids[s:e] == ids
        assert output.response_mask[s:e] == [1] * len(ids)
        assert output.response_logprobs[s:e] == lps
    assert sum(output.response_mask) == sum(len(ids) for ids, _ in sampler.emitted)
    assert output.extra_fields["itsm_logprob_sources"] == {"mask_1": "sampler", "mask_0": "placeholder_masked"}


def test_environment_spans_are_mask_0_placeholders_and_verbatim_observations(happy):
    loop, _, output, trace = happy
    env_spans = output.extra_fields["itsm_env_spans"]
    assert len(env_spans) == len(GOLD_SCRIPT) - 1  # no env turn after done
    covered = set()
    for span in env_spans:
        s, e = span["start"], span["end"]
        assert output.response_mask[s:e] == [0] * (e - s)
        assert output.response_logprobs[s:e] == [0.0] * (e - s)
        assert output.response_ids[s : s + len(loop.turn_separator)] == loop.turn_separator
        covered.update(range(s, e))
    policy = {i for sp in output.extra_fields["itsm_turn_spans"] for i in range(sp["start"], sp["end"])}
    assert covered | policy == set(range(len(output.response_ids))) and not covered & policy
    # Observations enter verbatim (no compression or cut), wrapped in OBSERVATION_PROMPT.
    decoded = TOKENIZER.decode(output.response_ids)
    for message in trace["messages"]:
        if message["role"] == "tool":
            assert message["content"] in decoded


def test_each_generate_call_sees_prompt_plus_response_so_far(happy):
    _, sampler, output, _ = happy
    spans = output.extra_fields["itsm_turn_spans"]
    for call, span in zip(sampler.calls, spans):
        assert call["prompt_ids"] == output.prompt_ids + output.response_ids[: span["start"]]
        assert call["sampling_params"]["max_tokens"] == 512
        assert IM_END in call["sampling_params"]["stop_token_ids"]
    assert len({c["request_id"] for c in sampler.calls}) == 1  # one sticky request, no rotation


def test_environment_turn_length_matches_the_step6_budget_measurement(happy):
    """Env turns are counted exactly as step 6 measured them (budget v1 basis)."""
    loop, _, output, _ = happy
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "data"))
    import measure_itsm_tokens as measure

    renderer = measure.Renderer(TOKENIZER, [])
    for span in output.extra_fields["itsm_env_spans"]:
        text = TOKENIZER.decode(output.response_ids[span["start"] + len(loop.turn_separator) : span["end"]])
        inner = text.split("<observation>\n", 1)[1].split("\n</observation>", 1)[0]
        assert span["end"] - span["start"] == renderer.environment_turn(OBSERVATION_PROMPT.format(observation=inner))


# -- one action per turn --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("Let me look that up. " + tool_call_text("get_user_using_name", {"first_name": "Karen", "last_name": "Watkins"}), "both text and tool calls"),
        (tool_call_text("get_user_using_name", {"first_name": "Karen", "last_name": "Watkins"}) * 2, "at most one tool call"),
        ("Checking now. <tool_call>\n{\"name\": \"get_user_using_name\", \"arguments\": {broken}\n</tool_call>", "malformed tool call"),
        ("<tool_call>\n{\"name\": \"get_user_using_name\"", "unterminated"),
        ("   ", "text or one tool call"),
    ],
)
def test_invalid_output_ends_with_invalid_action_and_outcome_0(text: str, reason: str):
    sampler = StubSampler([text])
    output = run(make_loop(sampler))
    info = output.extra_fields["reward_extra_info"]
    assert info["itsm_termination_reason"] == "invalid_action"
    assert output.reward_score == 0.0 and info["itsm_training_reward_source"] == "invalid_action"
    assert reason in output.extra_fields["itsm_turn_spans"][0]["invalid_action"]
    assert output.response_mask == [1] * len(output.response_ids)  # the attempt is kept, no env turn after it


def test_output_cut_at_the_action_limit_is_invalid():
    sampler = StubSampler([tool_call_text("get_user_using_name", {"first_name": "Karen", "last_name": "Watkins"})], stop=False)
    output = run(make_loop(sampler))
    assert output.extra_fields["itsm_turn_spans"][0]["invalid_action"].startswith("action_token_limit")


def test_parse_action_message_and_done():
    assert parse_action("Hello there<|im_end|>", "Hello there", stopped=True).action.kind == "message"
    assert parse_action(tool_call_text("done", {}) + "<|im_end|>", tool_call_text("done", {}), stopped=True).action.kind == "done"
    assert parse_action(tool_call_text("done", {"x": 1}), tool_call_text("done", {"x": 1}), stopped=True).error


# -- log-probs are never invented ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("missing", "no log-probs"),
        ("misaligned", "one log-prob per generated token"),
        ("empty_tokens", "no tokens"),
        ("nan", "non-finite log-prob"),
        ("inf", "non-finite log-prob"),
        ("-inf", "non-finite log-prob"),
    ],
)
def test_sampler_contract_violations_are_infrastructure_failures(mode: str, message: str):
    sampler = StubSampler(GOLD_SCRIPT, logprobs=mode)
    loop = make_loop(sampler)
    with pytest.raises(ItsmInfrastructureError, match=message) as raised:
        run(loop)
    assert raised.value.stage == "sampler_contract"
    assert loop.envs[0].export_trace()["reward_info"] is None  # no reward for a broken sample


# -- budget: no rotation --------------------------------------------------------------------------


def test_response_budget_ends_with_response_length_and_no_rotation():
    sampler = StubSampler(GOLD_SCRIPT)
    output = run(make_loop(sampler, response_length=1200))
    info = output.extra_fields["reward_extra_info"]
    assert info["itsm_termination_reason"] == "response_length"
    assert len(output.response_ids) <= 1200
    assert len({c["request_id"] for c in sampler.calls}) == 1
    assert all(c["prompt_ids"][: len(output.prompt_ids)] == output.prompt_ids for c in sampler.calls)


# -- purpose gating -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("purpose", "task_id"),
    [
        ("grpo", "itsmbench_eops_46ca2862_b6ef60ca"),  # SFT-only
        ("grpo", "itsmbench_single_005"),  # R1 pending
        ("grpo", "itsmbench_eops_78a5b71e_a4fb07a9"),  # SFT-only, never GRPO
        ("grpo", "itsmbench_msp_019"),  # dev
        ("grpo", "itsmbench_eops_99ba2325_54f35603"),  # held-out
        ("dev", TASK),  # train task in a dev loop
    ],
)
def test_purpose_gate_refuses_tasks_outside_the_purpose(purpose: str, task_id: str):
    with pytest.raises(PermissionError, match="not allowed"):
        run(make_loop(StubSampler([]), purpose=purpose), task_id=task_id)


def test_thinking_must_be_disabled():
    with pytest.raises(ValueError, match="enable_thinking"):
        make_loop(StubSampler([]), enable_thinking=True)


def test_simulator_defaults_disable_thinking():
    """D5 (7A-R1): deepseek-flash with thinking disabled through extra_body (offline check only)."""
    assert ItsmAgentLoopSettings(purpose="grpo").user_llm == "deepseek/deepseek-flash"
    assert ItsmAgentLoopSettings(purpose="grpo").user_llm_args == DEFAULT_USER_LLM_ARGS
    assert DEFAULT_USER_LLM_ARGS["extra_body"] == {"thinking": {"type": "disabled"}}


def test_default_simulator_sends_thinking_disabled_to_litellm(monkeypatch):
    """D5 end to end, offline: the config the loop hands its worker reaches litellm.completion
    with deepseek-flash, temperature 0 and extra_body thinking disabled (fake litellm, no network)."""
    import types

    import agent_rl.rollout.itsm_agent_loop as loop_module

    captured = {}
    monkeypatch.setattr(loop_module, "ItsmEnvWorker", lambda config, **kwargs: captured.setdefault("config", config))
    loop = object.__new__(ItsmAgentLoop)
    loop.settings = ItsmAgentLoopSettings(purpose="grpo")
    loop._make_env(TASK, "e1")

    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        message = types.SimpleNamespace(content="Hi, I need help with an incident.", tool_calls=None)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=completion))
    env = ItsmEnv(captured["config"])  # default user = Bench2 UserSimulator
    env.reset()
    env.step(to_itsm_action(content="Hello, how can I help?", tool_calls=[]))
    assert calls, "the simulator was not called"
    for call in calls:
        assert call["model"] == "deepseek/deepseek-flash"
        assert call["temperature"] == 0.0
        assert call["extra_body"] == {"thinking": {"type": "disabled"}}
        assert call["max_tokens"] == 1024  # D5 (step 8)
        assert call["timeout"] == 60.0


# -- process worker: cancellation and timeout (real child processes) ------------------------------

SLOW_TOOL_S = 5.0
SLOW_TOOL = "update_configuration_item"  # the first write of the gold script (turn index 5)


def _run_until_slow_tool(loop, sampler, *, cancel_after_s):
    async def go():
        loop.loop = asyncio.get_running_loop()
        task = asyncio.create_task(loop.run({"temperature": 1.0}, task_id=TASK, uid="g0", rollout_n=0))
        while len(sampler.calls) < 6 and not task.done():
            await asyncio.sleep(0.05)
        started = time.monotonic()
        if cancel_after_s is not None:
            await asyncio.sleep(cancel_after_s)
            task.cancel()
        try:
            return await task, started
        except BaseException as error:  # noqa: BLE001 - returned for the assertions
            return error, started

    return asyncio.run(go())


def _assert_child_stopped(worker: ItsmEnvWorker, marker, started: float, stage: str) -> None:
    assert not worker.is_alive
    assert worker.failure["stage"] == stage and worker.failure["execution_stopped"] is True
    time.sleep(max(0.0, started + SLOW_TOOL_S + 1.0 - time.monotonic()))
    assert not marker.exists()  # the slow write never completed: stopped, not abandoned
    assert multiprocessing.active_children() == []


def test_cancel_during_a_slow_tool_kills_the_worker_child(tmp_path):
    marker = tmp_path / "slow-write-finished"
    sampler = StubSampler(GOLD_SCRIPT)
    loop = make_loop(sampler, worker=True)
    loop.worker_initializer = functools.partial(install_slow_tool, SLOW_TOOL, SLOW_TOOL_S, str(marker))

    outcome, started = _run_until_slow_tool(loop, sampler, cancel_after_s=0.5)

    assert isinstance(outcome, asyncio.CancelledError)  # no AgentLoopOutput, so no reward
    assert time.monotonic() - started < SLOW_TOOL_S
    _assert_child_stopped(loop.envs[0], marker, started, "cancelled")
    with pytest.raises(RuntimeError):
        loop.envs[0].score()


def test_call_timeout_during_a_slow_tool_is_infrastructure(tmp_path):
    marker = tmp_path / "slow-write-finished"
    sampler = StubSampler(GOLD_SCRIPT)
    loop = make_loop(sampler, worker=True, call_timeout_s=2.0)
    loop.worker_initializer = functools.partial(install_slow_tool, SLOW_TOOL, SLOW_TOOL_S, str(marker))

    outcome, started = _run_until_slow_tool(loop, sampler, cancel_after_s=None)

    assert isinstance(outcome, ItsmInfrastructureError) and outcome.stage == "timeout"
    _assert_child_stopped(loop.envs[0], marker, started, "timeout")


def test_worker_path_completes_an_episode():
    sampler = StubSampler(GOLD_SCRIPT)
    loop = make_loop(sampler, worker=True)
    output = run(loop)
    assert output.extra_fields["reward_extra_info"]["itsm_termination_reason"] == "agent_done"
    assert output.reward_score == 1.0
    assert not loop.envs[0].is_alive and multiprocessing.active_children() == []


# -- process worker: cancellation while the worker is starting (7A-R2) ----------------------------

SLOW_STARTUP_S = 5.0


def _cancel_when(loop, ready, *, second_cancel_after_s=None):
    """Run one rollout; cancel it as soon as ``ready(worker)`` holds (polled on the event loop)."""

    async def go():
        loop.loop = asyncio.get_running_loop()
        task = asyncio.create_task(loop.run({"temperature": 1.0}, task_id=TASK, uid="g0", rollout_n=0))
        deadline = time.monotonic() + 60
        while not (loop.envs and ready(loop.envs[0])):
            assert time.monotonic() < deadline and not task.done(), "the worker never reached the cancel point"
            await asyncio.sleep(0.01)
        cancelled_at = time.monotonic()
        task.cancel()
        if second_cancel_after_s is not None:
            await asyncio.sleep(second_cancel_after_s)
            task.cancel()  # a second cancellation must not interrupt the cleanup
        try:
            return await task, cancelled_at
        except BaseException as error:  # noqa: BLE001 - returned for the assertions
            return error, cancelled_at

    return asyncio.run(go())


def _assert_reclaimed(worker: ItsmEnvWorker) -> None:
    assert not worker.is_alive
    assert worker.failure["stage"] == "cancelled" and worker.failure["execution_stopped"] is True
    assert multiprocessing.active_children() == []
    worker.close()
    worker.close()  # repeated close is safe
    assert not worker.is_alive


@pytest.mark.parametrize("second_cancel_after_s", [None, 0.05])
def test_cancel_while_the_worker_child_is_starting(tmp_path, second_cancel_after_s):
    """The child exists, but the worker start has not returned yet (slow initializer)."""
    marker = tmp_path / "startup-finished"
    sampler = StubSampler(GOLD_SCRIPT)
    loop = make_loop(sampler, worker=True)
    loop.worker_initializer = functools.partial(slow_startup, SLOW_STARTUP_S, str(marker))

    # child spawned, start() not returned (no pid reported back yet)
    outcome, cancelled_at = _cancel_when(loop, lambda w: w.is_alive and w.pid is None, second_cancel_after_s=second_cancel_after_s)

    assert isinstance(outcome, asyncio.CancelledError)  # no AgentLoopOutput, hence no reward
    assert time.monotonic() - cancelled_at < SLOW_STARTUP_S  # killed, not waited out
    assert sampler.calls == []  # nothing was generated
    worker = loop.envs[0]
    assert worker.pid is None  # start never returned a ready worker
    time.sleep(max(0.0, cancelled_at + SLOW_STARTUP_S + 1.0 - time.monotonic()))
    assert not marker.exists()  # the starting child was killed before it finished
    _assert_reclaimed(worker)


def test_cancel_when_the_background_start_returns_a_ready_worker(monkeypatch):
    """The start thread still hands back a ready worker after the rollout was cancelled:
    the loop must reclaim it, not lose it."""
    returned = threading.Event()
    real_start = ItsmEnvWorker.start

    def start_then_linger(self):
        real_start(self)  # child is ready here
        returned.set()
        time.sleep(1.0)  # the "late" return: the rollout is cancelled meanwhile

    monkeypatch.setattr(ItsmEnvWorker, "start", start_then_linger)
    sampler = StubSampler(GOLD_SCRIPT)
    loop = make_loop(sampler, worker=True)

    outcome, _ = _cancel_when(loop, lambda w: returned.is_set())

    assert isinstance(outcome, asyncio.CancelledError)
    assert sampler.calls == []
    worker = loop.envs[0]
    assert worker.pid is not None  # it really was a ready worker with a live child
    _assert_reclaimed(worker)


def test_a_failing_worker_startup_is_reclaimed():
    sampler = StubSampler(GOLD_SCRIPT)
    loop = make_loop(sampler, worker=True)
    loop.worker_initializer = functools.partial(failing_startup, "boom in child")
    with pytest.raises(RuntimeError, match="boom in child"):
        run(loop)
    assert sampler.calls == []
    assert not loop.envs[0].is_alive and multiprocessing.active_children() == []
    loop.envs[0].close()


# -- cleanup outcome propagation (7A-R3) ----------------------------------------------------------


def _outcome(loop):
    """Run one rollout; return (result or raised exception, seconds until run() finished).

    The time is taken inside the event loop: asyncio.run() itself waits for executor
    threads (e.g. a hung close) before it returns."""

    async def go():
        loop.loop = asyncio.get_running_loop()
        started = time.monotonic()
        try:
            result = await loop.run({"temperature": 1.0}, task_id=TASK, uid="g0", rollout_n=0)
        except BaseException as error:  # noqa: BLE001 - returned for the assertions
            result = error
        return result, time.monotonic() - started

    return asyncio.run(go())


def test_first_cancel_during_the_normal_cleanup_is_propagated(monkeypatch):
    """The episode finished normally; the rollout is cancelled (for the first time) while the
    worker is being closed: the close completes, then CancelledError -- no output, no reward."""
    entered, finished = threading.Event(), threading.Event()
    real_close = ItsmEnvWorker.close

    def slow_close(self):
        entered.set()
        time.sleep(1.0)
        real_close(self)
        finished.set()

    monkeypatch.setattr(ItsmEnvWorker, "close", slow_close)
    sampler = StubSampler(GOLD_SCRIPT)
    loop = make_loop(sampler, worker=True)

    outcome, _ = _cancel_when(loop, lambda w: entered.is_set())

    assert isinstance(outcome, asyncio.CancelledError)  # not an AgentLoopOutput
    assert len(sampler.calls) == len(GOLD_SCRIPT)  # the episode itself had completed
    assert finished.is_set()  # the cleanup ran to completion before the cancellation propagated
    worker = loop.envs[0]
    assert not worker.is_alive and multiprocessing.active_children() == []
    real_close(worker)
    real_close(worker)


@pytest.mark.parametrize("origin", ["normal_end", "cancelled", "original_error"])
def test_unreclaimed_worker_is_an_infrastructure_error(monkeypatch, origin):
    """Fault injection: the cleanup really runs, but reports the child as not reclaimed."""
    real = ItsmAgentLoop._reclaim_sync

    def reports_not_reclaimed(self, env, started, *, cancel):
        real(self, env, started, cancel=cancel)
        return False

    monkeypatch.setattr(ItsmAgentLoop, "_reclaim_sync", reports_not_reclaimed)
    sampler = StubSampler(GOLD_SCRIPT, logprobs="nan" if origin == "original_error" else "ok")
    loop = make_loop(sampler, worker=True)
    if origin == "cancelled":
        outcome, _ = _cancel_when(loop, lambda w: len(sampler.calls) >= 3)
    else:
        outcome, _ = _outcome(loop)

    assert isinstance(outcome, ItsmInfrastructureError), outcome  # no output, no reward 0 either
    assert outcome.stage == "cleanup" and outcome.kind == "not_reclaimed"
    worker = loop.envs[0]
    diagnosis = outcome.cleanup
    assert diagnosis["reason"] == "the child is not confirmed gone after close()"
    assert diagnosis["episode_id"] == worker.episode_id and diagnosis["pid"] == worker.pid
    assert diagnosis["cancel_requested"] is (origin == "cancelled")
    assert worker.episode_id in str(outcome) and "not_reclaimed" in str(outcome)
    if origin == "normal_end":
        assert outcome.__cause__ is None and diagnosis["original_error"] is None
    elif origin == "cancelled":
        assert isinstance(outcome.__cause__, asyncio.CancelledError)
    else:
        assert isinstance(outcome.__cause__, ItsmInfrastructureError) and outcome.__cause__.stage == "sampler_contract"
        assert "non-finite log-prob" in diagnosis["original_error"]
    assert multiprocessing.active_children() == []  # the injected fault is in the report only


def test_cleanup_over_its_bound_is_an_infrastructure_error(monkeypatch):
    sampler = StubSampler(GOLD_SCRIPT)
    loop = make_loop(sampler, worker=True, call_timeout_s=2.0, kill_grace_s=0.5, cleanup_margin_s=0.2)
    bound = loop.cleanup_bound_s(cancel=False)
    hang = bound + 3.0
    real_close = ItsmEnvWorker.close
    close_entered = []
    real_run = ItsmAgentLoop.run

    def hung_close(self):
        close_entered.append(time.monotonic())
        time.sleep(hang)
        real_close(self)

    async def timed_run(self, *args, **kwargs):
        try:
            return await real_run(self, *args, **kwargs)
        finally:
            raised_at.append(time.monotonic())

    raised_at = []
    monkeypatch.setattr(ItsmEnvWorker, "close", hung_close)
    monkeypatch.setattr(ItsmAgentLoop, "run", timed_run)
    outcome, _ = _outcome(loop)
    elapsed = raised_at[0] - close_entered[0]  # from the start of the close to the report

    assert isinstance(outcome, ItsmInfrastructureError), outcome
    assert outcome.stage == "cleanup" and outcome.kind == "not_reclaimed"
    assert outcome.cleanup["reason"] == f"cleanup exceeded its bound of {bound:.1f}s"
    assert outcome.cleanup["cancel_requested"] is False and outcome.__cause__ is None
    assert bound - 1.0 <= elapsed < hang  # reported at the bound, not after the hang
    # hygiene: the hung close eventually reclaims the child (asyncio.run waited for it)
    assert not loop.envs[0].is_alive and multiprocessing.active_children() == []

