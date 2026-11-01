"""Step 7B tooling, offline: vLLM adapter (fake engine), scripted user, LLM guard, episode
runner, recompute alignment (tiny random Qwen3) and the length-input builder.

These check the code paths before GPU time is spent; they say nothing about real vLLM
log-probs, memory or model behaviour.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import math
import multiprocessing
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_DIR = Path(os.environ.get("ITSM_TOKENIZER_DIR", Path.home() / ".cache/entergym/tokenizers/Qwen3-8B-b968826d"))
if not (TOKENIZER_DIR / "tokenizer_config.json").exists():
    pytest.skip("pinned Qwen3-8B tokenizer not available", allow_module_level=True)

from transformers import AutoTokenizer  # noqa: E402

from agent_rl.data.itsm_task_sets import resolve_task  # noqa: E402
from agent_rl.envs.itsm_scripted_user import CLOSING, FOLLOW_UPS, PersonaScriptedUser, forbid_external_llm, opening_message  # noqa: E402
from agent_rl.rollout.vllm_token_server import VERL_ENGINE_DEFAULTS, VllmTokenServer, extract_token_logprobs, stop_reason_of, verl_style_sampling  # noqa: E402


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOKENIZER = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
IM_END = TOKENIZER.convert_tokens_to_ids("<|im_end|>")
SPLIT = "itsm_split_v3"


# -- verl-equivalent request handling -------------------------------------------------------------


def test_sampling_params_follow_verl_generate():
    params, max_tokens = verl_style_sampling({"temperature": 1.0, "logprobs": True, "max_tokens": 512}, prompt_len=40700, max_model_len=40960)
    assert max_tokens == 260  # clamped to the remaining context, as verl does
    assert params == {"temperature": 1.0, "logprobs": 0, "repetition_penalty": 1.0, "ignore_eos": False}
    params, _ = verl_style_sampling({"max_tokens": 5}, prompt_len=1, max_model_len=10)
    assert params["logprobs"] is None  # no logprobs requested -> none returned (and the loop refuses)
    with pytest.raises(ValueError, match="no room"):
        verl_style_sampling({"max_tokens": 5}, prompt_len=10, max_model_len=10)
    assert stop_reason_of("length") == "completed" and stop_reason_of("abort") == "aborted"
    assert VERL_ENGINE_DEFAULTS["logprobs_mode"] == "processed_logprobs" and VERL_ENGINE_DEFAULTS["gpu_memory_utilization"] == 0.45


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@pytest.fixture
def fake_vllm(monkeypatch):
    vllm = types.ModuleType("vllm")
    vllm.SamplingParams = FakeSamplingParams
    inputs = types.ModuleType("vllm.inputs")
    inputs.TokensPrompt = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.inputs", inputs)


class FakeEngine:
    """Replays assistant texts; returns per-token logprob dicts like vLLM (sampled token + top-k)."""

    def __init__(self, texts):
        self.texts = list(texts)
        self.seen = []

    async def generate(self, *, prompt, sampling_params, request_id):
        self.seen.append((request_id, len(prompt["prompt_token_ids"]), sampling_params))
        text = self.texts[min(len(self.seen) - 1, len(self.texts) - 1)]
        ids = TOKENIZER.encode(text, add_special_tokens=False)[: sampling_params.max_tokens - 1] + [IM_END]
        logprobs = None
        if sampling_params.logprobs is not None:
            logprobs = [{t: SimpleNamespace(logprob=-0.5 - 0.01 * i), 7: SimpleNamespace(logprob=-9.0)} for i, t in enumerate(ids)]
        completion = SimpleNamespace(token_ids=ids, logprobs=logprobs, finish_reason="stop")
        yield SimpleNamespace(outputs=[completion])

    def reset_prefix_cache(self):
        return None

    def shutdown(self):
        pass


def test_adapter_returns_one_logprob_per_token_like_verl(fake_vllm):
    server = VllmTokenServer(engine=FakeEngine(["hello"]), max_model_len=100)
    out = asyncio.run(server.generate(request_id="r", prompt_ids=[1, 2, 3], sampling_params={"temperature": 1.0, "logprobs": True, "max_tokens": 50}))
    assert len(out.log_probs) == len(out.token_ids) and out.log_probs[0] == -0.5
    assert out.stop_reason == "completed" and out.finish_reason == "stop"
    record = server.records[0]
    assert record.prompt_tokens == 3 and record.output_tokens == len(out.token_ids) and record.total_s is not None
    assert server.engine.seen[0][2].logprobs == 0  # sampled token only
    server.keep_top_logprobs = 5
    out = asyncio.run(server.generate(request_id="r", prompt_ids=[1], sampling_params={"temperature": 0.0, "logprobs": True, "max_tokens": 50}))
    assert server.engine.seen[1][2].logprobs == 5 and out.top_logprobs[0][7] == -9.0
    assert server.engine.seen[0][0] != server.engine.seen[1][0]  # unique vLLM request ids


def test_extract_matches_sampled_token_not_the_top_one():
    completion = SimpleNamespace(token_ids=[5, 6], logprobs=[{9: SimpleNamespace(logprob=-0.1), 5: SimpleNamespace(logprob=-2.0)},
                                                            {6: SimpleNamespace(logprob=-0.3)}])
    assert extract_token_logprobs(SimpleNamespace(outputs=[completion])) == [-2.0, -0.3]


# -- scripted user and LLM guard ------------------------------------------------------------------


def test_scripted_user_discloses_only_persona_facts():
    task = resolve_task("itsmbench_msp_013", split_version=SPLIT)
    text = opening_message(task)
    assert "Joanne Simpson" in text and "CHG0000026" in text and "Printer Jam - Tray 2" in text
    assert task.scenario.task_description[:60] not in text and "You are" not in text
    user = PersonaScriptedUser("itsmbench_msp_013", SPLIT)
    state = user.get_init_state()
    replies = []
    for _ in range(len(FOLLOW_UPS) + 3):
        message, state = user.generate_next_message(SimpleNamespace(content="agent"), state)
        replies.append(message.content)
    assert replies[0] == text and replies[1:4] == list(FOLLOW_UPS) and replies[4] == CLOSING and replies[-1] == CLOSING
    assert PersonaScriptedUser.is_stop(SimpleNamespace(content=CLOSING))


def test_llm_guard_makes_any_litellm_call_raise(monkeypatch):
    monkeypatch.delitem(sys.modules, "litellm", raising=False)
    forbid_external_llm()
    import litellm

    with pytest.raises(RuntimeError, match="forbidden"):
        litellm.completion(model="x", messages=[])
    with pytest.raises(RuntimeError, match="forbidden"):
        litellm.anything_else()
    monkeypatch.delitem(sys.modules, "litellm")


# -- episode runner end to end (fake engine, real worker children) ------------------------------


def test_episode_runner_records_real_loop_outputs(fake_vllm, tmp_path, monkeypatch):
    gpu = _load("run_7b_gpu", "scripts/itsm/run_7b_gpu.py")
    monkeypatch.setattr(gpu, "EPISODE_TASKS", (("grpo", "itsmbench_msp_013"), ("dev", "itsmbench_msp_019")))
    texts = ["Hello, how can I help?", '<tool_call>\n{"name": "find_change_by_number", "arguments": {"number": "CHG0000026"}}\n</tool_call>',
             '<tool_call>\n{"name": "done", "arguments": {}}\n</tool_call>']

    class Engine(FakeEngine):
        async def generate(self, *, prompt, sampling_params, request_id):
            # restart the script for every episode
            async for out in FakeEngine.generate(self, prompt=prompt, sampling_params=sampling_params, request_id=request_id):
                yield out
            if len(self.seen) == len(texts):
                self.seen.clear()

    server = VllmTokenServer(engine=Engine(texts), max_model_len=40960)

    async def go():
        return await gpu.run_episodes(server, TOKENIZER, tmp_path)

    episodes = asyncio.run(go())
    assert [e["ok"] for e in episodes] == [True, True], [e.get("error") for e in episodes]
    for e in episodes:
        assert e["extra_fields"]["reward_extra_info"]["itsm_termination_reason"] == "agent_done"
        assert e["num_turns"] == 3 and e["logprob_check"]["all_finite"] and len(e["requests"]) == 3
        assert len(e["response_ids"]) == len(e["response_mask"]) == len(e["response_logprobs"])
        assert "find_change_by_number" in e["turn_texts"][1]
        assert e["sampling"]["temperature"] == 1.0
    assert json.loads((tmp_path / "episodes.json").read_text())[0]["task_id"] == "itsmbench_msp_013"
    assert multiprocessing.active_children() == []


# -- recompute alignment --------------------------------------------------------------------------


def test_recompute_logprobs_align_with_a_naive_full_forward():
    torch = pytest.importorskip("torch")
    from transformers import Qwen3Config, Qwen3ForCausalLM

    gpu = _load("run_7b_gpu", "scripts/itsm/run_7b_gpu.py")
    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=8, max_position_embeddings=128)
    model = Qwen3ForCausalLM(config).eval()
    ids = torch.randint(0, 97, (40,)).tolist()
    positions = [5, 6, 7, 20, 39]
    for temperature in (1.0, 0.7):
        verl_lp, fp32_lp = gpu.recompute_token_logprobs(model, ids, positions, temperature, device="cpu", chunk=2)
        with torch.no_grad():
            logits = model(torch.tensor([ids])).logits[0].float() / temperature
        naive = [torch.log_softmax(logits[p - 1], -1)[ids[p]].item() for p in positions]
        assert verl_lp == pytest.approx(naive, abs=1e-5) and fp32_lp == pytest.approx(naive, abs=1e-5)
    with pytest.raises(ValueError):
        gpu.recompute_token_logprobs(model, ids, [0], 1.0, device="cpu")


# -- length inputs ----------------------------------------------------------------------------------


def test_length_inputs_reproduce_the_committed_meta(tmp_path):
    """The CLI (which pins PYTHONHASHSEED=0) reproduces the committed meta byte for byte."""
    import subprocess

    out = tmp_path / "inputs.json"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}
    result = subprocess.run([sys.executable, "scripts/itsm/build_7b_length_inputs.py", "--tokenizer-dir", str(TOKENIZER_DIR), "--out", str(out),
                             "--check", "docs/itsm_step7b/length_inputs_meta.json"], cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stderr[-2000:]
    tiers = json.loads(out.read_text())["tiers"]
    assert tiers["a"]["prompt_tokens"] == 28104  # step 6 measurement
    assert tiers["b"]["prompt_tokens"] <= 32768 - 512 and tiers["c"]["prompt_tokens"] <= 40960 - 512
    assert tiers["c"]["natural"] is False and tiers["b"]["target_token_ids"] == tiers["c"]["target_token_ids"]
    assert tiers["b"]["prompt_token_ids"][: tiers["a"]["prompt_tokens"]] == tiers["a"]["prompt_token_ids"]
    assert all(not math.isnan(t["prompt_tokens"]) for t in tiers.values())
