"""Single-process stand-in for verl's vLLM rollout server (step 7B engineering runs).

verl serves rollouts from ``vLLMHttpServer`` inside a Ray actor. Step 7B runs
``ItsmAgentLoop`` without Ray/FSDP, so this module builds the same vLLM engine
the same way and mirrors ``vLLMHttpServer.generate`` (verl @ 0a30f52,
``verl/workers/rollout/vllm_rollout/vllm_async_server.py``):

* engine: ``AsyncEngineArgs.from_cli_args`` of the CLI args verl derives from its
  rollout config (``build_cli_args_from_config``), ``create_engine_config``,
  ``AsyncLLM.from_vllm_config``;
* request: ``max_tokens`` clamped to ``[1, max_model_len - len(prompt)]``,
  ``logprobs = 0 if logprobs else None`` (the sampled token's log-prob only),
  ``repetition_penalty`` 1.0, ``ignore_eos`` False, ``TokensPrompt``;
* result: ``log_probs[i] = outputs[0].logprobs[i][token_ids[i]].logprob`` and
  ``stop_reason`` "completed" for finish reasons stop/length.

Not mirrored: Ray, ``worker_extension_cls``/weight sync (weights are loaded from
disk with ``load_format=auto`` instead of verl's ``dummy`` + actor sync), the
``monkey_patch_model`` vocab-size patch, LoRA, PD disaggregation, prometheus.

Every request is recorded (prompt/output length, finish reason, time to first
output, total time) for the step-7B resource record.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

#: verl rollout config defaults (verl/trainer/config/rollout/rollout.yaml) and this repo's
#: configs/runtime/pro6000_single_gpu.yaml, which overrides gpu_memory_utilization,
#: max_num_batched_tokens and max_num_seqs.
VERL_ENGINE_DEFAULTS: dict[str, Any] = {
    "dtype": "bfloat16",
    "load_format": "auto",  # verl: "dummy" + weight sync from the actor; here: the real weights
    "skip_tokenizer_init": False,
    "distributed_executor_backend": "mp",
    "max_num_seqs": 24,  # pro6000_single_gpu.yaml
    "enable_chunked_prefill": True,
    "max_num_batched_tokens": 32768,  # pro6000_single_gpu.yaml
    "enable_prefix_caching": True,
    "enable_sleep_mode": False,
    "logprobs_mode": "processed_logprobs",
    "enforce_eager": False,
    "gpu_memory_utilization": 0.45,  # pro6000_single_gpu.yaml (colocated with the actor in training)
    "disable_log_stats": False,
    "tensor_parallel_size": 1,
    "seed": 42,  # replica_rank 0 + rollout.seed 42
    "compilation_config": {"cudagraph_mode": "FULL_AND_PIECEWISE"},
}


@dataclass
class RequestRecord:
    request_id: str
    prompt_tokens: int
    max_tokens: int
    output_tokens: int = 0
    finish_reason: str | None = None
    stop_reason: str | None = None
    first_output_s: float | None = None
    total_s: float | None = None
    error: str | None = None


def verl_style_sampling(sampling_params: dict[str, Any], *, prompt_len: int, max_model_len: int) -> tuple[dict[str, Any], int]:
    """verl ``generate``'s parameter handling, without vLLM (unit-testable)."""
    params = dict(sampling_params)
    max_possible = max_model_len - prompt_len
    if max_possible < 1:
        raise ValueError(f"Prompt length ({prompt_len}) leaves no room to generate within max_model_len {max_model_len}")
    if "max_tokens" in params:
        max_tokens = params.pop("max_tokens")
    elif "max_new_tokens" in params:
        max_tokens = params.pop("max_new_tokens")
    else:
        raise ValueError("max_tokens is required (ItsmAgentLoop always sets it)")
    max_tokens = max(1, min(max_tokens, max_possible))
    params["logprobs"] = 0 if params.pop("logprobs", False) else None
    params.setdefault("repetition_penalty", 1.0)
    params.setdefault("ignore_eos", False)
    return params, max_tokens


def extract_token_logprobs(output) -> list[float] | None:
    """``outputs[0].logprobs[i][token_ids[i]].logprob`` exactly as verl does."""
    completion = output.outputs[0]
    if completion.logprobs is None:
        return None
    token_ids = completion.token_ids
    return [entry[token_ids[i]].logprob for i, entry in enumerate(completion.logprobs)]


def stop_reason_of(finish_reason: str | None) -> str | None:
    if finish_reason == "abort":
        return "aborted"
    if finish_reason in ("stop", "length"):
        return "completed"
    return finish_reason


@dataclass
class VllmTokenServer:
    """``server_manager`` for ``ItsmAgentLoop``: ``await generate(request_id=, prompt_ids=, sampling_params=)``."""

    engine: Any
    max_model_len: int
    records: list[RequestRecord] = field(default_factory=list)
    keep_top_logprobs: int | None = None  # length tests: also return top-k alternatives
    _counter: int = 0

    @classmethod
    def build(cls, model_path: str, *, max_model_len: int, overrides: dict[str, Any] | None = None, **kwargs) -> "VllmTokenServer":
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.usage.usage_lib import UsageContext
        try:  # vLLM > 0.11 (the remote has 0.24), as in verl's vllm_async_server
            from vllm.utils.argparse_utils import FlexibleArgumentParser
        except ImportError:
            from vllm.utils import FlexibleArgumentParser
        from vllm.v1.engine.async_llm import AsyncLLM

        from verl.workers.rollout.vllm_rollout.utils import build_cli_args_from_config

        args = {**VERL_ENGINE_DEFAULTS, **(overrides or {}), "max_model_len": max_model_len}
        args["compilation_config"] = json.dumps(args["compilation_config"])
        parser = AsyncEngineArgs.add_cli_args(FlexibleArgumentParser())
        cli = ["--model", model_path, *build_cli_args_from_config(args)]
        engine_args = AsyncEngineArgs.from_cli_args(parser.parse_args(cli))
        usage = UsageContext.OPENAI_API_SERVER
        vllm_config = engine_args.create_engine_config(usage_context=usage)
        engine = AsyncLLM.from_vllm_config(vllm_config=vllm_config, usage_context=usage, disable_log_stats=engine_args.disable_log_stats)
        server = cls(engine=engine, max_model_len=max_model_len, **kwargs)
        server.cli_args = cli
        server.vllm_config = vllm_config
        return server

    async def generate(self, *, request_id: str, prompt_ids: list[int], sampling_params: dict[str, Any]):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        params, max_tokens = verl_style_sampling(sampling_params, prompt_len=len(prompt_ids), max_model_len=self.max_model_len)
        if self.keep_top_logprobs and params["logprobs"] is not None:
            params["logprobs"] = self.keep_top_logprobs
        self._counter += 1
        vllm_request_id = f"{request_id}-{self._counter}"  # verl reuses one id per episode; vLLM needs unique live ids
        record = RequestRecord(request_id=vllm_request_id, prompt_tokens=len(prompt_ids), max_tokens=max_tokens)
        self.records.append(record)
        started = time.monotonic()
        final = None
        try:
            async for output in self.engine.generate(
                prompt=TokensPrompt(prompt_token_ids=list(prompt_ids)),
                sampling_params=SamplingParams(max_tokens=max_tokens, **params),
                request_id=vllm_request_id,
            ):
                if record.first_output_s is None:
                    record.first_output_s = round(time.monotonic() - started, 4)
                final = output
        except BaseException as error:
            record.error = f"{type(error).__name__}: {error}"
            raise
        finally:
            record.total_s = round(time.monotonic() - started, 4)
        if final is None or not final.outputs:
            record.stop_reason = "aborted"
            return SimpleNamespace(token_ids=[], log_probs=None, stop_reason="aborted", top_logprobs=None)
        completion = final.outputs[0]
        record.output_tokens = len(completion.token_ids)
        record.finish_reason = completion.finish_reason
        record.stop_reason = stop_reason_of(completion.finish_reason)
        top = None
        if self.keep_top_logprobs and completion.logprobs is not None:
            top = [{int(t): float(lp.logprob) for t, lp in entry.items()} for entry in completion.logprobs]
        return SimpleNamespace(
            token_ids=list(completion.token_ids),
            log_probs=extract_token_logprobs(final),
            stop_reason=record.stop_reason,
            finish_reason=completion.finish_reason,
            top_logprobs=top,
        )

    def shutdown(self) -> None:
        self.engine.shutdown()
