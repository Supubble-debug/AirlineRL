"""verl agent loop for the ITSM environment (step 7A: offline contract).

One episode of ``ItsmEnv`` driven by the policy that verl's rollout server
samples. The token contract (protocol §10, budget v1):

* ``prompt_ids``: Qwen3 chat template of ``build_action_messages(policy,
  reset observation)`` with the public tool schemas (Bench2 tools + ``done``),
  ``add_generation_prompt=True`` and the data config's chat-template kwargs
  (``enable_thinking=False``). Nothing private (task, gold, known_info, IDs)
  is rendered: only ``ItsmReset.public_info`` and the observation.
* each policy turn appends the sampler's ``token_ids`` with mask 1 and the
  sampler's own ``log_probs`` (``logprob_source="sampler"``). A sampler that
  returns no tokens, no log-probs, a count that does not match the tokens, or a
  non-finite log-prob (NaN/Inf) is an infrastructure failure
  (``ItsmInfrastructureError(stage="sampler_contract")``): no reward, nothing
  zero-filled.
* each environment turn appends ``turn_separator + user turn`` with mask 0 and
  0.0 placeholders (``logprob_source="placeholder_masked"``); the observation
  is ``ItsmTransition.observation`` verbatim (no compression, no cut).
* no context rotation (D3): when the next observation plus one action no
  longer fits ``response_length`` the episode ends with
  ``env.truncate("response_length")``.
* one action per turn: text plus a tool call, several tool calls, a malformed
  tool call, empty output or an output cut at ``max_action_tokens`` ends the
  episode with ``env.truncate("invalid_action")`` (D2: outcome 0).

Every episode runs in an ``ItsmEnvWorker`` child process (step 5): each call
waits at most ``call_timeout_s``, the episode at most ``episode_timeout_s``. When
the rollout task is cancelled the worker kills and joins the child before the
cancellation propagates, so an in-flight tool cannot finish a late write and no
reward is produced.

Task selection is purpose-gated: a ``grpo`` loop accepts only GRPO-trainable
tasks of the usage registry, a ``dev`` loop only dev tasks, and held-out needs
``final_evaluation``.

Verified offline with a deterministic sampler stub only; real sampled log-probs
(and whether verl recomputes them) are checked in step 7B.
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from enterprise_worlds.evaluator.text_match_strategy import TextMatchConfig

from agent_rl.data import itsm_task_sets
from agent_rl.envs.action_parser import ActionFormatError, ModelToolCall
from agent_rl.envs.itsm_env import DEFAULT_USER_LLM_ARGS, ItsmAction, ItsmEnvConfig, to_itsm_action
from agent_rl.envs.itsm_scoring import ItsmInfrastructureError
from agent_rl.envs.itsm_worker import ItsmEnvWorker
from agent_rl.prompts.action_prompt import OBSERVATION_PROMPT, build_action_messages

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput


LOOP_PURPOSES = ("grpo", "dev", "final_evaluation")
LOGPROB_SOURCE_SAMPLER = "sampler"
LOGPROB_SOURCE_PLACEHOLDER = "placeholder_masked"
_TOOL_CALL_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


@dataclass(frozen=True, slots=True)
class ItsmAgentLoopSettings:
    purpose: str
    max_steps: int = 40  # configs/itsm/budget_v1.yaml train.max_steps
    max_action_tokens: int = 512  # budget v1
    split_version: str = "itsm_split_v3"
    usage_version: str = "itsm_usage_v1"
    text_match: str = "fuzzy"  # D1b
    user_llm: str = "deepseek/deepseek-flash"  # D5 (2026-10-31)
    user_llm_args: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_USER_LLM_ARGS))  # D5
    user_llm_timeout_s: float = 60.0
    call_timeout_s: float = 90.0  # budget v1 worker.call_timeout_s
    episode_timeout_s: float = 1800.0  # budget v1 worker.episode_timeout_s

    def __post_init__(self) -> None:
        if self.purpose not in LOOP_PURPOSES:
            raise ValueError(f"purpose must be one of {list(LOOP_PURPOSES)}")
        if self.max_steps < 1 or self.max_action_tokens < 1:
            raise ValueError("max_steps and max_action_tokens must be positive")
        if self.call_timeout_s <= 0 or self.episode_timeout_s <= 0:
            raise ValueError("worker timeouts must be positive")


@dataclass(frozen=True, slots=True)
class ParsedAction:
    action: ItsmAction | None
    error: str | None


def check_sampler_output(token_ids: list[int], log_probs: Any, task_id: str) -> None:
    """Raise ``ItsmInfrastructureError(stage="sampler_contract")`` unless the sampler returned
    tokens and exactly one finite log-prob per token."""

    def fail(detail: str) -> None:
        raise ItsmInfrastructureError("sampler_contract", task_id, detail)

    if not token_ids:
        fail("rollout server returned no tokens")
    if log_probs is None:
        fail(f"rollout server returned no log-probs for {len(token_ids)} tokens")
    if len(log_probs) != len(token_ids):
        fail(f"rollout server must return one log-prob per generated token (got {len(log_probs)} for {len(token_ids)})")
    bad = [i for i, value in enumerate(log_probs) if not isinstance(value, (int, float)) or not math.isfinite(value)]
    if bad:
        fail(f"non-finite log-prob at positions {bad[:8]} (of {len(token_ids)})")


def allowed_task_ids(settings: ItsmAgentLoopSettings) -> frozenset[str]:
    if settings.purpose == "grpo":
        return frozenset(itsm_task_sets.training_task_ids(purpose="grpo", usage_version=settings.usage_version))
    if settings.purpose == "dev":
        return frozenset(itsm_task_sets.task_ids("dev", split_version=settings.split_version))
    return frozenset(itsm_task_sets.task_ids("heldout", split_version=settings.split_version, final_evaluation=True))


def parse_action(text_with_special: str, text_plain: str, *, stopped: bool) -> ParsedAction:
    """One model output -> one ``ItsmAction`` or an invalid-action reason.

    ``text_with_special`` keeps special tokens (tool-call detection),
    ``text_plain`` drops them (message text). Unlike verl's hermes parser,
    a malformed tool-call block is an error, not silently dropped, and text
    next to a tool call is an error, not discarded.
    """
    if not stopped:
        return ParsedAction(None, "action_token_limit: output ended without a stop token")
    blocks = _TOOL_CALL_BLOCK.findall(text_with_special)
    if "<tool_call>" in text_with_special and text_with_special.count("<tool_call>") != len(blocks):
        return ParsedAction(None, "unterminated <tool_call> block")
    calls = []
    for index, block in enumerate(blocks):
        try:
            payload = json.loads(block)
            name, arguments = payload["name"], payload.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            calls.append(ModelToolCall(name=name, arguments=arguments, tool_call_id=f"model_{index}"))
        except (ValueError, KeyError, TypeError, ActionFormatError) as error:
            return ParsedAction(None, f"malformed tool call: {error}")
    content = _TOOL_CALL_BLOCK.sub("", text_plain).strip()
    try:
        return ParsedAction(to_itsm_action(content=content or None, tool_calls=calls), None)
    except ActionFormatError as error:
        return ParsedAction(None, str(error))


class ItsmAgentLoop(AgentLoopBase):
    """Generate one ITSM trajectory with verl's current policy."""

    def __init__(self, *args: Any, settings: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.settings = ItsmAgentLoopSettings(**(settings or {}))
        self._check_template()

    # -- hooks ---------------------------------------------------------------------------------
    #: Child-side hooks for ``ItsmEnvWorker`` (picklable; tests inject a scripted user and
    #: fault injectors here). ``None`` = Bench2's user simulator, no initializer.
    user_factory = None
    worker_initializer = None

    def _make_env(self, task_id: str, episode_id: str) -> ItsmEnvWorker:
        config = ItsmEnvConfig(
            task_id=task_id,
            text_match=TextMatchConfig(strategy=self.settings.text_match),
            max_steps=self.settings.max_steps,
            user_llm=self.settings.user_llm,
            user_llm_args=dict(self.settings.user_llm_args),
            user_llm_timeout_s=self.settings.user_llm_timeout_s,
            split_version=self.settings.split_version,
        )
        return ItsmEnvWorker(
            config,
            call_timeout_s=self.settings.call_timeout_s,
            episode_timeout_s=self.settings.episode_timeout_s,
            user_factory=self.user_factory,
            initializer=self.worker_initializer,
            episode_id=episode_id,
        )

    # -- contract ------------------------------------------------------------------------------

    def _check_template(self) -> None:
        if self.apply_chat_template_kwargs.get("enable_thinking") is not False:
            raise ValueError("ITSM rollout requires apply_chat_template_kwargs.enable_thinking=False (budget v1)")
        if self.rollout_config.response_length < self.settings.max_action_tokens:
            raise ValueError("response_length must hold at least one action")

    def _stop_token_ids(self) -> list[int]:
        ids = {self.tokenizer.convert_tokens_to_ids("<|im_end|>")}
        if self.tokenizer.eos_token_id is not None:
            ids.add(self.tokenizer.eos_token_id)
        return sorted(ids)

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        task_id = kwargs.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("dataset field 'task_id' must be a non-empty string")
        if task_id not in allowed_task_ids(self.settings):
            raise PermissionError(f"task {task_id!r} is not allowed for a {self.settings.purpose!r} rollout")
        group_id = str(kwargs.get("uid", task_id))
        sample_index = int(kwargs.get("rollout_n", 0))
        episode_id = f"{task_id}:{group_id}:{sample_index}:{uuid4().hex[:8]}"
        env = await self.loop.run_in_executor(None, lambda: self._make_env(task_id, episode_id))
        try:
            return await self._run_episode(env, sampling_params, group_id, sample_index, task_id)
        except asyncio.CancelledError:
            # Stop the child itself, not only the awaiting thread: kill + join before the
            # cancellation propagates, so no in-flight tool finishes a late write.
            cancel = getattr(env, "cancel", None)
            if cancel is not None:
                cancel()
            raise
        finally:
            env.close()

    async def _run_episode(self, env, sampling_params: dict[str, Any], group_id: str, sample_index: int, task_id: str) -> AgentLoopOutput:
        reset = await self.loop.run_in_executor(None, env.reset)
        tools = reset.public_info["tools"]
        messages = build_action_messages(domain_policy=reset.public_info["policy"], observation=reset.observation)
        prompt_ids = await self.apply_chat_template(messages, tools=tools)
        response_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        turn_spans: list[dict[str, Any]] = []
        env_spans: list[dict[str, Any]] = []
        metrics = AgentLoopMetrics()
        request_id = uuid4().hex
        stop_ids = self._stop_token_ids()
        response_length = self.rollout_config.response_length
        transition = None

        while True:
            remaining = response_length - len(response_ids)
            max_tokens = min(self.settings.max_action_tokens, remaining)
            started = time.perf_counter()
            output = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids + response_ids,
                sampling_params={
                    **sampling_params,
                    "stop_token_ids": sorted(set((sampling_params.get("stop_token_ids") or []) + stop_ids)),
                    "max_tokens": max_tokens,
                },
            )
            metrics.generate_sequences += time.perf_counter() - started
            generated = list(output.token_ids)
            check_sampler_output(generated, output.log_probs, task_id)
            start = len(response_ids)
            response_ids.extend(generated)
            response_mask.extend([1] * len(generated))
            response_logprobs.extend(float(x) for x in output.log_probs)
            turn_index = len(turn_spans)
            turn_spans.append({"turn_index": turn_index, "start": start, "end": len(response_ids)})

            parsed = parse_action(
                self.tokenizer.decode(generated, skip_special_tokens=False),
                self.tokenizer.decode(generated, skip_special_tokens=True),
                stopped=generated[-1] in stop_ids,
            )
            started = time.perf_counter()
            if parsed.action is None:
                turn_spans[-1]["invalid_action"] = parsed.error
                transition = await self.loop.run_in_executor(None, lambda: env.truncate("invalid_action"))
            else:
                turn_spans[-1]["action_kind"] = parsed.action.kind
                action = parsed.action
                transition = await self.loop.run_in_executor(None, lambda: env.step(action))
            metrics.tool_calls += time.perf_counter() - started
            if transition.terminated or transition.truncated:
                break

            next_message = {"role": "user", "content": OBSERVATION_PROMPT.format(observation=transition.observation)}
            environment_ids = self.turn_separator + await self.apply_chat_template([next_message], remove_system_prompt=True)
            if len(response_ids) + len(environment_ids) + self.settings.max_action_tokens > response_length:
                transition = await self.loop.run_in_executor(None, lambda: env.truncate("response_length"))
                break
            env_start = len(response_ids)
            response_ids.extend(environment_ids)
            response_mask.extend([0] * len(environment_ids))
            response_logprobs.extend([0.0] * len(environment_ids))
            env_spans.append({"after_turn": turn_index, "start": env_start, "end": len(response_ids)})

        reward = transition.reward_info
        if reward is None:
            raise RuntimeError("ITSM episode ended without reward info")
        trace = env.export_trace()
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            reward_score=float(reward.training_reward),
            num_turns=len(turn_spans),
            metrics=metrics,
            extra_fields={
                "itsm_turn_spans": turn_spans,
                "itsm_env_spans": env_spans,
                "itsm_logprob_sources": {"mask_1": LOGPROB_SOURCE_SAMPLER, "mask_0": LOGPROB_SOURCE_PLACEHOLDER},
                "reward_extra_info": {
                    "itsm_episode_id": trace["episode_id"],
                    "itsm_task_id": trace["task_id"],
                    "itsm_group_id": group_id,
                    "itsm_sample_index": sample_index,
                    "itsm_purpose": self.settings.purpose,
                    "itsm_termination_reason": trace["termination_reason"],
                    "itsm_db_reward": float(reward.db_reward),
                    "itsm_training_reward_source": reward.training_reward_source,
                    "itsm_agent_turns": trace["agent_turns"],
                    "itsm_step_count": trace["step_count"],
                    "itsm_prompt_tokens": len(prompt_ids),
                    "itsm_policy_tokens": sum(response_mask),
                    "itsm_response_tokens": len(response_ids),
                },
            },
        )
