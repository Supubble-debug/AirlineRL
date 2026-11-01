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
waits at most ``call_timeout_s``, the episode at most ``episode_timeout_s``.

Worker lifecycle (7A-R2): the loop takes the worker handle *before* any child
exists (``start=False``, cheap, on the event loop), then starts it in a thread
and waits at most ``startup_timeout_s`` plus the kill grace. Whatever happens
next -- normal end, error, startup timeout, or cancellation during startup or
during the episode -- the handle is reclaimed by one bounded cleanup that runs
in its own task: ``cancel()`` (kills the child, or stops it from being spawned),
a bounded wait for the start thread to return, then ``close()`` (kill + join).
A cancellation cannot interrupt that cleanup; it is remembered and re-raised
after it, so no worker outlives its rollout, no in-flight tool finishes a late
write and no reward is produced.

Propagation after the cleanup (7A-R3), first match wins:

1. cleanup failed (child not confirmed gone, cleanup error or cleanup over its
   bound) -> ``ItsmInfrastructureError(stage="cleanup", kind="not_reclaimed")``,
   chained (``__cause__``) to the original exception or cancellation, with the
   episode id, pid, reason, whether a cancellation was requested and the
   original error in ``detail`` and in ``.cleanup``. A possibly live child must
   not be hidden behind a ``CancelledError``.
2. a cancellation was requested at any point, including for the first time
   during a normal cleanup -> ``CancelledError`` (the original one if any).
3. the episode raised -> that exception, unchanged.
4. otherwise -> the ``AgentLoopOutput``.

No branch other than 4 returns an output, so none of them yields a reward
(and none is turned into reward 0).

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
import logging
import math
import re
import threading
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


logger = logging.getLogger(__name__)

LOOP_PURPOSES = ("grpo", "dev", "final_evaluation")
CLEANUP_MARGIN_S = 5.0
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
    startup_timeout_s: float = 120.0  # worker spawn + ItsmEnv construction (7A-R2)
    kill_grace_s: float = 10.0  # bound on kill + join of the child (7A-R2)
    #: step 8 (dev): after the fuzzy training reward, run the D7 judges in the worker on the same
    #: final DB (``aux_evaluator_factory``); results go to extra_fields["itsm_aux_eval"] only.
    auxiliary_evaluation: bool = False
    aux_timeout_s: float = 2 * 4 * 120.0 + 60.0  # two judges x four attempts x judge timeout + margin
    #: protocol §13.1: refuse an env worker whose interpreter was not started with PYTHONHASHSEED=0
    require_fixed_hash_seed: bool = True
    cleanup_margin_s: float = CLEANUP_MARGIN_S  # slack added to each cleanup bound (7A-R3: settable for tests)

    def __post_init__(self) -> None:
        if self.purpose not in LOOP_PURPOSES:
            raise ValueError(f"purpose must be one of {list(LOOP_PURPOSES)}")
        if self.max_steps < 1 or self.max_action_tokens < 1:
            raise ValueError("max_steps and max_action_tokens must be positive")
        if min(self.call_timeout_s, self.episode_timeout_s, self.startup_timeout_s, self.kill_grace_s, self.cleanup_margin_s) <= 0:
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
    aux_evaluator_factory = None

    def _make_env(self, task_id: str, episode_id: str) -> ItsmEnvWorker:
        """The worker *handle*: no child process yet (``start=False``); ``_start_env`` spawns it."""
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
            aux_evaluator_factory=self.aux_evaluator_factory if self.settings.auxiliary_evaluation else None,
            episode_id=episode_id,
            startup_timeout_s=self.settings.startup_timeout_s,
            kill_grace_s=self.settings.kill_grace_s,
            start=False,
            require_fixed_hash_seed=self.settings.require_fixed_hash_seed,
        )

    # -- worker lifecycle (7A-R2) --------------------------------------------------------------

    async def _start_env(self, env, started: threading.Event, task_id: str) -> None:
        """Start the worker in a thread; wait at most startup + kill grace + margin."""
        start = getattr(env, "start", None)  # in-process test envs have no start()
        if start is None:
            started.set()
            return
        outcome: dict[str, BaseException] = {}

        def run_start() -> None:
            try:
                start()
            except BaseException as error:  # noqa: BLE001 - re-raised on the event loop
                outcome["error"] = error
            finally:
                started.set()

        self.loop.run_in_executor(None, run_start)
        deadline = time.monotonic() + self.settings.startup_timeout_s + self.settings.kill_grace_s + self.settings.cleanup_margin_s
        while not started.is_set():
            if time.monotonic() > deadline:
                raise ItsmInfrastructureError("startup", task_id, "worker start did not return within its bound", kind="not_reclaimed")
            await asyncio.sleep(0.02)
        if "error" in outcome:
            raise outcome["error"]

    def _reclaim_sync(self, env, started: threading.Event, *, cancel: bool) -> bool:
        """Blocking, bounded: stop the child (if cancelling), let start() return, close; True if gone."""
        if cancel and hasattr(env, "cancel"):
            env.cancel()
        started.wait(self.settings.kill_grace_s + self.settings.cleanup_margin_s)
        if not cancel and not started.is_set() and hasattr(env, "cancel"):
            env.cancel()  # start() still running: never close under a live startup
            started.wait(self.settings.kill_grace_s + self.settings.cleanup_margin_s)
        env.close()
        return not getattr(env, "is_alive", False)

    def cleanup_bound_s(self, *, cancel: bool) -> float:
        """Upper bound of one cleanup: start wait(s) + (healthy export + shutdown) + kill/join."""
        step = self.settings.kill_grace_s + self.settings.cleanup_margin_s
        healthy_close = 0.0 if cancel else 2 * self.settings.call_timeout_s
        return 2 * step + healthy_close + step

    async def _reclaim(self, env, started: threading.Event, *, cancel: bool) -> tuple[bool, str | None]:
        """Run the bounded cleanup in its own task; a cancellation cannot interrupt it.

        Returns ``(cancel_requested, failure)``: whether the rollout was cancelled while the
        cleanup ran, and ``None`` or the reason the child is not confirmed reclaimed.
        """
        bound = self.cleanup_bound_s(cancel=cancel)
        cleanup = asyncio.ensure_future(
            asyncio.wait_for(self.loop.run_in_executor(None, lambda: self._reclaim_sync(env, started, cancel=cancel)), bound)
        )
        cancel_requested = False
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                if not cancel_requested and hasattr(env, "cancel"):
                    env.cancel()  # first cancellation during a normal cleanup: stop the child now
                cancel_requested = True  # remembered; run() re-raises it after the cleanup
            except BaseException:  # noqa: BLE001 - inspected below
                break
        failure = None
        if cleanup.cancelled():
            failure = "cleanup task was cancelled"
        elif isinstance(cleanup.exception(), (asyncio.TimeoutError, TimeoutError)):
            failure = f"cleanup exceeded its bound of {bound:.1f}s"
        elif cleanup.exception() is not None:
            error = cleanup.exception()
            failure = f"cleanup raised {type(error).__name__}: {error}"
        elif cleanup.result() is not True:
            failure = "the child is not confirmed gone after close()"
        if failure is not None:
            logger.error("ITSM worker %s not reclaimed (pid %s): %s", getattr(env, "episode_id", "?"), getattr(env, "pid", None), failure)
        return cancel_requested, failure

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
        env = self._make_env(task_id, episode_id)  # handle first, before any child exists
        started = threading.Event()
        output: AgentLoopOutput | None = None
        error: BaseException | None = None
        try:
            await self._start_env(env, started, task_id)
            output = await self._run_episode(env, sampling_params, group_id, sample_index, task_id)
        except BaseException as raised:  # noqa: BLE001 - re-raised below, after the cleanup
            error = raised
        # Only a cancellation kills the child directly; other errors take the normal close
        # (a failed worker skips the trace export and is killed, a healthy one keeps its trace).
        cancelled = isinstance(error, asyncio.CancelledError)
        cancel_during_cleanup, failure = await self._reclaim(env, started, cancel=cancelled)
        cancel_requested = cancelled or cancel_during_cleanup
        if failure is not None:
            diagnosis = {
                "episode_id": getattr(env, "episode_id", episode_id),
                "pid": getattr(env, "pid", None),
                "reason": failure,
                "cancel_requested": cancel_requested,
                "original_error": None if error is None else f"{type(error).__name__}: {error}",
                "worker_failure": getattr(env, "failure", None),
            }
            infra = ItsmInfrastructureError("cleanup", task_id, json.dumps(diagnosis, sort_keys=True, default=str), kind="not_reclaimed")
            infra.cleanup = diagnosis
            raise infra from error
        if cancel_requested:
            if cancelled:
                raise error
            raise asyncio.CancelledError("rollout cancelled during worker cleanup")
        if error is not None:
            raise error
        return output

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
        aux = None
        if self.settings.auxiliary_evaluation:
            # Same final DB and trajectory; the fuzzy reward above is not touched.
            aux = await self.loop.run_in_executor(None, lambda: env.aux_evaluate(self.settings.aux_timeout_s))
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
                "itsm_aux_eval": aux,
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
