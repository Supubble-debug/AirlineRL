"""Single-episode ITSM environment adapter (protocol §2–§9; steps 4–5).

One ``ItsmEnv`` drives one episode synchronously, with no background thread:
``step`` executes the policy's tool call, or obtains the next user turn, in the
calling thread under the task clock, and returns only what is new. Business
tools, the user simulator and DB scoring are Bench2's own; the message order
and the ``max_steps`` budget match Bench2's ``Orchestrator.run``.

Model-visible call IDs (``call_001`` …) are independent of ``episode_id`` and
``task_id``; the global key for a call is ``(episode_id, call_id)``.

Lifecycle (protocol §7.1)::

    new --reset--> active --step/truncate--> interaction_ended --score--> scored
                     |                              |  ^
                     |                              v  | score (same final DB)
                     |                       scoring_retry_pending
                     v                              |
                   failed  <------------------------+ (attempts exhausted)

``closed`` is a separate flag. Calls in this process cannot be interrupted, so
this class has no timeout; ``agent_rl.envs.itsm_worker.ItsmEnvWorker`` runs it in
a child process that is killed on timeout or cancellation.
"""

from __future__ import annotations

import copy
import inspect
import time
import typing
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, Literal, Protocol

from pydantic import TypeAdapter, ValidationError

from enterprise_worlds.data_model.message import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from enterprise_worlds.data_model.tasks import Task
from enterprise_worlds.domains.itsm.tools import ItsmError
from enterprise_worlds.environment.environment import Environment, _stringify
from enterprise_worlds.evaluator.text_match_strategy import TextMatchConfig
from enterprise_worlds.gym.gym_agent import DONE_TOOL_SCHEMA
from enterprise_worlds.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE
from enterprise_worlds.user.base import STOP
from enterprise_worlds.user.user_simulator import UserSimulator

from agent_rl.data.schemas import ToolCallRecord
from agent_rl.envs.action_parser import ActionFormatError, ModelToolCall
from agent_rl.envs.itsm_scoring import (
    ItsmInfrastructureError,
    TaskDefinitionError,
    build_task_environment,
    load_itsm_tasks,
    score_final_db,
    task_clock,
    validate_text_match,
)


DONE_TOOL_NAME = DONE_TOOL_SCHEMA["function"]["name"]
#: Decision D5: the user simulator model, temperature 0.
DEFAULT_USER_LLM = "deepseek/deepseek-v4-flash"
DEFAULT_USER_LLM_ARGS = {"temperature": 0.0}
#: Bound on each user-simulator HTTP call (passed to litellm as ``timeout``).
DEFAULT_USER_LLM_TIMEOUT_S = 60.0
#: Protocol §12: three retries after the first scoring attempt, four in total.
DEFAULT_MAX_SCORING_RETRIES = 3
ERROR_PREFIX = "Error: "

ActionKind = Literal["message", "tool_call", "done"]
ErrorClass = Literal["unknown_tool", "bad_arguments", "domain_error"]
TerminationReason = Literal[
    "agent_done", "user_stop", "invalid_action", "max_steps", "response_length"
]
EpisodeStatus = Literal[
    "new", "active", "interaction_ended", "scoring_retry_pending", "scored", "failed"
]
TERMINATED_REASONS = frozenset({"agent_done", "user_stop", "invalid_action"})
TRUNCATED_REASONS = frozenset({"max_steps", "response_length"})
#: Reasons the rollout loop may impose from outside (protocol §7).
EXTERNAL_END_REASONS = frozenset({"response_length", "invalid_action"})


@dataclass(frozen=True, slots=True)
class ItsmAction:
    """Exactly one of: a message to the user, one tool call, or ``done``."""

    kind: ActionKind
    text: str | None = None
    name: str | None = None
    arguments: dict[str, Any] | None = None
    model_call_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "message":
            if not self.text or not self.text.strip():
                raise ActionFormatError("message text must not be empty")
            if STOP in self.text:
                raise ActionFormatError(f"message text must not contain the reserved {STOP} token")
        elif self.kind == "tool_call":
            if not self.name or not self.name.strip():
                raise ActionFormatError("tool name must not be empty")
            if self.name == DONE_TOOL_NAME:
                raise ActionFormatError("end the episode with ItsmAction.done()")
            if not isinstance(self.arguments, dict):
                raise ActionFormatError("tool arguments must be a dictionary")
        elif self.kind != "done":
            raise ActionFormatError(f"unknown action kind {self.kind!r}")

    @classmethod
    def message(cls, text: str) -> ItsmAction:
        return cls("message", text=text)

    @classmethod
    def tool_call(
        cls, name: str, arguments: dict[str, Any], model_call_id: str | None = None
    ) -> ItsmAction:
        return cls("tool_call", name=name, arguments=arguments, model_call_id=model_call_id)

    @classmethod
    def done(cls) -> ItsmAction:
        return cls("done")


def to_itsm_action(
    *, content: str | None, tool_calls: Sequence[ModelToolCall] | None = None
) -> ItsmAction:
    """Convert one parsed model output into an action (same rules as ``to_tau_action``)."""
    calls = tuple(tool_calls or ())
    has_content = content is not None and bool(content.strip())

    if has_content and calls:
        raise ActionFormatError("An ITSM action cannot contain both text and tool calls")
    if len(calls) > 1:
        raise ActionFormatError("An ITSM action can contain at most one tool call")
    if calls:
        call = calls[0]
        if call.name == DONE_TOOL_NAME:
            if call.arguments:
                raise ActionFormatError("done() takes no arguments")
            return ItsmAction.done()
        return ItsmAction.tool_call(
            call.name, call.arguments, model_call_id=call.tool_call_id or None
        )
    if has_content:
        return ItsmAction.message(content.strip())
    raise ActionFormatError("Model output must contain text or one tool call")


@dataclass(frozen=True)
class ItsmEnvConfig:
    """``max_steps`` has no default: the formal budget is fixed in step 6."""

    task_id: str
    text_match: TextMatchConfig
    max_steps: int
    user_llm: str = DEFAULT_USER_LLM
    user_llm_args: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_USER_LLM_ARGS))
    user_llm_timeout_s: float = DEFAULT_USER_LLM_TIMEOUT_S
    max_scoring_retries: int = DEFAULT_MAX_SCORING_RETRIES
    score_on_finish: bool = True

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must not be empty")
        if not self.user_llm.strip():
            raise ValueError("user_llm must not be empty")
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int) or self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if self.user_llm_timeout_s <= 0:
            raise ValueError("user_llm_timeout_s must be positive")
        if self.max_scoring_retries < 0:
            raise ValueError("max_scoring_retries must be non-negative")
        validate_text_match(self.text_match)


class UserModel(Protocol):
    """The part of Bench2's ``UserSimulator`` interface the env drives."""

    def get_init_state(self) -> Any: ...

    def generate_next_message(
        self, agent_message: AssistantMessage, state: Any
    ) -> tuple[UserMessage, Any]: ...


@dataclass(frozen=True, slots=True)
class ItsmToolCallRecord:
    call_id: str
    name: str
    arguments: dict[str, Any]
    model_call_id: str | None
    result_received: bool
    result: str | None
    error: str | None
    error_class: ErrorClass | None
    error_code: str | None
    error_field: str | None
    duration_ms: float

    def __post_init__(self) -> None:
        # Same invariant as the public ToolCallRecord: an error is a received result.
        if self.error is not None and not self.result_received:
            raise ValueError("a tool error requires result_received=True")

    def to_tool_call_record(self) -> ToolCallRecord:
        return ToolCallRecord(
            call_id=self.call_id,
            name=self.name,
            arguments=copy.deepcopy(self.arguments),
            result=self.result,
            error=self.error,
            result_received=self.result_received,
            duration_ms=self.duration_ms,
        )


@dataclass(frozen=True, slots=True)
class ItsmRewardInfo:
    db_evaluated: bool
    db_match: bool | None
    db_reward: float | None
    db_mismatches: tuple[str, ...]
    text_match_requested: str
    text_match_effective: str
    text_judge_model: str | None
    nl_status: Literal["not_evaluated", "evaluated"]
    nl_reward: float | None
    joint_reward: float | None
    training_reward: float
    training_reward_source: str
    gold_db_hash: str
    final_db_hash: str
    scoring_attempts: int


@dataclass(frozen=True, slots=True)
class ItsmReset:
    observation: str
    public_info: dict[str, Any]
    episode_id: str
    step_count: int


@dataclass(frozen=True, slots=True)
class ItsmTransition:
    observation: str
    terminated: bool
    truncated: bool
    termination_reason: TerminationReason | None
    tool_record: ItsmToolCallRecord | None
    reward_info: ItsmRewardInfo | None
    step_count: int
    agent_turns: int


@lru_cache(maxsize=1)
def _tasks_by_id() -> dict[str, Task]:
    return load_itsm_tasks()


def public_call_id(seq: int) -> str:
    """Model-visible call ID: unique within one episode, carries no task or episode data."""
    return f"call_{seq:03d}"


@lru_cache(maxsize=None)
def _parameter_validators(function: Any) -> dict[str, TypeAdapter]:
    """Strict validators from the annotations that also generate the public tool schema."""
    hints = typing.get_type_hints(function)
    hints.pop("return", None)
    return {name: TypeAdapter(hint) for name, hint in hints.items()}


def argument_type_error(tool: Any, arguments: dict[str, Any]) -> str | None:
    """Return why ``arguments`` do not match the tool's annotated types, else ``None``."""
    validators = _parameter_validators(getattr(tool, "__func__", tool))
    for name, value in arguments.items():
        validator = validators.get(name)
        if validator is None:
            continue
        try:
            validator.validate_python(value, strict=True)
        except ValidationError as error:
            return f"argument {name!r}: {error.errors()[0]['msg']}"
    return None


class ItsmEnv:
    """One ITSM episode: ``reset`` once, ``step`` until it ends, ``score``, then ``close``."""

    def __init__(
        self,
        config: ItsmEnvConfig,
        *,
        user: UserModel | None = None,
        episode_id: str | None = None,
        task: Task | None = None,
    ) -> None:
        """``task`` supplies a task outside Bench2's ``tasks.json`` (e.g. the step 6B
        pilot); its id must equal ``config.task_id``."""
        if task is None:
            tasks = _tasks_by_id()
            if config.task_id not in tasks:
                raise ValueError(f"unknown ITSM task_id {config.task_id!r}")
            task = tasks[config.task_id]
        elif task.id != config.task_id:
            raise ValueError(f"task.id {task.id!r} does not match config.task_id {config.task_id!r}")
        self.config = config
        self.task = task
        self.episode_id = episode_id or f"{self.task.id}:{uuid.uuid4().hex[:8]}"
        self._user = user if user is not None else UserSimulator(
            self.task.scenario,
            llm=config.user_llm,
            llm_args={**config.user_llm_args, "timeout": config.user_llm_timeout_s},
        )
        self._environment: Environment | None = None
        self._user_state: Any = None
        self._messages: list[Message] = []
        self._tool_records: list[ItsmToolCallRecord] = []
        self._call_seq = 0
        self._step_count = 0
        self._agent_turns = 0
        self._status: EpisodeStatus = "new"
        self._closed = False
        self._termination_reason: TerminationReason | None = None
        self._end_tool_record: ItsmToolCallRecord | None = None
        self._final_db_hash: str | None = None
        self._reward_info: ItsmRewardInfo | None = None
        self._scoring_errors: list[dict[str, Any]] = []
        self._failure: dict[str, Any] | None = None

    # -- public API ----------------------------------------------------------

    def reset(self) -> ItsmReset:
        if self._closed:
            raise RuntimeError("the environment is closed")
        if self._status != "new":
            raise RuntimeError("ItsmEnv runs exactly one episode; create a new instance")
        self._status = "active"
        try:
            self._environment = build_task_environment(self.task)
            self._user_state = self._user.get_init_state()
        except TaskDefinitionError as error:
            self._fail("task_definition", error)
            raise
        except Exception as error:
            raise self._infrastructure("env_build", error) from error

        greeting = AssistantMessage(content=DEFAULT_FIRST_AGENT_MESSAGE)
        self._messages.append(greeting)
        user_message = self._user_turn(greeting)
        if STOP in user_message.content:
            raise self._infrastructure(
                "user_sim_empty_episode", "user stopped before the agent's first turn"
            )
        return ItsmReset(
            observation=f"assistant: {greeting.content}\nuser: {user_message.content}",
            public_info=self._public_info(),
            episode_id=self.episode_id,
            step_count=self._step_count,
        )

    def step(self, action: ItsmAction) -> ItsmTransition:
        self._require_active()
        if not isinstance(action, ItsmAction):
            raise TypeError(f"action must be an ItsmAction, got {type(action).__name__}")
        self._agent_turns += 1
        if action.kind == "done":
            # Orchestrator checks the agent's stop before the budget.
            return self._end_interaction("agent_done", AssistantMessage(content=STOP))
        if self._step_count >= self.config.max_steps:
            return self._end_on_step_budget(action)
        if action.kind == "tool_call":
            return self._step_tool_call(action)
        return self._step_message(action)

    def truncate(self, reason: str) -> ItsmTransition:
        """End the interaction for a reason the rollout loop decides (protocol §7).

        ``response_length``: the token budget cannot fit another action.
        ``invalid_action``: the policy output could not be parsed (counts as a turn).
        """
        if reason not in EXTERNAL_END_REASONS:
            raise ValueError(f"truncate() reason must be one of {sorted(EXTERNAL_END_REASONS)}")
        self._require_active()
        if reason == "invalid_action":
            self._agent_turns += 1
        return self._end_interaction(reason)

    def score(self) -> ItsmTransition:
        """Score the saved final DB once; a valid result (DB=1 or DB=0) is reused.

        After an infrastructure scoring failure the same final DB and trajectory
        are scored again on the next call; no tool, user or policy call is made.
        At most ``1 + max_scoring_retries`` attempts, then the episode fails.
        """
        if self._closed:
            raise RuntimeError("the environment is closed")
        if self._status == "scored":
            return self._terminal_transition()
        if self._status not in ("interaction_ended", "scoring_retry_pending"):
            raise RuntimeError(f"cannot score an episode in status {self._status!r}")
        if self._environment.get_db_hash() != self._final_db_hash:
            raise RuntimeError("the final DB changed after the interaction ended")

        attempt = len(self._scoring_errors) + 1
        try:
            score = score_final_db(self.task, self._environment, text_match=self.config.text_match)
        except ItsmInfrastructureError as error:
            self._scoring_errors.append(
                {"attempt": attempt, "stage": error.stage, "kind": error.kind, "detail": error.detail}
            )
            if attempt > self.config.max_scoring_retries:
                self._fail(
                    error.stage,
                    f"{error.kind + ': ' if error.kind else ''}{error.detail}",
                    reason="scoring_attempts_exhausted",
                    attempts=attempt,
                )
            else:
                self._status = "scoring_retry_pending"
            raise
        except TaskDefinitionError as error:
            self._fail("task_definition", error)
            raise
        except Exception as error:
            self._fail("score_internal", error)
            raise

        invalid = self._termination_reason == "invalid_action"
        self._reward_info = ItsmRewardInfo(
            db_evaluated=True,
            db_match=score.db_match,
            db_reward=score.db_reward,
            db_mismatches=score.mismatches,
            text_match_requested=score.text_match_requested,
            text_match_effective=score.text_match_effective,
            text_judge_model=score.text_judge_model,
            nl_status="not_evaluated",
            nl_reward=None,
            joint_reward=None,
            # Decision D2: an invalid action ends the episode with outcome 0.
            training_reward=0.0 if invalid else score.db_reward,
            training_reward_source="invalid_action" if invalid else "db_only",
            gold_db_hash=score.gold_db_hash,
            final_db_hash=score.final_db_hash,
            scoring_attempts=attempt,
        )
        self._status = "scored"
        return self._terminal_transition()

    def close(self) -> None:
        """Idempotent; drops the DB and the user model. ``export_trace`` still works."""
        self._closed = True
        self._environment = None
        self._user = None

    @property
    def status(self) -> EpisodeStatus:
        return self._status

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def done(self) -> bool:
        """The business interaction has ended (scored or not)."""
        return self._termination_reason is not None

    @property
    def trajectory(self) -> list[Message]:
        """Every message in Bench2 form, in ``Orchestrator.run`` order."""
        return list(self._messages)

    @property
    def tool_records(self) -> tuple[ItsmToolCallRecord, ...]:
        return tuple(self._tool_records)

    @property
    def failure(self) -> dict[str, Any] | None:
        """Failure detail (``stage``, ``detail``; ``call_id``/``reason``/``attempts`` if any)."""
        return dict(self._failure) if self._failure is not None else None

    @property
    def scoring_errors(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(error) for error in self._scoring_errors)

    def current_db_hash(self) -> str:
        if self._closed:
            raise RuntimeError("the environment is closed")
        if self._environment is None:
            raise RuntimeError("reset() must be called first")
        return self._environment.get_db_hash()

    def export_trace(self) -> dict[str, Any]:
        """JSON-serializable structured trajectory."""
        return {
            "episode_id": self.episode_id,
            "task_id": self.task.id,
            "status": self._status,
            "closed": self._closed,
            "step_count": self._step_count,
            "agent_turns": self._agent_turns,
            "termination_reason": self._termination_reason,
            "tool_calls": [asdict(record) for record in self._tool_records],
            "messages": [message.model_dump(mode="json") for message in self._messages],
            "final_db_hash": self._final_db_hash,
            "reward_info": asdict(self._reward_info) if self._reward_info else None,
            "scoring_errors": [dict(error) for error in self._scoring_errors],
            "failure": dict(self._failure) if self._failure is not None else None,
        }

    # -- steps ---------------------------------------------------------------

    def _new_tool_call(self, action: ItsmAction) -> tuple[str, dict[str, Any]]:
        self._call_seq += 1
        call_id = public_call_id(self._call_seq)
        arguments = copy.deepcopy(action.arguments)
        self._messages.append(
            AssistantMessage(tool_calls=[ToolCall(id=call_id, name=action.name, arguments=arguments)])
        )
        return call_id, arguments

    def _unanswered_record(
        self, call_id: str, action: ItsmAction, arguments: dict[str, Any], duration_ms: float
    ) -> ItsmToolCallRecord:
        record = ItsmToolCallRecord(
            call_id=call_id,
            name=action.name,
            arguments=copy.deepcopy(arguments),
            model_call_id=action.model_call_id,
            result_received=False,
            result=None,
            error=None,
            error_class=None,
            error_code=None,
            error_field=None,
            duration_ms=duration_ms,
        )
        self._tool_records.append(record)
        return record

    def _step_tool_call(self, action: ItsmAction) -> ItsmTransition:
        call_id, arguments = self._new_tool_call(action)

        started = time.perf_counter()
        with task_clock(self.task):
            try:
                content, error_class, error_code, error_field = self._execute(
                    action.name, arguments
                )
            except Exception as error:
                # Arguments already passed validation, so the tool itself is broken. No
                # result reached the model; the fault detail lives in ``failure``.
                self._unanswered_record(
                    call_id, action, arguments, (time.perf_counter() - started) * 1000.0
                )
                raise self._infrastructure("tool_internal", error, call_id=call_id) from error
        duration_ms = (time.perf_counter() - started) * 1000.0

        self._messages.append(
            ToolMessage(id=call_id, content=content, requestor="assistant", error=error_class is not None)
        )
        self._step_count += 1
        error = content.removeprefix(ERROR_PREFIX) if error_class is not None else None
        record = ItsmToolCallRecord(
            call_id=call_id,
            name=action.name,
            arguments=copy.deepcopy(arguments),
            model_call_id=action.model_call_id,
            result_received=True,
            result=content if error_class is None else None,
            error=error,
            error_class=error_class,
            error_code=error_code,
            error_field=error_field,
            duration_ms=duration_ms,
        )
        self._tool_records.append(record)

        if error_class is None:
            observation = f"tool[{call_id}] ok: {'null' if content is None else content}"
        else:
            observation = f"tool[{call_id}] error({error_class}): {error}"
        return self._transition(observation, record)

    def _step_message(self, action: ItsmAction) -> ItsmTransition:
        agent_message = AssistantMessage(content=action.text)
        self._messages.append(agent_message)
        user_message = self._user_turn(agent_message)
        if STOP in user_message.content:
            return self._end_interaction("user_stop")
        return self._transition(f"user: {user_message.content}", None)

    def _end_on_step_budget(self, action: ItsmAction) -> ItsmTransition:
        """Orchestrator records the agent's message but neither executes the call
        nor asks the user once ``steps >= max_steps``."""
        if action.kind == "tool_call":
            call_id, arguments = self._new_tool_call(action)
            record = self._unanswered_record(call_id, action, arguments, 0.0)
            return self._end_interaction("max_steps", tool_record=record)
        self._messages.append(AssistantMessage(content=action.text))
        return self._end_interaction("max_steps")

    def _end_interaction(
        self,
        reason: TerminationReason,
        final_message: AssistantMessage | None = None,
        *,
        tool_record: ItsmToolCallRecord | None = None,
    ) -> ItsmTransition:
        if final_message is not None:
            self._messages.append(final_message)
        self._termination_reason = reason
        self._end_tool_record = tool_record
        self._final_db_hash = self._environment.get_db_hash()
        self._status = "interaction_ended"
        if not self.config.score_on_finish:
            return self._terminal_transition()
        return self.score()

    def _transition(self, observation: str, record: ItsmToolCallRecord | None) -> ItsmTransition:
        return ItsmTransition(
            observation, False, False, None, record, None, self._step_count, self._agent_turns
        )

    def _terminal_transition(self) -> ItsmTransition:
        reason = self._termination_reason
        return ItsmTransition(
            "",
            reason in TERMINATED_REASONS,
            reason in TRUNCATED_REASONS,
            reason,
            self._end_tool_record,
            self._reward_info,
            self._step_count,
            self._agent_turns,
        )

    # -- helpers -------------------------------------------------------------

    def _execute(
        self, name: str, arguments: dict[str, Any]
    ) -> tuple[str | None, ErrorClass | None, str | None, str | None]:
        """Run one tool call; return ``(content, error_class, error_code, error_field)``.

        Model input errors (unknown tool, arguments that fail the signature or the
        strict type check, ``ItsmError``) are returned. Any other exception
        propagates: the caller treats it as an infrastructure failure. Content
        strings follow Bench2's ``Environment.get_response`` except for
        ``bad_arguments``, whose Bench2 text leaks the Python class name.
        """
        tools = self._environment.tools
        if not tools.has_tool(name):
            return f"{ERROR_PREFIX}Tool {name!r} not found.", "unknown_tool", None, None
        tool = tools.tools[name]
        try:
            inspect.signature(tool).bind(**arguments)
        except TypeError as error:
            problem: str | None = str(error)
        else:
            problem = argument_type_error(tool, arguments)
        if problem is not None:
            return (
                f"{ERROR_PREFIX}Invalid arguments for tool {name!r}: {problem}",
                "bad_arguments",
                None,
                None,
            )
        try:
            result = self._environment.make_tool_call(name, **arguments)
        except ItsmError as error:
            return f"{ERROR_PREFIX}{error}", "domain_error", error.code, error.field
        return _stringify(result), None, None, None

    def _user_turn(self, agent_message: AssistantMessage) -> UserMessage:
        try:
            message, self._user_state = self._user.generate_next_message(
                agent_message, self._user_state
            )
        except Exception as error:
            raise self._infrastructure("user_sim", error) from error
        if not isinstance(message, UserMessage) or not message.has_text_content():
            raise self._infrastructure("user_sim", "user simulator returned no text")
        self._messages.append(message)
        self._step_count += 1
        return message

    def _public_info(self) -> dict[str, Any]:
        """Only protocol §3 public fields: the policy and the tool schemas."""
        return {
            "policy": self._environment.get_policy(),
            "tools": self._environment.get_tool_schemas() + [copy.deepcopy(DONE_TOOL_SCHEMA)],
        }

    def _fail(self, stage: str, cause: Exception | str, **extra: Any) -> str:
        self._status = "failed"
        detail = cause if isinstance(cause, str) else f"{type(cause).__name__}: {cause}"
        self._failure = {"stage": stage, "detail": detail, **extra}
        return detail

    def _infrastructure(
        self, stage: str, cause: Exception | str, *, call_id: str | None = None
    ) -> ItsmInfrastructureError:
        extra = {"call_id": call_id} if call_id is not None else {}
        detail = self._fail(stage, cause, **extra)
        return ItsmInfrastructureError(stage, self.task.id, f"episode {self.episode_id}: {detail}")

    def _require_active(self) -> None:
        if self._closed:
            raise RuntimeError("the environment is closed")
        if self._status == "new":
            raise RuntimeError("reset() must be called before step()")
        if self._status == "failed":
            raise RuntimeError("the episode ended with an infrastructure error")
        if self._status != "active":
            raise RuntimeError("the episode has already finished")
