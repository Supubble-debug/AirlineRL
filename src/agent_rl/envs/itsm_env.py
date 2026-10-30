"""Single-episode ITSM environment adapter (protocol §2–§5, §8, §9; step 4).

One ``ItsmEnv`` drives one episode synchronously, with no background thread:
``step`` executes the policy's tool call, or obtains the next user turn, in the
calling thread under the task clock, and returns only what is new. Business
tools, the user simulator and DB scoring are Bench2's own; the message order
matches Bench2's ``Orchestrator.run``.

Not handled here (step 5): step and token truncation, timeouts, cancellation,
retries and process isolation.
"""

from __future__ import annotations

import copy
import inspect
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, Literal, Protocol

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
ERROR_PREFIX = "Error: "

ActionKind = Literal["message", "tool_call", "done"]
ErrorClass = Literal["unknown_tool", "bad_arguments", "domain_error", "tool_exception_other"]
TerminationReason = Literal["agent_done", "user_stop"]


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
    task_id: str
    text_match: TextMatchConfig
    user_llm: str = DEFAULT_USER_LLM
    user_llm_args: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_USER_LLM_ARGS))

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must not be empty")
        if not self.user_llm.strip():
            raise ValueError("user_llm must not be empty")
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


@lru_cache(maxsize=1)
def _tasks_by_id() -> dict[str, Task]:
    return load_itsm_tasks()


class ItsmEnv:
    """One ITSM episode: ``reset`` once, ``step`` until terminated, then ``close``."""

    def __init__(
        self,
        config: ItsmEnvConfig,
        *,
        user: UserModel | None = None,
        episode_id: str | None = None,
    ) -> None:
        tasks = _tasks_by_id()
        if config.task_id not in tasks:
            raise ValueError(f"unknown ITSM task_id {config.task_id!r}")
        self.config = config
        self.task = tasks[config.task_id]
        self.episode_id = episode_id or f"{self.task.id}:{uuid.uuid4().hex[:8]}"
        self._user = user if user is not None else UserSimulator(
            self.task.scenario, llm=config.user_llm, llm_args=dict(config.user_llm_args)
        )
        self._environment: Environment | None = None
        self._user_state: Any = None
        self._messages: list[Message] = []
        self._tool_records: list[ItsmToolCallRecord] = []
        self._call_seq = 0
        self._step_count = 0
        self._has_reset = False
        self._done = False
        self._failed = False
        self._closed = False
        self._termination_reason: TerminationReason | None = None
        self._reward_info: ItsmRewardInfo | None = None

    # -- public API ----------------------------------------------------------

    def reset(self) -> ItsmReset:
        if self._closed:
            raise RuntimeError("the environment is closed")
        if self._has_reset:
            raise RuntimeError("ItsmEnv runs exactly one episode; create a new instance")
        self._has_reset = True
        try:
            self._environment = build_task_environment(self.task)
            self._user_state = self._user.get_init_state()
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
        if action.kind == "tool_call":
            return self._step_tool_call(action)
        if action.kind == "message":
            return self._step_message(action)
        return self._finish("agent_done", AssistantMessage(content=STOP))

    def close(self) -> None:
        self._closed = True

    @property
    def done(self) -> bool:
        return self._done

    @property
    def trajectory(self) -> list[Message]:
        """Every message in Bench2 form, in ``Orchestrator.run`` order."""
        return list(self._messages)

    @property
    def tool_records(self) -> tuple[ItsmToolCallRecord, ...]:
        return tuple(self._tool_records)

    def current_db_hash(self) -> str:
        if self._environment is None:
            raise RuntimeError("reset() must be called first")
        return self._environment.get_db_hash()

    def export_trace(self) -> dict[str, Any]:
        """JSON-serializable structured trajectory."""
        return {
            "episode_id": self.episode_id,
            "task_id": self.task.id,
            "step_count": self._step_count,
            "termination_reason": self._termination_reason,
            "tool_calls": [asdict(record) for record in self._tool_records],
            "messages": [message.model_dump(mode="json") for message in self._messages],
            "reward_info": asdict(self._reward_info) if self._reward_info else None,
        }

    # -- steps ---------------------------------------------------------------

    def _step_tool_call(self, action: ItsmAction) -> ItsmTransition:
        self._call_seq += 1
        call_id = f"{self.episode_id}/c{self._call_seq:03d}"
        arguments = copy.deepcopy(action.arguments)
        self._messages.append(
            AssistantMessage(tool_calls=[ToolCall(id=call_id, name=action.name, arguments=arguments)])
        )

        started = time.perf_counter()
        with task_clock(self.task):
            content, error_class, error_code, error_field = self._execute(action.name, arguments)
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
        return ItsmTransition(observation, False, False, None, record, None, self._step_count)

    def _step_message(self, action: ItsmAction) -> ItsmTransition:
        agent_message = AssistantMessage(content=action.text)
        self._messages.append(agent_message)
        user_message = self._user_turn(agent_message)
        if STOP in user_message.content:
            return self._finish("user_stop", None)
        return ItsmTransition(
            f"user: {user_message.content}", False, False, None, None, None, self._step_count
        )

    def _finish(
        self, reason: TerminationReason, final_message: AssistantMessage | None
    ) -> ItsmTransition:
        if final_message is not None:
            self._messages.append(final_message)
        self._done = True
        self._termination_reason = reason
        score = score_final_db(self.task, self._environment, text_match=self.config.text_match)
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
            training_reward=score.db_reward,
            training_reward_source="db_only",
            gold_db_hash=score.gold_db_hash,
            final_db_hash=score.final_db_hash,
        )
        return ItsmTransition("", True, False, reason, None, self._reward_info, self._step_count)

    # -- helpers -------------------------------------------------------------

    def _execute(
        self, name: str, arguments: dict[str, Any]
    ) -> tuple[str | None, ErrorClass | None, str | None, str | None]:
        """Run one tool call; return ``(content, error_class, error_code, error_field)``.

        Content strings follow Bench2's ``Environment.get_response`` except for
        ``bad_arguments``, whose Bench2 text leaks the Python class name.
        """
        tools = self._environment.tools
        if not tools.has_tool(name):
            return f"{ERROR_PREFIX}Tool {name!r} not found.", "unknown_tool", None, None
        try:
            inspect.signature(tools.tools[name]).bind(**arguments)
        except TypeError as error:
            return (
                f"{ERROR_PREFIX}Invalid arguments for tool {name!r}: {error}",
                "bad_arguments",
                None,
                None,
            )
        try:
            result = self._environment.make_tool_call(name, **arguments)
        except ItsmError as error:
            return f"{ERROR_PREFIX}{error}", "domain_error", error.code, error.field
        except Exception as error:
            return f"{ERROR_PREFIX}{error}", "tool_exception_other", None, None
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

    def _infrastructure(self, stage: str, cause: Exception | str) -> ItsmInfrastructureError:
        self._failed = True
        detail = cause if isinstance(cause, str) else f"{type(cause).__name__}: {cause}"
        return ItsmInfrastructureError(stage, self.task.id, f"episode {self.episode_id}: {detail}")

    def _require_active(self) -> None:
        if self._closed:
            raise RuntimeError("the environment is closed")
        if not self._has_reset:
            raise RuntimeError("reset() must be called before step()")
        if self._failed:
            raise RuntimeError("the episode ended with an infrastructure error")
        if self._done:
            raise RuntimeError("the episode has already finished")
