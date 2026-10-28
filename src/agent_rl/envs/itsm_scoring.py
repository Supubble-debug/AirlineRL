"""Deterministic ITSM DB scoring under the task clock (protocol §6 and §9 rule 1).

Reuses Bench2's environment builder, tools, record comparison and judge prompt.
Differs from Bench2's ``calculate_db_reward`` in four ways: the task clock is
set in the executing thread, a failing gold action raises instead of being
logged, the free-text comparison strategy must be explicit and may not silently
degrade, and the ``llm`` judge's raw response is validated here so a malformed
response is a judge protocol error rather than a negative verdict.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from enterprise_worlds.config import DEFAULT_LLM_NL_JUDGE_ARGS
from enterprise_worlds.data_model.message import SystemMessage, UserMessage
from enterprise_worlds.data_model.tasks import Task
from enterprise_worlds.domains.itsm import environment as itsm_environment
from enterprise_worlds.environment.db import DB
from enterprise_worlds.environment.environment import Environment
from enterprise_worlds.evaluator import evaluator_env, text_match_strategy
from enterprise_worlds.evaluator.text_match_strategy import TextMatchConfig
from enterprise_worlds.utils.clock import get_now, set_now


SEED_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_REPORTED_MISMATCHES = 20


class TaskDefinitionError(ValueError):
    """Raised when a task lacks a field the ITSM protocol requires."""


class ItsmInfrastructureError(RuntimeError):
    """Raised when the environment, clock or a judge cannot produce a valid result."""

    def __init__(self, stage: str, task_id: str, detail: str = "", kind: str = "") -> None:
        self.stage = stage
        self.task_id = task_id
        self.detail = detail
        self.kind = kind
        message = f"ITSM infrastructure failure during {stage} (task_id={task_id!r})"
        if kind:
            message = f"{message} [{kind}]"
        super().__init__(f"{message}: {detail}" if detail else message)

    def __reduce__(
        self,
    ) -> tuple[type[ItsmInfrastructureError], tuple[str, str, str, str]]:
        """Reconstruct the exception across multiprocessing and Ray boundaries."""

        return type(self), (self.stage, self.task_id, self.detail, self.kind)


class JudgeProtocolError(ValueError):
    """The judge's raw response does not satisfy the verdict protocol."""

    def __init__(self, kind: str, detail: str) -> None:
        self.kind = kind
        self.detail = detail
        super().__init__(f"{kind}: {detail}")


@dataclass(frozen=True, slots=True)
class TextJudgeVerdict:
    key: str
    field: str
    equivalent: bool
    reasoning: str | None


@dataclass(frozen=True, slots=True)
class ItsmDbScore:
    db_match: bool
    db_reward: float
    mismatches: tuple[str, ...]
    text_match_requested: str
    text_match_effective: str
    text_judge_model: str | None
    text_judgments: tuple[TextJudgeVerdict, ...]
    gold_db_hash: str
    final_db_hash: str


def load_itsm_tasks() -> dict[str, Task]:
    return {task.id: task for task in itsm_environment.get_tasks()}


def canonical_task_time(task: Task) -> str:
    if not task.current_time:
        raise TaskDefinitionError(f"{task.id}: current_time is required")
    return datetime.fromisoformat(task.current_time).strftime(SEED_TIME_FORMAT)


@contextmanager
def task_clock(task: Task) -> Iterator[str]:
    """Freeze the Bench2 clock to the task time in the current thread.

    The clock is thread-local, so callers must enter this in the thread that
    executes tools or scoring. The previous value is restored on exit.
    """
    expected = canonical_task_time(task)
    previous = get_now()
    set_now(task.current_time)
    try:
        if get_now() != expected:
            raise ItsmInfrastructureError(
                "clock", task.id, f"expected {expected!r}, got {get_now()!r}"
            )
        yield expected
    finally:
        set_now(previous)


def build_task_environment(task: Task) -> Environment:
    """Build a fresh seed + delta environment with its own in-memory DB."""
    if not task.seed_db:
        raise TaskDefinitionError(f"{task.id}: seed_db is required")
    return itsm_environment.get_environment(
        db_delta=task.initial_state_delta,
        acting_user_id=task.acting_user_id,
        org_id=task.org_id,
        org_ids=task.org_ids,
        seed_db=task.seed_db,
    )


def environment_constructor(task: Task) -> Callable[..., Environment]:
    """Bench2-compatible ``environment_constructor(db_delta=...)`` for one task."""
    if not task.seed_db:
        raise TaskDefinitionError(f"{task.id}: seed_db is required")

    def construct(db_delta=None) -> Environment:
        return itsm_environment.get_environment(
            db_delta=db_delta,
            acting_user_id=task.acting_user_id,
            org_id=task.org_id,
            org_ids=task.org_ids,
            seed_db=task.seed_db,
        )

    return construct


def replay_gold_actions(task: Task) -> Environment:
    """Replay every gold action on a fresh environment; any failure is infrastructure."""
    if not task.evaluation_criteria.actions:
        raise TaskDefinitionError(f"{task.id}: gold actions are required for DB scoring")
    with task_clock(task):
        environment = build_task_environment(task)
        for index, action in enumerate(task.evaluation_criteria.actions):
            try:
                environment.make_tool_call(action.name, **action.arguments)
            except Exception as error:
                raise ItsmInfrastructureError(
                    "gold_replay",
                    task.id,
                    f"action {index} {action.name}: {type(error).__name__}: {error}",
                ) from error
    return environment


def validate_text_match(text_match: TextMatchConfig) -> str:
    """Return the strategy, refusing configs whose effective strategy differs."""
    effective = text_match.effective_strategy()
    if effective != text_match.strategy:
        raise ValueError(
            f"text-match strategy {text_match.strategy!r} would degrade to "
            f"{effective!r}; configure it explicitly"
        )
    return effective


def collect_db_differences(
    gold_db: DB,
    pred_db: DB,
    baseline_db: DB,
    text_match: TextMatchConfig,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Bench2's ``compare_dbs`` walk, stopping before the free-text judge call.

    Returns ``(mismatches, pending)``; ``pending`` is non-empty only under the
    ``llm`` strategy. Record comparison is Bench2's own ``_compare_record``.
    """
    strategy = validate_text_match(text_match)
    freetext = gold_db.freetext_fields()
    mismatches: list[str] = []
    pending: list[dict[str, Any]] = []
    for collection in type(gold_db).model_fields:
        gold_records = getattr(gold_db, collection)
        pred_records = getattr(pred_db, collection)
        if not isinstance(gold_records, dict):
            if gold_records != pred_records:
                mismatches.append(f"{collection}: {gold_records!r} != {pred_records!r}")
            continue
        gold_ids, pred_ids = set(gold_records), set(pred_records)
        if gold_ids != pred_ids:
            if missing := sorted(gold_ids - pred_ids):
                mismatches.append(f"{collection}: missing {missing}")
            if extra := sorted(pred_ids - gold_ids):
                mismatches.append(f"{collection}: unexpected {extra}")
            continue
        baseline_records = getattr(baseline_db, collection, {})
        for record_id in sorted(gold_ids):
            baseline = (
                evaluator_env._dump(baseline_records[record_id])
                if record_id in baseline_records
                else {}
            )
            evaluator_env._compare_record(
                f"{collection}/{record_id}",
                evaluator_env._dump(gold_records[record_id]),
                evaluator_env._dump(pred_records[record_id]),
                freetext.get(collection, []),
                baseline,
                text_match,
                strategy,
                mismatches,
                pending,
            )
    return mismatches, pending


def build_text_judge_request(
    pairs: list[dict[str, Any]], text_match: TextMatchConfig
) -> dict[str, Any]:
    """The exact ``generate`` kwargs Bench2's ``judge_free_text`` would send."""
    payload = [
        {"index": index, "field": pair["field"], "gold": pair["gold"], "pred": pair["pred"]}
        for index, pair in enumerate(pairs)
    ]
    llm_args = (
        text_match.llm_args
        if text_match.llm_args is not None
        else dict(DEFAULT_LLM_NL_JUDGE_ARGS)
    )
    return {
        "model": text_match.llm,
        "messages": [
            SystemMessage(content=text_match_strategy.JUDGE_SYSTEM_PROMPT),
            UserMessage(content="pairs:\n" + json.dumps(payload, indent=2, default=str)),
        ],
        **llm_args,
    }


def parse_text_judge_response(
    content: str | None, pair_count: int
) -> list[tuple[bool, str | None]]:
    """Validate a raw judge response; return ``(equivalent, reasoning)`` per index.

    A result item is a valid verdict when its ``index`` is an in-range integer
    seen once and ``equivalent`` is a boolean. ``reasoning`` is optional and
    never affects validity. Anything else raises :class:`JudgeProtocolError`.
    """
    try:
        data = json.loads(text_match_strategy._extract_json(content or ""))
    except json.JSONDecodeError as error:
        raise JudgeProtocolError("unparseable", str(error)) from error
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise JudgeProtocolError("unparseable", "response has no 'results' list")

    verdicts: dict[int, tuple[bool, str | None]] = {}
    for position, item in enumerate(data["results"]):
        if not isinstance(item, dict):
            raise JudgeProtocolError("invalid_item", f"result {position} is not an object")
        index = item.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise JudgeProtocolError("invalid_index", f"result {position} index={index!r}")
        if not 0 <= index < pair_count:
            raise JudgeProtocolError("invalid_index", f"index {index} outside 0..{pair_count - 1}")
        if index in verdicts:
            raise JudgeProtocolError("duplicate_index", f"index {index} appears twice")
        equivalent = item.get("equivalent")
        if not isinstance(equivalent, bool):
            raise JudgeProtocolError(
                "invalid_equivalent", f"index {index} equivalent={equivalent!r}"
            )
        reasoning = item.get("reasoning")
        verdicts[index] = (equivalent, reasoning if isinstance(reasoning, str) else None)

    if missing := sorted(set(range(pair_count)) - set(verdicts)):
        raise JudgeProtocolError("missing_index", f"no verdict for indices {missing}")
    return [verdicts[index] for index in range(pair_count)]


def judge_text_pairs(
    task: Task, pairs: list[dict[str, Any]], text_match: TextMatchConfig
) -> list[TextJudgeVerdict]:
    """One judge call for every pending pair, classified at the response boundary."""
    try:
        response = text_match_strategy.generate(**build_text_judge_request(pairs, text_match))
    except Exception as error:
        raise ItsmInfrastructureError(
            "text_judge_call", task.id, f"{type(error).__name__}: {error}"
        ) from error
    try:
        parsed = parse_text_judge_response(response.content, len(pairs))
    except JudgeProtocolError as error:
        raise ItsmInfrastructureError(
            "text_judge_protocol", task.id, error.detail, kind=error.kind
        ) from error
    return [
        TextJudgeVerdict(
            key=pair["key"], field=pair["field"], equivalent=equivalent, reasoning=reasoning
        )
        for pair, (equivalent, reasoning) in zip(pairs, parsed)
    ]


def score_final_db(
    task: Task,
    final_environment: Environment,
    *,
    text_match: TextMatchConfig,
) -> ItsmDbScore:
    """Compare the episode's final DB with the gold DB under the task clock."""
    strategy = validate_text_match(text_match)
    gold_environment = replay_gold_actions(task)
    with task_clock(task):
        baseline_db = build_task_environment(task).tools.db
        mismatches, pending = collect_db_differences(
            gold_environment.tools.db,
            final_environment.tools.db,
            baseline_db,
            text_match,
        )
        verdicts = judge_text_pairs(task, pending, text_match) if pending else []
    for pair, verdict in zip(pending, verdicts):
        if not verdict.equivalent:
            mismatches.append(f"{pair['key']}: {pair['gold']!r} !≈ {pair['pred']!r} (judge)")

    matched = not mismatches
    return ItsmDbScore(
        db_match=matched,
        db_reward=1.0 if matched else 0.0,
        mismatches=tuple(mismatches[:MAX_REPORTED_MISMATCHES]),
        text_match_requested=text_match.strategy,
        text_match_effective=strategy,
        text_judge_model=text_match.llm if strategy == "llm" else None,
        text_judgments=tuple(verdicts),
        gold_db_hash=gold_environment.get_db_hash(),
        final_db_hash=final_environment.get_db_hash(),
    )
