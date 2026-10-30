"""Offline ITSM environment and DB-scoring regression (step 3).

Uses the real Bench2 tools, seeds and tasks. No LLM is called: DB scoring uses
``exact`` or ``fuzzy``; the ``llm`` judge tests replace ``generate`` with a mock
and are labelled as such. Gold replay proves environment/scorer consistency,
not model capability.
"""

from __future__ import annotations

import collections
import copy
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import pytest

from enterprise_worlds.data_model.message import AssistantMessage, ToolCall
from enterprise_worlds.domains.itsm import environment as itsm_environment
from enterprise_worlds.domains.itsm.data_model import Problem
from enterprise_worlds.evaluator import evaluator_nl, text_match_strategy
from enterprise_worlds.evaluator.evaluator_env import calculate_db_reward, compare_dbs
from enterprise_worlds.evaluator.text_match_strategy import TextMatchConfig, judge_free_text
from enterprise_worlds.utils.clock import DEFAULT_NOW, get_now, reset_now, set_now

from agent_rl.envs.itsm_scoring import (
    ItsmInfrastructureError,
    JudgeProtocolError,
    TaskDefinitionError,
    build_task_environment,
    build_text_judge_request,
    canonical_task_time,
    collect_db_differences,
    environment_constructor,
    load_itsm_tasks,
    parse_text_judge_response,
    replay_gold_actions,
    score_final_db,
    task_clock,
)


TASKS = load_itsm_tasks()
TASK_IDS = sorted(TASKS)
EXACT = TextMatchConfig(strategy="exact")
FUZZY = TextMatchConfig(strategy="fuzzy")
MSP_021 = "itsmbench_msp_021"
SINGLE_017 = "itsmbench_single_017"
STAMP_FIELDS = ("created_on", "updated_on")


def _gold_actions(task_id: str) -> list[dict[str, Any]]:
    return [
        {"name": action.name, "arguments": copy.deepcopy(action.arguments)}
        for action in TASKS[task_id].evaluation_criteria.actions
    ]


def _edit_actions(task_id: str, **edits: Any) -> list[dict[str, Any]]:
    actions = _gold_actions(task_id)
    for index, arguments in edits.get("set", {}).items():
        actions[index]["arguments"].update(arguments)
    for index in sorted(edits.get("drop", ()), reverse=True):
        del actions[index]
    return actions + edits.get("extra", [])


def _run_actions(task_id: str, actions: list[dict[str, Any]], *, clock_task=None):
    """Execute candidate actions on a fresh env; every action must succeed."""
    task = TASKS[task_id]
    with task_clock(clock_task or task):
        environment = build_task_environment(task)
        for action in actions:
            environment.make_tool_call(action["name"], **action["arguments"])
    return environment


def _score(task_id: str, actions: list[dict[str, Any]], text_match=FUZZY, **kwargs):
    environment = _run_actions(task_id, actions, **kwargs)
    return score_final_db(TASKS[task_id], environment, text_match=text_match)


def _baseline_db(task_id: str):
    task = TASKS[task_id]
    with task_clock(task):
        return build_task_environment(task).tools.db


def _changed_records(task):
    """(collection, record_id, gold, baseline) for every record the gold changed."""
    gold_db = replay_gold_actions(task).tools.db
    baseline_db = _baseline_db(task.id)
    for collection in type(gold_db).model_fields:
        gold_records = getattr(gold_db, collection)
        if not isinstance(gold_records, dict):
            continue
        baseline_records = getattr(baseline_db, collection)
        for record_id, record in gold_records.items():
            gold = record.model_dump()
            baseline = (
                baseline_records[record_id].model_dump()
                if record_id in baseline_records
                else None
            )
            if gold != baseline:
                yield collection, record_id, gold, baseline


# -- task suite ---------------------------------------------------------------------------------


def test_task_suite_matches_audited_inventory() -> None:
    assert len(TASKS) == 53
    # ``Task`` drops the ``tenancy`` field, so read it from the raw task file.
    raw_tasks = json.loads(itsm_environment.ITSM_TASKS_PATH.read_text(encoding="utf-8"))
    tenancy = collections.Counter(task["tenancy"]["model"] for task in raw_tasks)
    assert tenancy == {"provider_client": 31, "single_tenant": 22}
    assert {MSP_021, SINGLE_017} <= set(TASKS)


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_task_has_scoring_prerequisites(task_id: str) -> None:
    task = TASKS[task_id]
    assert task.current_time
    assert task.seed_db in {"msp_db.json", "single_tenant_db.json"}
    assert task.evaluation_criteria.actions
    assert task.evaluation_criteria.nl_assertions


# -- gold replay: every task --------------------------------------------------------------------


@pytest.mark.parametrize("task_id", TASK_IDS)
@pytest.mark.parametrize("text_match", [EXACT, FUZZY], ids=["exact", "fuzzy"])
def test_gold_replay_scores_db_match(task_id: str, text_match: TextMatchConfig) -> None:
    task = TASKS[task_id]

    score = score_final_db(task, replay_gold_actions(task), text_match=text_match)

    assert score.db_match, score.mismatches
    assert score.db_reward == 1.0
    assert score.text_match_requested == score.text_match_effective == text_match.strategy
    assert score.gold_db_hash == score.final_db_hash


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_gold_replay_is_repeatable(task_id: str) -> None:
    task = TASKS[task_id]

    hashes = {replay_gold_actions(task).get_db_hash() for _ in range(3)}
    scores = {score_final_db(task, replay_gold_actions(task), text_match=FUZZY) for _ in range(3)}

    assert len(hashes) == 1
    assert len(scores) == 1


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_strict_path_agrees_with_bench2_calculate_db_reward(task_id: str) -> None:
    task = TASKS[task_id]
    gold_calls = [
        ToolCall(name=action.name, arguments=action.arguments)
        for action in task.evaluation_criteria.actions
    ]

    with task_clock(task):
        bench2 = calculate_db_reward(
            environment_constructor(task), task, agent_tool_calls=gold_calls, text_match=FUZZY
        )
    ours = score_final_db(task, replay_gold_actions(task), text_match=FUZZY)

    assert bench2.reward == ours.db_reward == 1.0


# -- task clock ---------------------------------------------------------------------------------


def test_written_timestamps_follow_the_task_clock() -> None:
    checked = 0
    for task in TASKS.values():
        expected = canonical_task_time(task)
        for collection, record_id, gold, baseline in _changed_records(task):
            for field in STAMP_FIELDS:
                if field not in gold:
                    continue
                if baseline is not None and gold[field] == baseline[field]:
                    continue
                assert gold[field] == expected, (task.id, collection, record_id, field)
                checked += 1
    assert checked > 0


def test_wrong_clock_changes_db_for_every_task_not_at_default_time() -> None:
    default_clock_tasks = []
    sensitive = []
    for task in TASKS.values():
        if canonical_task_time(task) == DEFAULT_NOW:
            default_clock_tasks.append(task.id)
            continue
        actions = _gold_actions(task.id)
        wrong_clock = task.model_copy(update={"current_time": DEFAULT_NOW})
        score = _score(task.id, actions, text_match=EXACT, clock_task=wrong_clock)
        assert not score.db_match, task.id
        sensitive.append(task.id)

    assert len(default_clock_tasks) == 19
    assert len(sensitive) == 34
    assert {MSP_021, SINGLE_017} <= set(sensitive)


def test_task_clock_restores_the_previous_thread_value() -> None:
    set_now("2030-01-01T00:00:00")
    try:
        score_final_db(TASKS[MSP_021], replay_gold_actions(TASKS[MSP_021]), text_match=FUZZY)
        assert get_now() == "2030-01-01 00:00:00"
    finally:
        reset_now()


def test_missing_current_time_is_a_task_definition_error() -> None:
    task = TASKS[MSP_021].model_copy(update={"current_time": None})

    with pytest.raises(TaskDefinitionError):
        replay_gold_actions(task)


# -- gold replay failures and text-match configuration ------------------------------------------


def test_failing_gold_action_is_infrastructure_not_reward() -> None:
    task = TASKS[MSP_021].model_copy(deep=True)
    task.evaluation_criteria.actions[1].arguments["incident_id"] = "INC_999999"

    with pytest.raises(ItsmInfrastructureError) as raised:
        replay_gold_actions(task)

    assert raised.value.stage == "gold_replay"
    assert "action 1 update_incident" in raised.value.detail


def test_bench2_legacy_path_scores_the_same_gold_failure_as_zero() -> None:
    """Documents protocol F8: Bench2 logs the failure and scores instead of raising."""
    task = TASKS[MSP_021].model_copy(deep=True)
    task.evaluation_criteria.actions[1].arguments["incident_id"] = "INC_999999"

    with task_clock(task):
        result = calculate_db_reward(
            environment_constructor(task), task, agent_tool_calls=[], text_match=FUZZY
        )

    assert result.reward == 0.0


def _fail_get_environment_on_call(monkeypatch, failing_call: int) -> None:
    original = itsm_environment.get_environment
    calls = 0

    def get_environment(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == failing_call:
            raise OSError("seed file unreadable")
        return original(*args, **kwargs)

    monkeypatch.setattr(itsm_environment, "get_environment", get_environment)


@pytest.mark.parametrize(("failing_call", "role"), [(1, "gold"), (2, "baseline")])
def test_scoring_env_build_failure_is_infrastructure(monkeypatch, failing_call: int, role: str) -> None:
    """Fault injection at the two env builds inside score_final_db (gold, then baseline)."""
    task = TASKS[MSP_021]
    final_environment = replay_gold_actions(task)
    _fail_get_environment_on_call(monkeypatch, failing_call)

    with pytest.raises(ItsmInfrastructureError) as raised:
        score_final_db(task, final_environment, text_match=FUZZY)

    assert (raised.value.stage, raised.value.kind) == ("score_env_build", role)
    assert isinstance(raised.value.__cause__, OSError)
    assert str(raised.value.__cause__) == "seed file unreadable"


def test_scoring_task_definition_error_is_not_wrapped() -> None:
    task = TASKS[MSP_021]
    final_environment = replay_gold_actions(task)

    with pytest.raises(TaskDefinitionError) as raised:
        score_final_db(task.model_copy(update={"seed_db": None}), final_environment, text_match=FUZZY)

    assert not isinstance(raised.value, ItsmInfrastructureError)


def test_infrastructure_error_round_trips_through_pickle() -> None:
    original = ItsmInfrastructureError("text_judge_protocol", MSP_021, "detail", kind="missing_index")

    restored = pickle.loads(pickle.dumps(original))

    assert (restored.stage, restored.task_id, restored.detail, restored.kind) == (
        "text_judge_protocol",
        MSP_021,
        "detail",
        "missing_index",
    )


@pytest.mark.parametrize(
    "text_match",
    [TextMatchConfig(), TextMatchConfig(strategy="llm")],
    ids=["implicit-default", "llm-without-judge"],
)
def test_llm_strategy_without_judge_is_rejected(text_match: TextMatchConfig) -> None:
    task = TASKS[MSP_021]

    with pytest.raises(ValueError, match="degrade"):
        score_final_db(task, replay_gold_actions(task), text_match=text_match)


# -- key cases: MSP-021 -------------------------------------------------------------------------


MSP_021_PARAPHRASE = (
    "Hi Sara, about INC-000054 (backup failure): to keep troubleshooting we need the backup "
    "job name, the error message and when the failure happened. The incident is on hold "
    "until we hear back. Thanks."
)
MSP_021_NOTIFICATION = "NOTIF_030"

MSP_021_POSITIVE = {
    "gold": _edit_actions(MSP_021),
    "extra_read_calls": _edit_actions(
        MSP_021,
        extra=[
            {"name": "find_incident_by_number", "arguments": {"number": "INC-000054"}},
            {"name": "find_incident_slas", "arguments": {"incident_id": "INC_054"}},
        ],
    ),
    "paraphrased_message_fuzzy": _edit_actions(MSP_021, set={5: {"message": MSP_021_PARAPHRASE}}),
}

# case -> (valid but wrong actions, mismatch the scorer must report)
MSP_021_NEGATIVE = {
    "sla_not_paused": (
        _edit_actions(MSP_021, drop=[3]),
        "incident_sla/TSLA_021 stage: 'paused' != 'in_progress'",
    ),
    "wrong_on_hold_reason": (
        _edit_actions(MSP_021, set={1: {"on_hold_reason": "Awaiting Change"}}),
        "incident/INC_054 on_hold_reason: 'Awaiting Caller' != 'Awaiting Change'",
    ),
    "notification_status_sent_on_create": (
        _edit_actions(MSP_021, set={5: {"status": "sent"}}),
        "notification/NOTIF_030 status: 'queued' != 'sent'",
    ),
    "notification_status_updated_after_send": (
        _edit_actions(
            MSP_021,
            extra=[
                {
                    "name": "update_notification",
                    "arguments": {"notification_id": MSP_021_NOTIFICATION, "status": "failed"},
                }
            ],
        ),
        "notification/NOTIF_030 status: 'queued' != 'failed'",
    ),
    "notification_to_wrong_user": (
        _edit_actions(MSP_021, set={5: {"email": "barbara.perry5@servicenow.com"}}),
        "notification/NOTIF_030 email:",
    ),
    "wrong_notification_type": (
        _edit_actions(MSP_021, set={5: {"type": "update"}}),
        "notification/NOTIF_030 type: 'reminder' != 'update'",
    ),
    "duplicate_notification": (
        _edit_actions(MSP_021, extra=[_gold_actions(MSP_021)[5]]),
        "notification: unexpected ['NOTIF_031']",
    ),
    "unrelated_message_text": (
        _edit_actions(MSP_021, set={5: {"message": "Please call the service desk."}}),
        "notification/NOTIF_030 message:",
    ),
    "read_only_no_writes": (
        [a for a in _edit_actions(MSP_021) if a["name"].startswith(("find_", "get_"))],
        "incident/INC_054 status: 'on_hold' != 'in_progress'",
    ),
}


def test_msp_021_gold_state_matches_plan() -> None:
    db = replay_gold_actions(TASKS[MSP_021]).tools.db
    baseline = _baseline_db(MSP_021)

    assert baseline.incident_sla["TSLA_021"].stage == "in_progress"
    assert db.incident_sla["TSLA_021"].stage == "paused"
    assert db.incident["INC_054"].status == "on_hold"
    assert db.incident["INC_054"].on_hold_reason == "Awaiting Caller"
    assert MSP_021_NOTIFICATION not in baseline.notification
    notification = db.notification[MSP_021_NOTIFICATION]
    # The gold call omits ``status``; send_notification defaults it to 'queued'.
    assert "status" not in TASKS[MSP_021].evaluation_criteria.actions[5].arguments
    assert (notification.type, notification.status, notification.email) == (
        "reminder",
        "queued",
        "sara.clark93@servicenow.com",
    )


@pytest.mark.parametrize("case", sorted(MSP_021_POSITIVE))
def test_msp_021_positive_cases(case: str) -> None:
    score = _score(MSP_021, MSP_021_POSITIVE[case])

    assert score.db_match, score.mismatches


@pytest.mark.parametrize("case", sorted(MSP_021_NEGATIVE))
def test_msp_021_negative_cases(case: str) -> None:
    actions, expected_mismatch = MSP_021_NEGATIVE[case]

    score = _score(MSP_021, actions)

    assert not score.db_match
    assert score.db_reward == 0.0
    assert any(expected_mismatch in mismatch for mismatch in score.mismatches), score.mismatches


def test_msp_021_paraphrase_fails_exact_match() -> None:
    score = _score(MSP_021, MSP_021_POSITIVE["paraphrased_message_fuzzy"], text_match=EXACT)

    assert not score.db_match
    assert any(m.startswith("notification/NOTIF_030 message:") for m in score.mismatches)


# -- key cases: Single-017 ----------------------------------------------------------------------


SINGLE_017_PROBLEM = "PRB_003"
SINGLE_017_SOURCE_INCIDENT = "INC_020"
SINGLE_017_CHANGE = "CHG_003"
SINGLE_017_STAMP_FIELDS = {"created_on", "updated_on"}
SINGLE_017_EMPTY_STRUCTURED_FIELDS = {
    "service": "SVC_004",
    "service_offering": "SVCOFF_004",
    "assigned_to": "USER_008",
    "assignment_group": "GROUP_003",
    "category": "software",
}
# Free-text columns: Bench2's DB match ignores a prose field the gold left unchanged.
SINGLE_017_EMPTY_FREETEXT_FIELDS = {"workaround": "Restart the server.", "fix_notes": "Patched ACL."}

SINGLE_017_POSITIVE = {
    "gold": _edit_actions(SINGLE_017),
    "extra_read_calls": _edit_actions(
        SINGLE_017,
        extra=[{"name": "find_incident_by_number", "arguments": {"number": "INC0000007"}}],
    ),
}

SINGLE_017_NEGATIVE = {
    "change_not_mapped": (
        _edit_actions(SINGLE_017, drop=[8, 9]),
        "change_request_mapping: missing ['CRM_006']",
    ),
    "mapping_links_incident_not_problem": (
        _edit_actions(SINGLE_017, set={8: {"problem_id": None, "incident_id": "INC_020"}}, drop=[9]),
        "change_request_mapping/CRM_006 problem_id: 'PRB_003' != None",
    ),
    "change_wrong_status": (
        _edit_actions(SINGLE_017, set={7: {"status": "implement"}}),
        "change/CHG_003 status: 'authorize' != 'implement'",
    ),
    "problem_wrong_impact": (
        _edit_actions(SINGLE_017, set={3: {"impact": "high"}}),
        "problem/PRB_003 impact: 'medium' != 'high'",
    ),
    "problem_on_workstation_ci": (
        _edit_actions(SINGLE_017, set={3: {"configuration_item": "CI_003"}}),
        "problem/PRB_003 configuration_item: 'CI_004' != 'CI_003'",
    ),
    "problem_not_advanced_to_root_cause": (
        _edit_actions(SINGLE_017, drop=[5]),
        "problem/PRB_003 status: 'root_cause' != 'assess'",
    ),
    "source_incident_modified": (
        _edit_actions(
            SINGLE_017,
            extra=[
                {
                    "name": "update_incident",
                    "arguments": {"incident_id": "INC_020", "problem": "PRB_003"},
                }
            ],
        ),
        "incident/INC_020 problem: None != 'PRB_003'",
    ),
    "duplicate_problem": (
        _edit_actions(SINGLE_017, extra=[_gold_actions(SINGLE_017)[3]]),
        "problem: unexpected",
    ),
}


def _gold_written_problem_fields() -> set[str]:
    written: set[str] = set()
    for action in TASKS[SINGLE_017].evaluation_criteria.actions:
        if action.name == "create_problem" or (
            action.name == "update_problem"
            and action.arguments.get("problem_id") == SINGLE_017_PROBLEM
        ):
            written |= set(action.arguments)
    return written


def test_single_017_gold_leaves_source_incident_unchanged() -> None:
    db = replay_gold_actions(TASKS[SINGLE_017]).tools.db
    baseline = _baseline_db(SINGLE_017)

    assert (
        db.incident[SINGLE_017_SOURCE_INCIDENT].model_dump()
        == baseline.incident[SINGLE_017_SOURCE_INCIDENT].model_dump()
    )
    assert db.incident[SINGLE_017_SOURCE_INCIDENT].problem is None


def test_single_017_gold_state_leaves_optional_problem_fields_empty() -> None:
    """Gold-state check only. It says nothing about what DB scoring constrains;
    see the two scoring-scope tests below and protocol §14.4 (D8)."""
    optional = {
        name
        for name, field in Problem.model_fields.items()
        if not field.is_required() and field.default is None
    }
    expected_empty = optional - _gold_written_problem_fields() - SINGLE_017_STAMP_FIELDS
    problem = replay_gold_actions(TASKS[SINGLE_017]).tools.db.problem[SINGLE_017_PROBLEM]

    assert expected_empty == set(SINGLE_017_EMPTY_STRUCTURED_FIELDS) | set(
        SINGLE_017_EMPTY_FREETEXT_FIELDS
    )
    assert {name: getattr(problem, name) for name in expected_empty} == dict.fromkeys(
        expected_empty
    )


def test_single_017_gold_maps_exactly_change_to_problem() -> None:
    db = replay_gold_actions(TASKS[SINGLE_017]).tools.db
    baseline = _baseline_db(SINGLE_017)

    new_mappings = [
        record.model_dump()
        for record_id, record in db.change_request_mapping.items()
        if record_id not in baseline.change_request_mapping
    ]

    assert [
        (m["change_id"], m["problem_id"], m["incident_id"]) for m in new_mappings
    ] == [(SINGLE_017_CHANGE, SINGLE_017_PROBLEM, None)]


@pytest.mark.parametrize("case", sorted(SINGLE_017_POSITIVE))
def test_single_017_positive_cases(case: str) -> None:
    score = _score(SINGLE_017, SINGLE_017_POSITIVE[case])

    assert score.db_match, score.mismatches


@pytest.mark.parametrize("case", sorted(SINGLE_017_NEGATIVE))
def test_single_017_negative_cases(case: str) -> None:
    actions, expected_mismatch = SINGLE_017_NEGATIVE[case]

    score = _score(SINGLE_017, actions)

    assert not score.db_match
    assert score.db_reward == 0.0
    assert any(expected_mismatch in mismatch for mismatch in score.mismatches), score.mismatches


@pytest.mark.parametrize("field", sorted(SINGLE_017_EMPTY_STRUCTURED_FIELDS))
def test_single_017_scoring_scope_constrains_structured_optional_field(field: str) -> None:
    value = SINGLE_017_EMPTY_STRUCTURED_FIELDS[field]
    actions = _edit_actions(SINGLE_017, set={3: {field: value}})

    score = _score(SINGLE_017, actions, text_match=EXACT)

    assert not score.db_match
    assert any(m.startswith(f"problem/PRB_003 {field}: None != ") for m in score.mismatches)


MOCK_LLM_TEXT_MATCH = TextMatchConfig(strategy="llm", llm="mock/judge", llm_args={"temperature": 0.0})


@pytest.mark.parametrize("field", sorted(SINGLE_017_EMPTY_FREETEXT_FIELDS))
@pytest.mark.parametrize(
    "text_match", [EXACT, FUZZY, MOCK_LLM_TEXT_MATCH], ids=["exact", "fuzzy", "llm"]
)
def test_single_017_scoring_scope_excludes_unchanged_freetext_field(
    monkeypatch, field: str, text_match: TextMatchConfig
) -> None:
    """D8: records a Bench2 scoring rule, not a desired property. A free-text field
    the gold left unchanged is skipped before the strategy is chosen, so DB=1 does
    not guarantee it stays empty under any strategy, and llm never judges it.

    Under llm the mock judge accepts every pair it is sent (the fields the gold did
    change); the assertion is that the unchanged field is not among them."""

    def accept_every_pair(**kwargs):
        pairs = json.loads(kwargs["messages"][1].content.removeprefix("pairs:\n"))
        return AssistantMessage(
            content=_verdicts(*({"index": p["index"], "equivalent": True} for p in pairs))
        )

    monkeypatch.setattr(text_match_strategy, "generate", accept_every_pair)
    actions = _edit_actions(SINGLE_017, set={3: {field: SINGLE_017_EMPTY_FREETEXT_FIELDS[field]}})

    score = _score(SINGLE_017, actions, text_match=text_match)

    assert score.db_match
    assert all(not v.key.endswith(f" {field}") for v in score.text_judgments)
    if text_match.strategy == "llm":
        assert {v.key for v in score.text_judgments} == {
            f"problem/PRB_003 {name}"
            for name in ("problem_statement", "short_description", "worknotes")
        }


# -- pinned Bench2 sources and judge prompts (protocol §14.2, §14.3) -----------------------------


BENCH2_SOURCE_ROOT = Path(itsm_environment.__file__).resolve().parents[2]
PINNED_BENCH2_SOURCES = {
    "evaluator/evaluator_env.py": "cd8b458c8f928b443a71c8dc7dbeb03c1a3461242725ce441bfda93826c2e95b",
    "evaluator/text_match_strategy.py": "6bdd229561c4c38c39bd3276fa626667931343365fd66cf08736b53036bc4b20",
    "evaluator/evaluator_nl.py": "a7c6f23b0a0f2ad735f6e54e7cf6c65e879fc4550fece6c587ec507c93e91b51",
    "evaluator/evaluator.py": "db5f87b82a3ded5df1fbd2c90f0faf200f4ed3b0b7ada60000085dd94ac60be4",
    "utils/clock.py": "2ca1286d862c9886bace4f4ecd835a23f9d166373c177a917372e0149a203088",
    "utils/text_match.py": "d4790b0113c6ec06de0b24caa8949c3a97bf3d57df04ae88d47cb8b5b942a27c",
    "environment/environment.py": "0f39c84f0c271d88129d773d4ae54778786fdf0c647bd3128130384d9eff67a9",
    "environment/toolkit.py": "67a943876e6cd3de49c375cfcbd9db978395194f1217b2a502a96bfe50972a78",
    "domains/itsm/environment.py": "4783009792689f3dc6e7064fb55d5a87c15ffdac7737eda23ad9d46518ca6da3",
    "domains/itsm/data_model.py": "0ad66f253e4ea52cc213667f0a4d892e53ebf122aeb923d3e82ab83161910ab5",
    "data_model/message.py": "618acdbb19dcb947b2fba5e734530523c577f483e746939dbcbbfd417662040b",
    "orchestrator/orchestrator.py": "731831c31506c1a1676a8fd1db842fd0c95c5f5489cef5f3ada95d1adbf7767c",
}
PINNED_JUDGE_PROMPTS = {
    "text_match": "2e2350908ca2b54271cc96dab31c414a14413427af177a2fbcbcd1df272761d4",
    "nl": "295597be7c47347429e8adf6f236e62eae335d7ba587cdc221e866f4dac7927f",
}


def test_bench2_sources_match_pinned_checksums() -> None:
    actual = {
        path: hashlib.sha256((BENCH2_SOURCE_ROOT / path).read_bytes()).hexdigest()
        for path in PINNED_BENCH2_SOURCES
    }

    assert actual == PINNED_BENCH2_SOURCES


def test_judge_prompts_match_pinned_versions() -> None:
    prompts = {
        "text_match": text_match_strategy.JUDGE_SYSTEM_PROMPT,
        "nl": evaluator_nl.JUDGE_SYSTEM_PROMPT,
    }

    assert {
        name: hashlib.sha256(prompt.encode("utf-8")).hexdigest() for name, prompt in prompts.items()
    } == PINNED_JUDGE_PROMPTS


# -- the DB walk used for the llm path equals Bench2's compare_dbs ------------------------------


def _walk_cases():
    for task_id, cases in ((MSP_021, MSP_021_NEGATIVE), (SINGLE_017, SINGLE_017_NEGATIVE)):
        for case, (actions, _) in sorted(cases.items()):
            yield f"{task_id}:{case}", task_id, actions, None
    for task_id in TASK_IDS:
        task = TASKS[task_id]
        yield f"{task_id}:gold", task_id, _gold_actions(task_id), None
        if canonical_task_time(task) != DEFAULT_NOW:
            wrong_clock = task.model_copy(update={"current_time": DEFAULT_NOW})
            yield f"{task_id}:wrong_clock", task_id, _gold_actions(task_id), wrong_clock


@pytest.mark.parametrize("text_match", [EXACT, FUZZY], ids=["exact", "fuzzy"])
def test_db_walk_matches_bench2_compare_dbs(text_match: TextMatchConfig) -> None:
    compared = 0
    for label, task_id, actions, clock_task in _walk_cases():
        task = TASKS[task_id]
        gold_db = replay_gold_actions(task).tools.db
        pred_db = _run_actions(task_id, actions, clock_task=clock_task).tools.db
        baseline_db = _baseline_db(task_id)

        mismatches, pending = collect_db_differences(gold_db, pred_db, baseline_db, text_match)

        assert compare_dbs(gold_db, pred_db, baseline_db, text_match) == (
            not mismatches,
            mismatches,
        ), label
        assert pending == []
        compared += 1
    assert compared == 2 * len(TASK_IDS) - 19 + len(MSP_021_NEGATIVE) + len(SINGLE_017_NEGATIVE)


# -- llm judge classification (mocked generate, no network) -------------------------------------


LLM_JUDGE = TextMatchConfig(strategy="llm", llm="mock/judge", llm_args={"temperature": 0.0})


def _verdicts(*items: dict[str, Any]) -> str:
    return json.dumps({"results": list(items)})


VALID_RESPONSES = {
    "negative_without_reasoning": (
        _verdicts({"index": 0, "equivalent": False}, {"index": 1, "equivalent": True}),
        [(False, None), (True, None)],
    ),
    "positive_without_reasoning": (
        _verdicts({"index": 0, "equivalent": True}, {"index": 1, "equivalent": True}),
        [(True, None), (True, None)],
    ),
    "reordered_with_reasoning": (
        _verdicts(
            {"index": 1, "equivalent": False, "reasoning": "wrong recipient"},
            {"index": 0, "equivalent": True, "reasoning": "same facts"},
        ),
        [(True, "same facts"), (False, "wrong recipient")],
    ),
    "think_block_and_fence": (
        "<think>compare</think>\n```json\n"
        + _verdicts({"index": 0, "equivalent": True}, {"index": 1, "equivalent": False})
        + "\n```",
        [(True, None), (False, None)],
    ),
    "extra_keys_ignored": (
        _verdicts(
            {"index": 0, "equivalent": True, "confidence": 0.9},
            {"index": 1, "equivalent": True, "field": "subject"},
        ),
        [(True, None), (True, None)],
    ),
}

PROTOCOL_ERRORS = {
    "not_json": ("not json", "unparseable"),
    "empty": ("", "unparseable"),
    "no_results_key": ('{"verdicts": []}', "unparseable"),
    "results_not_list": ('{"results": {}}', "unparseable"),
    "top_level_list": ("[1, 2]", "unparseable"),
    "item_not_object": ('{"results": [1, {"index": 1, "equivalent": true}]}', "invalid_item"),
    "index_missing": (_verdicts({"equivalent": True}, {"index": 1, "equivalent": True}), "invalid_index"),
    "index_string": (_verdicts({"index": "0", "equivalent": True}, {"index": 1, "equivalent": True}), "invalid_index"),
    "index_bool": (_verdicts({"index": True, "equivalent": True}, {"index": 1, "equivalent": True}), "invalid_index"),
    "index_out_of_range": (_verdicts({"index": 0, "equivalent": True}, {"index": 2, "equivalent": True}), "invalid_index"),
    "index_negative": (_verdicts({"index": -1, "equivalent": True}, {"index": 1, "equivalent": True}), "invalid_index"),
    "index_duplicate": (_verdicts({"index": 0, "equivalent": True}, {"index": 0, "equivalent": False}), "duplicate_index"),
    "equivalent_missing": (_verdicts({"index": 0}, {"index": 1, "equivalent": True}), "invalid_equivalent"),
    "equivalent_string": (_verdicts({"index": 0, "equivalent": "false"}, {"index": 1, "equivalent": True}), "invalid_equivalent"),
    "equivalent_int": (_verdicts({"index": 0, "equivalent": 0}, {"index": 1, "equivalent": True}), "invalid_equivalent"),
    "equivalent_null": (_verdicts({"index": 0, "equivalent": None}, {"index": 1, "equivalent": True}), "invalid_equivalent"),
    "result_missing": (_verdicts({"index": 0, "equivalent": True}), "missing_index"),
    "results_empty": (_verdicts(), "missing_index"),
}


@pytest.mark.parametrize("case", sorted(VALID_RESPONSES))
def test_judge_parser_accepts_valid_verdicts(case: str) -> None:
    content, expected = VALID_RESPONSES[case]

    assert parse_text_judge_response(content, 2) == expected


@pytest.mark.parametrize("case", sorted(PROTOCOL_ERRORS))
def test_judge_parser_rejects_protocol_violations(case: str) -> None:
    content, kind = PROTOCOL_ERRORS[case]

    with pytest.raises(JudgeProtocolError) as raised:
        parse_text_judge_response(content, 2)

    assert raised.value.kind == kind


def _paraphrased_msp_021_env():
    return _run_actions(MSP_021, MSP_021_POSITIVE["paraphrased_message_fuzzy"])


def _mock_judge(monkeypatch, content: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def generate(**kwargs):
        calls.append(kwargs)
        return AssistantMessage(content=content)

    monkeypatch.setattr(text_match_strategy, "generate", generate)
    return calls


def test_mock_judge_pending_pairs_are_message_then_subject() -> None:
    task = TASKS[MSP_021]
    _, pending = collect_db_differences(
        replay_gold_actions(task).tools.db,
        _paraphrased_msp_021_env().tools.db,
        _baseline_db(MSP_021),
        LLM_JUDGE,
    )

    assert [pair["key"] for pair in pending] == [
        "notification/NOTIF_030 message",
        "notification/NOTIF_030 subject",
    ]


def test_mock_judge_request_matches_bench2_judge_free_text(monkeypatch) -> None:
    task = TASKS[MSP_021]
    _, pending = collect_db_differences(
        replay_gold_actions(task).tools.db,
        _paraphrased_msp_021_env().tools.db,
        _baseline_db(MSP_021),
        LLM_JUDGE,
    )
    calls = _mock_judge(monkeypatch, VALID_RESPONSES["positive_without_reasoning"][0])

    judge_free_text(pending, LLM_JUDGE.llm, LLM_JUDGE.llm_args)

    assert calls == [build_text_judge_request(pending, LLM_JUDGE)]


def test_mock_judge_negative_without_reasoning_is_a_valid_verdict(monkeypatch) -> None:
    calls = _mock_judge(monkeypatch, VALID_RESPONSES["negative_without_reasoning"][0])

    score = score_final_db(TASKS[MSP_021], _paraphrased_msp_021_env(), text_match=LLM_JUDGE)

    assert len(calls) == 1
    assert not score.db_match
    assert score.db_reward == 0.0
    assert [(v.key, v.equivalent, v.reasoning) for v in score.text_judgments] == [
        ("notification/NOTIF_030 message", False, None),
        ("notification/NOTIF_030 subject", True, None),
    ]
    assert any(m.startswith("notification/NOTIF_030 message:") and m.endswith("(judge)") for m in score.mismatches)


def test_mock_judge_positive_without_reasoning_is_a_valid_verdict(monkeypatch) -> None:
    calls = _mock_judge(monkeypatch, VALID_RESPONSES["positive_without_reasoning"][0])

    score = score_final_db(TASKS[MSP_021], _paraphrased_msp_021_env(), text_match=LLM_JUDGE)

    assert len(calls) == 1
    assert score.db_match
    assert score.text_match_effective == "llm"
    assert score.text_judge_model == "mock/judge"
    assert all(v.equivalent and v.reasoning is None for v in score.text_judgments)


@pytest.mark.parametrize(
    "case", ["result_missing", "not_json", "index_duplicate", "equivalent_missing"]
)
def test_mock_judge_protocol_violation_is_classified_infrastructure(monkeypatch, case: str) -> None:
    content, kind = PROTOCOL_ERRORS[case]
    calls = _mock_judge(monkeypatch, content)

    with pytest.raises(ItsmInfrastructureError) as raised:
        score_final_db(TASKS[MSP_021], _paraphrased_msp_021_env(), text_match=LLM_JUDGE)

    assert len(calls) == 1
    assert (raised.value.stage, raised.value.kind) == ("text_judge_protocol", kind)


def test_mock_judge_call_exception_is_classified_infrastructure(monkeypatch) -> None:
    def generate(**kwargs):
        raise TimeoutError("judge timed out")

    monkeypatch.setattr(text_match_strategy, "generate", generate)

    with pytest.raises(ItsmInfrastructureError) as raised:
        score_final_db(TASKS[MSP_021], _paraphrased_msp_021_env(), text_match=LLM_JUDGE)

    assert raised.value.stage == "text_judge_call"
    assert "TimeoutError" in raised.value.detail
