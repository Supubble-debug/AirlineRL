"""The ITSM budget stays consistent with the split v2 token measurements (step 6C).

Reads committed files only; no tokenizer download, model or LLM.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from agent_rl.data.itsm_split_v2 import ACCEPTED_DERIVED_TASK_IDS, DEFAULT_V2_PATH


REPO_ROOT = Path(__file__).resolve().parents[2]
BUDGET = yaml.safe_load((REPO_ROOT / "configs" / "itsm" / "budget_v1.yaml").read_text(encoding="utf-8"))
TRAIN = BUDGET["train"]
BASIS = BUDGET["basis"]["itsm_split_v2"]
MEASURED = json.loads((REPO_ROOT / BASIS["measurements"]).read_text(encoding="utf-8"))
V1_MEASURED = json.loads((REPO_ROOT / BUDGET["basis"]["itsm_split_v1"]["measurements"]).read_text(encoding="utf-8"))
SUMMARY = MEASURED["summary"]
MANIFEST = json.loads(DEFAULT_V2_PATH.read_text(encoding="utf-8"))


def test_v2_measures_exactly_the_45_train_and_dev_tasks() -> None:
    measured = MEASURED["data"]["tasks_measured"]

    assert MEASURED["data"]["split_version"] == "itsm_split_v2"
    assert len(measured) == len(set(measured)) == 45
    assert set(measured) == set(MANIFEST["splits"]["train"]) | set(MANIFEST["splits"]["dev"])
    assert not set(measured) & set(MANIFEST["splits"]["heldout"])
    assert MEASURED["data"]["derived_tasks_measured"] == sorted(ACCEPTED_DERIVED_TASK_IDS)
    assert all(episode["db_match"] for episode in MEASURED["episodes"])


def test_same_tokenizer_and_full_observation_rule() -> None:
    assert MEASURED["tokenizer"] | {"transformers_version": None} == V1_MEASURED["tokenizer"] | {"transformers_version": None}
    assert MEASURED["observation_rule"] == TRAIN["observation_rule"] == "full_tool_results_no_truncation"
    assert TRAIN["context_rotation"] is False


def test_budget_values_still_cover_the_v2_gold_path() -> None:
    assert BASIS["values_changed"] is False and BASIS["status"] == "pending_step6c_acceptance"
    assert TRAIN["max_prompt_length"] >= SUMMARY["initial_prompt_tokens"]["max"] + 256
    assert TRAIN["max_response_length"] >= 2 * SUMMARY["response_tokens_gold_replay"]["max"]
    assert TRAIN["max_action_tokens"] >= 2 * SUMMARY["generated_tokens_per_tool_call"]["max"]
    assert TRAIN["max_steps"] >= 2 * SUMMARY["step_count_gold_replay"]["max"]
    # Gold-path maxima did not move with the three derived tasks.
    for key in ("initial_prompt_tokens", "response_tokens_gold_replay", "full_sequence_tokens",
                "generated_tokens_per_tool_call", "tool_observation_raw_tokens", "step_count_gold_replay"):
        assert SUMMARY[key]["max"] == V1_MEASURED["summary"][key]["max"], key


def test_unfiltered_reads_over_the_response_budget_are_recorded() -> None:
    over = [
        {"tool": r["tool"], "task_id": r["task_id"], "raw_tokens": r["max_raw_observation_tokens"]}
        for r in MEASURED["unfiltered_read_observations"]
        if r["max_raw_observation_tokens"] > TRAIN["max_response_length"]
    ]

    assert over == BASIS["unfiltered_reads_over_max_response_length"]
    assert {r["task_id"] for r in over} == {"itsmbench_eops_1d219d93_d1ba6c2d"}
    # v1 had none: every v1 unfiltered read fit the response budget.
    assert max(r["max_raw_observation_tokens"] for r in V1_MEASURED["unfiltered_read_observations"]) < TRAIN["max_response_length"]
    by_task = {r["task_id"]: r for r in MEASURED["unfiltered_read_max_by_task"]}
    assert all(not by_task[t]["org_sliced"] for t in ACCEPTED_DERIVED_TASK_IDS)
    assert all(r["org_sliced"] for t, r in by_task.items() if t not in ACCEPTED_DERIVED_TASK_IDS)
