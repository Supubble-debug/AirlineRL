"""The v1 ITSM budget stays consistent with the committed token measurements (step 6).

Reads committed files only; no tokenizer download, model or LLM.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from agent_rl.data.itsm_split import DEFAULT_MANIFEST_PATH


REPO_ROOT = Path(__file__).resolve().parents[2]
BUDGET = yaml.safe_load((REPO_ROOT / "configs" / "itsm" / "budget_v1.yaml").read_text(encoding="utf-8"))
MEASURED = json.loads((REPO_ROOT / "docs" / "itsm_step6" / "token_measurements.json").read_text(encoding="utf-8"))
SUMMARY = MEASURED["summary"]
TRAIN = BUDGET["train"]
SHARED_KEYS = (
    "max_model_len",
    "max_prompt_length",
    "max_response_length",
    "max_action_tokens",
    "max_steps",
    "max_agent_turns",
    "context_rotation",
    "observation_rule",
    "user_llm",
    "user_llm_temperature",
    "worker",
)


def test_budget_names_the_measured_tokenizer() -> None:
    tokenizer = MEASURED["tokenizer"]

    assert BUDGET["tokenizer"]["revision"] == tokenizer["revision"]
    assert BUDGET["tokenizer"]["chat_template_sha256"] == tokenizer["chat_template_sha256"]
    assert BUDGET["tokenizer"]["enable_thinking"] is tokenizer["enable_thinking"] is False


def test_training_and_evaluation_share_one_episode_budget() -> None:
    assert {key: TRAIN[key] for key in SHARED_KEYS} == {key: BUDGET["eval"][key] for key in SHARED_KEYS}


def test_context_fits_the_model_and_the_measured_prompt() -> None:
    assert TRAIN["max_model_len"] <= MEASURED["tokenizer"]["max_position_embeddings"]
    assert TRAIN["max_prompt_length"] + TRAIN["max_response_length"] <= TRAIN["max_model_len"]
    assert TRAIN["max_prompt_length"] >= SUMMARY["initial_prompt_tokens"]["max"] + 256


def test_40960_is_recorded_as_an_unverified_candidate_beyond_the_native_context() -> None:
    context = BUDGET["context"]

    assert (context["native_context"], context["max_model_len_candidate"]) == (32768, 40960)
    assert context["status"] == "candidate_pending"
    assert {"step7", "step9"} <= set(context["verify"])
    assert TRAIN["max_model_len"] == BUDGET["eval"]["max_model_len"] == context["max_model_len_candidate"]
    # The measured prompt does not leave room for the longest gold replay at 32768.
    room = context["native_context"] - SUMMARY["initial_prompt_tokens"]["max"]
    assert room < SUMMARY["response_tokens_gold_replay"]["max"]


def test_budget_and_measurements_use_the_same_full_observation_rule() -> None:
    rule = "full_tool_results_no_truncation"

    assert BUDGET["observation_rule"] == TRAIN["observation_rule"] == BUDGET["eval"]["observation_rule"] == rule
    assert MEASURED["observation_rule"] == rule
    assert "max_observation_tokens" not in TRAIN and "max_observation_tokens" not in BUDGET["eval"]


def test_checkpoints_are_not_selected_on_dev_or_heldout() -> None:
    manifest = json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))

    assert TRAIN["checkpoint_selection"] == "pre_registered_steps_and_rule_frozen_by_step8"
    assert "not checkpoint selection" in manifest["usage"]
    assert "workflow-composition" in manifest["positioning"]


def test_per_episode_and_per_action_budgets_have_headroom_over_gold() -> None:
    assert TRAIN["max_response_length"] >= 2 * SUMMARY["response_tokens_gold_replay"]["max"]
    assert TRAIN["max_action_tokens"] >= 2 * SUMMARY["generated_tokens_per_tool_call"]["max"]
    assert TRAIN["max_action_tokens"] != 256  # not inherited from airline
    assert TRAIN["max_steps"] >= 2 * SUMMARY["step_count_gold_replay"]["max"]


def test_measurements_never_read_heldout_and_replays_succeed() -> None:
    manifest = json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
    measured = set(MEASURED["data"]["tasks_measured"])

    assert measured == set(manifest["splits"]["train"]) | set(manifest["splits"]["dev"])
    assert not measured & set(manifest["splits"]["heldout"])
    assert all(episode["db_match"] for episode in MEASURED["episodes"])
