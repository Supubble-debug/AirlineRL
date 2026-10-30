"""Lossless tool-schema serializations and the committed token comparison (step 6).

Round-trips use the real ITSM tool schemas; no tokenizer is needed. The token
figures come from docs/itsm_step6/schema_serialization.json.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from enterprise_worlds.gym.gym_agent import DONE_TOOL_SCHEMA

from agent_rl.envs.itsm_scoring import build_task_environment, load_itsm_tasks
from agent_rl.prompts.tool_serialization import SERIALIZATIONS, roundtrip_equal


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = json.loads((REPO_ROOT / "docs" / "itsm_step6" / "schema_serialization.json").read_text(encoding="utf-8"))
MEASURED = json.loads((REPO_ROOT / "docs" / "itsm_step6" / "token_measurements.json").read_text(encoding="utf-8"))
BUDGET = yaml.safe_load((REPO_ROOT / "configs" / "itsm" / "budget_v1.yaml").read_text(encoding="utf-8"))
TASK = next(iter(load_itsm_tasks().values()))
TOOLS = build_task_environment(TASK).get_tool_schemas() + [copy.deepcopy(DONE_TOOL_SCHEMA)]


@pytest.mark.parametrize("name", sorted(SERIALIZATIONS))
def test_every_variant_parses_back_to_identical_schemas(name: str) -> None:
    parsed = SERIALIZATIONS[name].parse(SERIALIZATIONS[name].render(TOOLS))

    assert roundtrip_equal(name, TOOLS)
    assert json.dumps(parsed, sort_keys=True) == json.dumps(TOOLS, sort_keys=True)
    assert len(parsed) == len(TOOLS) == 94


def test_template_default_is_the_chat_template_tojson() -> None:
    block = SERIALIZATIONS["template_default"].render(TOOLS)

    assert block.split("\n")[0] == json.dumps(TOOLS[0], ensure_ascii=False)


def test_committed_comparison_is_lossless_and_matches_the_main_measurement() -> None:
    variants = RESULTS["variants"]

    assert set(variants) == set(SERIALIZATIONS)
    for result in variants.values():
        assert result["parse_equal"] and result["parse_equal_canonical_json"]
        assert result["outside_tools_block_unchanged"]
    assert variants["template_default"]["initial_prompt_tokens"] == MEASURED["summary"]["initial_prompt_tokens"]["max"]
    assert RESULTS["tokenizer"]["chat_template_sha256"] == MEASURED["tokenizer"]["chat_template_sha256"]
    # Measured only; the budget still uses the chat template's own serialization.
    assert BUDGET["tokenizer"]["tool_schema_serialization"] == "template_default"
