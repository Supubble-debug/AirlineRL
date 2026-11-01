"""summarize_8: a top-level auxiliary ``not_evaluated`` (aux failed as a whole) must be counted
as not evaluated for both the LLM-DB and the NL part -- never dropped, never counted as 0."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_summarize():
    spec = importlib.util.spec_from_file_location("summarize_8", REPO_ROOT / "scripts/itsm/summarize_8.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trajectory(index, aux, reward=0.0):
    return {"group_id": "A-t", "part": "A", "task_id": "t", "sample_index": index, "prompt_ids": [1] * 10, "response_ids": [2] * 5,
            "reward_score": reward, "extra_fields": {"itsm_aux_eval": aux, "itsm_turn_spans": [{"turn_index": 0, "start": 0, "end": 5}],
                                                     "itsm_env_spans": [], "reward_extra_info": {"itsm_termination_reason": "agent_done"}}}


def test_top_level_not_evaluated_is_counted_for_both_parts(tmp_path):
    completed = {"status": "completed", "db_llm": {"status": "evaluated", "db_match": True}, "nl": {"status": "evaluated", "nl_met": False}}
    failed_whole = {"status": "not_evaluated", "reason": "timeout", "error": "ItsmInfrastructureError: timeout"}
    rows = [trajectory(0, completed, 1.0), trajectory(1, failed_whole), trajectory(2, failed_whole), trajectory(3, completed)]
    (tmp_path / "groups.jsonl").write_text(json.dumps({"group_id": "A-t", "part": "A", "task_id": "t", "status": "complete", "sample_status": ["valid"] * 4}) + "\n")
    (tmp_path / "trajectories.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "attempts.jsonl").write_text("")
    a = load_summarize().summarise(tmp_path)["A_e0_base_dev_diagnostic"]["tasks"][0]
    assert a["n"] == 4 and a["successes_fuzzy"] == 1  # the fuzzy result is unaffected
    assert a["llm_db"] == {"evaluated": 2, "not_evaluated": 2, "matches": 2, "not_evaluated_reasons": {"timeout": 2}}
    assert a["nl"]["evaluated"] == 2 and a["nl"]["not_evaluated"] == 2 and a["nl"]["met"] == 0
    assert a["nl"]["not_evaluated_reasons"] == {"timeout": 2}
    assert a["joint_db_llm_and_nl"] == {"evaluated": 2, "successes": 0}
    assert a["llm_db"]["evaluated"] + a["llm_db"]["not_evaluated"] == a["n"]  # every sample accounted for
