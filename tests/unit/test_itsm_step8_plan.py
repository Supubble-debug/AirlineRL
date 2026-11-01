"""Step 8 plan, reservation numbers, concurrency ramp and the 32k counterfactual (offline, fast)."""

from __future__ import annotations

import pytest

from agent_rl.data import itsm_task_sets
from agent_rl.eval.itsm_length_counterfactual import COUNTERFACTUAL_LABEL, first_cut
from agent_rl.rollout.itsm_diagnostic_runner import ConcurrencyRamp
from agent_rl.rollout.itsm_step8 import build_plan, load_plan, load_prices, reservation_table


def test_plan_is_the_confirmed_scope_in_a_fixed_order():
    plan, prices = load_plan(), load_prices()
    groups = build_plan(plan, prices)
    a = [g for g in groups if g.part == "A"]
    b = [g for g in groups if g.part == "B"]
    assert [g.task_id for g in a] == sorted(itsm_task_sets.task_ids("dev", split_version="itsm_split_v3")) and len(a) == 13
    assert [g.task_id for g in b] == sorted(itsm_task_sets.training_task_ids(purpose="grpo", usage_version="itsm_usage_v1")) and len(b) == 26
    assert groups == a + b and all(g.samples == 4 for g in groups)
    heldout = set(itsm_task_sets.task_ids("heldout", split_version="itsm_split_v3", final_evaluation=True))
    assert not heldout & {g.task_id for g in groups}
    assert plan["sampling"]["temperature"] == 1.0 and plan["max_model_len"] == 40960 and plan["budget"]["cap_usd"] == 12.0
    assert plan["envelope"]["simulator_max_tokens"] == 1024 and plan["gpu"]["cap_gpu_h"] == 2.0
    assert plan["review"] == {"budget_id": "step8_api_review", "cap_usd": 1.0, "max_billed_requests": 20}


def test_reservation_numbers():
    table = reservation_table(load_plan(), load_prices())
    assert table["per_request_reservation_usd"] == {"simulator": pytest.approx(32768 * 0.30e-6 + 1024 * 1.20e-6, abs=1e-6),
                                                    "text_judge": pytest.approx(32768 * 1.32e-6 + 4096 * 3.96e-6, abs=1e-6),
                                                    "nl_judge": pytest.approx(98304 * 1.32e-6 + 4096 * 3.96e-6, abs=1e-6)}
    assert table["group_envelope_usd"]["A"] == pytest.approx(4 * (12 * table["per_request_reservation_usd"]["simulator"]
                                                                  + table["per_request_reservation_usd"]["text_judge"] + table["per_request_reservation_usd"]["nl_judge"]), abs=1e-4)  # table values are rounded
    assert table["max_committed_by_envelopes_in_flight_usd"] < table["cap_usd"]


def test_ramp_rules():
    ramp = ConcurrencyRamp(levels=(2, 4, 8), promote_after=4)
    for _ in range(4):
        ramp.on_clean()
    assert ramp.limit == 4
    ramp.on_unclean()
    for _ in range(3):
        ramp.on_clean()
    assert ramp.limit == 4  # the streak restarted
    ramp.on_clean()
    assert ramp.limit == 8
    for _ in range(10):
        ramp.on_clean()
    assert ramp.limit == 8  # never above 8
    assert ramp.on_engine_error() is False and ramp.limit == 4
    assert ramp.on_engine_error() is False and ramp.limit == 2
    assert [ramp.on_engine_error() for _ in range(3)] == [False, False, True]  # three at the floor: stop
    with pytest.raises(ValueError):
        ConcurrencyRamp(levels=(2, 16))


def test_counterfactual_cut_rules():
    # prompt 28,200; response budget at 32768 = min(32768-28672, 32768-28200) = 4096
    turns = [{"turn_index": 0, "start": 0, "end": 100}, {"turn_index": 1, "start": 2100, "end": 2200}, {"turn_index": 2, "start": 3700, "end": 3900}]
    envs = [{"after_turn": 0, "start": 100, "end": 2100}, {"after_turn": 1, "start": 2200, "end": 3700}]
    assert first_cut(28200, turns, envs, max_model_len=40960) == {"cut": False, "response_budget": 12288}
    cut = first_cut(28200, turns, envs, max_model_len=32768)
    assert cut["cut"] and cut["rule"] == "response_length" and cut["at_turn"] == 1  # 2200 + 1500 + 512 > 4096
    long_action = [{"turn_index": 0, "start": 3800, "end": 4100}]  # 300 tokens, 296 allowed
    assert first_cut(28200, long_action, [], max_model_len=32768)["rule"] == "action_token_limit"  # only 296 allowed
    assert "not an actual run" in COUNTERFACTUAL_LABEL
