"""Step 8 wiring: plan, reservation table, loops and the run (configs/itsm/step8/plan.json).

* ``build_plan``: groups in a fixed order -- part A (dev tasks, sorted) then part B
  (GRPO-trainable tasks, sorted); held-out is never read.
* ``envelope_usd`` / ``reservation_table``: the per-group pre-reservation and the
  per-request reservations, from the plan's bounds and the committed prices.
* ``make_step8_loop``: an ``ItsmAgentLoop`` whose env workers use the ledgered D5 user and,
  for part A, the ledgered D7 auxiliary judges; ``litellm`` is replaced by a raising stub in
  every worker child, so no request can bypass the ledger.
"""

from __future__ import annotations

import asyncio
import functools
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent_rl.rollout.itsm_diagnostic_runner import EngineGuard, GroupPlan
from agent_rl.utils.spend_ledger import Price

REPO_ROOT = Path(__file__).resolve().parents[3]
PLAN_PATH = REPO_ROOT / "configs" / "itsm" / "step8" / "plan.json"
PRICES_PATH = REPO_ROOT / "configs" / "itsm" / "step8" / "prices.json"


def load_plan(path: Path = PLAN_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_prices(path: Path = PRICES_PATH) -> dict[str, Price]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {role: Price(**raw[role]) for role in ("simulator", "judge")}


def per_request_usd(plan: dict[str, Any], prices: dict[str, Price]) -> dict[str, float]:
    e = plan["envelope"]
    return {"simulator": prices["simulator"].cost(e["simulator_input_bound_tokens"], e["simulator_max_tokens"]),
            "text_judge": prices["judge"].cost(e["text_judge_input_bound_tokens"], e["judge_max_tokens"]),
            "nl_judge": prices["judge"].cost(e["nl_judge_input_bound_tokens"], e["judge_max_tokens"])}


def attempt_envelope_usd(plan: dict[str, Any], prices: dict[str, Price], *, auxiliary: bool) -> float:
    e, r = plan["envelope"], per_request_usd(plan, prices)
    total = e["simulator_calls_per_attempt"] * r["simulator"]
    if auxiliary:
        total += e["text_judge_calls_per_attempt"] * r["text_judge"] + e["nl_judge_calls_per_attempt"] * r["nl_judge"]
    return total


def envelope_usd(plan: dict[str, Any], prices: dict[str, Price], part: str) -> float:
    spec = plan["parts"][part]
    return spec["samples_per_task"] * attempt_envelope_usd(plan, prices, auxiliary=spec["auxiliary_evaluation"])


def part_task_ids(plan: dict[str, Any], part: str) -> list[str]:
    from agent_rl.data import itsm_task_sets

    spec = plan["parts"][part]
    if spec["tasks"] == "dev":
        return sorted(itsm_task_sets.task_ids("dev", split_version=plan["split_version"]))
    if spec["tasks"] == "grpo_trainable":
        return sorted(itsm_task_sets.training_task_ids(purpose="grpo", usage_version=plan["usage_version"]))
    raise ValueError(f"unknown task set {spec['tasks']!r}")


def build_plan(plan: dict[str, Any], prices: dict[str, Price], parts: tuple[str, ...] = ("A", "B")) -> list[GroupPlan]:
    groups = []
    for part in parts:
        spec = plan["parts"][part]
        for task_id in part_task_ids(plan, part):
            groups.append(GroupPlan(group_id=f"{part}-{task_id}", part=part, purpose=spec["purpose"], task_id=task_id,
                                    samples=spec["samples_per_task"], envelope_usd=round(envelope_usd(plan, prices, part), 6)))
    return groups


def reservation_table(plan: dict[str, Any], prices: dict[str, Price]) -> dict[str, Any]:
    groups = build_plan(plan, prices)
    by_part = {p: [g for g in groups if g.part == p] for p in plan["parts"]}
    return {
        "per_request_reservation_usd": {k: round(v, 6) for k, v in per_request_usd(plan, prices).items()},
        "attempt_envelope_usd": {p: round(attempt_envelope_usd(plan, prices, auxiliary=plan["parts"][p]["auxiliary_evaluation"]), 6) for p in plan["parts"]},
        "group_envelope_usd": {p: round(envelope_usd(plan, prices, p), 6) for p in plan["parts"]},
        "groups": {p: len(g) for p, g in by_part.items()},
        "episodes": {p: sum(x.samples for x in g) for p, g in by_part.items()},
        "sum_of_all_first_envelopes_usd": round(sum(g.envelope_usd for g in groups), 6),
        "cap_usd": plan["budget"]["cap_usd"],
        # worst case: one group per running episode at the top concurrency level
        "max_groups_in_flight": plan["ramp"]["levels"][-1],
        "max_committed_by_envelopes_in_flight_usd": round(plan["ramp"]["levels"][-1] * max(envelope_usd(plan, prices, p) for p in plan["parts"]), 6),
    }


def make_step8_loop(group: GroupPlan, *, server: Any, tokenizer: Any, plan: dict[str, Any], budget_address: str, budget_authkey: bytes,
                    response_length: int = 12288, prompt_length: int = 28672):
    from verl.utils.tokenizer.chat_template import initialize_system_prompt, initialize_turn_separator

    from agent_rl.envs.itsm_ledgered_user import make_ledgered_user
    from agent_rl.envs.itsm_scripted_user import forbid_external_llm
    from agent_rl.eval.itsm_aux_eval import make_aux_evaluator
    from agent_rl.rollout.itsm_agent_loop import ItsmAgentLoop, ItsmAgentLoopSettings

    auxiliary = plan["parts"][group.part]["auxiliary_evaluation"]

    class Step8Loop(ItsmAgentLoop):
        worker_initializer = staticmethod(forbid_external_llm)  # nothing may reach a provider outside the ledger
        user_factory = functools.partial(make_ledgered_user, group.task_id, plan["split_version"], budget_address, budget_authkey, group.group_id)
        aux_evaluator_factory = functools.partial(make_aux_evaluator, budget_address, budget_authkey, group.group_id) if auxiliary else None

    loop = object.__new__(Step8Loop)
    loop.settings = ItsmAgentLoopSettings(purpose=group.purpose, split_version=plan["split_version"], usage_version=plan["usage_version"],
                                          auxiliary_evaluation=auxiliary)
    loop.tokenizer = tokenizer
    loop.processor = None
    loop.apply_chat_template_kwargs = {"enable_thinking": False}
    loop.system_prompt = initialize_system_prompt(tokenizer, enable_thinking=False)
    loop.turn_separator = initialize_turn_separator(tokenizer, enable_thinking=False)
    loop.rollout_config = SimpleNamespace(prompt_length=prompt_length, response_length=response_length)
    loop.server_manager = EngineGuard(server)
    loop.loop = asyncio.get_running_loop()  # called by the runner inside its event loop
    loop._check_template()
    return loop
