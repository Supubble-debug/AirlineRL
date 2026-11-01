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


# -- hard caps (R1 item 5) ------------------------------------------------------------------------

#: Measured on the pinned Qwen3-8B tokenizer (token 56940 decodes to 128 bytes of spaces).
QWEN_MAX_BYTES_PER_TOKEN = 128
#: json.dumps(ensure_ascii=False) turns one raw byte into at most 6 (control chars -> \\u00XX).
JSON_ESCAPE_FACTOR = 6
#: Every text the simulator/judges can see is either in the policy prompt (<= max_prompt_length)
#: or in the response (<= response_length) of the attempt, in Qwen tokens.
CONTEXT_TOKENS = 28672 + 12288


def max_system_prompt_bytes(plan: dict[str, Any]) -> dict[str, int]:
    """Largest D5 system prompt over the planned tasks, and the two D7 system prompts (raw bytes)."""
    from enterprise_worlds.evaluator import evaluator_nl, text_match_strategy
    from enterprise_worlds.user.user_simulator import UserSimulator

    from agent_rl.data.itsm_task_sets import resolve_task

    ids = part_task_ids(plan, "A") + part_task_ids(plan, "B")
    sim = max(len(UserSimulator(resolve_task(t, split_version=plan["split_version"]).scenario, llm="x", llm_args={}).system_prompt.encode()) for t in ids)
    return {"simulator": sim, "text_judge": len(text_match_strategy.JUDGE_SYSTEM_PROMPT.encode()), "nl_judge": len(evaluator_nl.JUDGE_SYSTEM_PROMPT.encode())}


def hard_cap_table(plan: dict[str, Any], prices: dict[str, Price], *, system_bytes: dict[str, int] | None = None,
                   request_byte_caps: dict[str, int] | None = None) -> dict[str, Any]:
    """Hard maximum spend of one COMPLETE group, every allowed retry included.

    Counts (all from the rules, nothing estimated):
    * simulator calls per attempt <= max_steps (40): every user turn, the opening one included,
      is a step in ``ItsmEnv``, and the user is not asked once steps reach max_steps;
    * attempts per sample <= max_attempts (4) (protocol §12);
    * auxiliary judges (part A) run once per valid sample, each <= 4 attempts: <= 2 x 4 calls;
    * per request: input bound = request bytes + 64 (as reserved), output = the max_tokens cap.
    Request bytes: without a size rule they are only bounded by the context
    (``CONTEXT_TOKENS`` x 128 bytes x JSON escape 6 + the system prompt); ``request_byte_caps``
    evaluates a proposed explicit size rule instead (not enforced by the current code).
    """
    system_bytes = system_bytes or max_system_prompt_bytes(plan)
    e, attempts, steps = plan["envelope"], plan["max_attempts"], 40
    out_cap = {"simulator": e["simulator_max_tokens"], "text_judge": e["judge_max_tokens"], "nl_judge": e["judge_max_tokens"]}
    role = {"simulator": "simulator", "text_judge": "judge", "nl_judge": "judge"}

    def body_bytes(kind: str) -> int:
        if request_byte_caps is not None:
            return request_byte_caps[kind]
        return JSON_ESCAPE_FACTOR * (system_bytes[kind] + CONTEXT_TOKENS * QWEN_MAX_BYTES_PER_TOKEN) + 1024  # + JSON framing

    per_request = {k: prices[role[k]].cost(body_bytes(k) + 64, out_cap[k]) for k in role}
    per_sample = {"B": attempts * steps * per_request["simulator"]}
    per_sample["A"] = per_sample["B"] + attempts * (per_request["text_judge"] + per_request["nl_judge"])
    samples = {p: plan["parts"][p]["samples_per_task"] for p in plan["parts"]}
    group = {p: samples[p] * per_sample[p] for p in plan["parts"]}
    cap = plan["budget"]["cap_usd"]
    return {
        "counts": {"simulator_calls_per_attempt": steps, "attempts_per_sample": attempts, "judge_calls_per_valid_sample": 2 * attempts,
                   "samples_per_group": samples},
        "request_bytes": {k: body_bytes(k) for k in role},
        "per_request_usd": {k: round(v, 4) for k, v in per_request.items()},
        "per_sample_usd": {k: round(v, 4) for k, v in per_sample.items()},
        "complete_group_hard_cap_usd": {k: round(v, 4) for k, v in group.items()},
        "fits_in_cap_usd": cap,
        "groups_that_fit_at_once": {k: int(cap // v) for k, v in group.items()},
    }


def make_step8_loop(group: GroupPlan, *, server: Any, tokenizer: Any, plan: dict[str, Any], budget_address: str, budget_authkey: bytes,
                    response_length: int = 12288, prompt_length: int = 28672, settings_overrides: dict[str, Any] | None = None):
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
                                          auxiliary_evaluation=auxiliary, **(settings_overrides or {}))
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
