"""Counterfactual length-budget analysis along trajectories sampled at max_model_len 40960.

THIS IS NOT A 32k RUN. It takes the token sequence that was actually sampled with
``max_model_len = 40960`` and asks, turn by turn, where ``ItsmAgentLoop``'s own length rules
would have stopped the SAME sequence under a smaller ``max_model_len`` (e.g. 32768):

* response budget R = ``response_length`` = max_model_len - max_prompt_length (budget v1:
  28672), as the loop is configured (``rollout_config.response_length``); the server also
  clamps each request to max_model_len - current input, which is applied as well;
* before policy turn t (starting at response offset s) the loop requests
  ``max_tokens = min(max_action_tokens, R - s)``; an actual turn longer than that would have
  been cut -> ``invalid_action`` (``action_token_limit``);
* after policy turn t (ending at e), the next environment turn of length L is appended only if
  ``e + L + max_action_tokens <= R``, else ``truncate("response_length")``.

What it cannot tell: the policy's later actions, the reward or the termination under the
smaller budget (a different cut changes everything after it). Reported only as "would have
been cut at turn t, by rule r"; the 40960 reward is shown next to it, never re-labelled.
"""

from __future__ import annotations

from typing import Any

COUNTERFACTUAL_LABEL = "counterfactual budget analysis along 40960 trajectories (not an actual run at this length)"


MAX_PROMPT_LENGTH = 28672  # budget v1


def first_cut(prompt_tokens: int, turn_spans: list[dict[str, Any]], env_spans: list[dict[str, Any]], *,
              max_model_len: int, max_action_tokens: int = 512, max_prompt_length: int = MAX_PROMPT_LENGTH) -> dict[str, Any]:
    if prompt_tokens > max_prompt_length:
        return {"cut": True, "rule": "prompt_over_max_prompt_length", "at_turn": 0}
    budget = min(max_model_len - max_prompt_length, max_model_len - prompt_tokens)
    if budget < max_action_tokens:
        return {"cut": True, "rule": "prompt_leaves_no_action_budget", "at_turn": 0, "response_budget": budget}
    env_after = {span["after_turn"]: span for span in env_spans}
    for turn in turn_spans:
        start, end, index = turn["start"], turn["end"], turn["turn_index"]
        allowed = min(max_action_tokens, budget - start)
        if end - start > allowed:
            return {"cut": True, "rule": "action_token_limit", "at_turn": index, "response_budget": budget,
                    "turn_tokens": end - start, "allowed_tokens": allowed}
        span = env_after.get(index)
        if span is not None and end + (span["end"] - span["start"]) + max_action_tokens > budget:
            return {"cut": True, "rule": "response_length", "at_turn": index, "response_budget": budget,
                    "tokens_at_cut": end, "next_env_tokens": span["end"] - span["start"]}
    return {"cut": False, "response_budget": budget}


def analyse(trajectories: list[dict[str, Any]], *, max_model_len: int = 32768, max_action_tokens: int = 512) -> dict[str, Any]:
    rows = []
    for t in trajectories:
        extra = t["extra_fields"]
        cut = first_cut(len(t["prompt_ids"]), extra["itsm_turn_spans"], extra["itsm_env_spans"], max_model_len=max_model_len,
                        max_action_tokens=max_action_tokens)
        total = len(t["prompt_ids"]) + len(t["response_ids"])
        rows.append({"group_id": t["group_id"], "task_id": t["task_id"], "sample_index": t["sample_index"], "total_tokens_at_40960": total,
                     "exceeds_length": total > max_model_len, "actual_reward_at_40960": t["reward_score"],
                     "actual_termination_at_40960": extra["reward_extra_info"]["itsm_termination_reason"], "counterfactual": cut})
    return {"label": COUNTERFACTUAL_LABEL, "max_model_len": max_model_len, "episodes": len(rows),
            "would_be_cut": sum(r["counterfactual"]["cut"] for r in rows),
            "would_be_cut_by_rule": {rule: sum(r["counterfactual"].get("rule") == rule for r in rows)
                                     for rule in ("response_length", "action_token_limit", "prompt_leaves_no_action_budget", "prompt_over_max_prompt_length")},
            "longer_than_length": sum(r["exceeds_length"] for r in rows),
            "rows": rows}
