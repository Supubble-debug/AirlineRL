"""Step 8 ledgered D5 user and worker-side D7 auxiliary evaluation (offline, mocked HTTP)."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import httpx
import pytest

from enterprise_worlds.data_model.message import AssistantMessage
from enterprise_worlds.evaluator import evaluator_nl, text_match_strategy
from enterprise_worlds.evaluator.text_match_strategy import TextMatchConfig
from enterprise_worlds.user import user_simulator as bench2_user

from agent_rl.data.itsm_task_sets import resolve_task
from agent_rl.envs.itsm_env import ItsmAction, ItsmEnv, ItsmEnvConfig
from agent_rl.envs.itsm_ledgered_user import LedgeredUserSimulator, UserSimContractError
from agent_rl.envs.itsm_scoring import ItsmInfrastructureError
from agent_rl.eval.itsm_aux_eval import AuxEvaluator, NL_PROMPT_SHA256, TEXT_PROMPT_SHA256, _sha, parse_nl_response
from agent_rl.llm.deepseek_chat import DeepSeekChat
from agent_rl.utils.budget_manager import BudgetManager
from agent_rl.utils.spend_ledger import Price, SpendLedger

sys.path.insert(0, str(Path(__file__).resolve().parent))
from itsm_test_support import ScriptedUser  # noqa: E402

SPLIT = "itsm_split_v3"
DEV_TASK = "itsmbench_msp_019"
PRICES = {"simulator": Price(input_per_mtok=0.30, output_per_mtok=1.20, source="test"), "judge": Price(input_per_mtok=1.32, output_per_mtok=3.96, source="test")}


def chat_for(tmp_path, role, handler, name="l.jsonl"):
    ledger = SpendLedger(tmp_path / name, cap_usd=5.0, max_billed_requests=1000, budget_id="t")
    manager = BudgetManager(ledger, PRICES)
    manager.open_group("g", 1.0, {})
    sent = []

    def counted(request):
        sent.append(json.loads(request.content))
        return handler(request, len(sent))

    return DeepSeekChat(api_key="k", budget=manager, group_id="g", role=role, http=httpx.Client(transport=httpx.MockTransport(counted))), sent, ledger


def reply(content, *, model="deepseek-flash", finish="stop", reasoning=None):
    message = {"role": "assistant", "content": content, **({"reasoning_content": reasoning} if reasoning else {})}
    return httpx.Response(200, json={"id": "x", "model": model, "choices": [{"finish_reason": finish, "message": message}],
                                     "usage": {"prompt_tokens": 50, "completion_tokens": 10}})


# -- D5 ledgered user ------------------------------------------------------------------------------


def test_ledgered_user_sends_exactly_bench2_messages(tmp_path, monkeypatch):
    scenario = resolve_task(DEV_TASK, split_version=SPLIT).scenario
    captured = []

    def fake_generate(model, messages, **kwargs):
        captured.append((model, bench2_user.generate.__module__ and [m.model_dump() for m in messages], kwargs))
        return AssistantMessage(content="ok")

    monkeypatch.setattr(bench2_user, "generate", fake_generate)
    reference = bench2_user.UserSimulator(scenario, llm="deepseek/deepseek-flash", llm_args={"temperature": 0.0})
    state = reference.get_init_state()
    reference.generate_next_message(AssistantMessage(content="Hi, how can I help?"), state)

    chat, sent, ledger = chat_for(tmp_path, "simulator", lambda r, n: reply("ok"))
    ours = LedgeredUserSimulator(scenario, chat=chat)
    ours.generate_next_message(AssistantMessage(content="Hi, how can I help?"), ours.get_init_state())
    from enterprise_worlds.utils.llm_utils import to_litellm_messages
    from enterprise_worlds.data_model.message import Message  # noqa: F401

    expected = to_litellm_messages(list(state.system_messages) + state.flip_roles()[:-1])  # before the reply was appended
    assert sent[0]["messages"] == expected
    assert sent[0]["model"] == "deepseek-flash" and sent[0]["temperature"] == 0.0 and sent[0]["max_tokens"] == 1024
    assert sent[0]["thinking"] == {"type": "disabled"}


@pytest.mark.parametrize(("response", "kind"), [
    (reply("cut mid-senten", finish="length"), "finish_reason_length"),
    (reply("x", finish="content_filter"), "finish_reason"),
    (reply("ok", reasoning="let me think"), "thinking_not_disabled"),
    (reply("   "), "empty_reply"),
])
def test_bad_replies_are_never_used_and_keep_their_cost(tmp_path, response, kind):
    scenario = resolve_task(DEV_TASK, split_version=SPLIT).scenario
    chat, sent, ledger = chat_for(tmp_path, "simulator", lambda r, n: response)
    user = LedgeredUserSimulator(scenario, chat=chat)
    state = user.get_init_state()
    with pytest.raises(UserSimContractError) as raised:
        user.generate_next_message(AssistantMessage(content="Hi"), state)
    assert raised.value.kind == kind
    assert [m.role for m in state.messages] == ["assistant"]  # the cut reply was not appended
    settles = [json.loads(l) for l in ledger.path.read_text().splitlines() if '"settle"' in l]
    assert len(settles) == 1 and settles[0]["usage_reported"] is True and settles[0]["amount_usd"] > 0  # cost kept


def test_truncated_simulator_reply_is_a_user_sim_failure_in_the_env(tmp_path):
    scenario = resolve_task(DEV_TASK, split_version=SPLIT)
    chat, sent, ledger = chat_for(tmp_path, "simulator", lambda r, n: reply("Hello, I need help.") if n == 1 else reply("cut", finish="length"))
    env = ItsmEnv(ItsmEnvConfig(task_id=DEV_TASK, text_match=TextMatchConfig(strategy="fuzzy"), max_steps=40, split_version=SPLIT),
                  user=LedgeredUserSimulator(scenario.scenario, chat=chat))
    env.reset()
    with pytest.raises(ItsmInfrastructureError) as raised:
        env.step(ItsmAction.message("Which incident is it?"))
    assert raised.value.stage == "user_sim" and "finish_reason_length" in str(raised.value)


# -- D7 auxiliary evaluation ---------------------------------------------------------------------


def scored_dev_env(paraphrase: bool):
    task = resolve_task(DEV_TASK, split_version=SPLIT)
    env = ItsmEnv(ItsmEnvConfig(task_id=DEV_TASK, text_match=TextMatchConfig(strategy="fuzzy"), max_steps=40, split_version=SPLIT),
                  user=ScriptedUser(["Hi, please reassign INC-000008 to Gregory Richards.", "Thanks, that is all."]))
    env.reset()
    for action in task.evaluation_criteria.actions:
        arguments = copy.deepcopy(action.arguments)
        if paraphrase and "worknotes" in arguments:
            arguments["worknotes"] = arguments["worknotes"].replace("Reassigned to", "Ticket reassigned to")
        transition = env.step(ItsmAction.tool_call(action.name, arguments))
    env.step(ItsmAction.message("The incident was reassigned and the caller notified."))
    transition = env.step(ItsmAction.done())
    assert transition.reward_info is not None
    return env, transition.reward_info


def judge_handler(*, fail_first=0, protocol_error=False, nl_met=True, text_equivalent=True):
    def handler(request, n):
        if n <= fail_first:
            return httpx.Response(500, json={"error": "upstream"})
        body = json.loads(request.content)
        system, user = body["messages"][0]["content"], body["messages"][1]["content"]
        if protocol_error:
            return reply("not json", model="deepseek-v4-pro")
        if system == text_match_strategy.JUDGE_SYSTEM_PROMPT:
            pairs = json.loads(user.split("pairs:\n", 1)[1])
            return reply(json.dumps({"results": [{"index": p["index"], "equivalent": text_equivalent} for p in pairs]}), model="deepseek-v4-pro")
        assert system == evaluator_nl.JUDGE_SYSTEM_PROMPT
        outcomes = json.loads(user.split("expectedOutcomes:\n", 1)[1])
        return reply(json.dumps({"results": [{"expectedOutcome": o, "metExpectation": nl_met, "reasoning": "r"} for o in outcomes]}), model="deepseek-v4-pro")
    return handler


def test_prompt_pins_match_bench2():
    assert _sha(text_match_strategy.JUDGE_SYSTEM_PROMPT) == TEXT_PROMPT_SHA256 and _sha(evaluator_nl.JUDGE_SYSTEM_PROMPT) == NL_PROMPT_SHA256


def test_aux_results_are_separate_from_the_fuzzy_reward(tmp_path):
    env, reward = scored_dev_env(paraphrase=True)
    chat, sent, ledger = chat_for(tmp_path, "judge", judge_handler())
    result = AuxEvaluator(chat=chat)(env)
    assert result["db_llm"]["status"] == "evaluated" and result["db_llm"]["pending_pairs"] >= 1 and result["db_llm"]["db_match"] is True
    assert result["nl"]["status"] == "evaluated" and result["nl"]["nl_met"] is True and len(result["nl"]["checks"]) == 4
    assert len(sent) == 2  # one DB-text call, one NL call
    assert env.export_trace()["reward_info"]["db_reward"] == reward.db_reward  # fuzzy result untouched
    assert sent[0]["model"] == "deepseek-v4-pro" and sent[0]["max_tokens"] == 4096 and sent[0]["thinking"] == {"type": "disabled"}


def test_judge_failure_rejudges_the_same_inputs_only(tmp_path):
    env, _ = scored_dev_env(paraphrase=True)
    trajectory_before = len(env.trajectory)
    chat, sent, ledger = chat_for(tmp_path, "judge", judge_handler(fail_first=1))
    result = AuxEvaluator(chat=chat)(env)
    assert [a.get("kind") for a in result["db_llm"]["attempts"]] == ["http_status", None]
    assert sent[0] == sent[1]  # the retry re-judged byte-identical inputs (same trajectory, same final DB)
    assert len(env.trajectory) == trajectory_before  # nothing was re-run


def test_valid_negative_is_final_and_unevaluated_is_never_zero(tmp_path):
    env, _ = scored_dev_env(paraphrase=True)
    chat, sent, ledger = chat_for(tmp_path, "judge", judge_handler(nl_met=False, text_equivalent=False))
    result = AuxEvaluator(chat=chat)(env)
    assert len(sent) == 2  # negative verdicts are not retried
    assert result["db_llm"]["db_match"] is False and result["nl"]["nl_met"] is False

    chat, sent, ledger = chat_for(tmp_path, "judge", judge_handler(protocol_error=True), name="l2.jsonl")
    result = AuxEvaluator(chat=chat)(env)
    assert len(sent) == 8  # 4 attempts for each part, then stop
    assert result["db_llm"]["status"] == "not_evaluated" and result["db_llm"]["db_match"] is None
    assert result["nl"]["status"] == "not_evaluated" and result["nl"]["nl_met"] is None


def test_nl_parser_is_strict():
    assertions = ["A happened.", "B happened."]
    good = json.dumps({"results": [{"expectedOutcome": a, "metExpectation": True} for a in assertions]})
    assert [c["met"] for c in parse_nl_response(good, assertions)] == [True, True]
    for bad in ["nonsense", json.dumps({"results": [{"expectedOutcome": "A happened.", "metExpectation": True}]}),
                json.dumps({"results": [{"expectedOutcome": a, "metExpectation": "yes"} for a in assertions]}),
                json.dumps({"results": [{"expectedOutcome": "other", "metExpectation": True}, {"expectedOutcome": "B happened.", "metExpectation": True}]})]:
        with pytest.raises(ValueError):
            parse_nl_response(bad, assertions)


def test_auxiliary_inputs_require_a_scored_episode():
    env = ItsmEnv(ItsmEnvConfig(task_id=DEV_TASK, text_match=TextMatchConfig(strategy="fuzzy"), max_steps=40, split_version=SPLIT),
                  user=ScriptedUser(["Hi.", "Bye."]))
    env.reset()
    with pytest.raises(RuntimeError, match="scored"):
        env.auxiliary_inputs()
