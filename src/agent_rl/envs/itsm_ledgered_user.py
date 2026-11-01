"""D5 user simulator whose every call is reserved in the run's single ledger (step 8).

Same messages as Bench2's ``UserSimulator.generate_next_message`` (system messages from its
``system_prompt`` + ``flip_roles()``, converted with ``to_litellm_messages``, ``<think>``
blocks stripped), same model and D5 arguments (``deepseek/deepseek-flash``, temperature 0,
``max_tokens`` 1024, thinking disabled); only the transport differs (``DeepSeekChat``: one
reservation per HTTP request, no SDK retries). Bench2 itself is not modified.

Contract checks on every reply, all raising (``ItsmEnv`` turns them into ``user_sim``
infrastructure failures, so the episode is retried with a fresh DB and nothing is scored):
* ``finish_reason == "length"``: the reply was cut at ``max_tokens``; it is NOT used and is
  not appended to the conversation; its cost stays booked;
* any other finish reason than ``stop``; reasoning content or reasoning tokens (thinking was
  supposed to be off); empty text; a different returned model (``DeepSeekChat``).
"""

from __future__ import annotations

import os
from typing import Any

from enterprise_worlds.data_model.message import UserMessage
from enterprise_worlds.user.user_simulator import UserSimulator
from enterprise_worlds.utils.llm_utils import _strip_think, to_litellm_messages

from agent_rl.envs.itsm_env import DEFAULT_USER_LLM, DEFAULT_USER_LLM_ARGS
from agent_rl.llm.deepseek_chat import DEEPSEEK_BASE_URL, DeepSeekChat
from agent_rl.utils.budget_manager import BudgetClient


class UserSimContractError(RuntimeError):
    def __init__(self, kind: str, detail: str) -> None:
        super().__init__(f"{kind}: {detail}")
        self.kind = kind


class LedgeredUserSimulator(UserSimulator):
    def __init__(self, scenario, *, chat: DeepSeekChat, llm: str = DEFAULT_USER_LLM, llm_args: dict[str, Any] | None = None) -> None:
        args = dict(DEFAULT_USER_LLM_ARGS if llm_args is None else llm_args)
        super().__init__(scenario, llm=llm, llm_args=args)
        self.chat = chat
        self.replies: list[dict[str, Any]] = []

    def generate_next_message(self, agent_message, state):
        if agent_message is not None:
            state.messages.append(agent_message)
        llm_messages = list(state.system_messages) + state.flip_roles()
        args = dict(self.llm_args)
        result = self.chat.complete(self.llm, to_litellm_messages(llm_messages), temperature=args.pop("temperature"),
                                    max_tokens=args.pop("max_tokens"), extra_body=args.pop("extra_body", None))
        if args:
            raise ValueError(f"unsupported D5 arguments {sorted(args)}")
        self.replies.append({"request_id": result.request_id, "finish_reason": result.finish_reason, "charged_usd": result.charged_usd})
        if result.finish_reason == "length":
            raise UserSimContractError("finish_reason_length", f"reply cut at max_tokens={self.llm_args['max_tokens']}; not used")
        if result.finish_reason != "stop":
            raise UserSimContractError("finish_reason", f"finish_reason={result.finish_reason!r}")
        if result.reasoning_present or result.reasoning_tokens:
            raise UserSimContractError("thinking_not_disabled", f"reasoning returned ({result.reasoning_tokens} tokens)")
        content = _strip_think(result.content)
        if not content or not content.strip():
            raise UserSimContractError("empty_reply", "no text")
        user_message = UserMessage(content=content)
        state.messages.append(user_message)
        return user_message, state


def make_ledgered_user(task_id: str, split_version: str, budget_address: str, budget_authkey: bytes, group_id: str,
                       timeout_s: float = 60.0) -> LedgeredUserSimulator:
    """Picklable worker ``user_factory`` (via functools.partial); runs in the env child."""
    from agent_rl.data.itsm_task_sets import resolve_task

    task = resolve_task(task_id, split_version=split_version)
    chat = DeepSeekChat(api_key=os.environ.get("DEEPSEEK_API_KEY", ""), budget=BudgetClient(budget_address, budget_authkey),
                        group_id=group_id, role="simulator", timeout_s=timeout_s, tag=f"user:{task_id}",
                        base_url=os.environ.get("ITSM_DEEPSEEK_BASE_URL", DEEPSEEK_BASE_URL))
    return LedgeredUserSimulator(task.scenario, chat=chat)
