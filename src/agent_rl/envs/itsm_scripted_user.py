"""Deterministic ITSM user stand-in for engineering runs (step 7B), and an LLM guard.

``PersonaScriptedUser`` replaces Bench2's LLM user simulator when no external LLM
may be called. It says only what the task persona would disclose about
itself -- name, e-mail, role description and the identifiers in
``persona.known_info`` -- and never the simulator-only ``task_description``. It
does not tell the agent the goal, so task success is NOT expected and its
episodes measure the engineering path (real policy outputs, tool calls,
observations, lengths, log-probs), not task ability.

Reply order (fixed): the opening message, then ``FOLLOW_UPS`` one per agent
message, then a closing reply containing Bench2's STOP token, which ends the
episode as a user stop.

``forbid_external_llm`` is a worker initializer: it replaces ``litellm`` in the
child process so any attempted LLM call raises instead of reaching a provider.
"""

from __future__ import annotations

import sys
import types

from enterprise_worlds.data_model.message import UserMessage
from enterprise_worlds.user.base import STOP

FOLLOW_UPS = (
    "I'm not sure about the exact steps. Please look up the records I mentioned and take care of what they need.",
    "I don't have any more details than that. Please go ahead with what you can find in the system.",
    "Please continue with what you think is right.",
)
CLOSING = f"Thanks, that is all for now. {STOP}"
SCRIPT_VERSION = "persona_scripted_user_v1"


def opening_message(task) -> str:
    persona = task.scenario.persona
    identity = persona.identity
    details = "; ".join(f"{key.replace('_', ' ')}: {value}" for key, value in sorted(persona.known_info.items()))
    who = f"{identity.first_name} {identity.last_name} ({identity.email})"
    text = f"Hi, this is {who}, {persona.role_description}. I need help with a request in our IT service management system."
    return f"{text} The details I have: {details}." if details else text


class PersonaScriptedUser:
    """Bench2 ``UserModel`` interface: get_init_state / generate_next_message / is_stop."""

    def __init__(self, task_id: str, split_version: str) -> None:
        from agent_rl.data.itsm_task_sets import resolve_task

        task = resolve_task(task_id, split_version=split_version)
        self.replies = [opening_message(task), *FOLLOW_UPS, CLOSING]
        self.received: list[str | None] = []

    def get_init_state(self) -> int:
        return 0

    def generate_next_message(self, agent_message, state: int):
        self.received.append(getattr(agent_message, "content", None))
        return UserMessage(content=self.replies[min(state, len(self.replies) - 1)]), state + 1

    @staticmethod
    def is_stop(message) -> bool:
        return STOP in (message.content or "")


class ExternalLlmForbidden(RuntimeError):
    pass


def _forbidden(*args, **kwargs):
    raise ExternalLlmForbidden("external LLM calls are forbidden in this run (step 7B: scripted user only)")


def _stub_getattr(name: str):
    if name.startswith("__"):
        raise AttributeError(name)  # module introspection (__file__, __path__, ...) behaves normally
    return _forbidden


def forbid_external_llm() -> None:
    """Install a ``litellm`` stub whose every call raises (run in the child and the parent)."""
    stub = types.ModuleType("litellm")
    stub.completion = _forbidden
    stub.acompletion = _forbidden
    stub.__getattr__ = _stub_getattr  # any other public entry point also raises
    sys.modules["litellm"] = stub
