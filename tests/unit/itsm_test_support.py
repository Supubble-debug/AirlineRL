"""Deterministic user stand-in and fault injectors for the ITSM tests.

Kept in a module of its own (not a test file) so a spawned ``ItsmEnvWorker``
child can import these by name: ``functools.partial`` of them pickles.
"""

from __future__ import annotations

import functools
import hashlib
import os
import time
from pathlib import Path

from enterprise_worlds.data_model.message import UserMessage
from enterprise_worlds.domains.itsm import environment as itsm_environment
from enterprise_worlds.domains.itsm.tools import ItsmTools
from enterprise_worlds.user.base import STOP


def seed_file_digests() -> dict[str, str]:
    """sha256 of every file in the ITSM data directory (seeds, tasks, policy)."""
    root = itsm_environment.ITSM_TASKS_PATH.parent
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.iterdir())
        if path.is_file()
    }


class ScriptedUser:
    """Deterministic user stand-in: replies from a fixed list, records what it was sent."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.received: list[str | None] = []

    def get_init_state(self) -> int:
        return 0

    def generate_next_message(self, agent_message, state: int):
        self.received.append(agent_message.content)
        return UserMessage(content=self.replies[state]), state + 1

    @staticmethod
    def is_stop(message) -> bool:
        return STOP in (message.content or "")


def _patch_tool(tool_name: str, before) -> None:
    original = getattr(ItsmTools, tool_name)

    @functools.wraps(original)
    def patched(self, *args, **kwargs):
        before()
        return original(self, *args, **kwargs)

    setattr(ItsmTools, tool_name, patched)


def install_slow_tool(tool_name: str, seconds: float, marker_path: str) -> None:
    """The tool sleeps, then leaves a marker file, then runs normally.

    A marker that never appears proves the call was stopped, not just abandoned.
    """

    def slow() -> None:
        time.sleep(seconds)
        Path(marker_path).write_text(str(os.getpid()), encoding="utf-8")

    _patch_tool(tool_name, slow)


def install_crashing_tool(tool_name: str, exit_code: int) -> None:
    """The tool kills its own process, as a native crash would."""
    _patch_tool(tool_name, lambda: os._exit(exit_code))
