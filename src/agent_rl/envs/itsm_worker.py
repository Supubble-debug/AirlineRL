"""Process-isolated ITSM episode with bounded waits (protocol §8.1; step 5).

``ItsmEnv`` runs synchronously and cannot interrupt a tool, user-simulator or
scoring call already in progress; a thread-based wait that gives up would leave
that call running against the same DB. ``ItsmEnvWorker`` therefore hosts one
``ItsmEnv`` in its own child process. Every call waits at most its deadline; on
timeout, cancellation or a crash the child is killed and joined, and the
failure reports ``execution_stopped`` only after the process is confirmed gone.
A killed child takes its in-memory DB with it, so it cannot touch any later
episode, and the episode is failed (no reward, no re-scoring).
"""

from __future__ import annotations

import multiprocessing
import os
import pickle
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

from agent_rl.envs.itsm_env import ItsmAction, ItsmEnv, ItsmEnvConfig, UserModel
from agent_rl.envs.itsm_scoring import ItsmInfrastructureError


FORWARDED_METHODS = frozenset({"reset", "step", "truncate", "score", "export_trace", "close"})
POLL_INTERVAL_S = 0.02


def _portable(error: BaseException) -> BaseException:
    """The exception itself if it pickles, else a RuntimeError carrying its text."""
    try:
        pickle.dumps(error)
    except Exception:
        return RuntimeError(f"{type(error).__name__}: {error}")
    return error


def _worker_main(
    conn: Any,
    config: ItsmEnvConfig,
    user_factory: Callable[[], UserModel] | None,
    initializer: Callable[[], None] | None,
    episode_id: str,
) -> None:
    try:
        if initializer is not None:
            initializer()
        user = user_factory() if user_factory is not None else None
        env = ItsmEnv(config, user=user, episode_id=episode_id)
    except BaseException as error:
        conn.send(("startup_error", _portable(error), None))
        return
    conn.send(("ready", os.getpid(), env.status))
    while True:
        try:
            method, args = conn.recv()
        except EOFError:
            return
        if method == "shutdown":
            env.close()
            conn.send(("ok", None, env.status))
            return
        try:
            value = getattr(env, method)(*args)
        except BaseException as error:
            conn.send(("error", _portable(error), env.status))
        else:
            conn.send(("ok", value, env.status))


class ItsmEnvWorker:
    """One ``ItsmEnv`` episode in a child process, with killable bounded calls.

    ``user_factory`` and ``initializer`` run in the child and must be picklable
    (module-level functions or ``functools.partial`` of them).
    """

    def __init__(
        self,
        config: ItsmEnvConfig,
        *,
        call_timeout_s: float,
        episode_timeout_s: float | None = None,
        user_factory: Callable[[], UserModel] | None = None,
        initializer: Callable[[], None] | None = None,
        episode_id: str | None = None,
        start_method: str = "spawn",
        startup_timeout_s: float = 120.0,
        kill_grace_s: float = 10.0,
    ) -> None:
        if call_timeout_s <= 0:
            raise ValueError("call_timeout_s must be positive")
        if episode_timeout_s is not None and episode_timeout_s <= 0:
            raise ValueError("episode_timeout_s must be positive")
        self.config = config
        self.episode_id = episode_id or f"{config.task_id}:{uuid.uuid4().hex[:8]}"
        self.call_timeout_s = call_timeout_s
        self.episode_timeout_s = episode_timeout_s
        self.kill_grace_s = kill_grace_s
        self._call_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._cancel = threading.Event()
        self._episode_deadline: float | None = None
        self._failure: dict[str, Any] | None = None
        self._closed = False
        self._final_trace: dict[str, Any] | None = None
        self._child_status: str | None = None

        context = multiprocessing.get_context(start_method)
        self._conn, child_conn = context.Pipe(duplex=True)
        self._process = context.Process(
            target=_worker_main,
            args=(child_conn, config, user_factory, initializer, self.episode_id),
            daemon=True,
            name=f"itsm-episode-{self.episode_id}",
        )
        self._process.start()
        child_conn.close()
        kind, payload, status = self._receive("startup", time.monotonic() + startup_timeout_s)
        if kind == "startup_error":
            self._stop_process()
            raise payload
        self.pid: int = payload
        self._child_status = status

    # -- forwarded episode API -------------------------------------------------

    def reset(self):
        self._episode_deadline = (
            time.monotonic() + self.episode_timeout_s if self.episode_timeout_s else None
        )
        return self._call("reset")

    def step(self, action: ItsmAction):
        return self._call("step", action)

    def truncate(self, reason: str):
        return self._call("truncate", reason)

    def score(self):
        return self._call("score")

    def export_trace(self) -> dict[str, Any]:
        if self._closed and self._final_trace is not None:
            return self._final_trace
        if self._failure is not None and self._failure.get("execution_stopped") is not None:
            raise RuntimeError("the trace was lost: the worker process was stopped")
        return self._call("export_trace")

    def cancel(self) -> None:
        """Stop the episode now, from any thread; an in-flight call raises ``cancelled``."""
        self._cancel.set()
        with self._state_lock:
            idle = not self._call_lock.locked()
        if idle and self._failure is None and not self._closed:
            self._terminate("cancelled", "cancelled while idle", raise_error=False)

    def close(self) -> None:
        """Idempotent. Keeps the final trace when the child is still healthy."""
        if self._closed:
            return
        if self._failure is None and self._process.is_alive():
            try:
                self._final_trace = self._call("export_trace")
                with self._call_lock:
                    self._conn.send(("shutdown", ()))
                    self._receive("shutdown", time.monotonic() + self.call_timeout_s)
            except Exception:
                pass
        self._closed = True
        self._stop_process()

    @property
    def status(self) -> str | None:
        """The child ``ItsmEnv`` status, or ``failed`` once the worker itself failed."""
        return "failed" if self._failure is not None else self._child_status

    @property
    def failure(self) -> dict[str, Any] | None:
        return dict(self._failure) if self._failure is not None else None

    @property
    def is_alive(self) -> bool:
        return self._process.is_alive()

    # -- internals ---------------------------------------------------------------

    def _call(self, method: str, *args: Any) -> Any:
        if method not in FORWARDED_METHODS:
            raise ValueError(f"unknown method {method!r}")
        with self._call_lock:
            if self._closed:
                raise RuntimeError("the worker is closed")
            if self._failure is not None:
                raise RuntimeError(f"the worker failed ({self._failure['stage']})")
            deadline = time.monotonic() + self.call_timeout_s
            if self._episode_deadline is not None:
                deadline = min(deadline, self._episode_deadline)
            try:
                self._conn.send((method, args))
            except (OSError, EOFError) as error:
                # The child exited while idle (BrokenPipeError and friends). Checking
                # is_alive() first would still race with its exit, so the failed send
                # itself is the signal; a send that slips in before the exit is caught
                # by _receive instead.
                self._terminate("worker_crashed", f"{method}: could not send to child", cause=error)
            kind, payload, status = self._receive(method, deadline)
            self._child_status = status
        if kind == "error":
            raise payload
        return payload

    def _receive(self, method: str, deadline: float) -> tuple[str, Any, Any]:
        while True:
            if self._cancel.is_set():
                self._terminate("cancelled", f"{method} cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._terminate("timeout", f"{method} exceeded its deadline")
            try:
                if self._conn.poll(min(POLL_INTERVAL_S, remaining)):
                    return self._conn.recv()
            except (EOFError, OSError):
                pass
            else:
                if self._process.is_alive():
                    continue
                try:
                    if self._conn.poll(0):
                        return self._conn.recv()
                except (EOFError, OSError):
                    pass
            if self._cancel.is_set():
                self._terminate("cancelled", f"{method} cancelled")
            self._terminate(
                "worker_crashed", f"{method}: child exited with code {self._process.exitcode}"
            )

    def _terminate(
        self,
        stage: str,
        detail: str,
        *,
        raise_error: bool = True,
        cause: BaseException | None = None,
    ) -> None:
        stopped = self._stop_process()
        with self._state_lock:
            if self._failure is None:
                self._failure = {
                    "stage": stage,
                    "detail": detail,
                    "execution_stopped": stopped,
                    "exitcode": self._process.exitcode,
                }
                if cause is not None:
                    self._failure["cause"] = f"{type(cause).__name__}: {cause}"
            failure = dict(self._failure)
        if raise_error:
            raise ItsmInfrastructureError(
                failure["stage"],
                self.config.task_id,
                f"episode {self.episode_id}: {failure['detail']}",
                kind="killed" if failure["execution_stopped"] else "not_reclaimed",
            ) from cause

    def _stop_process(self) -> bool:
        """Kill and join the child; ``True`` only if it is confirmed to have exited."""
        with self._state_lock:
            if self._process.is_alive():
                self._process.kill()
            self._process.join(self.kill_grace_s)
            stopped = not self._process.is_alive()
            if stopped:
                try:
                    self._conn.close()
                except OSError:
                    pass
            return stopped
