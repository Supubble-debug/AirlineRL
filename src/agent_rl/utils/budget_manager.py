"""One budget manager for every possibly billed request of a run (step 8).

The manager lives in the runner's process and is the only owner of the ``SpendLedger``.
Worker children (``ItsmEnvWorker``: user simulator, auxiliary judges) never touch the ledger;
they ask the manager over an authenticated Unix socket (``BudgetServer`` / ``BudgetClient``)
to reserve before each request and to settle after it. If the manager cannot be reached the
child gets ``BudgetUnavailable`` and sends nothing (fail closed).

Groups:
* ``open_group`` reserves the group's planned envelope before the group starts; refused ->
  the group is not started (``BudgetExceeded``) and the runner opens no further group.
* ``reserve`` prices one request from its input-token bound and output cap and reserves it
  inside the group's envelope; a shortfall is topped up from the cap first; refused -> the
  request is not sent (``BudgetExceeded``).
* ``settle`` books the provider-reported usage; ``settle_unknown`` (timeout, transport error,
  missing usage, cancelled in flight) books the full reservation.
* ``close_group`` books every still-open request of the group in full and releases the rest.
* ``halt`` (non-retryable: returned model differs, actual cost above its reservation) refuses
  every later reservation, top-up and group; in-flight requests can still be settled.
"""

from __future__ import annotations

import os
import secrets
import threading
import uuid
from multiprocessing import AuthenticationError
from multiprocessing.connection import Client, Listener
from pathlib import Path
from typing import Any

from agent_rl.utils.spend_ledger import BudgetExceeded, Price, ReservationTooSmall, RunHalted, SpendLedger


class BudgetUnavailable(RuntimeError):
    """The budget manager could not be reached; the request must not be sent."""


class BudgetManager:
    def __init__(self, ledger: SpendLedger, prices: dict[str, Price]) -> None:
        self.ledger = ledger
        self.prices = prices
        self._lock = threading.Lock()
        self._request_role: dict[str, str] = {}
        self._request_group: dict[str, str] = {}
        #: refusals per group: a worker's budget refusal surfaces as a user_sim failure inside the
        #: env, so the runner asks here before treating a failure as retryable infrastructure.
        self.refusals: dict[str, list[str]] = {}
        self.recovery = ledger.recover_unfinished()  # crash before us: open requests charged in full

    # -- groups ----------------------------------------------------------------------------------

    def open_group(self, group_id: str, envelope_usd: float, detail: dict[str, Any]) -> None:
        self.ledger.open_envelope(group_id, envelope_usd, detail)

    def close_group(self, group_id: str, *, reason: str) -> dict[str, Any]:
        return self.ledger.close_envelope(group_id, reason=reason)

    # -- requests --------------------------------------------------------------------------------

    def reservation_usd(self, role: str, input_bound_tokens: int, max_output_tokens: int) -> float:
        return self.prices[role].cost(input_bound_tokens, max_output_tokens)

    def reserve(self, *, group_id: str, role: str, input_bound_tokens: int, max_output_tokens: int, detail: dict[str, Any]) -> dict[str, Any]:
        if role not in self.prices:
            raise ValueError(f"no price for role {role!r}")
        amount = self.reservation_usd(role, input_bound_tokens, max_output_tokens)
        request_id = f"{group_id}/{role}/{uuid.uuid4().hex[:12]}"
        with self._lock:  # top-up and reserve as one step: no other request can take the top-up
            try:
                shortfall = amount - self.ledger.envelope_remaining(group_id)
                if shortfall > 1e-12:
                    self.ledger.extend_envelope(group_id, shortfall, {"for_request": request_id, "role": role})
                self.ledger.reserve(request_id, amount, {**detail, "role": role, "input_bound_tokens": input_bound_tokens,
                                                         "max_output_tokens": max_output_tokens, "price_source": self.prices[role].source},
                                    envelope=group_id)
            except BudgetExceeded as error:
                self.refusals.setdefault(group_id, []).append(str(error))
                raise
            self._request_role[request_id] = role
            self._request_group[request_id] = group_id
        return {"request_id": request_id, "reserved_usd": amount}

    # -- non-retryable faults ------------------------------------------------------------------

    def halt(self, *, reason: str) -> None:
        """Stop new reservations, requests and groups (model mismatch, actual above reservation)."""
        self.ledger.halt(reason)

    @property
    def halt_reason(self) -> str | None:
        return self.ledger.halt_reason

    def refused(self, group_id: str) -> list[str]:
        with self._lock:
            return list(self.refusals.get(group_id, []))

    def settle(self, *, request_id: str, input_tokens: int | None, output_tokens: int | None, usage: Any) -> float:
        role = self._request_role[request_id]
        actual = None if input_tokens is None or output_tokens is None else self.prices[role].cost(input_tokens, output_tokens)
        return self.ledger.settle(request_id, actual_usd=actual, usage=usage)

    def settle_unknown(self, *, request_id: str, note: str) -> float:
        return self.ledger.settle(request_id, actual_usd=None, usage=None, note=note)

    def summary(self) -> dict[str, Any]:
        return self.ledger.summary()


#: operations a child may call (nothing that opens/closes groups or reads other state)
CHILD_OPERATIONS = ("reserve", "settle", "settle_unknown", "halt")


class BudgetServer:
    """Serve ``BudgetManager`` operations to worker children over an authenticated Unix socket."""

    def __init__(self, manager: BudgetManager, socket_dir: Path) -> None:
        self.manager = manager
        socket_dir.mkdir(parents=True, exist_ok=True)
        self.address = str(socket_dir / f"budget-{os.getpid()}-{uuid.uuid4().hex[:8]}.sock")
        self.authkey = secrets.token_bytes(32)
        self._listener = Listener(self.address, family="AF_UNIX", authkey=self.authkey)
        os.chmod(self.address, 0o600)
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._serve, name="budget-server", daemon=True)
        self._thread.start()

    def client_config(self) -> tuple[str, bytes]:
        return self.address, self.authkey

    def _serve(self) -> None:
        while not self._closed.is_set():
            try:
                conn = self._listener.accept()
            except (OSError, EOFError, AuthenticationError):
                if self._closed.is_set():
                    return
                continue  # failed handshake (wrong key) or a dropped peer: keep serving the others
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn) -> None:
        with conn:
            while True:
                try:
                    op, kwargs = conn.recv()
                except (EOFError, OSError):
                    return
                try:
                    if op not in CHILD_OPERATIONS:
                        raise PermissionError(f"operation {op!r} is not available to workers")
                    conn.send(("ok", getattr(self.manager, op)(**kwargs)))
                except BaseException as error:  # noqa: BLE001 - returned to the child
                    conn.send(("error", type(error).__name__, str(error)))

    def close(self) -> None:
        self._closed.set()
        try:
            self._listener.close()
        except OSError:
            pass
        try:
            os.unlink(self.address)
        except FileNotFoundError:
            pass


_ERRORS = {"BudgetExceeded": BudgetExceeded, "RunHalted": RunHalted, "ReservationTooSmall": ReservationTooSmall}


class BudgetClient:
    """Child side; picklable config ``(address, authkey)``; one connection per process."""

    def __init__(self, address: str, authkey: bytes) -> None:
        self.address, self.authkey = address, authkey
        self._conn = None
        self._lock = threading.Lock()

    def _call(self, op: str, **kwargs: Any) -> Any:
        with self._lock:
            try:
                if self._conn is None:
                    self._conn = Client(self.address, family="AF_UNIX", authkey=self.authkey)
                self._conn.send((op, kwargs))
                reply = self._conn.recv()
            except (OSError, EOFError, AuthenticationError) as error:
                self._conn = None
                raise BudgetUnavailable(f"budget manager unreachable: {type(error).__name__}: {error}") from error
        if reply[0] == "ok":
            return reply[1]
        raise _ERRORS.get(reply[1], BudgetUnavailable)(f"{reply[1]}: {reply[2]}")

    def reserve(self, **kwargs: Any) -> dict[str, Any]:
        return self._call("reserve", **kwargs)

    def settle(self, **kwargs: Any) -> float:
        return self._call("settle", **kwargs)

    def settle_unknown(self, **kwargs: Any) -> float:
        return self._call("settle_unknown", **kwargs)

    def halt(self, **kwargs: Any) -> None:
        return self._call("halt", **kwargs)
