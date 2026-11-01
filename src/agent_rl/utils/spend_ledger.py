"""Reserve-before-request spend ledger for paid API calls (step 7A probe, S1).

Every possibly billed request reserves its worst-case cost *before* it is sent;
a request that cannot be reserved is not sent. After the request the
reservation is settled with the provider's reported usage, or charged in full
when no usage comes back (timeouts, errors). The ledger is an append-only JSONL
file, so a crash leaves unsettled reservations that count in full on reload.
Stopping depends only on the ledger, never on a later bill.

Prices are inputs (USD per million tokens, plus an optional per-request fee);
the ledger never assumes a price.

Ownership and concurrency (7A-R1): a ledger file has exactly one writer. The
constructor takes an exclusive, non-blocking ``flock`` on ``<path>.lock`` and
holds it until ``close()`` (or process exit); a second ``SpendLedger`` on the
same path -- in this process or another one on the same host -- is refused with
``LedgerBusy`` instead of replaying a stale snapshot and over-reserving. The
owner replays the file once at open and is then the only one appending, so its
in-memory totals equal the file. Within the owner, a thread lock serialises
reserve/settle. The lock is host-local: a ledger must be written on one host
only (a synced copy elsewhere is a read-only record), and readers that only
need totals use ``SpendLedger.read_summary`` without taking ownership.

Group envelopes (step 8): ``open_envelope(group, amount)`` reserves a whole group's worst
case BEFORE the group starts (refused -> the group is not started). Every request of the
group is then reserved against what is left of its envelope (``reserve(..., envelope=)``),
checked request by request (a request that does not fit is first covered by
``extend_envelope`` from the cap; refused -> the request is not sent), and settled with its actual cost (full reservation without
usage). ``close_envelope`` charges any still-open request of the group in full (abandoned:
it may have been billed) and releases the unused rest. Committed spend =
settled + open reservations + the unused rest of every open envelope, so the cap holds
across concurrent groups. ``recover_unfinished`` (new owner after a crash) charges open
reservations in full and closes open envelopes.
"""

from __future__ import annotations

import fcntl
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class BudgetExceeded(RuntimeError):
    """A reservation would exceed the cap or the request-count limit; the request is not sent."""


class ReservationTooSmall(RuntimeError):
    """Actual cost exceeded the reservation: the bound is wrong, stop sending."""


class RunHalted(BudgetExceeded):
    """A non-retryable fault halted the ledger: nothing new may be reserved, extended or opened."""


class LedgerBusy(RuntimeError):
    """Another writer already owns this ledger file; nothing is reserved."""


@dataclass(frozen=True, slots=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float
    per_request_fee: float = 0.0
    source: str = ""

    def __post_init__(self) -> None:
        if min(self.input_per_mtok, self.output_per_mtok, self.per_request_fee) < 0:
            raise ValueError("prices must be non-negative")
        if not self.source:
            raise ValueError("a price needs its verified source")

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return input_tokens * self.input_per_mtok / 1e6 + output_tokens * self.output_per_mtok / 1e6 + self.per_request_fee


class _LedgerState:
    """Event replay shared by the owner and the read-only summary."""

    def __init__(self) -> None:
        self.open: dict[str, float] = {}
        self.request_envelope: dict[str, str | None] = {}
        self.envelopes: dict[str, dict[str, Any]] = {}
        self.settled = 0.0
        self.requests = 0
        self.halted: str | None = None

    def apply(self, event: dict[str, Any]) -> None:
        kind = event["event"]
        if kind == "reserve":
            self.open[event["request_id"]] = event["amount_usd"]
            self.request_envelope[event["request_id"]] = event.get("envelope")
            self.requests += 1
            if event.get("envelope") is not None:
                self.envelopes[event["envelope"]]["used"] += event["amount_usd"]
        elif kind == "settle":
            reserved = self.open.pop(event["request_id"], 0.0)
            self.settled += event["amount_usd"]
            envelope = self.request_envelope.get(event["request_id"])
            if envelope is not None:
                self.envelopes[envelope]["used"] += event["amount_usd"] - reserved
        elif kind == "envelope_open":
            self.envelopes[event["envelope"]] = {"amount": event["amount_usd"], "used": 0.0, "open": True}
        elif kind == "envelope_extend":
            self.envelopes[event["envelope"]]["amount"] += event["amount_usd"]
        elif kind == "halt":
            self.halted = self.halted or event["reason"]
        elif kind == "envelope_close":
            self.envelopes[event["envelope"]]["open"] = False

    def envelope_remaining(self, envelope: str) -> float:
        entry = self.envelopes[envelope]
        return max(0.0, entry["amount"] - entry["used"]) if entry["open"] else 0.0

    @property
    def committed(self) -> float:
        return self.settled + sum(self.open.values()) + sum(self.envelope_remaining(e) for e in self.envelopes)

    def open_requests_of(self, envelope: str) -> list[str]:
        return [r for r in self.open if self.request_envelope.get(r) == envelope]


class SpendLedger:
    def __init__(self, path: Path, *, cap_usd: float, max_billed_requests: int, budget_id: str) -> None:
        if cap_usd <= 0 or max_billed_requests < 1:
            raise ValueError("cap and request limit must be positive")
        self.path = Path(path)
        self.cap_usd = cap_usd
        self.max_billed_requests = max_billed_requests
        self.budget_id = budget_id
        self._lock = threading.Lock()
        self._state = _LedgerState()
        self._owner_fd: int | None = None
        self._acquire_ownership()
        try:
            if self.path.exists():
                self._replay()
        except BaseException:
            self.close()
            raise

    # -- ownership -------------------------------------------------------------------------------

    @property
    def lock_path(self) -> Path:
        return self.path.with_name(self.path.name + ".lock")

    def _acquire_ownership(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise LedgerBusy(f"ledger {self.path} already has a writer (lock {self.lock_path}); one owner per ledger") from None
        self._owner_fd = fd

    def close(self) -> None:
        """Release ownership; the object can no longer reserve or settle."""
        with self._lock:
            if self._owner_fd is not None:
                fcntl.flock(self._owner_fd, fcntl.LOCK_UN)
                os.close(self._owner_fd)
                self._owner_fd = None

    def __enter__(self) -> "SpendLedger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _require_owner(self) -> None:
        if self._owner_fd is None:
            raise LedgerBusy(f"ledger {self.path} was closed; open a new SpendLedger to write")

    @staticmethod
    def read_summary(path: Path, budget_id: str) -> dict[str, Any]:
        """Totals from the file without taking ownership (reports, audits)."""
        state = _LedgerState()
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event["budget_id"] != budget_id:
                raise ValueError(f"ledger {path} belongs to budget {event['budget_id']!r}")
            state.apply(event)
        return {"budget_id": budget_id, "billed_requests": state.requests, "settled_usd": round(state.settled, 6),
                "open_reservations_usd": round(sum(state.open.values()), 6),
                "open_envelopes_unused_usd": round(sum(state.envelope_remaining(e) for e in state.envelopes), 6)}

    # -- state -----------------------------------------------------------------------------------

    def _replay(self) -> None:
        for line in self.path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event["budget_id"] != self.budget_id:
                raise ValueError(f"ledger {self.path} belongs to budget {event['budget_id']!r}")
            self._state.apply(event)

    def _append(self, event: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"budget_id": self.budget_id, "time": time.time(), **event}, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._state.apply(event)

    @property
    def committed_usd(self) -> float:
        """Settled spend + every open reservation (in full) + the unused rest of open envelopes."""
        return self._state.committed

    @property
    def billed_requests(self) -> int:
        return self._state.requests

    def envelope_remaining(self, envelope: str) -> float:
        with self._lock:
            return self._state.envelope_remaining(envelope)

    def open_envelopes(self) -> list[str]:
        with self._lock:
            return [e for e, entry in self._state.envelopes.items() if entry["open"]]

    # -- operations ------------------------------------------------------------------------------

    @property
    def halt_reason(self) -> str | None:
        return self._state.halted

    def _refuse_if_halted(self) -> None:
        if self._state.halted is not None:
            raise RunHalted(f"ledger halted: {self._state.halted}")

    def halt(self, reason: str) -> None:
        """Non-retryable fault (model mismatch, actual above reservation): stop all new spending.

        Recorded in the file, so a restarted owner stays halted; settling in-flight requests is
        still allowed so their costs are kept.
        """
        with self._lock:
            self._require_owner()
            if self._state.halted is None:
                self._append({"event": "halt", "reason": reason})

    def open_envelope(self, envelope: str, amount_usd: float, detail: dict[str, Any]) -> None:
        """Reserve a group's worst case before it starts; ``BudgetExceeded`` = do not start it."""
        with self._lock:
            self._require_owner()
            self._refuse_if_halted()
            if envelope in self._state.envelopes:
                raise ValueError(f"envelope {envelope} already exists")
            if amount_usd <= 0:
                raise ValueError("an envelope must be positive")
            if self._state.committed + amount_usd > self.cap_usd + 1e-12:
                raise BudgetExceeded(f"envelope {envelope} ({amount_usd:.6f} USD) would exceed the {self.cap_usd} USD cap (committed {self._state.committed:.6f})")
            self._append({"event": "envelope_open", "envelope": envelope, "amount_usd": amount_usd, "detail": detail})

    def extend_envelope(self, envelope: str, amount_usd: float, detail: dict[str, Any]) -> None:
        """Top up an open envelope from the cap BEFORE a request that would not fit; refused = not sent."""
        with self._lock:
            self._require_owner()
            self._refuse_if_halted()
            entry = self._state.envelopes.get(envelope)
            if entry is None or not entry["open"]:
                raise BudgetExceeded(f"envelope {envelope} is not open")
            if amount_usd <= 0:
                raise ValueError("an extension must be positive")
            if self._state.committed + amount_usd > self.cap_usd + 1e-12:
                raise BudgetExceeded(f"extending envelope {envelope} by {amount_usd:.6f} USD would exceed the {self.cap_usd} USD cap (committed {self._state.committed:.6f})")
            self._append({"event": "envelope_extend", "envelope": envelope, "amount_usd": amount_usd, "detail": detail})

    def reserve(self, request_id: str, amount_usd: float, detail: dict[str, Any], *, envelope: str | None = None) -> None:
        with self._lock:
            self._require_owner()
            self._refuse_if_halted()
            if request_id in self._state.open or request_id in self._state.request_envelope:
                raise ValueError(f"request {request_id} already reserved")
            if self._state.requests + 1 > self.max_billed_requests:
                raise BudgetExceeded(f"request limit {self.max_billed_requests} reached (retries count)")
            if envelope is not None:
                entry = self._state.envelopes.get(envelope)
                if entry is None or not entry["open"]:
                    raise BudgetExceeded(f"envelope {envelope} is not open")
                remaining = self._state.envelope_remaining(envelope)
                if amount_usd > remaining + 1e-12:
                    raise BudgetExceeded(f"reserving {amount_usd:.6f} USD exceeds the {remaining:.6f} USD left in envelope {envelope}")
            elif self._state.committed + amount_usd > self.cap_usd + 1e-12:
                raise BudgetExceeded(f"reserving {amount_usd:.6f} USD would exceed the {self.cap_usd} USD cap (committed {self._state.committed:.6f})")
            self._append({"event": "reserve", "request_id": request_id, "amount_usd": amount_usd, "detail": detail, "envelope": envelope})

    def settle(self, request_id: str, *, actual_usd: float | None, usage: dict[str, Any] | None, note: str | None = None) -> float:
        """Settle with the reported cost; without usage the full reservation is charged."""
        with self._lock:
            self._require_owner()
            if request_id not in self._state.open:
                raise KeyError(f"request {request_id} is not open")
            reserved = self._state.open[request_id]
            charged = reserved if actual_usd is None else actual_usd
            event = {"event": "settle", "request_id": request_id, "amount_usd": charged, "reserved_usd": reserved, "usage": usage,
                     "usage_reported": actual_usd is not None}
            if note:
                event["note"] = note
            self._append(event)
            over = actual_usd is not None and actual_usd > reserved + 1e-12
            if over and self._state.halted is None:
                # The bound was wrong: nothing else may be sent (non-retryable, survives a restart).
                self._append({"event": "halt", "reason": f"actual above reservation: request {request_id} {actual_usd:.6f} > {reserved:.6f} USD"})
        if over:
            raise ReservationTooSmall(f"request {request_id}: actual {actual_usd:.6f} > reserved {reserved:.6f} USD")
        return charged

    def close_envelope(self, envelope: str, *, reason: str) -> dict[str, Any]:
        """Charge the group's still-open requests in full (they may have been billed), release the rest."""
        with self._lock:
            self._require_owner()
            entry = self._state.envelopes.get(envelope)
            if entry is None:
                raise KeyError(f"no envelope {envelope}")
            if not entry["open"]:
                return {"envelope": envelope, "already_closed": True}
            abandoned = self._state.open_requests_of(envelope)
            for request_id in abandoned:
                reserved = self._state.open[request_id]
                self._append({"event": "settle", "request_id": request_id, "amount_usd": reserved, "reserved_usd": reserved, "usage": None,
                              "usage_reported": False, "note": f"abandoned at envelope close ({reason})"})
            used = entry["used"]
            self._append({"event": "envelope_close", "envelope": envelope, "reason": reason, "used_usd": used,
                          "released_usd": max(0.0, entry["amount"] - used)})
            return {"envelope": envelope, "used_usd": used, "abandoned_requests": abandoned}

    def recover_unfinished(self) -> dict[str, Any]:
        """New owner after a crash: open requests are charged in full, open envelopes are closed."""
        with self._lock:
            self._require_owner()
            open_requests = list(self._state.open)
            open_envelopes = [e for e, entry in self._state.envelopes.items() if entry["open"]]
        for request_id in open_requests:
            if request_id in self._state.open and self._state.request_envelope.get(request_id) is None:
                self.settle(request_id, actual_usd=None, usage=None, note="recovered after restart: charged in full")
        for envelope in open_envelopes:
            self.close_envelope(envelope, reason="recovered after restart")
        return {"charged_requests": open_requests, "closed_envelopes": open_envelopes}

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "budget_id": self.budget_id,
                "cap_usd": self.cap_usd,
                "max_billed_requests": self.max_billed_requests,
                "billed_requests": self._state.requests,
                "settled_usd": round(self._state.settled, 6),
                "open_reservations_usd": round(sum(self._state.open.values()), 6),
                "open_envelopes_unused_usd": round(sum(self._state.envelope_remaining(e) for e in self._state.envelopes), 6),
                "committed_usd": round(self._state.committed, 6),
                "halted": self._state.halted,
            }
