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


class SpendLedger:
    def __init__(self, path: Path, *, cap_usd: float, max_billed_requests: int, budget_id: str) -> None:
        if cap_usd <= 0 or max_billed_requests < 1:
            raise ValueError("cap and request limit must be positive")
        self.path = Path(path)
        self.cap_usd = cap_usd
        self.max_billed_requests = max_billed_requests
        self.budget_id = budget_id
        self._lock = threading.Lock()
        self._open: dict[str, float] = {}
        self._settled = 0.0
        self._requests = 0
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
        settled, open_, requests = 0.0, {}, 0
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event["budget_id"] != budget_id:
                raise ValueError(f"ledger {path} belongs to budget {event['budget_id']!r}")
            if event["event"] == "reserve":
                open_[event["request_id"]] = event["amount_usd"]
                requests += 1
            elif event["event"] == "settle":
                open_.pop(event["request_id"], None)
                settled += event["amount_usd"]
        return {"budget_id": budget_id, "billed_requests": requests, "settled_usd": round(settled, 6), "open_reservations_usd": round(sum(open_.values()), 6)}

    # -- state -----------------------------------------------------------------------------------

    def _replay(self) -> None:
        for line in self.path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event["budget_id"] != self.budget_id:
                raise ValueError(f"ledger {self.path} belongs to budget {event['budget_id']!r}")
            if event["event"] == "reserve":
                self._open[event["request_id"]] = event["amount_usd"]
                self._requests += 1
            elif event["event"] == "settle":
                self._open.pop(event["request_id"], None)
                self._settled += event["amount_usd"]

    def _append(self, event: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"budget_id": self.budget_id, "time": time.time(), **event}, sort_keys=True) + "\n")
            handle.flush()

    @property
    def committed_usd(self) -> float:
        """Settled spend plus every open reservation (counted in full)."""
        return self._settled + sum(self._open.values())

    @property
    def billed_requests(self) -> int:
        return self._requests

    # -- operations ------------------------------------------------------------------------------

    def reserve(self, request_id: str, amount_usd: float, detail: dict[str, Any]) -> None:
        with self._lock:
            self._require_owner()
            if request_id in self._open:
                raise ValueError(f"request {request_id} already reserved")
            if self._requests + 1 > self.max_billed_requests:
                raise BudgetExceeded(f"request limit {self.max_billed_requests} reached (retries count)")
            if self.committed_usd + amount_usd > self.cap_usd:
                raise BudgetExceeded(f"reserving {amount_usd:.6f} USD would exceed the {self.cap_usd} USD cap (committed {self.committed_usd:.6f})")
            self._open[request_id] = amount_usd
            self._requests += 1
            self._append({"event": "reserve", "request_id": request_id, "amount_usd": amount_usd, "detail": detail})

    def settle(self, request_id: str, *, actual_usd: float | None, usage: dict[str, Any] | None) -> float:
        """Settle with the reported cost; without usage the full reservation is charged."""
        with self._lock:
            self._require_owner()
            reserved = self._open.pop(request_id)
            charged = reserved if actual_usd is None else actual_usd
            self._settled += charged
            self._append({"event": "settle", "request_id": request_id, "amount_usd": charged, "reserved_usd": reserved, "usage": usage, "usage_reported": actual_usd is not None})
        if actual_usd is not None and actual_usd > reserved + 1e-12:
            raise ReservationTooSmall(f"request {request_id}: actual {actual_usd:.6f} > reserved {reserved:.6f} USD")
        return charged

    def summary(self) -> dict[str, Any]:
        return {
            "budget_id": self.budget_id,
            "cap_usd": self.cap_usd,
            "max_billed_requests": self.max_billed_requests,
            "billed_requests": self._requests,
            "settled_usd": round(self._settled, 6),
            "open_reservations_usd": round(sum(self._open.values()), 6),
        }
