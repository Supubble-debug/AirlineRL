"""Concurrent step-8 diagnostic runner (sampling only, no weight updates).

Runs pre-registered groups (A: dev task x n, B: GRPO-trainable task x G) through
``ItsmAgentLoop`` -- each episode in its own ``ItsmEnvWorker`` -- on one rollout server.

Rules (all fixed before a run; see docs/itsm_step8/proposal.md):
* Order: the plan's order (A then B, task ids sorted); never reordered by results.
* Budget: a group is started only after ``BudgetManager.open_group`` reserved its envelope.
  Refused while other groups run -> wait until one closes (its unused envelope returns) and
  retry; refused with nothing in flight -> no further group is started. Every billed request is
  reserved by the single manager before it is sent. A refusal inside a group fails that
  group and also stops new groups. Nothing promises that every planned group completes.
* Retries (protocol §12): an episode-level infrastructure failure re-runs that sample with a
  fresh worker and DB, at most ``max_attempts`` (4) in total; rollout-engine errors (e.g.
  CUDA OOM) count as infrastructure too. Auxiliary judge failures are handled inside the
  worker (same trajectory, same DB) and never re-run the policy. A group with a sample that
  exhausts its attempts is discarded as a whole (no reward-0 filler).
* Concurrency ramp: start at ``levels[0]``; after ``promote_after`` consecutive clean episodes
  (valid, no infrastructure failure in any attempt) move up one level, at most ``levels[-1]``
  (8). Any rollout-engine error moves down one level and resets the streak;
  ``engine_error_stop`` consecutive engine errors at the lowest level stop the run.
* Discard stop (D6): more than 25 % of the started groups discarded (after at least 4 started
  groups) stops the run.
* Deadline: no group is started when less than ``group_estimate_s`` is left before the
  soft deadline; at the hard deadline every running episode is cancelled.
* Cancellation / failure: the loop reclaims every worker (7A-R2/R3); groups in flight are
  closed in the ledger (open requests booked in full); diagnostics are kept in the records.

Records (JSON lines, written as they happen): ``attempts.jsonl``, ``groups.jsonl``,
``trajectories.jsonl`` (valid samples: tokens, mask, log-probs, spans, rewards, auxiliary
results) and ``events.jsonl`` (group opens, concurrency changes, stop); ``run_summary.json`` at the end.
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from agent_rl.envs.itsm_scoring import ItsmInfrastructureError
from agent_rl.utils.budget_manager import BudgetManager, BudgetUnavailable
from agent_rl.utils.spend_ledger import BudgetExceeded, ReservationTooSmall


class RolloutEngineError(RuntimeError):
    """The rollout server failed a generate call (e.g. CUDA OOM, aborted engine)."""


class EngineGuard:
    """Wrap a server manager so its failures are recognisable as rollout-engine errors."""

    def __init__(self, server: Any) -> None:
        self.server = server

    async def generate(self, **kwargs: Any):
        try:
            return await self.server.generate(**kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - reclassified, original kept as cause
            raise RolloutEngineError(f"{type(error).__name__}: {error}") from error


@dataclass(frozen=True)
class GroupPlan:
    group_id: str  # no ":" (it is part of the episode id)
    part: str  # "A" (dev diagnostic) or "B" (GRPO train diagnostic)
    purpose: str  # loop purpose: "dev" or "grpo"
    task_id: str
    samples: int
    envelope_usd: float

    def __post_init__(self) -> None:
        if ":" in self.group_id or self.samples < 1 or self.envelope_usd <= 0:
            raise ValueError(f"bad group plan {self}")


@dataclass
class ConcurrencyRamp:
    levels: tuple[int, ...] = (2, 4, 8)
    promote_after: int = 4
    engine_error_stop: int = 3
    index: int = 0
    clean_streak: int = 0
    engine_errors_at_floor: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.levels or list(self.levels) != sorted(set(self.levels)) or self.levels[0] < 1 or self.levels[-1] > 8:
            raise ValueError("levels must be increasing, within 1..8")

    @property
    def limit(self) -> int:
        return self.levels[self.index]

    def _note(self, event: str) -> None:
        self.history.append({"time": time.time(), "event": event, "limit": self.limit, "clean_streak": self.clean_streak})

    def on_clean(self) -> None:
        self.clean_streak += 1
        self.engine_errors_at_floor = 0
        if self.clean_streak >= self.promote_after and self.index < len(self.levels) - 1:
            self.index += 1
            self.clean_streak = 0
            self._note("promote")

    def on_unclean(self) -> None:
        self.clean_streak = 0

    def on_engine_error(self) -> bool:
        """Returns True when the run must stop."""
        self.clean_streak = 0
        if self.index > 0:
            self.index -= 1
            self._note("demote")
            return False
        self.engine_errors_at_floor += 1
        self._note("engine_error_at_floor")
        return self.engine_errors_at_floor >= self.engine_error_stop


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class DiagnosticRunner:
    def __init__(self, *, plans: list[GroupPlan], make_loop: Callable[[GroupPlan], Any], budget: BudgetManager,
                 sampling_params: dict[str, Any], out_dir: Path, max_attempts: int = 4, ramp: ConcurrencyRamp | None = None,
                 soft_deadline: float | None = None, hard_deadline: float | None = None, group_estimate_s: float = 0.0,
                 discard_stop_fraction: float = 0.25, discard_stop_min_groups: int = 4) -> None:
        self.plans = list(plans)
        if len({p.group_id for p in self.plans}) != len(self.plans):
            raise ValueError("group ids must be unique")
        self.make_loop = make_loop
        self.budget = budget
        self.sampling_params = dict(sampling_params)
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.max_attempts = max_attempts
        self.ramp = ramp or ConcurrencyRamp()
        self.soft_deadline, self.hard_deadline, self.group_estimate_s = soft_deadline, hard_deadline, group_estimate_s
        self.discard_stop_fraction, self.discard_stop_min_groups = discard_stop_fraction, discard_stop_min_groups
        self.stop_reason: str | None = None
        self._cond: asyncio.Condition | None = None
        self._running = 0
        self._waiting = 0
        self._unqueued = 0
        self._sample_tasks: set[asyncio.Task] = set()
        self.group_results: list[dict[str, Any]] = []

    # -- records ---------------------------------------------------------------------------------

    def _write(self, name: str, record: dict[str, Any]) -> None:
        with (self.out / name).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _stop(self, reason: str) -> None:
        if self.stop_reason is None:
            self.stop_reason = reason
            self._event("stop", reason=reason)

    # -- slots -----------------------------------------------------------------------------------

    async def _acquire(self, first: bool) -> None:
        async with self._cond:
            if first:
                self._unqueued -= 1  # this sample is now visible as waiting
            self._waiting += 1
            try:
                await self._cond.wait_for(lambda: self._running < self.ramp.limit)
            finally:
                self._waiting -= 1
                self._cond.notify_all()
            self._running += 1

    async def _release(self) -> None:
        async with self._cond:
            self._running -= 1
            self._cond.notify_all()

    async def _notify(self) -> None:
        async with self._cond:
            self._cond.notify_all()

    # -- one sample ------------------------------------------------------------------------------

    async def _sample(self, plan: GroupPlan, loop: Any, index: int) -> dict[str, Any]:
        attempts: list[dict[str, Any]] = []
        queued = False
        try:
            for attempt in range(1, self.max_attempts + 1):
                if self.stop_reason is not None and attempt > 1:
                    return {"status": "stopped", "attempts": attempts}
                await self._acquire(first=not queued)
                queued = True
                outcome = await self._attempt(plan, loop, index, attempt, attempts)
                if outcome is not None:
                    return outcome
            return {"status": "attempts_exhausted", "attempts": attempts}
        finally:
            if not queued:  # cancelled before it ever asked for a slot
                async with self._cond:
                    self._unqueued -= 1
                    self._cond.notify_all()

    async def _attempt(self, plan: GroupPlan, loop: Any, index: int, attempt: int, attempts: list) -> dict[str, Any] | None:
        """One attempt in an acquired slot; returns the sample's final outcome or None to retry."""
        started = time.monotonic()
        base = {"group_id": plan.group_id, "part": plan.part, "task_id": plan.task_id, "sample_index": index, "attempt": attempt,
                "limit": self.ramp.limit, "started_utc": utc()}
        outcome: dict[str, Any] | None = None
        try:
            output = await loop.run(dict(self.sampling_params), task_id=plan.task_id, uid=plan.group_id, rollout_n=index)
        except asyncio.CancelledError:
            self._write("attempts.jsonl", {**base, "status": "cancelled", "wall_s": round(time.monotonic() - started, 3)})
            raise
        except RolloutEngineError as error:
            row = {**base, "status": "engine_error", "error": str(error)[:2000]}
            if self.ramp.on_engine_error():
                self._stop("rollout engine errors at the lowest concurrency")
            self._event("ramp", limit=self.ramp.limit)
        except (BudgetExceeded, BudgetUnavailable, ReservationTooSmall) as error:
            row = {**base, "status": "budget", "error": f"{type(error).__name__}: {error}"[:2000]}
            self._stop(f"budget: {type(error).__name__}")
            outcome = {"status": "budget"}
        except ItsmInfrastructureError as error:
            refused = self.budget.refused(plan.group_id)
            row = {**base, "status": "budget" if refused else "infra_failure", "stage": error.stage, "kind": error.kind,
                   "error": str(error)[:2000], "cleanup": getattr(error, "cleanup", None), "budget_refusals": refused}
            self.ramp.on_unclean()
            if refused:
                self._stop("budget: a request inside a group was refused")
                outcome = {"status": "budget"}
        except Exception as error:  # noqa: BLE001 - an unexpected bug: keep the diagnosis, stop the run
            row = {**base, "status": "error", "error": f"{type(error).__name__}: {error}"[:2000], "traceback": traceback.format_exc()[-4000:]}
            self._stop(f"unexpected {type(error).__name__}")
            outcome = {"status": "error"}
        else:
            info = output.extra_fields["reward_extra_info"]
            row = {**base, "status": "valid", "reward_score": output.reward_score, "termination": info["itsm_termination_reason"],
                   "episode_id": info["itsm_episode_id"], "num_turns": output.num_turns, "prompt_tokens": len(output.prompt_ids),
                   "response_tokens": len(output.response_ids), "aux_eval": output.extra_fields.get("itsm_aux_eval")}
            before = self.ramp.limit
            self.ramp.on_clean() if attempt == 1 else self.ramp.on_unclean()
            if self.ramp.limit != before:
                self._event("ramp", limit=self.ramp.limit)
            self._write("trajectories.jsonl", {**base, "prompt_ids": list(output.prompt_ids), "response_ids": list(output.response_ids),
                                                "response_mask": list(output.response_mask), "response_logprobs": list(output.response_logprobs),
                                                "reward_score": output.reward_score, "extra_fields": output.extra_fields})
            outcome = {"status": "valid", "reward": output.reward_score}
        finally:
            await self._release()
        row["wall_s"] = round(time.monotonic() - started, 3)
        attempts.append(row)
        self._write("attempts.jsonl", row)
        return None if outcome is None else {**outcome, "attempts": attempts}

    def _event(self, event: str, **fields: Any) -> None:
        self._write("events.jsonl", {"time": time.time(), "utc": utc(), "event": event, **fields})

    # -- one group -------------------------------------------------------------------------------

    async def _group(self, plan: GroupPlan) -> dict[str, Any]:
        loop = self.make_loop(plan)
        tasks = [asyncio.ensure_future(self._sample(plan, loop, index)) for index in range(plan.samples)]
        self._sample_tasks.update(tasks)
        status = "complete"
        try:
            pending = set(tasks)
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                if any(t.result()["status"] != "valid" for t in done) and pending:
                    for task in pending:  # the group is lost: stop spending on its other samples
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    pending = set()
        except asyncio.CancelledError:
            status = "cancelled"
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            self._sample_tasks.difference_update(tasks)
            states = ["cancelled" if t.cancelled() else (t.result()["status"] if t.done() and t.exception() is None else "error") for t in tasks]
            if status != "cancelled" and any(state != "valid" for state in states):
                status = "discarded"
            ledger = self.budget.close_group(plan.group_id, reason=status)
            rewards = [t.result().get("reward") for t in tasks] if status == "complete" else None
            result = {"group_id": plan.group_id, "part": plan.part, "task_id": plan.task_id, "status": status, "rewards": rewards,
                      "sample_status": states, "ledger": ledger, "finished_utc": utc()}
            self.group_results.append(result)
            self._write("groups.jsonl", result)
            await self._notify()
        return result

    def _discard_stop(self) -> bool:
        done = [g for g in self.group_results if g["part"] == "B"]
        discarded = sum(g["status"] == "discarded" for g in done)
        return len(done) >= self.discard_stop_min_groups and discarded / len(done) > self.discard_stop_fraction

    # -- run -------------------------------------------------------------------------------------

    async def run(self) -> dict[str, Any]:
        self._cond = asyncio.Condition()
        started = time.monotonic()
        group_tasks: list[asyncio.Task] = []
        watchdog = asyncio.ensure_future(self._hard_deadline_watchdog(group_tasks)) if self.hard_deadline else None
        not_started = []
        try:
            for position, plan in enumerate(self.plans):
                async with self._cond:  # open the next group only when a slot is free and no sample is queued
                    await self._cond.wait_for(lambda: self.stop_reason is not None
                                              or (self._unqueued == 0 and self._waiting == 0 and self._running < self.ramp.limit))
                if self.stop_reason is None and self._discard_stop():
                    self._stop("more than 25% of the started B groups discarded (D6)")
                if self.stop_reason is None and self.soft_deadline is not None and time.monotonic() + self.group_estimate_s > self.soft_deadline:
                    self._stop("GPU time: not enough left to start another group")
                if self.stop_reason is not None:
                    not_started = [p.group_id for p in self.plans[position:]]
                    break
                opened = False
                while True:
                    try:
                        self.budget.open_group(plan.group_id, plan.envelope_usd, {"part": plan.part, "task_id": plan.task_id, "samples": plan.samples})
                        opened = True
                        break
                    except BudgetExceeded as error:
                        in_flight = [t for t in group_tasks if not t.done()]
                        if not in_flight or self.stop_reason is not None:
                            self._stop(f"budget: group {plan.group_id} envelope not reservable with no group in flight ({error})")
                            break
                        # Unused envelope money of running groups returns when they close: wait, then retry.
                        self._event("open_deferred", group_id=plan.group_id, reason=str(error)[:300])
                        await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
                if not opened:
                    not_started = [p.group_id for p in self.plans[position:]]
                    break
                async with self._cond:
                    self._unqueued += plan.samples  # counted before the group's tasks first run
                self._event("group_open", group_id=plan.group_id, envelope_usd=plan.envelope_usd)
                group_tasks.append(asyncio.ensure_future(self._group(plan)))
            results = await asyncio.gather(*group_tasks, return_exceptions=True)
        except asyncio.CancelledError:
            self._stop("cancelled")
            for task in group_tasks:
                task.cancel()
            await asyncio.gather(*group_tasks, return_exceptions=True)
            raise
        finally:
            if watchdog is not None:
                watchdog.cancel()
            summary = {"stop_reason": self.stop_reason, "wall_s": round(time.monotonic() - started, 3), "planned_groups": len(self.plans),
                       "groups": {s: sum(g["status"] == s for g in self.group_results) for s in ("complete", "discarded", "cancelled")},
                       "not_started": not_started, "ramp": self.ramp.history, "final_limit": self.ramp.limit,
                       "budget": self.budget.summary(), "leftover_child_processes": len(multiprocessing.active_children()),
                       "finished_utc": utc()}
            (self.out / "run_summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n", encoding="utf-8")
        errors = [r for r in results if isinstance(r, BaseException)]
        if errors:
            summary["group_task_errors"] = [repr(e) for e in errors]
        return summary

    async def _hard_deadline_watchdog(self, group_tasks: list[asyncio.Task]) -> None:
        await asyncio.sleep(max(0.0, self.hard_deadline - time.monotonic()))
        self._stop("GPU time: hard deadline reached, running episodes cancelled")
        for task in list(group_tasks):
            task.cancel()
