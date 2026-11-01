"""Recovery supervisor: get the drive going again on its own after a fault.

Replaces "hold for operator resume, forever" with: wait for the fault to clear
-> verify the pose (lidar, camera fallback) -> re-localize if it is wrong ->
resume, with backoff that never gives up. The operator can still Resume (skip
the wait) or Stop at any time; a teleop suspension is never auto-resumed.

Pure logic: no Qt, no I/O. The hierarchical drive calls ``step(now, ...)``
from its tick while it is suspended; everything that touches the robot is an
injected callable (see ``RecoveryHooks``).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Optional, Sequence

from desktop.localization.pose_verify import BAD, PENDING, Verdict

logger = logging.getLogger(__name__)

# step() actions
WAIT = "wait"
RESUME = "resume"


@dataclass(frozen=True)
class RecoveryConfig:
    # Link must be continuously healthy this long before verifying; grows
    # with recent fault count (flapping link) — index = recent faults - 1.
    stable_link_s: Sequence[float] = (2.0, 5.0, 15.0, 30.0)
    flap_window_s: float = 120.0
    # Give the lidar this long to produce fresh matches before deciding
    # with what it has (then the camera fallback).
    verify_timeout_s: float = 5.0
    # Re-localize failures back off through these waits, then repeat the last.
    relocalize_backoff_s: Sequence[float] = (10.0, 30.0, 60.0)
    # A pose that verifies "bad" again right after a successful re-localize
    # also backs off (not a hot loop of relocates).
    max_relocalize_per_fault: int = 3


@dataclass
class RecoveryHooks:
    """What the supervisor may do. All optional except ``verifier``."""
    verifier: object                                   # PoseVerifier-like
    relocalize: Optional[Callable[[], tuple]] = None   # -> (ok: bool, detail: str)
    # Any Pi e-stop latched? (No resume while it is.)
    estop_latched: Optional[Callable[[], bool]] = None
    # Try to clear it: called only when the link is healthy and no goal is
    # live; the hook itself refuses non-link e-stops (LinkEStopGuard). -> bool
    rearm: Optional[Callable[[], bool]] = None
    # Called from begin() so the hook side can note the fault start.
    on_begin: Optional[Callable[[float], None]] = None
    # Called on each phase change with (tag, detail) — diagnostic dumps.
    on_event: Optional[Callable[[str, str], None]] = None


_PHASES = ("idle", "wait_link", "verify", "relocalize_wait")


class RecoverySupervisor:
    def __init__(self, hooks: RecoveryHooks, cfg: RecoveryConfig = RecoveryConfig()) -> None:
        self._h = hooks
        self._cfg = cfg
        self._phase = "idle"
        self._reason = ""
        self._fault_times: Deque[float] = deque()
        self._stable_since: Optional[float] = None
        self._stable_req = cfg.stable_link_s[0]
        self._verify_started: float = 0.0
        self._next_try: float = 0.0
        self._reloc_fails = 0
        self._relocs = 0
        self._last_verdict: Optional[Verdict] = None
        self._last_rearm: float = -1e9
        self._rearm_gap_s = 10.0
        self._attempts = 0
        self.resume_detail = ""

    # ── lifecycle ────────────────────────────────────────────────────

    def begin(self, now: float, reason: str, *, link_fault: bool = True) -> None:
        """A fault started (the drive just suspended), or — ``link_fault=False``
        — a blocked leg wants its pose checked (no flap accounting, and the
        link is already up, so the stability wait is the shortest one)."""
        if link_fault:
            self._fault_times.append(now)
        while self._fault_times and now - self._fault_times[0] > self._cfg.flap_window_s:
            self._fault_times.popleft()
        k = min(len(self._fault_times), len(self._cfg.stable_link_s)) - 1
        self._stable_req = (self._cfg.stable_link_s[max(0, k)] if link_fault
                            else self._cfg.stable_link_s[0])
        self._reason = reason
        self._stable_since = None
        self._reloc_fails = 0
        self._relocs = 0
        self._attempts = 0
        self._last_verdict = None
        try:
            self._h.verifier.mark_fault()
        except Exception:
            logger.exception("recovery: verifier.mark_fault failed")
        if self._h.on_begin is not None:
            try:
                self._h.on_begin(now)
            except Exception:
                logger.exception("recovery: on_begin failed")
        self._set_phase("wait_link", f"{reason}; need link stable {self._stable_req:.0f}s")

    def cancel(self) -> None:
        self._phase = "idle"

    def active(self) -> bool:
        return self._phase != "idle"

    def status(self, now: float) -> dict:
        nxt = None
        if self._phase == "wait_link" and self._stable_since is not None:
            nxt = max(0.0, self._stable_since + self._stable_req - now)
        elif self._phase == "relocalize_wait":
            nxt = max(0.0, self._next_try - now)
        return {
            "phase": self._phase,
            "reason": self._reason,
            "stable_req_s": self._stable_req,
            "next_try_in_s": nxt,
            "attempts": self._attempts,
            "last_verdict": None if self._last_verdict is None else
            f"{self._last_verdict.kind}: {self._last_verdict.detail}",
        }

    # ── step ─────────────────────────────────────────────────────────

    def step(self, now: float, *, link_ok: bool, goal_live: bool = False) -> str:
        if self._phase == "idle":
            return WAIT
        self._maybe_rearm(now, link_ok=link_ok, goal_live=goal_live)
        if not link_ok:
            if self._stable_since is not None:
                logger.info("recovery: link dropped again — waiting")
            self._stable_since = None
            if self._phase != "wait_link":
                self._set_phase("wait_link", "link lost again")
            return WAIT
        if self._stable_since is None:
            self._stable_since = now
        if self._phase == "wait_link":
            if now - self._stable_since < self._stable_req:
                return WAIT
            if self._estop_latched():
                return WAIT       # re-arm (rate-limited) handles it
            self._start_verify(now)
        if self._phase == "relocalize_wait":
            if now < self._next_try:
                return WAIT
            return self._relocalize(now)
        if self._phase == "verify":
            return self._verify(now)
        return WAIT

    # ── phases ───────────────────────────────────────────────────────

    def _start_verify(self, now: float) -> None:
        self._attempts += 1
        self._verify_started = now
        try:
            self._h.verifier.restart_window()
        except Exception:
            logger.exception("recovery: verifier.restart_window failed")
        self._set_phase("verify", f"attempt {self._attempts}")

    def _verify(self, now: float) -> str:
        v = self._h.verifier
        try:
            v.poll()
            final = now - self._verify_started >= self._cfg.verify_timeout_s
            verdict = v.verdict(final=final)
        except Exception as e:
            logger.exception("recovery: verifier failed")
            verdict = Verdict("unknown", f"verifier error: {e}", "none")
        if verdict.kind == PENDING:
            return WAIT
        self._last_verdict = verdict
        if verdict.kind == BAD:
            logger.warning("recovery: pose check BAD (%s) — re-localizing", verdict.detail)
            return self._relocalize(now)
        if self._estop_latched():
            self._set_phase("wait_link", "pose checked; e-stop still latched")
            return WAIT
        # OK, or UNKNOWN after lidar + camera: the fault was the link, the
        # odometry chain is intact — same as an operator Resume.
        self.resume_detail = f"pose {verdict.kind} ({verdict.source}) — {verdict.detail}"
        self._set_phase("idle", "resume: " + self.resume_detail)
        return RESUME

    def _relocalize(self, now: float) -> str:
        if self._h.relocalize is None:
            return self._backoff(now, "no re-localizer")
        if self._relocs >= self._cfg.max_relocalize_per_fault:
            self._relocs = 0
            return self._backoff(now, "re-localize did not stick")
        try:
            ok, detail = self._h.relocalize()
        except Exception as e:
            logger.exception("recovery: relocalize failed")
            ok, detail = False, f"error: {e}"
        self._relocs += 1
        if ok:
            logger.info("recovery: re-localized (%s) — verifying again", detail)
            self._start_verify(now)
            return WAIT
        return self._backoff(now, f"re-localize failed: {detail}")

    def _backoff(self, now: float, why: str) -> str:
        b = self._cfg.relocalize_backoff_s
        wait = b[min(self._reloc_fails, len(b) - 1)]
        self._reloc_fails += 1
        self._next_try = now + wait
        self._set_phase("relocalize_wait", f"{why}; retry in {wait:.0f}s")
        return WAIT

    # ── e-stop ───────────────────────────────────────────────────────

    def _estop_latched(self) -> bool:
        f = self._h.estop_latched
        try:
            return bool(f()) if f is not None else False
        except Exception:
            return False

    def _maybe_rearm(self, now: float, *, link_ok: bool, goal_live: bool) -> None:
        if (self._h.rearm is None or not link_ok or goal_live
                or self._stable_since is None
                or now - self._stable_since < self._cfg.stable_link_s[0]
                or now - self._last_rearm < self._rearm_gap_s
                or not self._estop_latched()):
            return
        self._last_rearm = now
        try:
            sent = bool(self._h.rearm())
        except Exception:
            logger.exception("recovery: rearm failed")
            sent = False
        if sent:
            logger.warning("recovery: link healthy %.0fs — auto re-arm of the "
                           "link e-stop sent", now - self._stable_since)
        else:
            logger.warning("recovery: e-stop latched and not ours to clear "
                           "(not a link fault) — waiting for the operator's Re-arm")
        self._emit("rearm", "sent" if sent else "refused")

    # ── helpers ──────────────────────────────────────────────────────

    def _set_phase(self, phase: str, detail: str) -> None:
        assert phase in _PHASES
        if phase != self._phase:
            logger.info("recovery: %s -> %s (%s)", self._phase, phase, detail)
        self._phase = phase
        self._emit(phase, detail)

    def _emit(self, tag: str, detail: str) -> None:
        cb = self._h.on_event
        if cb is None:
            return
        try:
            cb(tag, detail)
        except Exception:
            logger.exception("recovery: on_event failed")


class LinkEStopGuard:
    """Decides whether a latched Pi e-stop was caused by the link fault and
    so may be re-armed automatically (spec §5.10 deliberately relaxed for
    this one case; the Pi still refuses a re-arm while the heartbeat is bad).

    May re-arm only when the e-stop was NOT latched when the fault began and
    either the latest emergency message is the watchdog's
    ``heartbeat_timeout`` (received during this fault) or no emergency
    message arrived at all (the watchdog's own message lost in the same
    drop). An operator / desktop e-stop, a boot latch, or any other reason
    stays operator-only.
    """

    AUTO_REASONS = ("heartbeat_timeout",)

    def __init__(self) -> None:
        self._fault_start: Optional[float] = None
        self._active_at_start = True

    def on_fault_start(self, now: float, *, estop_active: bool) -> None:
        self._fault_start = now
        self._active_at_start = bool(estop_active)

    def may_rearm(self, *, estop_active: bool, emergency: Optional[dict],
                  emergency_ts: float) -> bool:
        if not estop_active or self._fault_start is None or self._active_at_start:
            return False
        if emergency is not None and emergency_ts >= self._fault_start - 1.0:
            return (str(emergency.get("reason")) in self.AUTO_REASONS
                    and str(emergency.get("source", "watchdog")) == "watchdog")
        return True     # latched during this fault, no other reason seen


__all__ = ["RecoverySupervisor", "RecoveryHooks", "RecoveryConfig", "LinkEStopGuard",
           "WAIT", "RESUME"]
