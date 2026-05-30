"""Tier-2 debug session (pure, no Qt, no zenoh).

Drives a *single manual target* through the real Tier-2 step and watches the
Tier-3 exchange, so Tier-2 can be debugged in isolation — no PF, no Tier-1.
The target is held in the **odom** frame (set from a body-frame click), so it
stays world-fixed as the robot drives; each tick we re-derive the body bearing
and distance from the live odom pose, run ``plan_tier2``, optionally send the
sub-goal to Tier-3, and surface **events** (anomalies) edge-triggered.

The Qt window (`tier2_window.py`) owns rendering + rasterization and feeds this
session scalar signals; everything decision/event-related lives here so it is
unit-testable with fakes. ``DriveIO`` is satisfied by ``DriveClient`` as-is.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

from body.lib.local_drive_core import body_to_odom, odom_to_body
from body.lib.tier2_subgoal import Tier2Config, Tier2Decision, plan_tier2

Pose = Tuple[float, float, float]
Point2 = Tuple[float, float]


class DriveIO(Protocol):
    def send_goto_from_body(
        self, bx: float, by: float, *,
        arrival_tol_m: Optional[float] = None, v_max: Optional[float] = None,
    ) -> Optional[int]: ...
    def cancel(self) -> None: ...


@dataclass(frozen=True)
class Event:
    ts: float
    level: str        # "info" | "warn" | "error"
    code: str
    detail: str

    def as_dict(self) -> Dict[str, Any]:
        return {"ts": self.ts, "level": self.level, "code": self.code, "detail": self.detail}


@dataclass
class Tier2Tick:
    """Everything one tick produced — rendered live and written to JSONL."""
    ts: float
    has_target: bool
    target_body: Optional[Point2]            # target in the current body frame
    target_dist_m: float
    decision: Optional[Tier2Decision]        # None when no target / no scan
    sent_cmd_id: Optional[int]               # set the tick a goto was issued
    tier3: Optional[Dict[str, Any]]          # latest Tier-3 status echo
    events: List[Event] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts,
            "has_target": self.has_target,
            "target_body": list(self.target_body) if self.target_body else None,
            "target_dist_m": self.target_dist_m,
            "decision": self.decision.as_dict() if self.decision else None,
            "sent_cmd_id": self.sent_cmd_id,
            "tier3": self.tier3,
            "events": [e.as_dict() for e in self.events],
        }


@dataclass(frozen=True)
class Tier2SessionConfig:
    tier2_cfg: Tier2Config = field(default_factory=Tier2Config)
    subgoal_arrival_tol_m: float = 0.15
    target_arrival_tol_m: float = 0.20       # within this of the target → done
    sub_v_max: Optional[float] = None
    odom_stale_s: float = 0.5
    scan_stale_s: float = 0.5


class Tier2Session:
    def __init__(self, io: DriveIO, cfg: Optional[Tier2SessionConfig] = None):
        self._io = io
        self._cfg = cfg or Tier2SessionConfig()
        self._target_odom: Optional[Point2] = None
        self._drive = False
        self._cmd_id: Optional[int] = None     # last goto we sent
        self._reached = False
        # Edge-trigger memory for event detection.
        self._prev: Dict[str, Any] = {}

    # ── Operator controls ────────────────────────────────────────────

    def set_target_from_body(self, bx: float, by: float, odom_pose: Pose) -> None:
        """Pin a target: convert the body-frame click to odom so it stays put."""
        self._target_odom = body_to_odom((bx, by), odom_pose)
        self._reached = False
        self._cmd_id = None

    def clear_target(self) -> None:
        self._target_odom = None
        self._cmd_id = None
        self._reached = False
        self._io.cancel()

    def set_drive(self, on: bool) -> None:
        self._drive = bool(on)
        if not on:
            self._io.cancel()
            self._cmd_id = None

    def set_tunables(self, *, subgoal_arrival_tol_m: float, sub_v_max: float) -> None:
        """Update the per-goto knobs (from the UI spinners) in place."""
        c = self._cfg
        self._cfg = Tier2SessionConfig(
            tier2_cfg=c.tier2_cfg,
            subgoal_arrival_tol_m=subgoal_arrival_tol_m,
            target_arrival_tol_m=c.target_arrival_tol_m,
            sub_v_max=sub_v_max,
            odom_stale_s=c.odom_stale_s,
            scan_stale_s=c.scan_stale_s,
        )

    @property
    def has_target(self) -> bool:
        return self._target_odom is not None

    @property
    def driving(self) -> bool:
        return self._drive

    # ── Per-tick ─────────────────────────────────────────────────────

    def tick(
        self,
        now: float,
        *,
        odom_pose: Optional[Pose],
        grid: Optional[Any],
        meta: Optional[Dict[str, Any]],
        scan_age_s: Optional[float],
        tier3_status: Optional[Dict[str, Any]],
        e_stop_active: bool,
        heartbeat_ok: bool,
    ) -> Tier2Tick:
        evs: List[Event] = []
        self._health_events(now, evs, odom_pose, scan_age_s, e_stop_active, heartbeat_ok)
        self._tier3_events(now, evs, tier3_status)

        if self._target_odom is None or odom_pose is None:
            return Tier2Tick(now, self.has_target, None, 0.0, None, None, tier3_status, evs)

        target_body = odom_to_body(self._target_odom, odom_pose)
        dist = math.hypot(target_body[0], target_body[1])
        bearing = math.atan2(target_body[1], target_body[0])

        # Reached the target → stop driving toward it (debug: hold here).
        if dist <= self._cfg.target_arrival_tol_m:
            if not self._reached:
                self._reached = True
                self._io.cancel()
                self._cmd_id = None
                evs.append(Event(now, "info", "target_reached", f"dist={dist:.2f}m"))
            return Tier2Tick(now, True, target_body, dist, None, None, tier3_status, evs)

        if grid is None or meta is None:
            self._emit_on_change(now, evs, "no_scan", True,
                                 "no_scan", "warn", "no rasterizable scan")
            return Tier2Tick(now, True, target_body, dist, None, None, tier3_status, evs)
        self._prev["no_scan"] = False        # scan present → re-arm the edge

        decision = plan_tier2(grid, meta, bearing, dist, self._cfg.tier2_cfg)
        self._decision_events(now, evs, decision)

        sent: Optional[int] = None
        if self._drive and decision.ok:
            if self._should_send(tier3_status):
                cid = self._io.send_goto_from_body(
                    decision.body_xy[0], decision.body_xy[1],
                    arrival_tol_m=self._cfg.subgoal_arrival_tol_m,
                    v_max=self._cfg.sub_v_max,
                )
                if cid is None:
                    evs.append(Event(now, "error", "send_failed", "no odom/connection"))
                else:
                    self._cmd_id = cid
                    sent = cid

        return Tier2Tick(now, True, target_body, dist, decision, sent, tier3_status, evs)

    # ── Send policy ──────────────────────────────────────────────────

    def _should_send(self, tier3_status: Optional[Dict[str, Any]]) -> bool:
        if self._cmd_id is None:
            return True                       # nothing outstanding yet
        if tier3_status is None:
            return False
        if int(tier3_status.get("cmd_id", 0)) != self._cmd_id:
            return False                      # our goto not yet serviced; wait
        # Tier-3 finished this sub-goal → re-pick toward the (same) target.
        return tier3_status.get("state") in ("ARRIVED", "IDLE")

    # ── Event detection (edge-triggered) ─────────────────────────────

    def _emit_on_change(self, now, evs, key, value, code, level, detail):
        if self._prev.get(key) != value:
            self._prev[key] = value
            if value:                          # only emit when the condition is true
                evs.append(Event(now, level, code, detail))

    def _health_events(self, now, evs, odom_pose, scan_age_s, e_stop, hb_ok):
        self._emit_on_change(now, evs, "e_stop", bool(e_stop),
                             "e_stop_active", "error", "motor/watchdog e-stop latched")
        self._emit_on_change(now, evs, "hb", not hb_ok,
                             "heartbeat_stale", "warn", "chassis heartbeat not fresh")
        self._emit_on_change(now, evs, "odom_none", odom_pose is None,
                             "odom_missing", "warn", "no odom pose")
        stale_scan = scan_age_s is not None and scan_age_s > self._cfg.scan_stale_s
        self._emit_on_change(now, evs, "scan_stale", stale_scan,
                             "scan_stale", "warn", f"scan age {scan_age_s:.2f}s" if scan_age_s else "")

    def _decision_events(self, now, evs, d: Tier2Decision):
        if not d.ok:
            self._emit_on_change(now, evs, "t2_block", d.reason,
                                 d.reason, "warn", "Tier-2 found no usable point")
        else:
            self._prev["t2_block"] = None      # clear so a future block re-emits
            self._emit_on_change(now, evs, "t2_cap", d.capped_at_target,
                                 "capped_at_target", "info", "sub-goal == target (clear)")

    def _tier3_events(self, now, evs, st: Optional[Dict[str, Any]]):
        if st is None:
            return
        state = st.get("state")
        self._emit_on_change(now, evs, "t3_state", state,
                             "tier3_state", "info", f"Tier-3 {state}")
        br = st.get("blocked_reason")
        self._emit_on_change(now, evs, "t3_block", br,
                             f"tier3_{br}" if br else "tier3_clear",
                             "warn" if br else "info", f"mode={st.get('mode')}")
        # cmd_id mismatch: Tier-3 servicing an id below the one we last sent
        # (the stale-counter collision that froze the robot last session).
        mismatch = (
            self._cmd_id is not None
            and int(st.get("cmd_id", 0)) < self._cmd_id
        )
        self._emit_on_change(
            now, evs, "cmd_mismatch", mismatch, "cmd_id_mismatch", "error",
            f"sent {self._cmd_id}, Tier-3 servicing {st.get('cmd_id')}")
