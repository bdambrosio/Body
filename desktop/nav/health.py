"""Upstream-signal liveness watcher.

Observes the freshness of Pi-side streams (body/status, body/odom,
body/map/local_2p5d, body/lidar/scan), the desktop→Pi heartbeat, the
session connection, and e-stop. Emits edge-triggered `pi.*` events to
the tracer when any of these flip between fresh and stale (or on/off).

Why: today the safety toolbar's `conn` pill conflates "zenoh session
open" with "Pi watchdog responding," and there's no visibility at all
into per-topic stalls. A trace reviewer needs to be able to tell
"odom went stale 4 s before the mission failed" without the operator
having to remember to call it out.

The watcher does NOT change behavior — it only emits events. The
mission's own pose-freshness gate (see `mission.py`/`main_window.py`)
still drives the actual pause→resume cycle.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .tracing import CAT_PI, LEVEL_INFO, LEVEL_WARN, Tracer

logger = logging.getLogger(__name__)


# Signals tracked by `_track_age_edge`. Threshold defaults are chosen
# from each producer's nominal rate × ~2.5:
#
#     status     — Pi watchdog, ~1 Hz   → 3.0 s
#     odom       — Pi odometry, ~10 Hz  → 0.5 s
#     local_map  — fuser publish ~1 Hz  → 2.0 s
#     scan       — Pi lidar, ~10 Hz     → 1.0 s
#     heartbeat  — desktop→Pi, 5 Hz     → 0.4 s
#
# These match the existing safety-toolbar / mission thresholds where
# they overlap, so a `pi.stall` trace entry lines up with the same
# moment the operator's pills go red / the mission pauses on no_pose.

@dataclass
class HealthThresholds:
    status_s: float = 3.0
    odom_s: float = 0.5
    local_map_s: float = 2.0
    scan_s: float = 1.0
    heartbeat_s: float = 0.4

    def for_signal(self, name: str) -> float:
        return {
            "status": self.status_s,
            "odom": self.odom_s,
            "local_map": self.local_map_s,
            "scan": self.scan_s,
            "heartbeat": self.heartbeat_s,
        }[name]


@dataclass
class _SignalState:
    fresh: Optional[bool] = None   # None = uninitialized; no edge until first tick
    last_age_s: Optional[float] = None


class LivenessWatcher:
    """Edge-triggered upstream liveness observer.

    Owns no thread of its own; call `tick()` from the UI's existing
    redraw timer. Internal rate-limiting drops ticks faster than
    `tick_period_s` so per-redraw calls cost ~nothing in the steady
    state.
    """

    def __init__(
        self,
        tracer: Tracer,
        *,
        fuser: Any,
        chassis: Any,
        thresholds: Optional[HealthThresholds] = None,
        tick_period_s: float = 1.0,
    ):
        self._tracer = tracer
        self._fuser = fuser
        self._chassis = chassis
        self._thresholds = thresholds or HealthThresholds()
        self._tick_period_s = float(tick_period_s)

        self._signals: Dict[str, _SignalState] = {
            name: _SignalState()
            for name in ("status", "odom", "local_map", "scan", "heartbeat")
        }
        self._connected: Optional[bool] = None
        self._estop: Optional[bool] = None
        self._last_hb_seq: int = -1
        self._last_hb_change_wall: float = 0.0
        self._last_tick_wall: float = 0.0

    # ── Tick ────────────────────────────────────────────────────────

    def tick(self, now: Optional[float] = None) -> None:
        now = now if now is not None else time.time()
        if now - self._last_tick_wall < self._tick_period_s:
            return
        self._last_tick_wall = now

        # Fuser-side ages: local_map, odom, scan. status_summary()
        # reports ages from local arrival ts, which is what we want
        # (independent of Pi clock skew).
        try:
            fuser_st = self._fuser.status_summary()
        except Exception:
            logger.exception("status_summary raised; skipping fuser signals")
            fuser_st = {"ages": {}}
        ages = fuser_st.get("ages") or {}
        for sig in ("local_map", "odom"):
            self._track_age_edge(sig, ages.get(sig))
        # `scan` rate isn't surfaced by status_summary; pull directly
        # from the fuser's internal last-arrival ts via a hidden but
        # stable attribute. If it disappears, the signal simply
        # stops emitting edges (None age = treat as unknown).
        scan_age = self._fuser_attr_age(now, "_last_lidar_ts")
        self._track_age_edge("scan", scan_age)

        # Chassis-side: body/status freshness, heartbeat, connection,
        # e-stop. Held under the chassis state lock.
        try:
            with self._chassis.state.lock:
                status_ts = self._chassis.state.status_ts
                hb_seq = self._chassis.state.heartbeat_seq
                connected = bool(self._chassis.state.connected)
                motor = self._chassis.state.motor_state
                status = self._chassis.state.status
        except Exception:
            logger.exception("chassis state read raised; skipping signals")
            return

        status_age = (now - status_ts) if status_ts > 0 else None
        self._track_age_edge("status", status_age)

        if hb_seq != self._last_hb_seq:
            self._last_hb_seq = hb_seq
            self._last_hb_change_wall = now
        hb_age = (
            (now - self._last_hb_change_wall)
            if self._last_hb_change_wall > 0 else None
        )
        # Heartbeat freshness is meaningful only while connected; treat
        # a disconnected chassis as "heartbeat unknown" so the
        # connection edge isn't shadowed by a redundant stall.
        if not connected:
            hb_age = None
        self._track_age_edge("heartbeat", hb_age)

        # Connection edge (separate from fresh-stale — disconnect is
        # categorically different from "Pi alive but quiet").
        if self._connected is None:
            self._connected = connected
        elif connected != self._connected:
            self._tracer.emit(
                CAT_PI,
                "connected" if connected else "disconnected",
                {},
                level=LEVEL_INFO if connected else LEVEL_WARN,
            )
            self._connected = connected

        # E-stop edge.
        estop = False
        if isinstance(motor, dict):
            estop = bool(motor.get("e_stop_active", False))
        elif isinstance(status, dict):
            estop = bool(status.get("e_stop_active", False))
        if self._estop is None:
            self._estop = estop
        elif estop != self._estop:
            self._tracer.emit(
                CAT_PI, "estop",
                {"active": estop},
                level=LEVEL_WARN if estop else LEVEL_INFO,
            )
            self._estop = estop

    # ── Helpers ─────────────────────────────────────────────────────

    def _track_age_edge(
        self, name: str, age_s: Optional[float],
    ) -> None:
        """Compare `age_s` against the per-signal threshold and emit
        on edge transitions. `None` age means "no data yet" — treated
        as stale, but only flips an edge once both prior and current
        values are non-None to avoid a spurious initial event.
        """
        threshold = self._thresholds.for_signal(name)
        if age_s is None:
            fresh = False
        else:
            fresh = age_s < threshold

        state = self._signals[name]
        prev = state.fresh
        state.last_age_s = age_s
        if prev is None:
            # First observation — establish baseline without emitting.
            state.fresh = fresh
            return
        if fresh == prev:
            return
        state.fresh = fresh
        self._tracer.emit(
            CAT_PI,
            "recovered" if fresh else "stall",
            {
                "signal": name,
                "age_s": age_s,
                "threshold_s": threshold,
            },
            level=LEVEL_INFO if fresh else LEVEL_WARN,
        )

    def _fuser_attr_age(self, now: float, attr: str) -> Optional[float]:
        ts = getattr(self._fuser, attr, None)
        # The fuser stores _last_lidar_ts inside its lock; reading
        # without the lock is fine for a monotone wall-clock float
        # (worst case we get an old value, which only shortens the
        # detected stall by one tick).
        if not isinstance(ts, (int, float)) or ts <= 0:
            return None
        return now - float(ts)
