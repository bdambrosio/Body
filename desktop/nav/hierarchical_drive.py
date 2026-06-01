"""Tier-1/Tier-2 hierarchical drive orchestrator (pure).

Drives a hand-placed topological waypoint route (Tier-1, a ``PatrolRunner``)
by handing Tier-3 (``body/drive/goto``) a live-observed sub-goal each leg.
Per tick: get the robot's world pose, take the bearing to the next waypoint
(the *only* thing that crosses over from the world map), let Tier-2 pick the
furthest live-visible free point along that bearing in the body-frame scan
grid, and send that body-frame point to Tier-3. Arrival at a *waypoint* is
judged by the world pose, not by Tier-3's sub-goal ARRIVED — Tier-3 ARRIVED
just means "reached this sub-goal", so we re-pick the next one toward the
same waypoint until the pose is within tolerance.

Pure: no Qt, no zenoh. Collaborators are injected (``DriveClient`` satisfies
``DriveIO`` as-is; ``PFPoseProvider`` is the PF-relative re-alignment
bootstrap — swap it for an LPR-backed provider later without touching this).

Frames: the sub-goal stays body-frame all the way into
``send_goto_from_body`` (which re-anchors via live odom), so a constant
PF↔odom yaw offset is harmless; only temporal skew matters, and it is
self-correcting because we re-pick each leg from fresh pose + scan.
"""
from __future__ import annotations

import enum
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, Tuple

from body.lib.scan_raster import ScanRasterConfig, rasterize_scan
from body.lib.tier2_subgoal import Tier2Config, bearing_to_waypoint, plan_tier2
from desktop.nav.patrol import PatrolRunner

logger = logging.getLogger(__name__)

Pose = Tuple[float, float, float]   # world (x, y, theta)


class PoseProvider(Protocol):
    def world_pose(self) -> Optional[Pose]:
        """Robot world pose, or None when unavailable/stale. Re-align seam."""
        ...


class DriveIO(Protocol):
    def latest_scan(self) -> Optional[Dict[str, Any]]: ...
    def latest_status(self) -> Optional[Dict[str, Any]]: ...
    def send_goto_from_body(
        self, bx: float, by: float, *,
        arrival_tol_m: Optional[float] = None, v_max: Optional[float] = None,
    ) -> Optional[int]: ...
    def cancel(self) -> None: ...


class PFPoseProvider:
    """PF-relative bootstrap: world pose straight from the particle filter.

    Replace with an LPR-backed provider (or one that re-anchors the PF to an
    LPR fix) when LPR lands — the orchestrator only knows ``world_pose()``.
    """

    def __init__(self, fuser: Any):
        self._fuser = fuser

    def world_pose(self) -> Optional[Pose]:
        latest = self._fuser.pose_source.latest_pose()
        if latest is None:
            return None
        pose = latest[0] if isinstance(latest, tuple) and len(latest) == 2 else latest
        return (float(pose[0]), float(pose[1]), float(pose[2]))


class HierState(enum.Enum):
    IDLE = "IDLE"
    ALIGNING = "ALIGNING"
    SELECT_SUBGOAL = "SELECT_SUBGOAL"
    DRIVING_SUBGOAL = "DRIVING_SUBGOAL"
    ADVANCE_WAYPOINT = "ADVANCE_WAYPOINT"
    ARRIVED = "ARRIVED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class HierConfig:
    waypoint_tol_m: float = 0.30          # PF-pose distance to count a waypoint reached
    # Intermediate (non-terminal) waypoints advance at this looser radius and
    # WITHOUT canceling Tier-3 — the next goto supersedes seamlessly, so the
    # bot doesn't decelerate-stop at every sub-waypoint. Must exceed Tier-3's
    # slowdown_distance_m (~0.4) so the advance fires before it slows. The
    # terminal waypoint still uses the tight waypoint_tol_m + a real stop.
    passthrough_tol_m: float = 0.60
    subgoal_arrival_tol_m: float = 0.15   # tol passed to Tier-3 for each sub-goal
    repick_hysteresis_rad: float = 0.35   # re-pick mid-leg if the bearing drifts past this
    max_blocked_repicks: int = 3          # retries before BLOCKED becomes a pause
    sub_v_max: Optional[float] = None     # per-goto speed cap (None → Tier-3 default)
    raster_cfg: ScanRasterConfig = field(default_factory=ScanRasterConfig)
    tier2_cfg: Tier2Config = field(default_factory=Tier2Config)


def _dist(a: Pose, wp_xy: Tuple[float, float]) -> float:
    return math.hypot(a[0] - wp_xy[0], a[1] - wp_xy[1])


class HierarchicalDrive:
    """Single-step state machine; call ``tick()`` from the host's redraw loop."""

    def __init__(
        self,
        runner: PatrolRunner,
        pose: PoseProvider,
        io: DriveIO,
        cfg: Optional[HierConfig] = None,
    ):
        self._runner = runner
        self._pose = pose
        self._io = io
        self._cfg = cfg or HierConfig()
        self._state = HierState.IDLE
        self._cmd_id: Optional[int] = None
        self._sent_bearing: Optional[float] = None
        self._subgoal_body: Optional[Tuple[float, float]] = None
        self._waypoint: Optional[Tuple[float, float]] = None
        self._blocked_repicks = 0
        self._block_reason: Optional[str] = None

    # ── Control ──────────────────────────────────────────────────────

    def _to(self, state: HierState) -> None:
        if state is not self._state:
            logger.info("hier: %s -> %s", self._state.value, state.value)
        self._state = state

    def start(self) -> None:
        self._blocked_repicks = 0
        self._block_reason = None
        self._cmd_id = None
        self._subgoal_body = None
        logger.info("hier: start (%d waypoints)", self._runner.n)
        self._to(
            HierState.ALIGNING if self._runner.current_target() is not None
            else HierState.FAILED
        )

    def stop(self) -> None:
        self._io.cancel()
        self._subgoal_body = None
        self._to(HierState.IDLE)

    def _enter_blocked(self, reason: str) -> None:
        # Count *consecutive* blocks (only reset by real progress — a sub-goal
        # done or a waypoint advance), NOT by a successful re-send. Otherwise a
        # standstill that Tier-3 vetoes every time loops SELECT↔BLOCKED forever,
        # re-issuing the identical futile goto.
        self._block_reason = reason
        self._blocked_repicks += 1
        if self._blocked_repicks > self._cfg.max_blocked_repicks:
            logger.warning("hier: blocked (%s) — gave up after %d retries, pausing",
                           reason, self._cfg.max_blocked_repicks)
        self._to(HierState.BLOCKED)

    def _arrival_tol(self) -> float:
        """Tight stop tolerance on the terminal leg, loose pass-through radius
        for intermediates (so we advance + retarget before Tier-3 decelerates)."""
        return (self._cfg.waypoint_tol_m if self._runner.is_terminal_leg()
                else self._cfg.passthrough_tol_m)

    # ── Introspection (for the UI overlay / status label) ────────────

    def state(self) -> HierState:
        return self._state

    def block_reason(self) -> Optional[str]:
        return self._block_reason

    def current_subgoal_body(self) -> Optional[Tuple[float, float]]:
        return self._subgoal_body

    def current_waypoint_world(self) -> Optional[Tuple[float, float]]:
        return self._waypoint

    # ── Tick ─────────────────────────────────────────────────────────

    def tick(self, now: float) -> HierState:
        s = self._state
        if s == HierState.ALIGNING:
            self._tick_aligning()
        elif s == HierState.SELECT_SUBGOAL:
            self._tick_select()
        elif s == HierState.DRIVING_SUBGOAL:
            self._tick_driving()
        elif s == HierState.ADVANCE_WAYPOINT:
            self._tick_advance()
        elif s == HierState.BLOCKED:
            self._tick_blocked()
        # IDLE / ARRIVED / FAILED are terminal/inert.
        return self._state

    def _tick_aligning(self) -> None:
        # Re-alignment seam: today, just wait for a valid world pose.
        if self._pose.world_pose() is not None:
            self._to(HierState.SELECT_SUBGOAL)

    def _tick_select(self) -> None:
        pose = self._pose.world_pose()
        if pose is None:
            self._to(HierState.ALIGNING)
            return
        wp = self._runner.current_target()
        if wp is None:
            self._to(HierState.FAILED)
            return
        self._waypoint = (wp.x_m, wp.y_m)
        if _dist(pose, self._waypoint) <= self._arrival_tol():
            self._to(HierState.ADVANCE_WAYPOINT)
            return

        grid_meta = self._raster()
        if grid_meta is None:
            self._enter_blocked("no_scan")
            return
        grid, meta = grid_meta
        wp_dist = _dist(pose, self._waypoint)
        bearing = bearing_to_waypoint(pose[0], pose[1], pose[2], wp.x_m, wp.y_m)
        # Tier-2 step (shared with the debug console): furthest free point
        # along the bearing, capped at the waypoint — never aim past it.
        r = plan_tier2(grid, meta, bearing, wp_dist, self._cfg.tier2_cfg)
        if not r.ok:
            logger.info("hier: tier2 no point (%s) bearing=%.2f wp_dist=%.2f",
                        r.reason, bearing, wp_dist)
            self._enter_blocked(r.reason)
            return

        cid = self._io.send_goto_from_body(
            r.body_xy[0], r.body_xy[1],
            arrival_tol_m=self._cfg.subgoal_arrival_tol_m,
            v_max=self._cfg.sub_v_max,
        )
        if cid is None:
            self._enter_blocked("send_failed")
            return
        self._cmd_id = cid
        self._sent_bearing = bearing
        self._subgoal_body = r.body_xy
        self._block_reason = None
        # Where the robot *thinks* the waypoint is relative to its own nose
        # (+x fwd, +y left). Compare against where the obstacle/clear space
        # actually is to tell a bad world->body bearing from a real obstacle.
        dx, dy = wp.x_m - pose[0], wp.y_m - pose[1]
        cw, sw = math.cos(-pose[2]), math.sin(-pose[2])
        wp_bx, wp_by = dx * cw - dy * sw, dx * sw + dy * cw
        logger.info(
            "hier: goto cmd=%d pose=(%.2f,%.2f,%.0f°) wp=(%.2f,%.2f) "
            "wp_body=(%.2f,%.2f) bearing=%.0f° free=%.2f sub=(%.2f,%.2f)",
            cid, pose[0], pose[1], math.degrees(pose[2]), wp.x_m, wp.y_m,
            wp_bx, wp_by, math.degrees(bearing), r.free_dist_m,
            r.body_xy[0], r.body_xy[1])
        self._to(HierState.DRIVING_SUBGOAL)

    def _tick_driving(self) -> None:
        pose = self._pose.world_pose()
        # Waypoint reached (PF-judged) supersedes any sub-goal progress.
        if pose is not None and self._waypoint is not None:
            if _dist(pose, self._waypoint) <= self._arrival_tol():
                # Only stop at the terminal waypoint. For intermediates, leave
                # Tier-3 driving and let the next SELECT_SUBGOAL goto supersede
                # it (seamless via cmd_id) — no decelerate-stop at each hop.
                if self._runner.is_terminal_leg():
                    self._io.cancel()
                self._to(HierState.ADVANCE_WAYPOINT)
                return

        st = self._io.latest_status()
        if st is not None and int(st.get("cmd_id", 0)) == self._cmd_id:
            state = st.get("state")
            # ARRIVED is published for a single Tier-3 tick before it drops
            # the goal and reverts to IDLE; at our slower poll we usually see
            # the IDLE. Both mean "this sub-goal is done" → re-pick the next
            # one toward the same waypoint.
            if state in ("ARRIVED", "IDLE"):
                self._blocked_repicks = 0          # sub-goal progress
                self._to(HierState.SELECT_SUBGOAL)
                return
            if state in ("BLOCKED", "FAULT"):
                self._enter_blocked(st.get("blocked_reason") or state)
                return
            if state == "CANCELED":
                self._to(HierState.FAILED)
                return

        # Mid-leg drift: re-pick if the bearing to the waypoint has moved on.
        if pose is not None and self._waypoint is not None and self._sent_bearing is not None:
            b = bearing_to_waypoint(pose[0], pose[1], pose[2], self._waypoint[0], self._waypoint[1])
            if abs(b - self._sent_bearing) > self._cfg.repick_hysteresis_rad:
                self._to(HierState.SELECT_SUBGOAL)

    def _tick_advance(self) -> None:
        next_idx, _lap_done = self._runner.on_arrived()
        logger.info("hier: waypoint reached -> next=%s", next_idx)
        self._blocked_repicks = 0                  # fresh leg
        if next_idx is None:
            self._io.cancel()
            self._subgoal_body = None
            self._to(HierState.ARRIVED)
        else:
            self._to(HierState.SELECT_SUBGOAL)

    def _tick_blocked(self) -> None:
        # rotate_repick: retry selection up to the cap (the scan may clear, or
        # Tier-3's own fan finds a way); past the cap, hold as a pause (the
        # consecutive-block counter is only cleared by real progress) so we
        # surface the block to the operator instead of thrashing gotos.
        if self._blocked_repicks <= self._cfg.max_blocked_repicks:
            self._to(HierState.SELECT_SUBGOAL)

    # ── Helpers ──────────────────────────────────────────────────────

    def _raster(self):
        scan = self._io.latest_scan()
        if not scan:
            return None
        return rasterize_scan(
            scan.get("ranges"),
            float(scan.get("angle_min", 0.0)),
            float(scan.get("angle_increment", 0.0)),
            self._cfg.raster_cfg,
        )
