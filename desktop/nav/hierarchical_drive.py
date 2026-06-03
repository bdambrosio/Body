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
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Protocol, Tuple

from body.lib import schemas
from body.lib.local_costmap import (
    LocalCostmapConfig, build_local_costmap, dilate_bool,
)
from body.lib.local_planner import LocalPlanConfig
from body.lib.scan_raster import ScanRasterConfig, rasterize_scan
from body.lib.tier2_subgoal import (
    Tier2Config, bearing_to_waypoint, furthest_free_point,
)
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


class HandoffSink(Protocol):
    """Breakpoint/record sink for the Tier-1→2 and Tier-2→3 handoffs (a
    ``body.lib.handoff_gate.HandoffGate`` satisfies it). The default
    ``NullHandoffSink`` makes the drive behave exactly as before."""
    def record(self, tier: int, payload: Dict[str, Any]) -> None: ...
    def should_hold(self, tier: int) -> bool: ...
    def consume_continue(self, tier: int) -> bool: ...
    def is_armed(self, tier: int) -> bool: ...


class NullHandoffSink:
    """No-op sink: never holds, never records (production default / tests)."""
    def record(self, tier: int, payload: Dict[str, Any]) -> None:
        pass

    def should_hold(self, tier: int) -> bool:
        return False

    def consume_continue(self, tier: int) -> bool:
        return False

    def is_armed(self, tier: int) -> bool:
        return False


class PFPoseProvider:
    """PF-relative bootstrap: world pose straight from the particle filter.

    Replace with an LPR-backed provider (or one that re-anchors the PF to an
    LPR fix) when LPR lands — the orchestrator only knows ``world_pose()``.

    A *stale* pose is reported as None (no pose). ``latest_pose()`` never
    expires its estimate, so on a connectivity drop the PF keeps returning a
    frozen pose; without this the drive would keep steering on it. Staleness
    is judged by the pose source's ``odom_age_s()`` — a desktop-side monotonic
    receive age, skew-immune (no Pi timestamp in the comparison), so a clock
    offset can't spuriously null a fresh pose.
    """

    def __init__(self, fuser: Any, max_pose_age_s: float = 0.75):
        self._fuser = fuser
        self._max_pose_age_s = max_pose_age_s

    def world_pose(self) -> Optional[Pose]:
        latest = self._fuser.pose_source.latest_pose()
        if latest is None:
            return None
        age = self._fuser.pose_source.odom_age_s()
        if age is not None and age > self._max_pose_age_s:
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
    # Pose/telemetry went stale mid-drive (typically a connectivity drop).
    # Tier-3 is canceled and the drive HOLDS here — it does NOT auto-resume
    # when the pose returns. The operator must call request_resume() so the
    # bot can't silently lurch back into motion on an intermittent reconnect.
    SUSPENDED = "SUSPENDED"
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
    # Only horizon_m is used now (the cap on the projected sub-goal distance);
    # Tier-2 no longer rasterizes or does clearance — Tier-3 owns that.
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
        sink: Optional[HandoffSink] = None,
    ):
        self._runner = runner
        self._pose = pose
        self._io = io
        self._cfg = cfg or HierConfig()
        # Body-frame scan raster for the Tier-2 clear-run (must match Tier-3's
        # grid so what we hand down lands on the same local map it routes on).
        self._raster = ScanRasterConfig()
        # Tier-3's own footprint/clearance model — the clear-run marches on this
        # inflated mask so the sub-goal is a goal Tier-3 accepts without snapping
        # (incl. walls lateral to the ray). Same config as the Pi → same lethal.
        self._costmap_cfg = LocalCostmapConfig()
        self._goal_clearance_cells = LocalPlanConfig().goal_clearance_cells
        # Handoff inspector sink (HO-1/HO-2 record + breakpoint). No-op default.
        self._sink: HandoffSink = sink or NullHandoffSink()
        self._held_tier: Optional[int] = None    # which handoff we're paused at
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
        self._held_tier = None
        logger.info("hier: start (%d waypoints)", self._runner.n)
        self._to(
            HierState.ALIGNING if self._runner.current_target() is not None
            else HierState.FAILED
        )

    def stop(self) -> None:
        self._io.cancel()
        self._subgoal_body = None
        self._held_tier = None
        self._to(HierState.IDLE)

    def _enter_suspended(self, reason: str) -> None:
        # Pose/telemetry lost while driving. Revoke the in-flight goto so the
        # Pi isn't still chasing the old goal when the link returns, then hold
        # in SUSPENDED until the operator explicitly resumes.
        logger.warning("hier: suspended (%s) — holding for operator resume", reason)
        self._io.cancel()
        self._subgoal_body = None
        self._block_reason = reason
        self._to(HierState.SUSPENDED)

    def request_resume(self) -> bool:
        """Operator-initiated resume after a SUSPENDED (connectivity) hold.

        Returns True if a resume was armed. Re-acquires via ALIGNING, which
        only advances to driving once a fresh pose is available again — so a
        resume clicked while still offline simply waits rather than lurching.
        """
        if self._state is not HierState.SUSPENDED:
            return False
        logger.info("hier: operator resume from SUSPENDED")
        self._blocked_repicks = 0
        self._block_reason = None
        self._to(HierState.ALIGNING)
        return True

    def is_suspended(self) -> bool:
        return self._state is HierState.SUSPENDED

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
        # IDLE / ARRIVED / FAILED are terminal/inert. SUSPENDED is inert too —
        # it only leaves via request_resume() (operator) or stop().
        return self._state

    def _tick_aligning(self) -> None:
        # Re-alignment seam: today, just wait for a valid world pose.
        if self._pose.world_pose() is not None:
            self._to(HierState.SELECT_SUBGOAL)

    def _tick_select(self) -> None:
        pose = self._pose.world_pose()
        if pose is None:
            # Lost the pose after we were already driving — gate the resume.
            self._enter_suspended("pose_lost")
            return
        wp = self._runner.current_target()
        if wp is None:
            self._to(HierState.FAILED)
            return
        self._waypoint = (wp.x_m, wp.y_m)
        wp_dist = _dist(pose, self._waypoint)
        if wp_dist <= self._arrival_tol():
            self._to(HierState.ADVANCE_WAYPOINT)
            return

        bearing = bearing_to_waypoint(pose[0], pose[1], pose[2], wp.x_m, wp.y_m)

        # HO-1 Tier-1 → Tier-2: the chosen world-frame waypoint + bearing, plus
        # the FULL dense Tier-1 route (Tier-1 builds the whole route; only this
        # one waypoint is handed to Tier-2 each leg).
        self._sink.record(1, schemas.handoff_t1(
            pose=pose, wp=(wp.x_m, wp.y_m),
            wp_index=self._runner.wp_index, wp_total=self._runner.n,
            lap_index=getattr(self._runner, "lap_index", 0),
            terminal=self._runner.is_terminal_leg(),
            arrival_tol_m=self._arrival_tol(),
            bearing_rad=bearing, wp_dist_m=wp_dist,
            route=[(w.x_m, w.y_m) for w in self._runner.patrol.waypoints]))
        if self._hold(1):
            return

        # Tier-2 (contract I3): the sub-goal is the furthest live-visible CLEAR
        # point along the bearing in the body-frame scan grid (ray-march, back
        # off the first block), so what we hand Tier-3 is reachable ON ITS OWN
        # local map — not a blind projection that can land past a wall.
        (bx, by), src, free_dist, grid, meta = self._select_subgoal_body(bearing, wp_dist)

        # HO-2 Tier-2 → Tier-3: the body-frame sub-goal + the scan it sits on.
        # The scan grid is heavy, so only attach it when BP2 is armed.
        horizon_d = min(wp_dist, self._cfg.tier2_cfg.horizon_m)
        self._sink.record(2, schemas.handoff_t2(
            pose=pose, bearing_rad=bearing, src=src, free_dist_m=free_dist,
            subgoal_body=(bx, by),
            target_body=(horizon_d * math.cos(bearing), horizon_d * math.sin(bearing)),
            arrival_tol_m=self._cfg.subgoal_arrival_tol_m, v_max=self._cfg.sub_v_max,
            grid_rows=(grid.tolist() if grid is not None else None),
            meta=meta))
        if self._hold(2):
            return

        # Proceeding past both handoffs — clear the one-shot continue tokens.
        self._sink.consume_continue(1)
        self._sink.consume_continue(2)
        self._held_tier = None

        cid = self._io.send_goto_from_body(
            bx, by,
            arrival_tol_m=self._cfg.subgoal_arrival_tol_m,
            v_max=self._cfg.sub_v_max,
        )
        if cid is None:
            self._enter_blocked("send_failed")
            return
        self._cmd_id = cid
        self._sent_bearing = bearing
        self._subgoal_body = (bx, by)
        self._block_reason = None
        logger.info(
            "hier: goto cmd=%d pose=(%.2f,%.2f,%.0f°) wp=(%.2f,%.2f) "
            "bearing=%.0f° wp_dist=%.2f sub=(%.2f,%.2f) src=%s free=%.2f",
            cid, pose[0], pose[1], math.degrees(pose[2]), wp.x_m, wp.y_m,
            math.degrees(bearing), wp_dist, bx, by, src, free_dist)
        self._to(HierState.DRIVING_SUBGOAL)

    def _hold(self, tier: int) -> bool:
        """If ``tier``'s breakpoint is armed and unstepped, hold here. On the
        first hold of a pause, cancel the active goto so the robot stops instead
        of driving the prior sub-goal while we inspect. Returns True when the
        caller should return (paused); the state stays SELECT_SUBGOAL so the
        next tick re-records the (live) handoff and re-checks for a continue."""
        if not self._sink.should_hold(tier):
            return False
        if self._held_tier is None:
            self._io.cancel()
        self._held_tier = tier
        return True

    def _select_subgoal_body(
        self, bearing: float, wp_dist: float,
    ) -> Tuple[Tuple[float, float], str, float, Optional[Any], Optional[Dict[str, Any]]]:
        """Tier-2 sub-goal toward ``bearing``, capped at ``wp_dist``.

        Restores the clear-run guarantee (I3): march the live scan along the
        bearing and return the furthest confirmed-clear point (backed off the
        first block/unknown), so the sub-goal is reachable on Tier-3's local
        map. Only when there's no rasterizable scan do we fall back to the old
        blind horizon projection. Returns ``((bx, by), source, free_dist_m,
        grid, meta)`` — source ``clear`` (ray-march) or ``blind`` (fallback);
        ``grid``/``meta`` are the rasterized scan (None when no scan)."""
        t2 = self._cfg.tier2_cfg
        grid = meta = None
        scan = self._io.latest_scan()
        if scan is not None and scan.get("ranges"):
            grid, meta = rasterize_scan(
                scan.get("ranges"), float(scan.get("angle_min", 0.0)),
                float(scan.get("angle_increment", 0.0)), self._raster)
            # March on Tier-3's footprint-inflated + goal-clearance lethal mask
            # (its exact costmap model), not the raw scan: a "clear" point here
            # is a goal Tier-3 accepts without snapping, and the 2-D inflation
            # catches walls lateral to the ray a single ray-march would miss.
            cm = build_local_costmap(grid, meta, self._costmap_cfg)
            blocked = dilate_bool(cm.lethal, iters=self._goal_clearance_cells)
            march_grid = grid.copy()
            march_grid[blocked] = 0
            # Inflation already provides the clearance — only a half-cell backoff
            # to land cleanly inside the free region (not 0.15 m on top of it).
            march_cfg = replace(t2, backoff_m=0.5 * float(meta["resolution_m"]))
            res = furthest_free_point(march_grid, meta, bearing, march_cfg,
                                      max_dist_m=wp_dist)
            if res.ok and res.body_xy is not None:
                return res.body_xy, "clear", res.free_dist_m, grid, meta
        dist = min(wp_dist, t2.horizon_m)
        return ((dist * math.cos(bearing), dist * math.sin(bearing)),
                "blind", dist, grid, meta)

    def _tick_driving(self) -> None:
        pose = self._pose.world_pose()
        if pose is None:
            # Pose went stale while driving (typically a connectivity drop) —
            # cancel Tier-3 and hold for an explicit operator resume.
            self._enter_suspended("pose_lost")
            return
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

