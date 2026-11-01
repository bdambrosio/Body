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
``send_goto_from_body``. It is stamped with ``anchor_ts`` = the ts of the
scan it was computed from, and Tier-3 anchors it into odom with its own odom
ring buffer interpolated at that instant (time-consistent; X6) — so a
constant PF↔odom yaw offset is harmless and the scan→send latency no longer
rotates the goal during turns. Without a scan (blind projection) the goal
falls back to the latest-odom anchoring.
"""
from __future__ import annotations

import enum
import logging
import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np

from body.lib import drive_config, schemas, zenoh_helpers
from body.lib.depth_memory import memory_config_from_dict, plan_cells, stamp_blocked
from body.lib.local_costmap import build_local_costmap, dilate_bool, dilate_disk
from body.lib.local_drive_core import wrap_pi
from body.lib.scan_raster import rasterize_scan
from body.lib.tier2_subgoal import (
    Tier2Config, bearing_to_waypoint, furthest_free_point,
)
from desktop.nav.patrol import PatrolRunner, passed_waypoint
from desktop.nav.recovery import RESUME
from desktop.nav.route_planner import (
    RouteConfig, RoutePlan, path_to_body, plan_route, visible_subgoal_along_path,
)

logger = logging.getLogger(__name__)

Pose = Tuple[float, float, float]   # world (x, y, theta)


class PoseProvider(Protocol):
    def world_pose(self) -> Optional[Pose]:
        """Robot world pose, or None when unavailable/stale. Re-align seam."""
        ...

    def correction_seq(self) -> int:
        """Monotone count of *discrete* pose corrections the source has applied
        (checkpoint re-anchor snaps, relocates, scan-match jumps). The driver
        watches this: when it changes mid-leg, the world pose just stepped
        (e.g. a bump corrected by the PF that odom never saw), so the current
        odom-anchored sub-goal is steering on a stale heading and must be
        re-picked. Optional — sources that never correct may omit it (the
        driver reads it defensively and treats absence as a constant 0)."""
        ...


class DriveIO(Protocol):
    def latest_scan(self) -> Optional[Dict[str, Any]]: ...
    def latest_status(self) -> Optional[Dict[str, Any]]: ...
    def send_goto_from_body(
        self, bx: float, by: float, *,
        arrival_tol_m: Optional[float] = None, v_max: Optional[float] = None,
        anchor_ts: Optional[float] = None,
    ) -> Optional[int]: ...
    def cancel(self) -> None: ...
    # Optional (read defensively): True once the goto ``cmd_id`` exhausted its
    # ack resends without any matching body/drive/status (lost on the link,
    # or the Pi's Tier-3 is not running). ``DriveClient`` provides it.
    # def ack_failed(self, cmd_id: int) -> bool: ...


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

    def correction_seq(self) -> int:
        # Best-effort: the PF's *discrete*-correction count (relocates,
        # rebinds, posterior jumps past the discrete gates). NOT n_applied —
        # that increments on every 10 Hz scan observation, which would re-pick
        # the sub-goal continuously and defeat the bearing hysteresis.
        try:
            return int(self._fuser.pose_source.correction_summary().get("n_discrete", 0))
        except Exception:
            return 0


class HierState(enum.Enum):
    IDLE = "IDLE"
    ALIGNING = "ALIGNING"
    SELECT_SUBGOAL = "SELECT_SUBGOAL"
    DRIVING_SUBGOAL = "DRIVING_SUBGOAL"
    ADVANCE_WAYPOINT = "ADVANCE_WAYPOINT"
    ARRIVED = "ARRIVED"
    BLOCKED = "BLOCKED"
    # Pose/telemetry went stale mid-drive (typically a connectivity drop), or
    # the operator took teleop. Tier-3 is canceled. With a RecoverySupervisor
    # attached (set_recovery), a pose_lost hold recovers on its own: link
    # stable -> pose verified (lidar, camera fallback) -> re-localized if
    # wrong -> resumed (desktop/nav/recovery.py). Without one, or for teleop,
    # it holds for request_resume() (operator).
    SUSPENDED = "SUSPENDED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class HierConfig:
    # Terminal stall-guard radius (PF-pose). The terminal is reached primarily
    # by the passed-vertex test (drove past it along the final leg); this is
    # only the fallback, so it MUST stay below the min waypoint spacing or it
    # fires at the penultimate vertex and stops the patrol a leg short.
    waypoint_tol_m: float = 0.15
    # Intermediate sub-waypoints advance one at a time via a passed-vertex test
    # (drove past the vertex along its route segment), NOT a radius — so closely
    # spaced corner sub-waypoints aren't skipped (the old radius cut corners).
    # The carrot is always the next un-passed vertex, so the bearing tracks the
    # Tier-1 route. pass_proximity_m is only a small stall-guard fallback.
    pass_proximity_m: float = 0.20
    subgoal_arrival_tol_m: float = 0.15   # tol passed to Tier-3 for each sub-goal
    repick_hysteresis_rad: float = 0.35   # re-pick mid-leg if the bearing drifts past this
    # After a block, retry selection at this interval for the window below (so
    # a transient — a person stepping through the lidar — clears on its own),
    # then hold as a pause for the operator (request_resume() restarts the
    # window). Time-based, NOT count-based: a retry count silently changes
    # meaning whenever the host tick rate does (it already did once, 5→20 Hz).
    blocked_retry_interval_s: float = 0.5
    blocked_retry_window_s: float = 10.0
    # Per-leg (one carrot / waypoint) deadline. Re-picks (sub-goal ARRIVED,
    # drift, pose corrections, REJECTED) deliberately do NOT reset it — each
    # re-pick is a new Tier-3 cmd_id that resets all of Tier-3's own per-goal
    # bounds, so without a leg-level bound a re-pick loop never times out
    # (X1). Only a waypoint advance or an operator resume starts a new leg.
    # Expiry → BLOCKED("leg_timeout"), paused for the operator (no retries).
    leg_deadline_s: float = 60.0
    # With the route planner a leg is marker-to-marker (9 m on long_loop,
    # 52 s at v_max before any turn): the deadline is leg_deadline_s plus
    # the planned route length at this crawl speed (10-31: leg_timeout on
    # the first clean lap's return leg). 0 disables the distance term.
    leg_deadline_mps: float = 0.10
    # REJECTED gotos (cmd_id below the Pi's watermark): the DriveClient re-seeds
    # past the watermark, so a re-pick normally succeeds at once. More than
    # this many consecutive rejections in a leg → BLOCKED("rejected").
    max_consecutive_rejects: int = 3
    # Tier-2 fan-out: when the straight clear-run toward the carrot fails (an
    # obstacle — often one only Tier-3's depth memory knows — sits on the
    # bearing), try bearings ±fan_step_deg … ±fan_max_deg and pick the one
    # closest to the bearing that (a) is clear for ≥ fan_min_free_m on the
    # same inflated mask, (b) lands within fan_corridor_m of the Tier-1 route
    # segment (prev vertex → carrot) — no shortcuts through doorways/off the
    # authored route — and (c) gets closer to the carrot. 0 disables.
    fan_max_deg: float = 75.0
    fan_step_deg: float = 15.0
    fan_min_free_m: float = 0.35
    fan_corridor_m: float = 0.75
    # Detour re-pick: a Tier-2 sub-goal is "the furthest visible clear point",
    # chosen on the obstacles known at that instant. If Tier-3's A* path to
    # it is then much longer than the straight distance (obstacles appeared —
    # typically its depth memory filling in — and it now plans a loop around
    # them: 10-31, a 2 m loop for a 0.5 m goal while spinning in place), the
    # goal is stale: re-pick, fan first. Needs detour_ticks consecutive Tier-3
    # status ticks; at most detour_max_repicks per leg, so it cannot thrash
    # cmd_ids (X1). 0 disables.
    detour_ratio: float = 3.0
    detour_min_path_m: float = 0.8
    detour_ticks: int = 5
    detour_max_repicks: int = 2
    # A retry that has really moved this far (world pose since its send) has
    # got past what blocked it: the consecutive-block retry window restarts,
    # so a second, unrelated block later in the same leg isn't charged for
    # the first one's time ("window exhausted" after 1 m of driving).
    retry_progress_m: float = 0.20
    # Tier-3 ARRIVED/IDLE at a snapped/frontier endpoint (plan_reason, also
    # carried on the IDLE after it) with the robot not having moved and the
    # sub-goal not a tiny hop = a Tier-2 "clear" goal Tier-3 knows is
    # blocked (e.g. depth memory). That is a block, not progress — a
    # goto→ARRIVED loop otherwise runs at 10 Hz with no motion, invisible to
    # the retry window, until leg_deadline_s.
    arrived_min_motion_m: float = 0.05
    # An intermediate carrot this close that Tier-2 cannot reach (no clear
    # run ≥ min_subgoal_m toward it, no fan candidate that gains on it) is
    # treated as passed: it is a route vertex, not a destination, and it is
    # sitting inside an obstacle's inflation (10-31: waypoint 3 against a
    # cabinet base; the robot parked 10–20 cm short of it, retrying
    # clear_run_failed until the window ran out). The terminal waypoint is
    # never skipped. Must exceed pass_proximity_m to matter.
    blocked_carrot_skip_m: float = 0.40
    # Use Tier-3's depth memory (status depth_mem_body) in the desktop's
    # planning (global route stamp + live path check). Off since 10-31: the
    # remembered cells, inflated, sealed corridors on the session map and
    # were the trigger in most stalls; the 20 s stamp of an actual
    # depth_block's hits keeps the one behaviour that mattered (do not
    # re-approach what just stopped you). Tier-3 still plans around its own
    # memory locally. Redesign step 4 moves persistence into the world map.
    plan_with_depth_memory: bool = False
    # Stamp a depth_block's own hits into the global plan for 20 s. Off
    # with the veto (10-31): the hits were phantoms 92 % of the time.
    stamp_block_hits: bool = False
    sub_v_max: Optional[float] = None     # per-goto speed cap (None → Tier-3 default)
    # Tier-2 clear-run uses tier2_cfg (horizon, step, min_subgoal, …) on a
    # footprint-inflated scan raster shared with Tier-3 (I8). Blind horizon
    # projection is only the no-scan fallback.
    tier2_cfg: Tier2Config = field(default_factory=Tier2Config)


@dataclass(frozen=True)
class BlockAdvisory:
    """Slow, advisory read of a block (e.g. a VLM looking at the camera frame).

    Caution-only by construction — it can stop retries, never cause motion:
      * ``hold``  — something that moves on its own (person, door): send no
        retry gotos for ``hold_s``, then resume the normal retry cadence.
      * ``pause`` — a static obstacle: give up the retry window now and wait
        for the operator, as if the window had run out.
      * ``note``  — no change in behaviour; ``label`` is shown to the operator.
    """
    action: str            # "hold" | "pause" | "note"
    label: str
    hold_s: float = 0.0


ADVISORY_ACTIONS = ("hold", "pause", "note")


def _dist(a: Pose, wp_xy: Tuple[float, float]) -> float:
    return math.hypot(a[0] - wp_xy[0], a[1] - wp_xy[1])


def _dist_to_segment(p: Tuple[float, float], a: Tuple[float, float],
                     b: Tuple[float, float]) -> float:
    ex, ey = b[0] - a[0], b[1] - a[1]
    L = ex * ex + ey * ey
    t = 0.0 if L < 1e-12 else max(0.0, min(1.0, ((p[0] - a[0]) * ex + (p[1] - a[1]) * ey) / L))
    return math.hypot(a[0] + t * ex - p[0], a[1] + t * ey - p[1])


def _world_bearing(pose: Pose, wp_xy: Tuple[float, float]) -> float:
    """World-frame bearing robot→waypoint (robot yaw NOT subtracted). Used for
    the drift re-pick test: pure in-place rotation by Tier-3 leaves it
    unchanged, so Tier-3's own rotate/realign is never mistaken for drift."""
    return math.atan2(wp_xy[1] - pose[1], wp_xy[0] - pose[0])


# Tier-3 status modes during which the robot is (re)orienting in place toward
# the sub-goal. A drift re-pick here would supersede the goal and reset the
# rotate hysteresis / realign bound on the Pi (X1).
_TIER3_TURNING_MODES = ("rotate", "realign")


class HierarchicalDrive:
    """Single-step state machine; call ``tick()`` from the host's redraw loop."""

    def __init__(
        self,
        runner: PatrolRunner,
        pose: PoseProvider,
        io: DriveIO,
        cfg: Optional[HierConfig] = None,
        sink: Optional[HandoffSink] = None,
        lead_in: Optional[list] = None,
        costmap_provider: Optional[Callable[[], Any]] = None,
        route_cfg: Optional[RouteConfig] = None,
    ):
        """``costmap_provider`` (nav redesign step 1): returns the live world
        ``Costmap`` (or None). When given, each SELECT plans a global route
        robot → (relaxed) carrot on it and hands Tier-3 the furthest
        live-visible point along that path; the straight-bearing Tier-2
        march (+ fan) is only the fallback when no costmap is available."""
        self._runner = runner
        self._costmap_provider = costmap_provider
        self._route_cfg = route_cfg or RouteConfig()
        self._route: Optional[RoutePlan] = None          # last global plan
        self._carrot_relaxed: Optional[Tuple[float, float]] = None
        self._block_hits_world: List[Tuple[float, float]] = []   # recent depth-veto hits
        self._block_hits_at: float = -1e9
        # One-time routed lead-in (start pose → first marker) the carrot follows
        # BEFORE the patrol loop, so the start isn't an un-routed beeline. Kept
        # OUT of the runner so lap/terminal accounting stays anchored on the
        # markers. lead_in[0] ≈ the start pose, lead_in[-1] == the first marker.
        self._lead_in: list = list(lead_in or [])
        self._lead_idx: int = 1
        self._pose = pose
        self._io = io
        self._cfg = cfg or HierConfig()
        # Body-frame scan raster + Tier-3's footprint/clearance model for the
        # Tier-2 clear-run: the sub-goal is marched on Tier-3's own inflated
        # lethal mask so what we hand down is a goal Tier-3 accepts without
        # snapping (incl. walls lateral to the ray). Built from config.json by
        # the SAME body.lib.drive_config builders the Pi uses — same source,
        # same lethal; parallel dataclass defaults would drift silently.
        body_cfg = zenoh_helpers.load_body_config()
        self._raster = drive_config.scan_raster_config(body_cfg)
        # Same near-robot exclusion as Tier-3 when stamping its depth memory.
        self._mem_cfg = memory_config_from_dict(
            (body_cfg.get("local_drive") or {}).get("depth_memory"))
        plan_cfg = drive_config.local_plan_config(body_cfg)
        self._costmap_cfg = plan_cfg.costmap
        self._goal_clearance_cells = plan_cfg.goal_clearance_cells
        # Handoff inspector sink (HO-1/HO-2 record + breakpoint). No-op default.
        self._sink: HandoffSink = sink or NullHandoffSink()
        self._held_tier: Optional[int] = None    # which handoff we're paused at
        self._route_start: Optional[Tuple[float, float]] = None   # prev for vertex 0
        self._detour_ticks = 0                      # consecutive detour status ticks
        self._block_skipped_via = False             # one via skip per leg
        self._detour_last_ts: Optional[float] = None
        self._detour_repicks = 0                    # per leg
        self._prefer_fan = False                    # next SELECT: fan before straight
        self._sent_src: Optional[str] = None        # clear | fan | blind of the live goto
        self._sent_pose_xy: Optional[Tuple[float, float]] = None   # world pose at send
        self._sent_dist: float = 0.0                # |sub-goal| at send
        self._prev_waypoint_xy: Optional[Tuple[float, float]] = None  # last vertex left
        self._state = HierState.IDLE
        self._cmd_id: Optional[int] = None
        self._sent_bearing: Optional[float] = None
        # World-frame bearing to the waypoint at send time (drift reference).
        self._sent_world_bearing: Optional[float] = None
        # Leg deadline: start of the current leg (None until the first select
        # of the leg). Re-picks do not touch it.
        self._leg_started_at: Optional[float] = None
        self._leg_deadline_s: float = 0.0           # this leg's budget (set at leg start)
        self._consecutive_rejects = 0
        # Pose-correction count at the moment the live sub-goal was sent. If the
        # provider's count moves past this mid-leg, the world pose stepped under
        # us → re-pick from the corrected pose. -1 until the first goto.
        self._sent_correction_seq: int = -1
        self._subgoal_body: Optional[Tuple[float, float]] = None
        self._waypoint: Optional[Tuple[float, float]] = None
        # Consecutive-block retry window: start time of the current block run
        # (None = no active block), last retry time, and a gave-up latch so
        # the pause is logged once. Only real progress (sub-goal done or a
        # waypoint advance) clears the window — a successful re-send does not.
        self._block_started_at: Optional[float] = None
        self._block_last_retry: float = 0.0
        self._block_gave_up = False
        self._block_reason: Optional[str] = None
        # Block runs are numbered so a slow advisory (answered seconds later)
        # only applies to the run it was asked about — not to a later block
        # after the robot has moved on.
        self._block_run: int = 0
        self._block_run_reason: Optional[str] = None   # reason that opened the run
        self._advisory: Optional[BlockAdvisory] = None
        self._advisory_run: int = -1
        self._advisory_hold_until: float = 0.0
        self._tick_now: float = 0.0   # tick() stamp, for handlers entered mid-tick
        # Optional self-recovery for pose_lost suspends (None = operator only).
        self._recovery: Optional[Any] = None
        self._auto_resumes = 0
        self._suspend_link_back = False

    # ── Control ──────────────────────────────────────────────────────

    def _to(self, state: HierState) -> None:
        if state is not self._state:
            logger.info("hier: %s -> %s", self._state.value, state.value)
        self._state = state

    def start(self) -> None:
        if self._recovery is not None:
            self._recovery.cancel()
        self._block_started_at = None
        self._block_gave_up = False
        self._block_reason = None
        self._cmd_id = None
        self._leg_started_at = None
        self._consecutive_rejects = 0
        self._subgoal_body = None
        self._held_tier = None
        self._route_start = None
        self._prev_waypoint_xy = None
        self._lead_idx = 1
        self._route = None
        self._carrot_relaxed = None
        self._block_hits_world = []
        self._block_skipped_via = False
        self._detour_ticks = 0
        self._detour_last_ts = None
        self._detour_repicks = 0
        self._prefer_fan = False
        logger.info("hier: start (%d waypoints, %d lead-in)",
                    self._runner.n, max(0, len(self._lead_in) - 1))
        self._to(
            HierState.ALIGNING if self._runner.current_target() is not None
            else HierState.FAILED
        )

    def stop(self) -> None:
        if self._recovery is not None:
            self._recovery.cancel()
        self._io.cancel()
        self._subgoal_body = None
        self._held_tier = None
        self._to(HierState.IDLE)

    def set_recovery(self, supervisor: Optional[Any]) -> None:
        """Attach a RecoverySupervisor (desktop/nav/recovery.py): pose_lost
        suspends then recover without the operator. None = operator only."""
        self._recovery = supervisor

    def recovery_status(self) -> Optional[Dict[str, Any]]:
        r = self._recovery
        if r is None or not r.active():
            return None
        return r.status(self._tick_now)

    def _enter_suspended(self, reason: str) -> None:
        # Pose/telemetry lost while driving. Revoke the in-flight goto so the
        # Pi isn't still chasing the old goal when the link returns. A
        # pose_lost hold recovers on its own when a supervisor is attached;
        # otherwise (and always for teleop) it waits for request_resume().
        auto = self._recovery is not None and reason == "pose_lost"
        logger.warning("hier: suspended (%s) — %s", reason,
                       "self-recovering" if auto else "holding for operator resume")
        self._io.cancel()
        self._subgoal_body = None
        self._block_reason = reason
        self._to(HierState.SUSPENDED)
        if auto:
            self._suspend_link_back = False
            self._recovery.begin(self._tick_now, reason)
        elif self._recovery is not None:
            self._recovery.cancel()

    def _tick_suspended(self, now: float) -> None:
        r = self._recovery
        if r is None or self._block_reason != "pose_lost" or not r.active():
            return
        link_ok = self._pose.world_pose() is not None
        if link_ok and not self._suspend_link_back:
            # The cancel sent on entry rode the same failing link — repeat it
            # once the link is back so Tier-3 can't still hold the old goal.
            self._io.cancel()
        self._suspend_link_back = link_ok
        if r.step(now, link_ok=link_ok, goal_live=False) == RESUME:
            self._auto_resumes += 1
            logger.info("hier: auto-resume from SUSPENDED (pose_lost) — %s",
                        r.resume_detail)
            self._resume_resets()
            self._to(HierState.ALIGNING)

    def request_resume(self) -> bool:
        """Operator-initiated resume after a SUSPENDED (connectivity) hold or
        a BLOCKED pause that exhausted its retry window.

        Returns True if a resume was armed. Re-acquires via ALIGNING, which
        only advances to driving once a fresh pose is available again — so a
        resume clicked while still offline simply waits rather than lurching.
        From BLOCKED, the retry window restarts fresh.
        """
        if self._state not in (HierState.SUSPENDED, HierState.BLOCKED):
            return False
        logger.info("hier: operator resume from %s", self._state.value)
        if self._recovery is not None:
            self._recovery.cancel()
        self._resume_resets()
        self._to(HierState.ALIGNING)
        return True

    def _resume_resets(self) -> None:
        self._block_started_at = None
        self._block_gave_up = False
        self._block_reason = None
        self._leg_started_at = None          # resume = fresh leg budget
        self._consecutive_rejects = 0

    # States in which the drive may have (or is about to issue) a live Tier-3
    # goal — the ones an operator teleop takeover must interrupt.
    _ACTIVE_STATES = (
        HierState.ALIGNING, HierState.SELECT_SUBGOAL, HierState.DRIVING_SUBGOAL,
        HierState.ADVANCE_WAYPOINT, HierState.BLOCKED,
    )

    def notify_teleop(self, active: bool) -> None:
        """Host hook: the operator armed Live-cmd / teleop (desktop cmd_vel or
        cmd_direct streaming). Tier-3 publishes its own cmd_vel at 10 Hz while
        it holds a goal; interleaved with teleop the motors see alternating
        commands (X3). Revoke the Tier-3 goal and hold SUSPENDED until the
        operator explicitly resumes. Idempotent; ``active=False`` is a no-op
        (resume stays an explicit operator action)."""
        if active and self._state in self._ACTIVE_STATES:
            self._enter_suspended("teleop")

    def is_suspended(self) -> bool:
        return self._state is HierState.SUSPENDED

    def can_resume(self) -> bool:
        """True when request_resume() would act: a SUSPENDED hold or a BLOCKED
        pause (during the retry window a resume just restarts it — harmless)."""
        return self._state in (HierState.SUSPENDED, HierState.BLOCKED)

    def held_tier(self) -> Optional[int]:
        """The handoff tier we're paused at for an armed inspector breakpoint
        (HO-1/HO-2), or None. (HO-3 holds on the Pi; detect that via the drive
        status mode == 'held'.)"""
        return self._held_tier

    def _enter_blocked(self, reason: str) -> None:
        # The retry window spans *consecutive* blocks (only reset by real
        # progress — a sub-goal done or a waypoint advance), NOT by a
        # successful re-send. Otherwise a standstill that Tier-3 vetoes every
        # time would loop SELECT↔BLOCKED forever, re-issuing the same goto.
        #
        # Always revoke the Tier-3 goal on entry (X3): a BLOCKED Tier-3 keeps
        # holding its goal and streaming cmd_vel, and resumes real motion on
        # its own if the block clears (a depth ghost vanishes) — while the UI
        # says BLOCKED/paused. Retries re-send a fresh goto anyway.
        if self._state is not HierState.BLOCKED:
            self._io.cancel()
        self._subgoal_body = None
        self._block_reason = reason
        if self._block_started_at is None:
            self._block_started_at = self._tick_now
            self._block_last_retry = self._tick_now
            self._block_run += 1
            self._block_run_reason = reason
        self._to(HierState.BLOCKED)

    # ── Block advisories (slow, caution-only) ────────────────────────

    def block_run(self) -> int:
        """Id of the current (or last) consecutive-block run. Increments when a
        new run starts; retries inside a run keep the same id."""
        return self._block_run

    def apply_block_advisory(self, run: int, adv: BlockAdvisory) -> bool:
        """Attach an advisory to block run ``run``. Ignored (returns False)
        unless that run is still live — no sub-goal progress since it started
        — so a late answer about a block the robot already got past can't stop
        it now. A run is live across its own retries; if a retry is driving
        when a hold/pause lands, the goal is revoked and we go back to BLOCKED
        (the only effect is less motion)."""
        if adv.action not in ADVISORY_ACTIONS:
            raise ValueError(f"unknown advisory action {adv.action!r}")
        if not self._block_run_live(run):
            return False
        self._advisory = adv
        self._advisory_run = run
        if adv.action != "note" and self._state is not HierState.BLOCKED:
            self._enter_blocked(self._block_reason or self._block_run_reason
                                or "advisory")
        if adv.action == "hold":
            self._advisory_hold_until = self._tick_now + max(0.0, adv.hold_s)
            logger.info("hier: advisory hold %.1fs (%s)", adv.hold_s, adv.label)
        elif adv.action == "pause" and not self._block_gave_up:
            self._block_gave_up = True
            self._io.cancel()
            logger.warning("hier: blocked (%s) — advisory says %s, pausing for "
                           "operator resume", self._block_reason, adv.label)
        return True

    def block_advisory(self) -> Optional[BlockAdvisory]:
        """The advisory for the current block run, while that run is live."""
        if self._advisory_run == self._block_run and self._block_run_live(self._block_run):
            return self._advisory
        return None

    def _block_run_live(self, run: int) -> bool:
        return (run == self._block_run
                and self._block_started_at is not None
                and self._state in (HierState.BLOCKED, HierState.SELECT_SUBGOAL,
                                    HierState.DRIVING_SUBGOAL))

    # ── Carrot: the one-time lead-in prefix, then the patrol runner ──────

    def _in_lead_in(self) -> bool:
        return self._lead_idx < len(self._lead_in)

    def _carrot_xy(self) -> Optional[Tuple[float, float]]:
        if self._in_lead_in():
            return tuple(self._lead_in[self._lead_idx])
        wp = self._runner.current_target()
        return None if wp is None else (wp.x_m, wp.y_m)

    def _carrot_terminal(self) -> bool:
        return False if self._in_lead_in() else self._runner.is_terminal_leg()

    def _display_route(self) -> list:
        """The full route for the inspector: lead-in (minus its shared last
        point) + the marker route."""
        wps = [(w.x_m, w.y_m) for w in self._runner.patrol.waypoints]
        return (list(self._lead_in[:-1]) + wps) if self._lead_in else wps

    def _carrot_index_total(self) -> Tuple[int, int]:
        total = len(self._display_route())
        if self._in_lead_in():
            return self._lead_idx, total
        offset = (len(self._lead_in) - 1) if self._lead_in else 0
        return offset + self._runner.wp_index, total

    def _prev_vertex(self) -> Tuple[float, float]:
        """The vertex just before the current carrot, defining the segment for
        the passed-test: the previous lead-in point in the lead-in; the vertex
        we last advanced from (handles loop closure) or the captured start pose
        in the patrol."""
        if self._in_lead_in():
            return tuple(self._lead_in[self._lead_idx - 1])
        if self._prev_waypoint_xy is not None:
            return self._prev_waypoint_xy
        if self._route_start is not None:
            return self._route_start
        return self._waypoint if self._waypoint is not None else (0.0, 0.0)

    def _pose_correction_seq(self) -> int:
        """Provider's discrete-correction count, read defensively (providers /
        test fakes without it report a constant 0 → the check never fires)."""
        fn = getattr(self._pose, "correction_seq", None)
        if fn is None:
            return 0
        try:
            return int(fn())
        except Exception:
            return 0

    def _reached_waypoint(self, pose: Pose) -> bool:
        """Advance off the current carrot, for terminal and intermediate alike:
        only once we've driven PAST the vertex along its route segment (t ≥ 1),
        so closely spaced sub-waypoints aren't skipped and the carrot stays the
        next un-passed vertex. The per-case proximity is just a stall-guard.

        The terminal MUST use the same passed-test, not a bare proximity radius:
        a radius >= the final leg length fires the instant we advance onto the
        terminal (we're already within it from the penultimate vertex), stopping
        the patrol a leg short. The passed-test is spacing-invariant — t ≥ 1
        triggers only after the final leg is actually driven — and the terminal
        just gets a looser proximity guard for PF-pose noise."""
        if self._waypoint is None:
            return False
        prox = (self._cfg.waypoint_tol_m if self._carrot_terminal()
                else self._cfg.pass_proximity_m)
        if passed_waypoint((pose[0], pose[1]), self._prev_vertex(),
                           self._waypoint, proximity_m=prox):
            return True
        # A via-point relaxed by the route planner (its own cell is inside an
        # obstacle's inflation): reaching the relaxed point IS reaching it.
        rc = self._carrot_relaxed
        return rc is not None and _dist(pose, rc) <= prox

    # ── Introspection (for the UI overlay / status label) ────────────

    def state(self) -> HierState:
        return self._state

    def block_reason(self) -> Optional[str]:
        return self._block_reason

    def current_subgoal_body(self) -> Optional[Tuple[float, float]]:
        return self._subgoal_body

    def current_waypoint_world(self) -> Optional[Tuple[float, float]]:
        return self._waypoint

    def current_route_world(self) -> Optional[list]:
        """The last global path (world polyline) when the route planner is
        in use, for the map preview."""
        return list(self._route.path_world) if (self._route and self._route.ok) else None

    # ── Tick ─────────────────────────────────────────────────────────

    # States that do compute-only work and should chain within a single tick,
    # so a sub-goal completion turns into the next goto in the SAME tick instead
    # of bleeding a render tick per hop (ARRIVED->SELECT->send, ADVANCE->SELECT).
    _CHAIN_STATES = (HierState.SELECT_SUBGOAL, HierState.ADVANCE_WAYPOINT)

    def tick(self, now: float) -> HierState:
        # Bounded same-tick handoff loop: keep running handlers while each one
        # hands off into an immediately-actionable state. Normal worst case is
        # DRIVING->SELECT->(ADVANCE->SELECT)->DRIVING (<=4 hops); the cap is a
        # backstop against a pathological cycle, not an expected limit.
        self._tick_now = now
        for _ in range(6):
            prev = self._state
            s = prev
            if s == HierState.ALIGNING:
                # Startup hop is its own tick — not latency-critical, and keeps
                # the first SELECT (HO-1 record / breakpoint) on its own tick.
                self._tick_aligning()
                break
            elif s == HierState.SELECT_SUBGOAL:
                self._tick_select()
            elif s == HierState.DRIVING_SUBGOAL:
                self._tick_driving()
            elif s == HierState.ADVANCE_WAYPOINT:
                self._tick_advance()
            elif s == HierState.BLOCKED:
                self._tick_blocked(now)
            elif s == HierState.SUSPENDED:
                # Self-recovery (pose_lost + supervisor); else inert until
                # request_resume() (operator) or stop().
                self._tick_suspended(now)
                break
            # IDLE / ARRIVED / FAILED are terminal/inert.
            if self._state == prev or self._state not in self._CHAIN_STATES:
                break
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
        carrot = self._carrot_xy()
        if carrot is None:
            self._to(HierState.FAILED)
            return
        if self._route_start is None:
            self._route_start = (pose[0], pose[1])   # prev for the first vertex
        self._waypoint = carrot
        wp_dist = _dist(pose, self._waypoint)
        if self._leg_started_at is None:
            self._leg_started_at = self._tick_now
            self._start_leg_budget(wp_dist)
        elif self._leg_expired():
            self._enter_blocked("leg_timeout")
            return
        if self._reached_waypoint(pose):
            self._to(HierState.ADVANCE_WAYPOINT)
            return

        cx, cy = self._waypoint
        bearing = bearing_to_waypoint(pose[0], pose[1], pose[2], cx, cy)

        # Nav redesign step 1: global route on the live world costmap. The
        # carrot may be relaxed (moved to the nearest free cell) or skipped;
        # the bearing handed down is the path's initial heading.
        route = None
        if self._costmap_provider is not None:
            route = self._plan_global_route(pose, (cx, cy), wp_dist)
            if route is False:
                return                                  # advanced / blocked
            if route is not None:
                cx, cy = route.target_world
                self._carrot_relaxed = route.target_world
                wp_dist = _dist(pose, (cx, cy))
                if wp_dist <= (self._cfg.waypoint_tol_m if self._carrot_terminal()
                               else self._cfg.pass_proximity_m):
                    self._to(HierState.ADVANCE_WAYPOINT)
                    return
                bearing = self._route_initial_bearing(route, pose, bearing)

        # HO-1 Tier-1 → Tier-2: the chosen world-frame carrot + bearing, plus the
        # FULL route (lead-in + dense markers) — Tier-2 tracks the whole route
        # vertex-by-vertex; this is just the current carrot.
        terminal = self._carrot_terminal()
        idx, total = self._carrot_index_total()
        self._sink.record(1, schemas.handoff_t1(
            pose=pose, wp=(cx, cy),
            wp_index=idx, wp_total=total,
            lap_index=getattr(self._runner, "lap_index", 0),
            terminal=terminal,
            arrival_tol_m=(self._cfg.waypoint_tol_m if terminal
                           else self._cfg.pass_proximity_m),
            bearing_rad=bearing, wp_dist_m=wp_dist,
            route=self._display_route()))
        if self._hold(1):
            return

        # Tier-2 (contract I3): furthest live-visible CLEAR point along the
        # bearing on Tier-3's footprint-inflated scan (advisory clear-run).
        # Tier-3's A* snap remains the authority. No usable clear point with a
        # live scan → BLOCKED (do not blind-project into obstacles). No scan →
        # blind horizon projection; Tier-3 reports no_scan if it cannot plan.
        prefer_fan, self._prefer_fan = self._prefer_fan, False
        if route is not None:
            selected = self._select_along_route(route, pose)
            if selected is None and not terminal and wp_dist <= self._cfg.blocked_carrot_skip_m:
                logger.info("hier: carrot %.2f m away but the path to it is not "
                            "live-clear — treating it as passed", wp_dist)
                self._to(HierState.ADVANCE_WAYPOINT)
                return
            if selected is None:
                self._enter_blocked("path_blocked_live")
                return
        else:
            selected = self._select_subgoal_body(bearing, wp_dist, pose=pose,
                                                 carrot=(cx, cy), prefer_fan=prefer_fan)
        if selected is None:
            if (not terminal and wp_dist <= self._cfg.blocked_carrot_skip_m
                    and self._io.latest_scan() is not None):
                logger.info("hier: carrot %.2f m away but no clear run to it — "
                            "treating it as passed", wp_dist)
                self._to(HierState.ADVANCE_WAYPOINT)
                return
            self._enter_blocked("clear_run_failed")
            return
        (bx, by), src, free_dist, grid, meta = selected

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

        # Time-consistent anchoring (X6): the sub-goal is in the body frame of
        # the scan it was marched on, so stamp it with that scan's ts and let
        # Tier-3 anchor it from its odom ring buffer at that instant. Blind
        # projection has no scan → latest-odom anchoring (anchor_ts None).
        kwargs: Dict[str, Any] = {
            "arrival_tol_m": self._cfg.subgoal_arrival_tol_m,
            "v_max": self._cfg.sub_v_max,
        }
        anchor_ts = (meta or {}).get("_scan_ts") if src in ("clear", "fan", "route") else None
        if anchor_ts is not None:
            kwargs["anchor_ts"] = anchor_ts
        cid = self._io.send_goto_from_body(bx, by, **kwargs)
        if cid is None:
            # Send failed (no odom/connection). A previous goto may still be
            # live on the Pi (intermediate advances deliberately leave it
            # running) — _enter_blocked revokes it so BLOCKED in the UI
            # matches an actually stopped robot.
            self._enter_blocked("send_failed")
            return
        self._cmd_id = cid
        self._sent_bearing = bearing
        self._sent_world_bearing = _world_bearing(pose, (cx, cy))
        self._sent_correction_seq = self._pose_correction_seq()
        self._sent_src = src
        self._sent_pose_xy = (pose[0], pose[1])
        self._sent_dist = math.hypot(bx, by)
        self._subgoal_body = (bx, by)
        self._block_reason = None
        logger.info(
            "hier: goto cmd=%d pose=(%.2f,%.2f,%.0f°) wp=(%.2f,%.2f)%s "
            "bearing=%.0f° wp_dist=%.2f sub=(%.2f,%.2f) src=%s free=%.2f",
            cid, pose[0], pose[1], math.degrees(pose[2]), cx, cy,
            " [lead-in]" if self._in_lead_in() else "",
            math.degrees(bearing), wp_dist, bx, by, src, free_dist)
        self._to(HierState.DRIVING_SUBGOAL)

    # ── Global route (nav redesign step 1) ───────────────────────────

    def _plan_global_route(self, pose: Pose, carrot: Tuple[float, float],
                           wp_dist: float):
        """Plan robot → carrot on the live costmap. Returns the RoutePlan,
        None to fall back to the bearing march (no costmap), or False when
        the state was changed here (carrot skipped / BLOCKED)."""
        try:
            cm = self._costmap_provider()
        except Exception:
            logger.exception("hier: costmap provider failed; bearing fallback")
            cm = None
        if cm is None:
            return None
        extra = self._extra_lethal_world(pose)
        wp = None if self._in_lead_in() else self._runner.current_target()
        must_visit = bool(getattr(wp, "must_visit", False))
        plan = plan_route(cm, (pose[0], pose[1]), carrot, self._route_cfg,
                          extra_lethal_world=extra,
                          via_radius_m=getattr(wp, "radius_m", None),
                          must_visit=must_visit)
        terminal = self._carrot_terminal() or must_visit
        if plan.ok:
            self._route = plan
            if self._leg_started_at is not None and self._tick_now - self._leg_started_at < 1.0:
                self._start_leg_budget(max(wp_dist, plan.distance_m))   # the real route length
            if plan.reason == "frontier":
                logger.info("hier: via (%.2f,%.2f) sealed off — heading for the reachable "
                            "point nearest it (%.2f,%.2f), %.2f m short", carrot[0], carrot[1],
                            plan.target_world[0], plan.target_world[1], plan.relaxed_m)
            elif plan.relaxed_m > 0.01:
                logger.info("hier: via (%.2f,%.2f) relaxed %.2f m to (%.2f,%.2f)",
                            carrot[0], carrot[1], plan.relaxed_m,
                            plan.target_world[0], plan.target_world[1])
            return plan
        self._route = None
        if plan.reason == "via_unreachable":
            if terminal:
                logger.warning("hier: %s waypoint (%.2f,%.2f) has no free cell "
                               "within its radius — blocked",
                               "must-visit" if must_visit else "terminal",
                               carrot[0], carrot[1])
                self._enter_blocked("terminal_unreachable")
                return False
            logger.info("hier: via (%.2f,%.2f) has no free cell within %.2f m — skipped",
                        carrot[0], carrot[1], self._route_cfg.via_radius_m)
            self._to(HierState.ADVANCE_WAYPOINT)
            return False
        logger.info("hier: no global route to (%.2f,%.2f): %s", carrot[0], carrot[1],
                    plan.reason)
        self._dump_planning_map(cm, pose, carrot, extra, plan.reason)
        self._enter_blocked("no_route")
        return False

    _dump_last_at = -1e9

    def _dump_planning_map(self, cm, pose, carrot, extra, reason) -> None:
        """Save the planning costmap + pose + via on a failed plan (at most
        one per 5 s) to ~/body-logs/plan-fail-<ts>.npz, so a no_route can
        be reproduced offline instead of inferred from the trace."""
        import os
        import time as _time
        if self._tick_now - self._dump_last_at < 5.0:
            return
        self._dump_last_at = self._tick_now
        try:
            d = os.path.expanduser("~/body-logs")
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"plan-fail-{int(_time.time())}.npz")
            np.savez_compressed(
                path, lethal=cm.lethal, cost=np.asarray(cm.cost, dtype=np.float32),
                unknown=cm.unknown, meta=np.array([cm.meta], dtype=object),
                pose=np.array(pose, dtype=float), via=np.array(carrot, dtype=float),
                extra=np.array(extra, dtype=float).reshape(-1, 2), reason=np.array(reason))
            logger.info("hier: planning map dumped to %s", path)
        except Exception:
            logger.exception("hier: planning map dump failed")

    def _extra_lethal_world(self, pose: Pose) -> List[Tuple[float, float]]:
        """Obstacles Tier-3 knows that the world map may not: its depth
        memory (every status) and the hits of a recent depth_block."""
        out: List[Tuple[float, float]] = []
        c, s = math.cos(pose[2]), math.sin(pose[2])
        mem = self._depth_memory_body() or []
        for p in mem:
            out.append((pose[0] + c * p[0] - s * p[1], pose[1] + s * p[0] + c * p[1]))
        if self._cfg.stamp_block_hits and self._tick_now - self._block_hits_at <= 20.0:
            out.extend(self._block_hits_world)
        return out

    def _note_block_hits(self, st: Dict[str, Any], pose: Optional[Pose]) -> None:
        hits = st.get("depth_hit_body") if isinstance(st, dict) else None
        if not hits or pose is None:
            return
        c, s = math.cos(pose[2]), math.sin(pose[2])
        self._block_hits_world = [
            (pose[0] + c * p[0] - s * p[1], pose[1] + s * p[0] + c * p[1]) for p in hits]
        self._block_hits_at = self._tick_now

    @staticmethod
    def _route_initial_bearing(route: RoutePlan, pose: Pose, default: float) -> float:
        """Body-frame heading of the path's first ≥ 0.1 m."""
        for x, y in route.path_world[1:]:
            if math.hypot(x - pose[0], y - pose[1]) >= 0.1:
                return wrap_pi(math.atan2(y - pose[1], x - pose[0]) - pose[2])
        return default

    def _select_along_route(self, route: RoutePlan, pose: Pose):
        """Furthest live-visible point along the global path (same inflated
        scan mask Tier-2 marched on). None when no scan or the first
        min_subgoal_m of the path is not clear."""
        live = self._live_march_grid()
        if live is None:
            # No scan: blind — the first horizon of the path (Tier-3 reports
            # no_scan if it cannot plan).
            pb = path_to_body(route.path_world, pose)
            d = 0.0
            for a, b in zip(pb, pb[1:]):
                d += math.hypot(b[0] - a[0], b[1] - a[1])
                if d >= self._route_cfg.horizon_m:
                    return b, "blind", d, None, None
            return pb[-1], "blind", d, None, None
        march_grid, path_grid, grid, meta = live
        pb = path_to_body(route.path_world, pose)
        hit = visible_subgoal_along_path(
            pb, march_grid, meta, self._route_cfg, path_grid=path_grid,
            robot_clear_radius_m=self._raster.robot_clear_radius_m)
        if hit is None:
            return None
        (bx, by), free = hit
        return (bx, by), "route", free, grid, meta

    def _live_march_grid(self):
        """(goal_grid, path_grid, grid, meta) from the latest scan, depth
        memory stamped: goal_grid = Tier-3's full lethal (+ goal clearance)
        mask — where a goal is accepted as-is; path_grid = footprint-only
        inflation — where the body physically fits. None without a scan."""
        scan = self._io.latest_scan()
        if scan is None or not scan.get("ranges"):
            return None
        grid, meta = rasterize_scan(
            scan.get("ranges"), float(scan.get("angle_min", 0.0)),
            float(scan.get("angle_increment", 0.0)), self._raster,
            unknown_bins=scan.get("unknown_bins"))
        mem = self._depth_memory_body()
        if mem:
            grid = grid.copy()
            cells = plan_cells(np.asarray(mem, dtype=float), (0.0, 0.0, 0.0),
                               self._mem_cfg.plan_exclude_radius_m,
                               self._mem_cfg.plan_exclude_half_angle_deg)
            stamp_blocked(grid, meta, cells, keep_clear_xy=(0.0, 0.0),
                          keep_clear_radius_m=self._raster.robot_clear_radius_m)
        cm = build_local_costmap(grid, meta, self._costmap_cfg)
        blocked = dilate_bool(cm.lethal, iters=self._goal_clearance_cells)
        march_grid = grid.copy()
        march_grid[blocked] = 0
        res = float(meta["resolution_m"])
        core = dilate_disk(grid == 0, self._costmap_cfg.footprint_radius_m / res) | (grid == 0)
        path_grid = grid.copy()
        path_grid[core] = 0
        meta = dict(meta)
        ts = scan.get("ts")
        if isinstance(ts, (int, float)) and ts > 0:
            meta["_scan_ts"] = float(ts)
        return march_grid, path_grid, grid, meta

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
        # Time parked at an inspector breakpoint is not leg time: restart the
        # leg budget when the operator continues.
        self._leg_started_at = None
        return True

    def _select_subgoal_body(
        self, bearing: float, wp_dist: float, *,
        pose: Optional[Pose] = None, carrot: Optional[Tuple[float, float]] = None,
        prefer_fan: bool = False,
    ) -> Optional[Tuple[Tuple[float, float], str, float, Optional[Any], Optional[Dict[str, Any]]]]:
        """Tier-2 sub-goal toward ``bearing``, capped at ``wp_dist``.

        March the live scan along the bearing on Tier-3's footprint-inflated
        lethal mask; return the furthest confirmed-clear point. Returns None
        when a scan is present but the clear-run finds no usable point (caller
        should BLOCKED). Blind horizon projection only when there is no
        rasterizable scan. Returns ``((bx, by), source, free_dist_m, grid,
        meta)`` — source ``clear``, ``fan`` (side bearing, see
        ``HierConfig.fan_*``; needs ``pose`` + ``carrot``) or ``blind``.
        """
        t2 = self._cfg.tier2_cfg
        grid = meta = None
        scan = self._io.latest_scan()
        if scan is not None and scan.get("ranges"):
            grid, meta = rasterize_scan(
                scan.get("ranges"), float(scan.get("angle_min", 0.0)),
                float(scan.get("angle_increment", 0.0)), self._raster,
                unknown_bins=scan.get("unknown_bins"))
            # Tier-3's depth memory (obstacles only the depth camera saw, in
            # the live body frame): march around them too, so the sub-goal
            # isn't picked straight through what stopped the last goal.
            mem = self._depth_memory_body()
            if mem:
                grid = grid.copy()
                cells = plan_cells(np.asarray(mem, dtype=float), (0.0, 0.0, 0.0),
                                   self._mem_cfg.plan_exclude_radius_m,
                                   self._mem_cfg.plan_exclude_half_angle_deg)
                stamp_blocked(grid, meta, cells, keep_clear_xy=(0.0, 0.0),
                              keep_clear_radius_m=self._raster.robot_clear_radius_m)
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
            src = "clear"
            if prefer_fan:
                # Detour re-pick: the straight pick is what Tier-3 is looping
                # around — try the sides first, straight only as fallback.
                fan = self._fan_subgoal(march_grid, meta, march_cfg, bearing,
                                        wp_dist, pose, carrot)
                if fan is not None:
                    res, src = fan, "fan"
            straight_ok = res.ok and res.body_xy is not None
            # Fan when the bearing is blocked outright, or only a short hop
            # (< fan_min_free_m) short of an obstacle — inching up to it just
            # to fail next tick wastes a cycle. A short straight hop is still
            # the fallback when no side bearing qualifies.
            short = (straight_ok and res.free_dist_m < self._cfg.fan_min_free_m
                     and res.free_dist_m < wp_dist - 1e-6)
            if src != "fan" and (not straight_ok or short):
                fan = self._fan_subgoal(march_grid, meta, march_cfg, bearing,
                                        wp_dist, pose, carrot)
                if fan is not None:
                    res, src = fan, "fan"
            if res is not None and res.ok and res.body_xy is not None:
                meta = dict(meta)
                ts = scan.get("ts")
                if isinstance(ts, (int, float)) and ts > 0:
                    meta["_scan_ts"] = float(ts)
                return res.body_xy, src, res.free_dist_m, grid, meta
            return None
        dist = min(wp_dist, t2.horizon_m)
        return ((dist * math.cos(bearing), dist * math.sin(bearing)),
                "blind", dist, grid, meta)

    def _fan_subgoal(self, march_grid, meta, march_cfg, bearing: float,
                     wp_dist: float, pose: Optional[Pose],
                     carrot: Optional[Tuple[float, float]]):
        """Side-bearing clear-run when the straight one failed (see
        ``HierConfig.fan_*``). Every candidate is a clear-run on the same
        inflated mask (unknown = blocked), so the robot→sub-goal segment is
        live-visible free space — it can't cross a wall. The route-corridor
        and progress gates keep it from wandering off the authored route.
        Smallest deviation wins; ties → closer to the carrot."""
        cfg = self._cfg
        if (pose is None or carrot is None or cfg.fan_max_deg <= 0
                or cfg.fan_step_deg <= 0):
            return None
        prev = self._prev_vertex()
        c, s = math.cos(pose[2]), math.sin(pose[2])
        n = int(math.floor(cfg.fan_max_deg / cfg.fan_step_deg + 1e-9))
        for k in range(1, n + 1):
            best = None
            for sign in (1.0, -1.0):
                dev = sign * math.radians(k * cfg.fan_step_deg)
                r = furthest_free_point(march_grid, meta, bearing + dev, march_cfg,
                                        max_dist_m=wp_dist)
                if not r.ok or r.body_xy is None or r.free_dist_m < cfg.fan_min_free_m:
                    continue
                # Pull the point back along its own (clear) ray until it is
                # inside the route corridor: a sidestep, never a dive through
                # an off-route doorway. Points on the clear ray are free on
                # the inflated mask, so Tier-3 accepts them as-is.
                ang = math.atan2(r.body_xy[1], r.body_xy[0])
                ca, sa = math.cos(ang), math.sin(ang)
                d = r.free_dist_m
                step = float(march_cfg.step_m)
                wx = wy = 0.0
                d_wp = math.inf
                while d >= cfg.fan_min_free_m:
                    bx, by = d * ca, d * sa
                    wx = pose[0] + c * bx - s * by
                    wy = pose[1] + s * bx + c * by
                    d_wp = math.hypot(carrot[0] - wx, carrot[1] - wy)
                    if (_dist_to_segment((wx, wy), prev, carrot) <= cfg.fan_corridor_m
                            and d_wp < wp_dist - 0.02):
                        break
                    d -= step
                if d < cfg.fan_min_free_m:
                    continue
                if d < r.free_dist_m:
                    r = replace(r, body_xy=(d * ca, d * sa), free_dist_m=d)
                if best is None or d_wp < best[0]:
                    best = (d_wp, r, dev)
            if best is not None:
                logger.info("hier: fan-out %+.0f° (straight bearing blocked) "
                            "free=%.2f d_wp=%.2f", math.degrees(best[2]),
                            best[1].free_dist_m, best[0])
                return best[1]
        return None

    def _depth_memory_body(self) -> Optional[list]:
        """``depth_mem_body`` from the latest Tier-3 status, if any (and the
        desktop is configured to plan with it)."""
        if not self._cfg.plan_with_depth_memory:
            return None
        st = self._io.latest_status()
        mem = st.get("depth_mem_body") if isinstance(st, dict) else None
        return mem if isinstance(mem, list) and mem else None

    def _tick_driving(self) -> None:
        pose = self._pose.world_pose()
        if pose is None:
            # Pose went stale while driving (typically a connectivity drop) —
            # cancel Tier-3 and hold for an explicit operator resume.
            self._enter_suspended("pose_lost")
            return
        # Waypoint reached (PF-judged) supersedes any sub-goal progress.
        if pose is not None and self._waypoint is not None:
            if self._reached_waypoint(pose):
                # Only stop at the terminal waypoint. For intermediates (incl.
                # lead-in points), leave Tier-3 driving and let the next
                # SELECT_SUBGOAL goto supersede it (seamless via cmd_id).
                if self._carrot_terminal():
                    self._io.cancel()
                self._to(HierState.ADVANCE_WAYPOINT)
                return

        if self._leg_expired():
            self._enter_blocked("leg_timeout")
            return

        st = self._io.latest_status()
        tier3_mode = None
        if st is not None and int(st.get("cmd_id", 0) or 0) == self._cmd_id:
            state = st.get("state")
            tier3_mode = st.get("mode")
            if tier3_mode == "held":
                # Parked at the Pi-side HO-3 breakpoint: inspection time is
                # not leg time.
                self._leg_started_at = self._tick_now
            if state == "REJECTED":
                # Our cmd_id fell at/below the Pi's watermark (another client
                # pushed it up, or local_drive restarted ahead of us). The
                # DriveClient re-seeds past the watermark on this status, so a
                # re-pick sends an accepted id. Resends of an accepted id are
                # absorbed by Tier-3 without a REJECTED, so every REJECTED is
                # real (watermark == our id means the goto itself was malformed).
                wm = st.get("watermark")
                self._consecutive_rejects += 1
                if self._consecutive_rejects > self._cfg.max_consecutive_rejects:
                    self._enter_blocked("rejected")
                    return
                logger.warning("hier: goto cmd=%s REJECTED (watermark=%s) — re-pick",
                               self._cmd_id, wm)
                self._to(HierState.SELECT_SUBGOAL)
                return
            else:
                self._consecutive_rejects = 0
            # ARRIVED is published for a single Tier-3 tick before it drops
            # the goal and reverts to IDLE; at our slower poll we usually see
            # the IDLE. Both mean "this sub-goal is done" → re-pick the next
            # one toward the same waypoint.
            if state in ("ARRIVED", "IDLE"):
                moved = self._moved_since_send(pose)
                if (st.get("plan_reason") in ("snapped", "frontier")
                        and moved < self._cfg.arrived_min_motion_m
                        and self._sent_dist >= self._cfg.subgoal_arrival_tol_m + 0.1):
                    logger.info("hier: %s without motion (sub-goal %.2f m away, "
                                "moved %.2f m) — treating as a block", state,
                                self._sent_dist, moved)
                    self._enter_blocked("arrived_no_motion")
                    return
                self._block_started_at = None      # sub-goal progress
                self._block_gave_up = False
                self._detour_repicks = 0
                self._detour_ticks = 0
                logger.info("hier: re-pick (sub-goal %s)", state)
                self._to(HierState.SELECT_SUBGOAL)
                return
            if (state == "DRIVING" and self._block_started_at is not None
                    and self._moved_since_send(pose) >= self._cfg.retry_progress_m):
                logger.info("hier: retry drove %.2f m — block window reset",
                            self._moved_since_send(pose))
                self._block_started_at = None
                self._block_gave_up = False
            if state in ("BLOCKED", "FAULT"):
                self._note_block_hits(st, pose)
                self._enter_blocked(st.get("blocked_reason") or state)
                return
            if state == "CANCELED":
                self._to(HierState.FAILED)
                return
        elif self._cmd_id is not None and self._ack_failed(self._cmd_id):
            # The goto (and its resends) never produced a matching status:
            # lost on the link or Tier-3 is not running. Don't wait forever
            # in DRIVING_SUBGOAL on a robot that is sitting still (X5).
            self._enter_blocked("no_ack")
            return

        if st is not None and int(st.get("cmd_id", 0) or 0) == self._cmd_id \
                and st.get("state") == "DRIVING" and self._detour_check(st):
            return

        # World-pose correction (re-anchor snap / relocate the odom frame never
        # saw): the current sub-goal is anchored in odom and now points the wrong
        # way relative to the corrected pose — re-pick from the corrected pose
        # at once instead of waiting for ARRIVED or for drift to cross the gate.
        cur_seq = self._pose_correction_seq()
        if self._sent_correction_seq >= 0 and cur_seq != self._sent_correction_seq:
            logger.info("hier: re-pick (pose correction seq %d->%d)",
                        self._sent_correction_seq, cur_seq)
            self._to(HierState.SELECT_SUBGOAL)
            return

        # Mid-leg drift: re-pick if the WORLD-frame bearing to the waypoint has
        # moved on (the robot translated off the line). World frame, not body:
        # Tier-3 rotating in place toward the sub-goal changes the body-relative
        # bearing but not the world one, and re-picking on it superseded the
        # goal every ~20° of rotation, resetting Tier-3's rotate hysteresis and
        # bounds (X1). Belt and braces: never drift-re-pick while Tier-3 says
        # it is rotating/realigning. The difference must be wrapped — a tiny
        # real drift across ±π would otherwise read as ~2π.
        # A fan sub-goal moves the robot off the bearing ON PURPOSE; the
        # bearing to the carrot changes by design and a drift re-pick would
        # abort the hop (and re-send the straight bearing that failed). Fan
        # hops run to Tier-3 ARRIVED/IDLE.
        if (self._waypoint is not None and self._sent_world_bearing is not None
                and tier3_mode not in _TIER3_TURNING_MODES
                and self._sent_src != "fan"):
            b = _world_bearing(pose, self._waypoint)
            drift = abs(wrap_pi(b - self._sent_world_bearing))
            if drift > self._cfg.repick_hysteresis_rad:
                logger.info("hier: re-pick (bearing drift %.0f° > %.0f°)",
                            math.degrees(drift),
                            math.degrees(self._cfg.repick_hysteresis_rad))
                self._to(HierState.SELECT_SUBGOAL)

    def _moved_since_send(self, pose: Pose) -> float:
        if self._sent_pose_xy is None:
            return 0.0
        return math.hypot(pose[0] - self._sent_pose_xy[0], pose[1] - self._sent_pose_xy[1])

    def _detour_check(self, st: Dict[str, Any]) -> bool:
        """See ``HierConfig.detour_*``. True when a re-pick was triggered."""
        cfg = self._cfg
        if cfg.detour_ratio <= 0:
            return False
        ts = st.get("ts")
        if ts is not None and ts == self._detour_last_ts:
            return False                    # same Tier-3 tick seen twice (20 vs 10 Hz)
        self._detour_last_ts = ts
        path = st.get("path_body_xy") or []
        plen = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:]))
        straight = float(st.get("dist_remaining_m") or 0.0)
        detour = (plen >= cfg.detour_min_path_m
                  and plen > cfg.detour_ratio * max(straight, 0.05))
        self._detour_ticks = self._detour_ticks + 1 if detour else 0
        if self._detour_ticks < max(1, cfg.detour_ticks):
            return False
        self._detour_ticks = 0
        if self._detour_repicks >= cfg.detour_max_repicks:
            return False                    # let Tier-3 drive its detour
        self._detour_repicks += 1
        self._prefer_fan = True
        logger.info("hier: re-pick (Tier-3 path %.2f m for %.2f m straight — "
                    "sub-goal stale; fan first, re-pick %d/%d)",
                    plen, straight, self._detour_repicks, cfg.detour_max_repicks)
        self._to(HierState.SELECT_SUBGOAL)
        return True

    def _tick_advance(self) -> None:
        # Lead-in prefix: just step the lead-in carrot. The patrol runner is
        # untouched until the lead-in delivers us to the first marker.
        if self._in_lead_in():
            self._prev_waypoint_xy = tuple(self._lead_in[self._lead_idx])
            self._lead_idx += 1
            self._carrot_relaxed = None
            self._block_started_at = None
            self._block_gave_up = False
            self._leg_started_at = None
            self._consecutive_rejects = 0
            self._detour_repicks = 0
            self._block_skipped_via = False
            self._to(HierState.SELECT_SUBGOAL)
            return
        # Remember the vertex we're leaving — it's the prev for the passed-test
        # on the next carrot (handles loop closure: prev is whatever we left).
        left = self._runner.current_target()
        if left is not None:
            self._prev_waypoint_xy = (left.x_m, left.y_m)
        self._carrot_relaxed = None
        next_idx, _lap_done = self._runner.on_arrived()
        logger.info("hier: waypoint reached -> next=%s", next_idx)
        self._block_started_at = None              # fresh leg
        self._block_gave_up = False
        self._leg_started_at = None
        self._consecutive_rejects = 0
        self._detour_repicks = 0
        self._block_skipped_via = False
        if next_idx is None:
            self._io.cancel()
            self._subgoal_body = None
            self._to(HierState.ARRIVED)
        else:
            self._to(HierState.SELECT_SUBGOAL)

    def _tick_blocked(self, now: float) -> None:
        # Retry selection (the scan may clear, or Tier-3's A* finds a way) at
        # the configured interval while inside the retry window; past it, hold
        # as a pause so the block surfaces to the operator instead of thrashing
        # gotos. The window spans consecutive blocks — only real progress (or
        # an operator resume) restarts it.
        if self._block_started_at is None:        # defensive: never unset here
            self._block_started_at = now
            self._block_last_retry = now
        # Leg deadline / repeated rejection are terminal for the leg: retrying
        # would just re-trip them (and churn goto+cancel) — pause at once.
        terminal = self._block_reason in ("leg_timeout", "rejected")
        if self._block_gave_up:
            return                 # paused (window spent, or a pause advisory)
        if terminal or now - self._block_started_at > self._cfg.blocked_retry_window_s:
            # Escalation before a pause (redesign step 5, first rung): an
            # intermediate via that cannot be reached within the window is
            # skipped — it is a route hint. The terminal (or a must-visit
            # marker, or a leg_timeout/rejected) still pauses for the operator.
            wp = None if self._in_lead_in() else self._runner.current_target()
            if (not terminal and not self._carrot_terminal()
                    and not getattr(wp, "must_visit", False)
                    and not self._block_skipped_via):
                self._block_skipped_via = True
                logger.warning("hier: blocked (%s) for %.0fs at via %s — skipping it",
                               self._block_reason, now - self._block_started_at,
                               self._waypoint)
                self._block_started_at = None
                self._block_gave_up = False
                self._to(HierState.ADVANCE_WAYPOINT)
                return
            if not self._block_gave_up:
                self._block_gave_up = True
                # Re-send the revoke on give-up: the cancel on BLOCKED entry
                # may have been lost, and a paused drive must not leave Tier-3
                # holding (and possibly resuming) a goal (X3).
                self._io.cancel()
                logger.warning(
                    "hier: blocked (%s) — %s, pausing for operator resume",
                    self._block_reason,
                    "terminal for this leg" if terminal else
                    f"retry window ({self._cfg.blocked_retry_window_s:.1f}s) exhausted")
            return
        if self._advisory_run == self._block_run and now < self._advisory_hold_until:
            return
        if now - self._block_last_retry >= self._cfg.blocked_retry_interval_s:
            self._block_last_retry = now
            self._to(HierState.SELECT_SUBGOAL)


    # ── Helpers ──────────────────────────────────────────────────────

    def _leg_expired(self) -> bool:
        return (self._leg_started_at is not None
                and self._cfg.leg_deadline_s > 0
                and self._tick_now - self._leg_started_at > self._leg_deadline_s)

    def _start_leg_budget(self, dist_m: float) -> None:
        cfg = self._cfg
        extra = dist_m / cfg.leg_deadline_mps if cfg.leg_deadline_mps > 0 else 0.0
        self._leg_deadline_s = cfg.leg_deadline_s + extra

    def _ack_failed(self, cmd_id: int) -> bool:
        """DriveIO's optional ack-failure signal, read defensively (test fakes
        and older IO without it → never fails)."""
        fn = getattr(self._io, "ack_failed", None)
        if fn is None:
            return False
        try:
            return bool(fn(cmd_id))
        except Exception:
            return False
