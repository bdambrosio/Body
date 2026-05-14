"""Recovery scaffolding: classifier, policy, primitive protocol.

When the mission pauses (Phase 1a/1b), a recovery policy decides what
to do about it. Phase 1c puts the *machinery* in place with stub
actions; Phase 2c plugs real motion primitives (Rotate360, BackUp) in.

The split is:

    classify_replan_failure(...)
        Looks at goal cell, robot cell, and the costmap to bucket a
        no-path failure into one of `RECOVERY_*` reasons. Stable across
        Phase 1c → 2c — the policy reads this, not the planner result
        directly.

    RecoveryPolicy.select(reason, attempts) -> RecoveryPrimitive | None
        Picks an action given a classified pause reason and the count
        of attempts already made within this mission. Returns None to
        signal "give up — fail the mission." Phase 1c policy returns
        a WaitAndResume stub for everything; Phase 2c upgrades.

    RecoveryPrimitive
        Anything callable that produces a (cmd_vel, status) per tick
        and reports DONE / ABORTED. Lives in `primitives.py` once
        Phase 2 lands; Phase 1c only ships WaitAndResume here.

The policy never touches the mission state directly — main_window
holds the mission object and applies transitions based on the
primitive's status. That keeps the policy testable as a pure function
of (reason, attempts).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import numpy as np

from desktop.world_map.costmap import Costmap


Pose = Tuple[float, float, float]
Point2 = Tuple[float, float]


# ── Classification ──────────────────────────────────────────────────


# Stable reason strings — the policy switches on these. Extending later
# is fine; main_window passes whatever it gets straight through.
REASON_NO_POSE = "no_pose"
REASON_NO_LIVE_CMD = "no_live_cmd"
REASON_GOAL_IN_UNKNOWN = "no_path:goal_in_unknown"
REASON_GOAL_IN_LETHAL_HALO = "no_path:goal_in_lethal_halo"
REASON_BOXED_IN = "no_path:boxed_in"
REASON_START_UNREACHABLE = "no_path:start_unreachable"
REASON_NO_PATH_OTHER = "no_path:other"


def classify_replan_failure(
    costmap: Optional[Costmap],
    pose: Optional[Pose],
    goal: Optional[Point2],
) -> str:
    """Bucket a planner no-path failure. Conservative when inputs are
    missing — falls through to `no_path:other` rather than guessing.

    The classifier does NOT re-run the planner. It looks only at the
    goal cell's disposition (unknown / lethal / clear) and a small
    neighborhood of the robot cell (boxed_in heuristic). This is
    enough to pick a reasonable recovery action without paying for a
    second A* run.
    """
    if costmap is None or pose is None or goal is None:
        return REASON_NO_PATH_OTHER

    res = float(costmap.meta["resolution_m"])
    ox = float(costmap.meta["origin_x_m"])
    oy = float(costmap.meta["origin_y_m"])
    nx, ny = costmap.lethal.shape

    gi, gj = _cell_at(goal, res, ox, oy)
    si, sj = _cell_at((pose[0], pose[1]), res, ox, oy)

    # Goal off the grid is rare (UI clamps) but treat as unknown — the
    # operator wants to drive there, but we don't have data.
    if not _in_bounds(gi, gj, nx, ny):
        return REASON_GOAL_IN_UNKNOWN

    if costmap.unknown[gi, gj]:
        return REASON_GOAL_IN_UNKNOWN
    if costmap.lethal[gi, gj]:
        return REASON_GOAL_IN_LETHAL_HALO

    # Goal cell is observed clear, so the failure is on the start side.
    # boxed_in: robot's cell or its 8-neighborhood is dominated by lethal
    # cells (the planner relaxes start by 5 cells; if even that fails
    # something is pinning us).
    if _boxed_in(costmap.lethal, si, sj, radius=3):
        return REASON_BOXED_IN
    if _in_bounds(si, sj, nx, ny) and costmap.lethal[si, sj]:
        return REASON_START_UNREACHABLE

    return REASON_NO_PATH_OTHER


# ── Primitive protocol ──────────────────────────────────────────────


PRIM_RUNNING = "RUNNING"
PRIM_DONE = "DONE"
PRIM_ABORTED = "ABORTED"


@dataclass
class PrimitiveOutput:
    status: str             # one of PRIM_*
    v_mps: float = 0.0
    omega_radps: float = 0.0
    note: str = ""

    @classmethod
    def running(cls, v: float = 0.0, omega: float = 0.0, note: str = "") -> "PrimitiveOutput":
        return cls(status=PRIM_RUNNING, v_mps=v, omega_radps=omega, note=note)

    @classmethod
    def done(cls, note: str = "") -> "PrimitiveOutput":
        return cls(status=PRIM_DONE, note=note)

    @classmethod
    def aborted(cls, note: str = "") -> "PrimitiveOutput":
        return cls(status=PRIM_ABORTED, note=note)


class RecoveryPrimitive(Protocol):
    """Anything that produces a cmd_vel + status per tick.

    Implementations MUST be stateful re: progress (yaw integrated, etc.)
    and MUST tolerate sparse / missing inputs (return RUNNING with zero
    cmd_vel rather than crashing on a None pose).

    name() returns a short human-readable label used in mission state
    and logs ("rotate_360", "back_up_0.30m", etc.).
    """

    def name(self) -> str: ...

    def update(
        self,
        pose: Optional[Pose],
        costmap: Optional[Costmap],
    ) -> PrimitiveOutput: ...

    def cancel(self) -> None: ...


# ── Stub primitive (Phase 1c) ───────────────────────────────────────


class WaitAndResume:
    """Stand still for `duration_s` seconds, then declare DONE.

    The "recovery action" of last resort: gives a transient blocker
    (pedestrian, rolling object, scan-match settling) time to clear on
    its own. Used by the Phase 1c stub policy for every reason; in
    Phase 2c it becomes the fallback only.
    """

    def __init__(self, duration_s: float = 2.0):
        self.duration_s = duration_s
        self._started_at: Optional[float] = None
        self._canceled: bool = False

    def name(self) -> str:
        return f"wait_and_resume({self.duration_s:.1f}s)"

    def update(
        self,
        pose: Optional[Pose],
        costmap: Optional[Costmap],
    ) -> PrimitiveOutput:
        if self._canceled:
            return PrimitiveOutput.aborted(note="canceled")
        now = time.monotonic()
        if self._started_at is None:
            self._started_at = now
        elapsed = now - self._started_at
        if elapsed >= self.duration_s:
            return PrimitiveOutput.done(note=f"waited {elapsed:.1f}s")
        return PrimitiveOutput.running(
            note=f"waiting {self.duration_s - elapsed:.1f}s",
        )

    def cancel(self) -> None:
        self._canceled = True


# ── Policy ──────────────────────────────────────────────────────────


@dataclass
class RecoveryPolicyConfig:
    wait_duration_s: float = 2.0
    back_up_distance_m: float = 0.20


class RecoveryPolicy:
    """Picks a primitive given a classified pause reason and the
    attempt count for the current mission.

    Per-reason dispatch (attempt 0):

        goal_in_unknown        → WaitAndResume (map may extend)
        goal_in_lethal_halo    → Rotate360 (gather obs around goal)
        boxed_in               → Rotate360 (clear local phantoms)
        start_unreachable      → BackUp (off the phantom we're sitting on)
        no_path:other          → Rotate360 (generic look-around)
        recovery_failed:*      → WaitAndResume
        no_pose                → never reaches here (handled in tick)

    On retries (attempt > 0) we mix it up: rotate-after-back, back-after-
    rotate, then fall back to wait. After `max_attempts` exhaustion we
    return None — main_window converts that to mission FAIL.

    Imports of `primitives` are local to `select()` to keep recovery.py
    importable from primitives.py (primitives.py imports the protocol +
    PrimitiveOutput from here).
    """

    def __init__(self, config: Optional[RecoveryPolicyConfig] = None):
        self.config = config or RecoveryPolicyConfig()

    def select(
        self,
        reason: str,
        attempts: int,
        max_attempts: int,
    ) -> Optional[RecoveryPrimitive]:
        if attempts >= max_attempts:
            return None
        # Local import — primitives.py imports from this module, so
        # avoid a circular import at module load.
        from .primitives import BackUp, BackUpConfig, Rotate360
        from .safety import SafetyConfig

        wait = WaitAndResume(duration_s=self.config.wait_duration_s)
        rotate = Rotate360()
        back = BackUp(BackUpConfig(
            distance_m=self.config.back_up_distance_m,
            safety=SafetyConfig(),
        ))

        if reason == REASON_GOAL_IN_UNKNOWN:
            return wait if attempts == 0 else rotate
        if reason == REASON_GOAL_IN_LETHAL_HALO:
            return rotate if attempts == 0 else wait
        if reason == REASON_BOXED_IN:
            return rotate if attempts == 0 else back
        if reason == REASON_START_UNREACHABLE:
            return back if attempts == 0 else rotate
        # NO_PATH_OTHER, recovery_failed:*, anything else.
        return rotate if attempts == 0 else wait


# ── Helpers ─────────────────────────────────────────────────────────


def _cell_at(
    xy: Point2, res: float, ox: float, oy: float,
) -> Tuple[int, int]:
    return (
        int(math.floor((xy[0] - ox) / res + 1e-9)),
        int(math.floor((xy[1] - oy) / res + 1e-9)),
    )


def _in_bounds(i: int, j: int, nx: int, ny: int) -> bool:
    return 0 <= i < nx and 0 <= j < ny


def _boxed_in(
    lethal: np.ndarray, i: int, j: int, *, radius: int,
) -> bool:
    """Robot is "boxed in" when its cell + radius-neighborhood is mostly
    lethal. Threshold: > 60% of the (2r+1)² window is lethal.
    """
    nx, ny = lethal.shape
    if not _in_bounds(i, j, nx, ny):
        return False
    i_lo = max(0, i - radius)
    i_hi = min(nx, i + radius + 1)
    j_lo = max(0, j - radius)
    j_hi = min(ny, j + radius + 1)
    sub = lethal[i_lo:i_hi, j_lo:j_hi]
    if sub.size == 0:
        return False
    return float(sub.mean()) > 0.60
