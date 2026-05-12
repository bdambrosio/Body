"""Motion primitives — small, self-contained "do this one thing" units.

Used today as recovery actions (Phase 2c plugs them into
RecoveryPolicy). Built to satisfy `recovery.RecoveryPrimitive`:

    name() -> str                         # short label for logs / UI
    update(pose, costmap) -> Output       # tick — returns running/done/aborted
    cancel() -> None                      # operator-or-mission abort

Each primitive integrates its own progress from the pose stream (yaw
delta for Rotate360, position delta for BackUp). They DO NOT integrate
their own commanded velocities or rely on dead reckoning — pose is the
ground truth.

Pose freshness is the caller's responsibility. The mission tick checks
pose age before dispatching to the recovery, so primitives can assume
`pose is not None` on the first call. (They still tolerate a None pose
defensively — the very first call, before pose has arrived, just reports
RUNNING with zero cmd_vel.)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from desktop.world_map.costmap import Costmap

from .recovery import (
    PRIM_ABORTED, PRIM_DONE, PRIM_RUNNING, PrimitiveOutput,
)
from .safety import SafetyConfig, rear_arc_blocked


Pose = Tuple[float, float, float]


# ── Rotate360 ────────────────────────────────────────────────────────


class Rotate360:
    """Spin in place until the integrated yaw delta reaches the target.

    Defaults are conservative for v1: 0.30 rad/s keeps scan-match (when
    promoted) comfortably locked, and the full 2π takes ~21 s, which is
    long enough to gather a thorough local-map sweep.

    Direction is +1 (CCW) by default. Picking a direction at random or
    based on which side of the robot has more unknown cells could be
    smarter; not necessary for v1.

    `cmd_vel`: (v=0, omega=±omega_radps). Motor-driver-side ramp handles
    the start transient; we publish the target step.
    """

    def __init__(
        self,
        target_angle_rad: float = 2.0 * math.pi,
        omega_radps: float = 0.30,
        direction: int = +1,
    ):
        self.target_angle_rad = float(target_angle_rad)
        self.omega_radps = float(omega_radps)
        self.direction = +1 if direction >= 0 else -1
        self._start_yaw: Optional[float] = None
        self._prev_yaw: Optional[float] = None
        self._integrated_rad: float = 0.0
        self._canceled: bool = False

    def name(self) -> str:
        return f"rotate_360({math.degrees(self.target_angle_rad):.0f}°)"

    def update(
        self,
        pose: Optional[Pose],
        costmap: Optional[Costmap],
    ) -> PrimitiveOutput:
        if self._canceled:
            return PrimitiveOutput.aborted(note="canceled")
        if pose is None:
            return PrimitiveOutput.running(note="awaiting pose")

        yaw = pose[2]
        if self._start_yaw is None:
            self._start_yaw = yaw
            self._prev_yaw = yaw
            return PrimitiveOutput.running(
                v=0.0, omega=self.omega_radps * self.direction,
                note="rotating 0/{:.0f}°".format(
                    math.degrees(self.target_angle_rad)
                ),
            )

        # Integrate the wrapped delta so we accumulate true rotation
        # rather than getting stuck at ±π. Sign tracks direction so
        # accidental backspin (e.g. operator nudges) doesn't count.
        # Note: `or yaw` would be wrong when prev_yaw == 0.0 (falsy);
        # use an explicit None check.
        prev = self._prev_yaw if self._prev_yaw is not None else yaw
        delta = _wrap_pi(yaw - prev)
        self._prev_yaw = yaw
        # Accumulate signed by intended direction.
        self._integrated_rad += delta * self.direction
        progress = max(0.0, self._integrated_rad)
        if progress >= self.target_angle_rad:
            return PrimitiveOutput.done(
                note=f"rotated {math.degrees(progress):.0f}°"
            )
        return PrimitiveOutput.running(
            v=0.0, omega=self.omega_radps * self.direction,
            note=f"rotating {math.degrees(progress):.0f}/"
                 f"{math.degrees(self.target_angle_rad):.0f}°",
        )

    def cancel(self) -> None:
        self._canceled = True


# ── RotateToHeading ─────────────────────────────────────────────────


@dataclass
class RotateToHeadingConfig:
    # Spin rate; sign tracks direction (handled per-tick from the
    # measured error). Default matches Rotate360 — comfortable for
    # scan-match when SLAM is promoted.
    omega_radps: float = 0.30
    # Stop tolerance. ±5° is tighter than the follower's 35°
    # rotate-in-place gate, so on hand-off back to FOLLOWING the
    # robot is already aligned enough for smooth pure-pursuit.
    tolerance_rad: float = math.radians(5.0)


class RotateToHeading:
    """Spin in place until the robot's yaw is within `tolerance_rad`
    of `target_theta_rad` (world frame). Direction is picked from the
    sign of the (wrapped) heading error each tick, so a small
    overshoot self-corrects rather than spinning a full loop.

    Used by patrol execution at each waypoint to face the next leg's
    bearing before the follower takes over. Idempotent on pose-loss:
    if pose temporarily disappears, the primitive holds (returns
    RUNNING with zero cmd_vel) until pose returns.
    """

    def __init__(
        self,
        target_theta_rad: float,
        config: Optional[RotateToHeadingConfig] = None,
    ):
        self.target_theta_rad = float(target_theta_rad)
        self.config = config or RotateToHeadingConfig()
        self._canceled: bool = False

    def name(self) -> str:
        return f"rotate_to_heading({math.degrees(self.target_theta_rad):+.0f}°)"

    def update(
        self,
        pose: Optional[Pose],
        costmap: Optional[Costmap],
    ) -> PrimitiveOutput:
        if self._canceled:
            return PrimitiveOutput.aborted(note="canceled")
        if pose is None:
            # Hold position until pose returns. Mission's no_pose
            # guard owns the eventual fail; the primitive just waits.
            return PrimitiveOutput.running(note="awaiting pose")

        yaw = pose[2]
        error = _wrap_pi(self.target_theta_rad - yaw)
        if abs(error) <= self.config.tolerance_rad:
            return PrimitiveOutput.done(
                note=f"aligned (|err|={math.degrees(abs(error)):.1f}°)"
            )
        # Sign of error picks rotation direction so a small overshoot
        # self-corrects rather than spinning all the way around.
        sign = 1.0 if error >= 0.0 else -1.0
        omega = sign * self.config.omega_radps
        return PrimitiveOutput.running(
            v=0.0, omega=omega,
            note=f"rotating, err={math.degrees(error):+.1f}°",
        )

    def cancel(self) -> None:
        self._canceled = True


# ── BackUp ──────────────────────────────────────────────────────────


@dataclass
class BackUpConfig:
    distance_m: float = 0.30
    speed_mps: float = 0.10           # absolute value; commanded as -v
    safety: SafetyConfig = None       # type: ignore[assignment]


class BackUp:
    """Drive straight back until either `distance_m` is covered or the
    rear safety arc reports a lethal cell.

    Position-integrated from pose deltas; the commanded speed is the
    target step. The rear-arc check uses the same `SafetyConfig` shape
    as the forward arc, so the wedge geometry stays consistent.
    """

    def __init__(self, config: Optional[BackUpConfig] = None):
        cfg = config or BackUpConfig()
        if cfg.safety is None:
            cfg = BackUpConfig(
                distance_m=cfg.distance_m,
                speed_mps=cfg.speed_mps,
                safety=SafetyConfig(),
            )
        self.config = cfg
        self._start_xy: Optional[Tuple[float, float]] = None
        self._traveled_m: float = 0.0
        self._canceled: bool = False

    def name(self) -> str:
        return f"back_up({self.config.distance_m:.2f}m)"

    def update(
        self,
        pose: Optional[Pose],
        costmap: Optional[Costmap],
    ) -> PrimitiveOutput:
        if self._canceled:
            return PrimitiveOutput.aborted(note="canceled")
        if pose is None:
            return PrimitiveOutput.running(note="awaiting pose")

        # Rear-arc safety: if there's a lethal cell behind us, refuse
        # to keep going. ABORTED rather than DONE — the recovery flow
        # treats this as failure-of-this-attempt and may pick a
        # different action next time.
        if costmap is not None and rear_arc_blocked(
            costmap, pose, self.config.safety,
        ):
            return PrimitiveOutput.aborted(
                note="rear arc blocked"
            )

        x, y = pose[0], pose[1]
        if self._start_xy is None:
            self._start_xy = (x, y)
            return PrimitiveOutput.running(
                v=-self.config.speed_mps, omega=0.0,
                note=f"backing 0/{self.config.distance_m:.2f} m",
            )
        sx, sy = self._start_xy
        self._traveled_m = math.hypot(x - sx, y - sy)
        if self._traveled_m >= self.config.distance_m:
            return PrimitiveOutput.done(
                note=f"backed up {self._traveled_m:.2f} m"
            )
        return PrimitiveOutput.running(
            v=-self.config.speed_mps, omega=0.0,
            note=f"backing {self._traveled_m:.2f}/"
                 f"{self.config.distance_m:.2f} m",
        )

    def cancel(self) -> None:
        self._canceled = True


# ── Helpers ─────────────────────────────────────────────────────────


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi
