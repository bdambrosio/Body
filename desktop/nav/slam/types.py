"""Shared dataclasses + enums for the SLAM pipeline.

Kept minimal and free of Qt/zenoh imports so these are testable in
isolation and reusable from either the fuser thread or scripts.
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Pose2D:
    x: float  # m, world frame
    y: float  # m, world frame
    theta: float  # rad, CCW positive

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.theta)

    def compose(self, dx: float, dy: float, dtheta: float) -> "Pose2D":
        """Return self ⊕ (dx, dy, dtheta) with (dx, dy) in body frame."""
        c = math.cos(self.theta)
        s = math.sin(self.theta)
        return Pose2D(
            x=self.x + c * dx - s * dy,
            y=self.y + s * dx + c * dy,
            theta=_wrap(self.theta + dtheta),
        )


@dataclass(frozen=True)
class PoseEstimate:
    pose: Pose2D
    sigma_xy_m: float       # 1-σ on x, y (isotropic for v1)
    sigma_theta_rad: float  # 1-σ on θ
    ts: float               # wall-clock or sensor timestamp


class FusionMode(str, enum.Enum):
    ROTATION_VECTOR = "rotation_vector"         # mag+accel+gyro, absolute yaw
    GAME_ROTATION_VECTOR = "game_rotation_vector"  # accel+gyro only, relative yaw
    RAW = "raw"
    UNKNOWN = "unknown"

    @classmethod
    def from_str(cls, s: Optional[str]) -> "FusionMode":
        if not s:
            return cls.UNKNOWN
        try:
            return cls(s)
        except ValueError:
            return cls.UNKNOWN


@dataclass(frozen=True)
class ImuReading:
    """Subset of body/imu payload the SLAM pipeline cares about."""
    ts: float
    gyro_z: float               # rad/s, body frame
    quat_wxyz: Optional[Tuple[float, float, float, float]]
    fusion_mode: FusionMode
    accuracy_rad: float         # per-report orientation σ from BNO085

    @classmethod
    def from_payload(cls, msg: dict) -> Optional["ImuReading"]:
        """Parse a body/imu (or legacy body/oakd/imu) JSON payload.

        Returns None on malformed payload rather than raising, so a
        single bad publisher can't tear down the consumer.
        """
        try:
            ts = float(msg.get("ts", 0.0))
            gyro = msg.get("gyro") or {}
            gz = float(gyro.get("z", 0.0))

            quat_raw = msg.get("orientation")
            quat: Optional[Tuple[float, float, float, float]] = None
            if isinstance(quat_raw, dict):
                try:
                    quat = (
                        float(quat_raw["w"]), float(quat_raw["x"]),
                        float(quat_raw["y"]), float(quat_raw["z"]),
                    )
                except (KeyError, TypeError, ValueError):
                    quat = None

            fusion = msg.get("fusion") or {}
            mode = FusionMode.from_str(fusion.get("mode"))
            acc = float(fusion.get("accuracy_rad", 0.0))
            return cls(
                ts=ts, gyro_z=gz, quat_wxyz=quat,
                fusion_mode=mode, accuracy_rad=acc,
            )
        except Exception:
            return None


@dataclass(frozen=True)
class ScoreField:
    """Per-candidate correlation score grid from a scan-match search.

    Indexed as ``field[ix, iy, ith]``; the candidate pose at that cell is

        (prior.x + dx_axis[ix],
         prior.y + dy_axis[iy],
         prior.theta + dth_axis[ith]).

    Scores are raw correlation sums in evidence units — *not* normalized
    to probabilities. The particle filter treats them as log-likelihood
    up to an additive constant: only ratios between cells matter, so the
    overall offset cancels in importance-weight normalization.

    Frame: indexed by delta-from-prior in the world frame. Two callers
    with different priors can share one field as long as their priors
    both sit inside the search window.
    """
    field: "np.ndarray"      # (Nx, Ny, Nth) float32
    dx_axis: "np.ndarray"    # (Nx,) float64, world-frame x delta
    dy_axis: "np.ndarray"    # (Ny,) float64, world-frame y delta
    dth_axis: "np.ndarray"   # (Nth,) float64, theta delta (rad)


@dataclass(frozen=True)
class ScanMatchResult:
    """Output of ScanMatcher.search.

    Fields:
    - pose: the best candidate pose (world frame).
    - score: raw score at `pose`.
    - score_prior: score at the prior pose (uncorrected).
    - improvement: score - score_prior (how much the search helped).
    - accepted: True iff improvement cleared the confidence gate.
    - search_exhausted: True iff the best candidate was at the edge of
      the search window (prior was too far off; window should grow).
    - score_field: full (Nx, Ny, Nth) score grid + axes. Populated only
      when search() was called with return_field=True.
    """
    pose: Pose2D
    score: float
    score_prior: float
    improvement: float
    accepted: bool
    search_exhausted: bool
    score_field: Optional[ScoreField] = None


def _wrap(a: float) -> float:
    """Wrap an angle to (-π, π]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def quaternion_to_yaw(q_wxyz: Tuple[float, float, float, float]) -> float:
    """Extract yaw (rotation around body z-axis) from a wxyz quaternion.

    Convention: right-handed, z-up, yaw positive CCW from +x.
    """
    w, x, y, z = q_wxyz
    # Standard ZYX Tait-Bryan yaw extraction.
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)
