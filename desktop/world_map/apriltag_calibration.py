"""AprilTag calibration — loads ``config/apriltag_poses.yaml`` and
provides the SE(3) math that turns a detection (T_cam_tag) into an
implied bot world pose (x_w, y_w, θ_w).

Frame conventions
-----------------
- **World frame (W):** x forward (toward room-front for a freshly
  rebound session), y left, z up. Standard mobile-robotics ENU-ish.
- **Body frame (B):** x forward (bot heading), y left, z up. Origin at
  the bot's rotation center.
- **Camera frame (C):** x right, y down, z forward (OpenCV / OAK-D).

YAML schema (config/apriltag_poses.yaml)
----------------------------------------
::

    camera:
      intrinsics:
        fx: 798.0
        fy: 798.0
        cx: 320.0
        cy: 200.0
      # Camera mount expressed as T_body_cam: where the camera sits in
      # the body frame. Translation in meters, rotation as ZYX Euler in
      # degrees (yaw → pitch → roll, applied right-to-left).
      mount:
        x_m: 0.10            # camera 10 cm forward of body center
        y_m: 0.00
        z_m: 0.15            # 15 cm above body origin
        yaw_deg: 0.0
        pitch_deg: 0.0
        roll_deg: -90.0      # rotates camera +z to align with body +x

    # Default tag size if a tag entry doesn't override it. The size is
    # the edge length of the BLACK border of the printed tag (canonical
    # AprilTag convention).
    tag_size_m: 0.10

    tags:
      0:
        x_m: 2.50
        y_m: 0.00
        z_m: 1.00
        yaw_deg: 180.0       # tag faces -x in world (toward origin)
        pitch_deg: 0.0
        roll_deg: 0.0
        sigma_xy_m: 0.05     # observation σ on (x, y), meters
        sigma_theta_deg: 5.0 # observation σ on θ, degrees
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from .apriltag_detector import CameraIntrinsics

logger = logging.getLogger(__name__)


# ── SE(3) helpers ─────────────────────────────────────────────────────


def _R_from_euler_zyx(yaw_rad: float, pitch_rad: float, roll_rad: float) -> np.ndarray:
    """ZYX intrinsic Euler → rotation matrix. R = Rz(yaw) · Ry(pitch) · Rx(roll).

    Applied to a body-frame vector v, ``R · v`` gives the corresponding
    vector in the parent (world) frame.
    """
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    cr, sr = math.cos(roll_rad), math.sin(roll_rad)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    return Rz @ Ry @ Rx


def make_transform(
    x: float, y: float, z: float,
    yaw_rad: float, pitch_rad: float, roll_rad: float,
) -> np.ndarray:
    """Build a 4×4 homogeneous SE(3) transform from translation + Euler."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _R_from_euler_zyx(yaw_rad, pitch_rad, roll_rad)
    T[:3, 3] = (x, y, z)
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """SE(3) inverse: [R, t; 0 1] → [Rᵀ, -Rᵀt; 0 1]. Avoids np.linalg.inv."""
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def yaw_from_R(R: np.ndarray) -> float:
    """Extract the world-frame yaw (rotation around +z) from a rotation
    matrix. Assumes the body has small roll/pitch — typical for a wheeled
    robot on a flat floor. atan2(R[1, 0], R[0, 0]) is the standard ZYX
    Tait-Bryan yaw."""
    return math.atan2(float(R[1, 0]), float(R[0, 0]))


# ── Calibration dataclasses ───────────────────────────────────────────


@dataclass(frozen=True)
class AprilTagWorldPose:
    """One tag's known world pose + observation uncertainties."""
    tag_id: int
    T_world_tag: np.ndarray         # 4×4 float64
    sigma_xy_m: float
    sigma_theta_rad: float
    tag_size_m: float


@dataclass(frozen=True)
class AprilTagCalibration:
    """Top-level config: camera intrinsics + body-camera mount + tags."""
    intrinsics: CameraIntrinsics
    T_body_cam: np.ndarray          # 4×4 float64
    default_tag_size_m: float
    tags: Dict[int, AprilTagWorldPose] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> "AprilTagCalibration":
        # Imported lazily so module import doesn't fail if pyyaml isn't
        # installed (tests that don't load YAML still work).
        try:
            import yaml
        except ImportError as e:
            raise RuntimeError(
                "AprilTag calibration loader needs PyYAML. "
                "Install with: uv pip install pyyaml"
            ) from e
        with Path(path).open("r") as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw, source=str(path))

    @classmethod
    def from_dict(cls, raw: Dict[str, Any], *, source: str = "<dict>") -> "AprilTagCalibration":
        cam = raw.get("camera") or {}
        intr_raw = cam.get("intrinsics") or {}
        try:
            intrinsics = CameraIntrinsics(
                fx=float(intr_raw["fx"]),
                fy=float(intr_raw["fy"]),
                cx=float(intr_raw["cx"]),
                cy=float(intr_raw["cy"]),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"{source}: camera.intrinsics malformed: {e}") from e

        mount = cam.get("mount") or {}
        T_body_cam = make_transform(
            x=float(mount.get("x_m", 0.0)),
            y=float(mount.get("y_m", 0.0)),
            z=float(mount.get("z_m", 0.0)),
            yaw_rad=math.radians(float(mount.get("yaw_deg", 0.0))),
            pitch_rad=math.radians(float(mount.get("pitch_deg", 0.0))),
            roll_rad=math.radians(float(mount.get("roll_deg", 0.0))),
        )

        default_tag_size = float(raw.get("tag_size_m", 0.10))
        tags_raw = raw.get("tags") or {}
        tags: Dict[int, AprilTagWorldPose] = {}
        for tag_id_raw, tdef in tags_raw.items():
            try:
                tag_id = int(tag_id_raw)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"{source}: tag key {tag_id_raw!r} not an integer"
                ) from e
            if tag_id in tags:
                raise ValueError(f"{source}: duplicate tag id {tag_id}")
            try:
                T_world_tag = make_transform(
                    x=float(tdef.get("x_m", 0.0)),
                    y=float(tdef.get("y_m", 0.0)),
                    z=float(tdef.get("z_m", 0.0)),
                    yaw_rad=math.radians(float(tdef.get("yaw_deg", 0.0))),
                    pitch_rad=math.radians(float(tdef.get("pitch_deg", 0.0))),
                    roll_rad=math.radians(float(tdef.get("roll_deg", 0.0))),
                )
                sigma_xy = float(tdef.get("sigma_xy_m", 0.05))
                sigma_theta = math.radians(float(tdef.get("sigma_theta_deg", 5.0)))
                tag_size = float(tdef.get("tag_size_m", default_tag_size))
            except (TypeError, ValueError) as e:
                raise ValueError(f"{source}: tag {tag_id} malformed: {e}") from e
            if sigma_xy <= 0 or sigma_theta <= 0 or tag_size <= 0:
                raise ValueError(
                    f"{source}: tag {tag_id} σ_xy, σ_θ, and tag_size must be > 0"
                )
            tags[tag_id] = AprilTagWorldPose(
                tag_id=tag_id,
                T_world_tag=T_world_tag,
                sigma_xy_m=sigma_xy,
                sigma_theta_rad=sigma_theta,
                tag_size_m=tag_size,
            )
        return cls(
            intrinsics=intrinsics,
            T_body_cam=T_body_cam,
            default_tag_size_m=default_tag_size,
            tags=tags,
        )


# ── The observation math ──────────────────────────────────────────────


def implied_body_world_pose(
    T_world_tag: np.ndarray,
    T_cam_tag: np.ndarray,
    T_body_cam: np.ndarray,
) -> Tuple[float, float, float]:
    """Compute the implied body pose in the world frame.

    Chain:
        T_world_tag = T_world_body · T_body_cam · T_cam_tag
    Solve:
        T_world_body = T_world_tag · T_cam_tag⁻¹ · T_body_cam⁻¹

    Project to SE(2): (x, y) from the world-frame translation, θ from
    the yaw of the world-frame rotation. The bot is on a flat floor so
    pitch and roll of T_world_body are approximately zero; small
    deviations from this fold into the yaw extraction's noise budget.
    """
    T_world_body = T_world_tag @ invert_transform(T_cam_tag) @ invert_transform(T_body_cam)
    x = float(T_world_body[0, 3])
    y = float(T_world_body[1, 3])
    theta = yaw_from_R(T_world_body[:3, :3])
    return (x, y, theta)
