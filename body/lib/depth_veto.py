"""Forward near-field depth veto for Tier-3 (last-resort stop, not a planner).

Tier-3 continues to route on the live lidar scan. This module only answers:
"does the OAK depth image show an obstacle slab in a short forward envelope
along the commanded translation?" It never steers.

Fail-open: missing / stale / undecodable depth → not blocked (lidar still
guards the scan plane). Fail-closed on a confirmed slab hit after a short
streak so stereo speckles do not spam BLOCKED.

Camera extrinsics match ``body.local_map`` (OpenCV cam → body via fixed axis
fix + ZYX euler). Pure NumPy — unit-tested off-robot.
"""
from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class DepthVetoConfig:
    enabled: bool = True
    stale_s: float = 0.5
    min_range_m: float = 0.08
    max_range_m: float = 0.80
    lateral_half_width_m: float = 0.12
    floor_band_m: float = 0.04
    clearance_height_m: float = 0.35
    ground_z_body_m: float = 0.0
    min_hits: int = 8
    hit_streak: int = 2
    # Skip veto while |ω| exceeds this (angular smear / in-place turn).
    max_abs_omega_radps: float = 0.40
    # Normalized ROI on the depth image (outer stereo edges are noisy).
    roi_u0: float = 0.20
    roi_u1: float = 0.80
    roi_v0: float = 0.25
    roi_v1: float = 0.85
    depth_median_kernel: int = 3
    depth_hfov_deg: float = 70.0
    depth_vfov_deg: float = 55.0
    depth_x_body_m: float = 0.0
    depth_y_body_m: float = 0.0
    depth_z_body_m: float = 0.09
    depth_yaw_rad: float = 0.0
    depth_pitch_rad: float = 0.14
    depth_roll_rad: float = 0.0


def _rot_x(r: float) -> np.ndarray:
    c, s = math.cos(r), math.sin(r)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(p: float) -> np.ndarray:
    c, s = math.cos(p), math.sin(p)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_z(y: float) -> np.ndarray:
    c, s = math.cos(y), math.sin(y)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _cam_to_body_Rt(cfg: DepthVetoConfig) -> Tuple[np.ndarray, np.ndarray]:
    # OpenCV cam: x right, y down, z forward → body: x forward, y left, z up
    r_fix = np.array(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64
    )
    r_e = (
        _rot_z(cfg.depth_yaw_rad)
        @ _rot_y(cfg.depth_pitch_rad)
        @ _rot_x(cfg.depth_roll_rad)
    )
    r_bc = r_e @ r_fix
    t_bc = np.array(
        [cfg.depth_x_body_m, cfg.depth_y_body_m, cfg.depth_z_body_m],
        dtype=np.float64,
    )
    return r_bc, t_bc


def _intrinsics(
    msg: Dict[str, Any], w: int, h: int, hfov: float, vfov: float
) -> Tuple[float, float, float, float]:
    k = msg.get("intrinsics")
    if isinstance(k, dict):
        try:
            return float(k["fx"]), float(k["fy"]), float(k["cx"]), float(k["cy"])
        except (KeyError, TypeError, ValueError):
            pass
    fx = (w - 1) / (2.0 * math.tan(hfov / 2.0)) if w > 1 else 1.0
    fy = (h - 1) / (2.0 * math.tan(vfov / 2.0)) if h > 1 else 1.0
    return fx, fy, (w - 1) * 0.5, (h - 1) * 0.5


def decode_depth_mm(msg: Dict[str, Any]) -> Optional[Tuple[np.ndarray, int, int]]:
    if msg.get("format") != "depth_uint16_mm":
        return None
    w = int(msg.get("width", 0))
    h = int(msg.get("height", 0))
    b64 = msg.get("data")
    if not isinstance(b64, str) or w <= 0 or h <= 0:
        return None
    try:
        raw = base64.standard_b64decode(b64)
    except (ValueError, TypeError):
        return None
    need = w * h * 2
    if len(raw) < need:
        return None
    arr = np.frombuffer(raw, dtype=np.uint16, count=w * h).reshape((h, w))
    return arr, w, h


def _median_filter_depth_mm(arr: np.ndarray, kernel: int) -> np.ndarray:
    if kernel <= 1 or kernel % 2 == 0:
        return arr
    half = kernel // 2
    padded = np.pad(arr, half, mode="constant", constant_values=0)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel, kernel))
    wf = np.where(windows > 0, windows.astype(np.float32), np.float32(np.nan))
    flat = wf.reshape(arr.shape[0], arr.shape[1], -1)
    with np.errstate(all="ignore"):
        med = np.nanmedian(flat, axis=-1)
    return np.where(np.isnan(med), 0, med).astype(np.uint16)


def count_slab_hits(
    arr_mm: np.ndarray,
    msg: Dict[str, Any],
    cfg: DepthVetoConfig,
) -> int:
    """Count body-frame depth points in the forward envelope obstacle slab."""
    h, w = arr_mm.shape
    if h < 2 or w < 2:
        return 0
    kernel = int(cfg.depth_median_kernel)
    work = _median_filter_depth_mm(arr_mm, kernel) if kernel > 1 else arr_mm
    hfov = math.radians(cfg.depth_hfov_deg)
    vfov = math.radians(cfg.depth_vfov_deg)
    fx, fy, cx, cy = _intrinsics(msg, w, h, hfov, vfov)
    r_bc, t_bc = _cam_to_body_Rt(cfg)

    u0 = int(np.clip(min(cfg.roi_u0, cfg.roi_u1) * w, 0, max(0, w - 1)))
    u1 = int(np.clip(max(cfg.roi_u0, cfg.roi_u1) * w, 0, w))
    v0 = int(np.clip(min(cfg.roi_v0, cfg.roi_v1) * h, 0, max(0, h - 1)))
    v1 = int(np.clip(max(cfg.roi_v0, cfg.roi_v1) * h, 0, h))
    if u1 <= u0 or v1 <= v0:
        return 0

    sl = work[v0:v1, u0:u1]
    vv_r, uu_r = np.nonzero(sl > 0)
    if vv_r.size == 0:
        return 0
    vv = vv_r + v0
    uu = uu_r + u0
    z_c = work[vv, uu].astype(np.float64) / 1000.0
    xc = (uu.astype(np.float64) - cx) * z_c / fx
    yc = (vv.astype(np.float64) - cy) * z_c / fy
    pc = np.stack([xc, yc, z_c], axis=1)
    pb = pc @ r_bc.T + t_bc

    x = pb[:, 0]
    y = pb[:, 1]
    z = pb[:, 2]
    height = z - cfg.ground_z_body_m
    in_envelope = (
        (x >= cfg.min_range_m)
        & (x <= cfg.max_range_m)
        & (np.abs(y) <= cfg.lateral_half_width_m)
        & (height > cfg.floor_band_m)
        & (height <= cfg.clearance_height_m)
    )
    return int(np.count_nonzero(in_envelope))


def depth_frame_hits(
    depth_msg: Optional[Dict[str, Any]],
    *,
    now_wall: float,
    v_mps: float,
    omega_radps: float,
    cfg: DepthVetoConfig,
) -> int:
    """Hits this frame, or 0 when the veto should not fire (fail-open / gated)."""
    if not cfg.enabled:
        return 0
    if abs(v_mps) < 1e-3:
        return 0  # pure rotation — same policy as swept veto
    if abs(omega_radps) > cfg.max_abs_omega_radps:
        return 0
    if depth_msg is None:
        return 0
    age = now_wall - float(depth_msg.get("ts", 0.0))
    if age < 0.0 or age > cfg.stale_s:
        return 0
    decoded = decode_depth_mm(depth_msg)
    if decoded is None:
        return 0
    arr, _w, _h = decoded
    return count_slab_hits(arr, depth_msg, cfg)


def depth_nearfield_blocked(
    depth_msg: Optional[Dict[str, Any]],
    *,
    now_wall: float,
    v_mps: float,
    omega_radps: float,
    cfg: DepthVetoConfig,
    streak: int,
) -> Tuple[bool, int]:
    """Return (blocked, new_streak). ``streak`` is consecutive hit frames."""
    hits = depth_frame_hits(
        depth_msg,
        now_wall=now_wall,
        v_mps=v_mps,
        omega_radps=omega_radps,
        cfg=cfg,
    )
    if hits >= cfg.min_hits:
        new_streak = streak + 1
    else:
        new_streak = 0
    blocked = new_streak >= max(1, cfg.hit_streak)
    return blocked, new_streak
