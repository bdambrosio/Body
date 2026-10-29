"""Forward near-field depth veto for Tier-3 (last-resort stop, not a planner).

Tier-3 continues to route on the live lidar scan. This module only answers:
"does the OAK depth image show an obstacle slab in a short forward envelope
along the commanded translation?" It never steers.

Fail-open: missing / stale / undecodable depth → not blocked (lidar still
guards the scan plane). Fail-closed on a confirmed slab hit after a short
streak so stereo speckles do not spam BLOCKED.

Camera extrinsics match ``body.local_map`` (OpenCV cam → body via fixed axis
fix + ZYX euler). Pure NumPy — unit-tested off-robot.

Robustness (R2/R3/R7):
  * the envelope follows the commanded (v, ω) ARC, not a straight box;
  * low points (below ``structure_free_height_m``) count only where they form
    vertical structure (a height span ≥ ``min_vertical_extent_m`` within one
    ``structure_cell_m`` floor cell) — a pitched floor, braking nose-dip or a
    ramp lifts the floor as a gently tilted plane with almost no span per
    cell, while a real low obstacle has a face; ``floor_band_per_m`` can add
    a range-scaled band on top;
  * ``DepthVetoTracker`` advances the hit streak only on a NEW depth frame
    (``seq``, else ``ts``) so one frame scored on two control ticks is not
    "two frames".
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
    # Range-scaled floor band: band(x) = floor_band_m + floor_band_per_m · x
    # (absorbs a pitch error of atan(floor_band_per_m)). 0 = flat band.
    floor_band_per_m: float = 0.0
    # Vertical-structure test for low points (see module doc). A point higher
    # than structure_free_height_m always counts; a lower one only if its
    # floor cell spans ≥ min_vertical_extent_m. min_vertical_extent_m ≤ 0
    # disables the test (legacy: every point above the band counts).
    min_vertical_extent_m: float = 0.03
    structure_free_height_m: float = 0.10
    structure_cell_m: float = 0.04


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


def _arc_coords(
    x: np.ndarray, y: np.ndarray, v_mps: Optional[float], omega_radps: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """(arc length along the commanded path, lateral offset from it) for body
    points. Straight (x, |y|) when ω≈0 or no command is given."""
    if v_mps is None or omega_radps is None or abs(v_mps) < 1e-6:
        return x, np.abs(y)
    curv = omega_radps / v_mps
    if abs(curv) < 1e-3:
        return x * (1.0 if v_mps > 0 else -1.0), np.abs(y)
    r = 1.0 / curv                      # signed turn radius, centre at (0, r)
    dy = y - r
    lateral = np.abs(np.hypot(x, dy) - abs(r))
    phi = np.arctan2(r * x, -r * dy) * (1.0 if r > 0 else -1.0)
    if v_mps < 0:
        phi = -phi
    s = np.where(phi >= 0.0, phi * abs(r), -1.0)
    return s, lateral


def count_slab_hits(
    arr_mm: np.ndarray,
    msg: Dict[str, Any],
    cfg: DepthVetoConfig,
    *,
    v_mps: Optional[float] = None,
    omega_radps: Optional[float] = None,
) -> int:
    """Count body-frame depth points in the forward envelope obstacle slab.

    With ``v_mps``/``omega_radps`` the envelope is the band of half-width
    ``lateral_half_width_m`` around the commanded arc over arc length
    [min_range_m, max_range_m]; without, the legacy straight box."""
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
    s_arc, lateral = _arc_coords(x, y, v_mps, omega_radps)
    in_corridor = (
        (s_arc >= cfg.min_range_m)
        & (s_arc <= cfg.max_range_m)
        & (lateral <= cfg.lateral_half_width_m)
    )
    band = cfg.floor_band_m + cfg.floor_band_per_m * np.maximum(x, 0.0)
    in_envelope = in_corridor & (height > band) & (height <= cfg.clearance_height_m)
    if not np.any(in_envelope):
        return 0
    if cfg.min_vertical_extent_m <= 0.0:
        return int(np.count_nonzero(in_envelope))

    # Vertical-structure test for low points: per floor cell, the height span
    # of every point in the corridor (floor returns included) must reach
    # min_vertical_extent_m for a low point there to count.
    low = in_envelope & (height <= cfg.structure_free_height_m)
    hits = int(np.count_nonzero(in_envelope & ~low))
    if np.any(low):
        cs = max(1e-3, float(cfg.structure_cell_m))
        sel = in_corridor & (height <= cfg.clearance_height_m)
        ix = np.floor(x[sel] / cs).astype(np.int64)
        iy = np.floor(y[sel] / cs).astype(np.int64)
        ix0, iy0 = int(ix.min()), int(iy.min())
        nxc = int(ix.max()) - ix0 + 1
        nyc = int(iy.max()) - iy0 + 1
        lin = (ix - ix0) * nyc + (iy - iy0)
        hmax = np.full(nxc * nyc, -np.inf)
        hmin = np.full(nxc * nyc, np.inf)
        np.maximum.at(hmax, lin, height[sel])
        np.minimum.at(hmin, lin, height[sel])
        span = hmax - hmin
        lx = np.floor(x[low] / cs).astype(np.int64) - ix0
        ly = np.floor(y[low] / cs).astype(np.int64) - iy0
        hits += int(np.count_nonzero(span[lx * nyc + ly] >= cfg.min_vertical_extent_m))
    return hits


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
    return count_slab_hits(arr, depth_msg, cfg, v_mps=v_mps, omega_radps=omega_radps)


def depth_nearfield_blocked(
    depth_msg: Optional[Dict[str, Any]],
    *,
    now_wall: float,
    v_mps: float,
    omega_radps: float,
    cfg: DepthVetoConfig,
    streak: int,
) -> Tuple[bool, int]:
    """Return (blocked, new_streak). ``streak`` counts consecutive hit CALLS
    (legacy, stateless) — the control loop uses ``DepthVetoTracker``, which
    only advances on a new depth frame."""
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


def depth_frame_key(depth_msg: Optional[Dict[str, Any]]) -> Optional[Any]:
    """Identity of a depth frame: ``seq`` when the driver sends it, else the
    capture ``ts``; None when neither is present."""
    if not isinstance(depth_msg, dict):
        return None
    seq = depth_msg.get("seq")
    if isinstance(seq, (int, float)) and not isinstance(seq, bool):
        return ("seq", int(seq))
    ts = depth_msg.get("ts")
    if isinstance(ts, (int, float)) and not isinstance(ts, bool):
        return ("ts", float(ts))
    return None


class DepthVetoTracker:
    """Per-goal debounce for the near-field veto: the streak advances (or
    resets) only when a NEW depth frame is scored (R3). Re-scoring the same
    frame on a later control tick — with that tick's (v, ω) — can still
    block, but never counts as another frame. Frames with no identity (no
    ``seq`` and no ``ts``) fall back to per-call counting."""

    def __init__(self, cfg: DepthVetoConfig) -> None:
        self.cfg = cfg
        self.streak = 0
        self._last_key: Optional[Any] = None

    def reset(self) -> None:
        self.streak = 0
        self._last_key = None

    def update(
        self, depth_msg: Optional[Dict[str, Any]], *, now_wall: float,
        v_mps: float, omega_radps: float,
    ) -> bool:
        cfg = self.cfg
        if depth_msg is None:
            self.reset()
            return False
        age = now_wall - float(depth_msg.get("ts", 0.0) or 0.0)
        if age < 0.0 or age > cfg.stale_s:
            self.reset()
            return False
        hits = depth_frame_hits(depth_msg, now_wall=now_wall, v_mps=v_mps,
                                omega_radps=omega_radps, cfg=cfg)
        hit = hits >= cfg.min_hits
        key = depth_frame_key(depth_msg)
        if key is None or key != self._last_key:
            self.streak = self.streak + 1 if hit else 0
            self._last_key = key
        return hit and self.streak >= max(1, cfg.hit_streak)
