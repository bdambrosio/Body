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
from typing import Any, Dict, Optional, Sequence, Tuple

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


@dataclass
class DepthPoints:
    """Valid depth pixels of one frame (inside the ROI) in the body frame."""
    pts: np.ndarray        # (N, 3) body-frame x fwd, y left, z up (m)
    uv: np.ndarray         # (N, 2) int pixel (u, v) in the full depth image
    height: np.ndarray     # (N,) z above ``ground_z_body_m``
    width: int
    height_px: int


def depth_points_body(
    arr_mm: np.ndarray, msg: Dict[str, Any], cfg: DepthVetoConfig,
) -> Optional[DepthPoints]:
    """ROI + median filter + intrinsics + cam→body for one depth frame. None
    when the frame or ROI is empty. Shared by the veto and the depth memory
    so both see exactly the same points."""
    h, w = arr_mm.shape
    if h < 2 or w < 2:
        return None
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
        return None

    sl = work[v0:v1, u0:u1]
    vv_r, uu_r = np.nonzero(sl > 0)
    if vv_r.size == 0:
        return None
    vv = vv_r + v0
    uu = uu_r + u0
    z_c = work[vv, uu].astype(np.float64) / 1000.0
    xc = (uu.astype(np.float64) - cx) * z_c / fx
    yc = (vv.astype(np.float64) - cy) * z_c / fy
    pc = np.stack([xc, yc, z_c], axis=1)
    pb = pc @ r_bc.T + t_bc
    return DepthPoints(pts=pb, uv=np.stack([uu, vv], axis=1), height=pb[:, 2] - cfg.ground_z_body_m,
                       width=int(w), height_px=int(h))


def obstacle_mask(dp: DepthPoints, region: np.ndarray, cfg: DepthVetoConfig) -> np.ndarray:
    """Obstacle points inside ``region`` (bool over dp's points): above the
    floor band, under the clearance height, and — for low points — standing
    on a floor cell with real vertical extent (see module doc). The span is
    measured over every point of ``region`` (floor returns included)."""
    x = dp.pts[:, 0]
    y = dp.pts[:, 1]
    height = dp.height
    band = cfg.floor_band_m + cfg.floor_band_per_m * np.maximum(x, 0.0)
    in_envelope = region & (height > band) & (height <= cfg.clearance_height_m)
    if not np.any(in_envelope) or cfg.min_vertical_extent_m <= 0.0:
        return in_envelope
    low = in_envelope & (height <= cfg.structure_free_height_m)
    out = in_envelope & ~low
    if np.any(low):
        cs = max(1e-3, float(cfg.structure_cell_m))
        sel = region & (height <= cfg.clearance_height_m)
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
        low_idx = np.nonzero(low)[0]
        lx = np.floor(x[low_idx] / cs).astype(np.int64) - ix0
        ly = np.floor(y[low_idx] / cs).astype(np.int64) - iy0
        ok = span[lx * nyc + ly] >= cfg.min_vertical_extent_m
        out[low_idx[ok]] = True
    return out


_POINTS_CACHE: Dict[str, Any] = {"key": None, "val": None}


def depth_points_for_msg(
    depth_msg: Dict[str, Any], cfg: DepthVetoConfig,
) -> Optional[DepthPoints]:
    """Decode + ``depth_points_body`` for a depth message, cached for the
    last frame (by ``seq``/``ts`` and config). The veto (every control tick)
    and the depth memory share one median filter per frame — it dominates
    the cost (~4 ms on the desktop)."""
    fk = depth_frame_key(depth_msg)
    data = depth_msg.get("data")
    # The data hash guards against two different frames sharing a key (no
    # seq + same ts); hashing ~20 kB of base64 is microseconds.
    key = None if fk is None or not isinstance(data, str) else (fk, hash(data), cfg)
    if key is not None and _POINTS_CACHE["key"] == key:
        return _POINTS_CACHE["val"]
    decoded = decode_depth_mm(depth_msg)
    val = None if decoded is None else depth_points_body(decoded[0], depth_msg, cfg)
    if key is not None:
        _POINTS_CACHE["key"] = key
        _POINTS_CACHE["val"] = val
    return val


@dataclass
class SlabHits:
    """Veto evidence for one frame: count, and where (for the status / VLM)."""
    hits: int = 0
    bbox_px: Optional[Tuple[int, int, int, int, int, int]] = None  # u0,v0,u1,v1,w,h
    hit_body: Optional[np.ndarray] = None                          # (N, 2) body x, y


def slab_hits_detail(
    arr_mm: np.ndarray,
    msg: Dict[str, Any],
    cfg: DepthVetoConfig,
    *,
    v_mps: Optional[float] = None,
    omega_radps: Optional[float] = None,
) -> SlabHits:
    """``count_slab_hits`` plus the pixel bbox and body x/y of the hits."""
    return _slab_hits_from_points(depth_points_body(arr_mm, msg, cfg), cfg,
                                  v_mps=v_mps, omega_radps=omega_radps)


def _slab_hits_from_points(
    dp: Optional[DepthPoints], cfg: DepthVetoConfig, *,
    v_mps: Optional[float], omega_radps: Optional[float],
    path_body: Optional[Sequence[Tuple[float, float]]] = None,
    denoise: Any = None,
) -> SlabHits:
    """``path_body`` (body-frame polyline from the robot, e.g. Tier-3's
    planned path): the corridor follows IT instead of the instantaneous
    commanded arc — while turning, the arc projected 0.8 m ahead swept
    through things the path curved away from within 0.3 m (10-31: every
    surviving would-block was a real object beside the path). ``denoise``
    (DenoiseConfig) applies the blob gates to the points in the corridor."""
    if dp is None:
        return SlabHits()
    if path_body is not None and len(path_body) >= 2:
        s_arc, lateral = _path_coords(dp.pts[:, 0], dp.pts[:, 1], path_body)
    else:
        s_arc, lateral = _arc_coords(dp.pts[:, 0], dp.pts[:, 1], v_mps, omega_radps)
    in_corridor = (
        (s_arc >= cfg.min_range_m)
        & (s_arc <= cfg.max_range_m)
        & (lateral <= cfg.lateral_half_width_m)
    )
    if denoise is not None and getattr(denoise, "enabled", False):
        from body.lib.depth_denoise import denoised_hits   # lazy: it imports this module
        n, idx, _blobs = denoised_hits(dp, in_corridor, cfg, denoise)
        if n == 0 or idx is None:
            return SlabHits()
        m = np.zeros(len(dp.pts), dtype=bool)
        m[idx] = True
    else:
        m = obstacle_mask(dp, in_corridor, cfg)
    n = int(np.count_nonzero(m))
    if n == 0:
        return SlabHits()
    uv = dp.uv[m]
    bbox = (int(uv[:, 0].min()), int(uv[:, 1].min()),
            int(uv[:, 0].max()), int(uv[:, 1].max()), dp.width, dp.height_px)
    return SlabHits(hits=n, bbox_px=bbox, hit_body=dp.pts[m, :2])


def _path_coords(
    x: np.ndarray, y: np.ndarray, path_body: Sequence[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """(arc length along ``path_body`` at the nearest point, lateral distance
    to it) for body points. The path starts at the robot; segments of zero
    length are skipped."""
    pts = [(float(a), float(b)) for a, b in path_body]
    if len(pts) < 2:
        return x, np.abs(y)
    best_lat = np.full(x.shape, np.inf)
    best_s = np.zeros(x.shape)
    walked = 0.0
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        ex, ey = bx - ax, by - ay
        L2 = ex * ex + ey * ey
        if L2 < 1e-12:
            continue
        t = np.clip(((x - ax) * ex + (y - ay) * ey) / L2, 0.0, 1.0)
        px, py = ax + t * ex, ay + t * ey
        lat = np.hypot(x - px, y - py)
        closer = lat < best_lat
        best_lat = np.where(closer, lat, best_lat)
        best_s = np.where(closer, walked + t * math.sqrt(L2), best_s)
        walked += math.sqrt(L2)
    return best_s, best_lat


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
    return slab_hits_detail(arr_mm, msg, cfg, v_mps=v_mps, omega_radps=omega_radps).hits


def depth_frame_detail(
    depth_msg: Optional[Dict[str, Any]],
    *,
    now_wall: float,
    v_mps: float,
    omega_radps: float,
    cfg: DepthVetoConfig,
    path_body: Optional[Sequence[Tuple[float, float]]] = None,
    denoise: Any = None,
) -> SlabHits:
    """Hits this frame (with location), or empty when the veto should not
    fire (fail-open / gated)."""
    if not cfg.enabled:
        return SlabHits()
    if abs(v_mps) < 1e-3:
        return SlabHits()  # pure rotation — same policy as swept veto
    if abs(omega_radps) > cfg.max_abs_omega_radps:
        return SlabHits()
    if depth_msg is None:
        return SlabHits()
    age = now_wall - float(depth_msg.get("ts", 0.0))
    if age < 0.0 or age > cfg.stale_s:
        return SlabHits()
    return _slab_hits_from_points(depth_points_for_msg(depth_msg, cfg), cfg,
                                  v_mps=v_mps, omega_radps=omega_radps,
                                  path_body=path_body, denoise=denoise)


def depth_frame_hits(
    depth_msg: Optional[Dict[str, Any]],
    *,
    now_wall: float,
    v_mps: float,
    omega_radps: float,
    cfg: DepthVetoConfig,
) -> int:
    """Hits this frame, or 0 when the veto should not fire (fail-open / gated)."""
    return depth_frame_detail(depth_msg, now_wall=now_wall, v_mps=v_mps,
                              omega_radps=omega_radps, cfg=cfg).hits


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

    def __init__(self, cfg: DepthVetoConfig, denoise: Any = None) -> None:
        self.cfg = cfg
        self.denoise = denoise                # DenoiseConfig or None (raw pixel count)
        self.streak = 0
        self._last_key: Optional[Any] = None
        # Evidence for the last scored frame, reported with a depth_block so a
        # desktop observer can tie the veto to a depth frame and camera image.
        self.last_hits = 0
        self.last_seq: Optional[int] = None
        self.last_bbox_px: Optional[Tuple[int, int, int, int, int, int]] = None
        self.last_hit_body: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.streak = 0
        self._last_key = None
        self.last_hits = 0
        self.last_seq = None
        self.last_bbox_px = None
        self.last_hit_body = None

    def update(
        self, depth_msg: Optional[Dict[str, Any]], *, now_wall: float,
        v_mps: float, omega_radps: float,
        path_body: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> bool:
        cfg = self.cfg
        if depth_msg is None:
            self.reset()
            return False
        age = now_wall - float(depth_msg.get("ts", 0.0) or 0.0)
        if age < 0.0 or age > cfg.stale_s:
            self.reset()
            return False
        det = depth_frame_detail(depth_msg, now_wall=now_wall, v_mps=v_mps,
                                 omega_radps=omega_radps, cfg=cfg,
                                 path_body=path_body, denoise=self.denoise)
        hits = det.hits
        hit = hits >= cfg.min_hits
        self.last_hits = int(hits)
        self.last_bbox_px = det.bbox_px
        self.last_hit_body = det.hit_body
        seq = depth_msg.get("seq")
        self.last_seq = (int(seq) if isinstance(seq, (int, float))
                         and not isinstance(seq, bool) else None)
        key = depth_frame_key(depth_msg)
        if key is None or key != self._last_key:
            self.streak = self.streak + 1 if hit else 0
            self._last_key = key
        return hit and self.streak >= max(1, cfg.hit_streak)
