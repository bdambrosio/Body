"""Fuse lidar + OAK-D depth into an egocentric 2.5D max-height grid (``body/map/local_2p5d``).

Ground plane z = 0 in body frame (+x forward, +y left, +z up). Each cell stores the maximum
sampled body-frame z from the latest lidar slice and depth frustum.

Optional **driveable** grid: no obstacle in the vertical slab (fitted floor plane +
``driveable_floor_band_m``, up to ``driveable_clearance_height_m``) using depth-ground RANSAC
at ``floor_fit_interval_s``. See docs/local_map_spec.md.
"""

from __future__ import annotations

import base64
import math
import signal
import sys
import threading
import time
import warnings
from typing import Any

import numpy as np

from body.lib import schemas, zenoh_helpers

_D_NONE = np.int8(-1)
_D_BLOCK = np.int8(0)
_D_OK = np.int8(1)


def _rot_x(r: float) -> np.ndarray:
    c, s = math.cos(r), math.sin(r)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(p: float) -> np.ndarray:
    c, s = math.cos(p), math.sin(p)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_z(y: float) -> np.ndarray:
    c, s = math.cos(y), math.sin(y)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _R_body_from_cam_euler(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """R such that p_body = R @ R_fix @ p_cam + t (camera OpenCV frame)."""
    return _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)


def _intrinsics_for_depth(
    msg: dict[str, Any], w: int, h: int, hfov: float, vfov: float
) -> tuple[float, float, float, float]:
    """Prefer device-true fx/fy/cx/cy from the depth message; fall back to hfov/vfov-derived."""
    k = msg.get("intrinsics")
    if isinstance(k, dict):
        try:
            return float(k["fx"]), float(k["fy"]), float(k["cx"]), float(k["cy"])
        except (KeyError, TypeError, ValueError):
            pass
    fx = (w - 1) / (2.0 * math.tan(hfov / 2.0)) if w > 1 else 1.0
    fy = (h - 1) / (2.0 * math.tan(vfov / 2.0)) if h > 1 else 1.0
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    return fx, fy, cx, cy


def _decode_depth_mm(msg: dict[str, Any]) -> tuple[np.ndarray, int, int] | None:
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


def _default_floor_plane(ground_z: float) -> tuple[np.ndarray, float]:
    """Horizontal plane z = ground_z in body frame: n·p + d = 0 with n = +Z."""
    n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    d = float(-ground_z)
    return n, d


def _plane_from_three(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray
) -> tuple[np.ndarray, float] | None:
    v1 = p1 - p0
    v2 = p2 - p0
    cn = np.cross(v1, v2)
    ln = float(np.linalg.norm(cn))
    if ln < 1e-9:
        return None
    n = (cn / ln).astype(np.float64)
    if n[2] < 0.0:
        n = -n
    d = float(-np.dot(n, p0))
    return n, d


def _refit_plane_svd(inlier_pts: np.ndarray) -> tuple[np.ndarray, float]:
    mu = np.mean(inlier_pts, axis=0)
    x = inlier_pts - mu
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    n = vt[-1].astype(np.float64)
    ln = float(np.linalg.norm(n))
    if ln < 1e-12:
        return _default_floor_plane(0.0)
    n /= ln
    if n[2] < 0.0:
        n = -n
    d = float(-np.dot(n, mu))
    return n, d


def _fit_floor_plane_ransac(
    pts: np.ndarray,
    *,
    iters: int,
    inlier_m: float,
    min_inliers: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, int] | None:
    npts = int(pts.shape[0])
    if npts < 3:
        return None
    best_count = 0
    best_inl: np.ndarray | None = None
    for _ in range(iters):
        ii = rng.choice(npts, size=3, replace=False)
        pl = _plane_from_three(pts[ii[0]], pts[ii[1]], pts[ii[2]])
        if pl is None:
            continue
        n, d = pl
        dist = np.abs(pts @ n + d)
        inl = dist < inlier_m
        c = int(np.count_nonzero(inl))
        if c > best_count:
            best_count = c
            best_inl = inl
    if best_inl is None or best_count < min_inliers:
        return None
    refined = pts[best_inl]
    n, d = _refit_plane_svd(refined)
    return n, d, int(best_count)


def _collect_body_points_depth_roi(
    arr_mm: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    R_bc: np.ndarray,
    t_bc: np.ndarray,
    u0: int,
    u1: int,
    v0: int,
    v1: int,
    max_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    sl = arr_mm[v0:v1, u0:u1]
    vv_r, uu_r = np.nonzero(sl > 0)
    if vv_r.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    vv = vv_r + v0
    uu = uu_r + u0
    idx = np.arange(uu.size)
    if uu.size > max_samples:
        idx = rng.choice(idx, size=max_samples, replace=False)
        uu = uu[idx]
        vv = vv[idx]
    Z = arr_mm[vv, uu].astype(np.float64) / 1000.0
    xc = (uu.astype(np.float64) - cx) * Z / fx
    yc = (vv.astype(np.float64) - cy) * Z / fy
    pc = np.stack([xc, yc, Z], axis=1)
    return pc @ R_bc.T + t_bc


def _depth_points_body_vectorized(
    arr_mm: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    R_bc: np.ndarray,
    t_bc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Valid depth pixels → body-frame (N,3), row indices v, col indices u."""
    h, w = arr_mm.shape
    v_idx, u_idx = np.nonzero(arr_mm > 0)
    if v_idx.size == 0:
        z = np.zeros((0, 3), dtype=np.float64)
        return z, v_idx, u_idx
    Z = arr_mm[v_idx, u_idx].astype(np.float64) / 1000.0
    xc = (u_idx.astype(np.float64) - cx) * Z / fx
    yc = (v_idx.astype(np.float64) - cy) * Z / fy
    pc = np.stack([xc, yc, Z], axis=1)
    pb = pc @ R_bc.T + t_bc
    return pb, v_idx, u_idx


def _median_filter_depth_mm(arr: np.ndarray, kernel: int) -> np.ndarray:
    """Reduce stereo speckle before unprojection: per-pixel median over valid neighbors only.

    Pixels with value ``0`` are treated as invalid (DepthAI convention) and omitted from the
    median; if no valid samples exist in the window, output is ``0``. Kernel must be odd and >= 3.
    """
    if kernel <= 1 or kernel % 2 == 0:
        return arr
    half = kernel // 2
    padded = np.pad(arr, half, mode="constant", constant_values=0)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel, kernel))
    wf = np.where(windows > 0, windows.astype(np.float32), np.float32(np.nan))
    flat = wf.reshape(arr.shape[0], arr.shape[1], -1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(flat, axis=-1)
    return np.where(np.isnan(med), 0, med).astype(np.uint16)


def main() -> None:
    body_cfg = zenoh_helpers.load_body_config()
    lm = body_cfg.get("local_map", {})
    if not bool(lm.get("enabled", False)):
        print(
            "local_map: disabled (local_map.enabled=false); idling so launcher does not respawn.",
            flush=True,
        )
        stop_idle = threading.Event()

        def _idle_sig(_s: int, _f: object) -> None:
            stop_idle.set()

        signal.signal(signal.SIGTERM, _idle_sig)
        signal.signal(signal.SIGINT, _idle_sig)
        while not stop_idle.is_set():
            time.sleep(1.0)
        sys.exit(0)

    lidar_cfg = body_cfg.get("lidar", {})
    oakd_cfg = body_cfg.get("oakd", {})

    res = float(lm.get("resolution_m", 0.08))
    xf = float(lm.get("extent_forward_m", 4.0))
    xb = float(lm.get("extent_back_m", 0.25))
    yl = float(lm.get("extent_left_m", 2.5))
    yr = float(lm.get("extent_right_m", 2.5))
    ground_z = float(lm.get("ground_z_body_m", 0.0))
    hz = float(lm.get("publish_hz", 2.0))
    period = 1.0 / max(0.5, hz)

    lidar_z = float(lm.get("lidar_z_body_m", lidar_cfg.get("height_above_ground_m", 0.10)))
    lidar_x = float(lm.get("lidar_x_body_m", 0.0))
    lidar_y = float(lm.get("lidar_y_body_m", 0.0))
    lidar_yaw = float(lm.get("lidar_yaw_rad", 0.0))

    depth_z = float(
        lm.get("depth_z_body_m", oakd_cfg.get("depth_camera_height_above_ground_m", 0.09))
    )
    depth_x = float(lm.get("depth_x_body_m", 0.0))
    depth_y = float(lm.get("depth_y_body_m", 0.0))
    depth_yaw = float(lm.get("depth_yaw_rad", 0.0))
    depth_pitch = float(lm.get("depth_pitch_rad", 0.0))
    depth_roll = float(lm.get("depth_roll_rad", 0.0))
    hfov = math.radians(float(lm.get("depth_hfov_deg", 73.0)))
    vfov = math.radians(float(lm.get("depth_vfov_deg", 58.0)))
    depth_median_kernel = int(lm.get("depth_median_kernel", 3))

    driveable_on = bool(lm.get("driveable_enabled", True))
    clearance_m = float(lm.get("driveable_clearance_height_m", 0.35))
    floor_band_m = float(lm.get("driveable_floor_band_m", 0.04))
    clear_frames_need = max(1, int(lm.get("driveable_clear_frames", 4)))
    floor_fit_interval_s = float(lm.get("floor_fit_interval_s", 0.5))
    floor_fit_iters = max(10, int(lm.get("floor_fit_ransac_iters", 100)))
    floor_fit_inlier_m = float(lm.get("floor_fit_inlier_m", 0.04))
    floor_fit_min_inliers = max(20, int(lm.get("floor_fit_min_inliers", 80)))
    floor_fit_max_samples = max(100, int(lm.get("floor_fit_max_samples", 1200)))
    floor_u0 = float(lm.get("floor_roi_u0", 0.10))
    floor_u1 = float(lm.get("floor_roi_u1", 0.90))
    floor_v0 = float(lm.get("floor_roi_v0", 0.60))
    floor_v1 = float(lm.get("floor_roi_v1", 1.0))
    floor_log_interval_s = float(lm.get("floor_fit_log_interval_s", 5.0))
    depth_roi_u0 = float(lm.get("depth_roi_u0", 0.05))
    depth_roi_u1 = float(lm.get("depth_roi_u1", 0.95))
    depth_roi_v0 = float(lm.get("depth_roi_v0", 0.05))
    depth_roi_v1 = float(lm.get("depth_roi_v1", 0.95))
    slab_min_pixels = max(1, int(lm.get("driveable_slab_min_pixels", 2)))
    floor_seen_min_pixels = max(1, int(lm.get("driveable_floor_min_pixels", 2)))
    unobs_decay = max(0, int(lm.get("driveable_unobs_decay_frames", 1)))
    lidar_slab_min_range = float(lm.get("lidar_slab_min_range_m", 0.15))
    lidar_slab_block_frames = max(1, int(lm.get("lidar_slab_block_frames", 2)))
    lidar_slab_min_hits = max(1, int(lm.get("lidar_slab_min_hits", 1)))
    if not driveable_on or clearance_m <= 0.0:
        driveable_on = False

    # OpenCV cam: x right, y down, z forward  ->  body: x forward, y left, z up
    R_fix = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64)
    R_e = _R_body_from_cam_euler(depth_yaw, depth_pitch, depth_roll)
    R_bc = R_e @ R_fix
    t_bc = np.array([depth_x, depth_y, depth_z], dtype=np.float64)

    origin_x = -xb
    origin_y = -yr
    nx = max(1, int(math.ceil((xf + xb) / res)))
    ny = max(1, int(math.ceil((yl + yr) / res)))

    rng = np.random.default_rng()
    floor_n, floor_d = _default_floor_plane(ground_z)
    next_floor_fit = 0.0
    last_floor_log = 0.0
    driveable_prev = np.full((nx, ny), _D_NONE, dtype=np.int8)
    clear_streak = np.zeros((nx, ny), dtype=np.int32)
    lidar_streak = np.zeros((nx, ny), dtype=np.int32)

    lock = threading.Lock()
    last_lidar: dict[str, Any] | None = None
    last_depth: dict[str, Any] | None = None
    last_odom: dict[str, Any] | None = None

    session = zenoh_helpers.open_session(body_cfg)
    stop = threading.Event()

    def on_lidar(_k: str, msg: dict[str, Any]) -> None:
        nonlocal last_lidar
        with lock:
            last_lidar = msg

    def on_depth(_k: str, msg: dict[str, Any]) -> None:
        nonlocal last_depth
        with lock:
            last_depth = msg

    def on_odom(_k: str, msg: dict[str, Any]) -> None:
        nonlocal last_odom
        with lock:
            last_odom = msg

    zenoh_helpers.declare_subscriber_json(session, "body/lidar/scan", on_lidar)
    zenoh_helpers.declare_subscriber_json(session, "body/oakd/depth", on_depth)
    zenoh_helpers.declare_subscriber_json(session, "body/odom", on_odom)

    def handle_sigterm(_sig: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    next_tick = time.monotonic()
    while not stop.is_set():
        z_acc = np.full((nx, ny), -np.inf, dtype=np.float64)
        slab_count = np.zeros((nx, ny), dtype=np.int32) if driveable_on else None
        floor_count = np.zeros((nx, ny), dtype=np.int32) if driveable_on else None
        lidar_slab_count = np.zeros((nx, ny), dtype=np.int32)
        lidar_ts: float | None = None
        depth_ts: float | None = None

        with lock:
            lmsg = last_lidar
            dmsg = last_depth
            omsg = last_odom

        now = time.monotonic()
        if driveable_on and now >= next_floor_fit and dmsg is not None:
            next_floor_fit = now + max(0.05, floor_fit_interval_s)
            dec_fit = _decode_depth_mm(dmsg)
            if dec_fit is not None:
                arr_f, wf, hf = dec_fit
                if depth_median_kernel > 1:
                    arr_f = _median_filter_depth_mm(arr_f, depth_median_kernel)
                fx_f, fy_f, cx_f, cy_f = _intrinsics_for_depth(dmsg, wf, hf, hfov, vfov)
                ui0 = int(np.clip(min(floor_u0, floor_u1) * wf, 0, max(0, wf - 1)))
                ui1 = int(np.clip(max(floor_u0, floor_u1) * wf, 0, wf))
                vi0 = int(np.clip(min(floor_v0, floor_v1) * hf, 0, max(0, hf - 1)))
                vi1 = int(np.clip(max(floor_v0, floor_v1) * hf, 0, hf))
                if ui1 > ui0 and vi1 > vi0:
                    roi_pts = _collect_body_points_depth_roi(
                        arr_f,
                        fx_f,
                        fy_f,
                        cx_f,
                        cy_f,
                        R_bc,
                        t_bc,
                        ui0,
                        ui1,
                        vi0,
                        vi1,
                        floor_fit_max_samples,
                        rng,
                    )
                    fit = _fit_floor_plane_ransac(
                        roi_pts,
                        iters=floor_fit_iters,
                        inlier_m=floor_fit_inlier_m,
                        min_inliers=floor_fit_min_inliers,
                        rng=rng,
                    )
                    if fit is not None:
                        fit_n, fit_d, fit_inliers = fit
                        if floor_log_interval_s > 0.0 and (now - last_floor_log) >= floor_log_interval_s:
                            last_floor_log = now
                            nx_, ny_, nz_ = float(fit_n[0]), float(fit_n[1]), float(fit_n[2])
                            pitch_implied = math.atan2(nx_, nz_) if nz_ != 0.0 else 0.0
                            roll_implied = math.atan2(-ny_, nz_) if nz_ != 0.0 else 0.0
                            h_origin = float(fit_d)
                            print(
                                f"[local_map] floor_fit n=({nx_:+.3f},{ny_:+.3f},{nz_:+.3f}) "
                                f"d={h_origin:+.3f}m pitch≈{math.degrees(pitch_implied):+.2f}° "
                                f"roll≈{math.degrees(roll_implied):+.2f}° inliers={fit_inliers}/{int(roi_pts.shape[0])} "
                                f"(set depth_pitch_rad≈{depth_pitch - pitch_implied:+.4f}, "
                                f"depth_roll_rad≈{depth_roll - roll_implied:+.4f} to neutralize)",
                                flush=True,
                            )

        if lmsg is not None:
            lidar_ts = float(lmsg.get("ts", 0.0))
            amin = float(lmsg.get("angle_min", 0.0))
            ainc = float(lmsg.get("angle_increment", 0.0))
            ranges = lmsg.get("ranges")
            if isinstance(ranges, list):
                lidar_xy = np.array([lidar_x, lidar_y], dtype=np.float64)
                for i, rv in enumerate(ranges):
                    if rv is None:
                        continue
                    r = float(rv)
                    if not math.isfinite(r) or r <= 0.0:
                        continue
                    th = amin + i * ainc
                    c, s = math.cos(th + lidar_yaw), math.sin(th + lidar_yaw)
                    px = lidar_xy[0] + r * c
                    py = lidar_xy[1] + r * s
                    pz = lidar_z
                    if pz <= ground_z:
                        continue
                    ix = int(math.floor((px - origin_x) / res))
                    iy = int(math.floor((py - origin_y) / res))
                    if 0 <= ix < nx and 0 <= iy < ny:
                        np.maximum.at(z_acc, (ix, iy), pz)
                        if driveable_on and r >= lidar_slab_min_range:
                            h_ab = float(
                                floor_n[0] * px + floor_n[1] * py + floor_n[2] * pz + floor_d
                            )
                            if floor_band_m < h_ab <= clearance_m:
                                lidar_slab_count[ix, iy] += 1

        if dmsg is not None:
            depth_ts = float(dmsg.get("ts", 0.0))
            dec = _decode_depth_mm(dmsg)
            if dec is not None:
                arr, w, h = dec
                if depth_median_kernel > 1:
                    arr = _median_filter_depth_mm(arr, depth_median_kernel)
                du0 = int(np.clip(min(depth_roi_u0, depth_roi_u1) * w, 0, max(0, w - 1)))
                du1 = int(np.clip(max(depth_roi_u0, depth_roi_u1) * w, 0, w))
                dv0 = int(np.clip(min(depth_roi_v0, depth_roi_v1) * h, 0, max(0, h - 1)))
                dv1 = int(np.clip(max(depth_roi_v0, depth_roi_v1) * h, 0, h))
                arr_roi = np.zeros_like(arr)
                if du1 > du0 and dv1 > dv0:
                    arr_roi[dv0:dv1, du0:du1] = arr[dv0:dv1, du0:du1]
                fx, fy, cx, cy = _intrinsics_for_depth(dmsg, w, h, hfov, vfov)
                pb, _, _ = _depth_points_body_vectorized(arr_roi, fx, fy, cx, cy, R_bc, t_bc)
                if pb.shape[0] > 0:
                    px = pb[:, 0]
                    py = pb[:, 1]
                    pz = pb[:, 2]
                    ix = np.floor((px - origin_x) / res).astype(np.int32)
                    iy = np.floor((py - origin_y) / res).astype(np.int32)
                    inside_xy = (
                        (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
                    )
                    h_ab = (
                        px * floor_n[0]
                        + py * floor_n[1]
                        + pz * floor_n[2]
                        + floor_d
                    )
                    if driveable_on and floor_count is not None:
                        floor_hit = inside_xy & (h_ab <= floor_band_m)
                        if np.any(floor_hit):
                            np.add.at(
                                floor_count, (ix[floor_hit], iy[floor_hit]), 1
                            )
                    mask_z = pz > ground_z
                    inside = mask_z & inside_xy
                    if np.any(inside):
                        ix_in = ix[inside]
                        iy_in = iy[inside]
                        pz_in = pz[inside]
                        np.maximum.at(z_acc, (ix_in, iy_in), pz_in)
                        if driveable_on and slab_count is not None:
                            slab = inside & (h_ab > floor_band_m) & (
                                h_ab <= clearance_m
                            )
                            if np.any(slab):
                                np.add.at(slab_count, (ix[slab], iy[slab]), 1)

        grid = np.where(np.isneginf(z_acc), np.nan, z_acc)

        driveable_rows: list[list[bool | None]] | None = None
        if driveable_on:
            lidar_raw = lidar_slab_count >= lidar_slab_min_hits
            lidar_streak = np.where(lidar_raw, lidar_streak + 1, 0)
            lidar_blocked = lidar_streak >= lidar_slab_block_frames
            slab_hit = (
                slab_count >= slab_min_pixels
                if slab_count is not None
                else np.zeros((nx, ny), dtype=bool)
            )
            floor_seen = (
                floor_count >= floor_seen_min_pixels
                if floor_count is not None
                else np.zeros((nx, ny), dtype=bool)
            )
            instant_block = slab_hit | lidar_blocked
            # Height grid ignores floor (pz <= ground_z); driveable still needs
            # "observed" when depth classifies samples as on the fitted floor.
            observed = ~np.isnan(grid) | floor_seen
            new_streak = clear_streak.copy()
            new_streak[instant_block] = 0
            inc = observed & ~instant_block
            new_streak[inc] = clear_streak[inc] + 1
            # Memoryless decay: cells not observed this frame forget clear
            # evidence so speckle-induced "green" does not stick forever.
            unobs = ~observed & ~instant_block
            new_streak[unobs] = np.maximum(0, clear_streak[unobs] - unobs_decay)
            clear_streak = new_streak
            ok_mask = clear_streak >= clear_frames_need
            faded = unobs & (clear_streak <= 0)
            d_now = np.where(
                instant_block,
                _D_BLOCK,
                np.where(
                    observed,
                    np.where(ok_mask, _D_OK, _D_BLOCK),
                    np.where(faded, _D_NONE, driveable_prev),
                ),
            )
            driveable_prev = d_now
            driveable_rows = []
            for ix in range(nx):
                dr: list[bool | None] = []
                for iy in range(ny):
                    dv = int(d_now[ix, iy])
                    if dv == int(_D_NONE):
                        dr.append(None)
                    elif dv == int(_D_BLOCK):
                        dr.append(False)
                    else:
                        dr.append(True)
                driveable_rows.append(dr)

        rows: list[list[float | None]] = []
        for ix in range(nx):
            row: list[float | None] = []
            for iy in range(ny):
                val = float(grid[ix, iy])
                row.append(None if math.isnan(val) else round(val, 4))
            rows.append(row)

        src: dict[str, Any] = {}
        if lidar_ts is not None:
            src["lidar_ts"] = lidar_ts
        if depth_ts is not None:
            src["depth_ts"] = depth_ts

        anchor_pose: dict[str, Any] | None = None
        if omsg is not None:
            try:
                anchor_pose = {
                    "odom_ts": float(omsg["ts"]),
                    "x": float(omsg["x"]),
                    "y": float(omsg["y"]),
                    "theta": float(omsg["theta"]),
                    "source": str(omsg.get("source", "commanded_vel_playback")),
                }
            except (KeyError, TypeError, ValueError):
                anchor_pose = None

        zenoh_helpers.publish_json(
            session,
            "body/map/local_2p5d",
            schemas.local_map_2p5d(
                ts=time.time(),
                resolution_m=res,
                origin_x_m=origin_x,
                origin_y_m=origin_y,
                nx=nx,
                ny=ny,
                max_height_m=rows,
                sources=src or None,
                driveable=driveable_rows,
                driveable_clearance_height_m=clearance_m if driveable_on else None,
                anchor_pose=anchor_pose,
            ),
        )

        next_tick += period
        sleep_for = next_tick - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_tick = time.monotonic()

    session.close()
    sys.exit(0)


if __name__ == "__main__":
    main()
