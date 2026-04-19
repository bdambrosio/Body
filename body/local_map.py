"""Fuse lidar + OAK-D depth into an egocentric 2.5D max-height grid (``body/map/local_2p5d``).

Ground plane z = 0 in body frame (+x forward, +y left, +z up). Each cell stores the maximum
sampled height above ground from the latest lidar slice and depth frustum. See docs/local_map_spec.md.
"""

from __future__ import annotations

import base64
import math
import signal
import sys
import threading
import time
from typing import Any

import numpy as np

from body.lib import schemas, zenoh_helpers


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


def _median_filter_depth_mm(arr: np.ndarray, kernel: int) -> np.ndarray:
    """Reduce stereo speckle before unprojection: per-pixel median over valid neighbors only.

    Pixels with value ``0`` are treated as invalid (DepthAI convention) and omitted from the
    median; if no valid samples exist in the window, output is ``0``. Kernel must be odd and >= 3.
    """
    if kernel <= 1 or kernel % 2 == 0:
        return arr
    half = kernel // 2
    h, w = arr.shape
    out = np.zeros_like(arr)
    for v in range(h):
        for u in range(w):
            patch = arr[max(0, v - half) : min(h, v + half + 1), max(0, u - half) : min(w, u + half + 1)]
            valid = patch[patch > 0]
            if valid.size == 0:
                continue
            out[v, u] = np.uint16(np.median(valid))
    return out


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

    lidar_z = float(lm.get("lidar_z_body_m", lidar_cfg.get("height_above_ground_m", 0.1778)))
    lidar_x = float(lm.get("lidar_x_body_m", 0.0))
    lidar_y = float(lm.get("lidar_y_body_m", 0.0))
    lidar_yaw = float(lm.get("lidar_yaw_rad", 0.0))

    depth_z = float(
        lm.get("depth_z_body_m", oakd_cfg.get("depth_camera_height_above_ground_m", 0.1016))
    )
    depth_x = float(lm.get("depth_x_body_m", 0.0))
    depth_y = float(lm.get("depth_y_body_m", 0.0))
    depth_yaw = float(lm.get("depth_yaw_rad", 0.0))
    depth_pitch = float(lm.get("depth_pitch_rad", 0.0))
    depth_roll = float(lm.get("depth_roll_rad", 0.0))
    hfov = math.radians(float(lm.get("depth_hfov_deg", 73.0)))
    vfov = math.radians(float(lm.get("depth_vfov_deg", 58.0)))
    depth_median_kernel = int(lm.get("depth_median_kernel", 3))

    # OpenCV cam: x right, y down, z forward  ->  body: x forward, y left, z up
    R_fix = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64)
    R_e = _R_body_from_cam_euler(depth_yaw, depth_pitch, depth_roll)
    R_bc = R_e @ R_fix
    t_bc = np.array([depth_x, depth_y, depth_z], dtype=np.float64)

    origin_x = -xb
    origin_y = -yr
    nx = max(1, int(math.ceil((xf + xb) / res)))
    ny = max(1, int(math.ceil((yl + yr) / res)))

    lock = threading.Lock()
    last_lidar: dict[str, Any] | None = None
    last_depth: dict[str, Any] | None = None

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

    zenoh_helpers.declare_subscriber_json(session, "body/lidar/scan", on_lidar)
    zenoh_helpers.declare_subscriber_json(session, "body/oakd/depth", on_depth)

    def handle_sigterm(_sig: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    next_tick = time.monotonic()
    while not stop.is_set():
        grid = np.full((nx, ny), np.nan, dtype=np.float64)
        lidar_ts: float | None = None
        depth_ts: float | None = None

        with lock:
            lmsg = last_lidar
            dmsg = last_depth

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
                        old = grid[ix, iy]
                        grid[ix, iy] = pz if math.isnan(old) else max(old, pz)

        if dmsg is not None:
            depth_ts = float(dmsg.get("ts", 0.0))
            dec = _decode_depth_mm(dmsg)
            if dec is not None:
                arr, w, h = dec
                if depth_median_kernel > 1:
                    arr = _median_filter_depth_mm(arr, depth_median_kernel)
                fx = (w - 1) / (2.0 * math.tan(hfov / 2.0)) if w > 1 else 1.0
                fy = (h - 1) / (2.0 * math.tan(vfov / 2.0)) if h > 1 else 1.0
                cx = (w - 1) * 0.5
                cy = (h - 1) * 0.5
                for v in range(h):
                    for u in range(w):
                        zmm = int(arr[v, u])
                        if zmm <= 0:
                            continue
                        Z = zmm / 1000.0
                        x_c = (u - cx) * Z / fx
                        y_c = (v - cy) * Z / fy
                        p_c = np.array([x_c, y_c, Z], dtype=np.float64)
                        p_b = R_bc @ p_c + t_bc
                        px, py, pz = float(p_b[0]), float(p_b[1]), float(p_b[2])
                        if pz <= ground_z:
                            continue
                        ix = int(math.floor((px - origin_x) / res))
                        iy = int(math.floor((py - origin_y) / res))
                        if 0 <= ix < nx and 0 <= iy < ny:
                            old = grid[ix, iy]
                            grid[ix, iy] = pz if math.isnan(old) else max(old, pz)

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
