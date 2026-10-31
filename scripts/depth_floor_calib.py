#!/usr/bin/env python
"""Estimate the OAK-D depth camera's pitch / roll / height from the floor.

Point the robot at open floor (nothing within ~1.5 m ahead is best), then:

    .venv/bin/python scripts/depth_floor_calib.py [--seconds 4] [--router tcp/192.168.8.60:7447]

Subscribes to body/oakd/depth, converts each frame to body-frame points with
the CURRENT config (local_drive.depth_veto / local_map camera model), fits the
dominant plane (RANSAC) to points within --max-range, and reports how far the
floor is from z = 0: the residual pitch, roll and height error, plus the
config values that would zero them. A big residual means the mount moved or
the config never matched.

Prints only; edit config.json yourself (keys: local_map.depth_pitch_rad,
depth_roll_rad, oakd.depth_camera_height_above_ground_m; the veto and the
depth memory read the same values through drive_config.depth_veto_config).
"""
from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from dataclasses import replace

import numpy as np

from body.lib import drive_config, zenoh_helpers
from body.lib.depth_veto import decode_depth_mm, depth_points_body


def _fit_plane_ransac(p: np.ndarray, *, iters: int = 300, tol: float = 0.015,
                      rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Best plane z = a x + b y + c by inlier count; returns (coef, inlier mask)."""
    best_m = np.zeros(len(p), dtype=bool)
    best = None
    for _ in range(iters):
        idx = rng.choice(len(p), 3, replace=False)
        A = np.c_[p[idx, :2], np.ones(3)]
        try:
            coef = np.linalg.solve(A, p[idx, 2])
        except np.linalg.LinAlgError:
            continue
        res = np.abs(p[:, 0] * coef[0] + p[:, 1] * coef[1] + coef[2] - p[:, 2])
        m = res < tol
        if m.sum() > best_m.sum():
            best_m, best = m, coef
    if best is None or best_m.sum() < 50:
        raise SystemExit("no plane found — is the camera looking at floor?")
    A = np.c_[p[best_m, :2], np.ones(int(best_m.sum()))]
    coef = np.linalg.lstsq(A, p[best_m, 2], rcond=None)[0]
    return coef, best_m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--router", default="tcp/192.168.8.60:7447")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--max-range", type=float, default=1.5)
    ap.add_argument("--lateral", type=float, default=0.6)
    args = ap.parse_args()

    body_cfg = zenoh_helpers.load_body_config()
    vc = drive_config.depth_veto_config(body_cfg)
    full = replace(vc, roi_u0=0.0, roi_u1=1.0, roi_v0=0.0, roi_v1=1.0)
    print(f"config: pitch {vc.depth_pitch_rad:.3f} rad ({math.degrees(vc.depth_pitch_rad):.1f} deg) "
          f"roll {vc.depth_roll_rad:.3f} rad  height {vc.depth_z_body_m:.3f} m  "
          f"floor band {vc.floor_band_m:.3f} m")

    import zenoh
    zc = zenoh.Config()
    zc.insert_json5("mode", '"client"')
    zc.insert_json5("connect/endpoints", f'["{args.router}"]')
    session = zenoh.open(zc)
    lock = threading.Lock()
    frames: list = []

    def on_depth(_k, m):
        with lock:
            frames.append(m)

    zenoh_helpers.declare_subscriber_json(session, "body/oakd/depth", on_depth)
    time.sleep(args.seconds)
    with lock:
        fr = list(frames)
    session.close()
    if not fr:
        print("no depth frames received", file=sys.stderr)
        return 1

    pts = []
    for m in fr:
        d = decode_depth_mm(m)
        if d is None:
            continue
        dp = depth_points_body(d[0], m, full)
        if dp is None:
            continue
        x, y = dp.pts[:, 0], dp.pts[:, 1]
        sel = (x > 0.1) & (x < args.max_range) & (np.abs(y) < args.lateral)
        pts.append(dp.pts[sel])
    p = np.concatenate(pts)
    print(f"frames {len(fr)}  points used {len(p)}")

    coef, inl = _fit_plane_ransac(p, rng=np.random.default_rng(0))
    a, b, c = (float(v) for v in coef)
    res = p[inl, 0] * a + p[inl, 1] * b + c - p[inl, 2]
    pitch_err = math.atan(a)          # floor rising with range → camera pitched down LESS than config
    roll_err = math.atan(b)
    print(f"floor plane: z = {a:+.4f} x {b:+.4f} y {c:+.4f}   inliers {inl.sum()}/{len(p)} "
          f"rms {float(np.sqrt(np.mean(res**2)))*1000:.1f} mm")
    print(f"residual: pitch {math.degrees(pitch_err):+.1f} deg  roll {math.degrees(roll_err):+.1f} deg  "
          f"height {c*1000:+.0f} mm")
    print("suggested config:")
    print(f"  local_map.depth_pitch_rad: {vc.depth_pitch_rad - pitch_err:.3f}   (now {vc.depth_pitch_rad:.3f})")
    print(f"  local_map.depth_roll_rad:  {vc.depth_roll_rad - roll_err:.3f}   (now {vc.depth_roll_rad:.3f})")
    print(f"  oakd.depth_camera_height_above_ground_m: {vc.depth_z_body_m - c:.3f}   (now {vc.depth_z_body_m:.3f})")
    # What the current config does to the floor across the veto / memory range.
    print("floor height as the current config sees it (inlier median per range bin):")
    for lo in np.arange(0.1, args.max_range, 0.1):
        m = inl & (p[:, 0] >= lo) & (p[:, 0] < lo + 0.1)
        if m.sum() > 30:
            flag = "" if abs(np.median(p[m, 2])) <= vc.floor_band_m else "  <- outside floor band"
            print(f"  {lo:.1f}-{lo+0.1:.1f} m: {np.median(p[m, 2])*1000:+5.0f} mm (n={int(m.sum())}){flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
