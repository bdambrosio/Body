#!/usr/bin/env python3
"""Phase 0 / Experiment A: analyze IMU yaw drift on a stationary log.

Reads a JSONL recording from `record_body_topics.py` containing body/imu
samples captured while the robot was motionless. Reports:

  * Linear drift rate (deg/min and rad/s) — fits a straight line to
    yaw vs time, slope is the drift rate.
  * Per-sample noise σ (deg) — std of residuals from the linear fit.
  * Per-second noise σ (deg) — random-walk equivalent, useful for the
    particle filter's IMU observation Σ.
  * Sample rate (Hz) — sanity check.

The particle filter uses these as:
  * IMU yaw observation Σ at time t: σ²(t) ≈ σ²_sample + (drift_rate · t)²

Usage:
    PYTHONPATH=. python3 scripts/phase0_imu_stationary.py PATH.jsonl \\
        [--start-ts T] [--end-ts T] [--plot]

The optional [--start-ts / --end-ts] trim the log to the truly-stationary
window; useful if the recording captured a bit of setup motion on either
end.

Outputs a small summary on stdout and (with --plot) saves a PNG of yaw
vs time next to the input file.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional


def _quat_to_yaw(w: float, x: float, y: float, z: float) -> float:
    """Yaw in radians from a wxyz quaternion. Mirrors the desktop helper."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _unwrap_to(prev: float, raw: float) -> float:
    d = raw - prev
    while d > math.pi:
        raw -= 2.0 * math.pi
        d = raw - prev
    while d < -math.pi:
        raw += 2.0 * math.pi
        d = raw - prev
    return raw


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("jsonl", type=Path, help="recording from record_body_topics.py")
    p.add_argument("--start-ts", type=float, default=None,
                   help="Skip samples with recv_ts < this (wall-clock seconds).")
    p.add_argument("--end-ts", type=float, default=None,
                   help="Skip samples with recv_ts > this.")
    p.add_argument("--plot", action="store_true",
                   help="Save yaw-vs-time PNG next to the input.")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    # (sensor_ts, yaw_rad_unwrapped)
    samples: list[tuple[float, float]] = []
    prev_yaw: Optional[float] = None
    skipped_no_quat = 0
    skipped_no_ts = 0

    with args.jsonl.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("topic") != "body/imu":
                continue
            recv_ts = float(rec.get("recv_ts", 0.0))
            if args.start_ts is not None and recv_ts < args.start_ts:
                continue
            if args.end_ts is not None and recv_ts > args.end_ts:
                continue
            payload = rec.get("payload") or {}
            quat = payload.get("quat_wxyz") or payload.get("quaternion_wxyz")
            if not (isinstance(quat, list) and len(quat) == 4):
                skipped_no_quat += 1
                continue
            sensor_ts = float(payload.get("ts") or recv_ts)
            if sensor_ts <= 0:
                skipped_no_ts += 1
                continue
            yaw_raw = _quat_to_yaw(*[float(v) for v in quat])
            if prev_yaw is None:
                yaw = yaw_raw
            else:
                yaw = _unwrap_to(prev_yaw, yaw_raw)
            prev_yaw = yaw
            samples.append((sensor_ts, yaw))

    if len(samples) < 50:
        print(
            f"Only {len(samples)} usable body/imu samples found. "
            f"Need more for reliable statistics. "
            f"(skipped: no_quat={skipped_no_quat}, no_ts={skipped_no_ts})",
            file=sys.stderr,
        )
        return 1

    samples.sort(key=lambda r: r[0])
    t0 = samples[0][0]
    t_rel = [s[0] - t0 for s in samples]
    yaw = [s[1] for s in samples]
    n = len(samples)
    duration_s = t_rel[-1]
    rate_hz = n / duration_s if duration_s > 0 else 0.0

    # Linear fit yaw = a + b·t. b is drift rate in rad/s.
    mean_t = sum(t_rel) / n
    mean_y = sum(yaw) / n
    num = sum((t_rel[i] - mean_t) * (yaw[i] - mean_y) for i in range(n))
    den = sum((t_rel[i] - mean_t) ** 2 for i in range(n))
    drift_rad_per_s = num / den if den > 0 else 0.0
    intercept = mean_y - drift_rad_per_s * mean_t

    residuals = [yaw[i] - (intercept + drift_rad_per_s * t_rel[i]) for i in range(n)]
    var_sample = sum(r * r for r in residuals) / max(1, n - 2)
    sigma_sample_rad = math.sqrt(var_sample)
    sigma_per_s_rad = sigma_sample_rad * math.sqrt(rate_hz) if rate_hz > 0 else 0.0

    # Total drift over the window for sanity.
    total_drift_rad = drift_rad_per_s * duration_s

    print(f"# IMU stationary drift — {args.jsonl}")
    print(f"samples          : {n}")
    print(f"duration         : {duration_s:.2f} s")
    print(f"sample rate      : {rate_hz:.1f} Hz")
    print(f"drift rate       : {math.degrees(drift_rad_per_s):+.4f} deg/s  "
          f"({math.degrees(drift_rad_per_s)*60:+.3f} deg/min,  "
          f"{drift_rad_per_s:+.6e} rad/s)")
    print(f"total drift      : {math.degrees(total_drift_rad):+.3f} deg over window")
    print(f"per-sample σ     : {math.degrees(sigma_sample_rad):.4f} deg  "
          f"({sigma_sample_rad:.6e} rad)")
    print(f"per-second σ     : {math.degrees(sigma_per_s_rad):.4f} deg/√s "
          f"({sigma_per_s_rad:.6e} rad/√s)  (random-walk equivalent)")
    print()
    print("# Particle-filter consumption")
    print(f"# σ²_IMU(t) ≈ σ²_sample + (drift_rate · t)²")
    print(f"# σ_sample = {sigma_sample_rad:.6e} rad ({math.degrees(sigma_sample_rad):.4f} deg)")
    print(f"# drift_rate = {drift_rad_per_s:+.6e} rad/s ({math.degrees(drift_rad_per_s)*60:+.3f} deg/min)")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available; skipping --plot", file=sys.stderr)
            return 0
        out_png = args.jsonl.with_suffix(".imu_drift.png")
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(t_rel, [math.degrees(y) for y in yaw], lw=0.6)
        axes[0].plot(
            [t_rel[0], t_rel[-1]],
            [math.degrees(intercept), math.degrees(intercept + drift_rad_per_s * t_rel[-1])],
            "r--", lw=1.0, label="linear fit",
        )
        axes[0].set_ylabel("yaw (deg)")
        axes[0].set_title(f"IMU yaw drift — {args.jsonl.name}")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[1].plot(t_rel, [math.degrees(r) for r in residuals], lw=0.5)
        axes[1].set_ylabel("residual (deg)")
        axes[1].set_xlabel("t (s)")
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_png, dpi=120)
        print(f"plot → {out_png}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
