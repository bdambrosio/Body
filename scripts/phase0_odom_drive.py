#!/usr/bin/env python3
"""Phase 0 / Experiments B & C: analyze odom vs IMU during a recorded drive.

Reads a JSONL from `record_body_topics.py` that contains body/odom and
body/imu samples spanning a controlled motion: either a straight-line
drive of known distance (Experiment B) or an in-place rotation of known
angle (Experiment C). Reports:

  Translation (Experiment B):
    * encoder-reported distance  vs  --measured-distance-m (tape)
    * fractional translation error |reported - measured| / measured
    * heading drift during the drive: (encoder_θ_end - encoder_θ_start)
      vs (imu_yaw_end - imu_yaw_start) — IMU is treated as ground truth
      for orientation since BNO085 is far more accurate than encoder
      differential over short windows.

  Rotation (Experiment C):
    * encoder-reported total rotation vs IMU-reported total rotation
    * fractional rotation error |encoder - imu| / imu
    * derived α_4 (rotation noise per radian) and a per-tick rotation
      noise σ

Usage:
    # Translation experiment, e.g. drove a tape-measured 3.0 m forward
    PYTHONPATH=. python3 scripts/phase0_odom_drive.py PATH.jsonl \\
        --mode translation --measured-distance-m 3.0

    # Rotation experiment, e.g. rotated in place ~360°. The measured
    # angle comes from IMU integration here (because IMU is the ground
    # truth) — we don't need a hand measurement.
    PYTHONPATH=. python3 scripts/phase0_odom_drive.py PATH.jsonl \\
        --mode rotation

Outputs the noise-model coefficients the particle filter will consume:
  α_1 (translation σ per meter)
  α_4 (rotation σ per radian)
α_2 (translation σ from rotation) and α_3 (rotation σ from translation)
are cross-terms; left at 0 here, can be measured later with combined
drives if they turn out to matter.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

# Make `from desktop....` resolve when run as `python3 scripts/...`
# from the repo root, without requiring PYTHONPATH=. on the command
# line. Same pattern desktop/nav/__main__.py uses.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from desktop.nav.slam.types import ImuReading, quaternion_to_yaw


def _unwrap_to(prev: float, raw: float) -> float:
    d = raw - prev
    while d > math.pi:
        raw -= 2.0 * math.pi
        d = raw - prev
    while d < -math.pi:
        raw += 2.0 * math.pi
        d = raw - prev
    return raw


def _load_odom_imu(
    jsonl: Path, start_ts: Optional[float], end_ts: Optional[float],
) -> tuple[list[tuple[float, float, float, float]], list[tuple[float, float]]]:
    """Return (odom_samples, imu_samples).
    odom_samples: (sensor_ts, x, y, theta) in odom frame.
    imu_samples : (sensor_ts, yaw_unwrapped).
    """
    odom: list[tuple[float, float, float, float]] = []
    imu_raw: list[tuple[float, tuple[float, float, float, float]]] = []
    with jsonl.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            recv_ts = float(rec.get("recv_ts", 0.0))
            if start_ts is not None and recv_ts < start_ts:
                continue
            if end_ts is not None and recv_ts > end_ts:
                continue
            topic = rec.get("topic")
            payload = rec.get("payload") or {}
            if topic == "body/odom":
                sensor_ts = float(payload.get("ts") or recv_ts)
                try:
                    x = float(payload.get("x", 0.0))
                    y = float(payload.get("y", 0.0))
                    th = float(payload.get("theta", 0.0))
                except (TypeError, ValueError):
                    continue
                odom.append((sensor_ts, x, y, th))
            elif topic == "body/imu":
                reading = ImuReading.from_payload(payload)
                if reading is not None and reading.quat_wxyz is not None:
                    imu_raw.append((reading.ts, reading.quat_wxyz))

    odom.sort(key=lambda r: r[0])

    imu_raw.sort(key=lambda r: r[0])
    imu_samples: list[tuple[float, float]] = []
    prev_yaw: Optional[float] = None
    for ts, q in imu_raw:
        y = quaternion_to_yaw(q)
        y = y if prev_yaw is None else _unwrap_to(prev_yaw, y)
        prev_yaw = y
        imu_samples.append((ts, y))
    return odom, imu_samples


def _imu_at(imu_samples: list[tuple[float, float]], ts: float) -> Optional[float]:
    """Linear-interpolate IMU yaw at ts. Within the buffer: interpolate.
    Within one inter-sample period of either end: snap to that endpoint
    (same grace window pattern as ImuYawTracker.yaw_at — odom and IMU
    are published independently and the boundary samples will routinely
    miss each other by a few ms). Outside the grace: None.
    """
    if not imu_samples:
        return None
    n = len(imu_samples)
    first_ts = imu_samples[0][0]
    last_ts = imu_samples[-1][0]
    grace = (last_ts - first_ts) / max(1, n - 1) if n >= 2 else 0.05
    if ts < first_ts - grace or ts > last_ts + grace:
        return None
    if ts <= first_ts:
        return imu_samples[0][1]
    if ts >= last_ts:
        return imu_samples[-1][1]
    # Binary search for bracketing pair.
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if imu_samples[mid][0] <= ts:
            lo = mid
        else:
            hi = mid
    t0, y0 = imu_samples[lo]
    t1, y1 = imu_samples[hi]
    if t1 == t0:
        return y1
    alpha = (ts - t0) / (t1 - t0)
    return y0 + alpha * (y1 - y0)


def analyze_translation(
    odom: list[tuple[float, float, float, float]],
    imu_samples: list[tuple[float, float]],
    measured_m: float,
) -> None:
    if len(odom) < 10:
        print("not enough odom samples", file=sys.stderr); return
    x0, y0, th0 = odom[0][1], odom[0][2], odom[0][3]
    x1, y1, th1 = odom[-1][1], odom[-1][2], odom[-1][3]
    encoder_dist = math.hypot(x1 - x0, y1 - y0)
    encoder_dtheta = th1 - th0
    encoder_dtheta = ((encoder_dtheta + math.pi) % (2.0 * math.pi)) - math.pi

    imu_y0 = _imu_at(imu_samples, odom[0][0])
    imu_y1 = _imu_at(imu_samples, odom[-1][0])
    if imu_y0 is None or imu_y1 is None:
        print("IMU samples don't span the odom window; cannot compute heading drift",
              file=sys.stderr)
        imu_dtheta = float("nan")
    else:
        imu_dtheta = imu_y1 - imu_y0  # already unwrapped

    err_abs = abs(encoder_dist - measured_m)
    err_frac = err_abs / measured_m if measured_m > 0 else 0.0

    # σ_translation per √meter: assume independent error per unit, so σ
    # accumulates as σ_total = α_1 · D. From one drive we can only point-
    # estimate α_1 = err_abs / D. Need multiple drives for a real fit;
    # this is a starting value.
    alpha_1_point_est = err_abs / measured_m if measured_m > 0 else 0.0

    print("# Translation drift")
    print(f"odom samples       : {len(odom)}")
    print(f"encoder dist (m)   : {encoder_dist:.4f}")
    print(f"measured dist (m)  : {measured_m:.4f}")
    print(f"abs error (m)      : {err_abs:.4f}")
    print(f"fractional error   : {err_frac*100:.2f} %")
    print(f"encoder Δθ         : {math.degrees(encoder_dtheta):+.2f} deg over drive")
    print(f"IMU      Δθ        : {math.degrees(imu_dtheta):+.2f} deg over drive"
          if not math.isnan(imu_dtheta) else "IMU Δθ: n/a")
    print()
    print("# Particle-filter consumption (translation noise model)")
    print(f"# α_1 point estimate (σ_trans / meter): {alpha_1_point_est:.4f}")
    print("# Run ≥3 drives at different distances + speeds to fit α_1 properly.")


def analyze_rotation(
    odom: list[tuple[float, float, float, float]],
    imu_samples: list[tuple[float, float]],
) -> None:
    if len(odom) < 10 or len(imu_samples) < 10:
        print("not enough samples", file=sys.stderr); return
    # Total unwrapped rotation per source.
    odom_ths = []
    prev = None
    for _, _, _, th in odom:
        th_u = th if prev is None else _unwrap_to(prev, th)
        prev = th_u
        odom_ths.append(th_u)
    encoder_total = odom_ths[-1] - odom_ths[0]

    imu_y0 = _imu_at(imu_samples, odom[0][0])
    imu_y1 = _imu_at(imu_samples, odom[-1][0])
    if imu_y0 is None or imu_y1 is None:
        print("IMU samples don't span odom window", file=sys.stderr); return
    imu_total = imu_y1 - imu_y0

    err_abs = encoder_total - imu_total
    err_frac = err_abs / imu_total if abs(imu_total) > 1e-6 else 0.0
    alpha_4_point_est = abs(err_abs) / abs(imu_total) if abs(imu_total) > 1e-6 else 0.0

    print("# Rotation drift")
    print(f"odom samples       : {len(odom)}")
    print(f"encoder total Δθ   : {math.degrees(encoder_total):+.2f} deg")
    print(f"IMU     total Δθ   : {math.degrees(imu_total):+.2f} deg "
          "(ground truth for short windows)")
    print(f"encoder - IMU      : {math.degrees(err_abs):+.2f} deg")
    print(f"fractional error   : {err_frac*100:+.2f} %")
    print()
    print("# Particle-filter consumption (rotation noise model)")
    print(f"# α_4 point estimate (σ_rot / radian): {alpha_4_point_est:.4f}")
    print("# Run ≥3 rotations of varying magnitudes/rates to fit α_4 properly.")


def analyze_combined(
    odom: list[tuple[float, float, float, float]],
    imu_samples: list[tuple[float, float]],
    measured_endpoint_xy_m: Optional[tuple[float, float]],
    measured_endpoint_theta_rad: Optional[float],
) -> None:
    """Combined translation + rotation calibration → point estimate
    of α_2 (translation σ from rotation).

    The bot drives a path that interleaves translation and rotation
    (figure-8, square loop, S-curve…) and the operator records the
    measured *true* final pose relative to the seed. We compare:

      encoder_end:   raw odom integration to end-of-drive.
      truth_end:     operator-measured (tape + protractor, or
                     return-to-known-mark).

    Decompose end-position error in the bot's *initial* frame:
        Δ = encoder_end − truth_end
    The xy magnitude is the total translation drift. From a single
    drive we can fit a point estimate of α_2 by assuming:
        σ_trans² ≈ α_1² · S²  +  α_2² · |Θ|²
    where S is total arc length and Θ is total |Δθ| traversed.
    α_1 from Phase 0 is fixed (=0.04). Solve for α_2:
        α_2² = (Δxy² / N) − α_1² · S² / N      (single sample → N=1)
        α_2  = sqrt(max(Δxy² − α_1² · S², 0)) / |Θ|

    Multiple drives at varying rotation amounts give the standard fit.
    With one drive we get a point estimate plus a sanity check.
    """
    if len(odom) < 10:
        print("not enough odom samples", file=sys.stderr); return
    x0, y0, th0 = odom[0][1], odom[0][2], odom[0][3]
    xN, yN, thN = odom[-1][1], odom[-1][2], odom[-1][3]

    # Arc length S = Σ|chord|; total |rotation| = Σ|Δθ|.
    arc_length_m = 0.0
    abs_rotation_rad = 0.0
    for prev, curr in zip(odom[:-1], odom[1:]):
        dx = curr[1] - prev[1]
        dy = curr[2] - prev[2]
        dth = ((curr[3] - prev[3] + math.pi) % (2 * math.pi)) - math.pi
        arc_length_m += math.hypot(dx, dy)
        abs_rotation_rad += abs(dth)

    enc_dx = xN - x0
    enc_dy = yN - y0
    enc_dth = ((thN - th0 + math.pi) % (2 * math.pi)) - math.pi

    print(f"# Combined-motion analysis")
    print(f"# Drive: arc_length = {arc_length_m:.2f} m  "
          f"|rotation| = {math.degrees(abs_rotation_rad):.1f}°  "
          f"net_dθ = {math.degrees(enc_dth):+.1f}°")
    print(f"# Encoder Δ from seed (in seed-frame): "
          f"({enc_dx:+.3f}, {enc_dy:+.3f}) m, "
          f"net_θ = {math.degrees(enc_dth):+.1f}°")

    if measured_endpoint_xy_m is None:
        print("# Pass --measured-endpoint-xy mx,my to fit α_2.")
        print(f"# If the drive started and ended at the same physical mark, "
              f"--measured-endpoint-xy 0,0 works.")
        return

    truth_dx, truth_dy = measured_endpoint_xy_m
    # Rotate truth Δ from seed-physical frame into the seed *odom-frame*
    # if heading at seed wasn't aligned. We assume seed odom frame
    # aligns with the seed bot orientation, which it should after a
    # rebind. If not, pass --seed-heading-offset.
    err_dx = enc_dx - truth_dx
    err_dy = enc_dy - truth_dy
    err_xy = math.hypot(err_dx, err_dy)

    alpha_1 = 0.04  # locked from Phase 0
    expected_alpha1_drift = alpha_1 * arc_length_m
    print(f"# Expected drift from α_1 alone (Phase 0): "
          f"{expected_alpha1_drift*100:.1f} cm  "
          f"(α_1={alpha_1} × S={arc_length_m:.2f} m)")
    print(f"# Observed end-position error: "
          f"|err| = {err_xy*100:.1f} cm  "
          f"(dx={err_dx*100:+.1f}, dy={err_dy*100:+.1f})")

    # Residual after subtracting α_1 contribution (in quadrature).
    residual2 = err_xy**2 - expected_alpha1_drift**2
    if residual2 <= 0 or abs_rotation_rad < 1e-3:
        print(f"# α_2 cannot be fit from this drive — "
              f"observed drift ({err_xy*100:.1f} cm) does not exceed "
              f"α_1-only prediction ({expected_alpha1_drift*100:.1f} cm).")
        print(f"# Either α_2 ≈ 0, or this drive's rotation was too "
              f"small to detect the cross-term.")
        return
    alpha_2_point = math.sqrt(residual2) / abs_rotation_rad
    print(f"# α_2 point estimate: {alpha_2_point:.4f} m/rad")
    print(f"#   ({alpha_2_point*100:.1f} cm of σ per radian of rotation)")
    print(f"# Sanity: total |rotation| was "
          f"{math.degrees(abs_rotation_rad):.1f}° → α_2 contribution "
          f"= α_2·|Θ| = {alpha_2_point*abs_rotation_rad*100:.1f} cm")
    print(f"# Run ≥3 drives at varying rotation amounts to fit properly.")

    if measured_endpoint_theta_rad is not None:
        truth_dth = ((measured_endpoint_theta_rad + math.pi) % (2 * math.pi)) - math.pi
        th_err = abs((enc_dth - truth_dth + math.pi) % (2 * math.pi) - math.pi)
        print(f"# Bonus: encoder heading error = "
              f"{math.degrees(th_err):.2f}° vs measured.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("jsonl", type=Path)
    p.add_argument(
        "--mode", choices=("translation", "rotation", "combined"), required=True,
    )
    p.add_argument("--measured-distance-m", type=float, default=None,
                   help="Required for --mode translation. Tape-measured distance.")
    p.add_argument(
        "--measured-endpoint-xy", type=str, default=None,
        help="--mode combined: 'mx,my' (m) — true end-position relative "
             "to seed in the seed bot frame. e.g. '0,0' if you returned "
             "to the same physical mark, or '1.5,-0.3' if you measured "
             "the end position relative to start.",
    )
    p.add_argument(
        "--measured-endpoint-theta-deg", type=float, default=None,
        help="--mode combined: optional true end-heading relative to "
             "seed (degrees). Lets the script also report encoder yaw "
             "error as a sanity check.",
    )
    p.add_argument("--start-ts", type=float, default=None)
    p.add_argument("--end-ts", type=float, default=None)
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    if args.mode == "translation" and args.measured_distance_m is None:
        p.error("--measured-distance-m required for --mode translation")

    odom, imu_samples = _load_odom_imu(args.jsonl, args.start_ts, args.end_ts)
    if not odom:
        print(f"No body/odom in {args.jsonl}", file=sys.stderr)
        return 1
    if not imu_samples:
        print(f"No body/imu in {args.jsonl}", file=sys.stderr)
        return 1

    print(f"# {args.jsonl}")
    if args.mode == "translation":
        analyze_translation(odom, imu_samples, args.measured_distance_m)
    elif args.mode == "rotation":
        analyze_rotation(odom, imu_samples)
    else:
        endpoint_xy = None
        if args.measured_endpoint_xy is not None:
            parts = args.measured_endpoint_xy.split(",")
            if len(parts) != 2:
                p.error("--measured-endpoint-xy must be 'mx,my'")
            endpoint_xy = (float(parts[0]), float(parts[1]))
        endpoint_th = None
        if args.measured_endpoint_theta_deg is not None:
            endpoint_th = math.radians(args.measured_endpoint_theta_deg)
        analyze_combined(odom, imu_samples, endpoint_xy, endpoint_th)
    return 0


if __name__ == "__main__":
    sys.exit(main())
