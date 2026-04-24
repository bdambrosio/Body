#!/usr/bin/env python3
"""Standalone BNO085 diagnostic (no Zenoh, no motor_controller).

Pulses the RST line via lgpio, opens i2c via Blinka, enables the core
SH-2 reports (accel + gyro + rotation vector), and prints one sample
per second with quaternion + derived yaw + accuracy estimate.

Use to verify:
  - Module is wired correctly (3.3V / GND / SDA / SCL / INT / RST).
  - i2c address is 0x4B (default, ADDR open) or 0x4A (ADDR shorted).
  - Rotation Vector fusion converges under the current chassis layout.

  sudo .venv/bin/python3 scripts/bno085_probe.py
  sudo .venv/bin/python3 scripts/bno085_probe.py --addr 0x4A
  sudo .venv/bin/python3 scripts/bno085_probe.py --mode game_rotation_vector
  sudo .venv/bin/python3 scripts/bno085_probe.py --no-reset

See docs/imu_driver_spec.md §4 for pin allocation, §7 for mag-interference
test methodology (run this script stationary then under commanded motion
and compare accuracy_rad).

Ctrl+C to quit.
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time


def _pulse_reset(gpio: int, chip_idx: int) -> None:
    """Drive RST low ~20 ms, release, wait ~200 ms for SH-2 to boot."""
    try:
        import lgpio
    except ModuleNotFoundError:
        print(
            "lgpio not found — skipping RST pulse. Install python3-lgpio "
            "(sudo apt install python3-lgpio) or rerun with --no-reset.",
            file=sys.stderr,
        )
        return

    h = lgpio.gpiochip_open(chip_idx)
    try:
        ret = lgpio.gpio_claim_output(h, gpio, 1)
        if ret < 0:
            print(f"gpio_claim_output({gpio}) failed: {ret}", file=sys.stderr)
            return
        lgpio.gpio_write(h, gpio, 0)
        time.sleep(0.02)
        lgpio.gpio_write(h, gpio, 1)
        time.sleep(0.20)
    finally:
        try:
            lgpio.gpio_free(h, gpio)
        except Exception:
            pass
        lgpio.gpiochip_close(h)


def _quat_to_yaw_rad(i: float, j: float, k: float, real: float) -> float:
    """Yaw around +z (right-hand, CCW positive) from a unit quaternion.

    BNO085 Adafruit lib returns quaternion as (i, j, k, real) == (x, y, z, w).
    Matches desktop/nav/slam/types.py::quaternion_to_yaw convention.
    """
    siny = 2.0 * (real * k + i * j)
    cosy = 1.0 - 2.0 * (j * j + k * k)
    return math.atan2(siny, cosy)


def main() -> None:
    p = argparse.ArgumentParser(description="BNO085 i2c probe (Adafruit Blinka)")
    p.add_argument("--addr", type=lambda s: int(s, 0), default=0x4B,
                   help="i2c address (default 0x4B; 0x4A if ADDR jumper closed)")
    p.add_argument("--mode", choices=["rotation_vector", "game_rotation_vector"],
                   default="rotation_vector",
                   help="Fusion mode to enable (default rotation_vector)")
    p.add_argument("--rst-gpio", type=int, default=25,
                   help="BCM GPIO driving RST (default 25); ignored with --no-reset")
    p.add_argument("--chip", type=int, default=0,
                   help="gpiochip index for RST pulse (default 0)")
    p.add_argument("--no-reset", action="store_true",
                   help="Skip the RST pulse (use if RST is not wired)")
    p.add_argument("--period-s", type=float, default=1.0,
                   help="Print interval in seconds (default 1.0)")
    args = p.parse_args()

    if not args.no_reset:
        print(f"Pulsing RST on BCM {args.rst_gpio} …", flush=True)
        _pulse_reset(args.rst_gpio, args.chip)

    try:
        import board
        import busio
        from adafruit_bno08x.i2c import BNO08X_I2C
        from adafruit_bno08x import (
            BNO_REPORT_ACCELEROMETER,
            BNO_REPORT_GYROSCOPE,
            BNO_REPORT_ROTATION_VECTOR,
            BNO_REPORT_GAME_ROTATION_VECTOR,
        )
    except ModuleNotFoundError as e:
        print(
            f"Missing dependency: {e.name}. Install with:\n"
            "  pip install adafruit-circuitpython-bno08x adafruit-blinka",
            file=sys.stderr,
        )
        sys.exit(2)

    i2c = busio.I2C(board.SCL, board.SDA, frequency=400_000)
    bno = BNO08X_I2C(i2c, address=args.addr)

    quat_feature = (
        BNO_REPORT_ROTATION_VECTOR
        if args.mode == "rotation_vector"
        else BNO_REPORT_GAME_ROTATION_VECTOR
    )
    bno.enable_feature(BNO_REPORT_ACCELEROMETER)
    bno.enable_feature(BNO_REPORT_GYROSCOPE)
    bno.enable_feature(quat_feature)

    print(
        f"BNO085 ready at 0x{args.addr:02x}, mode={args.mode}. "
        f"Rotate the chassis by hand and watch yaw_deg. Ctrl+C to quit.",
        flush=True,
    )

    stop = False

    def handle_sig(_s, _f) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    t0 = time.monotonic()
    samples = 0
    while not stop:
        try:
            accel = bno.acceleration
            gyro = bno.gyro
            if args.mode == "rotation_vector":
                quat = bno.quaternion
            else:
                quat = bno.game_quaternion
        except Exception as e:
            print(f"read error: {type(e).__name__}: {e}", flush=True)
            time.sleep(0.2)
            continue

        samples += 1
        if quat is None or accel is None or gyro is None:
            print("  (waiting for first full report set…)", flush=True)
            time.sleep(0.2)
            continue

        i, j, k, real = quat
        yaw_rad = _quat_to_yaw_rad(i, j, k, real)
        ax, ay, az = accel
        gx, gy, gz = gyro

        elapsed = time.monotonic() - t0
        rate = samples / elapsed if elapsed > 0 else 0.0
        print(
            f"t={elapsed:6.1f}s rate={rate:5.1f}Hz | "
            f"yaw={math.degrees(yaw_rad):+7.2f}° "
            f"quat(ijk,r)=({i:+.3f},{j:+.3f},{k:+.3f},{real:+.3f}) | "
            f"accel=({ax:+5.2f},{ay:+5.2f},{az:+5.2f}) m/s² | "
            f"gyro=({gx:+5.2f},{gy:+5.2f},{gz:+5.2f}) rad/s",
            flush=True,
        )
        time.sleep(args.period_s)


if __name__ == "__main__":
    main()
