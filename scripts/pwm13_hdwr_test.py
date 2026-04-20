#!/usr/bin/env python3
"""
Pi 5 RP1 hardware PWM test on BCM GPIO13 (physical pin 33).

Uses:
  - pinctrl to switch GPIO13 to PWM0_CHAN1 alt function
  - Linux sysfs PWM interface on the RP1 controller

Expected mapping on Pi 5:
  GPIO13 -> PWM0_CHAN1
  RP1 PWM controller -> usually /sys/class/pwm/pwmchip2
  channel -> 1

Run with sudo if needed.

Examples:
  python3 scripts/pwm_gpio13_hw.py
  python3 scripts/pwm_gpio13_hw.py --freq 1000 --duty 50
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time


def write_text(path: pathlib.Path, value: str) -> None:
    path.write_text(value)


def read_text(path: pathlib.Path) -> str:
    return path.read_text().strip()


def find_pwmchip() -> pathlib.Path:
    pwm_root = pathlib.Path("/sys/class/pwm")
    chips = sorted(pwm_root.glob("pwmchip*"))
    if not chips:
        raise RuntimeError("No pwmchip devices found under /sys/class/pwm")

    # Prefer the RP1 controller on Pi 5, which is usually pwmchip2.
    preferred = pwm_root / "pwmchip2"
    if preferred.exists():
        return preferred

    # Fallback: look for a controller advertising >= 4 channels.
    for chip in chips:
        npwm_file = chip / "npwm"
        try:
            if int(read_text(npwm_file)) >= 4:
                return chip
        except Exception:
            pass

    raise RuntimeError(
        f"Could not identify RP1 PWM controller. Found: {[c.name for c in chips]}"
    )


def set_pinmux_gpio13_to_pwm() -> None:
    # GPIO13 -> PWM0_CHAN1 on Pi 5 RP1. "a0" selects ALT0 here.
    subprocess.run(
        ["pinctrl", "set", "13", "a0", "pn"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def export_channel(chip: pathlib.Path, channel: int) -> pathlib.Path:
    pwm_dir = chip / f"pwm{channel}"
    if not pwm_dir.exists():
        try:
            write_text(chip / "export", f"{channel}\n")
        except OSError:
            # It may already be exported by another process.
            pass

        # Give udev/sysfs a moment.
        for _ in range(50):
            if pwm_dir.exists():
                break
            time.sleep(0.02)

    if not pwm_dir.exists():
        raise RuntimeError(f"Failed to export channel {channel} on {chip}")
    return pwm_dir


def disable_if_enabled(pwm_dir: pathlib.Path) -> None:
    enable = pwm_dir / "enable"
    try:
        if read_text(enable) != "0":
            write_text(enable, "0\n")
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", type=float, default=1000.0, help="frequency in Hz")
    ap.add_argument("--duty", type=float, default=50.0, help="duty cycle percent")
    ap.add_argument("--channel", type=int, default=1, help="PWM channel (GPIO13 -> 1)")
    args = ap.parse_args()

    if args.freq <= 0:
        print("freq must be > 0", file=sys.stderr)
        return 1
    if not (0.0 <= args.duty <= 100.0):
        print("duty must be between 0 and 100", file=sys.stderr)
        return 1

    period_ns = int(round(1_000_000_000 / args.freq))
    duty_ns = int(round(period_ns * (args.duty / 100.0)))

    try:
        chip = find_pwmchip()
        set_pinmux_gpio13_to_pwm()
        pwm_dir = export_channel(chip, args.channel)

        # Sysfs PWM usually requires disable before changing period/duty.
        disable_if_enabled(pwm_dir)
        write_text(pwm_dir / "period", f"{period_ns}\n")
        write_text(pwm_dir / "duty_cycle", f"{duty_ns}\n")
        write_text(pwm_dir / "enable", "1\n")

        print(
            f"Hardware PWM enabled on GPIO13 via {chip.name}/pwm{args.channel}: "
            f"{args.freq:g} Hz, {args.duty:g}% duty"
        )
        print("Ctrl+C to stop.")

        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
        finally:
            try:
                write_text(pwm_dir / "enable", "0\n")
            except Exception:
                pass

        return 0

    except subprocess.CalledProcessError:
        print("Failed to run pinctrl. Is raspi-utils installed, and are permissions sufficient?", file=sys.stderr)
        return 1
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

