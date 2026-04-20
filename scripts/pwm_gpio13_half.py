#!/usr/bin/env python3
"""Drive one BCM GPIO with lgpio software PWM (Pi 5 / RP1). Default: GPIO13 @ 1 kHz, 50% duty.

Use this to verify scope probes and wiring without Zenoh or motor_controller.

  python3 scripts/pwm_gpio13_half.py
  python3 scripts/pwm_gpio13_half.py --gpio 12 --duty 25

Requires: python3-lgpio (apt: sudo apt install python3-lgpio). User must be in group ``gpio``
or run with sufficient rights to open /dev/gpiochip*.

Press Ctrl+C to stop (PWM set to 0% before exit).
"""

from __future__ import annotations

import argparse
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description="lgpio PWM test on one BCM pin")
    parser.add_argument("--chip", type=int, default=0, help="gpiochip index (Pi 5 RP1 is usually 0)")
    parser.add_argument("--gpio", type=int, default=13, help="BCM GPIO number (Body right motor PWM default: 13)")
    parser.add_argument("--freq", type=int, default=1000, help="PWM frequency Hz (lgpio allows 0.1–10000)")
    parser.add_argument("--duty", type=float, default=50.0, help="duty cycle %% (0–100)")
    args = parser.parse_args()

    duty = max(0.0, min(100.0, args.duty))
    if not (0.1 <= args.freq <= 10000):
        print("freq must be between 0.1 and 10000 Hz (or 0 to stop)", file=sys.stderr)
        return 1

    try:
        import lgpio
    except ModuleNotFoundError:
        print("lgpio not found. On Raspberry Pi OS: sudo apt install python3-lgpio", file=sys.stderr)
        return 1

    h = lgpio.gpiochip_open(args.chip)
    if h < 0:
        print(f"gpiochip_open({args.chip}) failed: {h}", file=sys.stderr)
        return 1

    def shutdown() -> None:
        try:
            lgpio.tx_pwm(h, args.gpio, args.freq, 0)
        except Exception:
            pass
        try:
            lgpio.gpiochip_close(h)
        except Exception:
            pass

    ret = lgpio.tx_pwm(h, args.gpio, args.freq, duty)
    if ret < 0:
        print(f"tx_pwm(BCM {args.gpio}, {args.freq} Hz, {duty}%%) failed: {ret}", file=sys.stderr)
        lgpio.gpiochip_close(h)
        return 1

    print(
        f"PWM on BCM {args.gpio}: {args.freq} Hz, {duty}% duty (chip {args.chip}). Ctrl+C to stop.",
        flush=True,
    )
    try:
        while True:
            time.sleep(3600.0)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
