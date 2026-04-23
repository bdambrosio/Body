#!/usr/bin/env python3
"""Standalone encoder diagnostic (no Zenoh, no motor_controller).

Claims two BCM pins as inputs with pull-ups and prints every edge seen,
plus a 1 Hz summary (total edges + current A/B level). Use to verify
that an encoder is wired to the expected pins and is powered (Vcc + GND
connected) before blaming the quadrature decoder.

  sudo .venv/bin/python3 scripts/encoder_read_test.py               # defaults: left encoder 23/24
  sudo .venv/bin/python3 scripts/encoder_read_test.py --a 27 --b 22 # right encoder (pins 13/15)

Ctrl+C to quit.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time

try:
    import lgpio
except ModuleNotFoundError:
    print(
        "lgpio not found. On Raspberry Pi OS: sudo apt install python3-lgpio "
        "(and in a venv, create it with --system-site-packages).",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser(description="Raw encoder edge tester (lgpio)")
    p.add_argument("--chip", type=int, default=0, help="gpiochip index (default 0)")
    p.add_argument("--a", type=int, default=23, help="BCM pin for channel A (default 23)")
    p.add_argument("--b", type=int, default=24, help="BCM pin for channel B (default 24)")
    p.add_argument("--print-edges", action="store_true", help="Print every edge event (noisy at speed)")
    args = p.parse_args()

    h = lgpio.gpiochip_open(args.chip)
    for pin in (args.a, args.b):
        ret = lgpio.gpio_claim_alert(h, pin, lgpio.BOTH_EDGES, lgpio.SET_PULL_UP)
        if ret < 0:
            print(f"gpio_claim_alert({pin}) failed: {ret}", file=sys.stderr)
            lgpio.gpiochip_close(h)
            sys.exit(2)

    a0 = lgpio.gpio_read(h, args.a) & 1
    b0 = lgpio.gpio_read(h, args.b) & 1
    print(
        f"Listening on BCM A={args.a} B={args.b}. Initial levels: A={a0} B={b0}. "
        "Rotate the wheel by hand. Ctrl+C to quit.",
        flush=True,
    )
    if a0 == 0 and b0 == 0:
        print(
            "  WARNING: both lines are LOW with pull-ups enabled — encoder likely not powered "
            "(check Vcc = Pi 3.3V, GND shared) or A/B shorted to GND.",
            flush=True,
        )

    lock = threading.Lock()
    counts = {args.a: 0, args.b: 0}
    total_edges = [0]

    def on_edge(pin_name: str, pin_bcm: int):
        def _cb(_handle, _gpio, level, _tick):
            with lock:
                counts[pin_bcm] += 1
                total_edges[0] += 1
            if args.print_edges:
                print(f"  edge {pin_name}(BCM {pin_bcm}) level={level}", flush=True)

        return _cb

    cb_a = lgpio.callback(h, args.a, lgpio.BOTH_EDGES, on_edge("A", args.a))
    cb_b = lgpio.callback(h, args.b, lgpio.BOTH_EDGES, on_edge("B", args.b))

    stop = threading.Event()

    def handle_sig(_s, _f) -> None:
        stop.set()

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    try:
        while not stop.is_set():
            time.sleep(1.0)
            with lock:
                ea = counts[args.a]
                eb = counts[args.b]
                total = total_edges[0]
            la = lgpio.gpio_read(h, args.a) & 1
            lb = lgpio.gpio_read(h, args.b) & 1
            print(
                f"edges A={ea} B={eb} total={total}  now A/B={la}/{lb}",
                flush=True,
            )
    finally:
        cb_a.cancel()
        cb_b.cancel()
        lgpio.gpiochip_close(h)


if __name__ == "__main__":
    main()
