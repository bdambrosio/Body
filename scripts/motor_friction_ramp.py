#!/usr/bin/env python3
"""Per-wheel static / kinetic friction ramp test (feeds motor.* WheelPI config).

For each selected wheel and direction, slowly ramps PWM duty UP from 0 until the
encoder shows sustained motion (static breakaway duty), keeps ramping a little to
sample the duty→speed line, then ramps DOWN until the wheel stops (kinetic /
drop-out duty). Repeats ``--trials`` times and prints suggested config values:

    motor.min_drive_pwm_<side>   static breakaway (+ margin) — the kick floor
    motor.ff_kinetic_pwm_<side>  Coulomb feed-forward offset (kinetic friction)
    motor.max_wheel_vel_ms       from the duty→speed slope (sanity check)

**Stop the stack first** (this script drives the GPIO/PWM directly)::

    sudo systemctl stop body-launcher
    PYTHONPATH=. .venv/bin/python scripts/motor_friction_ramp.py            # both wheels, both dirs
    PYTHONPATH=. .venv/bin/python scripts/motor_friction_ramp.py --wheel left --dir fwd --trials 5

Run it twice: robot ON BLOCKS (wheels free — pure drivetrain friction) and robot
ON THE FLOOR doing the same (adds tyre/scrub load; use the floor numbers for
config, the block numbers as a lower bound). In-place rotation scrub is higher
still — confirm with a slow Sweep-360 afterwards. Battery state matters: note
the pack voltage alongside the results.

``--simulate`` runs the same procedure against a built-in Coulomb-friction wheel
model (no hardware) to check the script itself.
"""

from __future__ import annotations

import argparse
import math
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

MOVING_TICKS_PER_WINDOW = 3  # |ticks| over one hold window that counts as "moving"
CONFIRM_WINDOWS = 2  # consecutive windows to confirm start / stop


class Rig(Protocol):
    def set_duty(self, side: str, duty: float, direction: str) -> None: ...
    def ticks(self, side: str) -> int: ...  # signed delta since last call
    def zero(self) -> None: ...
    def close(self) -> None: ...


@dataclass
class RampResult:
    side: str
    direction: str
    static_duty: float | None
    kinetic_duty: float | None
    samples: list[tuple[float, float]] = field(default_factory=list)  # (duty, |v| m/s) ramping down while moving


def fit_line(samples: list[tuple[float, float]]) -> tuple[float, float] | None:
    """Least squares |v| = k * (duty - d0). Returns (k [m/s per duty], d0) or None."""
    pts = [(d, v) for d, v in samples if v > 0.0]
    if len(pts) < 3:
        return None
    n = len(pts)
    mx = sum(d for d, _ in pts) / n
    my = sum(v for _, v in pts) / n
    sxx = sum((d - mx) ** 2 for d, _ in pts)
    if sxx <= 1e-12:
        return None
    k = sum((d - mx) * (v - my) for d, v in pts) / sxx
    if k <= 0.0:
        return None
    d0 = mx - my / k
    return k, d0


def run_ramp(
    rig: Rig,
    side: str,
    direction: str,
    *,
    step: float,
    hold_s: float,
    max_duty: float,
    overshoot: float,
    m_per_tick: float,
    sleep: Callable[[float], None] = time.sleep,
) -> RampResult:
    res = RampResult(side, direction, None, None)

    def window(duty: float) -> float:
        rig.set_duty(side, duty, direction)
        rig.ticks(side)  # discard ticks from the transition
        sleep(hold_s)
        return float(rig.ticks(side))

    # Ramp up → static breakaway.
    duty = 0.0
    streak = 0
    while duty < max_duty:
        duty = min(max_duty, duty + step)
        t = window(duty)
        streak = streak + 1 if abs(t) >= MOVING_TICKS_PER_WINDOW else 0
        if streak >= CONFIRM_WINDOWS:
            # First of the confirming windows is the breakaway duty.
            res.static_duty = round(duty - (CONFIRM_WINDOWS - 1) * step, 4)
            break
    if res.static_duty is None:
        rig.zero()
        return res

    # A little higher to sample the moving line, then ramp down → drop-out.
    top = min(max_duty, duty + overshoot)
    while duty < top:
        duty = min(top, duty + step)
        window(duty)
    streak = 0
    while duty > 0.0:
        t = window(duty)
        if abs(t) >= MOVING_TICKS_PER_WINDOW:
            res.samples.append((duty, abs(t) * m_per_tick / hold_s))
            streak = 0
        else:
            streak += 1
            if streak >= CONFIRM_WINDOWS:
                res.kinetic_duty = round(duty + (CONFIRM_WINDOWS - 1) * step, 4)
                break
        duty = max(0.0, duty - step)
    rig.zero()
    sleep(max(0.5, 3 * hold_s))  # let it coast to rest before the next run
    return res


def summarize(results: list[RampResult], max_wheel_vel_ms: float, margin: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    slopes: list[float] = []
    for side in ("left", "right"):
        rs = [r for r in results if r.side == side]
        if not rs:
            continue
        stat = [r.static_duty for r in rs if r.static_duty is not None]
        kin = [r.kinetic_duty for r in rs if r.kinetic_duty is not None]
        fits = [f for f in (fit_line(r.samples) for r in rs) if f is not None]
        d0s = [d0 for _, d0 in fits if 0.0 <= d0 < 1.0]
        slopes.extend(k for k, _ in fits)
        side_out: dict[str, Any] = {
            "static_median": statistics.median(stat) if stat else None,
            "static_max": max(stat) if stat else None,
            "kinetic_dropout_median": statistics.median(kin) if kin else None,
            "coulomb_intercept_median": statistics.median(d0s) if d0s else None,
            "runs": len(rs),
            "failed_to_move": sum(1 for r in rs if r.static_duty is None),
        }
        if stat:
            side_out["suggest_min_drive_pwm"] = round(max(stat) * (1.0 + margin), 3)
        kin_est = side_out["coulomb_intercept_median"]
        if kin_est is None and kin:
            kin_est = statistics.median(kin)
        if kin_est is not None:
            # Slightly under the measured offset: the PI makes up a shortfall
            # smoothly, an overestimate makes slow commands overshoot.
            side_out["suggest_ff_kinetic_pwm"] = round(0.9 * kin_est, 3)
        out[side] = side_out
    if slopes:
        k = statistics.median(slopes)
        kins = [out[s].get("suggest_ff_kinetic_pwm", 0.0) for s in out]
        kin = statistics.median(kins) if kins else 0.0
        out["duty_to_speed_slope_mps_per_duty"] = round(k, 4)
        out["suggest_max_wheel_vel_ms"] = round(k * (1.0 - kin / 0.9), 3)
        out["current_max_wheel_vel_ms"] = max_wheel_vel_ms
    return out


def print_report(results: list[RampResult], summary: dict[str, Any]) -> None:
    print("\n=== per-run results ===")
    for r in results:
        fit = fit_line(r.samples)
        fit_s = f"k={fit[0]:.3f} m/s/duty d0={fit[1]:.3f}" if fit else "fit n/a"
        print(
            f"  {r.side:5s} {r.direction}: static={r.static_duty}  dropout={r.kinetic_duty}  "
            f"{fit_s}  ({len(r.samples)} moving samples)"
        )
    print("\n=== summary ===")
    for side in ("left", "right"):
        if side in summary:
            print(f"  {side}: {summary[side]}")
    for k in ("duty_to_speed_slope_mps_per_duty", "suggest_max_wheel_vel_ms", "current_max_wheel_vel_ms"):
        if k in summary:
            print(f"  {k}: {summary[k]}")
    print("\n=== suggested config.json motor.* entries ===")
    for side in ("left", "right"):
        s = summary.get(side, {})
        if "suggest_min_drive_pwm" in s:
            print(f'    "min_drive_pwm_{side}": {s["suggest_min_drive_pwm"]},')
        if "suggest_ff_kinetic_pwm" in s:
            print(f'    "ff_kinetic_pwm_{side}": {s["suggest_ff_kinetic_pwm"]},')
    mins = [summary[s]["suggest_min_drive_pwm"] for s in ("left", "right") if "suggest_min_drive_pwm" in summary.get(s, {})]
    kins = [summary[s]["suggest_ff_kinetic_pwm"] for s in ("left", "right") if "suggest_ff_kinetic_pwm" in summary.get(s, {})]
    if mins:
        print(f'    "min_drive_pwm": {max(mins)},')
    if kins:
        print(f'    "ff_kinetic_pwm": {min(kins)},')
    print(
        "  (per-wheel *_left/_right keys override the shared ones. max_wheel_vel_ms: only change it if the\n"
        "   suggestion differs a lot — it also scales the open-loop fallback.)"
    )


class HardwareRig:
    def __init__(self, motor_cfg: dict[str, Any]) -> None:
        from body.lib import motor_gpio

        self._gpio = motor_gpio
        cfg = dict(motor_cfg)
        cfg["encoders_enabled"] = True
        self.h, self.pins = motor_gpio.open_mdd10a(cfg)
        self.enc = motor_gpio.setup_quadrature_encoders(self.h, cfg)
        if not self.enc or self.enc.get("left") is None or self.enc.get("right") is None:
            self.close()
            raise RuntimeError("both encoders are required (motor.encoder_left/right_enabled)")
        self.duty = {"left": 0.0, "right": 0.0}
        self.dir = {"left": "fwd", "right": "fwd"}

    def set_duty(self, side: str, duty: float, direction: str) -> None:
        self.duty = {"left": 0.0, "right": 0.0}
        self.duty[side] = max(0.0, min(1.0, duty))
        self.dir[side] = direction
        self._gpio.apply_outputs(
            self.h, self.pins, self.duty["left"], self.duty["right"], self.dir["left"], self.dir["right"]
        )

    def ticks(self, side: str) -> int:
        return int(self.enc[side].drain())

    def zero(self) -> None:
        self._gpio.apply_outputs(self.h, self.pins, 0.0, 0.0, "fwd", "fwd")

    def close(self) -> None:
        try:
            self._gpio.zero_outputs(self.pins)
            if self.enc:
                self._gpio.close_quadrature_encoders(self.enc)
        finally:
            self._gpio.shutdown(self.h, self.pins)


class SimRig:
    """Coulomb wheel: starts above d_static, keeps moving above d_kin, |v| = k (d - d_kin)."""

    def __init__(self, m_per_tick: float, d_static: dict[str, float], d_kin: dict[str, float], k: float = 1.0) -> None:
        self.m_per_tick = m_per_tick
        self.d_static, self.d_kin, self.k = d_static, d_kin, k
        self.duty = {"left": 0.0, "right": 0.0}
        self.moving = {"left": False, "right": False}
        self.acc = {"left": 0.0, "right": 0.0}

    def set_duty(self, side: str, duty: float, direction: str) -> None:
        self.duty = {"left": 0.0, "right": 0.0}
        self.duty[side] = duty

    def advance(self, dt: float) -> None:
        for s in ("left", "right"):
            d = self.duty[s]
            if not self.moving[s] and d > self.d_static[s]:
                self.moving[s] = True
            if self.moving[s] and d <= self.d_kin[s]:
                self.moving[s] = False
            v = self.k * (d - self.d_kin[s]) if self.moving[s] else 0.0
            self.acc[s] += v * dt / self.m_per_tick

    def ticks(self, side: str) -> int:
        t = int(self.acc[side])
        self.acc[side] -= t
        return t

    def zero(self) -> None:
        self.duty = {"left": 0.0, "right": 0.0}
        self.moving = {"left": False, "right": False}

    def close(self) -> None:
        pass


def _motor_controller_running() -> bool:
    try:
        r = subprocess.run(["pgrep", "-f", "body.motor_controller"], capture_output=True, text=True)
        return r.returncode == 0
    except OSError:
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wheel", choices=("left", "right", "both"), default="both")
    ap.add_argument("--dir", choices=("fwd", "rev", "both"), default="both")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--step", type=float, default=0.005, help="duty step per window (default 0.005)")
    ap.add_argument("--hold", type=float, default=0.15, help="seconds per duty step (default 0.15)")
    ap.add_argument("--max-duty", type=float, default=0.45, help="never exceed this duty (default 0.45)")
    ap.add_argument("--overshoot", type=float, default=0.10, help="ramp this far past breakaway before ramping down")
    ap.add_argument("--margin", type=float, default=0.05, help="fractional margin on the static suggestion")
    ap.add_argument("--yes", action="store_true", help="skip the safety prompt")
    ap.add_argument("--simulate", action="store_true", help="run against a built-in wheel model (no hardware)")
    args = ap.parse_args(argv)

    from body.lib.zenoh_helpers import load_body_config

    motor_cfg = load_body_config().get("motor", {})
    m_per_tick = 2.0 * math.pi * float(motor_cfg.get("wheel_radius_m", 0.0325)) / float(motor_cfg.get("ticks_per_rev", 1920))
    sides = ("left", "right") if args.wheel == "both" else (args.wheel,)
    dirs = ("fwd", "rev") if args.dir == "both" else (args.dir,)

    sleep: Callable[[float], None] = time.sleep
    if args.simulate:
        sim = SimRig(m_per_tick, {"left": 0.19, "right": 0.22}, {"left": 0.11, "right": 0.13})
        rig: Any = sim

        def sleep(dt: float) -> None:  # noqa: F811 — simulated time
            sim.advance(dt)
    else:
        if _motor_controller_running():
            print("motor_controller is running — stop the stack first (sudo systemctl stop body-launcher).", file=sys.stderr)
            return 2
        if not args.yes:
            print("This drives each wheel alone up to duty", args.max_duty, "— put the robot ON BLOCKS (or clear floor).")
            input("Press Enter to start (Ctrl-C to abort)... ")
        rig = HardwareRig(motor_cfg)

    results: list[RampResult] = []
    try:
        for trial in range(max(1, args.trials)):
            for side in sides:
                for direction in dirs:
                    r = run_ramp(
                        rig, side, direction,
                        step=args.step, hold_s=args.hold, max_duty=args.max_duty,
                        overshoot=args.overshoot, m_per_tick=m_per_tick, sleep=sleep,
                    )
                    results.append(r)
                    print(
                        f"trial {trial + 1} {side:5s} {direction}: static={r.static_duty} "
                        f"dropout={r.kinetic_duty}",
                        flush=True,
                    )
    except KeyboardInterrupt:
        print("\naborted — zeroing PWM", flush=True)
    finally:
        try:
            rig.zero()
        finally:
            rig.close()

    summary = summarize(results, float(motor_cfg.get("max_wheel_vel_ms", 0.0)), args.margin)
    print_report(results, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
