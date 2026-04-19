"""MDD10A motor driver: PWM + DIR on Raspberry Pi via lgpio (Pi 5). See docs/motor_controller_spec.md §4.6."""

from __future__ import annotations

from typing import Any


def _err(name: str, ret: int) -> None:
    if ret < 0:
        raise RuntimeError(f"{name} failed: {ret}")


def open_mdd10a(motor_cfg: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Open gpiochip, claim DIR outputs, start PWM at 0% on both channels. Returns (handle, pin_info)."""
    import lgpio

    chip = int(motor_cfg.get("gpio_chip", 0))
    h = lgpio.gpiochip_open(chip)
    _err("gpiochip_open", h)

    pwm_freq = int(motor_cfg.get("pwm_frequency_hz", 1000))
    lp = int(motor_cfg.get("gpio_left_pwm", 12))
    rp = int(motor_cfg.get("gpio_right_pwm", 13))
    ld = int(motor_cfg.get("gpio_left_dir", 5))
    rd = int(motor_cfg.get("gpio_right_dir", 6))

    _err("gpio_claim_output left DIR", lgpio.gpio_claim_output(h, ld, 0))
    _err("gpio_claim_output right DIR", lgpio.gpio_claim_output(h, rd, 0))

    _err("tx_pwm left", lgpio.tx_pwm(h, lp, pwm_freq, 0))
    _err("tx_pwm right", lgpio.tx_pwm(h, rp, pwm_freq, 0))

    pin_info: dict[str, Any] = {
        "left_pwm": lp,
        "right_pwm": rp,
        "left_dir": ld,
        "right_dir": rd,
        "pwm_freq": pwm_freq,
    }
    return h, pin_info


def apply_outputs(
    h: Any,
    pin_info: dict[str, Any],
    left_pwm: float,
    right_pwm: float,
    left_dir: str,
    right_dir: str,
) -> None:
    """Drive MDD10A: duty 0..1, DIR LOW=fwd HIGH=rev per motor_controller_spec truth table."""
    import lgpio

    freq = int(pin_info["pwm_freq"])
    lp = int(pin_info["left_pwm"])
    rp = int(pin_info["right_pwm"])
    ld = int(pin_info["left_dir"])
    rd = int(pin_info["right_dir"])

    lgpio.tx_pwm(h, lp, freq, max(0.0, min(100.0, left_pwm * 100.0)))
    lgpio.tx_pwm(h, rp, freq, max(0.0, min(100.0, right_pwm * 100.0)))
    lgpio.gpio_write(h, ld, 1 if left_dir == "rev" else 0)
    lgpio.gpio_write(h, rd, 1 if right_dir == "rev" else 0)


def shutdown(h: Any, pin_info: dict[str, Any]) -> None:
    """Zero PWM duty, then close chip."""
    import lgpio

    freq = int(pin_info["pwm_freq"])
    for pin in (int(pin_info["left_pwm"]), int(pin_info["right_pwm"])):
        lgpio.tx_pwm(h, pin, freq, 0)
    lgpio.gpiochip_close(h)
