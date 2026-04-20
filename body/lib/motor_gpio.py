"""MDD10A motor driver: hardware PWM (sysfs) on BCM 12/13 + DIR via lgpio (Pi 5). See docs/motor_controller_spec.md §4.6."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any


def _err(name: str, ret: int) -> None:
    if ret < 0:
        raise RuntimeError(f"{name} failed: {ret}")


def _read_text(path: Path) -> str:
    return path.read_text().strip()


def _write_text(path: Path, value: str) -> None:
    path.write_text(value)


def _find_pwmchip(motor_cfg: dict[str, Any]) -> Path:
    preferred_name = motor_cfg.get("sysfs_pwm_chip")
    pwm_root = Path("/sys/class/pwm")
    if preferred_name:
        preferred = pwm_root / preferred_name
        if preferred.exists():
            return preferred
    chips = sorted(pwm_root.glob("pwmchip*"))
    if not chips:
        raise RuntimeError("No pwmchip devices found under /sys/class/pwm")
    fallback = pwm_root / "pwmchip2"
    if fallback.exists():
        return fallback
    for chip in chips:
        npwm_file = chip / "npwm"
        try:
            if int(_read_text(npwm_file)) >= 4:
                return chip
        except Exception:
            pass
    raise RuntimeError(f"Could not identify RP1 PWM controller. Found: {[c.name for c in chips]}")


def _disable_if_enabled(pwm_dir: Path) -> None:
    enable = pwm_dir / "enable"
    try:
        if _read_text(enable) != "0":
            _write_text(enable, "0\n")
    except Exception:
        pass


def _export_channel(chip: Path, channel: int) -> tuple[Path, bool]:
    """Returns (pwm_dir, we_exported)."""
    pwm_dir = chip / f"pwm{channel}"
    we_exported = False
    if not pwm_dir.exists():
        try:
            _write_text(chip / "export", f"{channel}\n")
            we_exported = True
        except OSError:
            pass
        for _ in range(50):
            if pwm_dir.exists():
                break
            time.sleep(0.02)
    if not pwm_dir.exists():
        raise RuntimeError(f"Failed to export PWM channel {channel} on {chip}")
    return pwm_dir, we_exported


def _set_pinmux_pwm(bcm: int) -> None:
    subprocess.run(
        ["pinctrl", "set", str(bcm), "a0", "pn"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _setup_pwm_channel(
    pwm_dir: Path, period_ns: int, duty_ns: int
) -> None:
    _disable_if_enabled(pwm_dir)
    _write_text(pwm_dir / "period", f"{period_ns}\n")
    _write_text(pwm_dir / "duty_cycle", f"{duty_ns}\n")
    _write_text(pwm_dir / "enable", "1\n")


def _set_pwm_duty(pwm_dir: Path, period_ns: int, duty_0_1: float) -> None:
    duty_ns = min(period_ns, max(0, int(round(period_ns * duty_0_1))))
    dc = pwm_dir / "duty_cycle"
    try:
        _write_text(dc, f"{duty_ns}\n")
    except OSError:
        _disable_if_enabled(pwm_dir)
        _write_text(dc, f"{duty_ns}\n")
        en = pwm_dir / "enable"
        if en.exists():
            _write_text(en, "1\n")


def _unexport_channel(chip: Path, channel: int) -> None:
    try:
        _write_text(chip / "unexport", f"{channel}\n")
    except OSError:
        pass


def _dir_pin_level(rev: bool, invert: bool) -> int:
    """MDD10A: LOW=fwd, HIGH=rev. If invert, swap levels (e.g. left motor mounted 180°)."""
    level = 1 if rev else 0
    return 1 - level if invert else level


def open_mdd10a(motor_cfg: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Open gpiochip, claim DIR outputs, start hardware PWM at 0% on both motor channels. Returns (handle, pin_info)."""
    try:
        import lgpio
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "lgpio is required for motor GPIO. On Raspberry Pi OS: sudo apt install python3-lgpio. "
            "If you use a venv, create it with --system-site-packages (or set include-system-site-packages "
            "= true in .venv/pyvenv.cfg) so the venv can see distro packages. See README.md Install."
        ) from e

    chip_idx = int(motor_cfg.get("gpio_chip", 0))
    h = lgpio.gpiochip_open(chip_idx)
    _err("gpiochip_open", h)

    pwm_freq = int(motor_cfg.get("pwm_frequency_hz", 1000))
    lp = int(motor_cfg.get("gpio_left_pwm", 12))
    rp = int(motor_cfg.get("gpio_right_pwm", 13))
    ld = int(motor_cfg.get("gpio_left_dir", 5))
    rd = int(motor_cfg.get("gpio_right_dir", 6))
    left_ch = int(motor_cfg.get("pwm_sysfs_left_channel", 0))
    right_ch = int(motor_cfg.get("pwm_sysfs_right_channel", 1))
    skip_pinmux = bool(motor_cfg.get("pwm_skip_pinmux", False))
    invert_left_dir = bool(motor_cfg.get("invert_left_dir", False))

    _err(
        "gpio_claim_output left DIR",
        lgpio.gpio_claim_output(h, ld, _dir_pin_level(False, invert_left_dir)),
    )
    _err("gpio_claim_output right DIR", lgpio.gpio_claim_output(h, rd, 0))

    if pwm_freq <= 0:
        lgpio.gpiochip_close(h)
        raise ValueError("pwm_frequency_hz must be positive")

    period_ns = int(round(1_000_000_000 / pwm_freq))

    sysfs_chip = _find_pwmchip(motor_cfg)
    if not skip_pinmux:
        for bcm in (lp, rp):
            try:
                _set_pinmux_pwm(bcm)
            except subprocess.CalledProcessError as e:
                lgpio.gpiochip_close(h)
                raise RuntimeError(
                    "pinctrl failed — install raspi-utils and run with permissions to set pin mux "
                    "(see scripts/pwm13_hdwr_test.py)."
                ) from e

    left_pwm_dir, left_exported = _export_channel(sysfs_chip, left_ch)
    right_pwm_dir, right_exported = _export_channel(sysfs_chip, right_ch)

    try:
        _setup_pwm_channel(left_pwm_dir, period_ns, 0)
        _setup_pwm_channel(right_pwm_dir, period_ns, 0)
    except Exception:
        _disable_if_enabled(left_pwm_dir)
        _disable_if_enabled(right_pwm_dir)
        if left_exported:
            _unexport_channel(sysfs_chip, left_ch)
        if right_exported:
            _unexport_channel(sysfs_chip, right_ch)
        lgpio.gpiochip_close(h)
        raise

    pin_info: dict[str, Any] = {
        "left_pwm": lp,
        "right_pwm": rp,
        "left_dir": ld,
        "right_dir": rd,
        "pwm_freq": pwm_freq,
        "period_ns": period_ns,
        "sysfs_chip": sysfs_chip,
        "left_pwm_dir": left_pwm_dir,
        "right_pwm_dir": right_pwm_dir,
        "left_ch": left_ch,
        "right_ch": right_ch,
        "left_exported": left_exported,
        "right_exported": right_exported,
        "invert_left_dir": invert_left_dir,
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

    period_ns = int(pin_info["period_ns"])
    ld = int(pin_info["left_dir"])
    rd = int(pin_info["right_dir"])
    left_pwm_dir = pin_info["left_pwm_dir"]
    right_pwm_dir = pin_info["right_pwm_dir"]

    _set_pwm_duty(left_pwm_dir, period_ns, max(0.0, min(1.0, left_pwm)))
    _set_pwm_duty(right_pwm_dir, period_ns, max(0.0, min(1.0, right_pwm)))
    inv_left = bool(pin_info.get("invert_left_dir", False))
    lgpio.gpio_write(h, ld, _dir_pin_level(left_dir == "rev", inv_left))
    lgpio.gpio_write(h, rd, _dir_pin_level(right_dir == "rev", False))


def shutdown(h: Any, pin_info: dict[str, Any]) -> None:
    """Zero PWM duty, unexport sysfs channels we exported, then close chip."""
    import lgpio

    left_pwm_dir = pin_info["left_pwm_dir"]
    right_pwm_dir = pin_info["right_pwm_dir"]
    sysfs_chip = pin_info["sysfs_chip"]
    left_ch = int(pin_info["left_ch"])
    right_ch = int(pin_info["right_ch"])

    _disable_if_enabled(left_pwm_dir)
    _disable_if_enabled(right_pwm_dir)

    if pin_info.get("left_exported"):
        _unexport_channel(sysfs_chip, left_ch)
    if pin_info.get("right_exported"):
        _unexport_channel(sysfs_chip, right_ch)

    lgpio.gpiochip_close(h)
