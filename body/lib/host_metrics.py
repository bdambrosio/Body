"""Raspberry Pi (and generic Linux) host telemetry for watchdog / ``body/status``.

- CPU temperature: ``/sys/class/thermal/thermal_zone0/temp`` when present.
- Throttling / **input power problems**: ``vcgencmd get_throttled`` — the firmware sets
  **under_voltage_now** / **under_voltage_occurred** when the **5 V (USB-C/micro-USB) input**
  is too low for the PMIC. That is the right software signal for “5 V sag”, not ``core_volts``.
- **core_volts** (``vcgencmd measure_volts core``): the **ARM logic core rail**, often ~0.8–0.95 V
  depending on load and DVFS. It is **not** the 5 V bus and does **not** move in a way that
  substitutes for measuring input voltage.

There is **no** standard on-board readout of the actual **5 V pin voltage** in Linux on typical
Pi models; for a numeric 5 V reading you need external measurement (DMM, INA219/Power HAT, etc.).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from typing import Any


def _read_cpu_temp_c() -> float | None:
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", encoding="utf-8") as f:
            md = int(f.read().strip())
        return md / 1000.0
    except (OSError, ValueError):
        return None


def _vcgencmd_line(args: list[str]) -> str | None:
    exe = shutil.which("vcgencmd")
    if not exe:
        return None
    try:
        r = subprocess.run(
            [exe, *args],
            capture_output=True,
            text=True,
            timeout=0.5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    return r.stdout.strip()


def _parse_throttled(line: str) -> tuple[int | None, dict[str, bool]]:
    m = re.search(r"throttled=(0x[0-9a-fA-F]+|\d+)", line)
    if not m:
        return None, {}
    raw_s = m.group(1)
    try:
        val = int(raw_s, 0)
    except ValueError:
        return None, {}
    flags = {
        "under_voltage_now": bool(val & (1 << 0)),
        "arm_freq_capped_now": bool(val & (1 << 1)),
        "throttled_now": bool(val & (1 << 2)),
        "soft_temp_limit_now": bool(val & (1 << 3)),
        "under_voltage_occurred": bool(val & (1 << 16)),
        "arm_freq_capped_occurred": bool(val & (1 << 17)),
        "throttled_occurred": bool(val & (1 << 18)),
        "soft_temp_limit_occurred": bool(val & (1 << 19)),
    }
    return val, flags


def _parse_volts(line: str) -> float | None:
    m = re.search(r"([\d.]+)\s*V", line)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def read_host_metrics_dict() -> dict[str, Any]:
    """Return a JSON-serializable dict for ``body/status`` → ``host`` (may be sparse)."""
    out: dict[str, Any] = {"ts": time.time()}
    t = _read_cpu_temp_c()
    if t is not None:
        out["cpu_temp_c"] = round(t, 2)

    throttled_line = _vcgencmd_line(["get_throttled"])
    if throttled_line:
        raw, flags = _parse_throttled(throttled_line)
        if raw is not None:
            out["throttled"] = f"0x{raw:X}"
        out.update(flags)

    # SoC rail only (~0.8 V class); not 5 V input. See module docstring.
    core_line = _vcgencmd_line(["measure_volts", "core"])
    if core_line:
        v = _parse_volts(core_line)
        if v is not None:
            out["core_volts"] = round(v, 4)

    return out
