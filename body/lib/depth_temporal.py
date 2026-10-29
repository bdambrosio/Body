"""Depth-frame temporal smoothing and capture-time stamping (``body/oakd/depth`` producer).

Pure NumPy, no DepthAI/cv2, so it is unit-testable off the Pi.

Smoothing rule per pixel (depth in mm, 0 = invalid):

- current valid, previous valid, ``|D - S_prev| <= iir_max_delta_mm``:
  ``S = w_old*S_prev + w_new*D`` (IIR).
- current valid otherwise (no previous, or a step larger than the gate — a new
  surface, not noise): ``S = D``.
- current invalid, previous valid and held for fewer than ``hold_max_frames``
  consecutive frames: ``S = S_prev`` (short hold to bridge a one-frame dropout).
- otherwise ``S = 0``.

The hold is bounded so an invalid pixel (dark, textureless, too close, occluded)
can never keep a stale obstacle depth indefinitely (finding R1).
"""
from __future__ import annotations

from typing import Any

import numpy as np


class DepthTemporalSmoother:
    def __init__(
        self,
        hold_max_frames: int = 1,
        iir_max_delta_mm: float = 200.0,
        w_new: float = 2.0 / 3.0,
    ) -> None:
        self.hold_max_frames = max(0, int(hold_max_frames))
        self.iir_max_delta_mm = float(iir_max_delta_mm)
        self.w_new = float(w_new)
        self._prev: np.ndarray | None = None
        self._age: np.ndarray | None = None

    def reset(self) -> None:
        self._prev = None
        self._age = None

    def apply(self, frame_mm: np.ndarray) -> np.ndarray:
        """One frame (any numeric dtype, 0 = invalid) → smoothed float64 frame."""
        curr = np.asarray(frame_mm, dtype=np.float64)
        prev = self._prev
        age = self._age
        if prev is None or age is None or prev.shape != curr.shape:
            prev = np.zeros_like(curr)
            age = np.zeros(curr.shape, dtype=np.int32)
        valid_c = curr > 0.0
        valid_p = prev > 0.0
        out = np.zeros_like(curr)
        new_age = np.zeros_like(age)

        blend = valid_c & valid_p
        if self.iir_max_delta_mm > 0.0:
            blend &= np.abs(curr - prev) <= self.iir_max_delta_mm
        w_new = self.w_new
        out[blend] = (1.0 - w_new) * prev[blend] + w_new * curr[blend]
        take = valid_c & ~blend
        out[take] = curr[take]

        hold = ~valid_c & valid_p & (age < self.hold_max_frames)
        out[hold] = prev[hold]
        new_age[hold] = age[hold] + 1

        self._prev = out.copy()
        self._age = new_age
        return out


def capture_wall_ts(
    frame: Any,
    arrival_wall: float,
    arrival_mono: float,
    max_latency_s: float = 1.0,
) -> float:
    """Wall-clock capture time of a DepthAI frame.

    DepthAI's ``getTimestamp()`` is the capture time already synced to the host
    steady clock (CLOCK_MONOTONIC on Linux, i.e. ``time.monotonic()``). Map it to
    wall clock through the (arrival_wall, arrival_mono) pair taken when the frame
    was dequeued. Falls back to ``arrival_wall`` if the timestamp is missing or
    implausible (negative latency or older than ``max_latency_s``).
    """
    try:
        td = frame.getTimestamp()
        cap_mono = float(td.total_seconds()) if hasattr(td, "total_seconds") else float(td)
    except Exception:
        return float(arrival_wall)
    latency = float(arrival_mono) - cap_mono
    if not (0.0 <= latency <= max_latency_s):
        return float(arrival_wall)
    return float(arrival_wall) - latency
