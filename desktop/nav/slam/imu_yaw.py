"""IMU yaw tracker.

Consumes ImuReading samples (from body/imu, via ImuReading.from_payload),
extracts yaw from the fused quaternion, unwraps across π boundaries,
and answers yaw_at(ts) queries with a σ estimate.

Handles two fusion modes the same way — the only difference is what
the yaw *means*:
- ROTATION_VECTOR: absolute heading (relative to magnetic north at
  boot-time calibration). Yaw drift should be very small.
- GAME_ROTATION_VECTOR: relative heading (zeroed at boot). Yaw
  drift ~0.5–1°/min; periodic scan-match corrections expected.

Mode propagation: consumers see fusion_mode() and decide whether
the yaw is globally meaningful (e.g. for persistent landmarks).

Boot-time settle:
- ROTATION_VECTOR: the BNO085's accuracy_rad shrinks over ~1–2 s as
  its gyro bias settles. The tracker refuses to answer queries
  (returns None) until `min_settle_samples` consecutive readings
  report accuracy_rad <= settle_accuracy_rad.
- GAME_ROTATION_VECTOR: the SH-2 firmware does not produce a
  dynamic accuracy estimate, so accuracy_rad is the constant
  `imu.game_rotation_vector_accuracy_rad` from Pi config (0.175
  by default) on every sample. The accuracy gate is therefore
  meaningless in this mode; settle is purely a sample-count gate
  (`min_settle_samples` of any GAME_RV samples). Pi already waits
  `imu.settle_time_s` (2 s default) before publishing, so by the
  time desktop sees its first sample the BNO085 startup transient
  is already behind us.

After settle, all readings count and the gate never re-arms.

Thread-safety: Ingest and query may be called from different threads
(Zenoh callback thread ingests; fuser thread queries). All public
methods take an internal lock.
"""
from __future__ import annotations

import logging
import math
import threading
from collections import deque
from typing import Deque, Optional, Tuple

from .types import FusionMode, ImuReading, quaternion_to_yaw

logger = logging.getLogger(__name__)


class ImuYawTracker:
    # Defaults tuned for BNO085:
    #   accuracy_rad < ~3° (0.052 rad) == "calibrated" per SH-2.
    #   Require ~0.2 s of stable readings before going live
    #   (≈ 20 samples at 100 Hz).
    DEFAULT_SETTLE_ACCURACY_RAD = 0.06
    DEFAULT_MIN_SETTLE_SAMPLES = 20

    def __init__(
        self,
        *,
        buffer_seconds: float = 2.0,
        settle_accuracy_rad: float = DEFAULT_SETTLE_ACCURACY_RAD,
        min_settle_samples: int = DEFAULT_MIN_SETTLE_SAMPLES,
    ) -> None:
        self._lock = threading.RLock()
        # (ts, unwrapped_yaw_rad, accuracy_rad) samples, sorted by ts.
        self._buf: Deque[Tuple[float, float, float]] = deque(maxlen=512)
        self._buf_seconds = buffer_seconds
        self._settle_acc = float(settle_accuracy_rad)
        self._min_settle = int(min_settle_samples)
        self._settle_run = 0
        self._settled = False
        self._mode: FusionMode = FusionMode.UNKNOWN
        # Zero-yaw reference: set at first settled sample. For relative
        # fusion modes this becomes the "boot heading." Consumers can
        # ignore it and use raw unwrapped yaw if they prefer.
        self._yaw_zero: Optional[float] = None

    # ── Ingest ───────────────────────────────────────────────────────

    def update(self, reading: ImuReading) -> None:
        if reading.quat_wxyz is None:
            # No orientation report; nothing to track. gyro_z is not
            # used here (consumers can integrate themselves if they
            # want short-horizon propagation beyond the last sample).
            return
        raw_yaw = quaternion_to_yaw(reading.quat_wxyz)
        with self._lock:
            if self._buf:
                prev_yaw = self._buf[-1][1]
                yaw = _unwrap_to(prev_yaw, raw_yaw)
            else:
                yaw = raw_yaw

            if self._buf and reading.ts <= self._buf[-1][0]:
                # Out-of-order sample — skip (IMU ought to be monotonic;
                # treat any inversion as a glitch).
                return

            self._buf.append((reading.ts, yaw, reading.accuracy_rad))
            self._mode = reading.fusion_mode

            # Settle gate. GAME_RV reports a constant accuracy_rad
            # (≈ 0.175) so the accuracy comparison is degenerate;
            # fall back to a pure sample-count gate in that mode.
            sample_passes = (
                reading.fusion_mode == FusionMode.GAME_ROTATION_VECTOR
                or reading.accuracy_rad <= self._settle_acc
            )
            if sample_passes:
                self._settle_run += 1
                if not self._settled and self._settle_run >= self._min_settle:
                    self._settled = True
                    self._yaw_zero = yaw
                    logger.info(
                        f"imu_yaw: settled (mode={self._mode.value}, "
                        f"yaw_zero={math.degrees(yaw):.1f}°, "
                        f"acc={reading.accuracy_rad:.3f} rad)"
                    )
            else:
                self._settle_run = 0

            # Trim by time window
            cutoff = self._buf[-1][0] - self._buf_seconds
            while len(self._buf) > 2 and self._buf[0][0] < cutoff:
                self._buf.popleft()

    # ── Query ────────────────────────────────────────────────────────

    def is_settled(self) -> bool:
        with self._lock:
            return self._settled

    def fusion_mode(self) -> FusionMode:
        with self._lock:
            return self._mode

    def yaw_at(self, ts: float) -> Optional[Tuple[float, float]]:
        """Return (yaw_rad, sigma_rad) at `ts`, or None if unsettled
        or outside buffer grace window.

        yaw_rad is unwrapped (can exceed ±π for long continuous
        rotations). Consumers needing [-π, π] should wrap themselves.
        """
        with self._lock:
            if not self._settled or not self._buf:
                return None
            n = len(self._buf)
            first_ts = self._buf[0][0]
            last_ts = self._buf[-1][0]
            # Grace: one mean inter-arrival on each side. (Prior 0.5×
            # was too tight — shadow-mode logs showed scan timestamps
            # routinely landing 5–10 ms past the latest IMU sample due
            # to publish-jitter, missing the IMU prior on ~37% of
            # match attempts.)
            grace = (last_ts - first_ts) / max(1, n - 1) if n >= 2 else 0.05
            if ts < first_ts - grace or ts > last_ts + grace:
                return None

            # Clamp into buffer range for interpolation.
            if ts <= first_ts:
                _, y, s = self._buf[0]
                return (y, s)
            if ts >= last_ts:
                _, y, s = self._buf[-1]
                return (y, s)

            # Binary search for bracketing pair.
            lo, hi = 0, n - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if self._buf[mid][0] <= ts:
                    lo = mid
                else:
                    hi = mid
            t0, y0, s0 = self._buf[lo]
            t1, y1, s1 = self._buf[hi]
            if t1 == t0:
                return (y1, s1)
            alpha = (ts - t0) / (t1 - t0)
            yaw = y0 + alpha * (y1 - y0)
            # Simple max of neighboring σ; overestimates slightly but
            # is safe (we prefer a larger search window over a tighter
            # one we can't honor).
            sigma = max(s0, s1)
            return (yaw, sigma)

    def latest(self) -> Optional[Tuple[float, float, float]]:
        """Return (ts, yaw, sigma) of the newest sample, or None."""
        with self._lock:
            if not self._buf or not self._settled:
                return None
            return self._buf[-1]


def _unwrap_to(prev: float, raw: float) -> float:
    """Shift `raw` (in [-π, π]) by ±2π so it's within π of `prev`.

    Preserves the prior's branch; monotonic rotation produces a
    continuously increasing or decreasing unwrapped sequence.
    """
    d = raw - prev
    # Fold into (-π, π] first so the while-loops terminate fast.
    while d > math.pi:
        raw -= 2.0 * math.pi
        d = raw - prev
    while d < -math.pi:
        raw += 2.0 * math.pi
        d = raw - prev
    return raw
