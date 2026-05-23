"""Temporal pose ring buffer for timestamp-correct ``pose_at(ts)``."""

from __future__ import annotations

import math
from collections import deque
from threading import RLock
from typing import Deque, Optional, Tuple

import numpy as np

Pose = Tuple[float, float, float]


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _unwrap_pair(a: float, b: float) -> Tuple[float, float]:
    d = b - a
    while d > math.pi:
        b -= 2.0 * math.pi
        d = b - a
    while d < -math.pi:
        b += 2.0 * math.pi
        d = b - a
    return a, b


class PoseBuffer:
    """Thread-safe (ts, pose, optional cov) buffer with interpolation."""

    def __init__(self, *, buffer_seconds: float = 2.0, maxlen: int = 512):
        self._lock = RLock()
        self._buf: Deque[Tuple[float, Pose, Optional[np.ndarray]]] = deque(
            maxlen=maxlen,
        )
        self._buffer_seconds = float(buffer_seconds)

    def append(
        self,
        ts: float,
        pose: Pose,
        cov: Optional[np.ndarray] = None,
    ) -> None:
        with self._lock:
            if self._buf and ts <= self._buf[-1][0]:
                return
            self._buf.append((float(ts), pose, cov))
            cutoff = self._buf[-1][0] - self._buffer_seconds
            while len(self._buf) > 2 and self._buf[0][0] < cutoff:
                self._buf.popleft()

    def latest(self) -> Optional[Tuple[Pose, float]]:
        with self._lock:
            if not self._buf:
                return None
            ts, pose, _ = self._buf[-1]
            return pose, ts

    def pose_at(self, ts: float) -> Optional[Pose]:
        with self._lock:
            n = len(self._buf)
            if n == 0:
                return None
            if n == 1:
                return self._buf[0][1]
            first_ts = self._buf[0][0]
            last_ts = self._buf[-1][0]
            if ts < first_ts:
                grace = (last_ts - first_ts) / max(1, n - 1) * 0.5
                if first_ts - ts <= grace:
                    return self._buf[0][1]
                return None
            if ts > last_ts:
                grace = (last_ts - first_ts) / max(1, n - 1) * 0.5
                if ts - last_ts <= grace:
                    return self._buf[-1][1]
                return None
            lo, hi = 0, n - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if self._buf[mid][0] <= ts:
                    lo = mid
                else:
                    hi = mid
            t0, p0, _ = self._buf[lo]
            t1, p1, _ = self._buf[hi]
            if t1 == t0:
                return p1
            alpha = (ts - t0) / (t1 - t0)
            x = p0[0] + alpha * (p1[0] - p0[0])
            y = p0[1] + alpha * (p1[1] - p0[1])
            ua, ub = _unwrap_pair(p0[2], p1[2])
            th = _wrap(ua + alpha * (ub - ua))
            return (x, y, th)

    def cov_at(self, ts: float) -> Optional[np.ndarray]:
        with self._lock:
            n = len(self._buf)
            if n == 0:
                return None
            best_idx = 0
            best_dt = abs(self._buf[0][0] - ts)
            for i, (t, _, cov) in enumerate(self._buf):
                dt = abs(t - ts)
                if dt < best_dt:
                    best_dt = dt
                    best_idx = i
            return self._buf[best_idx][2]

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()
