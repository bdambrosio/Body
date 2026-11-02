"""Scan-to-scan judge for the EKF's wheel-vs-IMU heading contest.

When the EKF's IMU guard trips ("wheels say 21°, IMU says 0°") it opens a
*contest* and keeps the wheel heading until something says who lied. The
lidar can: the rotation between a scan before the dispute and one after,
measured by scan matching alone, is compared with both claims
(``desktop/localization/scan_rotation.py``). Nav (``MCLPoseSource``) has
done this since 11-01; the mapping app did not, so a wheel slip during a
mapping drive became a permanent heading error baked into the map. This
helper is the shared implementation for both.

Pure logic apart from the EKF calls; the caller owns threading.
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np

from desktop.localization.scan_rotation import estimate_rotation, judge

logger = logging.getLogger(__name__)


class RotationContestJudge:
    def __init__(self, ekf: Any, *, history: int = 48, tag: str = "ekf") -> None:
        """``ekf``: an ``EkfPoseTracker`` (needs ``rotation_contest()``,
        ``rotation_between()``, ``resolve_rotation_contest()``).
        ``history``: scans kept (~5 s at 10 Hz)."""
        self._ekf = ekf
        self._tag = tag
        self._hist: Deque[Tuple[float, np.ndarray]] = deque(maxlen=history)
        self.counts: Dict[str, int] = {"judged": 0, "imu_wins": 0}

    def clear(self) -> None:
        self._hist.clear()

    def observe(self, scan_ts: float, points_body: np.ndarray) -> Optional[Tuple[str, bool]]:
        """Record a scan (body-frame xy points, Pi-clock ts) and, if a
        contest is open and this scan closes its window, judge it.
        Returns (verdict, heading_changed) when a verdict was applied, else
        None."""
        self._hist.append((scan_ts, points_body))
        contest = self._ekf.rotation_contest()
        if contest is None or scan_ts < contest[1]:
            return None
        t0, _t1 = contest
        hist = list(self._hist)
        before = [h for h in hist if h[0] <= t0]
        # No scan that old (startup / rate gaps): the oldest one — the
        # claims are measured over the actual scan interval anyway.
        a = before[-1] if before else hist[0]
        b = hist[-1]
        claims = self._ekf.rotation_between(a[0], b[0]) if b[0] > a[0] else None
        if claims is None:
            verdict, detail = "unknown", "no wheel/IMU history over the scan pair"
        else:
            wheel, imu = claims
            hyps = {"wheels": wheel, "imu": imu}
            t_start = time.monotonic()
            result = estimate_rotation(a[1], b[1], hyps)
            verdict = judge(result, hyps)
            ms = (time.monotonic() - t_start) * 1000.0
            if result is None:
                detail = "scans too sparse"
            else:
                detail = (
                    f"scans {math.degrees(result.dtheta):+.0f}° (fit {result.score:.2f}) "
                    f"vs wheels {math.degrees(wheel):+.0f}° "
                    f"({result.hypothesis_scores['wheels']:.2f}) / IMU "
                    f"{math.degrees(imu):+.0f}° ({result.hypothesis_scores['imu']:.2f}) "
                    f"over {b[0] - a[0]:.1f}s, {ms:.0f}ms")
        changed = bool(self._ekf.resolve_rotation_contest(verdict))
        self.counts["judged"] += 1
        if changed:
            self.counts["imu_wins"] += 1
        logger.info("%s: wheel/IMU contest -> %s (%s)", self._tag, verdict, detail)
        return verdict, changed
