"""Tests for the scan-to-scan rotation tie-breaker."""

from __future__ import annotations

import math
import unittest

import numpy as np

from desktop.localization.scan_rotation import estimate_rotation, judge

# A furnished-ish room: outer walls + two boxes (asymmetric so rotation is
# observable). Segments as (x0, y0, x1, y1) in world metres.
_SEGS = [
    (-2.0, -1.5, 3.0, -1.5), (3.0, -1.5, 3.0, 2.0), (3.0, 2.0, -2.0, 2.0),
    (-2.0, 2.0, -2.0, -1.5),
    (1.0, 0.5, 1.6, 0.5), (1.6, 0.5, 1.6, 1.1), (1.6, 1.1, 1.0, 1.1), (1.0, 1.1, 1.0, 0.5),
    (-1.2, -0.9, -0.7, -0.6),
]


def _scan(x: float, y: float, th: float, n: int = 360, noise: float = 0.005,
          seed: int = 0) -> np.ndarray:
    """Body-frame points of a raycast scan from pose (x, y, th)."""
    rng = np.random.default_rng(seed)
    pts = []
    for k in range(n):
        a = -math.pi + 2 * math.pi * k / n
        dx, dy = math.cos(th + a), math.sin(th + a)
        best = None
        for x0, y0, x1, y1 in _SEGS:
            ex, ey = x1 - x0, y1 - y0
            den = dx * ey - dy * ex
            if abs(den) < 1e-12:
                continue
            t = ((x0 - x) * ey - (y0 - y) * ex) / den
            u = ((x0 - x) * dy - (y0 - y) * dx) / den
            if t > 0.05 and 0.0 <= u <= 1.0 and (best is None or t < best):
                best = t
        if best is not None:
            r = best + rng.normal(0.0, noise)
            pts.append((r * math.cos(a), r * math.sin(a)))
    return np.array(pts)


class TestScanRotation(unittest.TestCase):
    def test_recovers_rotation_and_sides_with_true_claim(self) -> None:
        a = _scan(0.3, 0.2, 0.4)
        b = _scan(0.32, 0.21, 0.4 + math.radians(3.0), seed=1)   # barely turned
        # The 2026-11-01 case: wheels claim 21°, IMU ~0°, truth ~3°.
        claims = {"wheels": math.radians(21.0), "imu": math.radians(-1.0)}
        res = estimate_rotation(a, b, claims)
        assert res is not None
        self.assertAlmostEqual(math.degrees(res.dtheta), 3.0, delta=1.5)
        self.assertEqual(judge(res, claims), "imu")

    def test_frozen_imu_case_sides_with_wheels(self) -> None:
        a = _scan(0.0, 0.0, -0.8)
        b = _scan(0.03, -0.02, -0.8 + math.radians(24.0), seed=2)
        claims = {"wheels": math.radians(25.0), "imu": math.radians(0.0)}
        res = estimate_rotation(a, b, claims)
        self.assertEqual(judge(res, claims), "wheels")

    def test_close_claims_are_not_judged(self) -> None:
        a = _scan(0.0, 0.0, 0.0)
        b = _scan(0.0, 0.0, math.radians(5.0), seed=3)
        claims = {"wheels": math.radians(6.0), "imu": math.radians(4.0)}
        self.assertEqual(judge(estimate_rotation(a, b, claims), claims), "unknown")

    def test_sparse_scan_is_unknown(self) -> None:
        a = _scan(0.0, 0.0, 0.0)[:10]
        b = _scan(0.0, 0.0, 0.3)[:10]
        claims = {"wheels": 0.3, "imu": 0.0}
        self.assertIsNone(estimate_rotation(a, b, claims))
        self.assertEqual(judge(None, claims), "unknown")

    def test_neither_claim_matches_is_unknown(self) -> None:
        a = _scan(0.0, 0.0, 0.0)
        b = _scan(0.0, 0.0, math.radians(12.0), seed=4)   # halfway between
        claims = {"wheels": math.radians(24.0), "imu": 0.0}
        self.assertEqual(judge(estimate_rotation(a, b, claims), claims), "unknown")


if __name__ == "__main__":
    unittest.main()
