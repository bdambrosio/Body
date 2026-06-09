"""Tests for MCLPoseSource scan-match observation."""

from __future__ import annotations

import math
import time
import unittest

import numpy as np

from desktop.localization.mcl_pose_source import MCLPoseSource, MCLPoseSourceConfig
from desktop.reference_map.reference_map import (
    ReferenceMap,
    build_reference_map_from_log_odds,
)
from desktop.world_map.particle_filter_pose import ParticleFilterConfig


def _corridor_map(*, res: float = 0.08) -> ReferenceMap:
    nx, ny = 80, 40
    log_odds = np.full((nx, ny), -0.4, dtype=np.float32)
    log_odds[:, :8] = 0.85
    log_odds[:, -8:] = 0.85
    return build_reference_map_from_log_odds(
        log_odds,
        resolution_m=res,
        origin_x_m=0.0,
        origin_y_m=0.0,
        session_id="test",
    )


def _raycast_scan(
    ref, pose, angles, *, max_range: float = 5.0, step: float = 0.02,
) -> np.ndarray:
    """Synthesize lidar ranges by ray-casting against the map occupancy."""
    x0, y0, th = pose
    occ = ref.occupied_mask()
    nx, ny = occ.shape
    res, ox, oy = ref.resolution_m, ref.origin_x_m, ref.origin_y_m
    ranges = np.full(len(angles), np.nan, dtype=np.float64)
    for k, a in enumerate(angles):
        ca, sa = math.cos(th + a), math.sin(th + a)
        r = step
        while r <= max_range:
            i = int((x0 + ca * r - ox) / res)
            j = int((y0 + sa * r - oy) / res)
            if not (0 <= i < nx and 0 <= j < ny):
                break
            if occ[i, j]:
                ranges[k] = r
                break
            r += step
    return ranges


class TestRelocateAt(unittest.TestCase):
    def _seeded_source(self, *, seed_pose, true_pose):
        ref = _corridor_map()
        src = MCLPoseSource(
            ref,
            pf_config=ParticleFilterConfig(n_particles=200, device="cpu"),
            config=MCLPoseSourceConfig(min_evidence_cells=50),
        )
        src._mcl.seed_at(*seed_pose, sigma_xy_m=0.05, sigma_theta_rad=0.05)
        angles = np.linspace(-math.pi, math.pi, 120, endpoint=False)
        ranges = _raycast_scan(ref, true_pose, angles)
        ts = time.time()
        src._seeded = True
        src._last_odom = (ts, true_pose[0], true_pose[1], true_pose[2])
        src._last_ranges = ranges
        src._last_angles = angles
        src._last_scan_ts = ts
        src._last_scan_recv_mono = time.monotonic()
        return src

    def test_not_seeded_guard(self) -> None:
        ref = _corridor_map()
        src = MCLPoseSource(
            ref,
            pf_config=ParticleFilterConfig(n_particles=50, device="cpu"),
            config=MCLPoseSourceConfig(min_evidence_cells=50),
        )
        self.assertEqual(src.relocate_at(3.0, 1.6)["reason"], "not_seeded")

    def test_locks_xy_and_returns_contract(self) -> None:
        # Seed at the right (x, y) but a wrong heading; relocate_at should
        # keep xy within the small window and report the recovered yaw.
        src = self._seeded_source(
            seed_pose=(3.0, 1.6, math.radians(90.0)),
            true_pose=(3.0, 1.6, 0.0),
        )
        result = src.relocate_at(3.0, 1.6)
        self.assertTrue(result["success"], result)
        self.assertEqual(result["method"], "relocate_at")
        bx, by, _bth = result["best_pose"]
        # xy stays within the configured window (+ one cell of slop).
        self.assertLessEqual(abs(bx - 3.0), 0.10 + src._map.resolution_m)
        self.assertLessEqual(abs(by - 1.6), 0.10 + src._map.resolution_m)
        self.assertIn("improvement", result)
        self.assertIn("evidence_cells", result)

    def test_relocate_at_bumps_discrete_seq_and_reanchors_yaw(self):
        from types import SimpleNamespace

        from desktop.localization.mcl_pose_source import _wrap

        src = self._seeded_source(
            seed_pose=(3.0, 1.6, math.radians(90.0)),
            true_pose=(3.0, 1.6, 0.0),
        )
        # Fake IMU: a fixed raw yaw reading so the re-anchor is checkable.
        imu_yaw = 1.0
        src._imu_tracker = SimpleNamespace(yaw_at=lambda ts: (imu_yaw, 0.0))
        self.assertEqual(src.correction_summary()["n_discrete"], 0)
        result = src.relocate_at(3.0, 1.6)
        self.assertTrue(result["success"], result)
        # An operator relocate is always a discrete correction.
        self.assertEqual(src.correction_summary()["n_discrete"], 1)
        # The IMU yaw constraint must be re-anchored to the relocated heading
        # (offset = imu - θ_seeded), not left in the pre-relocate frame where
        # every subsequent IMU observation drags the posterior back.
        bth = result["best_pose"][2]
        self.assertLess(abs(_wrap(src._yaw_offset - (imu_yaw - bth))), 0.10)


class TestMCLScanMatch(unittest.TestCase):
    def test_scan_match_summary_populated(self) -> None:
        ref = _corridor_map()
        src = MCLPoseSource(
            ref,
            pf_config=ParticleFilterConfig(n_particles=200, device="cpu"),
            config=MCLPoseSourceConfig(min_evidence_cells=50),
        )
        src._mcl.seed_at(2.0, 1.6, 0.0, sigma_xy_m=0.01, sigma_theta_rad=0.01)
        angles = np.linspace(-math.pi, math.pi, 90, endpoint=False)
        ranges = np.full(90, 2.5, dtype=np.float64)
        points = src._scan_points(ranges, angles)
        self.assertIsNotNone(points)
        ok = src._apply_scan_match_observation(points)  # type: ignore[arg-type]
        self.assertTrue(ok)
        sm = src.scan_match_summary()
        self.assertIn("best_pose", sm)
        self.assertIn("prior_pose", sm)
        self.assertIn("elapsed_ms", sm)
        # Ordinary tracking observation: n_applied counts it, but a tightly
        # seeded posterior barely moves, so it is NOT a discrete correction.
        summary = src.correction_summary()
        self.assertEqual(summary["n_applied"], 1)
        self.assertEqual(summary["n_discrete"], 0)


if __name__ == "__main__":
    unittest.main()
