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

    def test_scan_matched_around_pose_at_scan_ts(self) -> None:
        # S4: the search is centred on the posterior at the scan's ts (from
        # the pose buffer), not on the current (predicted-forward) posterior.
        ref = _corridor_map()
        src = MCLPoseSource(
            ref,
            pf_config=ParticleFilterConfig(n_particles=200, device="cpu"),
            config=MCLPoseSourceConfig(min_evidence_cells=50),
        )
        src._mcl.seed_at(2.0, 1.6, 0.0, sigma_xy_m=0.01, sigma_theta_rad=0.01)
        src._pose_buffer.append(10.0, (1.5, 1.6, 0.0))
        src._pose_buffer.append(10.2, (2.0, 1.6, 0.0))
        angles = np.linspace(-math.pi, math.pi, 90, endpoint=False)
        points = src._scan_points(np.full(90, 2.5), angles)
        self.assertTrue(src._apply_scan_match_observation(points, scan_ts=10.0))
        prior = src.scan_match_summary()["prior_pose"]
        self.assertAlmostEqual(prior[0], 1.5, places=6)
        self.assertEqual(src.match_summary()["scan_prior_at_scan_ts"], 1)
        # Robot motion since scan_ts must not count as a discrete jump.
        self.assertEqual(src.correction_summary()["n_discrete"], 0)
        # Outside the buffer → falls back to the current posterior.
        self.assertTrue(src._apply_scan_match_observation(points, scan_ts=99.0))
        self.assertEqual(src.match_summary()["scan_prior_fallback_now"], 1)


class TestPoseSupportInputs(unittest.TestCase):
    """Explore plan A2: map coverage + scan-time pose + scan listeners."""

    def _src(self):
        ref = _corridor_map()
        src = MCLPoseSource(
            ref,
            pf_config=ParticleFilterConfig(n_particles=200, device="cpu"),
            config=MCLPoseSourceConfig(min_evidence_cells=50),
        )
        src._mcl.seed_at(2.0, 1.6, 0.0, sigma_xy_m=0.01, sigma_theta_rad=0.01)
        return ref, src

    def test_coverage_on_and_off_map(self) -> None:
        _, src = self._src()
        # The fixture's corridor interior (-0.4) reads as unknown; walls
        # are known. Points from (2.0, 1.6) onto both walls:
        on = np.array([[0.0, -1.2], [1.0, 1.2], [-1.0, -1.3]])
        self.assertEqual(src._map_coverage(on, (2.0, 1.6, 0.0)), 1.0)
        off = np.array([[10.0, 0.0], [20.0, 0.0], [0.5, -1.2], [-30.0, 0.0]])
        self.assertAlmostEqual(src._map_coverage(off, (2.0, 1.6, 0.0)), 0.25)

    def test_summary_carries_coverage_and_scan_pose(self) -> None:
        ref, src = self._src()
        src._pose_buffer.append(10.0, (2.0, 1.6, 0.0))
        src._pose_buffer.append(10.2, (2.1, 1.6, 0.0))
        angles = np.linspace(-math.pi, math.pi, 90, endpoint=False)
        points = src._scan_points(_raycast_scan(ref, (2.0, 1.6, 0.0), angles), angles)
        self.assertTrue(src._apply_scan_match_observation(points, scan_ts=10.0))
        sm = src.scan_match_summary()
        self.assertGreater(sm["map_coverage"], 0.9)
        self.assertIsNotNone(sm["scan_pose_world"])
        self.assertAlmostEqual(sm["scan_pose_world"][1], 1.6, delta=0.1)
        # Scan ts outside the buffer → no trustworthy scan-time pose.
        self.assertTrue(src._apply_scan_match_observation(points, scan_ts=99.0))
        self.assertIsNone(src.scan_match_summary()["scan_pose_world"])

    def test_listener_called_from_on_scan(self) -> None:
        import json
        from types import SimpleNamespace
        ref, src = self._src()
        src._seeded = True
        got = []
        src.add_scan_listener(lambda ts, r, a, s: got.append((ts, len(r), s)))
        src.add_scan_listener(lambda *a: 1 / 0)   # a broken listener is contained
        angles = np.linspace(-math.pi, math.pi, 90, endpoint=False)
        ranges = _raycast_scan(ref, (2.0, 1.6, 0.0), angles)
        msg = {"ts": 10.0, "angle_min": float(angles[0]),
               "angle_increment": float(angles[1] - angles[0]),
               "ranges": [None if not np.isfinite(r) else float(r) for r in ranges]}
        src._on_scan(SimpleNamespace(payload=json.dumps(msg).encode()))
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0][0], 10.0)
        self.assertEqual(got[0][1], 90)
        self.assertIn("map_coverage", got[0][2])
        self.assertIn("scan_obs_run", got[0][2])


class TestMCLOdomReset(unittest.TestCase):
    def test_odom_session_change_does_not_predict_jump(self) -> None:
        ref = _corridor_map()
        src = MCLPoseSource(
            ref,
            pf_config=ParticleFilterConfig(n_particles=100, device="cpu"),
            config=MCLPoseSourceConfig(min_evidence_cells=50),
        )
        src._mcl.seed_at(2.0, 1.6, 0.0, sigma_xy_m=0.001, sigma_theta_rad=0.001)
        src._seeded = True
        src._last_odom = (10.0, 3.0, 0.0, 0.0)
        before = src._mcl.posterior_mean()
        w_before = src.to_world(3.0, 0.0, 0.0)
        # Force the EKF to report a reset on this sample.
        src._ekf.update_odom = lambda *a, **k: True
        src.update(10.02, 0.0, 0.0, 0.0, session_id="new", seq=0)
        after = src._mcl.posterior_mean()
        self.assertAlmostEqual(after[0], before[0], places=3)
        self.assertEqual(src.match_summary()["odom_resets"], 1)
        w_after = src.to_world(0.0, 0.0, 0.0)
        for a, b in zip(w_before, w_after):
            self.assertAlmostEqual(a, b, places=6)


if __name__ == "__main__":
    unittest.main()


class TestRotationContest(unittest.TestCase):
    """Wheel-vs-IMU dispute settled by scan-to-scan rotation (2026-11-01)."""

    def _src_with_trip(self):
        from desktop.fusion.test_ekf_pose_tracker import _imu

        src = MCLPoseSource(
            _corridor_map(),
            pf_config=ParticleFilterConfig(n_particles=200, device="cpu"),
            config=MCLPoseSourceConfig(min_evidence_cells=50),
        )
        ekf = src._ekf
        for i in range(250):
            ekf.update_imu(_imu(100.0 + i * 0.01, 0.0))
        ekf.update_odom(102.5, 0.0, 0.0, 0.0)
        t = 103.0
        for i in range(61):            # wheels +0.72 rad, IMU flat
            ts = t + i * 0.02
            ekf.update_imu(_imu(ts, 0.0))
            ekf.update_odom(ts, 0.0, 0.0, 0.012 * i)
        self.assertIsNotNone(ekf.rotation_contest())
        return src, t

    def test_scans_side_with_imu_and_bump_correction_seq(self) -> None:
        from desktop.localization.test_scan_rotation import _scan

        src, t = self._src_with_trip()
        seq0 = src.correction_summary()["n_discrete"]
        src._maybe_judge_rotation_contest(t - 0.05, _scan(0.3, 0.2, 0.4))
        src._maybe_judge_rotation_contest(t + 1.2, _scan(0.3, 0.2, 0.42, seed=1))  # ~1° turn
        self.assertIsNone(src._ekf.rotation_contest())
        self.assertEqual(src.match_summary()["rotation_contests_imu_wins"], 1)
        self.assertGreater(src.correction_summary()["n_discrete"], seq0)
        self.assertTrue(src._ekf.imu_trusted())
        self.assertAlmostEqual(src._ekf.pose_at(t + 1.2)[2], 0.0, places=2)

    def test_scans_side_with_wheels_keeps_distrust(self) -> None:
        from desktop.localization.test_scan_rotation import _scan

        src, t = self._src_with_trip()
        src._maybe_judge_rotation_contest(t - 0.05, _scan(0.3, 0.2, 0.4))
        src._maybe_judge_rotation_contest(t + 1.2, _scan(0.3, 0.2, 0.4 + 0.72, seed=1))
        self.assertIsNone(src._ekf.rotation_contest())
        self.assertEqual(src.match_summary()["rotation_contests_imu_wins"], 0)
        self.assertFalse(src._ekf.imu_trusted())

    def test_waits_for_a_scan_after_the_trip(self) -> None:
        from desktop.localization.test_scan_rotation import _scan

        src, t = self._src_with_trip()
        trip_t1 = src._ekf.rotation_contest()[1]
        src._maybe_judge_rotation_contest(t - 0.05, _scan(0.3, 0.2, 0.4))
        src._maybe_judge_rotation_contest(trip_t1 - 0.01, _scan(0.3, 0.2, 0.4))
        self.assertIsNotNone(src._ekf.rotation_contest())
