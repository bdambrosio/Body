"""Tests for ParticleFilterPoseSource — the production-mode wrapper.

Covers PoseSource interface conformance, the seed-when-IMU-ready
behavior, predict integration via the .update() path that
FuserController drives, and the connect/disconnect lifecycle.

Run:
    PYTHONPATH=. python3 -m unittest desktop.world_map.test_particle_filter_pose_source -v
"""
from __future__ import annotations

import json
import math
import unittest
from typing import Any, Callable, Dict, List
from unittest.mock import MagicMock

import numpy as np

from desktop.nav.slam.scan_matcher import ScanMatcherConfig
from desktop.nav.slam.types import FusionMode

from desktop.world_map.particle_filter_pose import ParticleFilterConfig
from desktop.world_map.particle_filter_pose_source import (
    ParticleFilterPoseSource,
    ParticleFilterPoseSourceConfig,
)


# ── Mocks ─────────────────────────────────────────────────────────────


class _FakeSample:
    def __init__(self, payload: bytes):
        self._b = payload

    @property
    def payload(self) -> "_FakeSample":
        return self

    def to_bytes(self) -> bytes:
        return self._b


class _FakeSession:
    def __init__(self) -> None:
        self.subs: Dict[str, Callable[[Any], None]] = {}

    def declare_subscriber(self, topic: str, cb):
        self.subs[topic] = cb
        h = MagicMock()
        h.undeclare = MagicMock()
        return h


class _FakeGrid:
    """Empty 8x6 m world with perimeter block-votes."""

    origin_x_m = -4.0
    origin_y_m = -3.0
    resolution_m = 0.04

    def __init__(self) -> None:
        nx = int(8.0 / self.resolution_m)
        ny = int(6.0 / self.resolution_m)
        v = np.zeros((nx, ny), dtype=np.int32)
        v[0, :] = 10
        v[-1, :] = 10
        v[:, 0] = 10
        v[:, -1] = 10
        self._v = v

    def snapshot_block_votes(self) -> np.ndarray:
        return self._v.copy()


def _imu_sample(ts: float, yaw_rad: float = 0.0) -> _FakeSample:
    half = yaw_rad * 0.5
    payload = json.dumps({
        "ts": ts,
        "gyro": {"x": 0.0, "y": 0.0, "z": 0.0},
        "orientation": {
            "w": math.cos(half), "x": 0.0, "y": 0.0, "z": math.sin(half),
        },
        "fusion": {"mode": "game_rotation_vector", "accuracy_rad": 0.05},
    }).encode("utf-8")
    return _FakeSample(payload)


def _settle_imu_to(ps: ParticleFilterPoseSource, until_ts: float, yaw_rad: float = 0.0, n: int = 25):
    """Feed n IMU samples ending at until_ts so the tracker settles."""
    dt = 0.005
    ts0 = until_ts - (n - 1) * dt
    for i in range(n):
        ps._on_imu(_imu_sample(ts0 + i * dt, yaw_rad))


# ── PoseSource interface tests ───────────────────────────────────────


class TestPoseSourceInterface(unittest.TestCase):
    def test_source_name(self):
        ps = ParticleFilterPoseSource(
            pf_config=ParticleFilterConfig(n_particles=200, seed=1),
        )
        self.assertEqual(ps.source_name(), "particle")

    def test_pose_at_returns_none_before_seed(self):
        ps = ParticleFilterPoseSource(
            pf_config=ParticleFilterConfig(n_particles=200, seed=1),
        )
        self.assertIsNone(ps.pose_at(1000.0))
        self.assertIsNone(ps.latest_pose())

    def test_seeds_only_when_imu_ready(self):
        ps = ParticleFilterPoseSource(
            pf_config=ParticleFilterConfig(n_particles=200, seed=2),
        )
        # No IMU samples yet — update() should defer seeding.
        ps.update(1000.0, 0.0, 0.0, 0.0)
        self.assertIsNone(ps.latest_pose())
        self.assertEqual(ps.match_summary()["odom_skipped_no_imu"], 1)

        # Settle IMU at 1000.0, retry — should seed.
        _settle_imu_to(ps, until_ts=1000.0)
        ps.update(1000.0, 0.0, 0.0, 0.0)
        latest = ps.latest_pose()
        self.assertIsNotNone(latest)
        pose, ts = latest
        # Seeded at world (0, 0, 0).
        self.assertAlmostEqual(pose[0], 0.0, places=2)
        self.assertAlmostEqual(pose[1], 0.0, places=2)
        self.assertAlmostEqual(pose[2], 0.0, places=2)
        self.assertEqual(ts, 1000.0)

    def test_update_advances_filter(self):
        ps = ParticleFilterPoseSource(
            pf_config=ParticleFilterConfig(
                n_particles=2000, seed=3,
                init_sigma_xy_m=0.001, init_sigma_theta_rad=0.0,
            ),
        )
        _settle_imu_to(ps, until_ts=1000.0)
        ps.update(1000.0, 0.0, 0.0, 0.0)
        # 10 cm forward over 20 ms.
        _settle_imu_to(ps, until_ts=1000.02)
        ps.update(1000.02, 0.10, 0.0, 0.0)
        latest = ps.latest_pose()
        self.assertIsNotNone(latest)
        pose, _ = latest
        # Filter advanced ~10 cm in world x.
        self.assertAlmostEqual(pose[0], 0.10, delta=0.02)
        self.assertEqual(ps.match_summary()["predicts_run"], 1)

    def test_teleport_detection(self):
        ps = ParticleFilterPoseSource(
            pf_config=ParticleFilterConfig(n_particles=200, seed=4),
        )
        _settle_imu_to(ps, until_ts=1000.0)
        ps.update(1000.0, 0.0, 0.0, 0.0)
        # 2 m jump — far past the 0.5 m threshold.
        _settle_imu_to(ps, until_ts=1000.02)
        ps.update(1000.02, 2.0, 0.0, 0.0)
        self.assertEqual(ps.match_summary()["teleports"], 1)
        self.assertEqual(ps.match_summary()["predicts_run"], 0)

    def test_rebind_world_to_current_zeros_world_pose(self):
        ps = ParticleFilterPoseSource(
            pf_config=ParticleFilterConfig(
                n_particles=2000, seed=5,
                init_sigma_xy_m=0.001, init_sigma_theta_rad=0.0,
            ),
        )
        _settle_imu_to(ps, until_ts=1000.0)
        ps.update(1000.0, 0.0, 0.0, 0.0)
        _settle_imu_to(ps, until_ts=1000.02)
        ps.update(1000.02, 0.10, 0.0, 0.0)
        latest_before = ps.latest_pose()
        self.assertGreater(latest_before[0][0], 0.05)

        # Rebind. Filter should snap to world (0, 0, 0).
        new_off = ps.rebind_world_to_current()
        self.assertIsNotNone(new_off)
        latest_after = ps.latest_pose()
        self.assertAlmostEqual(latest_after[0][0], 0.0, places=2)
        self.assertAlmostEqual(latest_after[0][1], 0.0, places=2)

    def test_to_world_uses_captured_offset(self):
        ps = ParticleFilterPoseSource(
            pf_config=ParticleFilterConfig(n_particles=200, seed=6),
        )
        _settle_imu_to(ps, until_ts=1000.0)
        # Seed at odom (1.0, 0.5, 30°) → world (0, 0, 0)
        ps.update(1000.0, 1.0, 0.5, math.radians(30.0))
        # to_world should map seed odom back to (0, 0, 0).
        x_w, y_w, th_w = ps.to_world(1.0, 0.5, math.radians(30.0))
        self.assertAlmostEqual(x_w, 0.0, places=9)
        self.assertAlmostEqual(y_w, 0.0, places=9)
        self.assertAlmostEqual(th_w, 0.0, places=9)

    def test_cov_at_returns_3x3_after_seed(self):
        ps = ParticleFilterPoseSource(
            pf_config=ParticleFilterConfig(n_particles=200, seed=7),
        )
        self.assertIsNone(ps.cov_at(1000.0))
        _settle_imu_to(ps, until_ts=1000.0)
        ps.update(1000.0, 0.0, 0.0, 0.0)
        cov = ps.cov_at(1000.0)
        self.assertIsNotNone(cov)
        self.assertEqual(cov.shape, (3, 3))


# ── Connection lifecycle ─────────────────────────────────────────────


class TestConnectionLifecycle(unittest.TestCase):
    def test_connect_subscribes_to_imu_and_scan(self):
        ps = ParticleFilterPoseSource(
            pf_config=ParticleFilterConfig(n_particles=200, seed=10),
        )
        session = _FakeSession()
        grid = _FakeGrid()
        ps.connect(session, grid)
        try:
            self.assertIn("body/imu", session.subs)
            self.assertIn("body/lidar/scan", session.subs)
        finally:
            ps.disconnect()

    def test_disconnect_idempotent(self):
        ps = ParticleFilterPoseSource(
            pf_config=ParticleFilterConfig(n_particles=200, seed=11),
        )
        session = _FakeSession()
        grid = _FakeGrid()
        ps.connect(session, grid)
        ps.disconnect()
        ps.disconnect()  # second call is a no-op

    def test_imu_callback_feeds_tracker(self):
        ps = ParticleFilterPoseSource(
            pf_config=ParticleFilterConfig(n_particles=200, seed=12),
        )
        session = _FakeSession()
        grid = _FakeGrid()
        ps.connect(session, grid)
        try:
            cb = session.subs["body/imu"]
            for ts in [1000.0 + 0.005 * i for i in range(25)]:
                cb(_imu_sample(ts, yaw_rad=0.0))
            self.assertEqual(ps.match_summary()["imu_received"], 25)
        finally:
            ps.disconnect()


if __name__ == "__main__":
    unittest.main()
