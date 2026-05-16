"""Integration smoke tests for ShadowParticleFilterDriver.

No live Pi required — uses mock zenoh sample/session/pose-source/grid
stand-ins. Verifies the message flow (decode → predict → observe →
trace write) without exercising real network or filesystem unduly.

Run:
    PYTHONPATH=. python3 -m unittest desktop.world_map.test_shadow_pf_driver -v
"""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np

from desktop.nav.slam.scan_matcher import ScanMatcherConfig

from .shadow_pf_driver import ShadowParticleFilterDriver, ShadowPfConfig


# ── Mocks ─────────────────────────────────────────────────────────────


class _MockSample:
    def __init__(self, payload_bytes: bytes):
        self._payload_bytes = payload_bytes

    @property
    def payload(self):
        return self

    def to_bytes(self) -> bytes:
        return self._payload_bytes


class _MockSession:
    """Captures registered subscribers so tests can poke them directly."""

    def __init__(self):
        self.subs: Dict[str, Callable[[Any], None]] = {}

    def declare_subscriber(
        self, topic: str, callback: Callable[[Any], None],
    ) -> Any:
        self.subs[topic] = callback
        handle = MagicMock()
        handle.undeclare = MagicMock()
        return handle


class _MockGrid:
    """Looks like a WorldGrid: empty world + one block-vote square so
    the matcher has something to score against."""

    origin_x_m = -4.0
    origin_y_m = -3.0
    resolution_m = 0.04

    def __init__(self):
        nx = int(8.0 / self.resolution_m)
        ny = int(6.0 / self.resolution_m)
        votes = np.zeros((nx, ny), dtype=np.int32)
        votes[0, :] = 10
        votes[-1, :] = 10
        votes[:, 0] = 10
        votes[:, -1] = 10
        self._votes = votes

    def snapshot_block_votes(self) -> np.ndarray:
        return self._votes.copy()


class _MockPoseSource:
    """Always-returns-fixed pose source, with a settable trajectory."""

    def __init__(self):
        self._poses: List[Tuple[float, float, float, float]] = []

    def push(self, ts: float, x: float, y: float, theta: float) -> None:
        self._poses.append((ts, x, y, theta))

    def pose_at(self, ts: float) -> Optional[Tuple[float, float, float]]:
        if not self._poses:
            return None
        # Find nearest by ts (test-time mock — simple linear scan).
        nearest = min(self._poses, key=lambda p: abs(p[0] - ts))
        return (nearest[1], nearest[2], nearest[3])


def _odom_sample(ts: float) -> _MockSample:
    payload = json.dumps({"ts": ts}).encode("utf-8")
    return _MockSample(payload)


def _scan_sample(
    ts: float, *, n_rays: int = 360, max_range: float = 5.0,
) -> _MockSample:
    angle_min = -math.pi
    angle_inc = 2.0 * math.pi / n_rays
    # Use a uniform 3 m range — gives the matcher real points to score.
    ranges = [3.0] * n_rays
    payload = json.dumps({
        "ts": ts,
        "angle_min": angle_min,
        "angle_increment": angle_inc,
        "ranges": ranges,
    }).encode("utf-8")
    return _MockSample(payload)


# ── Tests ─────────────────────────────────────────────────────────────


class TestShadowPfDriverFlow(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False,
        )
        self.tmp.close()
        self.trace_path = Path(self.tmp.name)

    def tearDown(self):
        self.trace_path.unlink(missing_ok=True)

    def _build(self):
        session = _MockSession()
        grid = _MockGrid()
        pose_source = _MockPoseSource()
        driver = ShadowParticleFilterDriver(
            session=session, grid=grid, pose_source=pose_source,
            trace_path=self.trace_path,
            scan_matcher_config=ScanMatcherConfig(),
            config=ShadowPfConfig(scan_hz=100.0),  # disable rate-limit in tests
        )
        driver.connect()
        return driver, session, pose_source

    def test_connect_subscribes_to_both_topics(self):
        driver, session, _ = self._build()
        try:
            self.assertIn("body/odom", session.subs)
            self.assertIn("body/lidar/scan", session.subs)
        finally:
            driver.disconnect()

    def test_odom_seed_then_predict(self):
        driver, session, pose_source = self._build()
        try:
            pose_source.push(1000.0, 0.0, 0.0, 0.0)
            session.subs["body/odom"](_odom_sample(1000.0))
            self.assertEqual(driver.counters()["odom_received"], 1)
            # First sample seeds; no predict yet.
            self.assertEqual(driver.counters()["predicts_run"], 0)

            # Second sample at +10 cm forward — drives a predict.
            pose_source.push(1000.02, 0.10, 0.0, 0.0)
            session.subs["body/odom"](_odom_sample(1000.02))
            self.assertEqual(driver.counters()["predicts_run"], 1)
        finally:
            driver.disconnect()

    def test_teleport_detection_reseeds(self):
        driver, session, pose_source = self._build()
        try:
            pose_source.push(1000.0, 0.0, 0.0, 0.0)
            session.subs["body/odom"](_odom_sample(1000.0))
            # Next sample jumps 2 m — well past the 0.5 m threshold.
            pose_source.push(1000.02, 2.0, 0.0, 0.0)
            session.subs["body/odom"](_odom_sample(1000.02))
            self.assertEqual(driver.counters()["teleports_detected"], 1)
            self.assertEqual(driver.counters()["predicts_run"], 0)
        finally:
            driver.disconnect()

    def test_scan_observation_writes_trace_record(self):
        driver, session, pose_source = self._build()
        try:
            # Seed via odom, then deliver a scan.
            pose_source.push(1000.0, 0.0, 0.0, 0.0)
            session.subs["body/odom"](_odom_sample(1000.0))
            pose_source.push(1000.5, 0.0, 0.0, 0.0)
            session.subs["body/lidar/scan"](_scan_sample(1000.5))
            self.assertEqual(driver.counters()["scan_obs_run"], 1)
            self.assertGreater(driver.counters()["trace_records_written"], 0)
        finally:
            driver.disconnect()

        # Inspect the trace file.
        records = [
            json.loads(line)
            for line in self.trace_path.read_text().splitlines()
            if line.strip()
        ]
        self.assertGreaterEqual(len(records), 3)  # start + scan_obs + end
        self.assertEqual(records[0]["type"], "session_start")
        scan_records = [r for r in records if r.get("type") == "scan_obs"]
        self.assertGreaterEqual(len(scan_records), 1)
        rec = scan_records[0]
        # Required summary fields populated.
        for field in (
            "legacy_pose", "filter_mean", "filter_cov_diag",
            "n_eff", "max_weight", "weight_entropy",
            "std_x", "std_y", "std_theta", "resampled",
            "scan_score", "scan_score_prior",
        ):
            self.assertIn(field, rec)
        # Legacy and filter means agree at seed time (within a cm or two).
        legacy = rec["legacy_pose"]
        mean = rec["filter_mean"]
        self.assertAlmostEqual(legacy[0], mean[0], delta=0.05)
        self.assertAlmostEqual(legacy[1], mean[1], delta=0.05)

    def test_malformed_odom_increments_counter_without_crashing(self):
        driver, session, _ = self._build()
        try:
            # Garbage payload — driver should swallow and move on.
            session.subs["body/odom"](_MockSample(b"\x00not-json"))
            self.assertEqual(driver.counters()["odom_malformed"], 1)
        finally:
            driver.disconnect()

    def test_disconnect_idempotent_and_flushes_trace(self):
        driver, _, _ = self._build()
        driver.disconnect()
        # Second disconnect is a no-op.
        driver.disconnect()
        # session_end record landed.
        records = [
            json.loads(line)
            for line in self.trace_path.read_text().splitlines()
            if line.strip()
        ]
        self.assertEqual(records[-1]["type"], "session_end")


if __name__ == "__main__":
    unittest.main()
