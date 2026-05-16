"""Unit tests for the Phase 3 AprilTag stack.

Covers the calibration loader, SE(3) math, the filter-side observation
method, and the observer's flow with a mocked detector. The detector
itself (pupil-apriltags) is exercised separately when the lib is
installed; here we mock it so the tests don't pull in the C extension.

Run:
    PYTHONPATH=. python3 -m unittest desktop.world_map.test_apriltag -v
"""
from __future__ import annotations

import json
import math
import threading
import unittest
from typing import Any, Callable, Dict, List
from unittest.mock import MagicMock

import numpy as np

from desktop.world_map.apriltag_calibration import (
    AprilTagCalibration,
    AprilTagWorldPose,
    implied_body_world_pose,
    invert_transform,
    make_transform,
    yaw_from_R,
)
from desktop.world_map.apriltag_detector import CameraIntrinsics, TagDetection
from desktop.world_map.apriltag_observer import (
    AprilTagObserver,
    AprilTagObserverConfig,
)
from desktop.world_map.particle_filter_pose import (
    ParticleFilterConfig,
    ParticleFilterPose,
)


# ── SE(3) math ────────────────────────────────────────────────────────


class TestSE3Math(unittest.TestCase):
    def test_invert_transform_is_inverse(self):
        T = make_transform(1.0, 2.0, 0.5, math.radians(30), math.radians(10), 0.0)
        I = T @ invert_transform(T)
        self.assertTrue(np.allclose(I, np.eye(4), atol=1e-9))

    def test_yaw_from_R_identity(self):
        self.assertAlmostEqual(yaw_from_R(np.eye(3)), 0.0)

    def test_yaw_from_R_pure_yaw(self):
        R = make_transform(0, 0, 0, math.radians(37.5), 0, 0)[:3, :3]
        self.assertAlmostEqual(yaw_from_R(R), math.radians(37.5), places=9)

    def test_implied_body_world_pose_identity_chain(self):
        # All identities → body sits at world origin.
        I4 = np.eye(4)
        x, y, th = implied_body_world_pose(I4, I4, I4)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 0.0)
        self.assertAlmostEqual(th, 0.0)

    def test_implied_body_world_pose_pure_translation(self):
        # Bot at (1, 0, 0°); camera mounted at (0.1, 0, 0) on body
        # (no rotation); tag at (2, 0, 0°) in world. The tag in the
        # camera frame is at world(tag) - world(camera) = (2, 0, 0) -
        # (1.1, 0, 0) = (0.9, 0, 0). Solve backward.
        T_world_tag = make_transform(2.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        T_body_cam = make_transform(0.1, 0.0, 0.0, 0.0, 0.0, 0.0)
        # If body sits at (1, 0, 0°), what's T_cam_tag?
        T_world_body_truth = make_transform(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        T_world_cam = T_world_body_truth @ T_body_cam
        T_cam_tag = invert_transform(T_world_cam) @ T_world_tag
        # Now solve:
        x, y, th = implied_body_world_pose(T_world_tag, T_cam_tag, T_body_cam)
        self.assertAlmostEqual(x, 1.0, places=9)
        self.assertAlmostEqual(y, 0.0, places=9)
        self.assertAlmostEqual(th, 0.0, places=9)

    def test_implied_body_world_pose_with_yaw(self):
        # Bot at (0.5, -0.2, 30°), camera mount offset & tilt, tag
        # somewhere with yaw=180°. Forward-then-backward must agree.
        T_world_body_truth = make_transform(
            0.5, -0.2, 0.0, math.radians(30.0), 0.0, 0.0,
        )
        T_body_cam = make_transform(
            0.1, 0.0, 0.15, 0.0, math.radians(-5.0), 0.0,
        )
        T_world_tag = make_transform(
            2.5, 1.0, 1.0, math.radians(180.0), 0.0, 0.0,
        )
        T_world_cam = T_world_body_truth @ T_body_cam
        T_cam_tag = invert_transform(T_world_cam) @ T_world_tag
        x, y, th = implied_body_world_pose(T_world_tag, T_cam_tag, T_body_cam)
        self.assertAlmostEqual(x, 0.5, places=8)
        self.assertAlmostEqual(y, -0.2, places=8)
        self.assertAlmostEqual(th, math.radians(30.0), places=8)


# ── Calibration loader ───────────────────────────────────────────────


class TestCalibrationLoader(unittest.TestCase):
    def _calib(self) -> Dict[str, Any]:
        return {
            "camera": {
                "intrinsics": {"fx": 800.0, "fy": 800.0, "cx": 320.0, "cy": 200.0},
                "mount": {
                    "x_m": 0.10, "y_m": 0.0, "z_m": 0.15,
                    "yaw_deg": 0.0, "pitch_deg": -5.0, "roll_deg": 0.0,
                },
            },
            "tag_size_m": 0.10,
            "tags": {
                0: {
                    "x_m": 2.5, "y_m": 0.0, "z_m": 1.0,
                    "yaw_deg": 180.0, "pitch_deg": 0.0, "roll_deg": 0.0,
                    "sigma_xy_m": 0.05, "sigma_theta_deg": 5.0,
                },
                7: {
                    "x_m": 0.0, "y_m": 1.5, "z_m": 0.8,
                    "yaw_deg": -90.0,
                    "sigma_xy_m": 0.03, "sigma_theta_deg": 3.0,
                    "tag_size_m": 0.05,
                },
            },
        }

    def test_loader_parses_minimal_config(self):
        c = AprilTagCalibration.from_dict(self._calib())
        self.assertEqual(c.intrinsics.fx, 800.0)
        self.assertEqual(len(c.tags), 2)
        self.assertIn(0, c.tags)
        self.assertIn(7, c.tags)
        self.assertAlmostEqual(c.tags[7].sigma_theta_rad, math.radians(3.0))
        self.assertAlmostEqual(c.tags[7].tag_size_m, 0.05)

    def test_loader_rejects_negative_sigma(self):
        bad = self._calib()
        bad["tags"][0]["sigma_xy_m"] = -0.01
        with self.assertRaises(ValueError):
            AprilTagCalibration.from_dict(bad)

    def test_loader_rejects_zero_tag_size(self):
        bad = self._calib()
        bad["tags"][0]["tag_size_m"] = 0.0
        with self.assertRaises(ValueError):
            AprilTagCalibration.from_dict(bad)

    def test_loader_missing_intrinsics_raises(self):
        bad = self._calib()
        del bad["camera"]["intrinsics"]["fx"]
        with self.assertRaises(ValueError):
            AprilTagCalibration.from_dict(bad)


# ── Filter observation ───────────────────────────────────────────────


class TestObserveXyWorld(unittest.TestCase):
    def test_observation_pulls_xy_toward_value(self):
        pf = ParticleFilterPose(ParticleFilterConfig(
            n_particles=2000,
            init_sigma_xy_m=0.20,
            init_sigma_theta_rad=0.0,
            seed=900,
        ))
        pf.seed_at(0.0, 0.0, 0.0)
        pf.observe_xy_world(0.5, -0.3, sigma_xy_m=0.05)
        x, y, _ = pf.posterior_mean()
        # Posterior xy should shift toward the observation by several cm.
        self.assertGreater(x, 0.2)
        self.assertLess(y, -0.1)

    def test_observation_rejects_zero_sigma(self):
        pf = ParticleFilterPose(ParticleFilterConfig(n_particles=100, seed=901))
        pf.seed_at(0.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            pf.observe_xy_world(0.0, 0.0, sigma_xy_m=0.0)


# ── Observer end-to-end (mocked detector) ────────────────────────────


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
        self.pubs: List[str] = []

    def declare_subscriber(self, topic: str, cb: Callable[[Any], None]):
        self.subs[topic] = cb
        return MagicMock()

    def declare_publisher(self, topic: str):
        self.pubs.append(topic)
        return MagicMock()


class _FakeDetector:
    """Drop-in replacement for AprilTagDetector that returns canned
    detections regardless of input."""

    def __init__(self, detections: List[TagDetection]) -> None:
        self._detections = detections

    def detect_jpeg(self, _jpeg_bytes, **_kwargs) -> List[TagDetection]:
        return list(self._detections)


def _build_calib() -> AprilTagCalibration:
    return AprilTagCalibration.from_dict({
        "camera": {
            "intrinsics": {"fx": 800.0, "fy": 800.0, "cx": 320.0, "cy": 200.0},
            "mount": {"x_m": 0.10},
        },
        "tag_size_m": 0.10,
        "tags": {
            0: {
                "x_m": 2.0, "y_m": 0.0, "z_m": 0.0,
                "yaw_deg": 180.0,
                "sigma_xy_m": 0.05, "sigma_theta_deg": 5.0,
            },
        },
    })


def _rgb_message(b64_payload: str = "AAAA", ok: bool = True) -> _FakeSample:
    msg = {
        "ts": 1000.0,
        "request_id": "test",
        "ok": ok,
        "format": "jpeg",
        "encoding": "base64",
        "data": b64_payload,
        "width": 640, "height": 400,
    }
    if not ok:
        msg = {"ts": 1000.0, "request_id": "test", "ok": False, "error": "oops"}
    return _FakeSample(json.dumps(msg).encode("utf-8"))


class TestObserverFlow(unittest.TestCase):
    def setUp(self):
        self.session = _FakeSession()
        self.pf = ParticleFilterPose(ParticleFilterConfig(
            n_particles=2000,
            init_sigma_xy_m=0.10,
            init_sigma_theta_rad=math.radians(5.0),
            seed=950,
        ))
        self.pf.seed_at(0.0, 0.0, 0.0)
        self.pf_lock = threading.RLock()
        self.calib = _build_calib()

    def _make_observer(
        self, detections: List[TagDetection],
        on_detection=None,
        request_hz: float = 0.0,
    ) -> AprilTagObserver:
        return AprilTagObserver(
            session=self.session, pf=self.pf, pf_lock=self.pf_lock,
            calibration=self.calib, detector=_FakeDetector(detections),
            config=AprilTagObserverConfig(request_hz=request_hz),
            on_detection=on_detection,
        )

    def test_connect_subscribes_to_rgb(self):
        obs = self._make_observer(detections=[])
        obs.connect()
        try:
            self.assertIn("body/oakd/rgb", self.session.subs)
        finally:
            obs.disconnect()

    def test_no_detections_no_observations(self):
        obs = self._make_observer(detections=[])
        obs.connect()
        try:
            log_w_before = self.pf.log_weights.clone()
            self.session.subs["body/oakd/rgb"](_rgb_message())
            self.assertEqual(obs.counters()["frames_processed"], 1)
            self.assertEqual(obs.counters()["detections_total"], 0)
            self.assertEqual(obs.counters()["observations_applied"], 0)
            import torch
            self.assertTrue(torch.allclose(self.pf.log_weights, log_w_before))
        finally:
            obs.disconnect()

    def test_known_tag_applies_observation(self):
        # Plant a fake detection that's geometrically consistent with
        # the bot being at world (0, 0, 0): the calibrated tag at world
        # (2, 0, 180°) seen from a body+camera with the configured
        # mount gives a specific T_cam_tag. Construct that.
        T_world_tag = self.calib.tags[0].T_world_tag
        T_body_cam = self.calib.T_body_cam
        T_world_body_truth = make_transform(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        T_world_cam = T_world_body_truth @ T_body_cam
        T_cam_tag = invert_transform(T_world_cam) @ T_world_tag

        captured: List[Dict[str, Any]] = []
        obs = self._make_observer(
            detections=[TagDetection(
                tag_id=0,
                T_cam_tag=T_cam_tag,
                decision_margin=60.0,
                pose_err=1e-5,
            )],
            on_detection=captured.append,
        )
        obs.connect()
        try:
            self.session.subs["body/oakd/rgb"](_rgb_message())
            self.assertEqual(obs.counters()["observations_applied"], 1)
            self.assertEqual(len(captured), 1)
            implied = captured[0]["implied_world_pose"]
            # Implied body pose should match the (0, 0, 0) we used to
            # construct T_cam_tag (within float tolerance).
            self.assertAlmostEqual(implied[0], 0.0, places=8)
            self.assertAlmostEqual(implied[1], 0.0, places=8)
            self.assertAlmostEqual(implied[2], 0.0, places=8)
        finally:
            obs.disconnect()

    def test_unknown_tag_skipped(self):
        obs = self._make_observer(detections=[TagDetection(
            tag_id=42, T_cam_tag=np.eye(4),
            decision_margin=60.0, pose_err=0.0,
        )])
        obs.connect()
        try:
            self.session.subs["body/oakd/rgb"](_rgb_message())
            self.assertEqual(obs.counters()["detections_unknown_tag"], 1)
            self.assertEqual(obs.counters()["observations_applied"], 0)
        finally:
            obs.disconnect()

    def test_malformed_payload_counted(self):
        obs = self._make_observer(detections=[])
        obs.connect()
        try:
            self.session.subs["body/oakd/rgb"](_FakeSample(b"not-json"))
            self.session.subs["body/oakd/rgb"](_rgb_message(ok=False))
            self.assertEqual(obs.counters()["rgb_malformed"], 1)
            self.assertEqual(obs.counters()["rgb_error_payload"], 1)
        finally:
            obs.disconnect()

    def test_active_request_mode_declares_publisher(self):
        obs = self._make_observer(detections=[], request_hz=2.0)
        obs.connect()
        try:
            self.assertIn("body/oakd/config", self.session.pubs)
        finally:
            obs.disconnect()


if __name__ == "__main__":
    unittest.main()
