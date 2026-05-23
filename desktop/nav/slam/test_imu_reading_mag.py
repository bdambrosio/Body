"""Tests for ImuReading mag block parsing."""

from __future__ import annotations

import math
import unittest

from desktop.nav.slam.types import FusionMode, ImuReading, quaternion_to_yaw


class TestImuReadingMag(unittest.TestCase):
    def test_from_payload_mag_valid(self) -> None:
        msg = {
            "ts": 1.0,
            "gyro": {"x": 0.0, "y": 0.0, "z": 0.0},
            "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
            "fusion": {"mode": "game_rotation_vector", "accuracy_rad": 0.175},
            "mag": {
                "valid": True,
                "accuracy_rad": 0.04,
                "orientation": {"w": 0.996, "x": 0.0, "y": 0.0, "z": 0.087},
            },
        }
        reading = ImuReading.from_payload(msg)
        self.assertIsNotNone(reading)
        assert reading is not None
        self.assertTrue(reading.mag_valid)
        self.assertAlmostEqual(reading.mag_accuracy_rad or 0.0, 0.04)
        self.assertIsNotNone(reading.mag_quat_wxyz)

    def test_from_payload_mag_absent(self) -> None:
        msg = {
            "ts": 1.0,
            "gyro": {"x": 0.0, "y": 0.0, "z": 0.0},
            "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
            "fusion": {"mode": "game_rotation_vector", "accuracy_rad": 0.175},
        }
        reading = ImuReading.from_payload(msg)
        self.assertIsNotNone(reading)
        assert reading is not None
        self.assertFalse(reading.mag_valid)
        self.assertIsNone(reading.mag_quat_wxyz)


if __name__ == "__main__":
    unittest.main()
