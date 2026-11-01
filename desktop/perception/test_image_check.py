"""Tests for the camera fallback of the post-fault pose check."""

from __future__ import annotations

import math
import unittest

import cv2
import numpy as np

from desktop.localization.pose_verify import BAD, OK, UNKNOWN
from desktop.perception.image_check import ImageCheckConfig, check_frames, image_yaw

W, H = 320, 200
CFG = ImageCheckConfig()
F = (W / 2.0) / math.tan(math.radians(CFG.hfov_deg) / 2.0)


def _scene(seed=0, width=W + 200):
    """A wide textured panorama; frames are horizontal crops of it."""
    rng = np.random.default_rng(seed)
    img = np.full((H, width), 128, np.uint8)
    for _ in range(160):
        x, y = int(rng.integers(0, width - 20)), int(rng.integers(0, H - 20))
        w, h = int(rng.integers(6, 30)), int(rng.integers(6, 30))
        cv2.rectangle(img, (x, y), (x + w, y + h), int(rng.integers(0, 255)), -1)
    return cv2.GaussianBlur(img, (3, 3), 0)


def _view(pano, yaw):
    """Camera turned by ``yaw`` (CCW +): scene shifts right by f*tan(yaw)."""
    off = 100 - int(round(F * math.tan(yaw)))
    return pano[:, off:off + W]


def _jpeg(img):
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()


class TestImageCheck(unittest.TestCase):
    def test_image_yaw_recovers_turn(self):
        pano = _scene()
        yaw, inl = image_yaw(_view(pano, 0.0), _view(pano, math.radians(10.0)))
        self.assertIsNotNone(yaw)
        self.assertGreaterEqual(inl, CFG.min_inliers)
        self.assertAlmostEqual(math.degrees(yaw), 10.0, delta=1.5)

    def test_pose_agrees_is_ok(self):
        pano = _scene(1)
        v = check_frames(_jpeg(_view(pano, 0.0)), (0.0, 0.0, 0.3),
                         _jpeg(_view(pano, math.radians(8.0))),
                         (0.05, 0.0, 0.3 + math.radians(8.0)))
        self.assertEqual(v.kind, OK, v.detail)

    def test_heading_wrong_is_bad(self):
        # Camera says it barely turned; the pose claims 25° (the slip case).
        pano = _scene(2)
        v = check_frames(_jpeg(_view(pano, 0.0)), (0.0, 0.0, 0.0),
                         _jpeg(_view(pano, math.radians(2.0))),
                         (0.0, 0.0, math.radians(25.0)))
        self.assertEqual(v.kind, BAD, v.detail)

    def test_featureless_is_unknown(self):
        blank = np.full((H, W), 120, np.uint8)
        v = check_frames(_jpeg(blank), (0, 0, 0), _jpeg(blank), (0, 0, 0.2))
        self.assertEqual(v.kind, UNKNOWN)

    def test_large_translation_is_unknown(self):
        pano = _scene(3)
        img = _jpeg(_view(pano, 0.0))
        v = check_frames(img, (0, 0, 0), img, (1.0, 0, 0))
        self.assertEqual(v.kind, UNKNOWN)


if __name__ == "__main__":
    unittest.main()
