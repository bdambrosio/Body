"""Object tasks, look-only phase: grammar, box geometry, depth range, memory."""
from __future__ import annotations

import math
import os
import tempfile
import unittest

import numpy as np

from desktop.tasks.commands import Command, parse
from desktop.tasks.locate import (CameraModel, bbox_to_bearing, depth_range_m,
                                  norm1000_to_px, sighting_from_answer,
                                  world_point)
from desktop.tasks.memory import ObjectMemory

RGB = CameraModel(558.3, 558.3, 320.0, 200.0, 640, 400)
DEPTH = CameraModel(85.3, 85.3, 60.0, 45.0, 120, 90)


class TestParse(unittest.TestCase):
    def test_verbs_and_aliases(self):
        self.assertEqual(parse("goto red ball"), Command("goto", "red ball"))
        self.assertEqual(parse("Go to the dog."), Command("goto", "dog"))
        self.assertEqual(parse("look for a red ball"), Command("find", "red ball"))
        self.assertEqual(parse("where is the cat?"), Command("find", "cat"))
        self.assertEqual(parse("follow me"), Command("follow", "person"))
        self.assertEqual(parse("look at the chair"), Command("look", "chair"))
        self.assertEqual(parse("  STOP "), Command("stop"))

    def test_rejects(self):
        self.assertIsNone(parse(""))
        self.assertIsNone(parse("goto"))
        self.assertIsNone(parse("dance"))
        self.assertIsNone(parse("gotoball"))


class TestGeometry(unittest.TestCase):
    def test_bearing_sign_and_size(self):
        self.assertAlmostEqual(bbox_to_bearing((300, 0, 340, 10), RGB), 0.0)
        right = bbox_to_bearing((600, 0, 640, 10), RGB)     # u=620
        self.assertLess(right, 0.0)
        self.assertAlmostEqual(right, math.atan2(-300, 558.3))
        self.assertGreater(bbox_to_bearing((0, 0, 40, 10), RGB), 0.0)

    def test_norm1000(self):
        self.assertEqual(norm1000_to_px((500, 0, 250, 1000), 640, 400),
                         (160.0, 0.0, 320.0, 400.0))

    def test_depth_range_picks_object_over_background(self):
        d = np.full((90, 120), 4000, np.uint16)        # wall at 4 m
        d[40:50, 55:65] = 1500                          # object at 1.5 m, centred
        r = depth_range_m((300, 180, 340, 220), RGB, d, DEPTH)
        self.assertAlmostEqual(r, 1.5, places=2)

    def test_depth_range_none_when_invalid(self):
        d = np.zeros((90, 120), np.uint16)
        self.assertIsNone(depth_range_m((300, 180, 340, 220), RGB, d, DEPTH))
        self.assertIsNone(depth_range_m((700, 0, 800, 10), RGB,
                                        np.full((90, 120), 1000, np.uint16), DEPTH))

    def test_depth_range_scales_depth_model(self):
        d = np.full((45, 60), 2000, np.uint16)          # half-res depth
        self.assertAlmostEqual(
            depth_range_m((300, 180, 340, 220), RGB, d, DEPTH), 2.0, places=2)

    def test_world_point(self):
        x, y = world_point((1.0, 2.0, math.pi / 2), 0.0, 2.0)
        self.assertAlmostEqual(x, 1.0)
        self.assertAlmostEqual(y, 4.0)


class TestSighting(unittest.TestCase):
    def test_present_with_depth_gets_world_xy(self):
        d = np.full((90, 120), 2000, np.uint16)
        s = sighting_from_answer(
            "ball", {"present": True, "bbox_2d": [470, 450, 530, 550],
                     "confidence": 0.9},
            RGB, ts=10.0, pose=(0.0, 0.0, 0.0), depth_mm=d, depth_cam=DEPTH)
        self.assertTrue(s.present)
        self.assertAlmostEqual(s.bearing, 0.0, places=2)
        self.assertAlmostEqual(s.range_m, 2.0, places=2)
        self.assertAlmostEqual(s.world_xy[0], 2.0, places=2)

    def test_not_present_and_junk(self):
        s = sighting_from_answer("ball", {"present": False, "bbox_2d": [0, 0, 0, 0],
                                          "confidence": 0.8}, RGB, ts=1.0)
        self.assertFalse(s.present)
        s = sighting_from_answer("ball", {"present": True, "bbox_2d": [5, 5, 5, 5],
                                          "confidence": 0.8}, RGB, ts=1.0)
        self.assertFalse(s.present)                     # degenerate box
        self.assertIsNone(sighting_from_answer("ball", None, RGB, ts=1.0))
        self.assertIsNone(sighting_from_answer("ball", {"present": "yes"}, RGB, ts=1.0))

    def test_no_depth_means_bearing_only(self):
        s = sighting_from_answer("ball", {"present": True, "bbox_2d": [0, 0, 100, 100],
                                          "confidence": 0.7}, RGB, ts=1.0,
                                 pose=(0, 0, 0))
        self.assertTrue(s.present)
        self.assertGreater(s.bearing, 0)
        self.assertIsNone(s.range_m)
        self.assertIsNone(s.world_xy)


def _seen(label, xy, conf=0.8, ts=0.0):
    from desktop.tasks.locate import Sighting
    return Sighting(label, True, conf, (0, 0, 1, 1), 0.0, 1.0, (0, 0, 0), xy, ts)


class TestMemory(unittest.TestCase):
    def test_merge_expire_and_best(self):
        m = ObjectMemory(merge_radius_m=0.5, max_age_s=100)
        m.add(_seen("ball", (1.0, 1.0), ts=0.0))
        m.add(_seen("ball", (1.2, 1.0), ts=5.0))       # same ball
        m.add(_seen("ball", (5.0, 5.0), ts=3.0))       # another one
        self.assertEqual(len(m.all("ball", now=10)), 2)
        best = m.best("ball", now=10)
        self.assertEqual(best.count, 2)
        self.assertAlmostEqual(best.x, 1.1)
        self.assertEqual(m.best("ball", now=104).x, best.x)  # ts 5 still live
        self.assertIsNone(m.best("ball", now=200))
        self.assertIsNone(m.best("dog", now=0))

    def test_ignores_unplaced(self):
        m = ObjectMemory()
        self.assertIsNone(m.add(_seen("ball", None)))

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "mem.json")
            m = ObjectMemory(path=p)
            m.add(_seen("dog", (2.0, 3.0), ts=1.0))
            m.save()
            m2 = ObjectMemory(path=p)
            m2.load()
            self.assertAlmostEqual(m2.best("dog", now=2.0).y, 3.0)


if __name__ == "__main__":
    unittest.main()
