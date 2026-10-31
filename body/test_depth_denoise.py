"""Blob gates (body/lib/depth_denoise.py) and the path-following veto
corridor (depth_veto._path_coords) — the two 10-31 fixes for a veto that
fired on ~92 % phantoms and on real objects beside a curving path."""
from __future__ import annotations

import math
import unittest

import numpy as np

from body.lib.depth_denoise import DenoiseConfig, denoised_hits, find_blobs, gate_blobs
from body.lib.depth_veto import DepthVetoTracker, depth_points_body, slab_hits_detail
from body.test_depth_memory import depth_msg, render_depth, veto_cfg

VC = veto_cfg()
DC = DenoiseConfig()


def _points(boxes, pose=(0, 0, 0)):
    arr = render_depth(pose, boxes, VC)
    return arr, depth_points_body(arr, {"width": int(arr.shape[1]), "height": int(arr.shape[0])}, VC)


class TestBlobGates(unittest.TestCase):
    def test_real_objects_survive(self):
        for name, box in (("bag", (0.5, -0.15, 0.8, 0.15, 0.30)),
                          ("shoe", (0.45, -0.05, 0.70, 0.05, 0.08)),
                          ("box", (0.6, -0.1, 0.8, 0.1, 0.15))):
            _arr, dp = _points([box])
            region = (dp.pts[:, 0] > 0.1) & (dp.pts[:, 0] < 0.9)
            n, idx, blobs = denoised_hits(dp, region, VC, DC)
            self.assertGreater(n, 20, name)
            self.assertEqual(len(blobs), 1, name)
            self.assertGreaterEqual(blobs[0].extent_m, 0.06)

    def test_coin_sized_cluster_and_near_field_dropped(self):
        # A 3 cm post 20 cm tall: a stereo-mismatch-sized blob (lidar sees
        # real posts anyway); and the same blob inside the near field.
        for box in ((0.5, -0.015, 0.53, 0.015, 0.20), (0.15, -0.05, 0.22, 0.05, 0.10)):
            _arr, dp = _points([box])
            region = (dp.pts[:, 0] > 0.05) & (dp.pts[:, 0] < 0.9)
            raw = len(gate_blobs(find_blobs(dp, region, VC, DC), DenoiseConfig(
                min_range_m=0.0, min_extent_m=0.0, min_height_m=0.0, min_cells=1)))
            self.assertGreaterEqual(raw, 1)
            n, _idx, blobs = denoised_hits(dp, region, VC, DC)
            self.assertEqual((n, len(blobs)), (0, 0), box)

    def test_thin_floor_slab_dropped_by_height(self):
        # A 4 cm "step" 30 cm wide: floor-transition class, no face.
        _arr, dp = _points([(0.5, -0.15, 0.7, 0.15, 0.045)])
        region = (dp.pts[:, 0] > 0.1) & (dp.pts[:, 0] < 0.9)
        n, _idx, _b = denoised_hits(dp, region, VC, DC)
        self.assertEqual(n, 0)

    def test_disabled_returns_raw_mask(self):
        _arr, dp = _points([(0.5, -0.015, 0.53, 0.015, 0.20)])
        region = (dp.pts[:, 0] > 0.1) & (dp.pts[:, 0] < 0.9)
        n, _idx, _b = denoised_hits(dp, region, VC, DenoiseConfig(enabled=False))
        self.assertGreater(n, 0)


class TestPathCorridor(unittest.TestCase):
    BESIDE = (0.45, 0.18, 0.65, 0.40, 0.30)     # bag to the LEFT of straight-ahead

    def test_arc_corridor_counts_object_the_path_curves_away_from(self):
        arr, _dp = _points([self.BESIDE])
        msg = {"width": int(arr.shape[1]), "height": int(arr.shape[0])}
        # Commanded arc curving left at this instant (ω = +0.25 at 0.18 m/s,
        # r = 0.72 m) sweeps into the bag; the planned path goes straight,
        # then right.
        arc = slab_hits_detail(arr, msg, VC, v_mps=0.18, omega_radps=0.25).hits
        path = [(0.0, 0.0), (0.4, 0.0), (0.8, -0.2)]
        along = DepthVetoTracker(VC)
        along_hits = along.update(depth_msg(arr, ts=1.0, seq=1), now_wall=1.0,
                                  v_mps=0.18, omega_radps=0.25, path_body=path)
        self.assertGreater(arc, VC.min_hits)
        self.assertFalse(along_hits)
        self.assertEqual(along.last_hits, 0)

    def test_path_corridor_counts_object_on_the_path(self):
        arr, _dp = _points([self.BESIDE])
        path = [(0.0, 0.0), (0.3, 0.15), (0.6, 0.30)]     # path runs into it
        tr = DepthVetoTracker(VC)
        for k in (1, 2):
            fired = tr.update(depth_msg(arr, ts=float(k), seq=k), now_wall=float(k),
                              v_mps=0.18, omega_radps=0.0, path_body=path)
        self.assertTrue(fired)
        self.assertGreater(tr.last_hits, VC.min_hits)


if __name__ == "__main__":
    unittest.main()
