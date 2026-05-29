"""Tests for BackUp's drift-immune (body-frame) rear safety check."""
import math
import unittest

import numpy as np

from desktop.nav.primitives import BackUp, BackUpConfig
from desktop.nav.recovery import PRIM_ABORTED, PRIM_RUNNING
from desktop.nav.safety import SafetyConfig
from desktop.world_map.costmap import Costmap


def _meta(res=0.08, n=80):
    return {
        "resolution_m": res,
        "origin_x_m": -(n * res) / 2.0,
        "origin_y_m": -(n * res) / 2.0,
        "nx": n,
        "ny": n,
    }


def _local_map(blocked_behind: bool):
    meta = _meta()
    grid = np.full((meta["nx"], meta["ny"]), 1, dtype=np.int8)
    if blocked_behind:
        res = meta["resolution_m"]
        # 25 cm behind the body (−x).
        i = int(math.floor((-0.25 - meta["origin_x_m"]) / res))
        j = int(math.floor((0.0 - meta["origin_y_m"]) / res))
        grid[i, j] = 0
    return grid, meta


def _costmap_all_lethal(n=200, res=0.05):
    meta = {
        "resolution_m": res, "origin_x_m": -(n * res) / 2.0,
        "origin_y_m": -(n * res) / 2.0, "nx": n, "ny": n,
    }
    return Costmap(
        cost=np.zeros((n, n), dtype=np.float32),
        lethal=np.ones((n, n), dtype=bool),
        unknown=np.zeros((n, n), dtype=bool),
        distance_m=np.zeros((n, n), dtype=np.float32),
        meta=meta,
    )


class TestBackUpRearCheck(unittest.TestCase):
    def test_body_frame_obstacle_aborts(self):
        drive, meta = _local_map(blocked_behind=True)
        back = BackUp(BackUpConfig(
            distance_m=0.2, safety=SafetyConfig(footprint_radius_m=0.22),
            local_map_provider=lambda: (drive, meta),
        ))
        out = back.update(pose=(0.0, 0.0, 0.0), costmap=None)
        self.assertEqual(out.status, PRIM_ABORTED)

    def test_body_frame_clear_overrides_lethal_costmap(self):
        # Drift case: world-frame costmap is all-lethal (wrong pose), but
        # the body-frame local_map is clear behind. Body frame wins, so
        # BackUp proceeds rather than aborting on the stale costmap.
        drive, meta = _local_map(blocked_behind=False)
        back = BackUp(BackUpConfig(
            distance_m=0.2, safety=SafetyConfig(footprint_radius_m=0.22),
            local_map_provider=lambda: (drive, meta),
        ))
        out = back.update(pose=(0.0, 0.0, 0.0), costmap=_costmap_all_lethal())
        self.assertEqual(out.status, PRIM_RUNNING)
        self.assertLess(out.v_mps, 0.0)  # commanding reverse

    def test_falls_back_to_costmap_when_no_local_map(self):
        back = BackUp(BackUpConfig(
            distance_m=0.2, safety=SafetyConfig(),
            local_map_provider=lambda: None,  # local_map missing/stale
        ))
        out = back.update(pose=(0.0, 0.0, 0.0), costmap=_costmap_all_lethal())
        self.assertEqual(out.status, PRIM_ABORTED)

    def test_no_provider_uses_costmap(self):
        # Legacy path: no provider configured at all.
        back = BackUp(BackUpConfig(distance_m=0.2, safety=SafetyConfig()))
        out = back.update(pose=(0.0, 0.0, 0.0), costmap=_costmap_all_lethal())
        self.assertEqual(out.status, PRIM_ABORTED)


if __name__ == "__main__":
    unittest.main()
