"""Unit tests for Tier-1 global patrol expansion.

Run: PYTHONPATH=. python3 -m unittest desktop.nav.test_patrol_expand -v
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from desktop.nav.patrol import Patrol, Waypoint
from desktop.nav.patrol_expand import (
    ExpandConfig, expand_patrol, resample_path,
)
from desktop.nav.planner import AStarConfig
from desktop.world_map.costmap import CostmapConfig, build_costmap

# Small fixtures use narrow corridors; drop the planner's 1-cell clearance
# gate so these exercise the routing logic, not clearance (tested in planner).
_NOCLR = AStarConfig(min_clearance_cells=0)


def _patrol(pts, *, loop=False, laps=1):
    return Patrol(
        name="t", session_id="s", authored_utc="now", loop=loop, laps=laps,
        waypoints=[Waypoint(x_m=x, y_m=y) for x, y in pts],
    )


def _costmap_from_driveable(drive, *, res=0.05, ox=0.0, oy=0.0):
    snap = {
        "driveable": drive.astype(np.int8),
        "meta": {"resolution_m": res, "origin_x_m": ox, "origin_y_m": oy,
                 "nx": drive.shape[0], "ny": drive.shape[1], "frame": "world"},
    }
    # No inflation/denoise for the small fixtures: lethal == blocked cells, so
    # the narrow test corridors stay open (real configs inflate by footprint).
    cfg = CostmapConfig(
        footprint_radius_m=0.0, safety_margin_m=0.0, inflation_decay_m=0.01,
        denoise=False, unknown_is_lethal=False,
    )
    return build_costmap(snap, cfg)


class TestResample(unittest.TestCase):
    def test_straight_line_keeps_endpoints(self):
        path = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0)]  # collinear, short
        out = resample_path(path, max_spacing_m=2.0, corner_thresh_rad=0.35)
        self.assertEqual(out[0], (0.0, 0.0))
        self.assertEqual(out[-1], (1.0, 0.0))

    def test_long_straight_caps_spacing(self):
        path = [(0.0, 0.0)] + [(i * 0.1, 0.0) for i in range(1, 51)]  # 5 m
        out = resample_path(path, max_spacing_m=1.0, corner_thresh_rad=0.35)
        gaps = [math.hypot(b[0] - a[0], b[1] - a[1])
                for a, b in zip(out, out[1:])]
        self.assertLessEqual(max(gaps), 1.0 + 1e-6)

    def test_right_angle_corner_preserved(self):
        # East 1 m then north 1 m; the corner vertex must be kept so the
        # straight hops don't cut diagonally across it.
        path = ([(i * 0.1, 0.0) for i in range(0, 11)]
                + [(1.0, j * 0.1) for j in range(1, 11)])
        out = resample_path(path, max_spacing_m=5.0, corner_thresh_rad=0.35)
        self.assertTrue(any(abs(x - 1.0) < 1e-6 and abs(y - 0.0) < 1e-6
                            for x, y in out), "corner (1,0) not kept")

    def test_degenerate(self):
        self.assertEqual(resample_path([], max_spacing_m=1.0,
                                       corner_thresh_rad=0.35), [])
        self.assertEqual(
            resample_path([(1.0, 2.0)], max_spacing_m=1.0,
                          corner_thresh_rad=0.35), [(1.0, 2.0)])


class TestExpand(unittest.TestCase):
    def _pocket_map(self):
        # 40x40 @0.05 (2x2 m). Corridor along the bottom row band; a dead-end
        # pocket pokes UP in the middle. Start left, goal right — the route
        # must run along the corridor, never up into the pocket.
        n = 40
        drive = np.full((n, n), 0, np.int8)   # blocked everywhere
        # Open a horizontal corridor (rows j in [2,6)) across all i.
        drive[:, 2:6] = 1
        # Pocket: a vertical clear stub up from the corridor at i in [18,22),
        # j in [6,30) — clear, but a dead end (no exit at top).
        drive[18:22, 6:30] = 1
        return _costmap_from_driveable(drive)

    def test_routes_around_dead_end(self):
        cm = self._pocket_map()
        # Waypoints at the two corridor ends (world = (i+0.5)*res, (j+0.5)*res).
        a = (2 * 0.05, 4 * 0.05)
        b = (37 * 0.05, 4 * 0.05)
        res = expand_patrol(_patrol([a, b]), cm,
                            ExpandConfig(max_spacing_m=0.3, astar=_NOCLR))
        self.assertTrue(res.ok, res.reason)
        # No sub-waypoint should be up inside the pocket (world y > 0.32 m,
        # i.e. cell j>=6, in the pocket x-band).
        for w in res.patrol.waypoints:
            in_pocket_x = 0.9 <= w.x_m <= 1.1
            self.assertFalse(in_pocket_x and w.y_m > 0.32,
                             f"sub-waypoint entered pocket: {(w.x_m, w.y_m)}")
        # And it actually got from a to b (spacing capped).
        gaps = [math.hypot(p.x_m - q.x_m, p.y_m - q.y_m)
                for p, q in zip(res.patrol.waypoints, res.patrol.waypoints[1:])]
        self.assertLessEqual(max(gaps), 0.3 + 1e-6)

    def test_unreachable_segment_reports(self):
        n = 30
        drive = np.full((n, n), 1, np.int8)
        drive[15, :] = 0          # wall at i=15 (all j) splits i<15 from i>15
        cm = _costmap_from_driveable(drive)
        a = (0.05 * 5, 0.05 * 15)    # i=5  (left of wall)
        b = (0.05 * 25, 0.05 * 15)   # i=25 (right of wall) → unreachable
        res = expand_patrol(_patrol([a, b]), cm, ExpandConfig(astar=_NOCLR))
        self.assertFalse(res.ok)
        self.assertEqual(res.failed_segment, (0, 1))

    def test_loop_closes(self):
        n = 40
        drive = np.full((n, n), 1, np.int8)   # wide open
        cm = _costmap_from_driveable(drive)
        pts = [(0.2, 0.2), (1.6, 0.2), (1.6, 1.6)]
        res = expand_patrol(_patrol(pts, loop=True), cm,
                            ExpandConfig(max_spacing_m=0.5))
        self.assertTrue(res.ok, res.reason)
        self.assertTrue(res.patrol.loop)
        # Loop should not duplicate the start as the final point.
        first = res.patrol.waypoints[0]
        last = res.patrol.waypoints[-1]
        self.assertFalse(abs(first.x_m - last.x_m) < 1e-6
                         and abs(first.y_m - last.y_m) < 1e-6)

    def test_single_waypoint_passthrough(self):
        cm = _costmap_from_driveable(np.full((10, 10), 1, np.int8))
        res = expand_patrol(_patrol([(0.2, 0.2)]), cm)
        self.assertTrue(res.ok)
        self.assertEqual(len(res.patrol.waypoints), 1)

    def test_lead_in_routed_from_start(self):
        cm = _costmap_from_driveable(np.full((40, 40), 1, np.int8))
        pts = [(0.5, 0.5), (1.5, 0.5)]
        start = (0.1, 0.1)
        res = expand_patrol(_patrol(pts), cm, ExpandConfig(max_spacing_m=0.5),
                            start_xy=start)
        self.assertTrue(res.ok, res.reason)
        self.assertIsNotNone(res.lead_in)
        self.assertLess(math.hypot(res.lead_in[0][0] - start[0],
                                   res.lead_in[0][1] - start[1]), 0.1)
        self.assertLess(math.hypot(res.lead_in[-1][0] - pts[0][0],
                                   res.lead_in[-1][1] - pts[0][1]), 0.1)
        # The patrol (marker route) is byte-for-byte unchanged by the lead-in.
        ref = expand_patrol(_patrol(pts), cm, ExpandConfig(max_spacing_m=0.5))
        self.assertEqual([(w.x_m, w.y_m) for w in res.patrol.waypoints],
                         [(w.x_m, w.y_m) for w in ref.patrol.waypoints])

    def test_no_lead_in_without_start(self):
        cm = _costmap_from_driveable(np.full((20, 20), 1, np.int8))
        res = expand_patrol(_patrol([(0.2, 0.2), (0.8, 0.2)]), cm)
        self.assertIsNone(res.lead_in)

    def test_no_lead_in_when_already_at_first_marker(self):
        cm = _costmap_from_driveable(np.full((20, 20), 1, np.int8))
        res = expand_patrol(_patrol([(0.2, 0.2), (0.8, 0.2)]), cm,
                            start_xy=(0.2, 0.2))
        self.assertIsNone(res.lead_in)


if __name__ == "__main__":
    unittest.main()
