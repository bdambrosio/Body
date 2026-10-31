"""Nav redesign step 1: live global route + soft via-points + live-visible
sub-goal along the path (desktop/nav/route_planner.py + HierarchicalDrive
with a costmap_provider)."""
from __future__ import annotations

import math
import unittest

import numpy as np

from desktop.nav.hierarchical_drive import HierConfig, HierState, HierarchicalDrive
from desktop.nav.route_planner import (
    RouteConfig, path_to_body, plan_route, relax_via, visible_subgoal_along_path,
)
import desktop.nav.test_hierarchical_drive as thd
from desktop.nav.test_hierarchical_drive import FakeDriveIO, FakePose, _clear_scan, _runner
from desktop.world_map.costmap import build_costmap

RES = 0.05
HALF = 3.0


def _world(blocked=()):
    """6×6 m clear world (robot at the origin); ``blocked`` = list of
    (x0, y0, x1, y1) rectangles, world metres."""
    n = int(2 * HALF / RES)
    drive = np.ones((n, n), np.int8)
    for (x0, y0, x1, y1) in blocked:
        i0, i1 = int((x0 + HALF) / RES), int((x1 + HALF) / RES)
        j0, j1 = int((y0 + HALF) / RES), int((y1 + HALF) / RES)
        drive[max(0, i0):min(n, i1 + 1), max(0, j0):min(n, j1 + 1)] = 0
    snap = {"driveable": drive,
            "meta": {"resolution_m": RES, "origin_x_m": -HALF, "origin_y_m": -HALF,
                     "nx": n, "ny": n}}
    return build_costmap(snap)


def _drive(cm, points, scan=None, pose=(0.0, 0.0, 0.0), **cfg):
    io = FakeDriveIO(scan=scan if scan is not None else _clear_scan())
    po = FakePose(pose)
    hd = HierarchicalDrive(_runner(points), po, io, HierConfig(**cfg),
                           costmap_provider=lambda: cm)
    hd.start()
    hd.tick(0.0)            # ALIGNING → SELECT
    hd.tick(0.0)            # SELECT → …
    return hd, io, po


WALL_WITH_GAP = [(1.0, -3.0, 1.05, 0.4)]          # x = 1 m, open for y > 0.4


class TestRoutePlanner(unittest.TestCase):
    def test_route_goes_around_wall_and_subgoal_follows_it(self):
        cm = _world(WALL_WITH_GAP)
        plan = plan_route(cm, (0.0, 0.0), (2.0, 0.0), RouteConfig())
        self.assertTrue(plan.ok, plan.reason)
        self.assertGreater(max(y for _x, y in plan.path_world), 0.5)     # via the gap
        hd, io, _ = _drive(cm, ((2.0, 0.0),))
        self.assertEqual(hd.state(), HierState.DRIVING_SUBGOAL, hd.block_reason())
        bx, by, _t, _v = io.sent[0]
        self.assertLessEqual(math.hypot(bx, by), 1.5 + 1e-6)
        self.assertGreater(by, 0.15)                 # heading for the gap, not the wall
        self.assertIsNotNone(hd.current_route_world())

    def test_via_inside_obstacle_is_relaxed_and_reached(self):
        cm = _world([(1.9, -0.1, 2.1, 0.1)])        # blob right on the marker
        rc = RouteConfig()
        target = relax_via(cm, (2.0, 0.0), rc.via_radius_m, min_clearance_cells=1)
        self.assertIsNotNone(target)
        self.assertGreater(math.hypot(target[0] - 2.0, target[1]), 0.15)
        self.assertLessEqual(math.hypot(target[0] - 2.0, target[1]), rc.via_radius_m + RES)
        hd, io, po = _drive(cm, ((2.0, 0.0), (2.0, 2.0)))
        self.assertEqual(hd.state(), HierState.DRIVING_SUBGOAL, hd.block_reason())
        # Robot arrives at the relaxed point (never at the marker itself):
        # that counts as reaching the via → carrot advances.
        po.pose = (target[0], target[1], 0.0)
        io.set_status(cmd_id=1, state="IDLE", ts=2.0)
        hd.tick(1.0)
        hd.tick(1.0)
        self.assertEqual(hd.current_waypoint_world(), (2.0, 2.0))

    def test_enclosed_via_is_skipped_terminal_blocks(self):
        enclosed = [(1.3, -0.7, 2.7, 0.7)]          # 1.4 m block around the marker
        cm = _world(enclosed)
        hd, io, _ = _drive(cm, ((2.0, 0.0), (0.0, 2.0)))
        self.assertEqual(hd.current_waypoint_world(), (0.0, 2.0))   # skipped
        self.assertEqual(hd.state(), HierState.DRIVING_SUBGOAL)
        hd2, io2, _ = _drive(cm, ((2.0, 0.0),))
        self.assertEqual(hd2.state(), HierState.BLOCKED)
        self.assertEqual(hd2.block_reason(), "terminal_unreachable")
        self.assertEqual(io2.sent, [])

    def test_live_scan_obstacle_shortens_subgoal_along_path(self):
        cm = _world()                                # map says all clear
        wall = thd.TestHierarchicalDrive._seg_scan([(0.6, -5.0, 0.6, 5.0)])
        hd, io, _ = _drive(cm, ((2.0, 0.0),), scan=wall)
        self.assertEqual(hd.state(), HierState.DRIVING_SUBGOAL, hd.block_reason())
        bx, by, _t, _v = io.sent[0]
        self.assertLess(bx, 0.6 - 0.11)              # stops short of the live wall
        self.assertGreaterEqual(bx, 0.2)
        # Wall right in front: nothing drivable along the path → honest block.
        near = thd.TestHierarchicalDrive._seg_scan([(0.3, -5.0, 0.3, 5.0)])
        hd2, io2, _ = _drive(cm, ((2.0, 0.0),), scan=near)
        self.assertEqual(hd2.state(), HierState.BLOCKED)
        self.assertEqual(hd2.block_reason(), "path_blocked_live")

    def test_depth_block_hits_are_stamped_into_the_next_plan(self):
        cm = _world()
        hd, io, po = _drive(cm, ((2.0, 0.0),), stamp_block_hits=True)
        straight = io.sent[0]
        self.assertLess(abs(straight[1]), 0.05)        # straight (± half a cell)
        hits = [[0.6 + k * 0.04, y] for k in range(4) for y in (-0.08, -0.04, 0.0, 0.04, 0.08)]
        io.set_status(cmd_id=1, state="BLOCKED", blocked_reason="depth_block",
                      depth_hit_body=hits, ts=1.0)
        hd.tick(1.0)
        self.assertEqual(hd.state(), HierState.BLOCKED)
        hd.tick(1.6)                                   # retry → replan with the hits
        self.assertEqual(len(io.sent), 2, io.sent)
        bx, by, _t, _v = io.sent[1]
        self.assertGreater(abs(by), 0.12, "retry still aims through the depth hits")

    def test_leaving_from_right_in_front_of_an_obstacle(self):
        # 10-31 14:30: the robot stopped 0.2 m from the bag (depth block),
        # the next via is off to the side. Inside the full inflation every
        # forward sample fails; the path only has to be physically passable
        # (footprint) and its END acceptable to Tier-3.
        cm = _world([(0.15, -0.25, 0.4, 0.25)])      # bag 0.2 m ahead, 0.5 m wide
        wall = thd.TestHierarchicalDrive._seg_scan(
            [(0.2, -0.25, 0.2, 0.25), (0.2, 0.25, 0.4, 0.25), (0.2, -0.25, 0.4, -0.25)])
        hd, io, _ = _drive(cm, ((0.4, 1.5),), scan=wall)
        self.assertEqual(hd.state(), HierState.DRIVING_SUBGOAL, hd.block_reason())
        bx, by, _t, _v = io.sent[0]
        self.assertGreater(by, 0.2)                  # leaves sideways, toward the via
        self.assertGreaterEqual(math.hypot(bx, by), 0.2)

    def test_stamped_hits_are_inflated_like_mapped_obstacles(self):
        cm = _world()
        from desktop.nav.route_planner import stamp_lethal_world
        cm2 = stamp_lethal_world(cm, [(1.0, 0.0)])
        i, j = int((1.0 + HALF) / RES), int((0.0 + HALF) / RES)
        self.assertTrue(cm2.lethal[i, j])
        self.assertTrue(cm2.lethal[i, j + 2])        # 10 cm away: inside the footprint
        self.assertFalse(cm2.lethal[i, j + 6])       # 30 cm away: free
        plan = plan_route(cm, (0.0, 0.0), (2.0, 0.0), RouteConfig(),
                          extra_lethal_world=[(1.0, 0.0)])
        self.assertTrue(plan.ok)
        self.assertGreaterEqual(
            min(math.hypot(x - 1.0, y) for x, y in plan.path_world), 0.15 - RES)

    def test_reference_prior_fills_unobserved_session_cells(self):
        from desktop.nav.route_planner import merge_prior, resample_prior
        ref = np.ones((40, 40), np.int8)                 # 2 m × 2 m at 5 cm, all free
        ref[20:22, :] = 0                                # a wall across y at x = 1.0
        ref[20:22, 30:34] = 1                            # with a doorway
        ref_meta = {"resolution_m": 0.05, "origin_x_m": 0.0, "origin_y_m": 0.0}
        grid_meta = {"resolution_m": 0.1, "origin_x_m": -1.0, "origin_y_m": -1.0,
                     "nx": 40, "ny": 40}
        prior = resample_prior(ref, ref_meta, grid_meta)
        self.assertEqual(prior.shape, (40, 40))
        self.assertEqual(int(prior[5, 5]), -1)           # outside the reference map
        self.assertEqual(int(prior[15, 15]), 1)          # (0.55, 0.55): free
        self.assertEqual(int(prior[20, 15]), 0)          # (1.05, 0.55): wall
        self.assertEqual(int(prior[20, 26]), 1)          # (1.05, 1.65): doorway
        session = np.full((40, 40), -1, np.int8)
        session[15, 15] = 0                              # session saw a box there
        session[20, 15] = 1                              # and the wall gone here
        merged = merge_prior(session, prior)
        self.assertEqual(int(merged[15, 15]), 0)         # session wins
        self.assertEqual(int(merged[20, 15]), 1)
        self.assertEqual(int(merged[20, 16]), 0)         # prior elsewhere

    def test_sealed_via_gets_a_frontier_path(self):
        # Marker inside a closed pocket (a bag on it, walls around): the
        # via's own cell may be free but unreachable. Plan to the reachable
        # cell nearest it instead of "no route".
        pocket = [(1.5, -0.6, 2.5, -0.5), (1.5, 0.5, 2.5, 0.6),
                  (2.4, -0.6, 2.5, 0.6), (1.5, -0.6, 1.6, 0.6)]      # closed box
        cm = _world(pocket)
        plan = plan_route(cm, (0.0, 0.0), (2.0, 0.0), RouteConfig())
        self.assertTrue(plan.ok, plan.reason)
        self.assertEqual(plan.reason, "frontier")
        self.assertLess(plan.target_world[0], 1.5)                  # outside the box
        self.assertLess(math.hypot(plan.target_world[0] - 2.0, plan.target_world[1]), 1.0)
        hd, io, _ = _drive(cm, ((2.0, 0.0), (0.0, 2.0)))
        self.assertEqual(hd.state(), HierState.DRIVING_SUBGOAL, hd.block_reason())
        # Robot wholly enclosed: no progress possible → honest no_route.
        ring = [(-0.4, -0.4, 0.4, -0.3), (-0.4, 0.3, 0.4, 0.4),
                (0.3, -0.4, 0.4, 0.4), (-0.4, -0.4, -0.3, 0.4)]
        plan2 = plan_route(_world(ring), (0.0, 0.0), (2.0, 0.0), RouteConfig())
        self.assertFalse(plan2.ok)
        self.assertEqual(plan2.reason, "no_route")

    def test_bearing_fallback_without_costmap(self):
        io = FakeDriveIO(scan=_clear_scan())
        hd = HierarchicalDrive(_runner(((2.0, 0.0),)), FakePose((0.0, 0.0, 0.0)), io,
                               costmap_provider=lambda: None)
        hd.start()
        hd.tick(0.0)
        hd.tick(0.0)
        self.assertEqual(hd.state(), HierState.DRIVING_SUBGOAL)
        self.assertIsNone(hd.current_route_world())

    def test_visible_subgoal_walks_path_and_backs_off(self):
        meta = {"resolution_m": 0.08, "origin_x_m": -2.5, "origin_y_m": -2.5}
        n = int(5.0 / 0.08)
        grid = np.ones((n, n), np.int8)
        gi = int((1.0 + 2.5) / 0.08)
        grid[gi:gi + 2, :] = 0                        # blocked band at x ≈ 1.0
        path = [(0.0, 0.0), (0.5, 0.3), (1.5, 0.3)]
        out = visible_subgoal_along_path(path, grid, meta, RouteConfig())
        self.assertIsNotNone(out)
        (bx, by), d = out
        self.assertLess(bx, 1.0)
        self.assertGreater(bx, 0.8)
        self.assertAlmostEqual(by, 0.3, delta=0.05)
        self.assertIsNone(visible_subgoal_along_path([(0.0, 0.0), (0.1, 0.0)], grid, meta,
                                                     RouteConfig()))
        pb = path_to_body([(1.0, 1.0), (2.0, 1.0)], (1.0, 1.0, math.pi / 2))
        self.assertAlmostEqual(pb[1][0], 0.0, places=6)
        self.assertAlmostEqual(pb[1][1], -1.0, places=6)


if __name__ == "__main__":
    unittest.main()
