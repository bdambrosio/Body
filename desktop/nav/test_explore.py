"""Explore Phase D: viewpoint planner, investigate spot, ExploreTask + TaskRunner."""
from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np

from desktop.nav.explore import (
    ExploreConfig, ExploreMap, ExplorePlanner, cast_visible, investigate_spot,
    reachable_distances,
)
from desktop.nav.hierarchical_drive import HierState
from desktop.nav.tasks import ExploreContext, ExploreTask, Leg, TaskRunner

RES = 0.05
META = {"resolution_m": RES, "origin_x_m": 0.0, "origin_y_m": 0.0, "nx": 120, "ny": 60}


def two_rooms(door=True):
    """6 m × 3 m, walls round the edge, a dividing wall at x = 3 m with a
    0.8 m door (y 1.1–1.9) — unknown outside."""
    d = np.full((120, 60), -1, np.int8)
    d[1:119, 1:59] = 1
    d[0, :] = d[119, :] = 0
    d[:, 0] = d[:, 59] = 0
    d[60, 1:59] = 0
    if door:
        d[60, 22:38] = 1
    return d


def costmap_of(drive, extra_lethal=None):
    lethal = drive != 1
    if extra_lethal is not None:
        lethal = lethal | extra_lethal
    return SimpleNamespace(lethal=lethal, unknown=drive == -1, meta=META)


def cell(x, y):
    return int(x / RES), int(y / RES)


class TestExplorePlanner(unittest.TestCase):
    def test_rays_stop_at_walls(self):
        d = two_rooms(door=False)
        vis = cast_visible(d, *cell(1.5, 1.5), res=RES, range_m=4.0, n_rays=180)
        xs = (vis // d.shape[1]) * RES
        self.assertLessEqual(xs.max(), 3.05)            # nothing past the dividing wall
        self.assertIn(60 * 60 + 30, set(vis.tolist()))  # the wall cell itself is seen

    def test_viewpoints_clear_of_walls_and_nogo(self):
        d = two_rooms()
        nogo = np.zeros_like(d, bool)
        nogo[80:120, :] = True
        em = ExploreMap(d, nogo, META, ExploreConfig(viewpoint_min_clearance_m=0.3))
        self.assertTrue(em.viewpoints)
        for i, j in em.viewpoints:
            self.assertLess(i, 80)
            self.assertTrue(np.all(d[i - 5:i + 6, j - 5:j + 6] == 1))

    def test_greedy_goes_to_the_unseen_room_and_coverage_grows(self):
        d = two_rooms()
        em = ExploreMap(d, None, META)
        pl = ExplorePlanner(em)
        seen = np.zeros_like(d, bool)
        seen[:60, :] = True                              # left room already seen
        dist = reachable_distances(costmap_of(d).lethal, None, META, (1.0, 1.5))
        c0 = pl.coverage(seen)
        ch = pl.choose(seen, dist)
        self.assertGreater(ch.xy[0], 3.0)
        pl.mark_visited(ch.index)
        self.assertGreater(pl.coverage(seen), c0)
        # Everything looked at: nothing left worth a leg.
        for k in range(len(em.viewpoints)):
            pl.mark_visited(k)
        self.assertIsNone(pl.choose(seen, dist))
        self.assertAlmostEqual(pl.coverage(seen), 1.0)

    def test_unreachable_and_abandoned_viewpoints_are_skipped(self):
        d = two_rooms(door=False)
        em = ExploreMap(d, None, META)
        pl = ExplorePlanner(em)
        dist = reachable_distances(costmap_of(d).lethal, None, META, (1.0, 1.5))
        self.assertFalse(np.isfinite(dist[cell(4.5, 1.5)]))
        ch = pl.choose(None, dist)
        self.assertLess(ch.xy[0], 3.0)
        pl.mark_abandoned(ch.index)
        self.assertNotEqual(pl.choose(None, dist).index, ch.index)

    def test_investigate_spot_range_sight_and_new_angle(self):
        d = two_rooms()
        dist = reachable_distances(costmap_of(d).lethal, None, META, (1.0, 1.5))
        change, first = (4.5, 1.5), (3.3, 1.5)          # first seen from the west
        sp = investigate_spot(d, META, dist, change, first, ExploreConfig())
        rng = math.hypot(sp.xy[0] - change[0], sp.xy[1] - change[1])
        self.assertTrue(1.0 <= rng <= 1.5 + RES)
        self.assertTrue(sp.new_angle)
        b0 = math.atan2(first[1] - change[1], first[0] - change[0])
        b = math.atan2(sp.xy[1] - change[1], sp.xy[0] - change[0])
        self.assertGreaterEqual(abs((b - b0 + math.pi) % (2 * math.pi) - math.pi),
                                math.radians(30) - 1e-6)
        self.assertGreater(sp.xy[0], 3.05)               # not through the dividing wall
        hd = math.atan2(change[1] - sp.xy[1], change[0] - sp.xy[0])
        self.assertAlmostEqual(sp.heading, hd)


class FakeLegDrive:
    def __init__(self, leg):
        self.leg = leg
        self.st = HierState.DRIVING_SUBGOAL
        self.reason = None
        self.stopped = False

    def state(self):
        return self.st

    def fail_reason(self):
        return self.reason

    def stop(self):
        self.stopped = True
        self.st = HierState.IDLE


class Sim:
    """ExploreTask over two_rooms with a fake drive: the robot teleports to a
    leg's goal when the test says it arrived."""

    def __init__(self, **cfg):
        self.d = two_rooms()
        self.cfg = ExploreConfig(**cfg)
        self.em = ExploreMap(self.d, None, META, self.cfg)
        self.robot = (0.5, 0.5)
        self.seen = np.zeros_like(self.d, bool)
        self.occ = {}
        self.inv = []
        ctx = ExploreContext(
            costmap=lambda: costmap_of(self.d),
            robot_xy=lambda: self.robot,
            seen=lambda: self.seen,
            occupied_near=lambda xy, r: self.occ.get(xy),
            home_xy=(0.5, 0.5),
            on_investigated=lambda cid, o, det: self.inv.append((cid, o, det)),
        )
        self.task = ExploreTask(self.em, ctx, self.cfg)
        self.drives = []
        self.finished = None
        self.runner = TaskRunner(self.task, self._start,
                                 on_finished=lambda why, s: setattr(self, "finished", (why, s)))

    def _start(self, leg):
        dr = FakeLegDrive(leg)
        self.drives.append(dr)
        return dr

    def arrive(self, now):
        dr = self.drives[-1]
        self.robot = dr.leg.xy
        dr.st = HierState.ARRIVED
        self.runner.tick(now)
        return dr.leg


class TestExploreTask(unittest.TestCase):
    def test_visits_viewpoints_then_goes_home(self):
        s = Sim()
        s.runner.start(0.0)
        purposes = []
        t = 0.0
        while s.finished is None and t < 500:
            t += 1.0
            purposes.append(s.arrive(t).purpose)
        self.assertEqual(s.finished[0], "done")
        self.assertEqual(purposes[-1], "home")
        self.assertGreaterEqual(purposes.count("viewpoint"), 2)
        self.assertGreaterEqual(s.finished[1]["coverage"], 0.9)

    def test_budget_preempts_and_returns_home(self):
        s = Sim(budget_min=1.0)
        s.runner.start(0.0)
        first = s.drives[-1]
        self.assertEqual(first.leg.purpose, "viewpoint")
        s.runner.tick(61.0)
        self.assertTrue(first.stopped)
        self.assertEqual(s.drives[-1].leg.purpose, "home")
        self.assertIsNone(s.drives[-1].leg.give_up_after)
        s.arrive(62.0)
        self.assertEqual(s.finished[0], "done")
        self.assertIn("budget", s.finished[1]["end_reason"])

    def test_abandoned_viewpoint_is_not_retried(self):
        s = Sim()
        s.runner.start(0.0)
        dr = s.drives[-1]
        dr.st, dr.reason = HierState.FAILED, "abandoned"
        s.runner.tick(1.0)
        self.assertIn(dr.leg.ref, s.task.planner.abandoned)
        self.assertNotEqual(s.drives[-1].leg.ref, dr.leg.ref)

    def test_operator_stop_ends_the_task(self):
        s = Sim()
        s.runner.start(0.0)
        s.drives[-1].st = HierState.IDLE
        s.runner.tick(1.0)
        self.assertEqual(s.finished[0], "stopped")

    def _report(self, s, cid, xy, verdict="novel", kind="new", robot=(3.3, 1.5)):
        s.task.on_change({"type": "change", "change_id": cid, "kind": kind,
                          "status": "appeared", "x": xy[0], "y": xy[1], "verdict": verdict,
                          "robot": [robot[0], robot[1], 0.0]})

    def test_novel_new_is_investigated_and_confirmed_with_dwell(self):
        s = Sim()
        s.runner.start(0.0)
        self._report(s, 7, (4.5, 1.5))
        self._report(s, 8, (4.5, 2.0), verdict="candidate")     # learning: report only
        self._report(s, 9, (2.0, 2.0), kind="missing")          # missing: not investigated
        s.task.on_change({"type": "label", "change_id": 7, "label": "box or bag"})
        s.arrive(1.0)                                            # current viewpoint done
        leg = s.drives[-1].leg
        self.assertEqual((leg.purpose, leg.ref), ("investigate", 7))
        self.assertEqual(leg.dwell_s, s.cfg.investigate_dwell_s)
        s.occ[(4.5, 1.5)] = True
        s.arrive(2.0)
        self.assertIs(s.drives[-1].leg, leg)                     # dwelling
        s.runner.tick(2.0 + s.cfg.investigate_dwell_s + 0.1)
        self.assertEqual(s.inv, [(7, "confirmed", {"label": "box or bag"})])
        self.assertEqual(s.drives[-1].leg.purpose, "viewpoint")
        summ = s.task.summary()
        self.assertEqual(summ["investigations"], {"confirmed": 1})
        self.assertEqual(summ["new_changes"][7]["label"], "box or bag")

    def test_cleared_change_is_dismissed_and_gone_is_not_visited(self):
        s = Sim()
        s.runner.start(0.0)
        self._report(s, 1, (4.5, 1.5))
        self._report(s, 2, (4.5, 2.2))
        s.task.on_change({"type": "change", "change_id": 2, "kind": "new", "status": "gone",
                          "x": 4.5, "y": 2.2})
        s.arrive(1.0)
        s.occ[(4.5, 1.5)] = False
        s.arrive(2.0)
        s.runner.tick(10.0)
        outcomes = {cid: o for cid, o, _ in s.inv}
        self.assertEqual(outcomes, {2: "gone", 1: "dismissed"})
        self.assertEqual(sum(1 for d in s.drives if d.leg.purpose == "investigate"), 1)

    def test_investigation_cap(self):
        s = Sim(max_investigations=1)
        s.runner.start(0.0)
        self._report(s, 1, (4.5, 1.5))
        self._report(s, 2, (1.5, 1.5), robot=(0.5, 0.5))
        s.arrive(1.0)
        s.occ[(4.5, 1.5)] = s.occ[(1.5, 1.5)] = True
        s.arrive(2.0)
        s.runner.tick(10.0)
        self.assertEqual(sum(1 for d in s.drives if d.leg.purpose == "investigate"), 1)


if __name__ == "__main__":
    unittest.main()
