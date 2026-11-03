"""Nav redesign step 3, desktop side: HierarchicalDrive streams the global
path to Tier-3 (goto kind "path") when the Pi advertises caps ["path"]."""
from __future__ import annotations

import math
import unittest

from desktop.nav.hierarchical_drive import HierConfig, HierState, HierarchicalDrive
from desktop.nav.test_hierarchical_drive import FakeDriveIO, FakePose, _clear_scan, _runner
from desktop.nav.test_route_planner import WALL_WITH_GAP, _world


class PathIO(FakeDriveIO):
    def __init__(self, caps=True, **kw):
        super().__init__(**kw)
        self.caps = caps
        self.paths = []        # dicts: cmd_id, new, path, anchor_ts, seq, route, end, ttl

    def supports_path(self):
        return self.caps

    def send_path(self, path_body, anchor_ts, *, cmd_id=None, path_seq=1, route_id=1,
                  end_is_goal=False, ttl_s=None, arrival_tol_m=None, v_max=None,
                  final_heading_rad=None):
        new = cmd_id is None
        if new:
            self._cid += 1
            cmd_id = self._cid
        self.paths.append(dict(cmd_id=cmd_id, new=new, path=list(path_body),
                               anchor_ts=anchor_ts, seq=path_seq, route=route_id,
                               end=end_is_goal, ttl=ttl_s))
        return cmd_id


class StampedPose(FakePose):
    def __init__(self, pose):
        super().__init__(pose)
        self.ts = 100.0

    def world_pose_stamped(self):
        return None if self.pose is None else (self.pose, self.ts)


def _drive(cm, points, caps=True, pose=(0.0, 0.0, 0.0), **cfg):
    io = PathIO(caps=caps, scan=_clear_scan())
    po = StampedPose(pose)
    hd = HierarchicalDrive(_runner(points), po, io, HierConfig(path_mode=True, **cfg),
                           costmap_provider=lambda: cm)
    hd.start()
    hd.tick(0.0)
    hd.tick(0.0)
    return hd, io, po


class TestPathMode(unittest.TestCase):
    def test_sends_the_route_as_a_path_goal(self):
        cm = _world(WALL_WITH_GAP)
        hd, io, _ = _drive(cm, ((2.0, 0.0),))
        self.assertEqual(hd.state(), HierState.DRIVING_SUBGOAL, hd.block_reason())
        self.assertEqual(io.sent, [])                     # no sub-goal goto
        p = io.paths[0]
        self.assertTrue(p["new"])
        self.assertEqual(p["anchor_ts"], 100.0)
        self.assertEqual(p["path"][0], (0.0, 0.0))        # robot-first, body frame
        self.assertGreater(max(y for _x, y in p["path"]), 0.4)   # through the gap
        self.assertTrue(p["end"])                         # terminal, not truncated
        self.assertEqual(p["ttl"], 1.0)
        seg = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(p["path"], p["path"][1:])]
        self.assertLessEqual(max(seg), 0.10 + 1e-6)

    def test_falls_back_to_subgoals_without_caps(self):
        cm = _world()
        hd, io, _ = _drive(cm, ((2.0, 0.0),), caps=False)
        self.assertEqual(io.paths, [])
        self.assertEqual(len(io.sent), 1)

    def test_streams_refreshes_on_one_goal(self):
        cm = _world()
        hd, io, po = _drive(cm, ((2.5, 0.0),))
        cid = io.paths[0]["cmd_id"]
        for k in range(1, 11):                            # 1 s at 10 Hz, robot moving
            po.pose = (0.018 * k, 0.0, 0.0)
            po.ts = 100.0 + 0.1 * k
            io.set_status(cmd_id=cid, state="DRIVING", mode="follow", path_age_s=0.1)
            hd.tick(0.1 * k)
        refreshes = io.paths[1:]
        self.assertGreaterEqual(len(refreshes), 3)
        self.assertLessEqual(len(refreshes), 5)           # ~4 Hz
        self.assertTrue(all(not p["new"] and p["cmd_id"] == cid for p in refreshes))
        seqs = [p["seq"] for p in io.paths]
        self.assertEqual(seqs, sorted(set(seqs)))         # strictly increasing
        self.assertTrue(all(p["route"] == io.paths[0]["route"] for p in refreshes))
        last = refreshes[-1]
        self.assertAlmostEqual(last["path"][1][0], 0.10, delta=0.02)   # re-anchored
        self.assertEqual(io.sent, [])
        self.assertEqual(io.cancels, 0)

    def test_via_advance_is_a_new_route_on_the_same_goal(self):
        cm = _world()
        hd, io, po = _drive(cm, ((1.0, 0.0), (1.0, 1.5)))
        cid = io.paths[0]["cmd_id"]
        self.assertFalse(io.paths[0]["end"])              # an intermediate via
        po.pose = (1.05, 0.0, 0.0)                        # drove past the via
        io.set_status(cmd_id=cid, state="DRIVING", mode="follow")
        with self.assertLogs("desktop.nav.hierarchical_drive", "INFO") as lg:
            hd.tick(0.5)
        self.assertEqual(hd.current_waypoint_world(), (1.0, 1.5))
        p = io.paths[-1]
        self.assertFalse(p["new"])
        self.assertEqual(p["cmd_id"], cid)
        self.assertEqual(p["route"], io.paths[0]["route"] + 1)
        self.assertTrue(p["end"])                         # now the terminal
        self.assertTrue(any("path stream (to waypoint" in m for m in lg.output))
        self.assertEqual(io.cancels, 0)

    def test_path_blocked_stamps_the_point_and_retries_with_a_new_goal(self):
        cm = _world()
        hd, io, po = _drive(cm, ((2.5, 0.0),))
        cid = io.paths[0]["cmd_id"]
        io.set_status(cmd_id=cid, state="BLOCKED", blocked_reason="path_blocked",
                      block_body_xy=[1.0, 0.0])
        hd.tick(0.3)
        self.assertEqual(hd.state(), HierState.BLOCKED)
        self.assertEqual(io.cancels, 1)
        hd.tick(1.0)                                       # retry
        p = io.paths[-1]
        self.assertTrue(p["new"])
        self.assertNotEqual(p["cmd_id"], cid)
        near = [pt for pt in p["path"] if abs(pt[0] - 1.0) < 0.15]
        self.assertTrue(near)
        self.assertGreater(min(abs(y) for _x, y in near), 0.15)   # goes round the point

    def test_stale_stop_is_counted_and_logged(self):
        cm = _world()
        hd, io, _ = _drive(cm, ((2.5, 0.0),))
        cid = io.paths[0]["cmd_id"]
        io.set_status(cmd_id=cid, state="DRIVING", mode="starved", path_age_s=1.3)
        hd.tick(0.1)
        io.set_status(cmd_id=cid, state="BLOCKED", blocked_reason="path_stale",
                      path_age_s=2.1)
        with self.assertLogs("desktop.nav.hierarchical_drive", "INFO") as lg:
            hd.tick(0.2)
            hd.stop()
        text = "\n".join(lg.output)
        self.assertIn("path stale", text)
        self.assertIn("starved=1 stale=1", text)
        self.assertIn("max_path_age=2.10s", text)

    def test_pose_correction_triggers_a_replan(self):
        cm = _world()
        hd, io, po = _drive(cm, ((2.5, 0.0),))
        cid = io.paths[0]["cmd_id"]
        io.set_status(cmd_id=cid, state="DRIVING", mode="follow")
        hd.tick(0.1)
        n = hd._pstats["replans"]
        po.corr_seq += 1
        hd.tick(0.2)
        self.assertEqual(hd._pstats["replans"], n + 1)
        self.assertEqual(hd.state(), HierState.DRIVING_SUBGOAL)

    def test_terminal_arrival_finishes(self):
        cm = _world()
        hd, io, po = _drive(cm, ((1.0, 0.0),))
        cid = io.paths[0]["cmd_id"]
        po.pose = (0.95, 0.0, 0.0)
        io.set_status(cmd_id=cid, state="ARRIVED")
        hd.tick(0.5)
        hd.tick(0.5)
        self.assertEqual(hd.state(), HierState.ARRIVED)


if __name__ == "__main__":
    unittest.main()
