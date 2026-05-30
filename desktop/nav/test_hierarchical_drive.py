"""Tests for the Tier-1/Tier-2 hierarchical drive orchestrator (pure)."""
import math
import unittest

from desktop.nav.hierarchical_drive import HierarchicalDrive, HierConfig, HierState
from desktop.nav.patrol import Patrol, PatrolRunner, Waypoint

N = 360
ANGLE_MIN = -math.pi
ANGLE_INC = 2.0 * math.pi / N


def _clear_scan():
    # All no-return beams → scan_raster clears to the horizon (open space).
    return {"ranges": [0.0] * N, "angle_min": ANGLE_MIN, "angle_increment": ANGLE_INC}


def _runner(points, *, loop=False, laps=1):
    patrol = Patrol(name="t", session_id="s", authored_utc="", loop=loop, laps=laps,
                    waypoints=[Waypoint(x_m=x, y_m=y) for x, y in points])
    return PatrolRunner(patrol)


class FakeDriveIO:
    def __init__(self, scan=None):
        self.scan = scan
        self.status = None
        self.sent = []          # list of (bx, by, arrival_tol_m, v_max)
        self.cancels = 0
        self._cid = 0

    def latest_scan(self):
        return self.scan

    def latest_status(self):
        return self.status

    def send_goto_from_body(self, bx, by, *, arrival_tol_m=None, v_max=None):
        self._cid += 1
        self.sent.append((bx, by, arrival_tol_m, v_max))
        return self._cid

    def cancel(self):
        self.cancels += 1

    def set_status(self, cmd_id, state, blocked_reason=None):
        self.status = {"cmd_id": cmd_id, "state": state, "blocked_reason": blocked_reason}


class FakePose:
    def __init__(self, pose=None):
        self.pose = pose

    def world_pose(self):
        return self.pose


class TestHierarchicalDrive(unittest.TestCase):
    def _build(self, points=((2.0, 0.0),), pose=(0.0, 0.0, 0.0), scan=None, **cfg):
        io = FakeDriveIO(scan=scan if scan is not None else _clear_scan())
        po = FakePose(pose)
        hd = HierarchicalDrive(_runner(points), po, io, HierConfig(**cfg))
        return hd, io, po

    def _drive_to_sending(self, hd):
        hd.start()
        hd.tick(0.0)   # ALIGNING → SELECT
        hd.tick(0.0)   # SELECT → DRIVING (sends one goto)

    def test_idle_until_started(self):
        hd, io, _ = self._build()
        self.assertEqual(hd.tick(0.0), HierState.IDLE)
        self.assertEqual(io.sent, [])

    def test_start_aligns_selects_and_sends(self):
        hd, io, _ = self._build()
        hd.start()
        self.assertEqual(hd.state(), HierState.ALIGNING)
        self.assertEqual(hd.tick(0.0), HierState.SELECT_SUBGOAL)
        self.assertEqual(hd.tick(0.0), HierState.DRIVING_SUBGOAL)
        self.assertEqual(len(io.sent), 1)
        # Sub-goal aims straight ahead toward the waypoint, within the scan horizon.
        bx, by, tol, _ = io.sent[0]
        self.assertGreater(bx, 0.0)
        self.assertAlmostEqual(by, 0.0, places=6)
        self.assertIsNotNone(tol)

    def test_no_pose_stays_aligning(self):
        hd, io, _ = self._build(pose=None)
        hd.start()
        self.assertEqual(hd.tick(0.0), HierState.ALIGNING)
        self.assertEqual(io.sent, [])

    def test_no_waypoints_fails(self):
        io = FakeDriveIO(scan=_clear_scan())
        hd = HierarchicalDrive(_runner(()), FakePose((0, 0, 0)), io)
        hd.start()
        self.assertEqual(hd.state(), HierState.FAILED)

    def test_subgoal_arrived_repicks_same_waypoint(self):
        hd, io, _ = self._build(points=((5.0, 0.0),))
        self._drive_to_sending(hd)
        runner_idx_before = hd._runner.wp_index
        io.set_status(cmd_id=1, state="ARRIVED")
        self.assertEqual(hd.tick(0.0), HierState.SELECT_SUBGOAL)
        self.assertEqual(hd.tick(0.0), HierState.DRIVING_SUBGOAL)
        self.assertEqual(len(io.sent), 2)                    # re-picked
        self.assertEqual(hd._runner.wp_index, runner_idx_before)  # same waypoint

    def test_waypoint_reached_advances(self):
        hd, io, po = self._build(points=((1.0, 0.0), (3.0, 0.0)))
        self._drive_to_sending(hd)
        po.pose = (1.0, 0.0, 0.0)        # now sitting on wp0
        self.assertEqual(hd.tick(0.0), HierState.ADVANCE_WAYPOINT)
        self.assertGreaterEqual(io.cancels, 1)
        self.assertEqual(hd.tick(0.0), HierState.SELECT_SUBGOAL)   # toward wp1
        self.assertEqual(hd._runner.wp_index, 1)

    def test_last_waypoint_terminal_arrived(self):
        hd, io, po = self._build(points=((1.0, 0.0),))
        self._drive_to_sending(hd)
        po.pose = (1.0, 0.0, 0.0)
        self.assertEqual(hd.tick(0.0), HierState.ADVANCE_WAYPOINT)
        self.assertEqual(hd.tick(0.0), HierState.ARRIVED)
        self.assertGreaterEqual(io.cancels, 1)
        self.assertIsNone(hd.current_subgoal_body())

    def test_tier3_blocked_sets_blocked(self):
        hd, io, _ = self._build(points=((5.0, 0.0),))
        self._drive_to_sending(hd)
        io.set_status(cmd_id=1, state="BLOCKED", blocked_reason="swept_block")
        self.assertEqual(hd.tick(0.0), HierState.BLOCKED)
        self.assertEqual(hd.block_reason(), "swept_block")

    def test_tier3_canceled_fails(self):
        hd, io, _ = self._build(points=((5.0, 0.0),))
        self._drive_to_sending(hd)
        io.set_status(cmd_id=1, state="CANCELED")
        self.assertEqual(hd.tick(0.0), HierState.FAILED)

    def test_stale_status_ignored(self):
        hd, io, _ = self._build(points=((5.0, 0.0),))
        self._drive_to_sending(hd)
        io.set_status(cmd_id=99, state="ARRIVED")   # not our cmd_id
        self.assertEqual(hd.tick(0.0), HierState.DRIVING_SUBGOAL)

    def test_no_scan_blocks_then_pauses(self):
        # No scan ever → SELECT blocks; rotate_repick retries to the cap, then holds.
        io = FakeDriveIO(scan=None)
        hd = HierarchicalDrive(_runner(((5.0, 0.0),)), FakePose((0, 0, 0)),
                               io, HierConfig(max_blocked_repicks=2))
        hd.start()
        hd.tick(0.0)                  # ALIGNING → SELECT
        states = [hd.tick(0.0) for _ in range(8)]
        self.assertEqual(states[-1], HierState.BLOCKED)
        self.assertEqual(io.sent, [])
        self.assertEqual(hd.block_reason(), "no_scan")

    def test_bearing_hysteresis_repicks(self):
        hd, io, po = self._build(points=((5.0, 0.0),), repick_hysteresis_rad=0.2)
        self._drive_to_sending(hd)
        # Robot rotated a lot in place → bearing to the waypoint moved well past hysteresis.
        po.pose = (0.0, 0.0, 1.0)
        self.assertEqual(hd.tick(0.0), HierState.SELECT_SUBGOAL)

    def test_stop_cancels_and_idles(self):
        hd, io, _ = self._build()
        self._drive_to_sending(hd)
        hd.stop()
        self.assertEqual(hd.state(), HierState.IDLE)
        self.assertGreaterEqual(io.cancels, 1)


if __name__ == "__main__":
    unittest.main()
