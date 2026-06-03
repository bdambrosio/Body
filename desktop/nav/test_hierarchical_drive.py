"""Tests for the Tier-1/Tier-2 hierarchical drive orchestrator (pure)."""
import math
import unittest

from desktop.nav.hierarchical_drive import (
    HierarchicalDrive, HierConfig, HierState, PFPoseProvider,
)
from desktop.nav.patrol import Patrol, PatrolRunner, Waypoint


class _FakePoseSource:
    """Minimal pose_source: a fixed pose + a settable skew-immune odom age."""
    def __init__(self, pose=(1.0, 2.0, 0.5), age_s=0.0):
        self._pose = pose
        self._age = age_s

    def latest_pose(self):
        return (self._pose, 12345.0)   # ts deliberately bogus — must be ignored

    def odom_age_s(self):
        return self._age


class _FakeFuser:
    def __init__(self, src):
        self.pose_source = src


class TestPFPoseProvider(unittest.TestCase):
    def test_fresh_age_returns_pose(self):
        src = _FakePoseSource(age_s=0.1)
        p = PFPoseProvider(_FakeFuser(src), max_pose_age_s=0.75)
        self.assertEqual(p.world_pose(), (1.0, 2.0, 0.5))

    def test_stale_age_returns_none(self):
        src = _FakePoseSource(age_s=1.5)
        p = PFPoseProvider(_FakeFuser(src), max_pose_age_s=0.75)
        self.assertIsNone(p.world_pose())

    def test_ignores_pi_timestamp(self):
        # Bogus Pi ts (12345.0, decades stale) must NOT make a fresh pose look
        # stale — staleness is judged by odom_age_s only (the skew-immune fix).
        src = _FakePoseSource(age_s=0.0)
        p = PFPoseProvider(_FakeFuser(src), max_pose_age_s=0.75)
        self.assertEqual(p.world_pose(), (1.0, 2.0, 0.5))

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


class FakeSink:
    """In-memory HandoffSink for breakpoint tests."""
    def __init__(self):
        self.records = []           # list of (tier, payload)
        self._armed = set()
        self._continue = set()

    def record(self, tier, payload):
        self.records.append((tier, payload))

    def is_armed(self, tier):
        return tier in self._armed

    def should_hold(self, tier):
        return tier in self._armed and tier not in self._continue

    def consume_continue(self, tier):
        if tier in self._continue:
            self._continue.discard(tier)
            return True
        return False

    # test controls
    def arm(self, tier):
        self._armed.add(tier)

    def cont(self, tier):
        self._continue.add(tier)

    def recorded_tiers(self):
        return [t for t, _ in self.records]


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
        po.pose = (1.0, 0.0, 0.0)        # now sitting on wp0 (intermediate)
        self.assertEqual(hd.tick(0.0), HierState.ADVANCE_WAYPOINT)
        # Intermediate advance is seamless — Tier-3 is NOT canceled; the next
        # SELECT_SUBGOAL goto supersedes it.
        self.assertEqual(io.cancels, 0)
        self.assertEqual(hd.tick(0.0), HierState.SELECT_SUBGOAL)   # toward wp1
        self.assertEqual(hd._runner.wp_index, 1)

    def test_passthrough_advances_intermediate_without_stop(self):
        # wp0 is non-terminal: a pose inside passthrough_tol (0.6) but outside
        # the tight waypoint_tol (0.3) should advance (and not cancel).
        hd, io, po = self._build(points=((1.0, 0.0), (3.0, 0.0)))
        self._drive_to_sending(hd)
        po.pose = (0.55, 0.0, 0.0)       # 0.45 m from wp0
        self.assertEqual(hd.tick(0.0), HierState.ADVANCE_WAYPOINT)
        self.assertEqual(io.cancels, 0)
        self.assertEqual(hd.tick(0.0), HierState.SELECT_SUBGOAL)
        self.assertEqual(hd._runner.wp_index, 1)

    def test_terminal_does_not_passthrough(self):
        # Single (terminal) waypoint: at 0.45 m — inside passthrough but outside
        # the tight tol — it must NOT advance yet (terminal uses the tight tol).
        hd, io, po = self._build(points=((1.0, 0.0),))
        self._drive_to_sending(hd)
        po.pose = (0.55, 0.0, 0.0)       # 0.45 m from the only (terminal) wp
        self.assertEqual(hd.tick(0.0), HierState.DRIVING_SUBGOAL)
        self.assertEqual(io.cancels, 0)

    def test_last_waypoint_terminal_arrived(self):
        hd, io, po = self._build(points=((1.0, 0.0),))
        self._drive_to_sending(hd)
        po.pose = (1.0, 0.0, 0.0)
        self.assertEqual(hd.tick(0.0), HierState.ADVANCE_WAYPOINT)
        self.assertEqual(hd.tick(0.0), HierState.ARRIVED)
        self.assertGreaterEqual(io.cancels, 1)
        self.assertIsNone(hd.current_subgoal_body())

    def test_subgoal_idle_repicks_same_waypoint(self):
        # Tier-3 publishes ARRIVED for one tick then reverts to IDLE; at our
        # slower poll we usually catch the IDLE. It must still re-pick.
        hd, io, _ = self._build(points=((5.0, 0.0),))
        self._drive_to_sending(hd)
        io.set_status(cmd_id=1, state="IDLE")
        self.assertEqual(hd.tick(0.0), HierState.SELECT_SUBGOAL)
        self.assertEqual(hd.tick(0.0), HierState.DRIVING_SUBGOAL)
        self.assertEqual(len(io.sent), 2)
        self.assertEqual(hd._runner.wp_index, 0)

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

    def test_select_no_scan_falls_back_to_blind_projection(self):
        # Tier-2 prefers the clear-run over the live scan, but with NO
        # rasterizable scan it falls back to a blind horizon projection so it
        # still sends + drives (no_scan is then Tier-3's BLOCKED to report).
        io = FakeDriveIO(scan=None)
        hd = HierarchicalDrive(_runner(((5.0, 0.0),)), FakePose((0.0, 0.0, 0.0)), io)
        hd.start()
        self.assertEqual(hd.tick(0.0), HierState.SELECT_SUBGOAL)
        self.assertEqual(hd.tick(0.0), HierState.DRIVING_SUBGOAL)
        self.assertEqual(len(io.sent), 1)
        # Fallback sub-goal is the waypoint direction clamped to the horizon
        # (1.5 m), not the 5 m waypoint — and NOT backed off (no scan).
        bx, by, _tol, _v = io.sent[0]
        self.assertAlmostEqual(bx, 1.5, places=6)
        self.assertAlmostEqual(by, 0.0, places=6)

    def test_select_uses_clear_run_when_scan_present(self):
        # With a live scan, Tier-2 marches the bearing and hands Tier-3 the
        # furthest CLEAR point backed off the horizon — NOT the blind 1.5 m
        # projection. An all-no-return scan is clear everywhere, so the sub-goal
        # is horizon - backoff along +x.
        n = 360
        scan = {
            "ranges": [10.0] * n,          # all beyond range_max → no-return → clear
            "angle_min": -math.pi,
            "angle_increment": 2.0 * math.pi / n,
            "ts": 1.0,
        }
        io = FakeDriveIO(scan=scan)
        hd = HierarchicalDrive(_runner(((5.0, 0.0),)), FakePose((0.0, 0.0, 0.0)), io)
        hd.start()
        self.assertEqual(hd.tick(0.0), HierState.SELECT_SUBGOAL)
        self.assertEqual(hd.tick(0.0), HierState.DRIVING_SUBGOAL)
        self.assertEqual(len(io.sent), 1)
        bx, by, _tol, _v = io.sent[0]
        # clear-run reached the 1.5 m horizon then backed off 0.15 m → ~1.35 m,
        # strictly short of the blind 1.5 m. Proves the scan path is wired.
        self.assertLess(bx, 1.5)
        self.assertAlmostEqual(bx, 1.35, places=2)
        self.assertAlmostEqual(by, 0.0, places=2)

    # ── handoff breakpoints (HO-1 / HO-2) ────────────────────────────
    def _build_sink(self, sink, points=((2.0, 0.0),)):
        io = FakeDriveIO(scan=_clear_scan())
        hd = HierarchicalDrive(_runner(points), FakePose((0.0, 0.0, 0.0)), io,
                               HierConfig(), sink=sink)
        hd.start()
        hd.tick(0.0)   # ALIGNING → SELECT
        return hd, io

    def test_records_emitted_without_arming(self):
        # With nothing armed, both handoffs record and the drive proceeds.
        sink = FakeSink()
        hd, io = self._build_sink(sink)
        self.assertEqual(hd.tick(0.0), HierState.DRIVING_SUBGOAL)
        self.assertEqual(sink.recorded_tiers(), [1, 2])
        self.assertEqual(len(io.sent), 1)

    def test_breakpoint_t2_holds_then_single_steps(self):
        sink = FakeSink()
        sink.arm(2)
        hd, io = self._build_sink(sink)
        # Armed at HO-2: holds in SELECT, cancels the bot, sends nothing.
        self.assertEqual(hd.tick(0.0), HierState.SELECT_SUBGOAL)
        self.assertEqual(io.sent, [])
        self.assertGreaterEqual(io.cancels, 1)
        self.assertIn(2, sink.recorded_tiers())
        # Re-holding does NOT re-cancel each tick.
        cancels_after_first = io.cancels
        hd.tick(0.0)
        self.assertEqual(io.cancels, cancels_after_first)
        self.assertEqual(io.sent, [])
        # Continue → single-step past HO-2, goto goes out.
        sink.cont(2)
        self.assertEqual(hd.tick(0.0), HierState.DRIVING_SUBGOAL)
        self.assertEqual(len(io.sent), 1)

    def test_breakpoint_t1_holds_before_subgoal_selection(self):
        sink = FakeSink()
        sink.arm(1)
        hd, io = self._build_sink(sink)
        self.assertEqual(hd.tick(0.0), HierState.SELECT_SUBGOAL)
        self.assertEqual(io.sent, [])
        # Held at HO-1 → HO-2 not reached, so no tier-2 record yet.
        self.assertEqual(sink.recorded_tiers(), [1])
        sink.cont(1)
        self.assertEqual(hd.tick(0.0), HierState.DRIVING_SUBGOAL)
        self.assertIn(2, sink.recorded_tiers())
        self.assertEqual(len(io.sent), 1)

    def test_t2_record_attaches_grid_only_when_armed(self):
        # Not armed → lean record (no grid). Armed → grid+meta attached.
        sink = FakeSink()
        hd, io = self._build_sink(sink)
        hd.tick(0.0)
        t2 = [p for t, p in sink.records if t == 2][0]
        self.assertNotIn("grid", t2)

        sink2 = FakeSink()
        sink2.arm(2)
        hd2, _ = self._build_sink(sink2)
        hd2.tick(0.0)
        t2b = [p for t, p in sink2.records if t == 2][0]
        self.assertIn("grid", t2b)
        self.assertIn("meta", t2b)

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

    def test_pose_loss_while_driving_suspends_and_cancels(self):
        # Lose the pose mid-drive → SUSPENDED, Tier-3 canceled (so the Pi stops
        # chasing the old goal), and it does NOT auto-resume when pose returns.
        hd, io, po = self._build(points=((5.0, 0.0),))
        self._drive_to_sending(hd)
        self.assertEqual(hd.state(), HierState.DRIVING_SUBGOAL)
        cancels_before = io.cancels
        po.pose = None                       # connectivity drop
        self.assertEqual(hd.tick(0.0), HierState.SUSPENDED)
        self.assertEqual(io.cancels, cancels_before + 1)
        self.assertEqual(hd.block_reason(), "pose_lost")
        # Pose comes back — must STAY suspended (no auto-lurch).
        po.pose = (0.5, 0.0, 0.0)
        sent_before = len(io.sent)
        self.assertEqual(hd.tick(0.0), HierState.SUSPENDED)
        self.assertEqual(len(io.sent), sent_before)

    def test_resume_from_suspended_drives_again(self):
        hd, io, po = self._build(points=((5.0, 0.0),))
        self._drive_to_sending(hd)
        po.pose = None
        hd.tick(0.0)                         # → SUSPENDED
        po.pose = (0.5, 0.0, 0.0)            # link restored
        self.assertTrue(hd.request_resume())
        self.assertEqual(hd.state(), HierState.ALIGNING)
        self.assertEqual(hd.tick(0.0), HierState.SELECT_SUBGOAL)
        self.assertEqual(hd.tick(0.0), HierState.DRIVING_SUBGOAL)   # drives again

    def test_resume_noop_when_not_suspended(self):
        hd, io, _ = self._build()
        self._drive_to_sending(hd)
        self.assertEqual(hd.state(), HierState.DRIVING_SUBGOAL)
        self.assertFalse(hd.request_resume())
        self.assertEqual(hd.state(), HierState.DRIVING_SUBGOAL)

    def test_select_pose_loss_suspends(self):
        # Pose lost at the SELECT step (between sub-goals) also suspends.
        hd, io, po = self._build(points=((5.0, 0.0),))
        self._drive_to_sending(hd)
        io.set_status(cmd_id=1, state="ARRIVED")
        self.assertEqual(hd.tick(0.0), HierState.SELECT_SUBGOAL)
        po.pose = None
        self.assertEqual(hd.tick(0.0), HierState.SUSPENDED)


if __name__ == "__main__":
    unittest.main()
