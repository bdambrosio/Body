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

    def test_correction_seq_counts_discrete_not_every_scan(self):
        # n_applied increments on every 10 Hz scan observation; re-picking on
        # it would defeat the bearing hysteresis. The provider must read the
        # discrete-jump count instead.
        src = _FakePoseSource()
        src.correction_summary = lambda: {"n_applied": 570, "n_discrete": 2}
        p = PFPoseProvider(_FakeFuser(src))
        self.assertEqual(p.correction_seq(), 2)

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
        self.corr_seq = 0          # bump to simulate a re-anchor/relocate snap

    def world_pose(self):
        return self.pose

    def correction_seq(self):
        return self.corr_seq


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


class TestPassedWaypoint(unittest.TestCase):
    def test_not_passed_when_short(self):
        from desktop.nav.patrol import passed_waypoint
        self.assertFalse(passed_waypoint((0.55, 0.0), (0.0, 0.0), (1.0, 0.0)))

    def test_passed_when_beyond(self):
        from desktop.nav.patrol import passed_waypoint
        self.assertTrue(passed_waypoint((1.1, 0.0), (0.0, 0.0), (1.0, 0.0)))

    def test_passed_abeam_even_if_offset(self):
        from desktop.nav.patrol import passed_waypoint
        # Abeam the vertex (t = 1) but laterally offset still counts as passed.
        self.assertTrue(passed_waypoint((1.0, 0.5), (0.0, 0.0), (1.0, 0.0)))

    def test_proximity_fallback(self):
        from desktop.nav.patrol import passed_waypoint
        # Short along the segment but within the small proximity guard.
        self.assertTrue(passed_waypoint((0.0, 0.85), (0.0, 0.0), (0.0, 1.0),
                                        proximity_m=0.2))
        self.assertFalse(passed_waypoint((0.0, 0.7), (0.0, 0.0), (0.0, 1.0),
                                         proximity_m=0.2))


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
        # Same-tick handoff: ARRIVED → re-pick → next goto all in one tick.
        self.assertEqual(hd.tick(0.0), HierState.DRIVING_SUBGOAL)
        self.assertEqual(len(io.sent), 2)                    # re-picked
        self.assertEqual(hd._runner.wp_index, runner_idx_before)  # same waypoint

    def test_waypoint_reached_advances(self):
        hd, io, po = self._build(points=((1.0, 0.0), (3.0, 0.0)))
        self._drive_to_sending(hd)
        po.pose = (1.0, 0.0, 0.0)        # now sitting on wp0 (intermediate)
        # Same-tick: reached wp0 → ADVANCE → SELECT toward wp1 → DRIVING.
        self.assertEqual(hd.tick(0.0), HierState.DRIVING_SUBGOAL)
        # Intermediate advance is seamless — Tier-3 is NOT canceled; the next
        # SELECT_SUBGOAL goto supersedes it.
        self.assertEqual(io.cancels, 0)
        self.assertEqual(hd._runner.wp_index, 1)   # advanced toward wp1

    def test_intermediate_advances_only_when_passed(self):
        # Non-terminal wp0: a pose SHORT of it (not yet driven past along the
        # start→wp0 segment) must NOT advance — the old proximity radius would
        # have skipped it (and cut the corner). It advances only once passed.
        hd, io, po = self._build(points=((1.0, 0.0), (3.0, 0.0)))
        self._drive_to_sending(hd)       # _route_start captured at (0,0)
        po.pose = (0.55, 0.0, 0.0)       # 0.45 m short of wp0 → not passed
        self.assertEqual(hd.tick(0.0), HierState.DRIVING_SUBGOAL)
        self.assertEqual(hd._runner.wp_index, 0)
        po.pose = (1.1, 0.0, 0.0)        # driven past wp0 (t ≥ 1) → advance
        # Same-tick: passed wp0 → ADVANCE → SELECT toward wp1 → DRIVING.
        self.assertEqual(hd.tick(0.0), HierState.DRIVING_SUBGOAL)
        self.assertEqual(io.cancels, 0)  # seamless — Tier-3 not canceled
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
        # Same-tick: reached terminal wp → ADVANCE → ARRIVED.
        self.assertEqual(hd.tick(0.0), HierState.ARRIVED)
        self.assertGreaterEqual(io.cancels, 1)
        self.assertIsNone(hd.current_subgoal_body())

    def test_subgoal_idle_repicks_same_waypoint(self):
        # Tier-3 publishes ARRIVED for one tick then reverts to IDLE; at our
        # slower poll we usually catch the IDLE. It must still re-pick.
        hd, io, _ = self._build(points=((5.0, 0.0),))
        self._drive_to_sending(hd)
        io.set_status(cmd_id=1, state="IDLE")
        # Same-tick handoff: IDLE → re-pick → next goto all in one tick.
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
        # All-clear scan → no inflation; clear-run reaches the 1.5 m horizon then
        # backs off a ½ cell (0.04 m) → ~1.46 m, short of the blind 1.5 m. Proves
        # the scan path is wired (and the new ½-cell backoff is in effect).
        self.assertLess(bx, 1.5)
        self.assertAlmostEqual(bx, 1.46, places=2)
        self.assertAlmostEqual(by, 0.0, places=2)

    def test_lead_in_followed_then_patrol(self):
        # lead_in [start, (1,0), marker0] is followed vertex-by-vertex BEFORE
        # the patrol runner engages; the runner stays at wp_index 0 until the
        # lead-in delivers us to the first marker.
        io = FakeDriveIO(scan=_clear_scan())
        po = FakePose((0.0, 0.0, 0.0))
        runner = _runner(((2.0, 0.0), (4.0, 0.0)))
        hd = HierarchicalDrive(runner, po, io, HierConfig(),
                               lead_in=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])
        hd.start()
        self.assertEqual(hd.tick(0.0), HierState.SELECT_SUBGOAL)    # ALIGNING→SELECT
        self.assertEqual(hd.tick(0.0), HierState.DRIVING_SUBGOAL)   # carrot=(1,0)
        self.assertTrue(hd._in_lead_in())
        self.assertEqual(hd._waypoint, (1.0, 0.0))     # first lead-in carrot
        self.assertEqual(runner.wp_index, 0)           # runner untouched

        po.pose = (1.1, 0.0, 0.0)                       # passed (1,0)
        # Same-tick: ADVANCE (step lead-in) → SELECT → DRIVING toward marker0.
        self.assertEqual(hd.tick(0.0), HierState.DRIVING_SUBGOAL)
        self.assertTrue(hd._in_lead_in())
        self.assertEqual(hd._waypoint, (2.0, 0.0))
        self.assertEqual(runner.wp_index, 0)

        po.pose = (2.1, 0.0, 0.0)                       # passed marker0
        # Same-tick: lead-in ends → patrol engages (on_arrived 0→1) → DRIVING
        # toward marker1, all collapsed into one tick.
        self.assertEqual(hd.tick(0.0), HierState.DRIVING_SUBGOAL)
        self.assertFalse(hd._in_lead_in())
        self.assertEqual(runner.wp_index, 1)
        self.assertEqual(hd._waypoint, (4.0, 0.0))

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

    def test_t2_record_always_attaches_grid_when_scan_present(self):
        # The local map is attached every select (so the inspector's map isn't
        # intermittent), independent of whether BP2 is armed.
        for armed in (False, True):
            sink = FakeSink()
            if armed:
                sink.arm(2)
            hd, _ = self._build_sink(sink)
            hd.tick(0.0)
            t2 = [p for t, p in sink.records if t == 2][0]
            self.assertIn("grid", t2)
            self.assertIn("meta", t2)

    def test_bearing_hysteresis_repicks(self):
        hd, io, po = self._build(points=((5.0, 0.0),), repick_hysteresis_rad=0.2)
        self._drive_to_sending(hd)
        # Robot rotated a lot in place → bearing to the waypoint moved well past hysteresis.
        po.pose = (0.0, 0.0, 1.0)
        # Same-tick: drift past hysteresis → re-pick → new goto, settles in DRIVING.
        self.assertEqual(hd.tick(0.0), HierState.DRIVING_SUBGOAL)
        self.assertEqual(len(io.sent), 2)

    def test_pose_correction_repicks_without_drift(self):
        # A re-anchor snap moves the world pose without odom seeing it — the
        # provider's correction_seq advances while the bearing hasn't drifted
        # past the hysteresis gate. The driver must still re-pick at once (from
        # the corrected pose), not wait for ARRIVED.
        hd, io, po = self._build(points=((5.0, 0.0),))
        self._drive_to_sending(hd)
        self.assertEqual(len(io.sent), 1)
        po.corr_seq += 1                 # re-anchor snap; pose otherwise unchanged
        self.assertEqual(hd.tick(0.0), HierState.DRIVING_SUBGOAL)  # same-tick re-pick
        self.assertEqual(len(io.sent), 2)

    def test_no_repick_without_correction_or_drift(self):
        # Steady driving with no correction and no drift must NOT re-pick.
        hd, io, po = self._build(points=((5.0, 0.0),))
        self._drive_to_sending(hd)
        self.assertEqual(hd.tick(0.0), HierState.DRIVING_SUBGOAL)
        self.assertEqual(len(io.sent), 1)

    def test_no_spurious_repick_across_pi_bearing(self):
        # Carrot dead-astern: both bearings live near ±π. A tiny real drift
        # that crosses the seam must not read as ~2π and re-pick every tick.
        hd, io, po = self._build(points=((-5.0, 0.0),), pose=(0.0, 0.0, 0.0))
        self._drive_to_sending(hd)
        self.assertEqual(len(io.sent), 1)
        po.pose = (0.0, 0.01, 0.0)       # bearing flips sign across ±π
        self.assertEqual(hd.tick(0.0), HierState.DRIVING_SUBGOAL)
        self.assertEqual(len(io.sent), 1)   # no re-pick

    def test_blocked_retry_is_time_based(self):
        hd, io, po = self._build(points=((5.0, 0.0),),
                                 blocked_retry_interval_s=0.5,
                                 blocked_retry_window_s=2.0)
        self._drive_to_sending(hd)                          # goto #1
        io.set_status(cmd_id=1, state="BLOCKED", blocked_reason="swept_block")
        self.assertEqual(hd.tick(1.0), HierState.BLOCKED)   # block run starts t=1
        # Inside the retry interval: hold, no goto churn (the old count-based
        # cap burned all its retries in a fraction of a second at 20 Hz).
        self.assertEqual(hd.tick(1.2), HierState.BLOCKED)
        self.assertEqual(len(io.sent), 1)
        # Interval elapsed: re-pick fires a fresh goto.
        self.assertEqual(hd.tick(1.6), HierState.DRIVING_SUBGOAL)
        self.assertEqual(len(io.sent), 2)

    def test_blocked_window_exhausts_then_operator_resume(self):
        hd, io, po = self._build(points=((5.0, 0.0),),
                                 blocked_retry_interval_s=0.5,
                                 blocked_retry_window_s=2.0)
        self._drive_to_sending(hd)
        io.set_status(cmd_id=1, state="BLOCKED", blocked_reason="swept_block")
        self.assertEqual(hd.tick(1.0), HierState.BLOCKED)
        self.assertEqual(hd.tick(1.6), HierState.DRIVING_SUBGOAL)   # retry
        io.set_status(cmd_id=2, state="BLOCKED", blocked_reason="swept_block")
        self.assertEqual(hd.tick(3.1), HierState.BLOCKED)   # past window start+2.0
        sent = len(io.sent)
        self.assertEqual(hd.tick(3.6), HierState.BLOCKED)   # paused — no retries
        self.assertEqual(hd.tick(9.9), HierState.BLOCKED)
        self.assertEqual(len(io.sent), sent)
        self.assertTrue(hd.can_resume())
        # Operator resume restarts the window and drives again.
        self.assertTrue(hd.request_resume())
        self.assertEqual(hd.state(), HierState.ALIGNING)
        self.assertEqual(hd.tick(10.0), HierState.SELECT_SUBGOAL)
        self.assertEqual(hd.tick(10.1), HierState.DRIVING_SUBGOAL)
        self.assertEqual(len(io.sent), sent + 1)

    def test_send_failed_cancels_live_goto(self):
        # A send failure enters BLOCKED — any still-live goto on the Pi must be
        # revoked so the UI's BLOCKED matches an actually-stopped robot.
        hd, io, po = self._build(points=((5.0, 0.0),))
        self._drive_to_sending(hd)
        cancels = io.cancels
        io.set_status(cmd_id=1, state="ARRIVED")            # forces a re-pick
        io.send_goto_from_body = lambda *a, **k: None       # next send fails
        hd.tick(0.0)
        self.assertEqual(hd.state(), HierState.BLOCKED)
        self.assertEqual(hd.block_reason(), "send_failed")
        self.assertEqual(io.cancels, cancels + 1)

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
        # Pose lost when a sub-goal completes (the re-pick step) must suspend
        # rather than send a goto from a stale pose. With the same-tick handoff
        # the completion and re-pick share a tick, so losing the pose first
        # means that tick suspends and emits no new goto.
        hd, io, po = self._build(points=((5.0, 0.0),))
        self._drive_to_sending(hd)
        sent_before = len(io.sent)
        io.set_status(cmd_id=1, state="ARRIVED")
        po.pose = None
        self.assertEqual(hd.tick(0.0), HierState.SUSPENDED)
        self.assertEqual(len(io.sent), sent_before)   # no goto from a lost pose


if __name__ == "__main__":
    unittest.main()
