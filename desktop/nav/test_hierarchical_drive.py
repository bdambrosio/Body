"""Tests for the Tier-1/Tier-2 hierarchical drive orchestrator (pure)."""
import math
import unittest

from desktop.nav.hierarchical_drive import (
    BlockAdvisory, HierarchicalDrive, HierConfig, HierState, PFPoseProvider,
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
        self.anchor_ts = []     # anchor_ts per send (None = latest-odom anchoring)
        self.cancels = 0
        self._cid = 0
        self.failed_acks = set()

    def latest_scan(self):
        return self.scan

    def latest_status(self):
        return self.status

    def send_goto_from_body(self, bx, by, *, arrival_tol_m=None, v_max=None,
                            anchor_ts=None):
        self._cid += 1
        self.sent.append((bx, by, arrival_tol_m, v_max))
        self.anchor_ts.append(anchor_ts)
        return self._cid

    def cancel(self):
        self.cancels += 1

    def ack_failed(self, cmd_id):
        return cmd_id in self.failed_acks

    def set_status(self, cmd_id, state, blocked_reason=None, **extra):
        self.status = {"cmd_id": cmd_id, "state": state,
                       "blocked_reason": blocked_reason, **extra}


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

    def test_tier2_marches_on_pi_config(self):
        # The Tier-2 clear-run must use Tier-3's EXACT costmap model. Pin that
        # the drive builds it from config.json via the shared builders — not
        # from parallel dataclass defaults that drift the moment someone tunes
        # the Pi (the corner-stall class of bug the I3 redesign removed).
        from body.lib import drive_config, zenoh_helpers
        body_cfg = zenoh_helpers.load_body_config()
        plan = drive_config.local_plan_config(body_cfg)
        hd, _, _ = self._build()
        self.assertEqual(hd._costmap_cfg, plan.costmap)
        self.assertEqual(hd._goal_clearance_cells, plan.goal_clearance_cells)
        self.assertEqual(hd._raster, drive_config.scan_raster_config(body_cfg))

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

    def test_select_clear_run_fail_blocks_not_blind(self):
        # When a live scan is present but the clear-run finds no usable point,
        # SELECT must BLOCKED (clear_run_failed) — never blind-project.
        io = FakeDriveIO(scan=_clear_scan())
        hd = HierarchicalDrive(_runner(((5.0, 0.0),)), FakePose((0.0, 0.0, 0.0)), io)
        hd.start()
        self.assertEqual(hd.tick(0.0), HierState.SELECT_SUBGOAL)
        hd._select_subgoal_body = lambda *a, **k: None
        self.assertEqual(hd.tick(0.0), HierState.BLOCKED)
        self.assertEqual(hd.block_reason(), "clear_run_failed")
        self.assertEqual(len(io.sent), 0)
        self.assertGreaterEqual(io.cancels, 1)

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
        # Robot slid sideways off the line → the WORLD bearing to the waypoint
        # moved past hysteresis (atan2(2, 5) ≈ 0.38 rad).
        po.pose = (0.0, -2.0, 0.0)
        # Same-tick: drift past hysteresis → re-pick → new goto, settles in DRIVING.
        self.assertEqual(hd.tick(0.0), HierState.DRIVING_SUBGOAL)
        self.assertEqual(len(io.sent), 2)

    def test_rotation_in_place_is_not_drift(self):
        # X1: Tier-3 rotating in place toward the sub-goal changes the body-
        # relative bearing but not the world bearing — must NOT re-pick (each
        # re-pick reset Tier-3's rotate hysteresis and per-goal bounds).
        hd, io, po = self._build(points=((0.0, 5.0),), repick_hysteresis_rad=0.2)
        self._drive_to_sending(hd)
        for yaw in (0.2, 0.5, 0.9, 1.3):
            po.pose = (0.0, 0.0, yaw)
            self.assertEqual(hd.tick(0.0), HierState.DRIVING_SUBGOAL)
        self.assertEqual(len(io.sent), 1)

    def test_no_drift_repick_while_tier3_rotating(self):
        hd, io, po = self._build(points=((5.0, 0.0),), repick_hysteresis_rad=0.2)
        self._drive_to_sending(hd)
        io.set_status(cmd_id=1, state="DRIVING", mode="rotate")
        po.pose = (0.0, -2.0, 0.0)            # would be drift...
        self.assertEqual(hd.tick(0.0), HierState.DRIVING_SUBGOAL)
        self.assertEqual(len(io.sent), 1)     # ...but Tier-3 is rotating
        io.set_status(cmd_id=1, state="DRIVING", mode="follow")
        self.assertEqual(hd.tick(0.0), HierState.DRIVING_SUBGOAL)
        self.assertEqual(len(io.sent), 2)     # now it re-picks

    def test_leg_deadline_not_reset_by_repicks(self):
        hd, io, po = self._build(points=((5.0, 0.0),), leg_deadline_s=10.0)
        hd.start()
        hd.tick(0.0)
        hd.tick(0.0)                                  # leg starts t=0, goto #1
        for i, t in enumerate((3.0, 6.0, 9.0)):       # sub-goal ARRIVED re-picks
            io.set_status(cmd_id=len(io.sent), state="ARRIVED")
            self.assertEqual(hd.tick(t), HierState.DRIVING_SUBGOAL)
        cancels = io.cancels
        self.assertEqual(hd.tick(10.5), HierState.BLOCKED)
        self.assertEqual(hd.block_reason(), "leg_timeout")
        self.assertGreater(io.cancels, cancels)
        # Terminal for the leg: no retry gotos, paused for the operator.
        sent = len(io.sent)
        self.assertEqual(hd.tick(11.0), HierState.BLOCKED)
        self.assertEqual(hd.tick(30.0), HierState.BLOCKED)
        self.assertEqual(len(io.sent), sent)
        self.assertTrue(hd.request_resume())          # fresh leg budget
        hd.tick(31.0)
        self.assertEqual(hd.tick(31.0), HierState.DRIVING_SUBGOAL)

    def test_leg_deadline_resets_on_waypoint_advance(self):
        hd, io, po = self._build(points=((1.0, 0.0), (5.0, 0.0)), leg_deadline_s=10.0)
        hd.start()
        hd.tick(0.0)
        hd.tick(0.0)
        po.pose = (1.1, 0.0, 0.0)                     # passed waypoint 0 at t=8
        self.assertEqual(hd.tick(8.0), HierState.DRIVING_SUBGOAL)
        self.assertEqual(hd._runner.wp_index, 1)
        self.assertEqual(hd.tick(15.0), HierState.DRIVING_SUBGOAL)   # 7 s into leg 2

    def test_tier3_blocked_cancels_goal(self):
        # X3: entering BLOCKED revokes the Tier-3 goal (it would otherwise keep
        # streaming cmd_vel and resume on its own if the block cleared).
        hd, io, _ = self._build(points=((5.0, 0.0),))
        self._drive_to_sending(hd)
        cancels = io.cancels
        io.set_status(cmd_id=1, state="BLOCKED", blocked_reason="swept_block")
        self.assertEqual(hd.tick(0.0), HierState.BLOCKED)
        self.assertEqual(io.cancels, cancels + 1)

    def test_teleop_suspends_and_cancels(self):
        hd, io, _ = self._build(points=((5.0, 0.0),))
        self._drive_to_sending(hd)
        cancels = io.cancels
        hd.notify_teleop(True)
        self.assertEqual(hd.state(), HierState.SUSPENDED)
        self.assertEqual(hd.block_reason(), "teleop")
        self.assertEqual(io.cancels, cancels + 1)
        hd.notify_teleop(True)                        # idempotent
        self.assertEqual(io.cancels, cancels + 1)
        self.assertEqual(hd.tick(1.0), HierState.SUSPENDED)
        hd2, io2, _ = self._build()
        hd2.notify_teleop(True)                       # IDLE: no-op
        self.assertEqual(hd2.state(), HierState.IDLE)
        self.assertEqual(io2.cancels, 0)

    def test_rejected_repicks_then_blocks(self):
        hd, io, _ = self._build(points=((5.0, 0.0),), max_consecutive_rejects=2)
        self._drive_to_sending(hd)
        io.set_status(cmd_id=1, state="REJECTED", watermark=50)
        self.assertEqual(hd.tick(0.0), HierState.DRIVING_SUBGOAL)   # re-pick
        self.assertEqual(len(io.sent), 2)
        io.set_status(cmd_id=2, state="REJECTED", watermark=50)
        self.assertEqual(hd.tick(0.1), HierState.DRIVING_SUBGOAL)
        io.set_status(cmd_id=3, state="REJECTED", watermark=50)
        self.assertEqual(hd.tick(0.2), HierState.BLOCKED)
        self.assertEqual(hd.block_reason(), "rejected")

    def test_rejected_at_watermark_is_real_rejection(self):
        # Tier-3 absorbs resends of an accepted id silently; a REJECTED whose
        # watermark equals our id means the goto itself was refused (malformed),
        # so hier must re-pick rather than wait on a goal Tier-3 never took.
        hd, io, _ = self._build(points=((5.0, 0.0),))
        self._drive_to_sending(hd)
        io.set_status(cmd_id=1, state="REJECTED", watermark=1)
        self.assertEqual(hd.tick(0.0), HierState.DRIVING_SUBGOAL)
        self.assertEqual(len(io.sent), 2)

    def test_no_ack_blocks(self):
        hd, io, _ = self._build(points=((5.0, 0.0),))
        self._drive_to_sending(hd)
        io.failed_acks.add(1)
        self.assertEqual(hd.tick(0.0), HierState.BLOCKED)
        self.assertEqual(hd.block_reason(), "no_ack")

    def test_clear_run_goal_anchored_at_scan_ts(self):
        # X6: the body-frame sub-goal is stamped with the ts of the scan it was
        # computed from; blind projection (no scan) uses latest-odom anchoring.
        scan = dict(_clear_scan(), ts=123.25)
        hd, io, _ = self._build(points=((5.0, 0.0),), scan=scan)
        self._drive_to_sending(hd)
        self.assertEqual(io.anchor_ts, [123.25])
        io2 = FakeDriveIO(scan=None)
        hd2 = HierarchicalDrive(_runner(((5.0, 0.0),)), FakePose((0.0, 0.0, 0.0)), io2)
        self._drive_to_sending(hd2)
        self.assertEqual(io2.anchor_ts, [None])

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

    # ── Block advisories (slow VLM verdicts; caution-only) ───────────

    def _blocked(self, **cfg):
        cfg.setdefault("blocked_retry_interval_s", 0.5)
        cfg.setdefault("blocked_retry_window_s", 10.0)
        hd, io, po = self._build(points=((5.0, 0.0),), **cfg)
        self._drive_to_sending(hd)
        io.set_status(cmd_id=1, state="BLOCKED", blocked_reason="depth_block")
        self.assertEqual(hd.tick(1.0), HierState.BLOCKED)
        return hd, io, po

    def test_block_run_numbers_runs_not_retries(self):
        hd, io, _ = self._blocked()
        run = hd.block_run()
        self.assertEqual(run, 1)
        self.assertEqual(hd.tick(1.6), HierState.DRIVING_SUBGOAL)   # retry
        io.set_status(cmd_id=2, state="BLOCKED", blocked_reason="depth_block")
        self.assertEqual(hd.tick(1.7), HierState.BLOCKED)
        self.assertEqual(hd.block_run(), run)                        # same run
        self.assertEqual(hd.tick(2.3), HierState.DRIVING_SUBGOAL)   # retry → goto 3
        io.set_status(cmd_id=3, state="ARRIVED")                     # progress
        hd.tick(2.4)                                                 # re-pick → goto 4
        io.set_status(cmd_id=4, state="BLOCKED", blocked_reason="depth_block")
        self.assertEqual(hd.tick(2.5), HierState.BLOCKED)
        self.assertEqual(hd.block_run(), run + 1)                    # new run

    def test_hold_suppresses_retries_then_resumes(self):
        hd, io, _ = self._blocked()
        self.assertTrue(hd.apply_block_advisory(
            hd.block_run(), BlockAdvisory("hold", "person", hold_s=3.0)))
        sent = len(io.sent)
        for t in (1.6, 2.5, 3.9):
            self.assertEqual(hd.tick(t), HierState.BLOCKED)
        self.assertEqual(len(io.sent), sent)                 # no retry gotos
        self.assertEqual(hd.block_advisory().label, "person")
        self.assertEqual(hd.tick(4.1), HierState.DRIVING_SUBGOAL)   # hold over
        self.assertEqual(len(io.sent), sent + 1)

    def test_pause_gives_up_window_now(self):
        hd, io, _ = self._blocked()
        cancels = io.cancels
        self.assertTrue(hd.apply_block_advisory(
            hd.block_run(), BlockAdvisory("pause", "box")))
        self.assertEqual(io.cancels, cancels + 1)
        sent = len(io.sent)
        for t in (1.6, 5.0, 9.0):
            self.assertEqual(hd.tick(t), HierState.BLOCKED)
        self.assertEqual(len(io.sent), sent)
        self.assertTrue(hd.can_resume())
        self.assertTrue(hd.request_resume())
        self.assertIsNone(hd.block_advisory())

    def test_note_changes_nothing(self):
        hd, io, _ = self._blocked()
        self.assertTrue(hd.apply_block_advisory(
            hd.block_run(), BlockAdvisory("note", "floor_transition")))
        self.assertEqual(hd.tick(1.6), HierState.DRIVING_SUBGOAL)   # normal retry
        self.assertEqual(hd.block_advisory().label, "floor_transition")

    def test_advisory_for_old_run_is_ignored(self):
        hd, io, _ = self._blocked()
        old = hd.block_run()
        self.assertEqual(hd.tick(1.6), HierState.DRIVING_SUBGOAL)
        io.set_status(cmd_id=2, state="ARRIVED")                     # got past it
        hd.tick(1.7)
        self.assertEqual(hd.state(), HierState.DRIVING_SUBGOAL)
        sent, cancels = len(io.sent), io.cancels
        self.assertFalse(hd.apply_block_advisory(old, BlockAdvisory("pause", "box")))
        self.assertEqual(hd.state(), HierState.DRIVING_SUBGOAL)
        self.assertEqual((len(io.sent), io.cancels), (sent, cancels))
        self.assertIsNone(hd.block_advisory())

    def test_advisory_landing_mid_retry_revokes_goal(self):
        hd, io, _ = self._blocked()
        self.assertEqual(hd.tick(1.6), HierState.DRIVING_SUBGOAL)   # retry in flight
        cancels = io.cancels
        self.assertTrue(hd.apply_block_advisory(
            hd.block_run(), BlockAdvisory("pause", "chair")))
        self.assertEqual(hd.state(), HierState.BLOCKED)
        self.assertGreater(io.cancels, cancels)
        self.assertEqual(hd.block_reason(), "depth_block")

    def test_advisory_ignored_when_not_blocked(self):
        hd, io, _ = self._build(points=((5.0, 0.0),))
        self._drive_to_sending(hd)
        self.assertFalse(hd.apply_block_advisory(
            hd.block_run(), BlockAdvisory("pause", "box")))
        self.assertEqual(hd.state(), HierState.DRIVING_SUBGOAL)

    def test_unknown_advisory_action_rejected(self):
        hd, _, _ = self._blocked()
        with self.assertRaises(ValueError):
            hd.apply_block_advisory(hd.block_run(), BlockAdvisory("go", "x"))

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
