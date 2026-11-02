"""Closed-loop tests for the Tier-3 controller (body/lib/tier3_controller.py).

A tiny kinematic simulator ray-casts a synthetic world into a 360-beam scan,
rasterizes it with the production scan_raster config, and integrates the
controller's (v, ω). Scenarios are the reviewer reproductions of the
"nav stalls / rotate-realign dither" findings (T1–T9, L2, T12, T13).
"""
import math
import unittest
from dataclasses import replace

from body.lib import drive_config
from body.lib import tier3_controller as tc
from body.lib.drive_safety import (
    explained_unknown_mask, seed_clear_disk, swept_path_blocked,
    unknown_block_mask,
)
from body.lib.local_costmap import build_local_costmap
from body.lib.local_drive_core import STATE_ARRIVED, STATE_BLOCKED, steer_to_body_point
from body.lib.local_planner import lookahead_on_path, path_remaining_m, plan_local
from body.lib.scan_raster import rasterize_scan
from body.lib.tier3_controller import Tier3Controller, Tier3Goal
from body.lib.zenoh_helpers import load_body_config

N_BEAMS = 360
W = 5.0

_BODY_CFG = load_body_config()
RASTER = drive_config.scan_raster_config(_BODY_CFG)
T3 = drive_config.tier3_config(_BODY_CFG)
T3 = replace(T3, depth=replace(T3.depth, enabled=False))


def raycast(pose, segs):
    x, y, th = pose
    out = []
    for k in range(N_BEAMS):
        a = th + k * 2 * math.pi / N_BEAMS
        dx, dy = math.cos(a), math.sin(a)
        best = None
        for (x1, y1, x2, y2) in segs:
            ex, ey = x2 - x1, y2 - y1
            den = dx * ey - dy * ex
            if abs(den) < 1e-12:
                continue
            t = ((x1 - x) * ey - (y1 - y) * ex) / den
            u = ((x1 - x) * dy - (y1 - y) * dx) / den
            if t > 0.02 and 0 <= u <= 1 and (best is None or t < best):
                best = t
        out.append(best if best is not None and best < 12 else None)
    return out


def min_clearance(pose, segs):
    x, y, _ = pose
    m = 9.0
    for (x1, y1, x2, y2) in segs:
        ex, ey = x2 - x1, y2 - y1
        L = ex * ex + ey * ey
        t = max(0.0, min(1.0, ((x - x1) * ex + (y - y1) * ey) / L))
        m = min(m, math.hypot(x1 + t * ex - x, y1 + t * ey - y))
    return m


def post(cx, cy, r):
    return [(cx - r, cy - r, cx + r, cy - r), (cx + r, cy - r, cx + r, cy + r),
            (cx + r, cy + r, cx - r, cy + r), (cx - r, cy + r, cx - r, cy - r)]


class Sim:
    """Kinematic unicycle + scan → controller loop at 10 Hz."""

    def __init__(self, segs, goal, *, pose=(0.0, 0.0, 0.0), tol=0.12, cfg=T3,
                 latency_ticks=0, final_heading=None, motor=lambda v, w: (v, w)):
        self.segs = segs
        self.pose = pose
        self.poses = [pose]
        self.latency = latency_ticks
        self.motor = motor
        self.ctrl = Tier3Controller(cfg)
        self.ctrl.start_goal(Tier3Goal(1, goal[0], goal[1], tol,
                                       final_heading=final_heading), 0.0, pose)
        self.t = 0.0
        self.results = []
        self.primary_blocked = 0
        self.min_clear = 9.0

    def run(self, T=30.0):
        dt = 0.1
        r = None
        while self.t < T:
            ps = self.poses[max(0, len(self.poses) - 1 - self.latency)]
            g, m = rasterize_scan(raycast(ps, self.segs), 0.0, 2 * math.pi / N_BEAMS, RASTER)
            r = self.ctrl.step(now_mono=self.t, now_wall=self.t, pose_now=self.pose,
                               pose_scan=ps, grid=g, meta=m)
            self.results.append(r)
            self.min_clear = min(self.min_clear, min_clearance(self.pose, self.segs))
            if r.done or r.terminal:
                return r
            v, w = self.motor(*(r.cmd or (0.0, 0.0)))
            x, y, th = self.pose
            self.pose = (x + v * math.cos(th) * dt, y + v * math.sin(th) * dt, th + w * dt)
            self.poses.append(self.pose)
            self.t += dt
        return r

    def reasons(self):
        return [x.reason for x in self.results if x.reason]


class _CountPrimary:
    """Counts follow ticks whose FIRST (nominal) command the veto rejects."""

    def __init__(self):
        self.calls = []

    def __enter__(self):
        self._orig = tc.swept_path_blocked
        self._orig_step = Tier3Controller.step
        outer = self

        def sw(*a, **k):
            r = outer._orig(*a, **k)
            outer.calls[-1].append(r)
            return r

        def step(ctrl, **kw):
            outer.calls.append([])
            return outer._orig_step(ctrl, **kw)

        tc.swept_path_blocked = sw
        Tier3Controller.step = step
        return self

    def __exit__(self, *exc):
        tc.swept_path_blocked = self._orig
        Tier3Controller.step = self._orig_step

    @property
    def primary_blocked(self):
        return sum(1 for c in self.calls if c and c[0])

    @property
    def follow_checks(self):
        return sum(1 for c in self.calls if c)


INVARIANT_WORLDS = {
    "thin_post": post(0.7, 0.0, 0.02),
    "post_offset": post(0.6, 0.15, 0.02),
    "post_near": post(0.5, -0.1, 0.03),
    "wall_end_left": [(0.6, 0.1, 0.6, W)],
    "wall_end_right": [(0.5, -W, 0.5, -0.15)],
    "door_jambs": [(0.8, -W, 0.8, -0.15), (0.8, 0.45, 0.8, W)],
    "wall_follow_0.15": [(-W, 0.15, W, 0.15)],
    "wall_follow_0.18": [(-W, 0.18, W, 0.18)],
    "wall_follow_0.20": [(-W, 0.20, W, 0.20)],
    "wall_follow_right_0.17": [(-W, -0.17, W, -0.17)],
}


class TestPlannerVetoInvariant(unittest.TestCase):
    """Fix 3: the planner never sees fewer obstacles than the veto — an A*
    path from plan_local, followed with the controller's nominal command,
    passes swept_path_blocked on the same grid."""

    def _grid(self, segs, pose=(0.0, 0.0, 0.0)):
        g, m = rasterize_scan(raycast(pose, segs), 0.0, 2 * math.pi / N_BEAMS, RASTER)
        work = g.copy()
        seed_clear_disk(work, m, (0.0, 0.0), T3.seed_clear_radius_m)
        return work, m

    def test_static_first_command_passes_veto(self):
        foot = T3.foot
        checked = 0
        for name, segs in INVARIANT_WORLDS.items():
            work, m = self._grid(segs)
            ex = explained_unknown_mask(work, foot.unknown_explained_cells)
            cm = build_local_costmap(work, m, T3.plan.costmap)
            ub = unknown_block_mask(work, m, (0.0, 0.0), foot, ex)
            cm.lethal |= ub
            for goal in [(1.4, 0.0), (1.2, 0.3), (1.2, -0.3), (0.9, 0.1), (1.5, -0.1)]:
                p = plan_local(work, m, goal, T3.plan, costmap=cm)
                if not p.ok:
                    continue
                look = lookahead_on_path(p.path_body, T3.lookahead_m)
                v, w, _d, _b, _rot = steer_to_body_point(look, T3.params, False)
                if v <= 0:
                    continue
                checked += 1
                blocked = swept_path_blocked(
                    work, m, v_mps=v, omega_radps=w, config=foot,
                    max_reach_m=path_remaining_m(p.path_body) + T3.stop_margin_m,
                    explained_unknown=ex)
                self.assertFalse(blocked, f"{name} goal={goal}: A* path vetoed")
        self.assertGreater(checked, 3)

    def test_closed_loop_nominal_command_never_vetoed(self):
        for name, segs in INVARIANT_WORLDS.items():
            for goal in [(1.4, 0.0), (1.2, 0.5), (1.2, -0.5)]:
                with _CountPrimary() as cp:
                    sim = Sim(segs, goal)
                    sim.run(T=30.0)
                self.assertEqual(cp.primary_blocked, 0,
                                 f"{name} goal={goal}: nominal follow command vetoed "
                                 f"{cp.primary_blocked}/{cp.follow_checks} ticks")
                self.assertNotIn("swept_block", sim.reasons(), f"{name} goal={goal}")


class TestReachesGoals(unittest.TestCase):
    def _arrive(self, segs, goal, **kw):
        sim = Sim(segs, goal, **kw)
        r = sim.run(T=kw.pop("T", 30.0) if "T" in kw else 30.0)
        self.assertEqual(r.state, STATE_ARRIVED,
                         f"goal={goal}: {r.state} {r.reason} after {sim.t:.1f}s")
        return sim, r

    def test_subgoal_in_front_of_wall_arrives(self):
        # T2/T8: Tier-2 places sub-goals 0.16–0.22 m in front of a wall. The
        # veto used to stop the robot ~0.37 m short and dither to deadline.
        for gap in (0.17, 0.22, 0.30):
            sim, r = self._arrive([(1.0, -W, 1.0, W)], (1.0 - gap, 0.0))
            self.assertLess(sim.t, 8.0)
            self.assertGreater(sim.min_clear, 0.2)

    def test_wall_following_close(self):
        # T5: following a wall at 0.15–0.20 m used to be a permanent swept_block.
        for off in (0.15, 0.20):
            sim, _r = self._arrive([(-W, off, W, off)], (1.5, 0.0))
            self.assertGreaterEqual(sim.min_clear, off - 0.01)

    def test_thin_posts_are_avoided_not_blocked_on(self):
        # T1: a 2–8 cm post was speckle-removed from the planner → A* drove
        # straight at it and the raw-grid veto froze the robot.
        for r_post in (0.01, 0.02, 0.04):
            sim, _r = self._arrive(post(0.7, 0.0, r_post), (1.4, 0.0))
            self.assertGreater(sim.min_clear, 0.15)

    def test_doorways(self):
        self._arrive([(0.8, -W, 0.8, -0.1), (0.8, 0.5, 0.8, W)], (1.6, 0.2))
        self._arrive([(0.8, -W, 0.8, 0.05), (0.8, 0.55, 0.8, W)], (1.6, 0.3))

    def test_goal_behind_robot_no_omega_flip(self):
        # T4: a goal behind the robot flipped ω every tick.
        sim, _r = self._arrive([(-W, 0.6, W, 0.6), (-W, -0.6, W, -0.6)], (-1.0, 0.02))
        rot = [x.omega for x in sim.results if x.mode == "rotate"]
        flips = sum(1 for a, b in zip(rot, rot[1:]) if a * b < 0)
        self.assertLessEqual(flips, 1)

    def test_goal_beyond_wall_arrives_at_best_endpoint(self):
        # T8: reaching the best reachable cell is ARRIVED (plan_reason says
        # it was a stand-in), not BLOCKED:boxed_in.
        sim, r = self._arrive([(1.0, -W, 1.0, W)], (1.4, 0.0))
        self.assertIn(r.plan_reason, ("frontier", "snapped"))
        self.assertGreater(r.dist, 0.4)                 # residual is honest

    def test_scan_latency_is_compensated(self):
        # T10: scans 0.3 s old (pose_scan lags) must not break following.
        self._arrive([(0.8, -W, 0.8, 0.05), (0.8, 0.55, 0.8, W)], (1.6, 0.3),
                     latency_ticks=3)
        self._arrive([(-W, 0.18, W, 0.18)], (1.5, 0.0), latency_ticks=3)


class TestBoundedRotation(unittest.TestCase):
    def test_rotate_timeout_when_robot_cannot_turn(self):
        # A stalled / muted motor: commanded rotation never happens. The
        # per-goal non-translating budget ends it as BLOCKED rotate_timeout
        # well before the 30 s deadline.
        sim = Sim([], (-1.0, 0.0), motor=lambda v, w: (0.0, 0.0))
        r = sim.run(T=30.0)
        self.assertEqual(r.state, STATE_BLOCKED)
        self.assertEqual(r.reason, "rotate_timeout")
        self.assertTrue(r.terminal)
        self.assertLess(sim.t, T3.rotate_timeout_s + 1.0)

    def test_terminal_block_goes_quiet_on_cmd_vel(self):
        sim = Sim([], (-1.0, 0.0), motor=lambda v, w: (0.0, 0.0))
        sim.run(T=30.0)
        ctrl = sim.ctrl
        cmds = []
        for k in range(6):
            r = ctrl.step(now_mono=sim.t + 0.1 * (k + 1), now_wall=0.0, pose_now=sim.pose,
                          pose_scan=sim.pose, grid=None, meta=None)
            self.assertEqual(r.reason, "rotate_timeout")
            cmds.append(r.cmd)
        # terminal_zero_ticks zeros in total (one already sent), then silence
        self.assertEqual(cmds[-1], None)
        self.assertIn((0.0, 0.0), cmds[:2])

    def test_final_heading_bounded_by_budget(self):
        # T13: the final-heading turn used to bypass every timer.
        sim = Sim([], (0.5, 0.0), final_heading=math.pi / 2,
                  motor=lambda v, w: (v, 0.0))       # wheels drive, never turn
        r = sim.run(T=40.0)
        self.assertEqual(r.state, STATE_BLOCKED)
        self.assertIn(r.reason, ("rotate_timeout", "deadline"))

    def test_final_heading_completes(self):
        sim = Sim([], (0.5, 0.0), final_heading=math.pi / 2)
        r = sim.run(T=20.0)
        self.assertEqual(r.state, STATE_ARRIVED)
        self.assertAlmostEqual(sim.pose[2], math.pi / 2, delta=0.1)

    def test_rotation_does_not_reset_translation_clock(self):
        # T3: realign/rotate ticks reset realign_since; only translation may.
        ctrl = Tier3Controller(T3)
        ctrl.start_goal(Tier3Goal(1, -1.0, 0.0, 0.12), 0.0, (0.0, 0.0, 0.0))
        g, m = rasterize_scan(raycast((0, 0, 0), []), 0.0, 2 * math.pi / N_BEAMS, RASTER)
        th = 0.0
        for k in range(5):
            r = ctrl.step(now_mono=0.1 * k, now_wall=0.0, pose_now=(0, 0, th),
                          pose_scan=(0, 0, th), grid=g, meta=m)
            th += 0.06
        self.assertEqual(r.mode, "rotate")
        self.assertEqual(ctrl.last_translate_at, 0.0)
        ctrl.step(now_mono=0.6, now_wall=0.0, pose_now=(0.06, 0.0, th),
                  pose_scan=(0.06, 0.0, th), grid=g, meta=m)
        self.assertAlmostEqual(ctrl.last_translate_at, 0.6)

    def test_rotating_in_place_keeps_committed_path(self):
        # T4 path hysteresis: while turning to line up, the path is frozen in
        # odom (not re-planned on the re-quantized rotating grid).
        sim = Sim(post(0.7, 0.0, 0.02), (0.8, 0.0))
        r = sim.run(T=30.0)
        self.assertEqual(r.state, STATE_ARRIVED)
        rot = [x.omega for x in sim.results if x.mode == "rotate"]
        flips = sum(1 for a, b in zip(rot, rot[1:]) if a * b < 0)
        self.assertLessEqual(flips, 1)


class TestFailures(unittest.TestCase):
    def test_no_scan_is_blocked_not_driving(self):
        ctrl = Tier3Controller(T3)
        ctrl.start_goal(Tier3Goal(1, 1.0, 0.0, 0.12), 0.0, (0.0, 0.0, 0.0))
        r = ctrl.step(now_mono=0.0, now_wall=0.0, pose_now=(0, 0, 0), pose_scan=None,
                      grid=None, meta=None)
        self.assertEqual((r.state, r.reason, r.cmd), (STATE_BLOCKED, "no_scan", (0.0, 0.0)))

    def test_deadline_is_terminal(self):
        cfg = replace(T3, goal_deadline_s=1.0)
        sim = Sim([(1.0, -W, 1.0, W)], (3.0, 0.0), cfg=cfg, motor=lambda v, w: (0.0, 0.0))
        r = sim.run(T=5.0)
        self.assertEqual(r.reason, "deadline")
        self.assertTrue(r.terminal)



class TestDepthMemoryNoCreep(unittest.TestCase):
    """2026-10-31 door edge: an obstacle only the depth camera sees. With no
    memory, every retry (a new goal) drives ≥ 2 depth frames into it before
    the veto stops it — creeping ~2 cm closer per cycle. With the memory it
    is planned around like a lidar hit and the veto never has to fire."""

    PANEL = (0.60, -0.05, 0.64, 0.80, 2.0)    # open door: edge on the forward axis
    GOAL = (1.4, 0.0)

    def _run(self, mem_on, retries=8, T=40.0):
        from body.lib.depth_memory import DepthMemoryConfig
        from body.test_depth_memory import depth_msg, render_depth
        cfg = replace(T3, depth=replace(T3.depth, enabled=True), depth_act=True,
                      memory=DepthMemoryConfig(enabled=mem_on))
        ctrl = Tier3Controller(cfg)
        pose, t, seq, cmd = (0.0, 0.0, 0.0), 0.0, 0, 1
        x0, y0, x1, y1, _ = self.PANEL

        def dist(p):
            return math.hypot(max(x0 - p[0], 0.0, p[0] - x1), max(y0 - p[1], 0.0, p[1] - y1))

        blocks, dmin, retry_at = [], 9.0, 0.0
        ctrl.start_goal(Tier3Goal(cmd, *self.GOAL, 0.12), t, pose)
        while t < T:
            seq += 1
            dm = depth_msg(render_depth(pose, [self.PANEL], cfg.depth), ts=t, seq=seq)
            ctrl.observe_depth(dm, now_wall=t, pose_now=pose, pose_depth=pose, omega_radps=0.0)
            if ctrl.goal is None:                       # hier retry cadence
                if t < retry_at:
                    t += 0.1
                    continue
                cmd += 1
                ctrl.start_goal(Tier3Goal(cmd, *self.GOAL, 0.12), t, pose)
            g, m = rasterize_scan(raycast(pose, []), 0.0, 2 * math.pi / N_BEAMS, RASTER)
            r = ctrl.step(now_mono=t, now_wall=t, pose_now=pose, pose_scan=pose,
                          grid=g, meta=m, depth=dm)
            if r.done and r.state == STATE_ARRIVED:
                return "ARRIVED", blocks, dmin
            if r.state == STATE_BLOCKED:
                blocks.append((r.reason, dist(pose)))
                ctrl.clear()
                retry_at = t + 0.5
                if len(blocks) > retries:
                    return "GAVEUP", blocks, dmin
            v, w = r.cmd or (0.0, 0.0)
            x, y, th = pose
            pose = (x + v * math.cos(th) * 0.1, y + v * math.sin(th) * 0.1, th + w * 0.1)
            dmin = min(dmin, dist(pose))
            t += 0.1
        return "TIMEOUT", blocks, dmin

    def test_without_memory_reproduces_creep(self):
        state, blocks, _ = self._run(False)
        self.assertEqual(state, "GAVEUP")
        self.assertTrue(all(b[0] == "depth_block" for b in blocks), blocks)
        d = [b[1] for b in blocks]
        self.assertLess(d[-1], d[0] - 0.1, d)          # crept > 10 cm closer

    def test_with_memory_plans_around_without_blocking(self):
        state, blocks, dmin = self._run(True)
        self.assertEqual(state, "ARRIVED", blocks)
        self.assertEqual([b for b in blocks if b[0] == "depth_block"], [])
        self.assertGreater(dmin, T3.plan.costmap.footprint_radius_m)

    def test_memory_survives_new_goal(self):
        from body.lib.depth_memory import DepthMemoryConfig
        from body.test_depth_memory import depth_msg, render_depth
        cfg = replace(T3, memory=DepthMemoryConfig(enabled=True))
        ctrl = Tier3Controller(cfg)
        for k in (1, 2):
            dm = depth_msg(render_depth((0, 0, 0), [self.PANEL], cfg.depth), ts=k * 0.1, seq=k)
            ctrl.observe_depth(dm, now_wall=k * 0.1, pose_now=(0, 0, 0),
                               pose_depth=(0, 0, 0), omega_radps=0.0)
        n = len(ctrl.memory)
        self.assertGreater(n, 0)
        ctrl.start_goal(Tier3Goal(7, 1.0, 0.0, 0.12), 0.3, (0, 0, 0))
        ctrl.clear()
        self.assertEqual(len(ctrl.memory), n)


    def test_stale_ring_next_to_robot_does_not_pin_it(self):
        """10-31: robot carried by hand → old cells land 0.14 m around it (just
        outside the footprint, in the camera's near blind zone, so never seen
        free). Planning must ignore cells that close, or the robot starts in
        lethal space and spins to rotate_timeout."""
        import numpy as np
        from body.lib.depth_memory import DepthMemoryConfig
        ring = np.array([[0.14 * math.cos(a), 0.14 * math.sin(a)]
                         for a in np.linspace(0, 2 * math.pi, 24, endpoint=False)])

        def first_moves(exclude):
            cfg = replace(T3, memory=DepthMemoryConfig(enabled=True,
                                                        plan_exclude_radius_m=exclude))
            ctrl = Tier3Controller(cfg)
            ctrl.memory.cells_body = lambda pose: ring
            ctrl.memory.__class__ = type("Pinned", (type(ctrl.memory),),
                                         {"__len__": lambda self: len(ring)})
            ctrl.start_goal(Tier3Goal(1, 1.0, 0.0, 0.12), 0.0, (0.0, 0.0, 0.0))
            g, m = rasterize_scan(raycast((0.0, 0.0, 0.0), []), 0.0,
                                  2 * math.pi / N_BEAMS, RASTER)
            r = ctrl.step(now_mono=0.0, now_wall=0.0, pose_now=(0.0, 0.0, 0.0),
                          pose_scan=(0.0, 0.0, 0.0), grid=g, meta=m)
            return r

        r = first_moves(0.20)
        self.assertEqual(r.state, "DRIVING", r)            # not start_blocked
        self.assertEqual(r.plan_reason, "ok")
        # Cells AHEAD stay (the camera can't see that close): the plan does
        # not drive through them — it leaves backwards/around.
        self.assertLess(r.path_body[1][0], 0.05, r.path_body[:3])
        pinned = first_moves(0.0)                           # old behaviour
        self.assertFalse(pinned.state == "DRIVING" and pinned.plan_reason == "ok",
                         pinned)

    def test_behind_only_cells_do_not_slow_departure(self):
        import numpy as np
        from body.lib.depth_memory import DepthMemoryConfig
        behind = np.array([[-0.14, 0.0], [-0.1, 0.1], [-0.1, -0.1], [0.0, 0.14], [0.0, -0.14]])
        ctrl = Tier3Controller(replace(T3, memory=DepthMemoryConfig(enabled=True)))
        ctrl.memory.cells_body = lambda pose: behind
        ctrl.memory.__class__ = type("Pinned", (type(ctrl.memory),),
                                     {"__len__": lambda self: len(behind)})
        ctrl.start_goal(Tier3Goal(1, 1.0, 0.0, 0.12), 0.0, (0.0, 0.0, 0.0))
        g, m = rasterize_scan(raycast((0.0, 0.0, 0.0), []), 0.0, 2 * math.pi / N_BEAMS, RASTER)
        r = ctrl.step(now_mono=0.0, now_wall=0.0, pose_now=(0.0, 0.0, 0.0),
                      pose_scan=(0.0, 0.0, 0.0), grid=g, meta=m)
        self.assertEqual((r.state, r.plan_reason), ("DRIVING", "ok"))
        self.assertGreater(r.v, 0.0)


class TestObserveOnlyVeto(unittest.TestCase):
    def test_observe_only_reports_but_does_not_block(self):
        """depth_act=False: the veto scores the frame, the status says
        depth_would_block with the hit evidence, and the robot keeps driving."""
        from body.test_depth_memory import depth_msg, render_depth
        cfg = replace(T3, depth=replace(T3.depth, enabled=True), depth_act=False)
        ctrl = Tier3Controller(cfg)
        panel = (0.40, -0.2, 0.44, 0.2, 0.3)
        ctrl.start_goal(Tier3Goal(1, 1.0, 0.0, 0.12), 0.0, (0.0, 0.0, 0.0))
        g, m = rasterize_scan(raycast((0.0, 0.0, 0.0), []), 0.0, 2 * math.pi / N_BEAMS, RASTER)
        seen = False
        for k in range(4):
            dm = depth_msg(render_depth((0.0, 0.0, 0.0), [panel], cfg.depth), ts=0.1 * k, seq=k)
            r = ctrl.step(now_mono=0.1 * k, now_wall=0.1 * k, pose_now=(0.0, 0.0, 0.0),
                          pose_scan=(0.0, 0.0, 0.0), grid=g, meta=m, depth=dm)
            self.assertNotEqual(r.reason, "depth_block")
            if r.depth_would_block:
                seen = True
                self.assertGreater(r.depth_hits, 0)
                self.assertIsNotNone(r.depth_bbox_px)
                self.assertGreater(r.v, 0.0)
        self.assertTrue(seen, "veto never scored the panel")


class TestEndpointLatch(unittest.TestCase):
    """_latch_endpoint: a latched snapped/frontier endpoint must not survive
    once the path to it has become a long detour (10-31: a 6.6 m loop kept
    over a 0.6 m near-side path for 6 s of rotating in place), and an "ok"
    plan clears a latch only after LATCH_OK_TICKS consecutive ticks (the
    goal cell flickers lethal/non-lethal at an inflation edge)."""

    WALL_FAR_GAP = [(0.5, -W, 0.5, 1.0), (0.5, 1.8, 0.5, W)]

    def _plan(self, ctrl, segs, goal):
        from body.lib.local_costmap import build_local_costmap
        from body.lib.drive_safety import seed_clear_disk
        pose = (0.0, 0.0, 0.0)
        g, m = rasterize_scan(raycast(pose, segs), 0.0, 2 * math.pi / N_BEAMS, RASTER)
        work = g.copy()
        seed_clear_disk(work, m, (0.0, 0.0), T3.seed_clear_radius_m)
        cm = build_local_costmap(work, m, T3.plan.costmap)
        plan = plan_local(work, m, goal, T3.plan, costmap=cm, start_body=(0.0, 0.0))
        return ctrl._latch_endpoint(plan, work, m, cm, goal, pose, pose), plan

    def test_latched_loop_yields_to_short_near_side_path(self):
        ctrl = Tier3Controller(T3)
        ctrl.start_goal(Tier3Goal(1, 0.5, 0.0, 0.12), 0.0, (0.0, 0.0, 0.0))
        # Goal ON the wall (snapped). Latch: a far-side point a little closer
        # to it than the near-side snap, reachable only via the far gap.
        ctrl.endpoint_odom = (0.7, 0.1)
        chosen, fresh = self._plan(ctrl, self.WALL_FAR_GAP, (0.5, 0.0))
        self.assertIn(fresh.reason, ("frontier", "snapped"))
        self.assertLess(path_remaining_m(chosen.path_body), 1.0,
                        "kept the loop over the near-side path")
        self.assertLess(max(x for x, _y in chosen.path_body), 0.5)

    def test_ok_plan_clears_latch_only_after_consecutive_ticks(self):
        ctrl = Tier3Controller(T3)
        ctrl.start_goal(Tier3Goal(1, 0.9, 0.0, 0.12), 0.0, (0.0, 0.0, 0.0))
        ctrl.endpoint_odom = (0.4, 0.0)            # a near-side latch
        reasons = []
        for _ in range(tc.LATCH_OK_TICKS + 1):
            chosen, fresh = self._plan(ctrl, [], (0.9, 0.0))   # open floor: ok
            self.assertEqual(fresh.reason, "ok")
            reasons.append(chosen.reason)
        self.assertEqual(reasons[:tc.LATCH_OK_TICKS - 1],
                         ["snapped"] * (tc.LATCH_OK_TICKS - 1))
        self.assertEqual(reasons[-1], "ok")
        self.assertIsNone(ctrl.endpoint_odom)


if __name__ == "__main__":
    unittest.main()


class TestGoalSpeedCaps(unittest.TestCase):
    """Per-goal v_max / omega_max can only slow the robot (explore plan A3)."""

    def _goal(self, **kw):
        return tc.Tier3Goal(cmd_id=1, x=1.0, y=0.0, tol=0.1, **kw)

    def test_no_caps_returns_base(self):
        base = tc.DriveParams()
        self.assertIs(tc.goal_params(base, self._goal()), base)

    def test_v_max_clamped_to_configured_max(self):
        base = tc.DriveParams(v_max=0.18)
        self.assertAlmostEqual(tc.goal_params(base, self._goal(v_max=0.5)).v_max, 0.18)

    def test_v_max_below_v_min_is_raised_to_v_min(self):
        base = tc.DriveParams(v_max=0.18, v_min_mps=0.08)
        self.assertAlmostEqual(tc.goal_params(base, self._goal(v_max=0.02)).v_max, 0.08)

    def test_slow_v_max_honoured(self):
        base = tc.DriveParams(v_max=0.18)
        self.assertAlmostEqual(tc.goal_params(base, self._goal(v_max=0.10)).v_max, 0.10)

    def test_omega_max_only_lowers(self):
        base = tc.DriveParams(omega_max=0.6)
        self.assertAlmostEqual(tc.goal_params(base, self._goal(omega_max=0.3)).omega_max, 0.3)
        self.assertAlmostEqual(tc.goal_params(base, self._goal(omega_max=2.0)).omega_max, 0.6)
        self.assertAlmostEqual(tc.goal_params(base, self._goal(omega_max=0.0)).omega_max, 0.6)
