"""Unit tests for the Tier-3 depth obstacle memory (body/lib/depth_memory.py).

``render_depth`` ray-casts a synthetic world (floor plane + axis-aligned
boxes in the odom frame) into an OAK-style depth image so the memory sees the
same kind of frames the Pi does. Also used by test_tier3_controller.
"""
from __future__ import annotations

import base64
import math
import unittest
from dataclasses import replace

import numpy as np

from body.lib.depth_memory import (
    DepthMemoryConfig, DepthObstacleMemory, memory_config_from_dict,
    rounded_xy, stamp_blocked,
)
from body.lib.depth_veto import DepthVetoConfig, _cam_to_body_Rt

W_PX, H_PX = 120, 90


def depth_msg(arr_mm: np.ndarray, *, ts: float, seq: int) -> dict:
    h, w = arr_mm.shape
    return {
        "ts": ts, "seq": seq, "format": "depth_uint16_mm", "width": w, "height": h,
        "data": base64.standard_b64encode(arr_mm.astype(np.uint16).tobytes()).decode("ascii"),
    }


def render_depth(pose, boxes, cfg: DepthVetoConfig, *, w=W_PX, h=H_PX,
                 max_m=4.0) -> np.ndarray:
    """Depth (mm, camera z) seen from robot ``pose`` (odom x, y, θ) of the
    floor (z=0) and ``boxes`` [(x0, y0, x1, y1, z_top)] in odom."""
    hfov, vfov = math.radians(cfg.depth_hfov_deg), math.radians(cfg.depth_vfov_deg)
    fx = (w - 1) / (2 * math.tan(hfov / 2))
    fy = (h - 1) / (2 * math.tan(vfov / 2))
    cx, cy = (w - 1) / 2, (h - 1) / 2
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    dc = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu, float)], axis=-1)
    r_bc, t_bc = _cam_to_body_Rt(cfg)
    db = dc @ r_bc.T                                   # body dirs (camera-z = 1)
    c, s = math.cos(pose[2]), math.sin(pose[2])
    dx = c * db[..., 0] - s * db[..., 1]
    dy = s * db[..., 0] + c * db[..., 1]
    dz = db[..., 2]
    ox = pose[0] + c * t_bc[0] - s * t_bc[1]
    oy = pose[1] + s * t_bc[0] + c * t_bc[1]
    oz = t_bc[2] + cfg.ground_z_body_m
    best = np.full(dz.shape, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        tf = np.where(dz < -1e-9, -oz / dz, np.inf)
        best = np.minimum(best, tf)
        for (x0, y0, x1, y1, ztop) in boxes:
            tmin = np.zeros_like(best)
            tmax = np.full_like(best, np.inf)
            for o, d, lo, hi in ((ox, dx, x0, x1), (oy, dy, y0, y1), (oz, dz, 0.0, ztop)):
                t1 = (lo - o) / d
                t2 = (hi - o) / d
                inside = (o >= lo) & (o <= hi)
                near = np.where(np.abs(d) < 1e-12, np.where(inside, -np.inf, np.inf),
                                np.minimum(t1, t2))
                far = np.where(np.abs(d) < 1e-12, np.where(inside, np.inf, -np.inf),
                               np.maximum(t1, t2))
                tmin = np.maximum(tmin, near)
                tmax = np.minimum(tmax, far)
            hit = (tmax >= tmin) & (tmin > 0)
            best = np.where(hit, np.minimum(best, tmin), best)
    best[best > max_m] = np.inf
    return np.where(np.isfinite(best), best * 1000.0, 0).astype(np.uint16)


def veto_cfg(**kw) -> DepthVetoConfig:
    base = dict(enabled=True, depth_median_kernel=1, depth_pitch_rad=0.14,
                depth_z_body_m=0.09)
    base.update(kw)
    return DepthVetoConfig(**base)


MEM = DepthMemoryConfig(enabled=True)
# A box straddling the forward axis 0.5 m ahead (well inside the veto ROI).
BOX = (0.50, -0.06, 0.56, 0.06, 0.30)


class Feeder:
    def __init__(self, mem_cfg=MEM, vcfg=None):
        self.vcfg = vcfg or veto_cfg()
        self.mem = DepthObstacleMemory(mem_cfg, self.vcfg)
        self.t = 100.0
        self.seq = 0

    def frame(self, pose, boxes, *, omega=0.0, dt=0.1, now=None):
        self.t += dt
        self.seq += 1
        msg = depth_msg(render_depth(pose, boxes, self.vcfg), ts=self.t, seq=self.seq)
        return self.mem.update(msg, now_wall=self.t if now is None else now,
                               pose_now=pose, pose_depth=pose, omega_radps=omega)

    def near(self, xy, r=0.1):
        p = self.mem.cells_odom()
        return int(np.count_nonzero(np.hypot(p[:, 0] - xy[0], p[:, 1] - xy[1]) < r)) if p.size else 0


class TestRender(unittest.TestCase):
    def test_floor_and_box(self):
        cfg = veto_cfg()
        arr = render_depth((0, 0, 0), [BOX], cfg)
        self.assertGreater(np.count_nonzero(arr), 0.5 * arr.size)
        mid = arr[45, 60]
        self.assertTrue(400 < mid < 600, mid)


class TestConfirmAndClear(unittest.TestCase):
    def test_one_frame_marks_nothing_two_confirm(self):
        f = Feeder()
        f.frame((0, 0, 0), [BOX])
        self.assertEqual(len(f.mem), 0)
        f.frame((0, 0, 0), [BOX])
        self.assertGreater(len(f.mem), 0)
        self.assertGreater(f.near((0.53, 0.0)), 0)
        # Nothing remembered away from the box (floor isn't an obstacle).
        p = f.mem.cells_odom()
        self.assertTrue(np.all(np.abs(p[:, 0] - 0.53) < 0.1), p)

    def test_floor_only_marks_nothing(self):
        f = Feeder()
        for _ in range(5):
            f.frame((0, 0, 0), [])
        self.assertEqual(len(f.mem), 0)

    def test_pending_cell_dropped_if_next_frame_misses(self):
        f = Feeder()
        f.frame((0, 0, 0), [BOX])
        f.frame((0, 0, 0), [])
        f.frame((0, 0, 0), [BOX])
        self.assertEqual(len(f.mem), 0)

    def test_seen_floor_clears_door_moved(self):
        f = Feeder()
        f.frame((0, 0, 0), [BOX])
        f.frame((0, 0, 0), [BOX])
        n = len(f.mem)
        self.assertGreater(n, 0)
        f.frame((0, 0, 0), [])          # one free frame: kept
        self.assertGreater(len(f.mem), 0)
        f.frame((0, 0, 0), [])          # second: forgotten
        self.assertEqual(len(f.mem), 0)

    def test_occluded_cells_not_cleared(self):
        """A cell hidden behind a nearer box isn't 'seen free'."""
        f = Feeder()
        far = (0.70, -0.06, 0.74, 0.06, 0.30)
        f.frame((0, 0, 0), [far])
        f.frame((0, 0, 0), [far])
        self.assertGreater(f.near((0.72, 0.0)), 0)
        near = (0.35, -0.20, 0.38, 0.20, 0.30)       # wall in front hides it
        for _ in range(3):
            f.frame((0, 0, 0), [near])
        self.assertGreater(f.near((0.72, 0.0), r=0.06), 0)

    def test_out_of_view_persists_until_ttl(self):
        f = Feeder(replace(MEM, ttl_s=5.0))
        f.frame((0, 0, 0), [BOX])
        f.frame((0, 0, 0), [BOX])
        n = len(f.mem)
        # Turn away (slowly — under the rotation gate): box out of view.
        for _ in range(10):
            f.frame((0, 0, math.pi / 2), [BOX])
        self.assertEqual(len(f.mem), n)
        f.frame((0, 0, math.pi / 2), [BOX], dt=6.0)
        self.assertEqual(len(f.mem), 0)

    def test_fast_rotation_frames_skipped(self):
        f = Feeder()
        self.assertFalse(f.frame((0, 0, 0), [BOX], omega=1.0))
        self.assertFalse(f.frame((0, 0, 0), [BOX], omega=1.0))
        self.assertEqual(len(f.mem), 0)

    def test_same_frame_counts_once(self):
        f = Feeder()
        f.frame((0, 0, 0), [BOX])
        msg = depth_msg(render_depth((0, 0, 0), [BOX], f.vcfg), ts=f.t, seq=f.seq)
        self.assertFalse(f.mem.update(msg, now_wall=f.t, pose_now=(0, 0, 0),
                                      pose_depth=(0, 0, 0), omega_radps=0.0))
        self.assertEqual(len(f.mem), 0)

    def test_stale_frame_ignored(self):
        f = Feeder()
        f.frame((0, 0, 0), [BOX], now=f.t + 5.0)
        f.frame((0, 0, 0), [BOX], now=f.t + 5.0)
        self.assertEqual(len(f.mem), 0)

    def test_odom_frame_consistent_while_moving(self):
        """Seen from two poses, the box lands on the same odom cells; the body
        query puts it where it is relative to the robot now."""
        f = Feeder()
        f.frame((0.0, 0.0, 0.0), [BOX])
        f.frame((0.1, 0.02, 0.05), [BOX])
        self.assertGreater(f.near((0.53, 0.0), r=0.08), 0)
        b = f.mem.cells_body((0.53, 0.0, math.pi / 2))
        self.assertTrue(np.all(np.hypot(b[:, 0], b[:, 1]) < 0.12), b)

    def test_keep_radius_and_footprint_prune(self):
        f = Feeder()
        f.frame((0, 0, 0), [BOX])
        f.frame((0, 0, 0), [BOX])
        self.assertGreater(len(f.mem), 0)
        f.frame((5.0, 0, 0), [])            # drove far away
        self.assertEqual(len(f.mem), 0)
        f.frame((0, 0, 0), [BOX])
        f.frame((0, 0, 0), [BOX])
        f.mem.update(None, now_wall=f.t, pose_now=(0.53, 0.0, 0.0),
                     pose_depth=None, omega_radps=0.0)   # robot now stands there
        self.assertEqual(f.near((0.53, 0.0), r=0.08), 0)

    def test_disabled_and_reset(self):
        f = Feeder(replace(MEM, enabled=False))
        f.frame((0, 0, 0), [BOX])
        f.frame((0, 0, 0), [BOX])
        self.assertEqual(len(f.mem), 0)
        g = Feeder()
        g.frame((0, 0, 0), [BOX])
        g.frame((0, 0, 0), [BOX])
        g.mem.reset()
        self.assertEqual(len(g.mem), 0)


class TestHelpers(unittest.TestCase):
    def test_stamp_blocked(self):
        grid = np.ones((10, 10), dtype=np.int8)
        meta = {"resolution_m": 0.1, "origin_x_m": -0.5, "origin_y_m": -0.5}
        n = stamp_blocked(grid, meta, np.array([[0.32, 0.0], [0.05, 0.0], [9.0, 0.0]]),
                          keep_clear_xy=(0.0, 0.0), keep_clear_radius_m=0.11)
        self.assertEqual(n, 1)                 # near-robot and off-grid skipped
        self.assertEqual(grid[8, 5], 0)
        self.assertEqual(int(np.count_nonzero(grid == 0)), 1)

    def test_rounded_xy_nearest_first(self):
        pts = np.array([[1.0, 0.0], [0.2, 0.0], [0.5, 0.123456]])
        self.assertEqual(rounded_xy(pts, 2), [[0.2, 0.0], [0.5, 0.12]])
        self.assertIsNone(rounded_xy(np.zeros((0, 2)), 5))

    def test_config_from_dict(self):
        c = memory_config_from_dict({"enabled": False, "ttl_s": 12, "bogus": 1})
        self.assertFalse(c.enabled)
        self.assertEqual(c.ttl_s, 12.0)
        self.assertTrue(memory_config_from_dict(None).enabled)


if __name__ == "__main__":
    unittest.main()
