"""Unit tests for the Tier-3 near-field depth veto."""
from __future__ import annotations

import base64
import unittest

import numpy as np

from body.lib.depth_veto import (
    DepthVetoConfig,
    count_slab_hits,
    depth_nearfield_blocked,
)
from body.lib.drive_config import depth_veto_config
from body.lib.zenoh_helpers import load_body_config


def _depth_msg(arr_mm: np.ndarray, *, ts: float = 1000.0) -> dict:
    h, w = arr_mm.shape
    return {
        "ts": ts,
        "format": "depth_uint16_mm",
        "width": w,
        "height": h,
        "layout": "row_major",
        "data": base64.standard_b64encode(arr_mm.astype(np.uint16).tobytes()).decode(
            "ascii"
        ),
    }


def _cfg(**kwargs) -> DepthVetoConfig:
    # Level camera, origin at body, so OpenCV Z maps to body +x.
    base = dict(
        enabled=True,
        stale_s=1.0,
        min_range_m=0.10,
        max_range_m=0.80,
        lateral_half_width_m=0.15,
        floor_band_m=0.04,
        clearance_height_m=0.35,
        ground_z_body_m=0.0,
        min_hits=5,
        hit_streak=2,
        max_abs_omega_radps=0.5,
        roi_u0=0.0,
        roi_u1=1.0,
        roi_v0=0.0,
        roi_v1=1.0,
        depth_median_kernel=1,
        depth_hfov_deg=70.0,
        depth_vfov_deg=55.0,
        depth_x_body_m=0.0,
        depth_y_body_m=0.0,
        depth_z_body_m=0.10,
        depth_yaw_rad=0.0,
        depth_pitch_rad=0.0,
        depth_roll_rad=0.0,
    )
    base.update(kwargs)
    return DepthVetoConfig(**base)


class TestCountSlabHits(unittest.TestCase):
    def test_wall_in_envelope_counts(self):
        # Synthetic: every pixel reports ~0.4 m forward (camera Z).
        # With R_fix, cam Z → body X; cam origin at z=0.10 → points at x≈0.4,
        # z≈0.10 → height 0.10 is in the slab (0.04–0.35).
        w, h = 40, 30
        arr = np.full((h, w), 400, dtype=np.uint16)  # 400 mm
        msg = _depth_msg(arr)
        hits = count_slab_hits(arr, msg, _cfg())
        self.assertGreaterEqual(hits, 5)

    def test_far_wall_ignored(self):
        w, h = 40, 30
        arr = np.full((h, w), 2000, dtype=np.uint16)  # 2 m — beyond max_range
        msg = _depth_msg(arr)
        self.assertEqual(count_slab_hits(arr, msg, _cfg()), 0)

    def test_floor_only_ignored(self):
        # Single principal-point pixel: body z ≈ camera height when pitch=0.
        # Height 0.02 m sits inside the floor band and must not veto.
        arr = np.full((1, 1), 400, dtype=np.uint16)
        msg = _depth_msg(arr)
        hits = count_slab_hits(arr, msg, _cfg(depth_z_body_m=0.02, floor_band_m=0.04))
        self.assertEqual(hits, 0)


class TestDepthNearfieldBlocked(unittest.TestCase):
    def test_fail_open_on_missing(self):
        blocked, streak = depth_nearfield_blocked(
            None,
            now_wall=1000.0,
            v_mps=0.12,
            omega_radps=0.0,
            cfg=_cfg(),
            streak=0,
        )
        self.assertFalse(blocked)
        self.assertEqual(streak, 0)

    def test_fail_open_on_stale(self):
        w, h = 40, 30
        arr = np.full((h, w), 400, dtype=np.uint16)
        msg = _depth_msg(arr, ts=1000.0)
        blocked, streak = depth_nearfield_blocked(
            msg,
            now_wall=1002.0,  # 2 s old vs stale_s=1
            v_mps=0.12,
            omega_radps=0.0,
            cfg=_cfg(),
            streak=0,
        )
        self.assertFalse(blocked)
        self.assertEqual(streak, 0)

    def test_skip_when_rotating(self):
        w, h = 40, 30
        arr = np.full((h, w), 400, dtype=np.uint16)
        msg = _depth_msg(arr)
        blocked, streak = depth_nearfield_blocked(
            msg,
            now_wall=1000.0,
            v_mps=0.12,
            omega_radps=0.8,
            cfg=_cfg(max_abs_omega_radps=0.5),
            streak=0,
        )
        self.assertFalse(blocked)
        self.assertEqual(streak, 0)

    def test_streak_required(self):
        w, h = 40, 30
        arr = np.full((h, w), 400, dtype=np.uint16)
        msg = _depth_msg(arr)
        cfg = _cfg(hit_streak=2, min_hits=5)
        blocked1, s1 = depth_nearfield_blocked(
            msg, now_wall=1000.0, v_mps=0.12, omega_radps=0.0, cfg=cfg, streak=0
        )
        self.assertFalse(blocked1)
        self.assertEqual(s1, 1)
        blocked2, s2 = depth_nearfield_blocked(
            msg, now_wall=1000.1, v_mps=0.12, omega_radps=0.0, cfg=cfg, streak=s1
        )
        self.assertTrue(blocked2)
        self.assertEqual(s2, 2)

    def test_pure_rotation_allowed(self):
        w, h = 40, 30
        arr = np.full((h, w), 400, dtype=np.uint16)
        msg = _depth_msg(arr)
        blocked, streak = depth_nearfield_blocked(
            msg,
            now_wall=1000.0,
            v_mps=0.0,
            omega_radps=0.3,
            cfg=_cfg(),
            streak=5,
        )
        self.assertFalse(blocked)
        self.assertEqual(streak, 0)


class TestDepthVetoConfigBuilder(unittest.TestCase):
    def test_builder_reads_repo_config(self):
        cfg = load_body_config()
        dv = depth_veto_config(cfg)
        self.assertIsInstance(dv, DepthVetoConfig)
        section = cfg.get("local_drive", {}).get("depth_veto", {})
        if "max_range_m" in section:
            self.assertAlmostEqual(dv.max_range_m, float(section["max_range_m"]))
        # Pitch defaults from local_map when not overridden.
        self.assertAlmostEqual(
            dv.depth_pitch_rad,
            float(section.get("depth_pitch_rad", cfg["local_map"]["depth_pitch_rad"])),
        )


if __name__ == "__main__":
    unittest.main()
