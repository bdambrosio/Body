"""Tests for the post-fault pose verifier (lidar part + camera fallback hook)."""

from __future__ import annotations

import math
import unittest

from desktop.localization.pose_verify import (
    BAD, OK, PENDING, UNKNOWN, PoseVerifier, PoseVerifyConfig, Verdict,
)


class _Src:
    def __init__(self):
        self.seq = 0
        self.s = {"valid": True, "score_best": 60.0, "n_points": 100,
                  "shift_m": 0.03, "dtheta_rad": 0.01, "scan_obs_run": 0}

    def push(self, **kw):
        self.seq += 1
        self.s = {**self.s, **kw, "scan_obs_run": self.seq}

    def __call__(self):
        return dict(self.s)


def _verifier(src, baseline=0.6, image=None):
    return PoseVerifier(src, lambda: baseline, image_check=image)


class TestPoseVerifier(unittest.TestCase):
    def _feed(self, v, src, n=3, **kw):
        for _ in range(n):
            src.push(**kw)
            v.poll()

    def test_pending_until_fresh_matches(self):
        src = _Src()
        v = _verifier(src)
        v.mark_fault()
        v.poll()                       # same seq as at mark: not fresh
        self.assertEqual(v.verdict().kind, PENDING)
        self._feed(v, src, n=2)
        self.assertEqual(v.verdict().kind, PENDING)
        self._feed(v, src, n=1)
        self.assertEqual(v.verdict().kind, OK)

    def test_repeat_reads_of_same_match_do_not_count(self):
        src = _Src()
        v = _verifier(src)
        v.mark_fault()
        src.push()
        for _ in range(5):
            v.poll()
        self.assertEqual(v.verdict().kind, PENDING)

    def test_quality_relative_to_baseline(self):
        src = _Src()
        v = _verifier(src, baseline=0.6)
        v.mark_fault()
        self._feed(v, src, score_best=20.0)          # 0.20 = 33% of 0.6
        self.assertEqual(v.verdict().kind, BAD)
        v.restart_window()
        self._feed(v, src, score_best=33.0)          # 55%: in between
        self.assertEqual(v.verdict().kind, UNKNOWN)

    def test_large_match_offset_is_bad(self):
        src = _Src()
        v = _verifier(src)
        v.mark_fault()
        self._feed(v, src, dtheta_rad=math.radians(30.0))
        self.assertEqual(v.verdict().kind, BAD)

    def test_sparse_matches_ignored(self):
        src = _Src()
        v = _verifier(src)
        v.mark_fault()
        self._feed(v, src, n=5, n_points=10, score_best=6.0)
        self.assertEqual(v.verdict().kind, PENDING)
        self.assertEqual(v.verdict(final=True).kind, UNKNOWN)

    def test_no_baseline_uses_fallback_quality(self):
        src = _Src()
        v = _verifier(src, baseline=None)
        v.mark_fault()
        self._feed(v, src, score_best=30.0)
        self.assertEqual(v.verdict().kind, OK)

    def test_camera_decides_when_lidar_unknown(self):
        src = _Src()
        v = _verifier(src, image=lambda: Verdict(OK, "yaw agrees", "image"))
        v.mark_fault()
        self._feed(v, src, score_best=33.0)          # lidar unknown
        out = v.verdict()
        self.assertEqual(out.kind, OK)
        self.assertEqual(out.source, "image")

    def test_camera_not_consulted_when_lidar_decides(self):
        called = []
        src = _Src()
        v = _verifier(src, image=lambda: called.append(1) or Verdict(BAD, "x", "image"))
        v.mark_fault()
        self._feed(v, src)
        self.assertEqual(v.verdict().kind, OK)
        self.assertEqual(called, [])

    def test_camera_error_is_unknown(self):
        def boom():
            raise RuntimeError("no frame")
        src = _Src()
        v = _verifier(src, image=boom)
        v.mark_fault()
        self.assertEqual(v.verdict(final=True).kind, UNKNOWN)

    def test_config_thresholds(self):
        self.assertLess(PoseVerifyConfig().bad_quality_frac, PoseVerifyConfig().ok_quality_frac)


if __name__ == "__main__":
    unittest.main()
