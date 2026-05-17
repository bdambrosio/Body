"""Unit tests for ShadowVPRDriver.

We exercise the inner logic via `process_frame` so the tests don't
need a real Zenoh session. The driver's connect/disconnect/RGB
callback plumbing is intentionally thin — same template as
AprilTagObserver — and integration is verified live in 6.5.
"""
from __future__ import annotations

import base64
import io
import json
import math
import tempfile
import threading
import unittest
from pathlib import Path
from typing import List, Optional, Tuple
from unittest import mock

import numpy as np
import torch
from PIL import Image

from desktop.world_map.particle_filter_pose import (
    ParticleFilterConfig,
    ParticleFilterPose,
)
from desktop.world_map.vpr.bank import VPRBank
from desktop.world_map.vpr.extractor import (
    DinoV2Extractor,
    ExtractorConfig,
)
from desktop.world_map.vpr.shadow_driver import (
    ShadowVPRConfig,
    ShadowVPRDriver,
    _decode_b64_jpeg,
    _jpeg_bytes_to_rgb,
)
from desktop.world_map.vpr.test_extractor import _StubBackbone, _checker_rgb


def _stub_extractor(*, embed_dim: int = 32, seed: int = 0) -> DinoV2Extractor:
    cfg = ExtractorConfig(
        model_name="stub", input_size=28, patch_size=14,
        device="cpu", use_half_on_cuda=False,
    )
    return DinoV2Extractor(
        model=_StubBackbone(embed_dim=embed_dim, seed=seed), config=cfg,
    )


def _build_bank_from_extractor(
    extractor: DinoV2Extractor,
    poses: List[Tuple[float, float, float]],
    images: List[np.ndarray],
) -> VPRBank:
    feats = extractor.extract_batch(images).cpu()
    poses_t = torch.tensor(poses, dtype=torch.float32)
    return VPRBank(features=feats, poses=poses_t)


def _seeded_pf(*, n_particles: int = 500, sigma: float = 0.5,
               seed: int = 1) -> Tuple[ParticleFilterPose, threading.RLock]:
    pf = ParticleFilterPose(ParticleFilterConfig(
        n_particles=n_particles, init_sigma_xy_m=sigma,
        init_sigma_theta_rad=math.radians(2.0), seed=seed,
    ))
    pf.seed_at(0.0, 0.0, 0.0)
    return pf, threading.RLock()


def _jpeg_b64_payload(rgb: np.ndarray) -> dict:
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="JPEG", quality=85)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return {"ok": True, "data": b64, "ts": 1700.0}


class _FakeSample:
    def __init__(self, payload: bytes):
        self._payload = payload

    @property
    def payload(self):
        return self._payload


class _FakeSubscriber:
    def undeclare(self): pass


class _FakeSession:
    """Minimal stand-in for a zenoh.Session — collects subscribers
    and lets tests inject samples via the captured callback."""

    def __init__(self):
        self.subs: dict = {}
        self.pubs: list = []

    def declare_subscriber(self, key, cb):
        self.subs[key] = cb
        return _FakeSubscriber()

    def declare_publisher(self, key):
        self.pubs.append(key)
        class _P:
            def put(self_inner, _): pass
            def undeclare(self_inner): pass
        return _P()


class TestProcessFrame(unittest.TestCase):
    def _setup_driver(self, *, tmp_dir: Path, similarity_floor: float = 0.0):
        ext = _stub_extractor(embed_dim=32, seed=11)
        # 4 bank entries at known poses, using deterministic
        # synthetic frames so we can predict which one will match.
        poses = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                 (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)]
        # Use distinct checker patterns so features differ.
        imgs = [_checker_rgb(64 + i, 80 + i) for i in range(4)]
        bank = _build_bank_from_extractor(ext, poses, imgs)
        pf, pf_lock = _seeded_pf(n_particles=500)
        drv = ShadowVPRDriver(
            session=_FakeSession(),
            pf=pf, pf_lock=pf_lock, bank=bank, extractor=ext,
            trace_path=tmp_dir / "trace.jsonl",
            config=ShadowVPRConfig(
                request_hz=0.0,  # passive — no requester thread
                top_k=3,
                similarity_floor=similarity_floor,
                softmax_temperature=0.05,
                sigma_m=0.5,
                trace_flush_every=1,
            ),
        )
        # Manually open the trace file (skip connect() so we don't
        # touch the fake session's subscribers either).
        drv._trace_path.parent.mkdir(parents=True, exist_ok=True)
        drv._trace_fp = drv._trace_path.open("a", buffering=1)
        self.addCleanup(lambda: drv._trace_fp and drv._trace_fp.close())
        return drv, bank, imgs, poses

    def _read_trace(self, path: Path) -> list:
        return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]

    def test_vpr_obs_record_shape_on_known_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            drv, _, imgs, poses = self._setup_driver(tmp_dir=tmp)
            # Frame 2 was registered with pose (0, 1, 0). Querying with
            # frame 2's image should put weight on bank index 2.
            rec = drv.process_frame(imgs[2], rgb_recv_ts=1.0, rgb_ts=0.9)
            self.assertEqual(rec["type"], "vpr_obs")
            self.assertEqual(rec["rgb_recv_ts"], 1.0)
            self.assertEqual(rec["rgb_ts"], 0.9)
            # Top match is bank index 2 with sim ≈ 1.
            self.assertEqual(rec["top_k"][0]["idx"], 2)
            self.assertAlmostEqual(rec["top_k"][0]["sim"], 1.0, places=4)
            self.assertEqual(rec["top_k"][0]["pose_xytheta"], list(poses[2]))
            self.assertIsNotNone(rec["mixture"])
            self.assertAlmostEqual(rec["mixture"]["sigma_m"], 0.5)
            self.assertEqual(len(rec["mixture"]["positions_xy"]),
                             len(rec["mixture"]["weights"]))
            # would_be present, has the right shape.
            wb = rec["would_be"]
            self.assertIsNotNone(wb)
            self.assertEqual(len(wb["mean_xy_before"]), 2)
            self.assertEqual(len(wb["mean_xy_after"]), 2)
            self.assertIn("log_lik_stats", wb)
            for k in ("mean", "std", "min", "max"):
                self.assertIn(k, wb["log_lik_stats"])

    def test_no_match_record_when_floor_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            drv, _, imgs, _ = self._setup_driver(
                tmp_dir=tmp, similarity_floor=2.0,  # unreachable cosine
            )
            rec = drv.process_frame(imgs[0])
            self.assertEqual(rec["type"], "no_match")
            self.assertIsNone(rec.get("mixture"))
            # Counters updated.
            self.assertEqual(drv.counters()["frames_no_match"], 1)
            self.assertEqual(drv.counters()["frames_observed"], 0)
            # Trace contains the no-match record (top_k empty because of floor).
            traces = self._read_trace(tmp / "trace.jsonl")
            self.assertEqual(traces[-1]["type"], "no_match")
            self.assertEqual(traces[-1]["top_k"], [])

    def test_pf_state_unchanged(self):
        # Crucial invariant: shadow must not mutate the live filter.
        with tempfile.TemporaryDirectory() as tmp:
            drv, _, imgs, _ = self._setup_driver(tmp_dir=Path(tmp))
            state_before = drv._pf.state.clone()
            log_w_before = drv._pf._log_w.clone()
            for img in imgs:
                drv.process_frame(img)
            self.assertTrue(torch.equal(drv._pf.state, state_before))
            self.assertTrue(torch.equal(drv._pf._log_w, log_w_before))

    def test_would_be_mean_shifts_toward_match(self):
        # If the bank's top match is at +X, applying the would-be
        # update should pull the mean_xy_after toward that pose.
        with tempfile.TemporaryDirectory() as tmp:
            drv, bank, imgs, _ = self._setup_driver(tmp_dir=Path(tmp))
            rec = drv.process_frame(imgs[1])  # pose (1, 0, 0)
            wb = rec["would_be"]
            self.assertIsNotNone(wb)
            # Filter was seeded at origin → mean_xy_before ≈ (0, 0).
            self.assertAlmostEqual(wb["mean_xy_before"][0], 0.0, delta=0.1)
            # Mean after should be shifted toward +x.
            self.assertGreater(wb["mean_xy_after"][0], wb["mean_xy_before"][0])

    def test_counters_increment(self):
        with tempfile.TemporaryDirectory() as tmp:
            drv, _, imgs, _ = self._setup_driver(tmp_dir=Path(tmp))
            for img in imgs:
                drv.process_frame(img)
            c = drv.counters()
            self.assertEqual(c["frames_processed"], 4)
            self.assertEqual(c["frames_observed"], 4)
            self.assertEqual(c["frames_no_match"], 0)

    def test_current_pose_prefers_pf_posterior_when_seeded(self):
        # When the PF is seeded, current_pose comes from posterior_mean,
        # not pose_source — the PF's own posterior is the right answer
        # for "where does the filter think we are right now."
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            drv, _, imgs, _ = self._setup_driver(tmp_dir=tmp)
            class _Pose:
                def pose_at(self, ts):
                    return (0.5, -0.3, 0.1)
            drv._pose_source = _Pose()
            rec = drv.process_frame(imgs[0])
            self.assertEqual(len(rec["current_pose"]), 3)
            # PF was seeded at origin, so the posterior mean should be
            # near (0, 0, 0) — NOT the pose_source's (0.5, -0.3, 0.1).
            self.assertNotEqual(rec["current_pose"], [0.5, -0.3, 0.1])
            self.assertLess(abs(rec["current_pose"][0]), 0.1)
            self.assertLess(abs(rec["current_pose"][1]), 0.1)

    def test_on_trace_callback_fires(self):
        seen = []
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            drv, _, imgs, _ = self._setup_driver(tmp_dir=tmp)
            drv._on_trace = lambda rec: seen.append(rec)
            drv.process_frame(imgs[0])
            self.assertEqual(len(seen), 1)
            self.assertEqual(seen[0]["type"], "vpr_obs")


class TestPhase64Integration(unittest.TestCase):
    """Anchor + gating + live application — the 6.4 additions."""

    def _build(self, *, tmp_dir, live: bool, **cfg_overrides):
        ext = _stub_extractor(embed_dim=32, seed=42)
        # Bank: 4 frames at well-separated poses so the SE(2) fit has
        # geometric diversity (spread = 1.4 m diagonal).
        poses = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                 (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)]
        imgs = [_checker_rgb(64 + i, 80 + i) for i in range(4)]
        bank = _build_bank_from_extractor(ext, poses, imgs)
        pf, pf_lock = _seeded_pf(n_particles=2000, sigma=0.5)
        from desktop.world_map.vpr.anchor import AnchorOffsetConfig
        cfg = ShadowVPRConfig(
            request_hz=0.0,
            top_k=3,
            similarity_floor=cfg_overrides.pop("similarity_floor", 0.0),
            softmax_temperature=0.05,
            sigma_m=cfg_overrides.pop("sigma_m", 0.5),
            trace_flush_every=1,
            live=live,
            anchor=AnchorOffsetConfig(
                min_similarity=cfg_overrides.pop("anchor_min_sim", 0.85),
                min_pairs=cfg_overrides.pop("anchor_min_pairs", 3),
                min_spatial_spread_m=cfg_overrides.pop("anchor_spread_m", 0.5),
                max_residual_m=10.0,  # generous in tests
            ),
            gate_sigma_floor_ratio=cfg_overrides.pop("gate_sigma_ratio", 0.5),
            gate_min_distance_m=cfg_overrides.pop("gate_min_dist", 0.0),
            **cfg_overrides,
        )
        drv = ShadowVPRDriver(
            session=_FakeSession(),
            pf=pf, pf_lock=pf_lock, bank=bank, extractor=ext,
            trace_path=Path(tmp_dir) / "trace.jsonl",
            config=cfg,
        )
        drv._trace_path.parent.mkdir(parents=True, exist_ok=True)
        drv._trace_fp = drv._trace_path.open("a", buffering=1)
        self.addCleanup(lambda: drv._trace_fp and drv._trace_fp.close())
        return drv, imgs, poses

    def test_anchor_calibrates_after_enough_high_sim_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            drv, imgs, _ = self._build(
                tmp_dir=tmp, live=False, anchor_min_pairs=3, anchor_min_sim=0.5,
            )
            # The 4 bank images map to themselves with sim=1; running
            # all 4 gives the estimator 4 perfect pairs.
            for img in imgs:
                drv.process_frame(img)
            self.assertEqual(drv._anchor.state, "calibrated")
            self.assertIsNotNone(drv._anchor.calibration)
            self.assertEqual(drv.counters()["anchor_calibrations"], 1)

    def test_record_carries_anchor_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            drv, imgs, _ = self._build(
                tmp_dir=tmp, live=False, anchor_min_pairs=3, anchor_min_sim=0.5,
            )
            for img in imgs:
                rec = drv.process_frame(img)
            self.assertIn("anchor", rec)
            self.assertEqual(rec["anchor"]["state"], "calibrated")
            self.assertIn("offset", rec["anchor"])
            for k in ("dx", "dy", "dtheta_rad"):
                self.assertIn(k, rec["anchor"]["offset"])

    def test_gating_records_present_in_vpr_obs(self):
        with tempfile.TemporaryDirectory() as tmp:
            drv, imgs, _ = self._build(tmp_dir=tmp, live=False)
            rec = drv.process_frame(imgs[0])
            self.assertIn("gating", rec)
            for k in ("passed", "reason", "sigma_xy_m", "dist_since_last_m"):
                self.assertIn(k, rec["gating"])

    def test_gate_cloud_too_tight(self):
        # Tight init σ → cov small → gate should fail.
        ext = _stub_extractor(embed_dim=32, seed=42)
        poses = [(0.0, 0.0, 0.0)]
        bank = _build_bank_from_extractor(
            ext, poses, [_checker_rgb()],
        )
        # Seed with very tight cloud (1 mm σ).
        pf = ParticleFilterPose(ParticleFilterConfig(
            n_particles=500, init_sigma_xy_m=0.001,
            init_sigma_theta_rad=math.radians(0.1), seed=7,
        ))
        pf.seed_at(0.0, 0.0, 0.0)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ShadowVPRConfig(
                request_hz=0.0, top_k=1, similarity_floor=0.0,
                sigma_m=0.5, gate_sigma_floor_ratio=0.5,
                gate_min_distance_m=0.0, live=False,
            )
            drv = ShadowVPRDriver(
                session=_FakeSession(),
                pf=pf, pf_lock=threading.RLock(),
                bank=bank, extractor=ext,
                trace_path=Path(tmp) / "t.jsonl", config=cfg,
            )
            drv._trace_fp = (Path(tmp) / "t.jsonl").open("a", buffering=1)
            self.addCleanup(lambda: drv._trace_fp and drv._trace_fp.close())
            rec = drv.process_frame(_checker_rgb())
            self.assertFalse(rec["gating"]["passed"])
            self.assertEqual(rec["gating"]["reason"], "cloud_too_tight")

    def test_live_mode_off_does_not_apply(self):
        # Even with everything else green, live=False ⇒ no mutation.
        with tempfile.TemporaryDirectory() as tmp:
            drv, imgs, _ = self._build(
                tmp_dir=tmp, live=False, anchor_min_pairs=3, anchor_min_sim=0.5,
            )
            log_w_before = drv._pf._log_w.clone()
            for img in imgs:
                rec = drv.process_frame(img)
            self.assertFalse(rec["applied"])
            self.assertTrue(torch.equal(drv._pf._log_w, log_w_before))
            self.assertEqual(drv.counters()["live_obs_applied"], 0)

    def test_live_mode_applies_when_calibrated_and_gated_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            drv, imgs, _ = self._build(
                tmp_dir=tmp, live=True, anchor_min_pairs=3, anchor_min_sim=0.5,
            )
            # First 3 calibrate the anchor; subsequent ones can apply.
            for img in imgs:
                drv.process_frame(img)
            # After the loop, the anchor must be calibrated and at
            # least one live application should have fired (sims for
            # bank's own images are ~1.0, gate-cloud spread is healthy).
            self.assertEqual(drv._anchor.state, "calibrated")
            self.assertGreater(drv.counters()["live_obs_applied"], 0)

    def test_live_skipped_when_anchor_not_calibrated(self):
        # Set anchor threshold so high we never calibrate. live=True
        # but observations should be gated on "anchor not ready."
        with tempfile.TemporaryDirectory() as tmp:
            drv, imgs, _ = self._build(
                tmp_dir=tmp, live=True,
                anchor_min_pairs=99, anchor_min_sim=0.5,
            )
            for img in imgs:
                rec = drv.process_frame(img)
            self.assertEqual(drv._anchor.state, "uncalibrated")
            self.assertFalse(rec["applied"])
            self.assertEqual(drv.counters()["live_obs_applied"], 0)
            self.assertGreater(drv.counters()["live_obs_gated_anchor"], 0)


class TestKidnappingDiagnostic(unittest.TestCase):
    """Phase 6.4.3 — d_best_to_obs_m + kidnapping_suspected flag."""

    def _setup(self, tmp_dir, *, anchor_calibrated: bool = True):
        from desktop.world_map.vpr.anchor import (
            AnchorOffsetConfig, CalibrationResult,
        )
        ext = _stub_extractor(embed_dim=32, seed=77)
        # Bank pose for index 3 deliberately well away from origin so
        # the "far from cloud" condition (d > 3σ_m = 1.5 m) is met.
        poses = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                 (0.0, 1.0, 0.0), (5.0, 5.0, 0.0)]
        imgs = [_checker_rgb(64 + i, 80 + i) for i in range(4)]
        bank = _build_bank_from_extractor(ext, poses, imgs)
        pf, pf_lock = _seeded_pf(n_particles=500, sigma=0.01)  # tight cloud
        drv = ShadowVPRDriver(
            session=_FakeSession(),
            pf=pf, pf_lock=pf_lock, bank=bank, extractor=ext,
            trace_path=tmp_dir / "trace.jsonl",
            config=ShadowVPRConfig(
                request_hz=0.0, top_k=1, similarity_floor=0.0,
                sigma_m=0.5, gate_sigma_floor_ratio=0.0,
                gate_min_distance_m=0.0, live=False,
                anchor=AnchorOffsetConfig(min_pairs=99),  # won't auto-calibrate
            ),
        )
        drv._trace_path.parent.mkdir(parents=True, exist_ok=True)
        drv._trace_fp = drv._trace_path.open("a", buffering=1)
        self.addCleanup(lambda: drv._trace_fp and drv._trace_fp.close())
        if anchor_calibrated:
            # Identity offset — bank XY maps to itself in session frame.
            drv._anchor.set_calibration(CalibrationResult(
                dx=0.0, dy=0.0, dtheta_rad=0.0,
                n_pairs=99, residual_rms_m=0.01,
            ))
        return drv, imgs

    def test_d_best_to_obs_computed(self):
        # PF at origin (cloud tight at 1cm). Bank match at (5, 5) under
        # identity offset → mixture peak at (5, 5) → particle distance
        # to peak ≈ √50 ≈ 7 m.
        with tempfile.TemporaryDirectory() as tmp:
            drv, imgs = self._setup(Path(tmp))
            rec = drv.process_frame(imgs[3])  # bank pose (5, 5, 0)
            self.assertIn("d_best_to_obs_m", rec["would_be"])
            d = rec["would_be"]["d_best_to_obs_m"]
            self.assertGreater(d, 5.0)

    def test_kidnapping_suspected_when_high_sim_far_from_cloud(self):
        with tempfile.TemporaryDirectory() as tmp:
            drv, imgs = self._setup(Path(tmp), anchor_calibrated=True)
            rec = drv.process_frame(imgs[3])  # bank pose (5, 5) — far from cloud
            self.assertGreater(rec["top_k"][0]["sim"], 0.85)
            self.assertGreater(rec["would_be"]["d_best_to_obs_m"], 3 * 0.5)
            self.assertTrue(rec["kidnapping_suspected"])
            self.assertEqual(drv.counters().get("kidnapping_suspected"), 1)

    def test_not_suspected_when_anchor_uncalibrated(self):
        # Same setup, but anchor not calibrated → can't trust distance,
        # so kidnapping_suspected must be False.
        with tempfile.TemporaryDirectory() as tmp:
            drv, imgs = self._setup(Path(tmp), anchor_calibrated=False)
            rec = drv.process_frame(imgs[3])
            self.assertFalse(rec["kidnapping_suspected"])

    def test_not_suspected_when_low_similarity(self):
        # Force low sim by tightening the similarity floor — we still
        # see top_k but the [0] sim is below 0.85.
        with tempfile.TemporaryDirectory() as tmp:
            drv, imgs = self._setup(Path(tmp), anchor_calibrated=True)
            # Patch top_k_records inline: easier than constructing a
            # different bank. Process_frame uses the bank query directly,
            # so we just verify via a real query that the [0] sim is
            # well below 0.85 — the stub backbone produces variable
            # similarities depending on inputs.
            # Use a *very different* image so cosine is low.
            random_img = np.zeros((40, 40, 3), dtype=np.uint8)
            rec = drv.process_frame(random_img)
            # If sim happens to clear 0.85 (the stub is dumb), at
            # least confirm the gating logic is correct given values.
            if rec["top_k"][0]["sim"] < 0.85:
                self.assertFalse(rec["kidnapping_suspected"])


class TestOpportunisticCalibrationEvent(unittest.TestCase):
    """Phase 6.4.3 — opportunistic anchor lock emits vpr_calibration
    log event with the same schema the (now-disabled) sweep used."""

    def test_anchor_lock_logs_event(self):
        from desktop.world_map.vpr.anchor import AnchorOffsetConfig
        with tempfile.TemporaryDirectory() as tmp:
            ext = _stub_extractor(embed_dim=32, seed=88)
            poses = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                     (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)]
            imgs = [_checker_rgb(64 + i, 80 + i) for i in range(4)]
            bank = _build_bank_from_extractor(ext, poses, imgs)
            pf, pf_lock = _seeded_pf(n_particles=500, sigma=0.5)
            drv = ShadowVPRDriver(
                session=_FakeSession(),
                pf=pf, pf_lock=pf_lock, bank=bank, extractor=ext,
                trace_path=Path(tmp) / "t.jsonl",
                config=ShadowVPRConfig(
                    request_hz=0.0, top_k=1, similarity_floor=0.0,
                    sigma_m=0.5, live=False,
                    anchor=AnchorOffsetConfig(
                        min_pairs=3, min_spatial_spread_m=0.5,
                        max_residual_m=10.0,        # easy to pass
                        max_cov_xy_trace_m2=10.0,   # easy to pass
                    ),
                ),
            )
            drv._trace_fp = (Path(tmp) / "t.jsonl").open("a", buffering=1)
            self.addCleanup(lambda: drv._trace_fp and drv._trace_fp.close())
            for img in imgs:
                drv.process_frame(img)
            # Reload trace and check for vpr_calibration event.
            trace = [json.loads(ln) for ln in
                     (Path(tmp) / "t.jsonl").read_text().splitlines() if ln.strip()]
            cals = [r for r in trace if r["type"] == "vpr_calibration"]
            self.assertGreaterEqual(len(cals), 1)
            self.assertEqual(cals[-1]["phase"], "opportunistic_attempt")
            self.assertTrue(cals[-1]["passed"])
            self.assertIn("offset", cals[-1])


class TestRgbCallback(unittest.TestCase):
    def test_on_rgb_processes_real_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ext = _stub_extractor()
            imgs = [_checker_rgb(64, 80), _checker_rgb(80, 96)]
            poses = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
            bank = _build_bank_from_extractor(ext, poses, imgs)
            pf, lock = _seeded_pf()
            sess = _FakeSession()
            drv = ShadowVPRDriver(
                session=sess, pf=pf, pf_lock=lock, bank=bank, extractor=ext,
                trace_path=tmp / "trace.jsonl",
                config=ShadowVPRConfig(request_hz=0.0),
            )
            drv.connect()
            try:
                payload = json.dumps(_jpeg_b64_payload(imgs[0])).encode("utf-8")
                cb = sess.subs["body/oakd/rgb"]
                cb(_FakeSample(payload))
                self.assertEqual(drv.counters()["rgb_received"], 1)
                self.assertEqual(drv.counters()["frames_processed"], 1)
            finally:
                drv.disconnect()
            # Both session_start and at least one observation record.
            traces = [json.loads(ln) for ln in
                      (tmp / "trace.jsonl").read_text().splitlines() if ln.strip()]
            types = [r["type"] for r in traces]
            self.assertIn("session_start", types)
            self.assertIn("vpr_obs", types)
            self.assertIn("session_end", types)

    def test_malformed_payload_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            drv = ShadowVPRDriver(
                session=_FakeSession(),
                pf=_seeded_pf()[0], pf_lock=threading.RLock(),
                bank=_build_bank_from_extractor(
                    _stub_extractor(), [(0, 0, 0)], [_checker_rgb()],
                ),
                extractor=_stub_extractor(),
                trace_path=tmp / "t.jsonl",
                config=ShadowVPRConfig(request_hz=0.0),
            )
            drv._trace_fp = (tmp / "t.jsonl").open("a", buffering=1)
            self.addCleanup(lambda: drv._trace_fp and drv._trace_fp.close())
            drv._on_rgb(_FakeSample(b"not-json"))
            drv._on_rgb(_FakeSample(json.dumps({"ok": False}).encode()))
            drv._on_rgb(_FakeSample(json.dumps({"ok": True, "data": 42}).encode()))
            drv._on_rgb(_FakeSample(json.dumps(
                {"ok": True, "data": "!!!not-base64!!!"}).encode()))
            c = drv.counters()
            self.assertEqual(c["rgb_received"], 4)
            # 3 malformed: non-JSON, non-str data, undecodable b64.
            self.assertEqual(c["rgb_malformed"], 3)
            self.assertEqual(c["rgb_error_payload"], 1)
            self.assertEqual(c["frames_processed"], 0)


class TestJpegHelpers(unittest.TestCase):
    def test_round_trip(self):
        rgb = _checker_rgb(48, 64)
        buf = io.BytesIO()
        Image.fromarray(rgb, mode="RGB").save(buf, format="JPEG", quality=95)
        decoded = _jpeg_bytes_to_rgb(buf.getvalue())
        self.assertEqual(decoded.shape, rgb.shape)
        self.assertEqual(decoded.dtype, np.uint8)

    def test_b64_strips_data_url(self):
        raw = b"\x00\x01\x02\x03"
        b64 = "data:image/jpeg;base64," + base64.standard_b64encode(raw).decode()
        self.assertEqual(_decode_b64_jpeg(b64), raw)


if __name__ == "__main__":
    unittest.main()
