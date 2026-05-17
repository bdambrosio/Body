"""End-to-end smoke test for vpr_build_bank.

Creates a tiny synthetic session dir (3 JPEGs + frames.jsonl),
swaps in a stub extractor (so no torch.hub download), runs the
build, and inspects the resulting .pt bank file.
"""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

from desktop.world_map.vpr.extractor import DinoV2Extractor, ExtractorConfig
from desktop.world_map.vpr.test_extractor import _StubBackbone, _checker_rgb


def _make_session(session_dir: Path, n: int = 3) -> None:
    rgb_dir = session_dir / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    frames_path = session_dir / "frames.jsonl"
    with frames_path.open("w") as f:
        for i in range(n):
            arr = _checker_rgb(64 + i, 80 + i)
            jpeg_name = f"{i:06d}.jpg"
            Image.fromarray(arr, mode="RGB").save(
                rgb_dir / jpeg_name, format="JPEG", quality=85,
            )
            f.write(json.dumps({
                "idx": i,
                "rgb_ts": 1_700_000_000.0 + i,
                "pose_ts": 1_700_000_000.0 + i,
                "pose_age_s": 0.0,
                "pose_world": {
                    "x_m": 0.1 * i, "y_m": 0.2 * i, "theta_rad": 0.0,
                },
                "pose_source": "particle",
                "jpeg": f"rgb/{jpeg_name}",
            }) + "\n")


class TestBuildBank(unittest.TestCase):
    def test_end_to_end(self):
        # Stub extractor: ExtractorConfig with stub backbone.
        cfg = ExtractorConfig(
            model_name="stub",
            input_size=28,
            patch_size=14,
            device="cpu",
            use_half_on_cuda=False,
        )
        stub = DinoV2Extractor(
            model=_StubBackbone(embed_dim=32, seed=7),
            config=cfg,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            session = tmp / "session"
            _make_session(session, n=3)
            bank_path = tmp / "bank.pt"

            with mock.patch(
                "scripts.vpr_build_bank.load_default_extractor",
                return_value=stub,
            ):
                # Import here so the mock patches before main() resolves.
                from scripts import vpr_build_bank
                rc = vpr_build_bank.main([
                    str(session),
                    "--out", str(bank_path),
                    "--device", "cpu",
                    "--input-size", "28",
                    "--batch-size", "2",
                ])
            self.assertEqual(rc, 0)
            self.assertTrue(bank_path.exists())

            # weights_only=False because metadata is a plain dict.
            bank = torch.load(bank_path, weights_only=False)
            self.assertEqual(bank["features"].shape, (3, 32))
            self.assertEqual(bank["poses"].shape, (3, 3))
            self.assertEqual(bank["timestamps"].shape, (3,))
            self.assertEqual(bank["frame_idx"].shape, (3,))
            # L2-normalized.
            norms = torch.linalg.norm(bank["features"], dim=-1)
            self.assertTrue(torch.allclose(norms, torch.ones(3), atol=1e-5))
            # Pose round-trip.
            self.assertAlmostEqual(float(bank["poses"][2, 0]), 0.2, places=5)
            self.assertAlmostEqual(float(bank["poses"][2, 1]), 0.4, places=5)
            meta = bank["metadata"]
            self.assertEqual(meta["schema_version"], 1)
            self.assertEqual(meta["n_frames_total"], 3)
            self.assertEqual(meta["n_frames_extracted"], 3)
            self.assertEqual(meta["extractor"]["feature_dim"], 32)

    def test_missing_jpeg_skipped(self):
        cfg = ExtractorConfig(
            model_name="stub", input_size=28, patch_size=14, device="cpu",
            use_half_on_cuda=False,
        )
        stub = DinoV2Extractor(model=_StubBackbone(embed_dim=16), config=cfg)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            session = tmp / "s"
            _make_session(session, n=2)
            # Delete one JPEG to simulate a torn write.
            (session / "rgb" / "000001.jpg").unlink()
            bank_path = tmp / "b.pt"
            with mock.patch(
                "scripts.vpr_build_bank.load_default_extractor",
                return_value=stub,
            ):
                from scripts import vpr_build_bank
                rc = vpr_build_bank.main([
                    str(session), "--out", str(bank_path),
                    "--device", "cpu", "--input-size", "28",
                ])
            self.assertEqual(rc, 0)
            bank = torch.load(bank_path, weights_only=False)
            self.assertEqual(bank["features"].shape, (1, 16))
            self.assertEqual(bank["metadata"]["n_frames_total"], 2)
            self.assertEqual(bank["metadata"]["n_frames_extracted"], 1)


if __name__ == "__main__":
    unittest.main()
