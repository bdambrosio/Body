"""Unit tests for the DINOv2 extractor wrapper.

Tests use a small stub backbone (avg-pool + linear) so they run
without downloading DINOv2 weights. The wrapper logic (preprocess,
device, normalize, batching, half) is what we want to verify; the
real model is exercised by integration in 6.2+.
"""
from __future__ import annotations

import unittest

import numpy as np
import torch

from desktop.world_map.vpr.extractor import (
    DinoV2Extractor,
    ExtractorConfig,
    _to_rgb_uint8,
    _resize_rgb,
)


class _StubBackbone(torch.nn.Module):
    """Tiny stand-in for DINOv2. (B, 3, S, S) → (B, embed_dim).

    Average-pools then linearly projects. Deterministic given a seed,
    so two extractor instances built from identical state_dicts give
    identical features.
    """

    def __init__(self, embed_dim: int = 32, seed: int = 0):
        super().__init__()
        self.embed_dim = embed_dim
        torch.manual_seed(seed)
        # Take per-channel mean (3 features) and lift to embed_dim.
        self.proj = torch.nn.Linear(3, embed_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) — average across spatial dims.
        pooled = x.mean(dim=(-1, -2))  # (B, 3)
        return self.proj(pooled)


def _make_extractor(
    *, device: str = "cpu", input_size: int = 28, embed_dim: int = 32,
    seed: int = 0, use_half_on_cuda: bool = True,
) -> DinoV2Extractor:
    cfg = ExtractorConfig(
        model_name="stub",
        input_size=input_size,
        patch_size=14,  # 28 % 14 == 0
        device=device,
        use_half_on_cuda=use_half_on_cuda,
    )
    return DinoV2Extractor(
        model=_StubBackbone(embed_dim=embed_dim, seed=seed), config=cfg,
    )


def _checker_rgb(h: int = 64, w: int = 80) -> np.ndarray:
    """A deterministic non-trivial RGB image."""
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    r = ((yy + xx) % 256).astype(np.uint8)
    g = ((yy * 2) % 256).astype(np.uint8)
    b = ((xx * 3) % 256).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


class TestConfigValidation(unittest.TestCase):
    def test_rejects_non_multiple_input_size(self):
        with self.assertRaises(ValueError):
            ExtractorConfig(input_size=518 + 1, patch_size=14)


class TestExtractorShape(unittest.TestCase):
    def test_extract_returns_1d_normalized(self):
        ext = _make_extractor(embed_dim=32)
        feat = ext.extract(_checker_rgb())
        self.assertEqual(feat.shape, (32,))
        self.assertAlmostEqual(
            float(torch.linalg.norm(feat).item()), 1.0, places=5,
        )

    def test_feature_dim_property(self):
        ext = _make_extractor(embed_dim=48)
        self.assertEqual(ext.feature_dim, 48)


class TestDeterminism(unittest.TestCase):
    def test_same_image_same_feature(self):
        ext = _make_extractor()
        img = _checker_rgb()
        f1 = ext.extract(img.copy())
        f2 = ext.extract(img.copy())
        self.assertTrue(torch.allclose(f1, f2, atol=1e-6))

    def test_two_extractors_same_seed_match(self):
        a = _make_extractor(seed=42)
        b = _make_extractor(seed=42)
        # Force same weights via state_dict copy (separate construction
        # would re-seed but other code paths might consume the RNG
        # between them — copying is the unambiguous check).
        b._model.load_state_dict(a._model.state_dict())
        img = _checker_rgb()
        self.assertTrue(torch.allclose(a.extract(img), b.extract(img), atol=1e-6))


class TestBatching(unittest.TestCase):
    def test_batch_matches_per_image(self):
        ext = _make_extractor()
        img1 = _checker_rgb(64, 80)
        img2 = _checker_rgb(40, 50)
        batch = ext.extract_batch([img1, img2])
        self.assertEqual(batch.shape, (2, 32))
        # Each row L2-normalized.
        norms = torch.linalg.norm(batch, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones(2), atol=1e-5))
        # Match individual calls.
        self.assertTrue(torch.allclose(batch[0], ext.extract(img1), atol=1e-6))
        self.assertTrue(torch.allclose(batch[1], ext.extract(img2), atol=1e-6))

    def test_empty_batch_rejected(self):
        ext = _make_extractor()
        with self.assertRaises(ValueError):
            ext.extract_batch([])


class TestImageInputs(unittest.TestCase):
    def test_pil_image_matches_numpy(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("PIL not installed")
        ext = _make_extractor()
        arr = _checker_rgb()
        pil = Image.fromarray(arr, mode="RGB")
        self.assertTrue(torch.allclose(ext.extract(arr), ext.extract(pil), atol=1e-6))

    def test_rejects_wrong_dtype(self):
        ext = _make_extractor()
        bad = _checker_rgb().astype(np.float32) / 255.0
        with self.assertRaises(TypeError):
            ext.extract(bad)

    def test_rejects_wrong_shape(self):
        ext = _make_extractor()
        gray = np.zeros((32, 32), dtype=np.uint8)
        with self.assertRaises(ValueError):
            ext.extract(gray)


class TestPreprocessHelpers(unittest.TestCase):
    def test_resize_passthrough_when_size_matches(self):
        img = _checker_rgb(28, 28)
        out = _resize_rgb(img, 28, 28)
        # Identity → same data (may or may not be same object, but
        # bytes must match).
        self.assertTrue(np.array_equal(out, img))

    def test_resize_changes_size(self):
        img = _checker_rgb(64, 80)
        out = _resize_rgb(img, 28, 28)
        self.assertEqual(out.shape, (28, 28, 3))
        self.assertEqual(out.dtype, np.uint8)

    def test_to_rgb_rejects_unknown_type(self):
        with self.assertRaises(TypeError):
            _to_rgb_uint8("not-an-image")  # type: ignore[arg-type]


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestCudaParity(unittest.TestCase):
    def _make_pair(self, *, use_half_on_cuda: bool):
        cpu = _make_extractor(device="cpu")
        gpu = _make_extractor(
            device="cuda", use_half_on_cuda=use_half_on_cuda,
        )
        # Force identical weights.
        gpu._model.load_state_dict(
            {k: v.cuda() for k, v in cpu._model.state_dict().items()},
        )
        if use_half_on_cuda:
            gpu._model = gpu._model.half()
        return cpu, gpu

    def test_cpu_fp32_vs_gpu_fp32(self):
        cpu, gpu = self._make_pair(use_half_on_cuda=False)
        img = _checker_rgb()
        f_cpu = cpu.extract(img)
        f_gpu = gpu.extract(img).cpu()
        self.assertTrue(torch.allclose(f_cpu, f_gpu, atol=1e-5))

    def test_cpu_fp32_vs_gpu_fp16(self):
        cpu, gpu = self._make_pair(use_half_on_cuda=True)
        img = _checker_rgb()
        f_cpu = cpu.extract(img)
        f_gpu = gpu.extract(img).cpu()
        # fp16 → looser tolerance.
        self.assertTrue(torch.allclose(f_cpu, f_gpu, atol=2e-3))


if __name__ == "__main__":
    unittest.main()
