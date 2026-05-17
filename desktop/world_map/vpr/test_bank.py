"""Tests for VPRBank: load, cosine top-k query, mixture-observation
conversion."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from desktop.world_map.vpr.bank import (
    VPRBank,
    cosine_sim_matrix,
    mixture_observation_from_query,
)


def _synthetic_bank(n: int = 20, d: int = 16, seed: int = 0) -> VPRBank:
    """N synthetic features at known random poses. Features are random
    but L2-normalized; poses are arranged on a 2D grid."""
    gen = torch.Generator().manual_seed(seed)
    feats = torch.randn(n, d, generator=gen)
    # Lay poses out on a 4x5 grid for n=20.
    rows = int(n ** 0.5)
    cols = (n + rows - 1) // rows
    poses = torch.zeros(n, 3)
    for i in range(n):
        r, c = divmod(i, cols)
        poses[i, 0] = float(c)
        poses[i, 1] = float(r)
    return VPRBank(features=feats, poses=poses, device="cpu")


class TestVPRBankConstruction(unittest.TestCase):
    def test_features_get_renormalized(self):
        feats = torch.tensor([[3.0, 4.0], [10.0, 0.0]])
        poses = torch.zeros(2, 3)
        bank = VPRBank(features=feats, poses=poses)
        # Both rows should now be unit length.
        norms = torch.linalg.norm(bank._features, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones(2), atol=1e-5))

    def test_rejects_mismatched_rows(self):
        with self.assertRaises(ValueError):
            VPRBank(features=torch.zeros(3, 4), poses=torch.zeros(2, 3))

    def test_rejects_bad_pose_shape(self):
        with self.assertRaises(ValueError):
            VPRBank(features=torch.zeros(3, 4), poses=torch.zeros(3, 2))

    def test_n_frames_and_feature_dim(self):
        bank = _synthetic_bank(n=12, d=8)
        self.assertEqual(bank.n_frames, 12)
        self.assertEqual(bank.feature_dim, 8)


class TestQuery(unittest.TestCase):
    def test_exact_hit_returns_self(self):
        bank = _synthetic_bank(n=20, d=16, seed=1)
        # Query with a row of the bank itself → top-1 must be that row,
        # similarity 1.
        q = bank._features[7].clone()
        out = bank.query(q, top_k=1)
        self.assertEqual(int(out.indices[0].item()), 7)
        self.assertAlmostEqual(float(out.similarities[0].item()), 1.0, places=5)
        self.assertTrue(torch.allclose(out.poses[0], bank.poses[7]))

    def test_top_k_descending_by_similarity(self):
        bank = _synthetic_bank(n=20, d=16, seed=2)
        q = bank._features[3] + 0.01 * torch.randn(16)
        out = bank.query(q, top_k=5)
        sims = out.similarities.tolist()
        self.assertEqual(sims, sorted(sims, reverse=True))
        self.assertEqual(out.indices.shape, (5,))
        self.assertEqual(out.poses.shape, (5, 3))

    def test_similarity_floor_drops_low_matches(self):
        bank = _synthetic_bank(n=20, d=16, seed=3)
        # Random query mostly orthogonal → high floor rejects all.
        gen = torch.Generator().manual_seed(99)
        q = torch.randn(16, generator=gen)
        out = bank.query(q, top_k=5, similarity_floor=0.99)
        self.assertEqual(out.indices.numel(), 0)
        self.assertEqual(out.poses.shape, (0, 3))

    def test_query_clamps_top_k_at_n_frames(self):
        bank = _synthetic_bank(n=4, d=8)
        out = bank.query(bank._features[0], top_k=100)
        self.assertEqual(out.indices.shape, (4,))

    def test_query_accepts_2d_single_feature(self):
        bank = _synthetic_bank(n=10, d=8)
        q = bank._features[2].unsqueeze(0)  # (1, D)
        out = bank.query(q, top_k=1)
        self.assertEqual(int(out.indices[0].item()), 2)

    def test_query_rejects_batch(self):
        bank = _synthetic_bank(n=10, d=8)
        with self.assertRaises(ValueError):
            bank.query(bank._features[:3], top_k=1)

    def test_query_rejects_wrong_dim(self):
        bank = _synthetic_bank(n=10, d=8)
        with self.assertRaises(ValueError):
            bank.query(torch.zeros(7), top_k=1)


class TestLoadSave(unittest.TestCase):
    def test_round_trip_through_torch_save(self):
        bank = _synthetic_bank(n=8, d=12)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bank.pt"
            torch.save({
                "features": bank._features.cpu(),
                "poses": bank.poses.cpu(),
                "timestamps": torch.zeros(8, dtype=torch.float64),
                "frame_idx": torch.arange(8, dtype=torch.int64),
                "metadata": {"schema_version": 1, "extractor": {}},
            }, path)
            loaded = VPRBank.load(path)
            self.assertEqual(loaded.n_frames, 8)
            self.assertEqual(loaded.feature_dim, 12)
            self.assertEqual(loaded.metadata["schema_version"], 1)
            self.assertTrue(torch.allclose(
                loaded._features, bank._features, atol=1e-5,
            ))

    def test_load_missing_required_key_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.pt"
            torch.save({"features": torch.zeros(2, 3)}, path)
            with self.assertRaises(ValueError):
                VPRBank.load(path)


class TestMixtureConversion(unittest.TestCase):
    def test_empty_query_returns_none(self):
        bank = _synthetic_bank(n=5, d=8)
        gen = torch.Generator().manual_seed(101)
        q = torch.randn(8, generator=gen)
        empty = bank.query(q, top_k=3, similarity_floor=0.999)
        self.assertEqual(empty.indices.numel(), 0)
        self.assertIsNone(mixture_observation_from_query(empty))

    def test_single_top_match_weight_one(self):
        bank = _synthetic_bank(n=10, d=8)
        q = bank._features[4]
        out = bank.query(q, top_k=1)
        result = mixture_observation_from_query(out, sigma_m=0.4)
        self.assertIsNotNone(result)
        positions, weights, sigma = result  # type: ignore[misc]
        self.assertEqual(positions.shape, (1, 2))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=5)
        self.assertAlmostEqual(float(weights[0]), 1.0, places=5)
        self.assertEqual(sigma, 0.4)

    def test_softmax_normalizes_weights(self):
        bank = _synthetic_bank(n=10, d=8)
        q = bank._features[4] + 0.05 * torch.randn(8)
        out = bank.query(q, top_k=3)
        result = mixture_observation_from_query(out, temperature=0.05)
        self.assertIsNotNone(result)
        _, weights, _ = result  # type: ignore[misc]
        self.assertEqual(weights.shape, (3,))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=5)
        # Lower temperature concentrates weight on the top match —
        # the top-similarity weight should exceed the bottom one.
        self.assertGreater(float(weights[0]), float(weights[-1]))

    def test_higher_temperature_flattens_weights(self):
        bank = _synthetic_bank(n=10, d=8, seed=4)
        q = bank._features[2] + 0.05 * torch.randn(8)
        out = bank.query(q, top_k=4)
        r_sharp = mixture_observation_from_query(out, temperature=0.01)
        r_flat = mixture_observation_from_query(out, temperature=1.0)
        assert r_sharp is not None and r_flat is not None
        from desktop.world_map.vpr.bank import _entropy_of_weights
        h_sharp = _entropy_of_weights(r_sharp[1])
        h_flat = _entropy_of_weights(r_flat[1])
        self.assertGreater(h_flat, h_sharp)


class TestCosineSimMatrix(unittest.TestCase):
    def test_self_similarity_is_one(self):
        gen = torch.Generator().manual_seed(5)
        a = torch.randn(4, 8, generator=gen)
        sim = cosine_sim_matrix(a, a)
        self.assertEqual(sim.shape, (4, 4))
        diag = sim.diagonal()
        self.assertTrue(torch.allclose(diag, torch.ones(4), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
