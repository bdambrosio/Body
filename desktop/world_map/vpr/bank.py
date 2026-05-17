"""Feature-bank load + cosine top-k query for Phase 6 VPR.

A bank is the on-disk artifact produced by ``scripts/vpr_build_bank.py``:
N L2-normalized DINOv2 features, each tagged with the (x_m, y_m,
theta_rad) world pose of the bot when the frame was captured. At
runtime we query the bank with a feature from the current camera
frame and get the K most-similar bank entries, with their poses and
cosine similarities.

Two-step conversion to a filter observation
-------------------------------------------
1. ``VPRBank.query(feat, top_k)`` → top-K matches.
2. ``mixture_observation_from_query(query, …)`` → ``(positions_xy,
   weights, sigma_m)`` ready to feed into
   ``ParticleFilterPose.observe_xy_mixture``.

Step 2 is split out so the gating policy (similarity floor, soft-max
temperature, σ) can be tuned without touching the bank itself.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryResult:
    """Top-K bank matches for one query feature."""

    indices: torch.Tensor       # (K,) int64
    similarities: torch.Tensor  # (K,) float32 in [-1, 1]
    poses: torch.Tensor         # (K, 3) float32 [x_m, y_m, theta_rad]


class VPRBank:
    """In-memory feature bank loaded from a Phase 6.1 ``.pt`` file.

    Features live on the chosen device; query is a single matmul +
    topk, sub-millisecond for the bank sizes a single-room run
    produces (< few thousand frames). Single-query API; batch-query
    would be a trivial loop and we don't need it for the runtime
    observer (one camera frame per VPR tick).
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        features: torch.Tensor,
        poses: torch.Tensor,
        *,
        timestamps: Optional[torch.Tensor] = None,
        frame_idx: Optional[torch.Tensor] = None,
        metadata: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
    ) -> None:
        if features.ndim != 2:
            raise ValueError(
                f"features must be (N, D), got shape {tuple(features.shape)}"
            )
        if poses.ndim != 2 or poses.shape[1] != 3:
            raise ValueError(
                f"poses must be (N, 3), got shape {tuple(poses.shape)}"
            )
        if features.shape[0] != poses.shape[0]:
            raise ValueError(
                f"features ({features.shape[0]}) and poses "
                f"({poses.shape[0]}) row counts must match"
            )
        self._device = torch.device(device)
        # features stored normalized — re-normalize defensively so old
        # banks built before the L2 step in extractor are still usable.
        feats = features.to(self._device, dtype=torch.float32)
        norms = torch.linalg.norm(feats, dim=-1, keepdim=True).clamp_min(1e-12)
        self._features = feats / norms
        self._poses = poses.to(self._device, dtype=torch.float32)
        self._timestamps = (
            timestamps.to(self._device) if timestamps is not None else None
        )
        self._frame_idx = (
            frame_idx.to(self._device) if frame_idx is not None else None
        )
        self._metadata = dict(metadata) if metadata else {}

    # ── Properties ───────────────────────────────────────────────────

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def feature_dim(self) -> int:
        return int(self._features.shape[1])

    @property
    def n_frames(self) -> int:
        return int(self._features.shape[0])

    @property
    def metadata(self) -> Dict[str, Any]:
        return dict(self._metadata)

    @property
    def poses(self) -> torch.Tensor:
        return self._poses

    # ── Query ────────────────────────────────────────────────────────

    @torch.inference_mode()
    def query(
        self,
        feature: torch.Tensor,
        top_k: int = 5,
        *,
        similarity_floor: Optional[float] = None,
    ) -> QueryResult:
        """Return the top-K bank entries closest to ``feature`` by cosine
        similarity.

        Args:
            feature: (D,) or (1, D) tensor. Normalized internally so the
                caller doesn't have to.
            top_k: number of matches to return; capped at ``n_frames``.
            similarity_floor: if provided, drop matches below this
                threshold (cosine in [-1, 1]). Returned result may have
                fewer than ``top_k`` entries — including zero if no match
                clears the floor.

        Returns:
            QueryResult on the bank's device. Sorted descending by
            similarity.
        """
        if self.n_frames == 0:
            raise RuntimeError("query against an empty bank")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        q = feature.to(self._device, dtype=torch.float32)
        if q.ndim == 2:
            if q.shape[0] != 1:
                raise ValueError(
                    f"query expects a single feature, got batch shape {tuple(q.shape)}"
                )
            q = q[0]
        if q.ndim != 1 or q.shape[0] != self.feature_dim:
            raise ValueError(
                f"query feature shape mismatch: expected ({self.feature_dim},), "
                f"got {tuple(q.shape)}"
            )
        q = q / q.norm().clamp_min(1e-12)
        sims = self._features @ q  # (N,)
        k = min(int(top_k), self.n_frames)
        vals, idx = torch.topk(sims, k=k, sorted=True)
        if similarity_floor is not None:
            mask = vals >= float(similarity_floor)
            vals = vals[mask]
            idx = idx[mask]
        poses = self._poses[idx]
        return QueryResult(indices=idx, similarities=vals, poses=poses)

    # ── Load / save ─────────────────────────────────────────────────

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> "VPRBank":
        """Load a bank from a Phase 6.1 ``.pt`` file."""
        import os as _os
        path = Path(_os.path.expanduser(str(path)))
        # weights_only=False because the metadata field is a plain dict.
        raw = torch.load(path, weights_only=False, map_location="cpu")
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: bank file must be a dict, got {type(raw)}")
        for k in ("features", "poses"):
            if k not in raw:
                raise ValueError(f"{path}: bank file missing key {k!r}")
        meta = raw.get("metadata") or {}
        schema = int(meta.get("schema_version", 0))
        if schema != cls.SCHEMA_VERSION:
            logger.warning(
                "VPRBank: %s schema_version=%d (expected %d) — "
                "loading anyway; may break.",
                path, schema, cls.SCHEMA_VERSION,
            )
        return cls(
            features=raw["features"],
            poses=raw["poses"],
            timestamps=raw.get("timestamps"),
            frame_idx=raw.get("frame_idx"),
            metadata=meta,
            device=device,
        )


# ── Query → mixture-observation conversion ───────────────────────────


def mixture_observation_from_query(
    query: QueryResult,
    *,
    temperature: float = 0.05,
    sigma_m: float = 0.50,
    min_components: int = 1,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, float]]:
    """Convert a ``QueryResult`` to the inputs of
    ``ParticleFilterPose.observe_xy_mixture``.

    Args:
        query: result of ``VPRBank.query``. May be empty if every match
            was rejected by ``similarity_floor`` — returns ``None``.
        temperature: softmax temperature on cosine similarities. Smaller
            → sharper mixture weights (top match dominates). Default
            0.05 → ~exp((s_max - s) / 0.05); typical DINOv2 cosines for
            same-room views are ~0.5–0.8, so 0.05 keeps the dominant
            match clearly ahead while still letting close runners-up
            contribute.
        sigma_m: per-component Gaussian σ in meters. VPR is room-scale,
            not cm-scale; 0.5 m is a reasonable default for the soft
            anchor role 6.4's gating policy will exploit.
        min_components: refuse to emit an observation if fewer than this
            many components survived ``similarity_floor`` (default 1 —
            zero components always means no observation).

    Returns:
        ``(positions_xy: (K, 2) float32, weights: (K,) float32, sigma_m)``
        on the query's device, or ``None`` if the query was rejected.
        ``weights`` sum to 1.
    """
    n = int(query.similarities.shape[0])
    if n < max(1, min_components):
        return None
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if sigma_m <= 0:
        raise ValueError(f"sigma_m must be > 0, got {sigma_m}")

    sims = query.similarities.to(torch.float32)
    # Standard softmax in (sim / T) space. Numerically stable: subtract
    # max before exp.
    logits = sims / temperature
    weights = torch.softmax(logits, dim=0)
    positions_xy = query.poses[:, :2]  # drop theta — observation is XY only
    return positions_xy, weights, float(sigma_m)


# ── Convenience helpers (mostly for tests / interactive use) ──────────


def cosine_sim_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarity between rows of a (N, D) and b (M, D),
    handling L2 normalization. Returns (N, M) float32."""
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    a = a / torch.linalg.norm(a, dim=-1, keepdim=True).clamp_min(1e-12)
    b = b / torch.linalg.norm(b, dim=-1, keepdim=True).clamp_min(1e-12)
    return a @ b.T


def _entropy_of_weights(weights: torch.Tensor) -> float:
    """Shannon entropy in nats of a (K,) mixture-weight vector. Handy
    for tuning ``temperature`` — sharp = 0, uniform = log(K).
    """
    w = weights.clamp_min(1e-12)
    return float(-(w * w.log()).sum())


__all__ = [
    "QueryResult",
    "VPRBank",
    "mixture_observation_from_query",
    "cosine_sim_matrix",
]
