"""Radius-limited checkpoint match — the runtime "fast pose" primitive.

Given the reference map's occupancy, a set of LPR checkpoints, an odom prior,
and a live scan: pick the checkpoint(s) near the prior, slice the healed
occupancy patch around each (so only the certified, locally-correct region is
matched — the distorted far field is excluded), and search a small pose window
with the occlusion-aware ray-cast scorer (``raycast_match``). Accept the best
match whose inlier fraction clears a gate.

This is what re-anchors the dead-reckoned pose at a node (Direction A) and what
``.nav`` Re-localize becomes (Direction B); see
docs/topological_localization_design.md §6 / Phase 3. Pure: numpy only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence, Tuple

import numpy as np

from desktop.localization.checkpoints import Checkpoint
from desktop.localization.raycast_match import (
    RaycastConfig,
    best_pose_in_window,
)

Pose = Tuple[float, float, float]


def crop_disk(
    occupied: np.ndarray,
    origin_x_m: float,
    origin_y_m: float,
    resolution_m: float,
    center_xy: Tuple[float, float],
    radius_m: float,
) -> Tuple[np.ndarray, float, float]:
    """Crop `occupied` to the bbox around `center_xy` ± `radius_m` and zero
    everything outside the disk. Returns (sub_occupied, sub_origin_x,
    sub_origin_y) — the matchable patch, so rays can't reach occupied cells
    beyond the healed radius. Empty (0,0) grid if the disk is off-map."""
    nx, ny = occupied.shape
    cx, cy = float(center_xy[0]), float(center_xy[1])
    r_cells = int(math.ceil(radius_m / resolution_m)) + 1
    ci = int(math.floor((cx - origin_x_m) / resolution_m))
    cj = int(math.floor((cy - origin_y_m) / resolution_m))
    i0, i1 = max(0, ci - r_cells), min(nx, ci + r_cells + 1)
    j0, j1 = max(0, cj - r_cells), min(ny, cj + r_cells + 1)
    if i0 >= i1 or j0 >= j1:
        return np.zeros((0, 0), dtype=bool), origin_x_m, origin_y_m
    sub = np.array(occupied[i0:i1, j0:j1], dtype=bool)
    sub_ox = origin_x_m + i0 * resolution_m
    sub_oy = origin_y_m + j0 * resolution_m
    ii = np.arange(i0, i1)[:, None]
    jj = np.arange(j0, j1)[None, :]
    wx = origin_x_m + (ii + 0.5) * resolution_m
    wy = origin_y_m + (jj + 0.5) * resolution_m
    sub &= (wx - cx) ** 2 + (wy - cy) ** 2 <= radius_m ** 2
    return sub, sub_ox, sub_oy


@dataclass(frozen=True)
class CheckpointMatchConfig:
    select_radius_m: float = 1.5          # only test checkpoints within this of the prior
    xy_half_m: float = 0.30               # pose-search window (covers odom drift since last anchor)
    xy_step_m: float = 0.05
    theta_half_rad: float = math.radians(15.0)
    theta_step_rad: float = math.radians(3.0)
    min_inlier_frac: float = 0.60         # acceptance gate
    max_short_frac: float = 0.25          # reject if too many beams are contradicted
    raycast: RaycastConfig = field(default_factory=RaycastConfig)


@dataclass(frozen=True)
class CheckpointMatch:
    checkpoint_id: str
    pose: Pose
    inlier_frac: float
    short_frac: float
    score: float


class CheckpointMatcher:
    def __init__(
        self,
        occupied: np.ndarray,
        origin_x_m: float,
        origin_y_m: float,
        resolution_m: float,
        checkpoints: Sequence[Checkpoint],
        cfg: CheckpointMatchConfig = CheckpointMatchConfig(),
    ) -> None:
        self._occ = np.asarray(occupied, dtype=bool)
        self._ox = float(origin_x_m)
        self._oy = float(origin_y_m)
        self._res = float(resolution_m)
        self._checkpoints = list(checkpoints)
        self._cfg = cfg

    def _candidates(self, prior: Pose) -> List[Checkpoint]:
        px, py = prior[0], prior[1]
        near = [
            (math.hypot(c.x_m - px, c.y_m - py), c)
            for c in self._checkpoints
        ]
        return [c for d, c in sorted(near, key=lambda t: t[0])
                if d <= self._cfg.select_radius_m]

    def n_candidates(self, prior: Pose) -> int:
        """Checkpoints within ``select_radius_m`` of `prior` — lets a caller
        tell "no checkpoint nearby" apart from "match attempted but rejected"."""
        return len(self._candidates(prior))

    def match(
        self,
        prior: Pose,
        angles: Sequence[float],
        ranges: Sequence[float],
    ) -> Optional[CheckpointMatch]:
        """Best accepted checkpoint match near `prior`, or None. `angles`
        (body-frame rad) + `ranges` (m) are the live scan."""
        best: Optional[CheckpointMatch] = None
        for c in self._candidates(prior):
            sub, sox, soy = crop_disk(
                self._occ, self._ox, self._oy, self._res,
                (c.x_m, c.y_m), c.radius_m)
            if sub.size == 0 or not sub.any():
                continue
            rc = replace(
                self._cfg.raycast,
                max_range_m=min(self._cfg.raycast.max_range_m, c.radius_m))
            pose, s = best_pose_in_window(
                sub, sox, soy, self._res, prior, angles, ranges,
                xy_half_m=self._cfg.xy_half_m, xy_step_m=self._cfg.xy_step_m,
                theta_half_rad=self._cfg.theta_half_rad,
                theta_step_rad=self._cfg.theta_step_rad,
                cfg=rc)
            if (s.inlier_frac >= self._cfg.min_inlier_frac
                    and s.short_frac <= self._cfg.max_short_frac):
                m = CheckpointMatch(c.id, pose, s.inlier_frac, s.short_frac, s.score)
                if best is None or m.score > best.score:
                    best = m
        return best

    def relocalize(
        self,
        angles: Sequence[float],
        ranges: Sequence[float],
        *,
        yaw_hint: Optional[float] = None,
        xy_half_m: float = 0.6,
        xy_step_m: float = 0.10,
        theta_half_rad: float = math.pi,
        theta_step_rad: float = math.radians(5.0),
    ) -> Optional[CheckpointMatch]:
        """Cold-start / recovery: test **all** checkpoints, searching a window
        around *each checkpoint's own pose* (not an odom prior). Heading is
        swept around ``yaw_hint`` (IMU-primed) over ±``theta_half_rad``, or the
        full circle when ``yaw_hint`` is None. Returns the best accepted match.
        Slower than ``match`` — a deliberate operator action, not per-tick."""
        best: Optional[CheckpointMatch] = None
        for c in self._checkpoints:
            sub, sox, soy = crop_disk(
                self._occ, self._ox, self._oy, self._res,
                (c.x_m, c.y_m), c.radius_m)
            if sub.size == 0 or not sub.any():
                continue
            rc = replace(
                self._cfg.raycast,
                max_range_m=min(self._cfg.raycast.max_range_m, c.radius_m))
            center_theta = c.theta_rad if yaw_hint is None else float(yaw_hint)
            pose, s = best_pose_in_window(
                sub, sox, soy, self._res,
                (c.x_m, c.y_m, center_theta), angles, ranges,
                xy_half_m=xy_half_m, xy_step_m=xy_step_m,
                theta_half_rad=theta_half_rad, theta_step_rad=theta_step_rad,
                cfg=rc)
            if (s.inlier_frac >= self._cfg.min_inlier_frac
                    and s.short_frac <= self._cfg.max_short_frac):
                m = CheckpointMatch(c.id, pose, s.inlier_frac, s.short_frac, s.score)
                if best is None or m.score > best.score:
                    best = m
        return best
