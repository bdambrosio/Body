"""Log-odds occupancy grid builder for mapping sessions."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from desktop.reference_map.reference_map import (
    LOG_ODDS_FREE,
    LOG_ODDS_MAX,
    LOG_ODDS_MIN,
    LOG_ODDS_OCC,
    build_reference_map_from_log_odds,
    ReferenceMap,
)


def _clamp_cell(log_odds: np.ndarray, i: int, j: int) -> None:
    v = log_odds[i, j]
    if v < LOG_ODDS_MIN:
        log_odds[i, j] = LOG_ODDS_MIN
    elif v > LOG_ODDS_MAX:
        log_odds[i, j] = LOG_ODDS_MAX


class OccupancyBuilder:
    """Ray-cast lidar scans into a log-odds grid."""

    def __init__(
        self,
        *,
        extent_m: float,
        resolution_m: float,
        max_range_m: float = 5.0,
    ):
        self._res = float(resolution_m)
        self._extent = float(extent_m)
        n = 2 * int(math.ceil(extent_m / resolution_m / 2.0))
        self._n = n
        self._origin_x = -extent_m / 2.0
        self._origin_y = -extent_m / 2.0
        self._max_range = float(max_range_m)
        self.log_odds = np.zeros((n, n), dtype=np.float32)

    @property
    def resolution_m(self) -> float:
        return self._res

    @property
    def origin_x_m(self) -> float:
        return self._origin_x

    @property
    def origin_y_m(self) -> float:
        return self._origin_y

    @property
    def n_cells(self) -> int:
        return self._n

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        i = int(math.floor((x - self._origin_x) / self._res + 1e-9))
        j = int(math.floor((y - self._origin_y) / self._res + 1e-9))
        return i, j

    def integrate_scan(
        self,
        ranges_m: np.ndarray,
        angles_rad: np.ndarray,
        pose_world: Tuple[float, float, float],
    ) -> int:
        """Bresenham-style ray cast; returns cells updated."""
        x0, y0, theta = pose_world
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        updated = 0
        for r, a in zip(ranges_m, angles_rad):
            if not np.isfinite(r) or r <= 0 or r > self._max_range:
                continue
            bx = r * math.cos(a)
            by = r * math.sin(a)
            wx = x0 + cos_t * bx - sin_t * by
            wy = y0 + sin_t * bx + cos_t * by
            updated += self._cast_ray(x0, y0, wx, wy, hit=True)
        return updated

    def _cast_ray(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        *,
        hit: bool,
    ) -> int:
        i0, j0 = self.world_to_cell(x0, y0)
        i1, j1 = self.world_to_cell(x1, y1)
        n = self._n
        cells = list(_bresenham(i0, j0, i1, j1))
        if not cells:
            return 0
        free_cells = cells[:-1] if hit and len(cells) > 1 else cells
        count = 0
        for i, j in free_cells:
            if 0 <= i < n and 0 <= j < n:
                self.log_odds[i, j] += LOG_ODDS_FREE
                _clamp_cell(self.log_odds, i, j)
                count += 1
        if hit:
            i, j = cells[-1]
            if 0 <= i < n and 0 <= j < n:
                self.log_odds[i, j] += LOG_ODDS_OCC
                _clamp_cell(self.log_odds, i, j)
                count += 1
        return count

    def occupied_mask(self, threshold: float = 0.5) -> np.ndarray:
        return self.log_odds > threshold

    def to_reference_map(
        self,
        *,
        session_id: str = "",
        trajectory: np.ndarray | None = None,
        metadata: dict | None = None,
    ) -> ReferenceMap:
        return build_reference_map_from_log_odds(
            self.log_odds.copy(),
            resolution_m=self._res,
            origin_x_m=self._origin_x,
            origin_y_m=self._origin_y,
            session_id=session_id or None,
            trajectory=trajectory,
            metadata=metadata,
        )

    def snapshot_for_ui(self) -> dict:
        drive = np.full(self.log_odds.shape, -1, dtype=np.int8)
        drive[self.log_odds > 0.5] = 0
        drive[self.log_odds < -0.5] = 1
        known = np.abs(self.log_odds) > 0.1
        if not np.any(known):
            bounds = None
        else:
            ii, jj = np.where(known)
            bounds = (int(ii.min()), int(ii.max()), int(jj.min()), int(jj.max()))
        return {
            "driveable": drive,
            "meta": {
                "resolution_m": self._res,
                "origin_x_m": self._origin_x,
                "origin_y_m": self._origin_y,
                "nx": self._n,
                "ny": self._n,
                "frame": "world",
            },
            "bounds_ij": bounds,
        }


def _bresenham(i0: int, j0: int, i1: int, j1: int) -> list:
    cells = []
    di = abs(i1 - i0)
    dj = abs(j1 - j0)
    si = 1 if i0 < i1 else -1
    sj = 1 if j0 < j1 else -1
    err = di - dj
    i, j = i0, j0
    while True:
        cells.append((i, j))
        if i == i1 and j == j1:
            break
        e2 = 2 * err
        if e2 > -dj:
            err -= dj
            i += si
        if e2 < di:
            err += di
            j += sj
    return cells
