"""One supported scan → session grid + change detector (explore plan A1/B1).

Shared by the live nav window and ``scripts/replay_changes.py`` so a replay
scores exactly what the robot ran.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from desktop.perception.change_detector import (
    ChangeDetector, ChangeDetectorConfig, ChangeEvent,
)
from desktop.world_map.session_grid import SessionGrid, SessionGridConfig

Pose = Tuple[float, float, float]


class ChangeSession:
    def __init__(self, reference_map, *,
                 grid_cfg: Optional[SessionGridConfig] = None,
                 det_cfg: Optional[ChangeDetectorConfig] = None,
                 sweep_s: float = 2.0, detect: bool = True) -> None:
        rm = reference_map
        meta = {"nx": rm.nx, "ny": rm.ny, "resolution_m": rm.resolution_m,
                "origin_x_m": rm.origin_x_m, "origin_y_m": rm.origin_y_m}
        self.grid = SessionGrid(meta, grid_cfg)
        self.detector = (ChangeDetector(rm.driveable_int8(), rm.distance_field_m, meta, det_cfg)
                         if detect else None)
        self.sweep_s = sweep_s
        self._last_sweep = -1e18

    def meta_key(self):
        return self.grid.meta_key()

    def observe(self, ts: float, ranges: np.ndarray, angles: np.ndarray,
                pose: Pose) -> List[ChangeEvent]:
        """Integrate one scan the map supports; returns change events."""
        g = self.grid
        g.integrate(ranges, angles, pose, ts)
        det = self.detector
        if det is None:
            return []
        r = np.asarray(ranges, dtype=np.float64)
        a = np.asarray(angles, dtype=np.float64)
        ok = np.isfinite(r) & (r >= g.cfg.min_range_m) & (r <= g.cfg.max_range_m)
        pts = np.stack([r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])], axis=1)
        events = det.observe_scan(ts, pts, pose, free_at=g.is_free)
        if ts - self._last_sweep >= self.sweep_s:
            self._last_sweep = ts
            score, first = g.layers()
            events += det.sweep(ts, score, first, robot_xy=(pose[0], pose[1]))
        return events

    def reset(self) -> None:
        self.grid.reset()
        if self.detector is not None:
            self.detector.reset()
        self._last_sweep = -1e18
