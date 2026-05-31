"""Pure (no Qt / no zenoh) editable reference-map model.

The editor edits a **mapping-run reference map** (`reference_map.npz`):
the artifact the production MCL localizer
(`desktop/localization` → `MCLPoseSource`) loads. Its editable state is
`occupancy_log_odds`; the MCL scan matcher actually scores against the
**precomputed `likelihood_field`**, so on save we regenerate the
likelihood + distance fields from the edited occupancy via
`build_reference_map_from_log_odds`. The reference map is a static frozen
file (loaded once, never fused), so edits persist by construction.

`EditorMap` is a thin wrapper over `desktop.reference_map`:
  - paint sets log-odds to confident occupied / free / unknown,
  - `driveable_grid()` reuses `driveable_from_occupancy` for rendering,
  - `save_npz` rebuilds the derived fields and writes the npz.
"""
from __future__ import annotations

import math
import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from desktop.reference_map.reference_map import (
    LOG_ODDS_MAX,
    LOG_ODDS_MIN,
    build_reference_map_from_log_odds,
    driveable_from_occupancy,
    load_reference_map,
    save_reference_map,
)

# Paint kinds → confident log-odds. Wall/Free use the mapper's clamp
# extremes so they sit well past OCCUPIED_LOG_ODDS_THRESHOLD (0.5) and
# survive the hysteresis/denoise pass on save.
WALL = "wall"
FREE = "free"
UNKNOWN = "unknown"
PAINT_KINDS = (WALL, FREE, UNKNOWN)

_PAINT_LOG_ODDS = {
    WALL: float(LOG_ODDS_MAX),    # +4.0  → occupied
    FREE: float(LOG_ODDS_MIN),    # -4.0  → clear
    UNKNOWN: 0.0,                 #  0.0  → unknown
}


@dataclass
class EditorMap:
    """In-memory editable reference map. `log_odds` is the (nx, ny)
    editable state; everything else is carried through to save."""

    log_odds: np.ndarray
    resolution_m: float
    origin_x_m: float
    origin_y_m: float
    session_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    trajectory: Optional[np.ndarray] = None

    # ── Geometry (mirrors ReferenceMap) ─────────────────────────────

    @property
    def shape(self) -> Tuple[int, int]:
        return self.log_odds.shape  # (nx, ny)

    @property
    def meta(self) -> Dict[str, Any]:
        """View/costmap-compatible meta dict (matches
        ReferenceMap.snapshot_for_ui)."""
        nx, ny = self.shape
        return {
            "resolution_m": self.resolution_m,
            "origin_x_m": self.origin_x_m,
            "origin_y_m": self.origin_y_m,
            "nx": nx, "ny": ny, "frame": "world",
        }

    def world_to_cell(self, x_w: float, y_w: float) -> Tuple[int, int]:
        i = int(math.floor((x_w - self.origin_x_m) / self.resolution_m + 1e-9))
        j = int(math.floor((y_w - self.origin_y_m) / self.resolution_m + 1e-9))
        return i, j

    def cell_to_world(self, i: int, j: int) -> Tuple[float, float]:
        return (
            self.origin_x_m + (i + 0.5) * self.resolution_m,
            self.origin_y_m + (j + 0.5) * self.resolution_m,
        )

    def in_bounds(self, i: int, j: int) -> bool:
        nx, ny = self.shape
        return 0 <= i < nx and 0 <= j < ny

    def bounds_ij(self) -> Optional[Tuple[int, int, int, int]]:
        """Tight bbox of known (non-zero) cells, for auto-fit framing."""
        known = np.isfinite(self.log_odds) & (np.abs(self.log_odds) > 1e-6)
        if not np.any(known):
            return None
        ii, jj = np.where(known)
        return int(ii.min()), int(ii.max()), int(jj.min()), int(jj.max())

    # ── Render helper ───────────────────────────────────────────────

    def driveable_grid(self) -> np.ndarray:
        """int8 (1 clear / 0 blocked / -1 unknown), reusing the same
        thresholding the live stack uses."""
        return driveable_from_occupancy(self.log_odds)

    # ── Paint (the only mutator) ────────────────────────────────────

    def brush_cells(
        self, i0: int, j0: int, radius_cells: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """(ii, jj) for a filled disk of `radius_cells` at (i0, j0),
        clamped to the grid. radius 0 → the single center cell."""
        nx, ny = self.shape
        r = max(0, int(radius_cells))
        i_lo, i_hi = max(0, i0 - r), min(nx - 1, i0 + r)
        j_lo, j_hi = max(0, j0 - r), min(ny - 1, j0 + r)
        if i_lo > i_hi or j_lo > j_hi:
            return (np.empty(0, np.intp), np.empty(0, np.intp))
        ii, jj = np.meshgrid(
            np.arange(i_lo, i_hi + 1),
            np.arange(j_lo, j_hi + 1),
            indexing="ij",
        )
        if r > 0:
            mask = (ii - i0) ** 2 + (jj - j0) ** 2 <= r * r
            ii, jj = ii[mask], jj[mask]
        return (ii.ravel(), jj.ravel())

    def paint(self, ii: np.ndarray, jj: np.ndarray, kind: str) -> None:
        """Set `log_odds` at cells (ii, jj) to the confident value for
        `kind` (wall=+max, free=-max, unknown=0)."""
        if kind not in PAINT_KINDS:
            raise ValueError(f"unknown paint kind: {kind!r}")
        if len(ii) == 0:
            return
        self.log_odds[ii, jj] = _PAINT_LOG_ODDS[kind]

    def snapshot_occ(self) -> np.ndarray:
        """Copy of log_odds for the undo stack."""
        return self.log_odds.copy()

    def restore_occ(self, log_odds: np.ndarray) -> None:
        np.copyto(self.log_odds, log_odds)


# ── npz I/O ─────────────────────────────────────────────────────────


def load_npz(path: str) -> EditorMap:
    """Load a `reference_map.npz` into an editable `EditorMap`."""
    rm = load_reference_map(path)
    return EditorMap(
        log_odds=np.array(rm.occupancy_log_odds, dtype=np.float32),
        resolution_m=rm.resolution_m,
        origin_x_m=rm.origin_x_m,
        origin_y_m=rm.origin_y_m,
        session_id=rm.session_id,
        metadata=dict(rm.metadata),
        trajectory=(None if rm.trajectory is None
                    else np.array(rm.trajectory, dtype=np.float64)),
    )


def save_npz(emap: EditorMap, path: str, *, backup: bool = True) -> str:
    """Rebuild the likelihood + distance fields from the edited
    occupancy and write a `reference_map.npz`. If `backup` and `path`
    exists, copy it to `path + '.bak'` first. Returns the path."""
    if backup and os.path.exists(path):
        shutil.copy2(path, path + ".bak")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rm = build_reference_map_from_log_odds(
        emap.log_odds,
        resolution_m=emap.resolution_m,
        origin_x_m=emap.origin_x_m,
        origin_y_m=emap.origin_y_m,
        session_id=emap.session_id or None,
        metadata=emap.metadata,
        trajectory=emap.trajectory,
    )
    save_reference_map(path, rm)
    return path
