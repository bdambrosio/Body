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
PAINT_KINDS = (WALL, FREE, UNKNOWN)        # write the occupancy layer

# Keep-out (policy) layer kinds — write the bool `nogo` mask, never
# occupancy, so localization is unaffected.
NOGO = "nogo"
ERASE_NOGO = "erase_nogo"
NOGO_KINDS = (NOGO, ERASE_NOGO)

_PAINT_LOG_ODDS = {
    WALL: float(LOG_ODDS_MAX),    # +4.0  → occupied
    FREE: float(LOG_ODDS_MIN),    # -4.0  → clear
    UNKNOWN: 0.0,                 #  0.0  → unknown
}


def _bresenham(x0: int, y0: int, x1: int, y1: int):
    """Integer line from (x0,y0) to (x1,y1) inclusive. Yields
    (i, j, is_endpoint) for each cell along the ray."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        is_end = (x == x1 and y == y1)
        yield x, y, is_end
        if is_end:
            return
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


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
    # Keep-out mask (bool, same shape as log_odds). Operator policy layer;
    # carried through to save_npz, never touches occupancy. Filled to an
    # all-False array on construction when None.
    nogo: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.nogo is None:
            self.nogo = np.zeros(self.log_odds.shape, dtype=bool)
        else:
            self.nogo = np.asarray(self.nogo, dtype=bool)

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
        """Paint cells (ii, jj). Occupancy kinds (wall/free/unknown) set
        `log_odds`; keep-out kinds (nogo/erase_nogo) set the `nogo` mask."""
        if len(ii) == 0:
            return
        if kind in PAINT_KINDS:
            self.log_odds[ii, jj] = _PAINT_LOG_ODDS[kind]
        elif kind == NOGO:
            self.nogo[ii, jj] = True
        elif kind == ERASE_NOGO:
            self.nogo[ii, jj] = False
        else:
            raise ValueError(f"unknown paint kind: {kind!r}")

    def stamp_cells_from_scan(
        self, world_xy: Optional[np.ndarray],
        pose: Tuple[float, float, float], *, max_range_m: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Cells (ii, jj) to promote to Wall from live-scan world points.

        Keeps hits within ``max_range_m`` of the robot (``pose`` x, y) that
        land on a currently free-or-unknown cell (never an existing wall).
        One cell per hit — NO thickening — deduplicated. Pure: returns the
        cells; the caller paints + handles undo."""
        empty = (np.empty(0, np.intp), np.empty(0, np.intp))
        if world_xy is None or len(world_xy) == 0:
            return empty
        pts = np.asarray(world_xy, dtype=np.float64)
        rng = np.hypot(pts[:, 0] - pose[0], pts[:, 1] - pose[1])
        pts = pts[rng <= max_range_m]
        if len(pts) == 0:
            return empty
        res = self.resolution_m
        ii = np.floor((pts[:, 0] - self.origin_x_m) / res + 1e-9).astype(np.intp)
        jj = np.floor((pts[:, 1] - self.origin_y_m) / res + 1e-9).astype(np.intp)
        nx, ny = self.shape
        inb = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
        ii, jj = ii[inb], jj[inb]
        if len(ii) == 0:
            return empty
        # free+unknown only (driveable: 0=wall, 1=free, -1=unknown).
        drive = driveable_from_occupancy(self.log_odds)
        fu = drive[ii, jj] != 0
        ii, jj = ii[fu], jj[fu]
        if len(ii) == 0:
            return empty
        # Dedup repeated hits in the same cell so counts are honest.
        _, uniq = np.unique(ii.astype(np.int64) * ny + jj, return_index=True)
        return ii[uniq], jj[uniq]

    def restamp_from_scans(
        self,
        scans,                      # list[(world_xy (N,2), origin_xy (x,y))]
        *,
        center_xy: Tuple[float, float],
        radius_m: float,
        max_range_m: float = 6.0,
    ) -> int:
        """Replace the *observed* occupancy within ``radius_m`` of
        ``center_xy``, rebuilt from one or more scans by ray-tracing:
        FREE along each beam, WALL at the endpoint, in **set** mode.

        Cells no beam crosses are left untouched, so geometry behind a wall
        is preserved (rays stop at the first hit). Endpoint wins over free
        where they collide. ``scans`` is a list of (world-frame endpoints
        (N,2), sensor-origin (x,y)); pass several odom-stitched scans in the
        same world frame to fill occlusion shadows. Returns cells changed.
        Mutates ``log_odds``; the caller snapshots for undo."""
        nx, ny = self.shape
        ci, cj = self.world_to_cell(center_xy[0], center_xy[1])
        r_cells = max(1, int(math.ceil(radius_m / self.resolution_m)))
        r2 = r_cells * r_cells
        free: set = set()
        occ: set = set()
        for world_xy, origin in scans:
            if world_xy is None or len(world_xy) == 0:
                continue
            pts = np.asarray(world_xy, dtype=np.float64)
            ox, oy = float(origin[0]), float(origin[1])
            oi, oj = self.world_to_cell(ox, oy)
            rng = np.hypot(pts[:, 0] - ox, pts[:, 1] - oy)
            pts = pts[(rng > 1e-3) & (rng <= max_range_m)]
            for k in range(pts.shape[0]):
                ei, ej = self.world_to_cell(pts[k, 0], pts[k, 1])
                end_in = (ei - ci) ** 2 + (ej - cj) ** 2 <= r2
                for i, j, is_end in _bresenham(oi, oj, ei, ej):
                    if (i - ci) ** 2 + (j - cj) ** 2 > r2:
                        break  # straight ray that leaves the disk won't re-enter
                    if not (0 <= i < nx and 0 <= j < ny):
                        break
                    if is_end:
                        if end_in:
                            occ.add((i, j))
                    else:
                        free.add((i, j))
        free -= occ
        changed = 0
        for cells, kind in ((free, FREE), (occ, WALL)):
            if not cells:
                continue
            ii = np.fromiter((c[0] for c in cells), np.intp, len(cells))
            jj = np.fromiter((c[1] for c in cells), np.intp, len(cells))
            self.log_odds[ii, jj] = _PAINT_LOG_ODDS[kind]
            changed += len(cells)
        return changed

    def snapshot_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Copy of (log_odds, nogo) for the undo stack — a stroke may
        touch either layer."""
        return self.log_odds.copy(), self.nogo.copy()

    def restore_state(self, state: Tuple[np.ndarray, np.ndarray]) -> None:
        log_odds, nogo = state
        np.copyto(self.log_odds, log_odds)
        np.copyto(self.nogo, nogo)


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
        nogo=(None if rm.nogo_mask is None
              else np.array(rm.nogo_mask, dtype=bool)),
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
        nogo_mask=emap.nogo,
    )
    save_reference_map(path, rm)
    return path
