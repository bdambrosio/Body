"""Session observation grid: what the lidar has seen *this* nav session, on
the reference map's own grid (explore plan A1 / nav-redesign step 4).

Why: the planning map used to merge the reference map with itself (the
localization controller's "snapshot" *is* the reference map), so nothing
observed during a run reached the route planner — a box on the route was
only ever seen by Tier-3, one blocked leg at a time. This grid closes that
gap and is the substrate for change detection (new / missing / moving) and
for exploring past the mapped area.

Per cell it keeps:
  * ``score``        — clamped log-odds-style evidence (+hit, -free), so a
                       moved chair flips within a handful of scans;
  * ``n_obs``        — how many scans touched the cell;
  * ``last_hit_ts``  — Pi-clock ts of the last scan that ended on it;
  * ``last_free_ts`` — Pi-clock ts of the last scan that passed through it;
  * ``first_hit_ts`` — start of the cell's current occupied episode (0 when
                       the cell is not currently leaning occupied) — how long
                       a new object has been there.

Only lidar feeds it (depth stays out until the denoiser passes), and the
caller feeds it only while the pose is trustworthy — a bad pose would paint
walls in the wrong place (see ``desktop/nav/pose_health.PoseSupportMonitor``).
Beams with no return are skipped (no clearing), same as the mapper; a real
return beyond ``max_range_m`` clears up to ``max_range_m`` without a hit
(otherwise an object removed from in front of a far wall never clears).

Pure numpy, no I/O. Thread-safe: integrate() runs on the Zenoh scan thread,
readers run on the UI thread.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

Pose = Tuple[float, float, float]


@dataclass(frozen=True)
class SessionGridConfig:
    max_range_m: float = 5.0          # same cap as the mapper / MCL
    min_range_m: float = 0.12         # self-hits / mixed pixels near the body
    hit_inc: float = 0.85
    free_inc: float = -0.4
    clamp: float = 3.0
    # A cell is "blocked" above this score, "free" below its negative.
    occ_thresh: float = 0.5
    # Overriding a reference-map WALL with "free" needs more evidence than
    # clearing unknown: shallow rays graze wall cells and would erode them.
    free_over_prior_wall: float = -2.0
    # Overriding reference-map FREE with "blocked" (a new object) uses
    # occ_thresh; the planner's inflation does the rest.
    # "Seen through" a reference wall (the evidence a wall is MISSING, as
    # opposed to a beam grazing its face under a 1-cell pose/map offset):
    # the beam must carry on at least through_beyond_m past the cell, and
    # be out of the wall within through_cross_m on both sides of it (a beam
    # running along inside a wall line does not count).
    through_beyond_m: float = 0.30
    through_cross_m: float = 0.15
    # ...and must meet the wall at least this steeply (angle between beam and
    # wall line). 11-03: every "missing" report on four patrols sat on a
    # reference wall, mostly walls the lidar sweeps at a glancing angle — a
    # shallow beam slips past a 5 cm wall cell under a few cm of pose error.
    # Corners / blobs (no clear wall direction) keep the old test. 0 = off.
    through_min_incidence_deg: float = 30.0



def _box_sum(a: np.ndarray, r: int) -> np.ndarray:
    """Sum over a (2r+1)² window (zero padded), via cumulative sums."""
    p = np.pad(a, ((r + 1, r), (r + 1, r)))
    c = p.cumsum(0).cumsum(1)
    k = 2 * r + 1
    return c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]


def _wall_normals(wall: np.ndarray, r: int = 2, min_coherence: float = 0.5) -> np.ndarray:
    """Unit normal (nx, ny, 2) of the wall line at each wall cell, from the
    structure tensor of the wall mask over a (2r+1)² window — robust for
    1-cell-thin lines, where a plain gradient vanishes at the centre. Cells
    without a clear direction (corners, blobs: coherence below
    ``min_coherence``) and non-wall cells get (0, 0)."""
    w = wall.astype(np.float64)
    gx = np.zeros_like(w)
    gy = np.zeros_like(w)
    gx[1:-1, :] = 0.5 * (w[2:, :] - w[:-2, :])
    gy[:, 1:-1] = 0.5 * (w[:, 2:] - w[:, :-2])
    jxx, jxy, jyy = _box_sum(gx * gx, r), _box_sum(gx * gy, r), _box_sum(gy * gy, r)
    tr = jxx + jyy
    diff = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2)
    coh = np.where(tr > 1e-9, diff / np.maximum(tr, 1e-9), 0.0)
    ang = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)     # dominant gradient = normal direction
    out = np.zeros(wall.shape + (2,))
    ok = wall & (coh >= min_coherence)
    out[ok, 0] = np.cos(ang[ok])
    out[ok, 1] = np.sin(ang[ok])
    return out


class SessionGrid:
    def __init__(self, meta: Dict[str, Any],
                 config: Optional[SessionGridConfig] = None) -> None:
        self.cfg = config or SessionGridConfig()
        self.nx = int(meta["nx"])
        self.ny = int(meta["ny"])
        self.res = float(meta["resolution_m"])
        self.ox = float(meta["origin_x_m"])
        self.oy = float(meta["origin_y_m"])
        self._lock = threading.Lock()
        self.score = np.zeros((self.nx, self.ny), dtype=np.float32)
        self.n_obs = np.zeros((self.nx, self.ny), dtype=np.uint16)
        self.last_hit_ts = np.zeros((self.nx, self.ny), dtype=np.float64)
        self.last_free_ts = np.zeros((self.nx, self.ny), dtype=np.float64)
        self.first_hit_ts = np.zeros((self.nx, self.ny), dtype=np.float64)
        # Scans that saw THROUGH a reference wall cell (see config); reset
        # when the cell is hit again. Needs set_wall_mask().
        self.through = np.zeros((self.nx, self.ny), dtype=np.uint16)
        self._wall: Optional[np.ndarray] = None
        self._wall_normal: Optional[np.ndarray] = None
        self.version = 0          # bumps on every integrate / reset
        self.scans_integrated = 0
        self.scans_skipped = 0

    def meta_key(self) -> Tuple[int, int, float, float, float]:
        return (self.nx, self.ny, self.res, self.ox, self.oy)

    # ── writes ────────────────────────────────────────────────────────
    def reset(self) -> None:
        with self._lock:
            self.score.fill(0.0)
            self.n_obs.fill(0)
            self.last_hit_ts.fill(0.0)
            self.last_free_ts.fill(0.0)
            self.first_hit_ts.fill(0.0)
            self.through.fill(0)
            self.version += 1

    def set_wall_mask(self, wall: Optional[np.ndarray]) -> None:
        """Reference walls (bool, this grid's shape) for the see-through
        layer; None disables it."""
        if wall is not None and wall.shape != (self.nx, self.ny):
            raise ValueError("wall mask shape mismatch")
        self._wall = None if wall is None else wall.astype(bool)
        self._wall_normal = None if wall is None else _wall_normals(self._wall)

    def note_skipped(self) -> None:
        self.scans_skipped += 1

    def _cells(self, wx: np.ndarray, wy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        i = np.floor((wx - self.ox) / self.res).astype(np.int64)
        j = np.floor((wy - self.oy) / self.res).astype(np.int64)
        return i, j

    def integrate(self, ranges_m: np.ndarray, angles_rad: np.ndarray,
                  pose_world: Pose, ts: float) -> int:
        """Ray-cast one body-frame scan from ``pose_world`` (the pose at the
        scan's own ts). Each cell is counted at most once per scan; a cell
        that is both passed through and hit in the same scan counts as hit.
        Returns the number of cells touched."""
        cfg = self.cfg
        r = np.asarray(ranges_m, dtype=np.float64)
        a = np.asarray(angles_rad, dtype=np.float64)
        ok = np.isfinite(r) & (r >= cfg.min_range_m)
        if not ok.any():
            return 0
        r, a = r[ok], a[ok]
        is_hit = r <= cfg.max_range_m
        x0, y0, th = pose_world
        wa = a + th
        cos_a, sin_a = np.cos(wa), np.sin(wa)

        # Endpoints (hits) — only returns within max range.
        hi, hj = self._cells(x0 + r[is_hit] * cos_a[is_hit], y0 + r[is_hit] * sin_a[is_hit])

        # Free samples every half cell along each beam, stopping one cell
        # short of the hit so the hit cell itself is never cleared; beams
        # that returned beyond max range clear up to max range.
        step = 0.5 * self.res
        n_max = int(math.ceil(cfg.max_range_m / step))
        t = (np.arange(n_max, dtype=np.float64) + 0.5) * step     # (S,)
        stop = np.where(is_hit, np.maximum(r - self.res, 0.0), cfg.max_range_m)
        mask = t[None, :] < stop[:, None]                          # (B,S)
        bi, si = np.nonzero(mask)
        fi, fj = self._cells(x0 + t[si] * cos_a[bi], y0 + t[si] * sin_a[bi])

        def _flat(i, j):
            inb = (i >= 0) & (i < self.nx) & (j >= 0) & (j < self.ny)
            return np.unique(i[inb] * self.ny + j[inb])

        hit_flat = _flat(hi, hj)
        free_flat = np.setdiff1d(_flat(fi, fj), hit_flat, assume_unique=True)

        through_flat = np.zeros(0, dtype=np.int64)
        wall = self._wall
        if wall is not None and bi.size:
            inb = (fi >= 0) & (fi < self.nx) & (fj >= 0) & (fj < self.ny)
            on_wall = np.zeros(fi.shape, dtype=bool)
            on_wall[inb] = wall[fi[inb], fj[inb]]
            if on_wall.any():
                wb, ws = bi[on_wall], si[on_wall]
                tt = t[ws]
                far = np.where(is_hit, r, cfg.max_range_m)[wb]

                def _wall_at(d):
                    ci, cj = self._cells(x0 + d * cos_a[wb], y0 + d * sin_a[wb])
                    ok_ = (ci >= 0) & (ci < self.nx) & (cj >= 0) & (cj < self.ny) & (d > 0)
                    out = np.zeros(d.shape, dtype=bool)
                    out[ok_] = wall[ci[ok_], cj[ok_]]
                    return out
                crossing = ((far - tt >= cfg.through_beyond_m)
                            & ~_wall_at(tt - cfg.through_cross_m)
                            & ~_wall_at(tt + cfg.through_cross_m))
                if cfg.through_min_incidence_deg > 0 and self._wall_normal is not None:
                    # |beam · wall normal| = sin(angle between beam and wall).
                    nrm = self._wall_normal[fi[on_wall], fj[on_wall]]       # (K, 2), 0 = no direction
                    has_dir = (nrm[:, 0] != 0) | (nrm[:, 1] != 0)
                    sin_inc = np.abs(cos_a[wb] * nrm[:, 0] + sin_a[wb] * nrm[:, 1])
                    steep = sin_inc >= math.sin(math.radians(cfg.through_min_incidence_deg))
                    crossing &= steep | ~has_dir
                through_flat = np.setdiff1d(
                    np.unique(fi[on_wall][crossing] * self.ny + fj[on_wall][crossing]),
                    hit_flat, assume_unique=True)

        with self._lock:
            s = self.score.reshape(-1)
            n = self.n_obs.reshape(-1)
            lh = self.last_hit_ts.reshape(-1)
            lf = self.last_free_ts.reshape(-1)
            s[hit_flat] = np.minimum(cfg.clamp, s[hit_flat] + cfg.hit_inc)
            s[free_flat] = np.maximum(-cfg.clamp, s[free_flat] + cfg.free_inc)
            touched = np.concatenate([hit_flat, free_flat])
            n[touched] = np.minimum(np.iinfo(np.uint16).max - 1, n[touched]) + 1
            lh[hit_flat] = ts
            lf[free_flat] = ts
            fh = self.first_hit_ts.reshape(-1)
            new_ep = hit_flat[fh[hit_flat] == 0.0]
            fh[new_ep] = ts
            ended = free_flat[s[free_flat] <= 0.0]
            fh[ended] = 0.0
            th_ = self.through.reshape(-1)
            th_[through_flat] = np.minimum(np.iinfo(np.uint16).max - 1, th_[through_flat]) + 1
            th_[hit_flat] = 0
            self.version += 1
            self.scans_integrated += 1
        return int(touched.size)

    # ── reads ─────────────────────────────────────────────────────────
    def driveable_int8(self) -> np.ndarray:
        """1 free / 0 blocked / -1 not (yet) decided by this session."""
        with self._lock:
            s = self.score.copy()
        out = np.full(s.shape, -1, dtype=np.int8)
        out[s > self.cfg.occ_thresh] = 0
        out[s < -self.cfg.occ_thresh] = 1
        return out

    def planning_layer(self, prior: np.ndarray, *,
                       exclude_xy: Optional[Tuple[float, float]] = None,
                       exclude_radius_m: float = 0.3) -> np.ndarray:
        """Merge with the reference prior (int8 1/0/-1 on this grid): what
        this session saw wins, except that clearing a prior wall needs
        ``free_over_prior_wall`` evidence. Session-only blocked cells within
        ``exclude_radius_m`` of ``exclude_xy`` (the robot) are left out:
        inflated, they would seal the robot's own start cell (the
        start_blocked stalls of the depth-memory days); Tier-3's live scan
        owns anything that close."""
        with self._lock:
            s = self.score.copy()
        cfg = self.cfg
        out = prior.astype(np.int8, copy=True)
        blocked = s > cfg.occ_thresh
        if exclude_xy is not None and exclude_radius_m > 0:
            ci = (exclude_xy[0] - self.ox) / self.res - 0.5
            cj = (exclude_xy[1] - self.oy) / self.res - 0.5
            k = int(math.ceil(exclude_radius_m / self.res))
            i0, i1 = max(0, int(ci) - k), min(self.nx, int(ci) + k + 2)
            j0, j1 = max(0, int(cj) - k), min(self.ny, int(cj) + k + 2)
            if i0 < i1 and j0 < j1:
                ii, jj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
                near = np.hypot(ii - ci, jj - cj) * self.res <= exclude_radius_m
                sub = blocked[i0:i1, j0:j1]
                sub[near] = False
        out[blocked] = 0
        free = s < -cfg.occ_thresh
        wall = prior == 0
        out[free & ~wall] = 1
        out[free & wall & (s <= cfg.free_over_prior_wall)] = 1
        return out

    def is_free(self, x: float, y: float) -> bool:
        """Is world (x, y) currently seen free by this session?"""
        i = int(math.floor((x - self.ox) / self.res))
        j = int(math.floor((y - self.oy) / self.res))
        if not (0 <= i < self.nx and 0 <= j < self.ny):
            return False
        return bool(self.score[i, j] < -self.cfg.occ_thresh)

    def layers(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Copies of (score, first_hit_ts, through) for the change detector."""
        with self._lock:
            return self.score.copy(), self.first_hit_ts.copy(), self.through.copy()

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            s = self.score
            return {
                "scans": self.scans_integrated,
                "skipped": self.scans_skipped,
                "blocked": int((s > self.cfg.occ_thresh).sum()),
                "free": int((s < -self.cfg.occ_thresh).sum()),
                "version": self.version,
            }
