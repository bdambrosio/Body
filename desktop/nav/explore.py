"""Explore Phase D: where to look next in the mapped house (docs/explore_phase_d.md).

Pure numpy, no I/O, no Qt.

* ``ExploreMap`` is precomputed once per reference map:
  * **Viewpoints**: the most open free cell in each ``viewpoint_spacing_m``
    block (most clearance from walls, outside no-go).
  * **Visibility**: what a 360° lidar would touch from each viewpoint,
    ray-cast on the reference map. Rays stop at a wall (counted) or at
    unknown (not counted).
  * **Observable**: the union of those sets, which is the denominator for
    coverage.
* ``ExplorePlanner`` holds per-run state:
  * cells looked at from a visited viewpoint, viewpoints abandoned this run;
  * greedy next-best-view: ``gain / (travel_m + travel_bias_m)``, where gain
    counts observable cells neither seen by this session's grid nor looked
    at from a viewpoint already visited. Counting "looked at" keeps a
    viewpoint whose cells the real lidar can't see (lidar height, glass)
    from being chosen again and again.
* ``investigate_spot`` picks where to look at a reported change from: 1.0–1.5 m
  away, with line of sight on the reference map, from a bearing at least 30°
  off the first report's, and the shortest drive.

Travel distances come from one Dijkstra over the known, non-lethal cells of
the live planning costmap (``reachable_distances``).
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

Pt = Tuple[float, float]
Pose = Tuple[float, float, float]


@dataclass(frozen=True)
class ExploreConfig:
    budget_min: float = 15.0
    target_coverage: float = 0.90
    min_gain_cells: int = 40              # ≈ 0.1 m² at 5 cm: stop chasing slivers
    viewpoint_spacing_m: float = 0.75
    viewpoint_min_clearance_m: float = 0.30
    view_range_m: float = 4.0
    n_rays: int = 180
    travel_bias_m: float = 1.0            # c in gain / (travel + c)
    min_leg_m: float = 0.5                # no 10 cm hops: what's that close is seen in transit
    viewpoint_arrival_tol_m: float = 0.30
    investigate_min_m: float = 1.0
    investigate_max_m: float = 1.5
    investigate_min_bearing_deg: float = 30.0
    investigate_dwell_s: float = 3.0
    investigate_arrival_tol_m: float = 0.20
    max_investigations: int = 6
    home_arrival_tol_m: float = 0.15
    give_up_after_cycles: int = 1


def _cell(meta: dict, x: float, y: float) -> Tuple[int, int]:
    r = float(meta["resolution_m"])
    return (int(math.floor((x - float(meta["origin_x_m"])) / r + 1e-9)),
            int(math.floor((y - float(meta["origin_y_m"])) / r + 1e-9)))


def _world(meta: dict, i: int, j: int) -> Pt:
    r = float(meta["resolution_m"])
    return (float(meta["origin_x_m"]) + (i + 0.5) * r,
            float(meta["origin_y_m"]) + (j + 0.5) * r)


def _meta_key(meta: dict) -> Tuple[int, int, float, float, float]:
    return (int(meta["nx"]), int(meta["ny"]), round(float(meta["resolution_m"]), 6),
            round(float(meta["origin_x_m"]), 6), round(float(meta["origin_y_m"]), 6))


def _clearance_m(free: np.ndarray, res: float, cap_m: float) -> np.ndarray:
    """Distance (m, capped) from each cell to the nearest non-free cell, by
    4/8-neighbour wavefront: coarse, but only used to rank cells."""
    nx, ny = free.shape
    d = np.where(free, np.inf, 0.0)
    blocked = ~free
    front = blocked.copy()
    k = 0
    while front.any() and k * res < cap_m:
        k += 1
        grow = np.zeros_like(front)
        grow[1:, :] |= front[:-1, :]
        grow[:-1, :] |= front[1:, :]
        grow[:, 1:] |= front[:, :-1]
        grow[:, :-1] |= front[:, 1:]
        grow[1:, 1:] |= front[:-1, :-1]
        grow[:-1, :-1] |= front[1:, 1:]
        grow[1:, :-1] |= front[:-1, 1:]
        grow[:-1, 1:] |= front[1:, :-1]
        new = grow & np.isinf(d)
        d[new] = k * res
        front = new
    d[np.isinf(d)] = cap_m
    return d


def cast_visible(drive: np.ndarray, i: int, j: int, *, res: float, range_m: float,
                 n_rays: int) -> np.ndarray:
    """Flat indices (into ``drive``) a 360° scan from cell (i, j) touches on the
    reference map (1 free / 0 wall / -1 unknown): free cells up to and
    including the first wall cell; a ray stops before unknown."""
    nx, ny = drive.shape
    step = 0.5 * res
    ks = np.arange(1, int(range_m / step) + 1) * step              # (K,)
    ang = np.linspace(0.0, 2 * math.pi, n_rays, endpoint=False)     # (A,)
    ci, cj = (i + 0.5) * res, (j + 0.5) * res
    xi = np.floor((ci + np.cos(ang)[:, None] * ks[None, :]) / res).astype(np.int64)
    yj = np.floor((cj + np.sin(ang)[:, None] * ks[None, :]) / res).astype(np.int64)
    inb = (xi >= 0) & (xi < nx) & (yj >= 0) & (yj < ny)
    v = np.full(xi.shape, -1, dtype=np.int8)
    v[inb] = drive[xi[inb], yj[inb]]
    wall = v == 0
    unk = v < 0
    big = xi.shape[1]
    first_wall = np.where(wall.any(1), wall.argmax(1), big)
    first_unk = np.where(unk.any(1), unk.argmax(1), big)
    stop = np.minimum(first_wall, first_unk)                         # exclusive …
    keep = np.arange(big)[None, :] < stop[:, None]
    keep |= (np.arange(big)[None, :] == first_wall[:, None]) & (first_wall < first_unk)[:, None]
    flat = (xi * ny + yj)[keep & inb]
    return np.unique(flat)


class ExploreMap:
    """Viewpoints and their visible cells, precomputed on the reference map."""

    def __init__(self, drive: np.ndarray, nogo: Optional[np.ndarray], meta: dict,
                 cfg: Optional[ExploreConfig] = None) -> None:
        self.cfg = cfg or ExploreConfig()
        self.meta = dict(meta)
        self.key = _meta_key(meta)
        self.drive = np.asarray(drive, dtype=np.int8)
        nx, ny = self.drive.shape
        res = float(meta["resolution_m"])
        nogo = np.zeros((nx, ny), bool) if nogo is None else np.asarray(nogo, bool)
        free = (self.drive == 1) & ~nogo
        clear = _clearance_m(self.drive == 1, res, cap_m=1.0)
        ok = free & (clear >= self.cfg.viewpoint_min_clearance_m)
        b = max(1, int(round(self.cfg.viewpoint_spacing_m / res)))
        vps: List[Tuple[int, int]] = []
        ii, jj = np.nonzero(ok)
        if len(ii):
            blk = (ii // b) * ((ny // b) + 1) + (jj // b)
            order = np.lexsort((-clear[ii, jj], blk))   # by block, most open first
            seen_blk = set()
            for o in order:
                k = int(blk[o])
                if k in seen_blk:
                    continue
                seen_blk.add(k)
                vps.append((int(ii[o]), int(jj[o])))
        self.viewpoints: List[Tuple[int, int]] = vps
        self.visible: List[np.ndarray] = [
            cast_visible(self.drive, i, j, res=res, range_m=self.cfg.view_range_m,
                         n_rays=self.cfg.n_rays) for i, j in vps]
        obs = np.zeros(nx * ny, dtype=bool)
        for v in self.visible:
            obs[v] = True
        self.observable = obs.reshape(nx, ny)
        self.n_observable = int(obs.sum())

    def viewpoint_world(self, k: int) -> Pt:
        i, j = self.viewpoints[k]
        return _world(self.meta, i, j)


def reachable_distances(lethal: np.ndarray, unknown: Optional[np.ndarray], meta: dict,
                        start_xy: Pt, *, max_m: float = 60.0) -> np.ndarray:
    """Driving distance (m) from ``start_xy`` to every known, non-lethal cell
    (8-connected Dijkstra); inf where unreachable. The start cell counts as
    passable even if lethal (the robot is in it)."""
    nx, ny = lethal.shape
    res = float(meta["resolution_m"])
    passable = ~np.asarray(lethal, bool)
    if unknown is not None:
        passable &= ~np.asarray(unknown, bool)
    dist = np.full((nx, ny), np.inf)
    si, sj = _cell(meta, *start_xy)
    if not (0 <= si < nx and 0 <= sj < ny):
        return dist
    if not passable[si, sj]:
        # Step to the nearest passable cell within 0.3 m (the robot can sit in
        # inflation).
        r = int(math.ceil(0.3 / res))
        best = None
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                a, c = si + di, sj + dj
                if 0 <= a < nx and 0 <= c < ny and passable[a, c]:
                    d = math.hypot(di, dj)
                    if best is None or d < best[0]:
                        best = (d, a, c)
        if best is None:
            return dist
        dist[best[1], best[2]] = best[0] * res
        heap = [(best[0] * res, best[1], best[2])]
    else:
        dist[si, sj] = 0.0
        heap = [(0.0, si, sj)]
    steps = [(1, 0, res), (-1, 0, res), (0, 1, res), (0, -1, res),
             (1, 1, res * math.sqrt(2)), (1, -1, res * math.sqrt(2)),
             (-1, 1, res * math.sqrt(2)), (-1, -1, res * math.sqrt(2))]
    while heap:
        d, i, j = heapq.heappop(heap)
        if d > dist[i, j] or d > max_m:
            continue
        for di, dj, w in steps:
            a, c = i + di, j + dj
            if 0 <= a < nx and 0 <= c < ny and passable[a, c]:
                nd = d + w
                if nd < dist[a, c]:
                    dist[a, c] = nd
                    heapq.heappush(heap, (nd, a, c))
    return dist


@dataclass
class ViewpointChoice:
    index: int
    xy: Pt
    gain: int
    travel_m: float
    score: float


class ExplorePlanner:
    """Per-run state on top of an ``ExploreMap``."""

    def __init__(self, emap: ExploreMap) -> None:
        self.emap = emap
        self.cfg = emap.cfg
        nx, ny = emap.drive.shape
        self.looked = np.zeros(nx * ny, dtype=bool)
        self.abandoned: Set[int] = set()
        self.visited: List[int] = []

    def _covered(self, seen: Optional[np.ndarray]) -> np.ndarray:
        cov = self.looked.copy()
        if seen is not None:
            cov |= np.asarray(seen, bool).reshape(-1)
        return cov

    def coverage(self, seen: Optional[np.ndarray]) -> float:
        if self.emap.n_observable == 0:
            return 1.0
        obs = self.emap.observable.reshape(-1)
        return float((self._covered(seen) & obs).sum()) / self.emap.n_observable

    def gains(self, seen: Optional[np.ndarray]) -> np.ndarray:
        cov = self._covered(seen)
        return np.array([int((~cov[v]).sum()) for v in self.emap.visible], dtype=np.int64)

    def choose(self, seen: Optional[np.ndarray], dist: np.ndarray) -> Optional[ViewpointChoice]:
        """Best next viewpoint, or None when nothing reachable gains enough."""
        g = self.gains(seen)
        best: Optional[ViewpointChoice] = None
        for k, (i, j) in enumerate(self.emap.viewpoints):
            if k in self.abandoned or g[k] < self.cfg.min_gain_cells:
                continue
            t = float(dist[i, j])
            if not math.isfinite(t) or t < self.cfg.min_leg_m:
                continue
            s = g[k] / (t + self.cfg.travel_bias_m)
            if best is None or s > best.score:
                best = ViewpointChoice(k, self.emap.viewpoint_world(k), int(g[k]), t, s)
        return best

    def mark_visited(self, k: int) -> None:
        self.visited.append(k)
        self.looked[self.emap.visible[k]] = True

    def mark_abandoned(self, k: int) -> None:
        self.abandoned.add(k)


def line_of_sight(drive: np.ndarray, meta: dict, a: Pt, b: Pt, *, end_skip_m: float = 0.15) -> bool:
    """No reference wall or unknown on the segment a → b (the last
    ``end_skip_m`` before b ignored: b is the change itself)."""
    res = float(meta["resolution_m"])
    d = math.hypot(b[0] - a[0], b[1] - a[1])
    n = int(max(0.0, d - end_skip_m) / (0.5 * res))
    nx, ny = drive.shape
    for k in range(1, n + 1):
        t = k * 0.5 * res / d
        i, j = _cell(meta, a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        if not (0 <= i < nx and 0 <= j < ny) or drive[i, j] != 1:
            return False
    return True


@dataclass
class InvestigateSpot:
    xy: Pt
    heading: float          # facing the change
    travel_m: float
    new_angle: bool         # bearing ≥ min_bearing off the first report's


def investigate_spot(drive: np.ndarray, meta: dict, dist: np.ndarray, change_xy: Pt,
                     first_view_xy: Optional[Pt], cfg: ExploreConfig) -> Optional[InvestigateSpot]:
    """Where to look at ``change_xy`` from (see module doc), or None."""
    res = float(meta["resolution_m"])
    ci, cj = _cell(meta, *change_xy)
    r = int(math.ceil(cfg.investigate_max_m / res))
    nx, ny = dist.shape
    b0 = (math.atan2(first_view_xy[1] - change_xy[1], first_view_xy[0] - change_xy[0])
          if first_view_xy is not None else None)
    min_b = math.radians(cfg.investigate_min_bearing_deg)
    best: Optional[Tuple[Tuple[int, float], InvestigateSpot]] = None
    for i in range(max(0, ci - r), min(nx, ci + r + 1)):
        for j in range(max(0, cj - r), min(ny, cj + r + 1)):
            t = float(dist[i, j])
            if not math.isfinite(t):
                continue
            x, y = _world(meta, i, j)
            rng = math.hypot(x - change_xy[0], y - change_xy[1])
            if not (cfg.investigate_min_m <= rng <= cfg.investigate_max_m):
                continue
            if not line_of_sight(drive, meta, (x, y), change_xy):
                continue
            b = math.atan2(y - change_xy[1], x - change_xy[0])
            new = b0 is None or abs((b - b0 + math.pi) % (2 * math.pi) - math.pi) >= min_b
            key = (0 if new else 1, t)
            if best is None or key < best[0]:
                best = (key, InvestigateSpot(
                    (x, y), math.atan2(change_xy[1] - y, change_xy[0] - x), t, new))
    return best[1] if best else None
