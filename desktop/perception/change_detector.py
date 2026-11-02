"""Change detection against the reference map (explore plan B1).

Two time scales, both lidar-first (depth stays out until the denoiser
passes; the camera/VLM only *labels* what this finds):

* **Moving** — per scan. Endpoints that land on space the reference map
  calls free (away from its walls) are residuals; they are clustered and
  tracked by nearest neighbour in the world frame. A track whose net
  displacement passes ``moving_min_disp_m`` is a moving object (person, pet,
  door swinging); its end is reported with path length and speed.

* **Static** — swept every few seconds over the session grid:
  - ``new``: cells the reference map calls free that this session has hit
    solidly for ``new_min_age_s`` (a bag, a moved chair);
  - ``missing``: reference walls this session repeatedly sees *through*
    (an object removed, a door opened).
  A static change "goes" only when its cells are *observed* back to the
  reference state — not when they merely fall out of view.

The caller must feed only scans the map supports (``PoseSupportMonitor.
scan_ok``): a wrong pose turns every wall into a change. Pure logic, no
I/O, not thread-safe (call from one thread).
"""
from __future__ import annotations

import itertools
import math
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

Pose = Tuple[float, float, float]

KIND_MOVING = "moving"
KIND_NEW = "new"
KIND_MISSING = "missing"
STATUS_APPEARED = "appeared"
STATUS_GONE = "gone"


@dataclass(frozen=True)
class ChangeDetectorConfig:
    # Ignore evidence this close to a reference wall (pose and map error
    # live there). 0.15 on the 11-02 patrol left 111 of 170 "new" reports
    # within 0.25 m of a wall — registration churn, not objects.
    wall_margin_m: float = 0.30
    cluster_eps_m: float = 0.20        # bucket size for clustering endpoints
    cluster_min_points: int = 3
    cluster_max_extent_m: float = 1.0  # bigger → misalignment, not an object
    # Tracking.
    track_gate_m: float = 0.5
    track_lost_s: float = 1.0
    moving_min_disp_m: float = 0.4
    moving_min_hits: int = 4
    # A real mover leaves free space behind: its first position must be
    # seen free (``free_at``) before it counts as moving. A static edge
    # whose visible part slides as the robot drives never passes.
    moving_require_vacated: bool = True
    # ... checked on the cells the track's FIRST points hit (a desk or lamp
    # seen as several legs has its centroid in free space between them).
    moving_vacated_frac: float = 0.6
    # Static sweep.
    new_min_score: float = 1.5         # SessionGrid score ("solidly hit")
    new_min_age_s: float = 3.0         # occupied episode at least this long
    new_min_cells: int = 3              # a 15 cm face (the 4-cell try dropped a 16 cm box)
    # Seen from two viewpoints: the robot must have moved this far between
    # the first sweep that found the blob and the one that reports it.
    new_min_baseline_m: float = 0.5
    missing_max_score: float = -2.0    # reference wall seen through this much
    # ... and crossed by at least this many scans (SessionGrid.through):
    # beams that carried on behind it, not ones grazing its face. The
    # 11-02 hallway_loop lap: 65 "missing" on a 20-minute-old map, every one
    # on a wall face.
    missing_min_through: int = 3
    missing_min_cells: int = 3
    missing_min_extent_m: float = 0.25
    # Only solid reference walls can go missing: a wall cell with at least
    # this many wall cells in its 3x3 neighbourhood (not mapping speckle).
    missing_solid_neighbours: int = 3
    static_match_m: float = 0.4        # same change across sweeps
    near_track_m: float = 0.4          # a stopped moving object isn't "new" yet


@dataclass
class ChangeEvent:
    kind: str                  # moving | new | missing
    status: str                # appeared | gone
    change_id: int
    ts: float                  # Pi-clock ts of the evidence
    x: float
    y: float
    extent_m: float = 0.0
    n_cells: int = 0
    speed_mps: Optional[float] = None
    path_m: Optional[float] = None
    duration_s: Optional[float] = None
    label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class _Track:
    tid: int
    first_ts: float
    last_ts: float
    pts: Deque[Tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=200))
    hits: int = 0
    first_pts: Optional[np.ndarray] = None   # world xy of the first sighting
    moving: bool = False
    path_m: float = 0.0
    extent_m: float = 0.0

    @property
    def xy(self) -> Tuple[float, float]:
        return self.pts[-1][1], self.pts[-1][2]

    def disp(self) -> float:
        (_, x0, y0), (_, x1, y1) = self.pts[0], self.pts[-1]
        return math.hypot(x1 - x0, y1 - y0)


@dataclass
class _Static:
    cid: int
    kind: str
    x: float
    y: float
    cells: np.ndarray          # flat indices
    first_ts: float
    last_seen_ts: float
    first_robot: Optional[Tuple[float, float]] = None
    reported: bool = False
    extent_m: float = 0.0


def _components(coords: np.ndarray, connect8: bool = True) -> List[np.ndarray]:
    """Connected components of integer (N, 2) cell coords → list of index
    arrays into ``coords``."""
    if coords.shape[0] == 0:
        return []
    where = {(int(i), int(j)): k for k, (i, j) in enumerate(coords)}
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connect8:
        nbrs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    seen = np.zeros(coords.shape[0], dtype=bool)
    out = []
    for k in range(coords.shape[0]):
        if seen[k]:
            continue
        seen[k] = True
        stack, comp = [k], []
        while stack:
            m = stack.pop()
            comp.append(m)
            i, j = coords[m]
            for di, dj in nbrs:
                n = where.get((int(i) + di, int(j) + dj))
                if n is not None and not seen[n]:
                    seen[n] = True
                    stack.append(n)
        out.append(np.asarray(comp))
    return out


class ChangeDetector:
    def __init__(self, ref_driveable: np.ndarray, ref_distance_m: np.ndarray,
                 meta: Dict[str, Any], config: Optional[ChangeDetectorConfig] = None):
        """``ref_driveable``: reference int8 1/0/-1; ``ref_distance_m``: the
        reference distance-to-wall field; ``meta``: resolution_m,
        origin_x_m, origin_y_m (same grid as the SessionGrid)."""
        self.cfg = config or ChangeDetectorConfig()
        self.ref = ref_driveable
        self.ref_dist = ref_distance_m
        self.res = float(meta["resolution_m"])
        self.ox = float(meta["origin_x_m"])
        self.oy = float(meta["origin_y_m"])
        self.nx, self.ny = ref_driveable.shape
        wall = (ref_driveable == 0)
        nb = np.zeros(wall.shape, dtype=np.int16)
        pw = np.pad(wall, 1).astype(np.int16)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di or dj:
                    nb += pw[1 + di:1 + di + self.nx, 1 + dj:1 + dj + self.ny]
        self.solid_wall = wall & (nb >= self.cfg.missing_solid_neighbours)
        self._ids = itertools.count(1)
        self._tracks: List[_Track] = []
        self._static: List[_Static] = []

    # ── helpers ───────────────────────────────────────────────────────
    def _cell(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return (np.floor((x - self.ox) / self.res).astype(np.int64),
                np.floor((y - self.oy) / self.res).astype(np.int64))

    def _center(self, i: np.ndarray, j: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.ox + (i + 0.5) * self.res, self.oy + (j + 0.5) * self.res

    def tracks(self) -> List[Dict[str, Any]]:
        return [{"id": t.tid, "x": t.xy[0], "y": t.xy[1], "moving": t.moving}
                for t in self._tracks]

    def static_changes(self) -> List[Dict[str, Any]]:
        return [{"id": c.cid, "kind": c.kind, "x": c.x, "y": c.y,
                 "n_cells": int(c.cells.size), "reported": c.reported}
                for c in self._static]

    def reset(self) -> None:
        self._tracks.clear()
        self._static.clear()

    # ── moving: per scan ──────────────────────────────────────────────
    def residual_clusters(self, points_body: np.ndarray, pose: Pose
                          ) -> List[Tuple[float, float, float, int]]:
        """(x, y, extent_m, n_points) of endpoint clusters on reference-free
        space away from reference walls."""
        return [c[:4] for c in self._residual_clusters(points_body, pose)]

    def _residual_clusters(self, points_body: np.ndarray, pose: Pose):
        """Like residual_clusters plus each cluster's world points."""
        cfg = self.cfg
        if points_body.shape[0] == 0:
            return []
        c, s = math.cos(pose[2]), math.sin(pose[2])
        wx = pose[0] + c * points_body[:, 0] - s * points_body[:, 1]
        wy = pose[1] + s * points_body[:, 0] + c * points_body[:, 1]
        i, j = self._cell(wx, wy)
        inb = (i >= 0) & (i < self.nx) & (j >= 0) & (j < self.ny)
        keep = np.zeros(wx.shape[0], dtype=bool)
        keep[inb] = ((self.ref[i[inb], j[inb]] == 1)
                     & (self.ref_dist[i[inb], j[inb]] >= cfg.wall_margin_m))
        wx, wy = wx[keep], wy[keep]
        if wx.size < cfg.cluster_min_points:
            return []
        b = np.stack([np.floor(wx / cfg.cluster_eps_m), np.floor(wy / cfg.cluster_eps_m)],
                     axis=1).astype(np.int64)
        ub, inv = np.unique(b, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        out = []
        for comp in _components(ub):
            sel = np.isin(inv, comp)
            n = int(sel.sum())
            if n < cfg.cluster_min_points:
                continue
            px, py = wx[sel], wy[sel]
            extent = float(math.hypot(px.max() - px.min(), py.max() - py.min()))
            if extent > cfg.cluster_max_extent_m:
                continue
            out.append((float(px.mean()), float(py.mean()), extent, n,
                        np.stack([px, py], axis=1)))
        return out

    def observe_scan(self, ts: float, points_body: np.ndarray, pose: Pose,
                     free_at: Optional[Callable[[float, float], bool]] = None
                     ) -> List[ChangeEvent]:
        """``free_at(x, y)``: is that world point currently seen free (the
        SessionGrid)? Needed for ``moving_require_vacated``; None skips the
        check."""
        cfg = self.cfg
        events: List[ChangeEvent] = []
        clusters = self._residual_clusters(points_body, pose)
        # Greedy nearest-neighbour association, closest pairs first.
        pairs = []
        for ti, t in enumerate(self._tracks):
            for ci, (x, y, _e, _n, _p) in enumerate(clusters):
                d = math.hypot(x - t.xy[0], y - t.xy[1])
                if d <= cfg.track_gate_m:
                    pairs.append((d, ti, ci))
        used_t, used_c = set(), set()
        for d, ti, ci in sorted(pairs):
            if ti in used_t or ci in used_c:
                continue
            used_t.add(ti)
            used_c.add(ci)
            t = self._tracks[ti]
            x, y, e, _n, _pts = clusters[ci]
            t.path_m += math.hypot(x - t.xy[0], y - t.xy[1])
            t.pts.append((ts, x, y))
            t.last_ts = ts
            t.hits += 1
            t.extent_m = max(t.extent_m, e)
            vacated = True
            if free_at is not None and cfg.moving_require_vacated:
                fp = t.first_pts if t.first_pts is not None else np.array([[t.pts[0][1], t.pts[0][2]]])
                vacated = (np.mean([bool(free_at(float(px_), float(py_))) for px_, py_ in fp])
                           >= cfg.moving_vacated_frac)
            if (not t.moving and t.hits >= cfg.moving_min_hits
                    and t.disp() >= cfg.moving_min_disp_m and vacated):
                t.moving = True
                dt = max(1e-3, ts - t.first_ts)
                events.append(ChangeEvent(
                    KIND_MOVING, STATUS_APPEARED, t.tid, ts, x, y,
                    extent_m=t.extent_m, speed_mps=t.path_m / dt))
        for ci, (x, y, e, _n, pts) in enumerate(clusters):
            if ci not in used_c:
                t = _Track(next(self._ids), ts, ts, hits=1, extent_m=e,
                           first_pts=pts[:: max(1, len(pts) // 20)])
                t.pts.append((ts, x, y))
                self._tracks.append(t)
        alive = []
        for t in self._tracks:
            if ts - t.last_ts > cfg.track_lost_s:
                if t.moving:
                    dur = max(1e-3, t.last_ts - t.first_ts)
                    events.append(ChangeEvent(
                        KIND_MOVING, STATUS_GONE, t.tid, t.last_ts, t.xy[0], t.xy[1],
                        extent_m=t.extent_m, speed_mps=t.path_m / dur,
                        path_m=t.path_m, duration_s=dur))
            else:
                alive.append(t)
        self._tracks = alive
        return events

    # ── static: periodic sweep over the session grid ──────────────────
    def _blobs(self, mask: np.ndarray, min_cells: int) -> List[np.ndarray]:
        coords = np.argwhere(mask)
        return [coords[c] for c in _components(coords) if c.size >= min_cells]

    def sweep(self, now_ts: float, score: np.ndarray, first_hit_ts: np.ndarray,
              robot_xy: Optional[Tuple[float, float]] = None,
              through: Optional[np.ndarray] = None) -> List[ChangeEvent]:
        """``score`` / ``first_hit_ts``: the SessionGrid layers (same grid as
        the reference). ``now_ts`` on the Pi clock (latest scan ts).
        ``robot_xy``: robot position now, for the two-viewpoint gate on
        "new" (None → the gate is skipped)."""
        cfg = self.cfg
        events: List[ChangeEvent] = []
        new_mask = ((score >= cfg.new_min_score) & (self.ref == 1)
                    & (self.ref_dist >= cfg.wall_margin_m)
                    & (first_hit_ts > 0.0)
                    & (first_hit_ts <= now_ts - cfg.new_min_age_s))
        missing_mask = (score <= cfg.missing_max_score) & self.solid_wall
        if through is not None:
            missing_mask &= through >= cfg.missing_min_through
        missing = []
        for b in self._blobs(missing_mask, cfg.missing_min_cells):
            bx, by = self._center(b[:, 0], b[:, 1])
            if math.hypot(np.ptp(bx), np.ptp(by)) + self.res >= cfg.missing_min_extent_m:
                missing.append(b)
        found: List[Tuple[str, np.ndarray]] = (
            [(KIND_NEW, b) for b in self._blobs(new_mask, cfg.new_min_cells)]
            + [(KIND_MISSING, b) for b in missing])
        matched = set()
        for kind, cells in found:
            cx, cy = self._center(cells[:, 0], cells[:, 1])
            x, y = float(cx.mean()), float(cy.mean())
            if kind == KIND_NEW and any(
                    math.hypot(x - t.xy[0], y - t.xy[1]) < cfg.near_track_m
                    for t in self._tracks if t.moving):
                continue
            flat = cells[:, 0] * self.ny + cells[:, 1]
            hit = None
            for sc in self._static:
                if sc.kind == kind and sc.cid not in matched and \
                        math.hypot(sc.x - x, sc.y - y) <= cfg.static_match_m:
                    hit = sc
                    break
            extent = float(math.hypot(np.ptp(cx), np.ptp(cy)) + self.res)
            if hit is None:
                hit = _Static(next(self._ids), kind, x, y, flat, now_ts, now_ts,
                              first_robot=robot_xy)
                self._static.append(hit)
            matched.add(hit.cid)
            hit.x, hit.y, hit.cells, hit.last_seen_ts, hit.extent_m = x, y, flat, now_ts, extent
            if hit.reported:
                continue
            if kind == KIND_NEW and robot_xy is not None and hit.first_robot is not None:
                base = math.hypot(robot_xy[0] - hit.first_robot[0],
                                  robot_xy[1] - hit.first_robot[1])
                if base < cfg.new_min_baseline_m:
                    continue
            hit.reported = True
            events.append(ChangeEvent(kind, STATUS_APPEARED, hit.cid, now_ts, x, y,
                                      extent_m=extent, n_cells=int(flat.size)))
        # Unmatched: gone only if its cells are *observed* back to normal.
        keep = []
        flat_score = score.reshape(-1)
        for sc in self._static:
            if sc.cid in matched:
                keep.append(sc)
                continue
            vals = flat_score[sc.cells]
            back = (vals < 0.0).all() if sc.kind == KIND_NEW else (vals > 0.5).all()
            if back and not sc.reported:
                continue                       # never reported: drop quietly
            if back:
                events.append(ChangeEvent(
                    sc.kind, STATUS_GONE, sc.cid, now_ts, sc.x, sc.y,
                    n_cells=int(sc.cells.size),
                    duration_s=max(0.0, sc.last_seen_ts - sc.first_ts)))
            else:
                keep.append(sc)
        self._static = keep
        return events
