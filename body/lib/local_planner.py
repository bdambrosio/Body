"""Local A* planner over the body-frame scan grid (pure Python + numpy).

The single authority for local feasibility/routing: build a footprint-inflated,
clearance-graded costmap from the live scan, snap the goal onto it, and A* from
the robot (body origin) to the goal. The returned path is feasible on the
inflated grid — the body fits the whole way — so whatever Tier-2 hands in, this
either returns a drivable path or honestly reports no path. Tier-3
(``body/local_drive.py``) follows the path via pure-pursuit.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from body.lib.astar import astar_toward, nearest_non_lethal
from body.lib.local_costmap import (
    LocalCostmap, LocalCostmapConfig, build_local_costmap, dilate_bool,
    lethal_radius_m,
)

Point2 = Tuple[float, float]
Cell = Tuple[int, int]


@dataclass(frozen=True)
class LocalPlanConfig:
    costmap: LocalCostmapConfig = field(default_factory=LocalCostmapConfig)
    cost_per_unit: float = 0.10
    heuristic_weight: float = 1.0
    max_expansions: int = 50_000
    min_clearance_cells: int = 0       # hard extra margin beyond footprint (0 = halo only)
    goal_clearance_cells: int = 0      # extra goal clearance beyond the footprint.
    #   0 = accept a goal at the footprint edge (the swept veto still guards
    #   actual motion). Tier-2 reads this same value to stay ½-cell more
    #   conservative than Tier-3 by construction (no goal snapping).
    downsample_step_cells: int = 3     # path point spacing (cells)
    # Snapped/frontier endpoints within this many cells of the closest
    # reachable distance are ties → cheapest path wins (stable endpoint).
    snap_tie_cells: float = 1.0


@dataclass
class LocalPlan:
    ok: bool
    path_body: List[Point2]            # robot→goal in body metres (downsampled)
    goal_body_snapped: Optional[Point2]
    # ok        — path reaches the requested goal cell
    # snapped   — goal cell is lethal; path ends at the best reachable cell
    #             within the snap radius of it
    # frontier  — goal unreachable; path ends at the closest reachable cell
    # at_frontier (ok=False) — the robot already IS at the closest reachable
    #             cell (nothing reachable is nearer the goal); the caller
    #             treats this as arrival at the best-effort endpoint
    # boxed_in | start_blocked | start_out_of_map | goal_out_of_map (ok=False)
    reason: str
    n_expansions: int = 0
    elapsed_ms: float = 0.0


def _world_to_cell(xy: Point2, res: float, ox: float, oy: float) -> Cell:
    return (int(math.floor((xy[0] - ox) / res + 1e-9)),
            int(math.floor((xy[1] - oy) / res + 1e-9)))


def _cell_to_world(ij: Cell, res: float, ox: float, oy: float) -> Point2:
    return (ox + (ij[0] + 0.5) * res, oy + (ij[1] + 0.5) * res)


def plan_local(
    grid: np.ndarray, meta: Dict[str, Any], goal_body: Point2,
    cfg: Optional[LocalPlanConfig] = None,
    costmap: Optional[LocalCostmap] = None,
    *,
    start_body: Point2 = (0.0, 0.0),
) -> LocalPlan:
    """Plan a footprint-feasible path from the robot to `goal_body`.

    ``start_body`` is the robot's position in the grid's frame (the scan's
    body frame); (0, 0) unless the robot has moved since the scan.

    The goal is never pre-snapped by ring order: A* searches toward the raw
    goal and, if the goal cell is lethal or unreachable, ends at the
    reachable cell nearest to it (ties → cheaper path) — the best reachable
    snap (T6). ``goal_clearance_cells`` > 0 prefers an endpoint with that much
    extra clearance."""
    cfg = cfg or LocalPlanConfig()
    t0 = time.monotonic()
    cm = costmap or build_local_costmap(grid, meta, cfg.costmap)
    res = float(meta["resolution_m"]); ox = float(meta["origin_x_m"]); oy = float(meta["origin_y_m"])
    nx, ny = cm.lethal.shape

    lethal = (dilate_bool(cm.lethal, iters=cfg.min_clearance_cells)
              if cfg.min_clearance_cells > 0 else cm.lethal)

    si, sj = _world_to_cell(start_body, res, ox, oy)
    gi, gj = _world_to_cell(goal_body, res, ox, oy)

    def _fail(reason: str, n_exp: int = 0) -> LocalPlan:
        return LocalPlan(False, [], None, reason, n_expansions=n_exp,
                         elapsed_ms=(time.monotonic() - t0) * 1000.0)

    if not (0 <= si < nx and 0 <= sj < ny):
        return _fail("start_out_of_map")
    if not (0 <= gi < nx and 0 <= gj < ny):
        return _fail("goal_out_of_map")

    if lethal[si, sj]:
        # Robot centre inside the inflated band (bumped close to something):
        # step out to the nearest legal cell, but never across a blocked cell.
        lr = lethal_radius_m(cfg.costmap, res)
        radius = int(math.ceil(lr / res)) + 1 + cfg.min_clearance_cells
        relaxed = nearest_non_lethal(lethal, si, sj, radius=radius,
                                     blocked=(grid == 0))
        if relaxed is None:
            return _fail("start_blocked")
        si, sj = relaxed

    goal_ok = None
    if cfg.goal_clearance_cells > 0:
        goal_ok = ~dilate_bool(lethal, iters=cfg.goal_clearance_cells)

    cells, n_exp, msg = astar_toward(
        cost=cm.cost, lethal=lethal, start=(si, sj), goal=(gi, gj),
        cost_per_unit=cfg.cost_per_unit, h_weight=cfg.heuristic_weight,
        max_expansions=cfg.max_expansions, goal_ok=goal_ok,
        tie_cells=cfg.snap_tie_cells)
    elapsed = (time.monotonic() - t0) * 1000.0
    if cells is None or len(cells) <= 1:
        # Nothing reachable is nearer the goal than where we stand. If the
        # robot can move at all, it is AT the best reachable cell (an arrival
        # for a best-effort goal); if not, it is genuinely boxed in.
        free_nb = any(
            0 <= si + di < nx and 0 <= sj + dj < ny and not lethal[si + di, sj + dj]
            for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj)
        return LocalPlan(False, [], _cell_to_world((si, sj), res, ox, oy),
                         "at_frontier" if free_nb else "boxed_in",
                         n_expansions=n_exp, elapsed_ms=elapsed)

    end = cells[-1]
    if msg == "frontier" and lethal[gi, gj]:
        snap_r = max(8, 6 + 2 * cfg.min_clearance_cells)
        if math.hypot(end[0] - gi, end[1] - gj) <= snap_r:
            msg = "snapped"
    elif msg == "frontier" and goal_ok is not None and not goal_ok[gi, gj]:
        msg = "snapped"
    path = _downsample(cells, cfg.downsample_step_cells)
    path_body = [_cell_to_world(c, res, ox, oy) for c in path]
    # Anchor the path at the robot so the follower's lookahead starts there.
    if path_body and math.hypot(path_body[0][0] - start_body[0],
                                path_body[0][1] - start_body[1]) > res:
        path_body = [(float(start_body[0]), float(start_body[1]))] + path_body
    return LocalPlan(True, path_body, _cell_to_world(end, res, ox, oy),
                     msg, n_expansions=n_exp, elapsed_ms=elapsed)


def _downsample(cells: List[Cell], step: int) -> List[Cell]:
    if step <= 1 or len(cells) <= 2:
        return list(cells)
    out = cells[::step]
    if out[-1] != cells[-1]:
        out.append(cells[-1])
    return out


# ── pure-pursuit lookahead on a body-frame path (robot at origin) ────


def lookahead_on_path(path: List[Point2], lookahead_m: float) -> Optional[Point2]:
    """Point ~`lookahead_m` along `path` ahead of the robot (body origin)."""
    if len(path) < 2:
        return path[-1] if path else None
    robot = (0.0, 0.0)
    ni = _nearest_index(path, robot)
    if ni >= len(path) - 1:
        return path[-1]
    a, b = path[ni], path[ni + 1]
    seg_proj, _t = _project_onto_segment(robot, a, b)
    remaining = max(0.0, lookahead_m - math.hypot(seg_proj[0], seg_proj[1]))
    cur = seg_proj
    i = ni
    while i < len(path) - 1:
        nxt = path[i + 1]
        seg_len = math.hypot(nxt[0] - cur[0], nxt[1] - cur[1])
        if seg_len >= remaining:
            if seg_len < 1e-9:
                return nxt
            t = remaining / seg_len
            return (cur[0] + t * (nxt[0] - cur[0]), cur[1] + t * (nxt[1] - cur[1]))
        remaining -= seg_len
        cur = nxt
        i += 1
    return path[-1]


def _nearest_index(path: List[Point2], xy: Point2) -> int:
    best_i, best_d2 = 0, float("inf")
    for i, (px, py) in enumerate(path):
        d2 = (px - xy[0]) ** 2 + (py - xy[1]) ** 2
        if d2 < best_d2:
            best_d2, best_i = d2, i
    return best_i


def _project_onto_segment(xy: Point2, a: Point2, b: Point2) -> Tuple[Point2, float]:
    abx, aby = b[0] - a[0], b[1] - a[1]
    denom = abx * abx + aby * aby
    if denom < 1e-12:
        return a, 0.0
    t = max(0.0, min(1.0, ((xy[0] - a[0]) * abx + (xy[1] - a[1]) * aby) / denom))
    return (a[0] + t * abx, a[1] + t * aby), t


# ── path geometry helpers (path hysteresis / progress) ───────────────


def transform_path(path: List[Point2], pose: Tuple[float, float, float]) -> List[Point2]:
    """Express points given in the frame of ``pose`` (x, y, θ relative to the
    target frame) in the target frame: p' = R(θ)·p + (x, y)."""
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return [(pose[0] + px * c - py * s, pose[1] + px * s + py * c) for px, py in path]


def path_remaining_m(path: List[Point2], robot: Point2 = (0.0, 0.0)) -> float:
    """Polyline length from the robot's projection onto ``path`` to its end."""
    if not path:
        return 0.0
    if len(path) == 1:
        return math.hypot(path[0][0] - robot[0], path[0][1] - robot[1])
    ni = _nearest_index(path, robot)
    # project onto the better of the two segments touching the nearest vertex
    best = None
    for k in (ni - 1, ni):
        if 0 <= k < len(path) - 1:
            proj, _t = _project_onto_segment(robot, path[k], path[k + 1])
            d = math.hypot(proj[0] - robot[0], proj[1] - robot[1])
            if best is None or d < best[0]:
                best = (d, k, proj)
    if best is None:
        return 0.0
    _d, k, proj = best
    total = math.hypot(path[k + 1][0] - proj[0], path[k + 1][1] - proj[1])
    for i in range(k + 1, len(path) - 1):
        total += math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
    return total


def trim_path(path: List[Point2], robot: Point2) -> Tuple[List[Point2], float]:
    """Drop the part of ``path`` already behind the robot. Returns
    (path starting at the robot's projection, lateral distance robot→path).
    Projects onto the closest segment; near-ties (within 2 cm) go to the
    segment furthest along, so a path that doubles back never re-captures
    the robot on its earlier leg."""
    if len(path) < 2:
        d = math.hypot(path[0][0] - robot[0], path[0][1] - robot[1]) if path else 0.0
        return list(path), d
    projs = []
    for k in range(len(path) - 1):
        proj, _t = _project_onto_segment(robot, path[k], path[k + 1])
        projs.append((math.hypot(proj[0] - robot[0], proj[1] - robot[1]), k, proj))
    dmin = min(p[0] for p in projs)
    d, k, proj = max((p for p in projs if p[0] <= dmin + 0.02), key=lambda p: p[1])
    tail = [q for q in path[k + 1:]]
    if tail and math.hypot(tail[0][0] - proj[0], tail[0][1] - proj[1]) < 1e-6:
        tail = tail[1:]
    return [proj] + tail, d


def _sample_polyline(path: List[Point2], step: float) -> List[Point2]:
    out: List[Point2] = []
    if not path:
        return out
    out.append(path[0])
    for a, b in zip(path[:-1], path[1:]):
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(math.ceil(seg / max(step, 1e-6))))
        for k in range(1, n + 1):
            t = k / n
            out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    return out


def path_cost(path: List[Point2], cm: LocalCostmap, cfg: LocalPlanConfig,
              lethal: Optional[np.ndarray] = None) -> float:
    """A*-comparable cost of a body-frame polyline on ``cm`` (inf if it
    touches a lethal cell or leaves the grid). Sampled every half cell.

    ``lethal`` overrides ``cm.lethal`` for the validity test — the caller
    passes the UNPADDED footprint mask when re-validating a previous path,
    because a path that hugs the padded lethal edge flickers in and out of
    it as the grid re-quantizes under a rotating robot; cells that are only
    pad-lethal then cost the full halo."""
    res = float(cm.meta["resolution_m"]); ox = float(cm.meta["origin_x_m"]); oy = float(cm.meta["origin_y_m"])
    nx, ny = cm.lethal.shape
    hard = cm.lethal if lethal is None else lethal
    halo_max = float(cfg.costmap.halo_max)
    pts = _sample_polyline(path, 0.5 * res)
    total = 0.0
    prev = None
    for p in pts:
        i, j = _world_to_cell(p, res, ox, oy)
        if not (0 <= i < nx and 0 <= j < ny) or hard[i, j]:
            return float("inf")
        if prev is not None:
            c = float(cm.cost[i, j])
            if not math.isfinite(c):
                c = halo_max
            ds = math.hypot(p[0] - prev[0], p[1] - prev[1]) / res
            total += ds * (1.0 + cfg.cost_per_unit * c)
        prev = p
    return total
