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
    goal_clearance_cells: int = 1      # snap the goal to a cell with this much extra clearance
    downsample_step_cells: int = 3     # path point spacing (cells)


@dataclass
class LocalPlan:
    ok: bool
    path_body: List[Point2]            # robot→goal in body metres (downsampled)
    goal_body_snapped: Optional[Point2]
    reason: str                        # ok|no_path|max_expansions|start_blocked|goal_unreachable|goal_out_of_map
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
) -> LocalPlan:
    """Plan a footprint-feasible path from the robot (0,0) to `goal_body`."""
    cfg = cfg or LocalPlanConfig()
    t0 = time.monotonic()
    cm = costmap or build_local_costmap(grid, meta, cfg.costmap)
    res = float(meta["resolution_m"]); ox = float(meta["origin_x_m"]); oy = float(meta["origin_y_m"])
    nx, ny = cm.lethal.shape

    lethal = (dilate_bool(cm.lethal, iters=cfg.min_clearance_cells)
              if cfg.min_clearance_cells > 0 else cm.lethal)

    si, sj = _world_to_cell((0.0, 0.0), res, ox, oy)
    gi, gj = _world_to_cell(goal_body, res, ox, oy)

    def _fail(reason: str) -> LocalPlan:
        return LocalPlan(False, [], None, reason,
                         elapsed_ms=(time.monotonic() - t0) * 1000.0)

    if not (0 <= si < nx and 0 <= sj < ny):
        return _fail("start_out_of_map")
    if not (0 <= gi < nx and 0 <= gj < ny):
        return _fail("goal_out_of_map")

    if lethal[si, sj]:
        relaxed = nearest_non_lethal(lethal, si, sj, radius=max(5, 4 + 2 * cfg.min_clearance_cells))
        if relaxed is None:
            return _fail("start_blocked")
        si, sj = relaxed
    # Snap the goal to a cell with extra clearance (don't aim right at a wall —
    # e.g. a world-frame goal that mis-registers onto a wall in the live scan).
    # Prefer a clearance-dilated mask; fall back to the nearest merely-legal
    # cell so we never fail just for lack of the extra margin.
    goal_radius = max(8, 6 + 2 * cfg.min_clearance_cells)
    goal_lethal = (dilate_bool(lethal, iters=cfg.goal_clearance_cells)
                   if cfg.goal_clearance_cells > 0 else lethal)
    if goal_lethal[gi, gj]:
        relaxed = nearest_non_lethal(goal_lethal, gi, gj, radius=goal_radius)
        if relaxed is None:
            relaxed = nearest_non_lethal(lethal, gi, gj, radius=goal_radius)
            if relaxed is None:
                return _fail("goal_unreachable")
        gi, gj = relaxed

    # Reachable-by-construction (Invariant I3): if the requested goal isn't
    # reachable, head to the reachable cell closest to it (round the corner)
    # rather than reporting no-path. Only a genuine box-in ("no path") fails.
    cells, n_exp, msg = astar_toward(
        cost=cm.cost, lethal=lethal, start=(si, sj), goal=(gi, gj),
        cost_per_unit=cfg.cost_per_unit, h_weight=cfg.heuristic_weight,
        max_expansions=cfg.max_expansions)
    elapsed = (time.monotonic() - t0) * 1000.0
    if cells is None or len(cells) <= 1:
        return LocalPlan(False, [], None, "boxed_in",
                         n_expansions=n_exp, elapsed_ms=elapsed)

    end = cells[-1]
    path = _downsample(cells, cfg.downsample_step_cells)
    path_body = [_cell_to_world(c, res, ox, oy) for c in path]
    # Anchor the path at the robot so the follower's lookahead starts there.
    if path_body and math.hypot(path_body[0][0], path_body[0][1]) > res:
        path_body = [(0.0, 0.0)] + path_body
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
