"""A* path planner over a Costmap.

Pure Python + numpy, 8-connected, octile heuristic. The cost-step
between two adjacent cells is

    edge = base + cost_per_unit * cm.cost[neighbor]

where `base` is 1.0 for cardinal moves and √2 for diagonals, and
`cost_per_unit` translates halo cost (typically 0..100 from
`CostmapConfig.halo_max`) into "extra distance equivalents." With
defaults, a cell deep in the halo (cost ≈ halo_max) adds 5.0 to
the path length — a strong "hug the center" incentive without making
halo cells un-traversable.

Lethal cells are absolutely impassable (skipped during expansion).
Unknown cells inherit `unknown_cost` from CostmapConfig; the planner
will route through them when no observed-clear path exists, but
prefers observed-clear when one does.

Performance: for the 500×500 maps the fuser publishes, typical
goals (a few meters away) expand 1–10 k nodes; ~50–200 ms in pure
Python on a Pi-class host. One-shot per goal-set, not per tick,
so the budget is generous.
"""
from __future__ import annotations

import heapq
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from desktop.world_map.costmap import Costmap

logger = logging.getLogger(__name__)


# ── Config + result types ──────────────────────────────────────────


@dataclass
class AStarConfig:
    cost_per_unit: float = 0.05      # halo_max=100 → 5.0 distance penalty
    heuristic_weight: float = 1.0    # 1.0 = optimal; >1 = greedy/faster
    max_expansions: int = 200_000    # safety cap before declaring failure


@dataclass
class PlanResult:
    ok: bool
    msg: str
    waypoints_world: List[Tuple[float, float]]    # [(x_w, y_w), ...]
    waypoints_cells: List[Tuple[int, int]]
    distance_m: float                              # path length, end-to-end
    n_expansions: int
    elapsed_ms: float

    @classmethod
    def fail(cls, msg: str, *, n: int = 0, elapsed_ms: float = 0.0) -> "PlanResult":
        return cls(
            ok=False, msg=msg, waypoints_world=[], waypoints_cells=[],
            distance_m=0.0, n_expansions=n, elapsed_ms=elapsed_ms,
        )


# ── Public API ─────────────────────────────────────────────────────


def plan_path(
    costmap: Costmap,
    start_world: Tuple[float, float],
    goal_world: Tuple[float, float],
    config: Optional[AStarConfig] = None,
) -> PlanResult:
    """Plan an 8-connected path on `costmap` from `start_world` to
    `goal_world` (both in world meters). Returns a `PlanResult`."""
    cfg = config or AStarConfig()
    t0 = time.monotonic()

    res = float(costmap.meta["resolution_m"])
    ox = float(costmap.meta["origin_x_m"])
    oy = float(costmap.meta["origin_y_m"])

    si, sj = _world_to_cell(start_world, res, ox, oy)
    gi, gj = _world_to_cell(goal_world, res, ox, oy)

    nx, ny = costmap.cost.shape
    if not (0 <= si < nx and 0 <= sj < ny):
        return PlanResult.fail("start out of grid")
    if not (0 <= gi < nx and 0 <= gj < ny):
        return PlanResult.fail(
            "goal out of grid (pan view further or drive closer)",
        )

    if costmap.lethal[si, sj]:
        # Try a small relaxation: nearest non-lethal cell within a
        # 5-cell neighborhood. Common when the robot is pinned to
        # the inflation halo of its own pose-trail false positives.
        relaxed = _nearest_non_lethal(costmap.lethal, si, sj, radius=5)
        if relaxed is None:
            return PlanResult.fail("start cell is lethal — robot stuck?")
        si, sj = relaxed
    if costmap.lethal[gi, gj]:
        relaxed = _nearest_non_lethal(costmap.lethal, gi, gj, radius=8)
        if relaxed is None:
            return PlanResult.fail("goal is inside an obstacle (or its halo)")
        gi, gj = relaxed

    cells, n_expansions, msg = _astar_8c(
        cost=costmap.cost,
        lethal=costmap.lethal,
        start=(si, sj),
        goal=(gi, gj),
        cost_per_unit=cfg.cost_per_unit,
        h_weight=cfg.heuristic_weight,
        max_expansions=cfg.max_expansions,
    )
    elapsed_ms = (time.monotonic() - t0) * 1000.0
    if cells is None:
        return PlanResult.fail(msg, n=n_expansions, elapsed_ms=elapsed_ms)

    waypoints_world = [_cell_to_world(c, res, ox, oy) for c in cells]
    distance_m = _path_distance_m(waypoints_world)

    logger.debug(
        f"plan_path: ok cells={len(cells)} dist={distance_m:.2f} m "
        f"expansions={n_expansions} elapsed={elapsed_ms:.1f} ms"
    )
    return PlanResult(
        ok=True, msg="ok",
        waypoints_world=waypoints_world,
        waypoints_cells=cells,
        distance_m=distance_m,
        n_expansions=n_expansions,
        elapsed_ms=elapsed_ms,
    )


# ── Core A* ─────────────────────────────────────────────────────────


_NEIGHBORS_8 = (
    (-1, -1, math.sqrt(2.0)), (-1, 0, 1.0), (-1, 1, math.sqrt(2.0)),
    ( 0, -1, 1.0),                          ( 0, 1, 1.0),
    ( 1, -1, math.sqrt(2.0)), ( 1, 0, 1.0), ( 1, 1, math.sqrt(2.0)),
)


def _astar_8c(
    *,
    cost: np.ndarray,
    lethal: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    cost_per_unit: float,
    h_weight: float,
    max_expansions: int,
) -> Tuple[Optional[List[Tuple[int, int]]], int, str]:
    """Return (cells_path, expansions, msg). cells_path is None on
    failure. Path is start..goal inclusive, in cell-index pairs."""
    nx, ny = cost.shape
    if start == goal:
        return [start], 0, "ok"

    # g_score: dict cell → cumulative cost. Open: heap of (f, counter, cell).
    # Counter breaks ties so heapq doesn't try to compare cells.
    g: Dict[Tuple[int, int], float] = {start: 0.0}
    parents: Dict[Tuple[int, int], Tuple[int, int]] = {}
    open_heap: List[Tuple[float, int, Tuple[int, int]]] = []
    counter = 0
    h_start = _octile(start, goal) * h_weight
    heapq.heappush(open_heap, (h_start, counter, start))

    closed: Dict[Tuple[int, int], bool] = {}
    expansions = 0

    while open_heap:
        _f, _cnt, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        closed[cur] = True
        if cur == goal:
            return _reconstruct(parents, cur), expansions, "ok"
        expansions += 1
        if expansions > max_expansions:
            return None, expansions, "max_expansions exceeded"

        ci, cj = cur
        cur_g = g[cur]
        for di, dj, base in _NEIGHBORS_8:
            ni, nj = ci + di, cj + dj
            if not (0 <= ni < nx and 0 <= nj < ny):
                continue
            if lethal[ni, nj]:
                continue
            if (ni, nj) in closed:
                continue
            step = base + cost_per_unit * float(cost[ni, nj])
            tentative = cur_g + step
            prev = g.get((ni, nj))
            if prev is not None and tentative >= prev:
                continue
            g[(ni, nj)] = tentative
            parents[(ni, nj)] = cur
            f = tentative + h_weight * _octile((ni, nj), goal)
            counter += 1
            heapq.heappush(open_heap, (f, counter, (ni, nj)))

    return None, expansions, "no path"


def _octile(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    di = abs(a[0] - b[0])
    dj = abs(a[1] - b[1])
    return (di + dj) + (math.sqrt(2.0) - 2.0) * min(di, dj)


def _reconstruct(
    parents: Dict[Tuple[int, int], Tuple[int, int]],
    end: Tuple[int, int],
) -> List[Tuple[int, int]]:
    out = [end]
    cur = end
    while cur in parents:
        cur = parents[cur]
        out.append(cur)
    out.reverse()
    return out


# ── Helpers ────────────────────────────────────────────────────────


def _world_to_cell(
    xy: Tuple[float, float], res: float, ox: float, oy: float,
) -> Tuple[int, int]:
    x_w, y_w = xy
    i = int(math.floor((x_w - ox) / res + 1e-9))
    j = int(math.floor((y_w - oy) / res + 1e-9))
    return i, j


def _cell_to_world(
    ij: Tuple[int, int], res: float, ox: float, oy: float,
) -> Tuple[float, float]:
    i, j = ij
    return (ox + (i + 0.5) * res, oy + (j + 0.5) * res)


def _nearest_non_lethal(
    lethal: np.ndarray, i: int, j: int, *, radius: int,
) -> Optional[Tuple[int, int]]:
    """Search outward in expanding rings for the nearest non-lethal
    cell. Returns (i, j) or None if none found within radius."""
    nx, ny = lethal.shape
    for r in range(1, radius + 1):
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if max(abs(di), abs(dj)) != r:
                    continue
                ii, jj = i + di, j + dj
                if 0 <= ii < nx and 0 <= jj < ny and not lethal[ii, jj]:
                    return (ii, jj)
    return None


def _path_distance_m(pts: List[Tuple[float, float]]) -> float:
    d = 0.0
    for k in range(1, len(pts)):
        d += math.hypot(pts[k][0] - pts[k - 1][0],
                        pts[k][1] - pts[k - 1][1])
    return d
