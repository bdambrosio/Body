"""Grid-agnostic 8-connected A* (pure Python + numpy).

The Pi-side sibling of the A* in ``desktop/nav/planner.py`` — same algorithm,
extracted as a pure core so the local planner (``body/lib/local_planner.py``)
can run it on the tiny body-frame scan grid. No world-frame coupling, no Qt,
no zenoh. Operates on cell-index arrays: a float ``cost`` grid (halo cost) and
a bool ``lethal`` grid.

Edge cost = base + cost_per_unit * cost[neighbor], base = 1.0 cardinal / √2
diagonal. Lethal cells are impassable. ``cost_per_unit`` turns the costmap halo
(0..halo_max) into "extra distance equivalents" so A* bows toward clearance when
there's lateral room, without forbidding tight passages.
"""
from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

Cell = Tuple[int, int]

_NEIGHBORS_8 = (
    (-1, -1, math.sqrt(2.0)), (-1, 0, 1.0), (-1, 1, math.sqrt(2.0)),
    (0, -1, 1.0),                           (0, 1, 1.0),
    (1, -1, math.sqrt(2.0)), (1, 0, 1.0), (1, 1, math.sqrt(2.0)),
)


def _octile(a: Cell, b: Cell) -> float:
    di = abs(a[0] - b[0])
    dj = abs(a[1] - b[1])
    return (di + dj) + (math.sqrt(2.0) - 2.0) * min(di, dj)


def _reconstruct(parents: Dict[Cell, Cell], end: Cell) -> List[Cell]:
    out = [end]
    cur = end
    while cur in parents:
        cur = parents[cur]
        out.append(cur)
    out.reverse()
    return out


def astar_8c(
    *,
    cost: np.ndarray,
    lethal: np.ndarray,
    start: Cell,
    goal: Cell,
    cost_per_unit: float = 0.10,
    h_weight: float = 1.0,
    max_expansions: int = 50_000,
) -> Tuple[Optional[List[Cell]], int, str]:
    """Return (cells_path | None, n_expansions, msg). Path is start..goal
    inclusive in (i, j) cell pairs; None on failure ("no path" /
    "max_expansions exceeded")."""
    nx, ny = cost.shape
    if start == goal:
        return [start], 0, "ok"

    g: Dict[Cell, float] = {start: 0.0}
    parents: Dict[Cell, Cell] = {}
    open_heap: List[Tuple[float, int, Cell]] = []
    counter = 0
    heapq.heappush(open_heap, (_octile(start, goal) * h_weight, counter, start))
    closed: Dict[Cell, bool] = {}
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
            if lethal[ni, nj] or (ni, nj) in closed:
                continue
            step = base + cost_per_unit * float(cost[ni, nj])
            tentative = cur_g + step
            prev = g.get((ni, nj))
            if prev is not None and tentative >= prev:
                continue
            g[(ni, nj)] = tentative
            parents[(ni, nj)] = cur
            counter += 1
            heapq.heappush(
                open_heap, (tentative + h_weight * _octile((ni, nj), goal), counter, (ni, nj)))

    return None, expansions, "no path"


def astar_toward(
    *,
    cost: np.ndarray,
    lethal: np.ndarray,
    start: Cell,
    goal: Cell,
    cost_per_unit: float = 0.10,
    h_weight: float = 1.0,
    max_expansions: int = 50_000,
) -> Tuple[Optional[List[Cell]], int, str]:
    """Like ``astar_8c``, but goal-unreachable is not failure: return the path
    to the **reachable** cell closest (octile) to ``goal``. This is what makes
    the Tier-3 sub-goal reachable by construction — the robot always heads as
    far toward the requested point as the live free space allows (rounding
    corners), and only reports "boxed in" when no reachable cell is any closer
    than the start.

    Returns (cells_path, n_expansions, msg) with msg:
      * "ok"       — reached the goal exactly,
      * "frontier" — best-effort path to the closest reachable cell,
      * "no path"  — nothing reachable makes progress (path is just [start])."""
    nx, ny = cost.shape
    if start == goal:
        return [start], 0, "ok"
    g: Dict[Cell, float] = {start: 0.0}
    parents: Dict[Cell, Cell] = {}
    open_heap: List[Tuple[float, int, Cell]] = []
    counter = 0
    heapq.heappush(open_heap, (_octile(start, goal) * h_weight, counter, start))
    closed: Dict[Cell, bool] = {}
    expansions = 0
    best_cell: Cell = start
    best_h: float = _octile(start, goal)
    while open_heap:
        _f, _cnt, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        closed[cur] = True
        h_cur = _octile(cur, goal)
        if h_cur < best_h:
            best_h, best_cell = h_cur, cur
        if cur == goal:
            return _reconstruct(parents, cur), expansions, "ok"
        expansions += 1
        if expansions > max_expansions:
            break
        ci, cj = cur
        cur_g = g[cur]
        for di, dj, base in _NEIGHBORS_8:
            ni, nj = ci + di, cj + dj
            if not (0 <= ni < nx and 0 <= nj < ny):
                continue
            if lethal[ni, nj] or (ni, nj) in closed:
                continue
            step = base + cost_per_unit * float(cost[ni, nj])
            tentative = cur_g + step
            prev = g.get((ni, nj))
            if prev is not None and tentative >= prev:
                continue
            g[(ni, nj)] = tentative
            parents[(ni, nj)] = cur
            counter += 1
            heapq.heappush(
                open_heap,
                (tentative + h_weight * _octile((ni, nj), goal), counter, (ni, nj)))
    if best_cell == start:
        return [start], expansions, "no path"
    return _reconstruct(parents, best_cell), expansions, "frontier"


def nearest_non_lethal(
    lethal: np.ndarray, i: int, j: int, *, radius: int,
) -> Optional[Cell]:
    """Nearest non-lethal cell to (i, j), searching outward in rings.
    Returns (i, j) or None within `radius`."""
    nx, ny = lethal.shape
    if 0 <= i < nx and 0 <= j < ny and not lethal[i, j]:
        return (i, j)
    for r in range(1, radius + 1):
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if max(abs(di), abs(dj)) != r:
                    continue
                ii, jj = i + di, j + dj
                if 0 <= ii < nx and 0 <= jj < ny and not lethal[ii, jj]:
                    return (ii, jj)
    return None
