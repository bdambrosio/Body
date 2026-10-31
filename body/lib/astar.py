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
    goal_ok: Optional[np.ndarray] = None,
    tie_cells: float = 0.0,
    detour_ratio: float = 3.0,
    detour_slack_cells: float = 4.0,
) -> Tuple[Optional[List[Cell]], int, str]:
    """Like ``astar_8c``, but goal-unreachable is not failure: return the path
    to the **reachable** cell closest to ``goal``. This is what makes the
    Tier-3 sub-goal reachable by construction — the robot always heads as far
    toward the requested point as the live free space allows (rounding
    corners), and only reports "no path" when no reachable cell is any closer
    than the start.

    "Closest" is Euclidean (cell units) to the goal, ties broken by the lower
    path cost ``g`` — so a goal that lands on/behind a wall resolves to the
    best cell on the robot's side of it, never to a cell behind the wall in
    scan order (T6). The goal cell itself may be lethal (that is how a goal
    is "snapped"). ``goal_ok`` (optional bool mask) restricts which cells may
    be the endpoint (e.g. extra goal clearance); when no reachable cell
    satisfies it, the best reachable cell is used anyway. ``tie_cells`` > 0
    treats endpoints within that many cells of the best distance as ties and
    picks the cheapest of them. ``detour_ratio``: a cell whose geometric
    path length exceeds ``detour_ratio`` × its straight-line (octile)
    distance from the start + ``detour_slack_cells`` is a detour and is not
    eligible as an endpoint — a cell one cell nearer the goal on the far
    side of a wall, reachable only via a 75-cell loop, must not beat the
    near side (10-31: a 6 m loop for a 0.5 m goal, spinning in place toward
    it). A hard gate, not a penalty: ranking among eligible cells stays pure
    distance, so the bend around a post (ratio < 2) does not shuffle the
    endpoint as the grid re-quantizes. 0 disables.

    Returns (cells_path, n_expansions, msg) with msg:
      * "ok"       — reached the goal exactly,
      * "frontier" — best-effort path to the closest reachable cell,
      * "no path"  — nothing reachable makes progress (path is just [start])."""
    nx, ny = cost.shape
    goal_accept = goal_ok is None or (
        0 <= goal[0] < nx and 0 <= goal[1] < ny and bool(goal_ok[goal]))
    if start == goal and goal_accept:
        return [start], 0, "ok"
    g: Dict[Cell, float] = {start: 0.0}
    plen: Dict[Cell, float] = {start: 0.0}      # geometric path length (cells)
    parents: Dict[Cell, Cell] = {}
    open_heap: List[Tuple[float, int, Cell]] = []
    counter = 0
    heapq.heappush(open_heap, (_octile(start, goal) * h_weight, counter, start))
    closed: Dict[Cell, bool] = {}
    expansions = 0
    gi, gj = goal

    def _key(c: Cell) -> Tuple[float, float]:
        d = math.hypot(c[0] - gi, c[1] - gj)
        # Geometric length, not g: g carries costmap cost, which would brand
        # a straight approach along a wall as a detour.
        if (detour_ratio > 0 and c != start
                and plen[c] > detour_ratio * _octile(start, c) + detour_slack_cells):
            d += 1e6                        # ineligible unless nothing else progresses
        return (d, g[c])

    start_key = _key(start)
    best_cell: Cell = start
    best_key = start_key
    best_ok_cell: Optional[Cell] = (
        start if goal_ok is None or bool(goal_ok[start]) else None)
    best_ok_key = start_key
    while open_heap:
        _f, _cnt, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        closed[cur] = True
        k = _key(cur)
        if k < best_key:
            best_key, best_cell = k, cur
        if goal_ok is not None and bool(goal_ok[cur]) and (
                best_ok_cell is None or k < best_ok_key):
            best_ok_key, best_ok_cell = k, cur
        if cur == goal and goal_accept:
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
            plen[(ni, nj)] = plen[cur] + base
            parents[(ni, nj)] = cur
            counter += 1
            heapq.heappush(
                open_heap,
                (tentative + h_weight * _octile((ni, nj), goal), counter, (ni, nj)))
    if goal_ok is not None and best_ok_cell is not None:
        best_cell = best_ok_cell
    if tie_cells > 0 and best_cell != start:
        # Near-ties (within tie_cells of the best distance) go to the cheapest
        # path: the approach side, stable as the grid re-quantizes each tick
        # (a ring of equally-near cells around a lethal goal otherwise flips
        # the endpoint — and the path's side — from tick to tick).
        dbest = math.hypot(best_cell[0] - gi, best_cell[1] - gj)
        cand = best_cell
        for c in closed:
            if c == start:
                continue            # standing still is never a tie-break winner
            if goal_ok is not None and best_ok_cell is not None and not bool(goal_ok[c]):
                continue
            d = math.hypot(c[0] - gi, c[1] - gj)
            if d <= dbest + tie_cells and g[c] < g[cand]:
                cand = c
        best_cell = cand
    # "No progress" = no reachable cell strictly closer to the goal than the
    # start (a lower-g tie at the same distance is not progress).
    if best_cell == start or _key(best_cell)[0] >= start_key[0] - 1e-9:
        return [start], expansions, "no path"
    return _reconstruct(parents, best_cell), expansions, "frontier"


def nearest_non_lethal(
    lethal: np.ndarray, i: int, j: int, *, radius: int,
    blocked: Optional[np.ndarray] = None,
) -> Optional[Cell]:
    """Nearest (Euclidean) non-lethal cell to (i, j) within `radius` (ring
    distance), or None. Ties keep scan order. ``blocked`` (optional raw
    obstacle mask): skip candidates whose straight segment from (i, j)
    crosses a blocked cell — relaxing a lethal start must not hop through
    the wall that made it lethal."""
    nx, ny = lethal.shape
    if 0 <= i < nx and 0 <= j < ny and not lethal[i, j]:
        return (i, j)
    cands: List[Tuple[float, int, Cell]] = []
    n = 0
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            if di == 0 and dj == 0:
                continue
            ii, jj = i + di, j + dj
            if 0 <= ii < nx and 0 <= jj < ny and not lethal[ii, jj]:
                cands.append((math.hypot(di, dj), n, (ii, jj)))
                n += 1
    cands.sort()
    for _d, _n, c in cands:
        if blocked is None or not _segment_hits(blocked, (i, j), c):
            return c
    return None


def _segment_hits(mask: np.ndarray, a: Cell, b: Cell) -> bool:
    """True if any cell on the sampled segment a→b (inclusive) is set."""
    nx, ny = mask.shape
    steps = max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2
    for k in range(steps + 1):
        t = k / steps if steps else 0.0
        ii = int(round(a[0] + t * (b[0] - a[0])))
        jj = int(round(a[1] + t * (b[1] - a[1])))
        if 0 <= ii < nx and 0 <= jj < ny and mask[ii, jj]:
            return True
    return False
