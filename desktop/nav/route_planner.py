"""Live global route planning for the hierarchical drive (nav redesign, step 1).

Until 2026-10-31 the patrol was A*-expanded on the world costmap ONCE at Go
and the dense vertices then became hard targets that Tier-2 marched toward
in a straight line on the live scan. Nothing re-consulted the map, and a
vertex that ended up inside an obstacle's inflation could only be fought:
BLOCKED, retry, pause (docs/nav_redesign.md).

This module replaces that with:

  * ``plan_route``: A* on the LIVE world costmap from the robot to the
    current via-point, with the via-point **relaxed** to the nearest free
    cell within ``via_radius_m`` (a marker 20 cm inside a cabinet's
    inflation is a route hint, not a destination) or reported unreachable so
    the caller can skip it. Obstacles the map may not hold yet (Tier-3's
    depth-veto hits / depth memory) are stamped lethal for the plan.
  * ``visible_subgoal_along_path``: the furthest point along that path, up
    to ``horizon_m``, that the live scan raster (footprint-inflated) shows
    clear — the one thing Tier-2 was right to insist on. Tier-3 gets THAT
    as its goal, so it never has to invent a way around something the map
    already knows.

Pure functions; the orchestrator (``HierarchicalDrive``) owns state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from desktop.nav.planner import AStarConfig, _dilate_bool, _nearest_non_lethal, plan_path

Pt = Tuple[float, float]
Pose = Tuple[float, float, float]


@dataclass(frozen=True)
class RouteConfig:
    # A via-point whose cell is lethal is moved to the nearest non-lethal
    # cell within this radius; none within it → "via_unreachable".
    via_radius_m: float = 0.50
    # Sub-goal is at most this far along the path (Tier-3's local raster is
    # 2.5 m half-extent; keep context beyond the goal).
    horizon_m: float = 1.5
    # A sub-goal closer than this is no use (Tier-3's arrival tolerance +
    # one cell): the caller falls back / blocks.
    min_subgoal_m: float = 0.20
    # Step along the path when checking live visibility (≈ half a cell).
    march_step_m: float = 0.04
    # Pull the visible sub-goal back from the first non-clear cell.
    backoff_m: float = 0.04
    astar: AStarConfig = field(default_factory=AStarConfig)


@dataclass
class RoutePlan:
    ok: bool
    reason: str                       # ok | via_unreachable | no_route | no_costmap | <plan_path msg>
    path_world: List[Pt]              # dense, robot → target (empty on failure)
    target_world: Optional[Pt]        # the (possibly relaxed) via-point actually planned to
    relaxed_m: float = 0.0            # how far the via-point was moved
    distance_m: float = 0.0
    elapsed_ms: float = 0.0


def _world_to_cell(xy: Pt, meta: dict) -> Tuple[int, int]:
    res = float(meta["resolution_m"])
    return (int(math.floor((xy[0] - float(meta["origin_x_m"])) / res + 1e-9)),
            int(math.floor((xy[1] - float(meta["origin_y_m"])) / res + 1e-9)))


def _cell_to_world(ij: Tuple[int, int], meta: dict) -> Pt:
    res = float(meta["resolution_m"])
    return (float(meta["origin_x_m"]) + (ij[0] + 0.5) * res,
            float(meta["origin_y_m"]) + (ij[1] + 0.5) * res)


def stamp_lethal_world(costmap: Any, points_world: Iterable[Pt]) -> Any:
    """A shallow copy of ``costmap`` whose lethal mask also holds
    ``points_world`` (e.g. Tier-3 depth hits the map doesn't carry yet),
    inflated by the costmap's footprint radius exactly like a mapped
    blocked cell — so the global path keeps the same clearance from them
    that the live scan check demands (a bare cell let the path hug a
    remembered obstacle the live check then rejected: path_blocked_live
    every retry, 10-31). Returns the original object when nothing stamps."""
    pts = [p for p in points_world]
    if not pts:
        return costmap
    stamp = np.zeros(costmap.lethal.shape, dtype=bool)
    nx, ny = stamp.shape
    n = 0
    for p in pts:
        i, j = _world_to_cell(p, costmap.meta)
        if 0 <= i < nx and 0 <= j < ny:
            stamp[i, j] = True
            n += 1
    if n == 0:
        return costmap
    res = float(costmap.meta["resolution_m"])
    fp = float(getattr(getattr(costmap, "config", None), "footprint_radius_m", 0.15))
    iters = int(math.ceil(fp / res))
    if iters > 0:
        stamp = _dilate_bool(stamp, iters=iters)
    return replace(costmap, lethal=costmap.lethal | stamp)


def relax_via(costmap: Any, via: Pt, radius_m: float, *,
              min_clearance_cells: int) -> Optional[Pt]:
    """``via`` itself when its (clearance-dilated) cell is free, else the
    nearest free cell within ``radius_m``, else None."""
    meta = costmap.meta
    res = float(meta["resolution_m"])
    lethal = (_dilate_bool(costmap.lethal, iters=int(min_clearance_cells))
              if min_clearance_cells > 0 else costmap.lethal)
    i, j = _world_to_cell(via, meta)
    nx, ny = lethal.shape
    if not (0 <= i < nx and 0 <= j < ny):
        return None
    if not lethal[i, j]:
        return via
    radius = max(1, int(math.ceil(radius_m / res)))
    rel = _nearest_non_lethal(lethal, i, j, radius=radius)
    if rel is None:
        return None
    wx, wy = _cell_to_world(rel, meta)
    if math.hypot(wx - via[0], wy - via[1]) > radius_m + 0.5 * res:
        return None
    return (wx, wy)


def plan_route(costmap: Any, pose_xy: Pt, via: Pt, cfg: RouteConfig, *,
               extra_lethal_world: Sequence[Pt] = ()) -> RoutePlan:
    """Global path robot → (relaxed) via on the live costmap."""
    if costmap is None:
        return RoutePlan(False, "no_costmap", [], None)
    cm = stamp_lethal_world(costmap, extra_lethal_world)
    target = relax_via(cm, via, cfg.via_radius_m,
                       min_clearance_cells=cfg.astar.min_clearance_cells)
    if target is None:
        return RoutePlan(False, "via_unreachable", [], None)
    res = plan_path(cm, pose_xy, target, cfg.astar)
    if not res.ok:
        return RoutePlan(False, "no_route" if res.msg == "no path" else res.msg, [], target)
    path = list(res.waypoints_world)
    # plan_path's cells start at the robot's (relaxed) cell centre: lead with
    # the exact robot position so the body-frame march starts at (0, 0).
    if not path or math.hypot(path[0][0] - pose_xy[0], path[0][1] - pose_xy[1]) > 1e-6:
        path = [pose_xy] + path
    return RoutePlan(True, "ok", path, target,
                     relaxed_m=math.hypot(target[0] - via[0], target[1] - via[1]),
                     distance_m=res.distance_m, elapsed_ms=res.elapsed_ms)


def path_to_body(path_world: Sequence[Pt], pose: Pose) -> List[Pt]:
    c, s = math.cos(pose[2]), math.sin(pose[2])
    out = []
    for x, y in path_world:
        dx, dy = x - pose[0], y - pose[1]
        out.append((c * dx + s * dy, -s * dx + c * dy))
    return out


def visible_subgoal_along_path(
    path_body: Sequence[Pt], goal_grid: np.ndarray, meta: dict, cfg: RouteConfig,
    *, path_grid: Optional[np.ndarray] = None, max_dist_m: Optional[float] = None,
    robot_clear_radius_m: float = 0.0,
) -> Optional[Tuple[Pt, float]]:
    """Furthest point along ``path_body`` (from the robot at the origin) the
    live scan shows drivable, up to ``horizon_m`` of path length (and
    ``max_dist_m`` if given).

    Two masks (int8: 1 clear, else not): the path is walked on ``path_grid``
    (footprint-only inflation — physically passable; Tier-3's swept veto is
    the real gate), the chosen END point must be clear on ``goal_grid``
    (Tier-3's full lethal + goal clearance, so it accepts the goal without
    snapping). One mask for both made a robot standing 0.2 m from an
    obstacle unable to leave along any path (10-31). Samples within
    ``robot_clear_radius_m`` of the robot are never obstructions (it is
    already there). Returns ((bx, by), path_dist) or None when no sample
    ≥ ``min_subgoal_m`` qualifies. Off-grid counts as open."""
    if len(path_body) < 2:
        return None
    if path_grid is None:
        path_grid = goal_grid
    res = float(meta["resolution_m"])
    ox, oy = float(meta["origin_x_m"]), float(meta["origin_y_m"])
    nx, ny = goal_grid.shape
    limit = cfg.horizon_m if max_dist_m is None else min(cfg.horizon_m, max_dist_m)
    step = cfg.march_step_m

    def cell(px: float, py: float):
        i = int(math.floor((px - ox) / res))
        j = int(math.floor((py - oy) / res))
        return (i, j) if (0 <= i < nx and 0 <= j < ny) else None

    samples: List[Tuple[Pt, float]] = []       # path-clear samples in order
    obstructed = False
    walked = 0.0
    done = False
    for (ax, ay), (bx, by) in zip(path_body, path_body[1:]):
        seg = math.hypot(bx - ax, by - ay)
        if seg < 1e-9:
            continue
        n = max(1, int(math.ceil(seg / step)))
        for k in range(1, n + 1):
            t = k / n
            px, py = ax + t * (bx - ax), ay + t * (by - ay)
            d = walked + t * seg
            if d > limit:
                done = True
                break
            c = cell(px, py)
            near_robot = math.hypot(px, py) <= robot_clear_radius_m
            if c is not None and not near_robot and int(path_grid[c]) != 1:
                obstructed = True
                done = True
                break
            samples.append(((px, py), d))
        walked += seg
        if done:
            break
    if not samples:
        return None
    # Pull back from an obstruction, then take the furthest sample Tier-3
    # would accept as a goal.
    d_cap = samples[-1][1] - (cfg.backoff_m if obstructed else 0.0)
    for (px, py), d in reversed(samples):
        if d > d_cap or d < cfg.min_subgoal_m:
            continue
        c = cell(px, py)
        if c is None or int(goal_grid[c]) == 1:
            return (px, py), d
    return None


def _back_off(path_body: Sequence[Pt], d: float, back: float) -> Pt:
    """Point at path distance d - back along the polyline."""
    target = max(0.0, d - back)
    walked = 0.0
    for (ax, ay), (bx, by) in zip(path_body, path_body[1:]):
        seg = math.hypot(bx - ax, by - ay)
        if walked + seg >= target and seg > 1e-9:
            t = (target - walked) / seg
            return (ax + t * (bx - ax), ay + t * (by - ay))
        walked += seg
    return path_body[-1]


# ── Planning map = reference-map prior + live session grid ───────────────

def resample_prior(ref_driveable: np.ndarray, ref_meta: dict, grid_meta: dict) -> np.ndarray:
    """``ref_driveable`` (int8 1/0/-1 on the reference map's grid) sampled
    (nearest cell) onto the world grid described by ``grid_meta``
    (resolution_m, origin_x_m, origin_y_m, nx, ny). Cells outside the
    reference map are unknown (-1)."""
    nx, ny = int(grid_meta["nx"]), int(grid_meta["ny"])
    res = float(grid_meta["resolution_m"])
    ox, oy = float(grid_meta["origin_x_m"]), float(grid_meta["origin_y_m"])
    rres = float(ref_meta["resolution_m"])
    rox, roy = float(ref_meta["origin_x_m"]), float(ref_meta["origin_y_m"])
    xs = ox + (np.arange(nx) + 0.5) * res
    ys = oy + (np.arange(ny) + 0.5) * res
    ri = np.floor((xs - rox) / rres).astype(np.int64)
    rj = np.floor((ys - roy) / rres).astype(np.int64)
    out = np.full((nx, ny), -1, dtype=np.int8)
    oki = (ri >= 0) & (ri < ref_driveable.shape[0])
    okj = (rj >= 0) & (rj < ref_driveable.shape[1])
    ii, jj = np.nonzero(oki)[0], np.nonzero(okj)[0]
    if ii.size and jj.size:
        out[np.ix_(ii, jj)] = ref_driveable[np.ix_(ri[ii], rj[jj])]
    return out


def merge_prior(session: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Planning driveable layer: what this session has observed wins; the
    reference-map prior fills what it has not seen yet. (The session grid
    starts empty on every nav start — with unknown lethal to the planner,
    the next room was a wall until the robot had looked at it: no_route
    at waypoints 3 and 6, 10-31.)"""
    return np.where(session != -1, session, prior).astype(np.int8)


def path_length(path: Sequence[Pt]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:]))
