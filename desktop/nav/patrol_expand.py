"""Tier-1 global patrol expansion.

Expand a sparse, hand-placed patrol into a dense chain of sub-waypoints by
routing each high-level segment with **global A* on the reference-map costmap**
(`planner.plan_path`), then resampling the path so the straight hop between
consecutive sub-waypoints stays in free space (routes *around* dead-end pockets)
and is short enough to sit inside Tier-3's local horizon. The expanded patrol
feeds the unchanged `PatrolRunner` → Tier-2 → Tier-3 pipeline.

Scope (v1): only the **inter-waypoint** segments (Wi→Wi+1, plus Wn→W0 when the
patrol loops) are globally routed. The one-time lead-in from the robot's current
pose to the first waypoint still uses Tier-2's greedy projection, so place the
first waypoint reachable from the start (typically the robot starts on the loop).
Routing the lead-in is a clean v2 follow-up.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from desktop.nav.patrol import Patrol, Waypoint
from desktop.nav.planner import AStarConfig, plan_path

Pt = Tuple[float, float]


@dataclass
class ExpandConfig:
    # Sub-waypoint spacing cap. Must stay under Tier-3's local horizon
    # (scan half_extent ≈ 2.5 m) so the next sub-waypoint is always reachable.
    max_spacing_m: float = 1.5
    # Keep a vertex wherever the path heading turns more than this (~20°),
    # so a straight hop between sub-waypoints never cuts a corner into a pocket.
    corner_thresh_rad: float = 0.35
    astar: AStarConfig = field(default_factory=AStarConfig)


@dataclass
class ExpandResult:
    ok: bool
    patrol: Optional[Patrol]
    reason: str = ""
    # (from_index, to_index) into the high-level route when a segment is
    # unreachable, for an operator-facing message. None on success.
    failed_segment: Optional[Tuple[int, int]] = None


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _heading(a: Pt, b: Pt) -> float:
    return math.atan2(b[1] - a[1], b[0] - a[0])


def _close(a: Pt, b: Pt, tol: float = 1e-3) -> bool:
    return math.hypot(a[0] - b[0], a[1] - b[1]) <= tol


def resample_path(
    path: List[Pt], *, max_spacing_m: float, corner_thresh_rad: float,
) -> List[Pt]:
    """Resample a dense world polyline into sub-waypoints.

    Walks the polyline emitting a point at pts[i] when EITHER the path heading
    has turned more than ``corner_thresh_rad`` since the last emit (a corner) OR
    keeping it would let the next point's straight-line hop from the last emit
    exceed ``max_spacing_m``. Emitting *on the path* (not interpolating between
    sparse vertices) follows curves; the look-ahead keeps every hop ≤ the cap.
    Endpoints are always kept.
    """
    pts: List[Pt] = []
    for x, y in path:                       # drop consecutive duplicates
        p = (float(x), float(y))
        if not pts or not _close(pts[-1], p, 1e-9):
            pts.append(p)
    if len(pts) < 2:
        return list(pts)

    out: List[Pt] = [pts[0]]
    ref_h = _heading(pts[0], pts[1])        # heading leaving the last emit
    for i in range(1, len(pts) - 1):
        out_h = _heading(pts[i], pts[i + 1])
        corner = abs(_wrap(out_h - ref_h)) > corner_thresh_rad
        nxt = pts[i + 1]
        overflow = math.hypot(nxt[0] - out[-1][0], nxt[1] - out[-1][1]) > max_spacing_m
        if corner or overflow:
            out.append(pts[i])
            ref_h = out_h
    out.append(pts[-1])
    return out


def _mk_patrol(src: Patrol, pts: List[Pt]) -> Patrol:
    return Patrol(
        name=src.name, session_id=src.session_id, authored_utc=src.authored_utc,
        loop=src.loop, laps=src.laps,
        # face_next is irrelevant to the hierarchical driver (it never rotates
        # in place at a waypoint); intermediates are pure path-following points.
        waypoints=[Waypoint(x_m=x, y_m=y, face_next=False) for (x, y) in pts],
    )


def expand_patrol(
    patrol: Patrol, costmap, cfg: Optional[ExpandConfig] = None,
) -> ExpandResult:
    """Globally route + resample each high-level segment into sub-waypoints.

    `costmap` is a `desktop.world_map.costmap.Costmap` over the reference map.
    Returns the expanded `Patrol`, or `ok=False` with the unreachable segment.
    """
    cfg = cfg or ExpandConfig()
    hi: List[Pt] = [(w.x_m, w.y_m) for w in patrol.waypoints]
    if not hi:
        return ExpandResult(False, None, "patrol has no waypoints")
    if len(hi) == 1:
        return ExpandResult(True, _mk_patrol(patrol, hi))

    # Route = consecutive high-level waypoints, closing back to the first when
    # the patrol loops. Indices are into this `route` list for error reporting.
    route: List[Pt] = list(hi)
    if patrol.loop:
        route = route + [hi[0]]

    out: List[Pt] = []
    for k in range(len(route) - 1):
        res = plan_path(costmap, route[k], route[k + 1], cfg.astar)
        if not res.ok:
            return ExpandResult(
                False, None,
                f"no route for segment {k}->{k + 1}: {res.msg}",
                failed_segment=(k, k + 1),
            )
        sub = resample_path(
            res.waypoints_world,
            max_spacing_m=cfg.max_spacing_m,
            corner_thresh_rad=cfg.corner_thresh_rad,
        )
        if out and sub and _close(out[-1], sub[0]):
            sub = sub[1:]                   # dedupe the shared segment endpoint
        out.extend(sub)

    # For a loop, drop a duplicated closing point so PatrolRunner's own
    # last->first closure is the real final leg (not a zero-length hop).
    if patrol.loop and len(out) >= 2 and _close(out[-1], out[0]):
        out = out[:-1]

    if len(out) < 2:
        return ExpandResult(False, None, "expanded route degenerate")
    return ExpandResult(True, _mk_patrol(patrol, out))
