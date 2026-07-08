"""Tier-2 visibility sub-goal selection (pure).

Tier-1 (the topological waypoint list) only ever hands the metric loop a
coarse *direction*: the bearing from the robot's current world pose to the
next waypoint. Tier-2 turns that bearing into a concrete point the robot can
actually see right now — the furthest live-visible free point along the
bearing in the body-frame scan grid produced by ``scan_raster.rasterize_scan``
— and hands *that* body-frame point to Tier-3 (``body/drive/goto``). Nothing
from the world map crosses into the drive command except the bearing.

Production (``HierarchicalDrive``) and the ``pi_drive`` Tier-2 console both
use ``furthest_free_point`` / ``plan_tier2`` (the latter wraps the former into
a ``Tier2Decision`` for the debug UI). Hierarchical drive additionally
marches on Tier-3's footprint-inflated lethal mask before calling
``furthest_free_point``; the debug console may pass a raw or pre-masked grid.

Pure NumPy, no zenoh — importable on both the desktop orchestrator and the Pi.
Single ray, not a fan: Tier-3 runs local A* around whatever point we hand it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from body.lib.local_drive_core import wrap_pi


@dataclass(frozen=True)
class Tier2Config:
    horizon_m: float = 1.5          # cap sub-goal distance; < scan half_extent (2.5)
    #   so Tier-3's local A* keeps room/context beyond the goal and the swept
    #   preview doesn't clip the grid edge (2.0 m put the goal at the edge →
    #   unreliable Tier-3 targets).
    step_m: float = 0.04            # ray-march step (~half a cell)
    # Pull the sub-goal back from the first block/unknown. Small on purpose:
    # Tier-3 owns the real footprint safety gate (its own swept-footprint
    # check at radius ~0.14 m), so this only keeps the sub-goal off the
    # obstacle, not a full safety margin. Too large here makes waypoints near
    # walls unreachable and forces tight vetoed arcs.
    backoff_m: float = 0.15
    min_subgoal_m: float = 0.20     # below this → "no usable free point"
    require_clear: bool = True      # treat unknown (-1) as non-free (conservative)


@dataclass(frozen=True)
class Tier2Result:
    ok: bool
    body_xy: Optional[Tuple[float, float]]   # body-frame sub-goal (bx, by), or None
    free_dist_m: float                       # confirmed-clear distance along the bearing
    reason: str                              # "ok"|"blocked_at_origin"|"all_unknown"|"too_short"


@dataclass(frozen=True)
class Tier2Decision:
    """One Tier-2 step packaged for the orchestrator AND the debug console.

    ``plan_tier2`` wraps ``furthest_free_point`` into this shape for the
    pi_drive UI / JSONL trace. Production HierarchicalDrive calls
    ``furthest_free_point`` directly (after footprint-masking the grid).
    """
    bearing_rad: float                       # the CHOSEN bearing (may be off the direct line)
    max_dist_m: float                        # cap = distance to the target/waypoint
    ok: bool
    body_xy: Optional[Tuple[float, float]]
    free_dist_m: float
    reason: str
    capped_at_target: bool                   # clear all the way → sub-goal IS the target
    backoff_applied: bool                    # stopped short of an obstacle (backed off)
    bearing_offset_rad: float = 0.0          # chosen − direct (angular-search swing)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "bearing_rad": self.bearing_rad,
            "max_dist_m": self.max_dist_m,
            "ok": self.ok,
            "body_xy": list(self.body_xy) if self.body_xy is not None else None,
            "free_dist_m": self.free_dist_m,
            "reason": self.reason,
            "capped_at_target": self.capped_at_target,
            "backoff_applied": self.backoff_applied,
            "bearing_offset_rad": self.bearing_offset_rad,
        }


def bearing_to_waypoint(
    rx: float, ry: float, r_yaw: float, wx: float, wy: float,
) -> float:
    """Body-frame bearing from a robot world pose to a world waypoint.

    Result in (−π, π]: 0 = straight ahead (+x body), positive = to the
    robot's left (+y body), matching ``scan_raster`` axis conventions.
    """
    return wrap_pi(math.atan2(wy - ry, wx - rx) - r_yaw)


def furthest_free_point(
    grid: np.ndarray,
    meta: Dict[str, Any],
    bearing_rad: float,
    cfg: Optional[Tier2Config] = None,
    max_dist_m: Optional[float] = None,
) -> Tier2Result:
    """Live-visible free point along ``bearing_rad``, capped at the waypoint.

    March from the robot (body origin) outward along the bearing over the
    int8 scan grid (-1 unknown / 0 blocked / 1 clear). The march stops at the
    first blocked/unknown cell, the grid edge, the horizon, or ``max_dist_m``
    (the distance to the waypoint) — whichever comes first. We never aim
    *past* the waypoint.

      * Path clear all the way to ``max_dist_m`` → the sub-goal IS the
        waypoint (no backoff: it's a goal, not an obstacle).
      * Stopped early by an obstacle/unknown/horizon → the sub-goal is the
        clear distance backed off by ``backoff_m`` so Tier-3's swept-footprint
        check has room.

    ``ok=False`` when the resulting distance is below ``min_subgoal_m``.
    """
    cfg = cfg or Tier2Config()
    res = float(meta["resolution_m"])
    ox = float(meta["origin_x_m"])
    oy = float(meta["origin_y_m"])
    nx = int(meta["nx"])
    ny = int(meta["ny"])

    c = math.cos(bearing_rad)
    s = math.sin(bearing_rad)

    # Hard cap: the waypoint distance (if given), never beyond the horizon.
    limit = cfg.horizon_m if max_dist_m is None else min(cfg.horizon_m, max_dist_m)

    clear_run = 0.0          # furthest distance confirmed clear from the origin
    stop_value = 1           # cell value that ended the run (1 = reached the limit)
    reached_limit = False    # ran out to `limit` with everything clear

    n_steps = int(math.floor(limit / cfg.step_m))
    for k in range(1, n_steps + 1):
        d = k * cfg.step_m
        i = int(math.floor((d * c - ox) / res))
        j = int(math.floor((d * s - oy) / res))
        if i < 0 or i >= nx or j < 0 or j >= ny:
            stop_value = 1   # off-grid → treat as open
            break
        v = int(grid[i, j])
        if v == 0 or (cfg.require_clear and v == -1):
            stop_value = v
            break
        clear_run = d        # clear (or unknown when not require_clear) → advance
    else:
        clear_run = limit
        reached_limit = True

    # Reached the waypoint with a clear path → aim AT it (no backoff). Any
    # other stop (obstacle/unknown/horizon) → back off to leave Tier-3 room.
    reached_waypoint = (
        reached_limit and max_dist_m is not None and max_dist_m <= cfg.horizon_m
    )
    free_dist = clear_run if reached_waypoint else clear_run - cfg.backoff_m

    if free_dist >= cfg.min_subgoal_m:
        return Tier2Result(
            ok=True,
            body_xy=(free_dist * c, free_dist * s),
            free_dist_m=free_dist,
            reason="ok",
        )

    if clear_run <= 0.0:
        reason = "blocked_at_origin" if stop_value == 0 else "all_unknown"
    else:
        reason = "too_short"
    return Tier2Result(ok=False, body_xy=None, free_dist_m=free_dist, reason=reason)


def plan_tier2(
    grid: np.ndarray,
    meta: Dict[str, Any],
    bearing_rad: float,
    max_dist_m: float,
    cfg: Optional[Tier2Config] = None,
) -> Tier2Decision:
    """Production Tier-2 step: clear-run along ``bearing_rad`` (same as
    ``furthest_free_point``), packaged as a ``Tier2Decision`` for the debug
    console. Callers that already footprint-mask the grid (HierarchicalDrive)
    should call ``furthest_free_point`` directly; this wrapper is for
    pi_drive / tests that want the Decision shape.
    """
    cfg = cfg or Tier2Config()
    r = furthest_free_point(grid, meta, bearing_rad, cfg, max_dist_m=max_dist_m)
    limit = min(max_dist_m, cfg.horizon_m)
    capped = max_dist_m > cfg.horizon_m + 1e-9
    if not r.ok or r.body_xy is None:
        return Tier2Decision(
            bearing_rad=bearing_rad, max_dist_m=max_dist_m, ok=False,
            body_xy=None, free_dist_m=r.free_dist_m, reason=r.reason,
            capped_at_target=False, backoff_applied=False, bearing_offset_rad=0.0,
        )
    # Backoff was applied when we did not land exactly on the waypoint/limit.
    reached_target = (
        not capped and abs(r.free_dist_m - max_dist_m) <= 1e-6
    )
    backoff_applied = not reached_target and r.free_dist_m < limit - 1e-9
    return Tier2Decision(
        bearing_rad=bearing_rad, max_dist_m=max_dist_m, ok=True,
        body_xy=r.body_xy, free_dist_m=r.free_dist_m, reason=r.reason,
        capped_at_target=reached_target, backoff_applied=backoff_applied,
        bearing_offset_rad=0.0,
    )
