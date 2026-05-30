"""Tier-2 visibility sub-goal selection (pure).

Tier-1 (the topological waypoint list) only ever hands the metric loop a
coarse *direction*: the bearing from the robot's current world pose to the
next waypoint. Tier-2 turns that bearing into a concrete point the robot can
actually see right now — the furthest live-visible free point along the
bearing in the body-frame scan grid produced by ``scan_raster.rasterize_scan``
— and hands *that* body-frame point to Tier-3 (``body/drive/goto``). Nothing
from the world map crosses into the drive command except the bearing.

Pure NumPy, no zenoh — importable on both the desktop orchestrator and the Pi
(should Tier-2 ever move on-robot). Single ray, not a fan: Tier-3 runs its own
reactive fan/centering around whatever point we hand it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from body.lib.local_drive_core import wrap_pi


@dataclass(frozen=True)
class Tier2Config:
    horizon_m: float = 2.0          # cap sub-goal distance (≤ scan half_extent)
    step_m: float = 0.04            # ray-march step (~half a cell)
    backoff_m: float = 0.30         # pull the sub-goal back from the first block/unknown
    min_subgoal_m: float = 0.20     # below this → "no usable free point"
    require_clear: bool = True      # treat unknown (-1) as non-free (conservative)


@dataclass(frozen=True)
class Tier2Result:
    ok: bool
    body_xy: Optional[Tuple[float, float]]   # body-frame sub-goal (bx, by), or None
    free_dist_m: float                       # confirmed-clear distance along the bearing
    reason: str                              # "ok"|"blocked_at_origin"|"all_unknown"|"too_short"


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
) -> Tier2Result:
    """Furthest live-visible free point along ``bearing_rad`` in a body-frame grid.

    March from the robot (body origin) outward along the bearing over the
    int8 scan grid (-1 unknown / 0 blocked / 1 clear). The free run ends at
    the first blocked cell, the first unknown cell (when ``require_clear``),
    the grid edge, or the horizon. The sub-goal is the free distance backed
    off by ``backoff_m`` so Tier-3's swept-footprint check has room. Returns
    ``ok=False`` when the backed-off distance is below ``min_subgoal_m``.
    """
    cfg = cfg or Tier2Config()
    res = float(meta["resolution_m"])
    ox = float(meta["origin_x_m"])
    oy = float(meta["origin_y_m"])
    nx = int(meta["nx"])
    ny = int(meta["ny"])

    c = math.cos(bearing_rad)
    s = math.sin(bearing_rad)

    clear_run = 0.0          # furthest distance confirmed clear from the origin
    stop_value = 1           # cell value that ended the run (1 = reached horizon clear)

    n_steps = int(math.floor(cfg.horizon_m / cfg.step_m))
    for k in range(1, n_steps + 1):
        d = k * cfg.step_m
        bx = d * c
        by = d * s
        i = int(math.floor((bx - ox) / res))
        j = int(math.floor((by - oy) / res))
        if i < 0 or i >= nx or j < 0 or j >= ny:
            stop_value = 1   # off-grid → treat as open horizon
            break
        v = int(grid[i, j])
        if v == 0 or (cfg.require_clear and v == -1):
            stop_value = v
            break
        clear_run = d        # clear (or unknown when not require_clear) → advance
    else:
        clear_run = min(cfg.horizon_m, n_steps * cfg.step_m)

    free_dist = clear_run - cfg.backoff_m
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
