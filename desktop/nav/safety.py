"""Forward-arc safety check.

The planner routes around `lethal` cells by construction, so during
normal driving the forward arc is clean. But two things can change
between replans:

1. A new obstacle appears (someone walks in front of the robot, a
   chair gets shoved into the corridor) — the next local_map frame
   marks it blocked, but the follower has already pushed cmd_vel
   for this tick.
2. A scan-match correction shifts the world-frame pose by up to
   ~0.30 m, so a path that was correct relative to the old pose may
   now skim a wall in the new frame.

The safety check is the per-tick last line: scan a small wedge in
front of the robot for any `lethal=True` cells in the current
costmap, and if so, override cmd_vel to (0, 0) for this tick.
Mission stays in FOLLOWING — the moment the obstacle clears or the
next replan routes around it, driving resumes.

Vectorized over a small bounding box, ~0.5 ms per call on a 500×500
costmap. Additive — does nothing when there's no lethal in the arc.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from desktop.world_map.costmap import Costmap


@dataclass
class SafetyConfig:
    # How far ahead of the robot the arc reaches. Should be > the
    # follower's stopping distance at v_max. With v_max=0.20 m/s and
    # accel cap 0.50 m/s², stopping distance is ~0.04 m + ~0.04 m
    # of one-tick latency = ~0.08 m. 0.30 m gives generous margin.
    arc_distance_m: float = 0.30

    # Half-angle of the wedge. Wider = more conservative (catches
    # things off the heading axis) but more likely to spurious-stop
    # in tight corridors where walls run alongside the path.
    arc_half_angle_rad: float = math.radians(20.0)


def forward_arc_blocked(
    costmap: Costmap,
    pose: Tuple[float, float, float],
    config: Optional[SafetyConfig] = None,
) -> bool:
    """Return True if any lethal cell in `costmap` is inside the
    forward wedge of `pose`. Vectorized via a bounding-box crop.
    """
    return _arc_blocked(costmap, pose, config, direction=+1)


def rear_arc_blocked(
    costmap: Costmap,
    pose: Tuple[float, float, float],
    config: Optional[SafetyConfig] = None,
) -> bool:
    """Symmetric to `forward_arc_blocked`, but the wedge points
    backward (theta_w + π). Used by the BackUp primitive to refuse to
    reverse into a wall.
    """
    return _arc_blocked(costmap, pose, config, direction=-1)


def _arc_blocked(
    costmap: Costmap,
    pose: Tuple[float, float, float],
    config: Optional[SafetyConfig],
    *,
    direction: int,
) -> bool:
    cfg = config or SafetyConfig()
    x_w, y_w, theta_w = pose
    if direction < 0:
        theta_w = theta_w + math.pi

    res = float(costmap.meta["resolution_m"])
    ox = float(costmap.meta["origin_x_m"])
    oy = float(costmap.meta["origin_y_m"])
    nx, ny = costmap.lethal.shape
    r_max = cfg.arc_distance_m

    # Bounding box of the arc in cell indices. arc fits inside the
    # square [robot ± r_max] in world coords.
    i_lo = max(0, int(math.floor((x_w - r_max - ox) / res)))
    i_hi = min(nx, int(math.ceil((x_w + r_max - ox) / res)) + 1)
    j_lo = max(0, int(math.floor((y_w - r_max - oy) / res)))
    j_hi = min(ny, int(math.ceil((y_w + r_max - oy) / res)) + 1)
    if i_hi <= i_lo or j_hi <= j_lo:
        return False

    sub = costmap.lethal[i_lo:i_hi, j_lo:j_hi]
    if not np.any(sub):
        return False  # cheap exit when there's nothing lethal nearby

    # Cell centers as broadcast arrays.
    ii = np.arange(i_lo, i_hi).reshape(-1, 1).astype(np.float64)
    jj = np.arange(j_lo, j_hi).reshape(1, -1).astype(np.float64)
    cell_x = ox + (ii + 0.5) * res    # shape (H, 1)
    cell_y = oy + (jj + 0.5) * res    # shape (1, W)

    dx = cell_x - x_w                  # (H, 1)
    dy = cell_y - y_w                  # (1, W)
    dist = np.hypot(dx, dy)            # (H, W) via broadcasting
    bearing = np.arctan2(dy, dx)       # (H, W)
    angle_off = bearing - theta_w
    # Wrap to [-π, π] so |angle| comparison works across the
    # ±π discontinuity.
    angle_off = (angle_off + np.pi) % (2.0 * np.pi) - np.pi

    in_arc = (dist <= r_max) & (np.abs(angle_off) <= cfg.arc_half_angle_rad)
    return bool(np.any(sub & in_arc))
