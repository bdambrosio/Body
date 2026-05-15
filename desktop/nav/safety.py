"""Forward-arc safety check.

Two variants live here:

* `*_arc_blocked_local` reads the body-frame `local_map.driveable` —
  the freshest fused (lidar + depth) observation the Pi publishes. This
  is what the per-tick safety stop *should* use: it depends on no
  pose transform, so it's immune to SLAM drift, and it's
  recomputed each frame so stale votes can't haunt it.
* `*_arc_blocked` (world-frame, `Costmap`-based) is the older form,
  kept for the BackUp primitive's rear-arc check. Planning + recovery
  still consume world-frame data, so this function isn't going away —
  but the per-tick `GO BLOCKED` decision in main_window has moved to
  the local variant.

The architectural call: the planner trusts the world-frame costmap
(it needs the global view); safety trusts the live local_map (it
needs ground truth about "what's physically in front of me?"). When
those disagree, the disagreement is itself a high-confidence drift
signal — exactly the trigger for an auto-relocate decision.

Vectorized over a small bounding box, ~0.5 ms per call on a 100×100
local_map or 500×500 world costmap. Additive — does nothing when
there's no blocked/lethal cell in the arc.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from desktop.world_map.costmap import Costmap


@dataclass
class SafetyConfig:
    # How far ahead of the robot the arc reaches. Should be > the
    # follower's stopping distance at v_max plus enough lead for the
    # operator to perceive the block. With v_max=0.20 m/s and accel
    # cap 0.50 m/s², stopping distance is ~0.08 m; we use 0.50 m so
    # the bot stops a half-meter shy of the first observed obstacle.
    # Tune up if you want a longer preview horizon (the user asked
    # for ≥ 2 m in nav-safety discussion; trade-off is wider wedges
    # cause more spurious stops in tight corridors).
    arc_distance_m: float = 0.50

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


# ── Body-frame variants (read local_map directly) ──────────────────


def forward_arc_blocked_local(
    driveable: np.ndarray,
    meta: Dict[str, Any],
    config: Optional[SafetyConfig] = None,
) -> bool:
    """Return True if any blocked cell in the body-frame local_map is
    inside the forward wedge.

    Arguments come straight from `chassis.state.local_map_driveable`
    and `chassis.state.local_map_meta`. The local_map is in body frame
    (robot at origin, +x forward, +y left), so no pose transform is
    needed — this check is drift-immune.

    `driveable` is int8 with -1 unknown, 0 blocked, 1 clear. Only
    `== 0` cells trigger a block: unknown is not treated as an
    obstacle (the bot may need to drive into never-observed space
    on the way to the goal, and the Pi's watchdogs are the last line
    if real geometry turns out to be blocking).
    """
    return _arc_blocked_local(driveable, meta, config, direction=+1)


def rear_arc_blocked_local(
    driveable: np.ndarray,
    meta: Dict[str, Any],
    config: Optional[SafetyConfig] = None,
) -> bool:
    """Symmetric to `forward_arc_blocked_local`, wedge points backward."""
    return _arc_blocked_local(driveable, meta, config, direction=-1)


def _arc_blocked_local(
    driveable: np.ndarray,
    meta: Dict[str, Any],
    config: Optional[SafetyConfig],
    *,
    direction: int,
) -> bool:
    cfg = config or SafetyConfig()
    res = float(meta.get("resolution_m", 0.0))
    if res <= 0:
        return False  # bogus meta — fall through to "not blocked"
    ox = float(meta.get("origin_x_m", 0.0))
    oy = float(meta.get("origin_y_m", 0.0))
    nx, ny = driveable.shape
    r_max = cfg.arc_distance_m

    # Robot is at (0, 0) in body frame. Wedge axis is +x for forward,
    # -x (≡ θ=π) for rear.
    bearing_axis = 0.0 if direction > 0 else math.pi

    # Bounding box of the arc within the local_map (cell indices).
    i_lo = max(0, int(math.floor((0.0 - r_max - ox) / res)))
    i_hi = min(nx, int(math.ceil((0.0 + r_max - ox) / res)) + 1)
    j_lo = max(0, int(math.floor((0.0 - r_max - oy) / res)))
    j_hi = min(ny, int(math.ceil((0.0 + r_max - oy) / res)) + 1)
    if i_hi <= i_lo or j_hi <= j_lo:
        return False

    sub = driveable[i_lo:i_hi, j_lo:j_hi]
    blocked = (sub == 0)
    if not np.any(blocked):
        return False  # cheap exit when there's nothing observed-blocked nearby

    ii = np.arange(i_lo, i_hi).reshape(-1, 1).astype(np.float64)
    jj = np.arange(j_lo, j_hi).reshape(1, -1).astype(np.float64)
    cell_x = ox + (ii + 0.5) * res    # (H, 1)
    cell_y = oy + (jj + 0.5) * res    # (1, W)

    dist = np.hypot(cell_x, cell_y)
    bearing = np.arctan2(cell_y, cell_x)
    angle_off = bearing - bearing_axis
    angle_off = (angle_off + np.pi) % (2.0 * np.pi) - np.pi

    in_arc = (dist <= r_max) & (np.abs(angle_off) <= cfg.arc_half_angle_rad)
    return bool(np.any(blocked & in_arc))
