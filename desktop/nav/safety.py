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
    #
    # NOTE: `arc_distance_m` / `arc_half_angle_rad` are the *fixed
    # wedge* parameters used by the legacy `*_arc_blocked[_local]`
    # functions (still consumed by world-frame planning/recovery
    # checks). The per-tick motion veto in main_window now uses
    # `swept_path_blocked_local`, which traces the actual footprint
    # along the commanded (v, ω) arc — see the swept-* fields below.
    arc_distance_m: float = 0.50

    # Half-angle of the wedge. Wider = more conservative (catches
    # things off the heading axis) but more likely to spurious-stop
    # in tight corridors where walls run alongside the path.
    arc_half_angle_rad: float = math.radians(20.0)

    # ── Swept-footprint check (body-frame per-tick veto) ────────────
    # Radius of the robot's circular footprint. The swept check inflates
    # every obstacle by this radius (equivalently: traces a disc of this
    # radius along the path), so an obstacle at the robot's shoulder —
    # outside the old ±20° wedge — now correctly blocks. Set from the
    # localizer's footprint_radius_m at construction.
    footprint_radius_m: float = 0.22

    # Arc length the footprint center is traced along the predicted
    # (v, ω) motion. Total reach ahead ≈ this + footprint_radius_m.
    # Scales with speed (preview_time_s) between the min floor and this
    # cap so slow creeping still previews a sane distance.
    preview_distance_m: float = 0.35
    preview_min_distance_m: float = 0.15
    preview_time_s: float = 1.5

    # Treat observed-`unknown` (-1) cells as blocking, but only within
    # `unknown_block_range_m` of the body. Far-ahead unknown stays
    # passable (the robot must be able to drive into never-observed
    # space toward a goal); only an unknown cell right in front — the
    # signature of an obstacle the sensors saw but couldn't classify —
    # stops translation. Set block_on_unknown=False to restore the
    # old "unknown is always passable" behavior.
    block_on_unknown: bool = True
    unknown_block_range_m: float = 0.25

    # Empty-local_map guard. If the swept region contains fewer than
    # this many *observed* (clear-or-blocked, != -1) cells, refuse to
    # drive — an all-unknown local_map (e.g. the launcher startup race)
    # is untrustworthy, and the staleness gate alone won't catch it
    # because the data is "fresh", just empty.
    min_observed_cells: int = 3


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


# ── Swept-footprint variant (body-frame, footprint- and curve-aware) ─


def _arc_samples(
    v_mps: float, omega_radps: float, reach_m: float, n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Body-frame centers of the footprint traced along the constant-
    (v, ω) arc, from the origin out to arc length `reach_m`. Returns
    (cx, cy) arrays of length n+1. Unicycle integration; handles the
    straight-line (ω≈0) and reverse (v<0) cases by sign.
    """
    speed = abs(v_mps)
    ks = np.arange(n + 1, dtype=np.float64)
    if speed < 1e-6:
        return np.zeros(n + 1), np.zeros(n + 1)
    t_total = reach_m / speed
    t = t_total * ks / n
    if abs(omega_radps) < 1e-6:
        return v_mps * t, np.zeros(n + 1)
    radius = v_mps / omega_radps
    phi = omega_radps * t
    cx = radius * np.sin(phi)
    cy = radius * (1.0 - np.cos(phi))
    return cx, cy


def swept_path_blocked_local(
    driveable: np.ndarray,
    meta: Dict[str, Any],
    *,
    v_mps: float,
    omega_radps: float,
    config: Optional[SafetyConfig] = None,
) -> bool:
    """Body-frame swept-footprint obstacle check along the predicted arc.

    The robot is a disc of radius `footprint_radius_m` at the body origin
    (+x forward, +y left). We trace its center along the constant-(v, ω)
    arc out to a short preview distance and flag a block when any cell in
    the union of footprint discs is:

      * observed-blocked (`== 0`), or
      * observed-unknown (`== -1`) within `unknown_block_range_m` of the
        body, when `block_on_unknown` is set, or
      * the swept region holds fewer than `min_observed_cells` observed
        (`!= -1`) cells — an empty / not-yet-populated local_map we refuse
        to trust (fail-safe; the staleness gate misses this).

    Pure rotation (v≈0) sweeps no new ground for a circular footprint, so
    it returns False — rotation in place stays allowed (it's how the robot
    turns away from a facing obstacle). Reads the body-frame local_map, so
    it needs no pose transform and is immune to localization drift.

    Fail-safe: malformed meta or a path that falls outside the local_map
    coverage returns True (blocked) rather than driving blind.
    """
    cfg = config or SafetyConfig()
    speed = abs(v_mps)
    if speed < 1e-3:
        return False  # not translating — rotation is always permitted

    res = float(meta.get("resolution_m", 0.0))
    if res <= 0:
        return True  # malformed meta while moving — refuse to drive blind
    ox = float(meta.get("origin_x_m", 0.0))
    oy = float(meta.get("origin_y_m", 0.0))
    nx, ny = driveable.shape

    reach_m = min(
        cfg.preview_distance_m,
        max(cfg.preview_min_distance_m, speed * cfg.preview_time_s),
    )
    # Sample roughly one center per cell along the arc (≥3, capped).
    n = int(max(3, min(25, math.ceil(reach_m / max(res, 1e-3)))))
    cx, cy = _arc_samples(v_mps, omega_radps, reach_m, n)

    # Inflate the footprint by half a cell so a disc edge that clips a
    # cell without covering its center still counts (conservative).
    r_foot = cfg.footprint_radius_m + 0.5 * res
    pad = r_foot + res

    i_lo = max(0, int(math.floor((float(cx.min()) - pad - ox) / res)))
    i_hi = min(nx, int(math.ceil((float(cx.max()) + pad - ox) / res)) + 1)
    j_lo = max(0, int(math.floor((float(cy.min()) - pad - oy) / res)))
    j_hi = min(ny, int(math.ceil((float(cy.max()) + pad - oy) / res)) + 1)
    if i_hi <= i_lo or j_hi <= j_lo:
        return True  # arc falls outside local_map coverage — fail safe

    sub = driveable[i_lo:i_hi, j_lo:j_hi]
    ii = np.arange(i_lo, i_hi).reshape(-1, 1).astype(np.float64)
    jj = np.arange(j_lo, j_hi).reshape(1, -1).astype(np.float64)
    cell_x = ox + (ii + 0.5) * res    # (H, 1)
    cell_y = oy + (jj + 0.5) * res    # (1, W)

    # Union of footprint discs along the arc: a cell is "swept" if it
    # lies within r_foot of any sampled center.
    r2 = r_foot * r_foot
    in_swept = np.zeros(sub.shape, dtype=bool)
    for sx, sy in zip(cx, cy):
        d2 = (cell_x - sx) ** 2 + (cell_y - sy) ** 2
        in_swept |= d2 <= r2
    if not np.any(in_swept):
        return True  # nothing of the path is on the grid — fail safe

    if np.any((sub == 0) & in_swept):
        return True

    if cfg.block_on_unknown:
        dist_origin = np.hypot(cell_x, cell_y)
        unknown_close = (
            (sub == -1) & in_swept & (dist_origin <= cfg.unknown_block_range_m)
        )
        if np.any(unknown_close):
            return True

    observed_in_swept = int(np.count_nonzero((sub != -1) & in_swept))
    if observed_in_swept < cfg.min_observed_cells:
        return True  # local_map too empty here to trust "clear"

    return False


# ── Rotation rate limiter ──────────────────────────────────────────


class OmegaRateLimiter:
    """Caps |ω| and inserts a settle window between direction reversals.

    Two constraints:
      * Magnitude cap: |emitted ω| ≤ `omega_max_radps`.
      * Reversal hold: when input ω flips sign, emit 0.0 for at least
        `reversal_hold_s` before allowing the new sign through. The
        hold timer is reset whenever input ω comes back to zero or to
        the previously-emitted sign — only a sustained reversal
        request actually pays the hold.

    Reason this exists: on a small differential-drive bot, rapid
    left/right ω commands (e.g. when the follower is hunting heading
    while the forward arc keeps clipping translation) make the wheels
    slip, which loses IMU yaw lock + encoder odometry alignment. A
    15 dps cap with a 500 ms inter-reversal hold prevents the slip
    regime in practice without slowing legitimate continuous turning.
    """

    def __init__(self, omega_max_radps: float, reversal_hold_s: float):
        self._omega_max = float(omega_max_radps)
        self._hold_s = float(reversal_hold_s)
        self._last_sign: int = 0  # sign of the last non-zero emission
        self._zero_emit_started: Optional[float] = None

    def limit(self, omega_in: float, now: float) -> float:
        omega = max(-self._omega_max, min(self._omega_max, omega_in))
        if abs(omega) < 1e-6:
            if self._zero_emit_started is None:
                self._zero_emit_started = now
            return 0.0
        sign = 1 if omega > 0 else -1
        if self._last_sign == 0 or sign == self._last_sign:
            self._last_sign = sign
            self._zero_emit_started = None
            return omega
        # Reversal requested. Hold 0 until we've been emitting 0 for
        # at least `hold_s`. _zero_emit_started is set either by a
        # prior zero emit (input was 0) or by us, the first time we
        # see the reversal request.
        if self._zero_emit_started is None:
            self._zero_emit_started = now
            return 0.0
        if now - self._zero_emit_started < self._hold_s:
            return 0.0
        self._last_sign = sign
        self._zero_emit_started = None
        return omega

    def reset(self) -> None:
        self._last_sign = 0
        self._zero_emit_started = None
