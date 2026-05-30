"""Body-frame swept-footprint obstacle check for the Pi-side Tier-3 driver.

Traces the robot's circular footprint along the commanded (v, ω) arc over a
short preview distance and reports whether it would sweep an obstacle in the
body-frame ``local_map`` driveable layer. Drift-immune (body frame, no pose
transform). Pure NumPy — unit-tested off-robot.

This is the Pi-runtime sibling of ``desktop/nav/safety.py:swept_path_blocked_local``
(same algorithm); the two live in separate runtimes with no shared package, so
the logic is intentionally duplicated. Keep them in sync if you change either.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class FootprintConfig:
    footprint_radius_m: float = 0.22
    preview_distance_m: float = 0.35
    preview_min_distance_m: float = 0.15
    preview_time_s: float = 1.5
    block_on_unknown: bool = True
    unknown_block_range_m: float = 0.25
    min_observed_cells: int = 3
    # Half-angle of the forward cone for the (larger) *preview* footprint: an
    # obstacle must be within this cone of the velocity to veto via the preview
    # radius. Narrow (~60°) lets the robot drive *past* an abeam obstacle that
    # is beyond the body — e.g. squeeze through a gap.
    forward_cone_rad: float = math.radians(60.0)
    # Hard body radius, checked over the *full forward half-plane* (not the
    # narrow cone): an obstacle this close to the swept body is a real
    # collision at any side angle, so it always vetoes. Sized to the true
    # half-width plus a stopping margin — must stay BELOW the gaps you want to
    # pass (clearance ≈ hard_radius_m − true_half_width). This is what stops the
    # robot clipping a doorjamb that's abeam of the forward cone.
    hard_radius_m: float = 0.07


def _arc_samples(
    v_mps: float, omega_radps: float, reach_m: float, n: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Body-frame footprint centers + heading along the constant-(v, ω) arc
    from the origin out to arc length ``reach_m``. (n+1,) arrays (cx, cy,
    theta) where theta is the body heading at each sample (0 for straight).
    Handles the straight (ω≈0) and reverse (v<0) cases by sign."""
    speed = abs(v_mps)
    ks = np.arange(n + 1, dtype=np.float64)
    if speed < 1e-6:
        z = np.zeros(n + 1)
        return z, z.copy(), z.copy()
    t = (reach_m / speed) * ks / n
    theta = omega_radps * t
    if abs(omega_radps) < 1e-6:
        return v_mps * t, np.zeros(n + 1), theta
    radius = v_mps / omega_radps
    phi = omega_radps * t
    return radius * np.sin(phi), radius * (1.0 - np.cos(phi)), theta


def swept_path_blocked(
    driveable: np.ndarray,
    meta: Dict[str, Any],
    *,
    v_mps: float,
    omega_radps: float,
    config: Optional[FootprintConfig] = None,
) -> bool:
    """True if the footprint swept along the predicted (v, ω) arc hits an
    obstacle in the body-frame ``driveable`` grid (int8: -1 unknown, 0
    blocked, 1 clear), treats close-range unknown as blocking, or finds the
    swept region too empty to trust. Pure rotation (v≈0) returns False.
    Fail-safe (malformed meta / path off the grid → True)."""
    cfg = config or FootprintConfig()
    speed = abs(v_mps)
    if speed < 1e-3:
        return False  # rotation in place is always permitted

    res = float(meta.get("resolution_m", 0.0))
    if res <= 0:
        return True
    ox = float(meta.get("origin_x_m", 0.0))
    oy = float(meta.get("origin_y_m", 0.0))
    nx, ny = driveable.shape

    reach_m = min(
        cfg.preview_distance_m,
        max(cfg.preview_min_distance_m, speed * cfg.preview_time_s),
    )
    n = int(max(3, min(25, math.ceil(reach_m / max(res, 1e-3)))))
    cx, cy, theta = _arc_samples(v_mps, omega_radps, reach_m, n)
    sgn = 1.0 if v_mps >= 0 else -1.0

    r_foot = cfg.footprint_radius_m + 0.5 * res
    pad = r_foot + res
    i_lo = max(0, int(math.floor((float(cx.min()) - pad - ox) / res)))
    i_hi = min(nx, int(math.ceil((float(cx.max()) + pad - ox) / res)) + 1)
    j_lo = max(0, int(math.floor((float(cy.min()) - pad - oy) / res)))
    j_hi = min(ny, int(math.ceil((float(cy.max()) + pad - oy) / res)) + 1)
    if i_hi <= i_lo or j_hi <= j_lo:
        return True

    sub = driveable[i_lo:i_hi, j_lo:j_hi]
    ii = np.arange(i_lo, i_hi).reshape(-1, 1).astype(np.float64)
    jj = np.arange(j_lo, j_hi).reshape(1, -1).astype(np.float64)
    cell_x = ox + (ii + 0.5) * res
    cell_y = oy + (jj + 0.5) * res

    # Directional swept region: a cell counts only if it is within r_foot of a
    # sample AND inside the forward *cone* (half-angle ``forward_cone_rad``) of
    # the body's velocity at that sample. This drops the trailing/lateral part
    # of the (stationary) origin footprint, so an obstacle beside the robot no
    # longer vetoes motion that drives past it — only obstacles the motion
    # actually carries the body toward block. ``cone = π/2`` recovers the old
    # forward-half-plane behaviour.
    r2 = r_foot * r_foot
    hard_r = cfg.hard_radius_m + 0.5 * res
    hard_r2 = hard_r * hard_r
    cos2 = math.cos(cfg.forward_cone_rad) ** 2
    in_swept = np.zeros(sub.shape, dtype=bool)
    for sx, sy, th in zip(cx, cy, theta):
        dx = cell_x - sx                     # (H, 1)
        dy = cell_y - sy                     # (1, W)
        d2 = dx * dx + dy * dy               # (H, W)
        dirx = sgn * math.cos(th)
        diry = sgn * math.sin(th)
        dot = dx * dirx + dy * diry          # (H, W)
        fwd = dot >= 0.0                     # in the forward half-plane of travel
        # Preview footprint: within r_foot AND inside the narrow forward cone
        # (angle ≤ cone ⇔ dot² ≥ cos²·|d|²). Lets the robot pass abeam obstacles
        # that are beyond the body.
        preview = fwd & (d2 <= r2) & (dot * dot >= cos2 * d2)
        # Hard body: within the (smaller) hard radius anywhere in the forward
        # half-plane — a real clip at any side angle (e.g. an abeam doorjamb).
        hard = fwd & (d2 <= hard_r2)
        in_swept |= preview | hard
    if not np.any(in_swept):
        return True

    if np.any((sub == 0) & in_swept):
        return True

    if cfg.block_on_unknown:
        dist_origin = np.hypot(cell_x, cell_y)
        if np.any((sub == -1) & in_swept & (dist_origin <= cfg.unknown_block_range_m)):
            return True

    if int(np.count_nonzero((sub != -1) & in_swept)) < cfg.min_observed_cells:
        return True

    return False


def driveable_from_rows(rows: Any, nx: int, ny: int) -> Optional[np.ndarray]:
    """Convert the wire form of local_map ``driveable`` (list of lists of
    bool|None: True clear, False blocked, None unknown) to int8 (-1/0/1).
    Returns None if shape is wrong."""
    if not isinstance(rows, list) or len(rows) != nx:
        return None
    out = np.full((nx, ny), -1, dtype=np.int8)
    for i, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != ny:
            return None
        for j, v in enumerate(row):
            if v is True:
                out[i, j] = 1
            elif v is False:
                out[i, j] = 0
    return out
