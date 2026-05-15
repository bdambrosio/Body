"""Costmap construction for the planner.

Inputs: a snapshot from `WorldGrid.snapshot_for_ui()` carrying the
int8 driveable layer (1=clear, 0=blocked, -1=unknown).

Output: a `Costmap` with three layers aligned to the same world frame:
    - lethal: bool, robot must not enter (blocked or within
      footprint+safety-margin of blocked).
    - cost: float32, cost of routing through each non-lethal cell.
      Free clear is 0; halo around lethal decays exponentially toward 0;
      unknown gets a configurable cost so the planner prefers
      observed-clear when one exists, but is willing to plan into
      unknown when nothing else gets to the goal.
    - unknown: bool, cells never observed (passthrough from driveable).

Implementation notes
--------------------
We avoid pulling scipy into the desktop deps; the operations needed
(3×3 morphological open + a small-radius distance transform) are
cheap to implement in numpy:

* 3×3 binary erode/dilate: shift-and-AND / shift-and-OR over the 8
  neighbors. ~10 lines each.
* Distance transform: iterative wavefront dilation, capped at the
  worst-case useful radius (lethal radius + halo decay × ~5). For
  the parameters we use (lethal_radius ≈ 0.25 m, halo_decay ≈ 0.30 m),
  ~24 cells of propagation is plenty. That's 24 × 8 = 192 numpy ops
  on a 500×500 grid per costmap build — fast enough to rebuild every
  UI tick.

If costmap construction ever shows up in a profile, the natural next
step is to crop to the bounded region (`snapshot["bounds_ij"]` plus
a halo) before building, since cells far from any data don't need
processing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


_DEFAULT_LIVE_OVERRIDE_HALF_ANGLE_RAD = math.radians(15.0)


# ── Config ──────────────────────────────────────────────────────────


@dataclass
class CostmapConfig:
    # Robot footprint radius. Cells whose center is within this
    # distance of any blocked cell are LETHAL — robot literally
    # cannot fit. Smaller than the previous "footprint + safety"
    # combo: safety now lives in the halo, not in lethal, so a
    # speckle of blocked cells doesn't paint a giant no-go disk.
    footprint_radius_m: float = 0.15
    # Beyond the footprint, a band of width safety_margin_m carries
    # the maximum halo cost (planner strongly prefers to avoid).
    # Past that, cost decays exponentially.
    #
    # 0.10 m → 2 cells at 0.05 m resolution. Gives A* an incentive to
    # leave a ~2-cell buffer between the path and lethal cells when an
    # alternative route exists within ~10 cells of extra length per
    # avoided full-cost cell. Smaller bands let A* hug walls to save
    # one pixel — bad behavior next to scattered lidar speckle.
    safety_margin_m: float = 0.10

    # Exponential decay length-scale beyond the safety band.
    inflation_decay_m: float = 0.15
    halo_max: float = 100.0

    # Cost assigned to cells that have never been observed. Low enough
    # to plan through if no observed-clear path exists; high enough
    # that "go through observed clear" is preferred.
    unknown_cost: float = 25.0

    # Speckle filter on the blocked layer before inflation. Drops
    # blocked cells whose number of blocked 8-neighbors is below
    # this threshold.
    #   =1 → drops only fully-isolated cells
    #   =2 → also drops sparse 2-cell pairs (DEFAULT)
    #   =3 → eats 1-cell-thick walls (each cell only has 2 line-
    #        direction neighbors); too aggressive for lidar maps.
    denoise: bool = True
    denoise_min_neighbors: int = 2

    # Traversal protection. Cells the robot has physically driven
    # over (traversed_ts != NaN) are forced clear (cost=0, not
    # lethal) — they are known drivable, regardless of how the
    # blocked-cell inflation around them tries to paint them.
    # The traversed mask (Pi stamps a footprint_radius_m disk per
    # pose) is dilated by `traversal_protection_extra_radius_m`
    # before applying, so the protected corridor extends past the
    # exact wheel path and covers the body-width swath around it.
    # Override: if a traversed cell is currently within the
    # forward-cone of regard (radius + half-angle of robot heading),
    # the live classification stands instead. This lets a freshly
    # observed obstacle (chair pulled into a doorway, person in the
    # way) inflate normally even into previously-driven cells.
    traversal_protection_extra_radius_m: float = 0.15
    live_override_radius_m: float = 1.0
    live_override_half_angle_rad: float = _DEFAULT_LIVE_OVERRIDE_HALF_ANGLE_RAD


# ── Output type ────────────────────────────────────────────────────


@dataclass
class Costmap:
    cost: np.ndarray            # float32, finite for non-lethal
    lethal: np.ndarray          # bool
    unknown: np.ndarray         # bool
    distance_m: np.ndarray      # float32, distance to nearest blocked (capped)
    meta: Dict[str, Any]        # resolution_m, origin_x_m, origin_y_m, nx, ny
    bounds_ij: Optional[Tuple[int, int, int, int]] = None
    config: CostmapConfig = field(default_factory=CostmapConfig)


# ── Public API ─────────────────────────────────────────────────────


def build_costmap(
    snap: Dict[str, Any],
    config: Optional[CostmapConfig] = None,
    *,
    pose: Optional[Tuple[float, float, float]] = None,
) -> Costmap:
    """Construct a Costmap from a `WorldGrid.snapshot_for_ui()` result.

    `snap` must carry a `driveable` int8 layer (1=clear, 0=blocked,
    -1=unknown) and a `meta` dict with `resolution_m`. If `snap`
    also carries `traversed_ts`, traversal protection is applied
    (see CostmapConfig.live_override_*); `pose` is used to build the
    forward-cone exception. With pose=None, all traversed cells are
    unconditionally protected. Free of side effects; safe to call on
    the UI thread once per redraw.
    """
    cfg = config or CostmapConfig()
    drive = snap["driveable"]
    meta = snap["meta"]
    res = float(meta["resolution_m"])
    if res <= 0:
        raise ValueError(f"bad resolution_m={res} in costmap input")

    blocked = (drive == 0)
    unknown = (drive == -1)

    if cfg.denoise:
        blocked = _drop_speckle(
            blocked, min_neighbors=cfg.denoise_min_neighbors,
        )

    # Lethal radius is footprint only — the safety margin contributes
    # to halo cost, not to "robot cannot enter." This stops a single
    # speckle from painting a 50-cell lethal disk.
    lethal_radius_m = cfg.footprint_radius_m
    safety_band_m = cfg.safety_margin_m
    halo_extent_m = (
        lethal_radius_m + safety_band_m + 5.0 * cfg.inflation_decay_m
    )
    max_cells = max(2, int(math.ceil(halo_extent_m / res)))
    dist_cells = _wavefront_distance(blocked, max_cells=max_cells)
    dist_m = (dist_cells * res).astype(np.float32)

    # Lethal: blocked plus footprint-radius dilation.
    lethal = blocked | (dist_m < lethal_radius_m)

    # Cost field. Inside the safety band (lethal_radius..lethal_radius
    # + safety_margin) the halo is at full halo_max — strongly avoided
    # but not lethal. Beyond, exponential decay.
    cost = np.zeros_like(dist_m, dtype=np.float32)
    halo = (~lethal) & (dist_m < halo_extent_m)
    if np.any(halo):
        d_excess = dist_m[halo] - lethal_radius_m
        in_safety = d_excess <= safety_band_m
        decay_d = np.maximum(0.0, d_excess - safety_band_m)
        cost[halo] = np.where(
            in_safety,
            cfg.halo_max,
            cfg.halo_max * np.exp(-decay_d / cfg.inflation_decay_m),
        )
    cost[unknown] = cfg.unknown_cost
    # Lethal cells: leave as +inf so any consumer that ignores `lethal`
    # but uses `cost` gets sane behavior.
    cost[lethal] = np.inf

    # Traversal protection. Cells the robot has driven over are forced
    # clear unless they're in the forward cone of regard right now (in
    # which case the live observation wins — chair pulled into the
    # doorway must inflate normally).
    traversed_ts = snap.get("traversed_ts")
    if traversed_ts is not None:
        traversed = ~np.isnan(traversed_ts)
        extra_cells = max(
            0,
            int(math.ceil(cfg.traversal_protection_extra_radius_m / res)),
        )
        if extra_cells > 0 and np.any(traversed):
            traversed = _dilate_bool(traversed, iters=extra_cells)
        if pose is not None:
            in_cone = _forward_cone_mask(
                pose=pose, meta=meta,
                radius_m=cfg.live_override_radius_m,
                half_angle_rad=cfg.live_override_half_angle_rad,
            )
            protected = traversed & ~in_cone
        else:
            protected = traversed
        # Only force-clear cells that aren't currently raw-blocked.
        # The original intent of traversal protection was to clear
        # *inflation-induced* lethal (a prior pose's halo surrounding
        # the bot, blocking retreat through its own trail) — not to
        # override live lidar that reports the cell is actually
        # occupied. Without this `& ~blocked` guard, a chair pulled
        # into a previously-driven corridor stays cleared in the
        # costmap (lethal=False, cost=0), and A* happily routes the
        # bot into it. With the guard, raw blocked always wins; only
        # the inflation-only lethal around traversed cells is cleared.
        # Forward-cone exception still matters for halo cost, just not
        # for the blocked-cell case (which never gets cleared now).
        clearable = protected & ~blocked
        if np.any(clearable):
            lethal[clearable] = False
            cost[clearable] = 0.0

    return Costmap(
        cost=cost,
        lethal=lethal,
        unknown=unknown,
        distance_m=dist_m,
        meta=dict(meta),
        bounds_ij=snap.get("bounds_ij"),
        config=cfg,
    )


# ── Numpy helpers ──────────────────────────────────────────────────


def _forward_cone_mask(
    *,
    pose: Tuple[float, float, float],
    meta: Dict[str, Any],
    radius_m: float,
    half_angle_rad: float,
) -> np.ndarray:
    """Boolean mask of cells inside a forward cone from the robot.

    A cell qualifies when its center is within `radius_m` of the
    pose (x, y) AND the bearing from pose to cell is within
    `±half_angle_rad` of the pose heading θ.
    """
    nx = int(meta["nx"])
    ny = int(meta["ny"])
    res = float(meta["resolution_m"])
    ox = float(meta["origin_x_m"])
    oy = float(meta["origin_y_m"])
    x_pose, y_pose, theta = pose
    ii = np.arange(nx, dtype=np.float32)
    jj = np.arange(ny, dtype=np.float32)
    xs = ox + (ii + 0.5) * res
    ys = oy + (jj + 0.5) * res
    Xs, Ys = np.meshgrid(xs, ys, indexing="ij")
    dx = Xs - np.float32(x_pose)
    dy = Ys - np.float32(y_pose)
    in_range = (dx * dx + dy * dy) <= np.float32(radius_m * radius_m)
    bearing = np.arctan2(dy, dx)
    diff = bearing - np.float32(theta)
    diff = np.mod(diff + np.pi, 2.0 * np.pi) - np.pi
    in_angle = np.abs(diff) <= np.float32(half_angle_rad)
    return in_range & in_angle


def _shift(arr: np.ndarray, di: int, dj: int, fill) -> np.ndarray:
    """Return arr shifted by (di, dj). Vacated cells get `fill`."""
    out = np.full_like(arr, fill)
    h, w = arr.shape
    si0 = max(0, -di); si1 = min(h, h - di)
    sj0 = max(0, -dj); sj1 = min(w, w - dj)
    di0 = max(0,  di); di1 = min(h, h + di)
    dj0 = max(0,  dj); dj1 = min(w, w + dj)
    if si1 > si0 and sj1 > sj0:
        out[di0:di1, dj0:dj1] = arr[si0:si1, sj0:sj1]
    return out


def _dilate_bool(mask: np.ndarray, *, iters: int) -> np.ndarray:
    """8-connected boolean dilation, `iters` times. Each iteration
    extends the True region by one cell in every direction.
    """
    out = mask
    for _ in range(iters):
        nxt = out
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                nxt = nxt | _shift(out, di, dj, fill=False)
        out = nxt
    return out


def _drop_speckle(
    mask: np.ndarray, *, min_neighbors: int = 2,
) -> np.ndarray:
    """Drop True cells whose count of True 8-neighbors is below
    min_neighbors. With min_neighbors=1, drops only fully-isolated
    specks (preserves 1-cell-thick walls). With min_neighbors=2,
    also drops sparse 2-cell pairs that would each inflate to a
    50-cell lethal disk under footprint+safety inflation. Walls
    survive either way: each cell on a line has 2 line-direction
    neighbors.
    """
    count = np.zeros(mask.shape, dtype=np.int8)
    src = mask.astype(np.int8)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            count += _shift(src, di, dj, fill=0)
    return mask & (count >= int(min_neighbors))


def _wavefront_distance(blocked: np.ndarray, *, max_cells: int) -> np.ndarray:
    """Iterative-dilation Euclidean distance transform, capped at
    `max_cells`. Cells beyond that get max_cells + 1.

    Uses 8-neighbor wavefront with weights 1.0 / √2 to approximate
    Euclidean. Error vs. exact EDT is < 4 % at the radii we care
    about — fine for inflation purposes.
    """
    SQRT2 = math.sqrt(2.0)
    INF = float(max_cells + 1)
    dist = np.where(blocked, 0.0, INF).astype(np.float32)
    if max_cells <= 0:
        return dist

    # Small precomputed neighbor list.
    neighbors = [
        (-1, -1, SQRT2), (-1, 0, 1.0), (-1, 1, SQRT2),
        ( 0, -1, 1.0),                 ( 0, 1, 1.0),
        ( 1, -1, SQRT2), ( 1, 0, 1.0), ( 1, 1, SQRT2),
    ]
    for _ in range(max_cells):
        prev = dist
        cur = prev
        for di, dj, w in neighbors:
            shifted = _shift(prev, di, dj, fill=INF) + np.float32(w)
            cur = np.minimum(cur, shifted)
        if np.array_equal(cur, dist):
            break
        dist = cur
    return dist


# ── Visualization helper (used by WorldCostmapView) ────────────────


def costmap_to_rgb(cm: Costmap) -> np.ndarray:
    """Render a Costmap as a uint8 (nx, ny, 3) RGB image.

    Color choices kept distinct so the operator can tell at a glance
    what the planner is being told:
      lethal     → bright red (robot literally cannot enter here)
      halo, hot  → orange-yellow at full cost (high-cost band, planner
                   strongly avoids; previously rendered red and was
                   indistinguishable from lethal)
      halo, cool → green-yellow as cost falls toward 0
      free       → dark green
      unknown    → medium gray
    The whole halo gradient stays in the green→yellow range so red is
    reserved exclusively for "no go."
    """
    nx, ny = cm.cost.shape
    rgb = np.zeros((nx, ny, 3), dtype=np.uint8)

    # Free baseline (cost == 0, observed clear & not in halo).
    rgb[:] = (10, 64, 32)

    # Halo: green-yellow gradient on cost magnitude. t=0 (cost→0) is
    # dark olive; t=1 (cost==halo_max) is saturated yellow. Crucially
    # nothing in this gradient is red, so it can't be confused with
    # lethal at a glance.
    halo_max = float(cm.config.halo_max)
    finite = np.isfinite(cm.cost)
    halo = finite & (cm.cost > 0) & (~cm.unknown)
    if np.any(halo):
        t = np.clip(cm.cost[halo] / max(halo_max, 1e-6), 0.0, 1.0)
        # t=0 → (40, 110, 50) dark green
        # t=1 → (235, 215, 60) saturated yellow
        r = (40 + 195 * t).astype(np.uint8)
        g = (110 + 105 * t).astype(np.uint8)
        b = (50 + 10 * t).astype(np.uint8)
        rgb[halo] = np.stack([r, g, b], axis=-1)

    # Unknown.
    rgb[cm.unknown] = (90, 90, 90)

    # Lethal last so it overwrites halo edges. Pure red — the only
    # red anywhere in this image.
    rgb[cm.lethal] = (235, 50, 50)

    return rgb
