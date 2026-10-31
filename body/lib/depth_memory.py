"""Short-term odom-frame memory of depth-camera obstacles for Tier-3.

The depth veto (``depth_veto.py``) only stops the robot; it keeps no record.
Tier-2 and Tier-3 plan on the lidar scan alone, so an obstacle that only the
depth camera sees (a door edge, a chair seat, something below the lidar
plane) was re-approached on every retry: drive ≥ 2 depth frames, veto, cancel,
retry the same heading — creeping a few cm closer each cycle (2026-10-31).

This module remembers what depth saw, in the odom frame, so Tier-3 can stamp
it into the planning grid: A* routes around it and the swept-footprint check
refuses an arc into it *before* moving. Policy:

  * Confirm: a cell (``cell_m``) is remembered after ≥ ``min_pts`` obstacle
    points land in it on ``confirm_frames`` consecutive NEW depth frames —
    the same floor-band + vertical-structure test as the veto
    (``depth_veto.obstacle_mask``) over the veto's ROI and range.
  * Clear, evidence first: a remembered cell that is seen as floor (≥
    ``floor_min_pts`` floor points, 0 obstacle points) on ``clear_frames``
    consecutive new frames is dropped — e.g. the door was closed.
  * Clear, backstops: ``ttl_s`` after the last confirming frame; farther than
    ``keep_radius_m`` from the robot; under the robot's own footprint; and
    ``reset()`` on an odom-frame reset.
  * Planning skips cells within ``plan_exclude_radius_m`` of the robot.
  * The desktop resets it (``body/drive/depth_memory`` {"action": "reset"})
    on Go / relocate / a UI button: wheel odom can't see the robot being
    carried, so old cells would land around it wherever it is put down.
  * Out of view → kept (that is the point: going around a door edge takes it
    out of the camera's view). NOT cleared by a new goal.
  * Frames taken while |ω| > ``max_abs_omega_radps`` are skipped (smear).

Pure NumPy, no clocks: callers pass wall time and poses.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from body.lib.depth_veto import (
    DepthVetoConfig, depth_frame_key, depth_points_for_msg, obstacle_mask,
)

Pose = Tuple[float, float, float]
Cell = Tuple[int, int]


@dataclass(frozen=True)
class DepthMemoryConfig:
    enabled: bool = True
    cell_m: float = 0.04
    min_pts: int = 3              # obstacle points in a cell to count a frame
    confirm_frames: int = 2       # consecutive new frames to remember a cell
    clear_frames: int = 2         # consecutive "seen as floor" frames to forget
    floor_min_pts: int = 4        # floor points in a cell to call it seen-free
    ttl_s: float = 30.0           # forget this long after the last confirming frame
    keep_radius_m: float = 3.0    # forget cells farther than this from the robot
    clear_radius_m: float = 0.11  # the robot's own footprint can't hold an obstacle
    # Planning ignores remembered cells this close to the robot (they stay
    # remembered). Up close the lidar, the swept check and the depth veto
    # guard; a stale cell there (robot carried, or sitting in the camera's
    # near blind zone where it can never be seen free) otherwise pins the
    # robot: start-in-lethal → snapped goal → rotate_timeout (10-31).
    plan_exclude_radius_m: float = 0.20
    min_range_m: float = 0.08     # forward (body x) window that is scored
    max_range_m: float = 0.80
    max_abs_omega_radps: float = 0.40
    stale_s: float = 0.5
    max_cells: int = 400


def memory_config_from_dict(d: Optional[Dict[str, Any]]) -> DepthMemoryConfig:
    """Build from a ``local_drive.depth_memory`` config block; unknown keys
    are ignored so config.json can carry comments-as-keys safely."""
    base = DepthMemoryConfig()
    if not isinstance(d, dict):
        return base
    kw: Dict[str, Any] = {}
    for f in base.__dataclass_fields__:
        if f in d:
            kw[f] = type(getattr(base, f))(d[f])
    return DepthMemoryConfig(**kw)


@dataclass
class _CellState:
    hit_streak: int = 0
    free_streak: int = 0
    confirmed: bool = False
    last_hit_ts: float = 0.0


@dataclass
class FrameStats:
    """What one integrated frame did (for logging/tests)."""
    obstacle_cells: int = 0
    floor_cells: int = 0
    confirmed: int = 0
    cleared: int = 0


class DepthObstacleMemory:
    """See module doc. ``veto_cfg`` supplies the camera model, ROI, floor band
    and structure test (shared with the veto so both see the same points)."""

    def __init__(self, cfg: DepthMemoryConfig, veto_cfg: DepthVetoConfig) -> None:
        self.cfg = cfg
        self.veto_cfg = veto_cfg
        self._cells: Dict[Cell, _CellState] = {}
        self._last_key: Optional[Any] = None
        self.last_stats = FrameStats()

    # ── lifecycle ────────────────────────────────────────────────────
    def reset(self) -> None:
        self._cells.clear()
        self._last_key = None
        self.last_stats = FrameStats()

    def __len__(self) -> int:
        return sum(1 for c in self._cells.values() if c.confirmed)

    # ── geometry ─────────────────────────────────────────────────────
    def _cell_of(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cs = self.cfg.cell_m
        return np.floor(x / cs).astype(np.int64), np.floor(y / cs).astype(np.int64)

    def _centre(self, c: Cell) -> Tuple[float, float]:
        cs = self.cfg.cell_m
        return (c[0] + 0.5) * cs, (c[1] + 0.5) * cs

    # ── update ───────────────────────────────────────────────────────
    def update(
        self, depth_msg: Optional[Dict[str, Any]], *, now_wall: float,
        pose_now: Pose, pose_depth: Optional[Pose], omega_radps: float,
    ) -> bool:
        """Integrate ``depth_msg`` if it is a new, fresh frame (pose_depth =
        odom pose at the frame's capture ts), then expire/prune. Returns True
        when a frame was integrated."""
        cfg = self.cfg
        if not cfg.enabled:
            return False
        integrated = False
        key = depth_frame_key(depth_msg)
        if (depth_msg is not None and pose_depth is not None
                and (key is None or key != self._last_key)):
            age = now_wall - float(depth_msg.get("ts", 0.0) or 0.0)
            if 0.0 <= age <= cfg.stale_s and abs(omega_radps) <= cfg.max_abs_omega_radps:
                self._last_key = key
                integrated = self._integrate(depth_msg, pose_depth,
                                             float(depth_msg.get("ts", now_wall)))
        self._expire(now_wall, pose_now)
        return integrated

    def _integrate(self, msg: Dict[str, Any], pose: Pose, ts: float) -> bool:
        cfg, vcfg = self.cfg, self.veto_cfg
        if msg.get("format") != "depth_uint16_mm":
            return False
        dp = depth_points_for_msg(msg, vcfg)
        stats = FrameStats()
        obs_counts: Dict[Cell, int] = {}
        floor_counts: Dict[Cell, int] = {}
        if dp is not None:
            x = dp.pts[:, 0]
            region = (x >= cfg.min_range_m) & (x <= cfg.max_range_m)
            obs = obstacle_mask(dp, region, vcfg)
            band = vcfg.floor_band_m + vcfg.floor_band_per_m * np.maximum(x, 0.0)
            floor = region & (np.abs(dp.height) <= band)
            c, s = math.cos(pose[2]), math.sin(pose[2])
            ox = pose[0] + c * dp.pts[:, 0] - s * dp.pts[:, 1]
            oy = pose[1] + s * dp.pts[:, 0] + c * dp.pts[:, 1]
            ix, iy = self._cell_of(ox, oy)
            obs_counts = _count_cells(ix[obs], iy[obs])
            floor_counts = _count_cells(ix[floor], iy[floor])

        hit_cells = {k for k, n in obs_counts.items() if n >= cfg.min_pts}
        stats.obstacle_cells = len(hit_cells)
        for k in hit_cells:
            st = self._cells.get(k)
            if st is None:
                st = self._cells[k] = _CellState()
            st.hit_streak += 1
            st.free_streak = 0
            if st.hit_streak >= max(1, cfg.confirm_frames):
                if not st.confirmed:
                    stats.confirmed += 1
                st.confirmed = True
                st.last_hit_ts = ts
        # Every other known cell: a pending cell's streak is broken by any new
        # frame without hits; a remembered one only forgets on seen-free.
        for k in list(self._cells):
            if k in hit_cells:
                continue
            st = self._cells[k]
            st.hit_streak = 0
            if not st.confirmed:
                del self._cells[k]
                continue
            if (floor_counts.get(k, 0) >= cfg.floor_min_pts
                    and obs_counts.get(k, 0) == 0):
                st.free_streak += 1
                stats.floor_cells += 1
                if st.free_streak >= max(1, cfg.clear_frames):
                    del self._cells[k]
                    stats.cleared += 1
            else:
                st.free_streak = 0
        self.last_stats = stats
        return True

    def _expire(self, now_wall: float, pose_now: Pose) -> None:
        cfg = self.cfg
        if not self._cells:
            return
        keep2 = cfg.keep_radius_m ** 2
        clr2 = cfg.clear_radius_m ** 2
        for k in list(self._cells):
            st = self._cells[k]
            if st.confirmed and cfg.ttl_s > 0 and now_wall - st.last_hit_ts > cfg.ttl_s:
                del self._cells[k]
                continue
            cx, cy = self._centre(k)
            d2 = (cx - pose_now[0]) ** 2 + (cy - pose_now[1]) ** 2
            if d2 > keep2 or d2 < clr2:
                del self._cells[k]
        if len(self._cells) > cfg.max_cells:
            # Drop the stalest first (pending cells have last_hit_ts 0).
            order = sorted(self._cells, key=lambda k: self._cells[k].last_hit_ts)
            for k in order[: len(self._cells) - cfg.max_cells]:
                del self._cells[k]

    # ── queries ──────────────────────────────────────────────────────
    def cells_odom(self) -> np.ndarray:
        """(N, 2) centres of remembered cells, odom frame."""
        pts = [self._centre(k) for k, st in self._cells.items() if st.confirmed]
        return np.asarray(pts, dtype=np.float64).reshape(-1, 2)

    def cells_body(self, pose: Pose) -> np.ndarray:
        """(N, 2) centres of remembered cells in the body frame at ``pose``."""
        p = self.cells_odom()
        if p.size == 0:
            return p
        dx = p[:, 0] - pose[0]
        dy = p[:, 1] - pose[1]
        c, s = math.cos(pose[2]), math.sin(pose[2])
        return np.stack([c * dx + s * dy, -s * dx + c * dy], axis=1)


def _count_cells(ix: np.ndarray, iy: np.ndarray) -> Dict[Cell, int]:
    if ix.size == 0:
        return {}
    pairs, counts = np.unique(np.stack([ix, iy], axis=1), axis=0, return_counts=True)
    return {(int(a), int(b)): int(n) for (a, b), n in zip(pairs, counts)}


def stamp_blocked(grid: np.ndarray, meta: Dict[str, Any], pts_body: np.ndarray,
                  *, keep_clear_xy: Optional[Tuple[float, float]] = None,
                  keep_clear_radius_m: float = 0.0) -> int:
    """In place: mark the raster cells holding ``pts_body`` (grid's body frame)
    as blocked (0). Points within ``keep_clear_radius_m`` of ``keep_clear_xy``
    (the robot) are skipped so a stale cell can never pin the robot. Returns
    the number of points stamped."""
    if pts_body is None or len(pts_body) == 0:
        return 0
    res = float(meta.get("resolution_m", 0.0))
    if res <= 0:
        return 0
    pts = np.asarray(pts_body, dtype=np.float64).reshape(-1, 2)
    if keep_clear_xy is not None and keep_clear_radius_m > 0:
        d2 = (pts[:, 0] - keep_clear_xy[0]) ** 2 + (pts[:, 1] - keep_clear_xy[1]) ** 2
        pts = pts[d2 >= keep_clear_radius_m ** 2]
    ox = float(meta.get("origin_x_m", 0.0))
    oy = float(meta.get("origin_y_m", 0.0))
    ix = np.floor((pts[:, 0] - ox) / res).astype(np.int64)
    iy = np.floor((pts[:, 1] - oy) / res).astype(np.int64)
    ok = (ix >= 0) & (ix < grid.shape[0]) & (iy >= 0) & (iy < grid.shape[1])
    grid[ix[ok], iy[ok]] = 0
    return int(np.count_nonzero(ok))


def rounded_xy(pts: Optional[np.ndarray], limit: int) -> Optional[List[List[float]]]:
    """Compact [[x, y], …] (2 dp, ≤ limit, nearest first) for a status message."""
    if pts is None or len(pts) == 0:
        return None
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(p) > limit:
        p = p[np.argsort(np.hypot(p[:, 0], p[:, 1]))[:limit]]
    return [[round(float(a), 2), round(float(b), 2)] for a, b in p]
