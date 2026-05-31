"""Phase 2: live read-only lidar overlay link.

Connects (lazily, on demand) to a running bot to provide:
  - a world-frame pose from a read-only MCL localizer
    (`LocalizationController`; **never fuses** — it localizes against the
    on-disk reference map only and publishes nothing that edits it),
  - the live lidar scan (via `DriveClient`),
  - relocate / relocate-at hooks to seat the pose.

`scan_to_world` is the pure transform (body-frame endpoints → world via
the pose); the rest is thin zenoh plumbing reused from the proven Tier-2
console pattern (PF + DriveClient in one process, two zenoh sessions).

The localizer matches against the reference map *as loaded from disk*,
decoupled from the editor's in-memory edits — edits are for improving
*future* localization, and the overlay's job is to show ground truth at
the pose the robot's own localizer would report.
"""
from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def resolve_device(choice: str) -> str:
    """Mirror pi_drive/__main__._resolve_device."""
    if choice != "auto":
        return choice
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def scan_to_world(
    ranges,
    angle_min: float,
    angle_increment: float,
    pose: Tuple[float, float, float],
    *,
    min_range_m: float = 0.05,
    max_range_m: float = 12.0,
) -> np.ndarray:
    """Transform a lidar scan into world-frame endpoints (N, 2).

    Body frame: +x forward, +y left, beam i at
    ``angle_min + i*angle_increment``. World point =
    ``R(theta)·[bx, by] + [px, py]``. Non-finite / out-of-range beams
    are dropped.
    """
    r = np.asarray(ranges, dtype=np.float64).ravel()
    if r.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    ang = angle_min + np.arange(r.size, dtype=np.float64) * angle_increment
    valid = np.isfinite(r) & (r > min_range_m) & (r < max_range_m)
    r, ang = r[valid], ang[valid]
    if r.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    bx = r * np.cos(ang)
    by = r * np.sin(ang)
    px, py, th = float(pose[0]), float(pose[1]), float(pose[2])
    c, s = math.cos(th), math.sin(th)
    wx = px + c * bx - s * by
    wy = py + s * bx + c * by
    return np.column_stack([wx, wy])


class LiveLink:
    """Owns the read-only localizer + scan client. Construct cheaply;
    `connect()` does the heavy work (imports torch, opens zenoh)."""

    def __init__(self, router: str, map_path: str, *,
                 pf_device: str = "auto", pf_particles: int = 5000) -> None:
        self._router = router
        self._map_path = map_path
        self._pf_device = pf_device
        self._pf_particles = pf_particles
        self._localizer = None
        self._drive = None
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> Tuple[bool, Optional[str]]:
        if self._connected:
            return True, None
        try:
            from desktop.localization.config import LocalizationConfig
            from desktop.localization.controller import LocalizationController
            from desktop.pi_drive.drive_client import DriveClient
            from desktop.reference_map.legacy_convert import load_map_auto

            reference_map = load_map_auto(self._map_path)
            loc_cfg = LocalizationConfig(
                router=self._router, map_path=self._map_path,
                pf_device=resolve_device(self._pf_device),
                pf_n_particles=self._pf_particles,
            )
            self._localizer = LocalizationController(loc_cfg, reference_map)
            self._drive = DriveClient(self._router)
            ok_loc, err_loc = self._localizer.connect()
            ok_drv, err_drv = self._drive.connect()
            if not (ok_loc and ok_drv):
                self.disconnect()
                return False, err_loc or err_drv or "connect failed"
        except Exception as e:  # noqa: BLE001 — surface to operator
            logger.exception("live connect failed")
            self.disconnect()
            return False, f"{type(e).__name__}: {e}"
        self._connected = True
        return True, None

    def disconnect(self) -> None:
        for obj, name in ((self._localizer, "shutdown"), (self._drive, "shutdown")):
            if obj is not None:
                try:
                    getattr(obj, name)()
                except Exception:
                    logger.exception("live disconnect: %s failed", name)
        self._localizer = None
        self._drive = None
        self._connected = False

    # ── Reads ────────────────────────────────────────────────────────

    def latest_pose(self) -> Optional[Tuple[float, float, float]]:
        if self._localizer is None:
            return None
        lp = self._localizer.pose_source.latest_pose()
        if lp is None:
            return None
        pose = lp[0]  # (Pose, ts) → Pose is indexable (x, y, theta)
        return (float(pose[0]), float(pose[1]), float(pose[2]))

    def latest_scan_world(self, *, max_range_m: float = 12.0
                          ) -> Optional[np.ndarray]:
        if self._drive is None:
            return None
        scan = self._drive.latest_scan()
        pose = self.latest_pose()
        if not scan or pose is None or not scan.get("ranges"):
            return None
        return scan_to_world(
            scan.get("ranges"),
            float(scan.get("angle_min", 0.0)),
            float(scan.get("angle_increment", 0.0)),
            pose, max_range_m=max_range_m,
        )

    def scan_age_s(self, now: float) -> Optional[float]:
        if self._drive is None:
            return None
        scan = self._drive.latest_scan()
        if not scan:
            return None
        ts = float(scan.get("ts", 0.0))
        return (now - ts) if ts else None

    # ── Relocate ───────────────────────────────────────────────────────

    def relocate(self) -> dict:
        if self._localizer is None:
            return {"success": False, "reason": "not_connected"}
        return self._localizer.request_relocate(reason="map_editor")

    def relocate_at(self, x: float, y: float) -> dict:
        if self._localizer is None:
            return {"success": False, "reason": "not_connected"}
        return self._localizer.request_relocate_at(x, y, reason="map_editor")
