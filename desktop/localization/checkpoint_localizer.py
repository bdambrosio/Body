"""Checkpoint-based pose: odom dead-reckon + discrete re-anchor at checkpoints.

The runtime realization of Direction A (docs §6 / Phase 3). Instead of a
continuous metric scan-match against a map we don't trust globally, the pose in
the *map frame* is propagated by raw odom (locally true, drifts globally) and
**re-anchored** whenever the live scan confidently matches a nearby checkpoint
patch (`CheckpointMatcher`). The "local filter" is plain odom — no global
correlation against the distorted map.

`CheckpointLocalizer` is the pure state machine (no Qt / zenoh).
`CheckpointPoseProvider` is the thin adapter implementing the
`PoseProvider.world_pose()` seam used by `HierarchicalDrive`, so it is a drop-in
alternative to `PFPoseProvider`.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

from desktop.localization.checkpoint_matcher import (
    CheckpointMatch,
    CheckpointMatcher,
)

logger = logging.getLogger(__name__)

Pose = Tuple[float, float, float]


@dataclass(frozen=True)
class ReanchorSnap:
    """How far a checkpoint re-anchor moved the pose = the accumulated odom
    dead-reckon drift since the last successful anchor. ``trans_m`` / ``rot_rad``
    are the correction magnitude; ``dist_since_anchor_m`` is the path length the
    drift built up over (so ``trans_m / dist_since_anchor_m`` ≈ drift rate)."""
    checkpoint_id: str
    trans_m: float
    rot_rad: float
    dist_since_anchor_m: float
    score: float


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def pose_relative(a: Pose, b: Pose) -> Pose:
    """Express pose `b` in pose `a`'s frame → local (dx, dy, dθ). The
    frame-independent robot motion from a to b. Inverse of `pose_compose`."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    ca, sa = math.cos(-a[2]), math.sin(-a[2])
    return (dx * ca - dy * sa, dx * sa + dy * ca, _wrap(b[2] - a[2]))


def pose_compose(p: Pose, d: Pose) -> Pose:
    """Apply local delta `d` (in `p`'s frame) to world pose `p` → new pose."""
    cp, sp = math.cos(p[2]), math.sin(p[2])
    return (p[0] + d[0] * cp - d[1] * sp,
            p[1] + d[0] * sp + d[1] * cp,
            _wrap(p[2] + d[2]))


class CheckpointLocalizer:
    """Pure: seed a map-frame pose, dead-reckon it by odom, re-anchor on a
    confident checkpoint match. Times are caller-supplied (monotonic seconds)
    so it stays testable."""

    def __init__(
        self,
        matcher: CheckpointMatcher,
        *,
        reanchor_min_interval_s: float = 0.5,
    ) -> None:
        self._matcher = matcher
        self._reanchor_min_interval_s = reanchor_min_interval_s
        self._map_pose: Optional[Pose] = None
        self._last_odom: Optional[Pose] = None
        self._last_reanchor_t: float = -1e18
        self._last_match: Optional[CheckpointMatch] = None
        self._last_reanchor_snap: Optional[ReanchorSnap] = None
        self._dist_since_anchor: float = 0.0
        self._reanchor_count: int = 0
        self._seeded = False

    @property
    def seeded(self) -> bool:
        return self._seeded

    @property
    def reanchor_count(self) -> int:
        """Number of successful re-anchor snaps applied since construction. The
        hierarchical driver watches this to re-pick the moment a snap moves the
        world pose under a sub-goal anchored in the (uncorrected) odom frame."""
        return self._reanchor_count

    @property
    def last_match(self) -> Optional[CheckpointMatch]:
        return self._last_match

    @property
    def last_reanchor_snap(self) -> Optional[ReanchorSnap]:
        """The most recent re-anchor's drift correction (None until one fires)."""
        return self._last_reanchor_snap

    def seed(self, map_pose: Pose, odom_pose: Pose) -> None:
        """Set the initial map-frame pose and the odom reference it rides on."""
        self._map_pose = (float(map_pose[0]), float(map_pose[1]), float(map_pose[2]))
        self._last_odom = (float(odom_pose[0]), float(odom_pose[1]), float(odom_pose[2]))
        self._seeded = True

    def on_odom(self, odom_pose: Pose) -> None:
        """Advance the map-frame pose by the odom motion since the last call."""
        if not self._seeded:
            return
        if self._last_odom is not None and self._map_pose is not None:
            d = pose_relative(self._last_odom, odom_pose)
            self._map_pose = pose_compose(self._map_pose, d)
            self._dist_since_anchor += math.hypot(d[0], d[1])
        self._last_odom = (float(odom_pose[0]), float(odom_pose[1]), float(odom_pose[2]))

    def try_reanchor(
        self, now: float, angles: Sequence[float], ranges: Sequence[float],
    ) -> Optional[CheckpointMatch]:
        """Throttled checkpoint match against the live scan, using the current
        dead-reckoned pose as the prior. On a confident match, snap the
        map-frame pose to it (odom keeps flowing from where it was). Returns
        the match or None."""
        if not self._seeded or self._map_pose is None:
            return None
        if now - self._last_reanchor_t < self._reanchor_min_interval_s:
            return None
        self._last_reanchor_t = now
        m = self._matcher.match(self._map_pose, angles, ranges)
        if m is not None:
            # The correction (dead-reckoned pose → matched pose) IS the odom
            # drift that accumulated over dist_since_anchor since the last fix.
            rel = pose_relative(self._map_pose, m.pose)
            self._last_reanchor_snap = ReanchorSnap(
                checkpoint_id=m.checkpoint_id,
                trans_m=math.hypot(rel[0], rel[1]),
                rot_rad=rel[2],
                dist_since_anchor_m=self._dist_since_anchor,
                score=m.score,
            )
            self._dist_since_anchor = 0.0
            self._map_pose = m.pose
            self._last_match = m
            self._reanchor_count += 1
        return m

    def pose(self) -> Optional[Pose]:
        return self._map_pose


class CheckpointPoseProvider:
    """`PoseProvider.world_pose()` adapter — drop-in for `PFPoseProvider`.

    Collaborators are injected as callables so this stays decoupled from the
    live stack (and testable):
      * ``odom_fn``  → latest raw odom pose (x, y, θ), or None.
      * ``scan_fn``  → latest live scan as (angles_rad, ranges_m), or None.
      * ``seed_fn``  → an initial map-frame pose to bootstrap from (e.g. the PF
                       posterior or an operator Set-location), or None.
      * ``age_fn``   → odom age in seconds (skew-immune), or None — reported
                       stale past ``max_pose_age_s`` so the drive holds.
    """

    def __init__(
        self,
        localizer: CheckpointLocalizer,
        *,
        odom_fn: Callable[[], Optional[Pose]],
        scan_fn: Callable[[], Optional[Tuple[Sequence[float], Sequence[float]]]],
        seed_fn: Callable[[], Optional[Pose]],
        age_fn: Optional[Callable[[], Optional[float]]] = None,
        max_pose_age_s: float = 0.75,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._loc = localizer
        self._odom_fn = odom_fn
        self._scan_fn = scan_fn
        self._seed_fn = seed_fn
        self._age_fn = age_fn
        self._max_pose_age_s = max_pose_age_s
        self._clock = clock

    def world_pose(self) -> Optional[Pose]:
        if self._age_fn is not None:
            age = self._age_fn()
            if age is not None and age > self._max_pose_age_s:
                return None
        odom = self._odom_fn()
        if odom is None:
            return None
        if not self._loc.seeded:
            seed = self._seed_fn()
            if seed is None:
                return None
            self._loc.seed(seed, odom)
        else:
            self._loc.on_odom(odom)
        scan = self._scan_fn()
        if scan is not None:
            m = self._loc.try_reanchor(self._clock(), scan[0], scan[1])
            if m is not None:
                s = self._loc.last_reanchor_snap
                if s is not None:
                    # The drift instrumentation: snap magnitude = accumulated
                    # dead-reckon error since the last fix. Greppable on the nav
                    # console as "checkpoint: re-anchor".
                    logger.info(
                        "checkpoint: re-anchor cp=%s snap=%.3fm/%+.1f° "
                        "drift-over=%.2fm score=%.2f",
                        s.checkpoint_id, s.trans_m, math.degrees(s.rot_rad),
                        s.dist_since_anchor_m, s.score)
        return self._loc.pose()

    def correction_seq(self) -> int:
        """PoseProvider hook: re-anchor snaps move the world pose without
        touching odom, so the driver re-picks when this advances."""
        return self._loc.reanchor_count
