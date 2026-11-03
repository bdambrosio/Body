"""Path geometry for Tier-3 path mode (nav redesign step 3).

The desktop streams a polyline (``body/drive/goto`` ``kind: "path"``); Tier-3
follows it instead of re-planning to a point. Pure functions: arc-length
projection with a monotone hint, resampling, checking the path ahead against
the live costmap, finding a rejoin point past an obstruction, and how far a
local detour strays from the path. See docs/nav_step3_path_streaming.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

Point2 = Tuple[float, float]


@dataclass(frozen=True)
class PathFollowConfig:
    horizon_m: float = 2.0               # path checked against the live raster this far ahead
    max_rejoin_deviation_m: float = 0.5  # a local detour may stray this far from the path
    step_m: float = 0.04                 # resampling step for checks (half a raster cell)
    rejoin_clear_m: float = 0.12         # a rejoin point needs this much clear path after it
    rejoin_lead_m: float = 0.5           # aim the detour this far past the first clear point
    end_hold_m: float = 0.10             # end is not the goal: hold this close to it
    default_ttl_s: float = 1.0           # refresh deadline when the message has none
    stale_factor: float = 2.0            # no refresh for factor × ttl → BLOCKED path_stale
    back_window_m: float = 0.30          # projection may slide back this far (re-anchoring jitter)


def cumulative(path: Sequence[Point2]) -> np.ndarray:
    p = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    if len(p) < 2:
        return np.zeros(len(p))
    d = np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1]))
    return np.concatenate([[0.0], np.cumsum(d)])


def project(path: Sequence[Point2], cum: np.ndarray, pt: Point2,
            s_hint: Optional[float] = None, back_m: float = 0.3) -> Tuple[float, float]:
    """(arc length of ``pt``'s projection, lateral distance). With ``s_hint``
    the projection may not go further back than ``s_hint - back_m``, so a
    path that doubles back never re-captures the robot on its earlier leg.
    Near-ties (2 cm) go to the segment furthest along."""
    p = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    if len(p) == 1:
        return 0.0, math.hypot(p[0, 0] - pt[0], p[0, 1] - pt[1])
    a, b = p[:-1], p[1:]
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(axis=1), 1e-12)
    t = np.clip(((pt[0] - a[:, 0]) * ab[:, 0] + (pt[1] - a[:, 1]) * ab[:, 1]) / L2, 0.0, 1.0)
    q = a + ab * t[:, None]
    d = np.hypot(q[:, 0] - pt[0], q[:, 1] - pt[1])
    s = cum[:-1] + t * np.sqrt(L2)
    if s_hint is not None:
        d = np.where(s < s_hint - back_m, np.inf, d)
        if not np.isfinite(d).any():
            k = len(d) - 1
            return float(s[k]), float(np.hypot(q[k, 0] - pt[0], q[k, 1] - pt[1]))
    dmin = float(d.min())
    cand = np.nonzero(d <= dmin + 0.02)[0]
    k = int(cand[-1])
    return float(s[k]), float(d[k])


def point_at(path: Sequence[Point2], cum: np.ndarray, s: float) -> Point2:
    p = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    if len(p) == 1 or s <= 0.0:
        return (float(p[0, 0]), float(p[0, 1]))
    if s >= cum[-1]:
        return (float(p[-1, 0]), float(p[-1, 1]))
    k = int(np.searchsorted(cum, s, side="right") - 1)
    seg = cum[k + 1] - cum[k]
    t = 0.0 if seg <= 0 else (s - cum[k]) / seg
    return (float(p[k, 0] + t * (p[k + 1, 0] - p[k, 0])),
            float(p[k, 1] + t * (p[k + 1, 1] - p[k, 1])))


def resample(path: Sequence[Point2], cum: np.ndarray, s0: float, s1: float,
             step: float) -> List[Tuple[float, Point2]]:
    """[(s, point)] from ``s0`` to ``s1`` every ``step`` (both ends included)."""
    s1 = min(s1, float(cum[-1]) if len(cum) else 0.0)
    if s1 <= s0:
        return [(s0, point_at(path, cum, s0))]
    n = max(1, int(math.ceil((s1 - s0) / step)))
    return [(s0 + (s1 - s0) * k / n, point_at(path, cum, s0 + (s1 - s0) * k / n))
            for k in range(n + 1)]


def raster_lookup(mask: np.ndarray, meta) -> Callable[[Point2], Optional[bool]]:
    """f(point in the raster's frame) → mask value, or None off the raster."""
    res = float(meta["resolution_m"])
    ox, oy = float(meta["origin_x_m"]), float(meta["origin_y_m"])
    nx, ny = mask.shape

    def f(pt: Point2) -> Optional[bool]:
        i = int(math.floor((pt[0] - ox) / res))
        j = int(math.floor((pt[1] - oy) / res))
        if not (0 <= i < nx and 0 <= j < ny):
            return None
        return bool(mask[i, j])
    return f


def first_blocked(samples: List[Tuple[float, Point2]],
                  lethal: Callable[[Point2], Optional[bool]]) -> Tuple[Optional[int], int]:
    """(index of the first lethal sample or None, number of samples before
    the path leaves the raster). A block past the raster edge is not seen."""
    first = None
    for k, (_s, pt) in enumerate(samples):
        v = lethal(pt)
        if v is None:
            return first, k
        if v and first is None:
            first = k
    return first, len(samples)


def rejoin_index(samples: List[Tuple[float, Point2]], i0: int,
                 lethal: Callable[[Point2], Optional[bool]], clear_m: float) -> Optional[int]:
    """First sample after ``i0`` that is clear and followed by ``clear_m`` of
    clear path (or the path / raster end). None if the path never clears on
    the raster."""
    n = len(samples)
    k = i0 + 1
    while k < n:
        v = lethal(samples[k][1])
        if v is None:
            return None
        if not v:
            j = k
            ok = True
            while j + 1 < n and samples[j + 1][0] - samples[k][0] <= clear_m:
                j += 1
                vj = lethal(samples[j][1])
                if vj is None:
                    break
                if vj:
                    ok = False
                    break
            if ok:
                return k
            k = j + 1
            continue
        k += 1
    return None


def max_deviation(poly: Sequence[Point2], path: Sequence[Point2], cum: np.ndarray) -> float:
    """Largest distance from any vertex of ``poly`` to the polyline ``path``."""
    if not len(poly):
        return 0.0
    return max(project(path, cum, p)[1] for p in poly)
