"""Patrols — closed-loop waypoint missions and waypoint-to-costmap
adaptation.

A patrol is an ordered list of waypoints the robot drives through, in
sequence, rotating in place at each to face the next. When `loop=True`
the last waypoint connects back to the first, and the robot drives
the loop `laps` times before terminating (default: 1 — one full
circuit and stop).

Two storage locations, one format:

  1. Per-snapshot embed: `~/Body/sessions/<sid>/snap_<ts>/patrols.json`
     written next to `layers.npz` when the operator saves a bundle.
  2. Standalone library: `~/Body/patrols/<name>.json` — what the Load
     dropdown reads.

The `session_id` field anchors a patrol to the snapshot it was
authored against. On reload the UI compares against the live session
and warns on mismatch ("warn + allow" per the locked design); the
operator can Re-localize first if they want strict alignment.

The runtime state machine — wp_index, lap_index, terminal detection —
lives in `PatrolRunner` so that `Patrol` itself stays pure data.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Data model ──────────────────────────────────────────────────────


@dataclass
class Waypoint:
    x_m: float
    y_m: float
    # If False, the patrol skips the rotate-to-face-next step at this
    # waypoint (drives the next leg with whatever heading the follower
    # picks). Defaults to True — "rotate before leaving" is the
    # intended behavior.
    face_next: bool = True
    # Optional pause after arrival, before rotating / advancing. v1.1
    # — schema slot only; the runtime ignores it for now.
    hold_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x_m": float(self.x_m),
            "y_m": float(self.y_m),
            "face_next": bool(self.face_next),
            "hold_s": float(self.hold_s),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Waypoint":
        return cls(
            x_m=float(d["x_m"]),
            y_m=float(d["y_m"]),
            face_next=bool(d.get("face_next", True)),
            hold_s=float(d.get("hold_s", 0.0)),
        )


@dataclass
class Patrol:
    name: str
    session_id: str
    authored_utc: str
    # Closed-loop *shape*: the polyline drawn on the map closes back
    # to wp[0]. Independent of how many `laps` the runtime executes.
    loop: bool = True
    # Number of laps to execute. None = unlimited (operator cancels).
    # Default 1 — one full circuit then stop.
    laps: Optional[int] = 1
    waypoints: List[Waypoint] = field(default_factory=list)

    # ── Serialization ────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "name": self.name,
            "session_id": self.session_id,
            "authored_utc": self.authored_utc,
            "loop": bool(self.loop),
            "laps": self.laps,  # None or int
            "waypoints": [w.to_dict() for w in self.waypoints],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Patrol":
        laps_raw = d.get("laps", 1)
        laps: Optional[int] = None if laps_raw is None else int(laps_raw)
        return cls(
            name=str(d.get("name") or "unnamed"),
            session_id=str(d.get("session_id") or ""),
            authored_utc=str(d.get("authored_utc") or ""),
            loop=bool(d.get("loop", True)),
            laps=laps,
            waypoints=[Waypoint.from_dict(w) for w in d.get("waypoints", [])],
        )

    # ── Convenience ──────────────────────────────────────────────────

    def append(self, x_m: float, y_m: float, *,
               face_next: bool = True, hold_s: float = 0.0) -> Waypoint:
        wp = Waypoint(x_m=float(x_m), y_m=float(y_m),
                      face_next=face_next, hold_s=hold_s)
        self.waypoints.append(wp)
        return wp

    def remove_last(self) -> Optional[Waypoint]:
        return self.waypoints.pop() if self.waypoints else None

    def clear(self) -> None:
        self.waypoints.clear()

    def __len__(self) -> int:
        return len(self.waypoints)


# ── File I/O ────────────────────────────────────────────────────────


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(name: str) -> str:
    """Sanitize an operator-supplied patrol name into something safe
    for a filename. Empty / all-special input collapses to 'patrol'.
    """
    s = _SAFE_NAME_RE.sub("_", name.strip()) if name else ""
    return s.strip("._") or "patrol"


def write_to_file(patrol: Patrol, path: str) -> str:
    """Write `patrol` to `path` as pretty-printed JSON. Returns the
    absolute path written. Creates parent dirs as needed.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(patrol.to_dict(), fh, indent=2)
    return os.path.abspath(path)


def load_from_file(path: str) -> Patrol:
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    return Patrol.from_dict(d)


# ── Library ─────────────────────────────────────────────────────────


def library_dir() -> str:
    """Default standalone-library directory: `~/Body/patrols/`. Created
    on first use."""
    p = os.path.expanduser("~/Body/patrols")
    os.makedirs(p, exist_ok=True)
    return p


def list_library() -> List[str]:
    """Names (without .json) of patrols in the library, sorted."""
    d = library_dir()
    try:
        out = []
        for fn in os.listdir(d):
            if fn.endswith(".json"):
                out.append(fn[:-len(".json")])
        out.sort()
        return out
    except FileNotFoundError:
        return []


def library_path(name: str) -> str:
    return os.path.join(library_dir(), f"{safe_filename(name)}.json")


def load_from_library(name: str) -> Patrol:
    return load_from_file(library_path(name))


def save_to_library(patrol: Patrol) -> str:
    """Write to `library_dir()/<safe(name)>.json`. Returns the path."""
    return write_to_file(patrol, library_path(patrol.name))


def delete_from_library(name: str) -> bool:
    """Returns True if a file was removed, False if it didn't exist."""
    path = library_path(name)
    try:
        os.remove(path)
        return True
    except FileNotFoundError:
        return False


def new_empty(*, session_id: str, name: Optional[str] = None,
              loop: bool = True, laps: Optional[int] = 1) -> Patrol:
    """Construct a fresh empty patrol stamped with the given session
    id and the current UTC timestamp. `name` defaults to a
    timestamp-based placeholder the operator can rename later."""
    if not name:
        name = f"unsaved_{time.strftime('%H%M%S')}"
    return Patrol(
        name=name,
        session_id=session_id or "",
        authored_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        loop=loop,
        laps=laps,
        waypoints=[],
    )


# ── Runtime state machine ───────────────────────────────────────────


@dataclass
class PatrolRunner:
    """Tracks the iterator-like state of a running patrol.

    main_window owns one of these for the duration of a patrol
    mission. Mission's `wp_index` / `lap_index` mirror this state for
    the UI; the authoritative ticker is here.

    Lap accounting: a lap closes when we arrive back at wp[0] after
    visiting at least one other waypoint. The initial drive *to* wp[0]
    is NOT a lap closure.
    """

    patrol: Patrol
    wp_index: int = 0       # index of current target waypoint
    lap_index: int = 0      # number of laps *completed* so far
    _legs_completed: int = 0

    @property
    def n(self) -> int:
        return len(self.patrol.waypoints)

    def current_target(self) -> Optional[Waypoint]:
        if self.n == 0:
            return None
        return self.patrol.waypoints[self.wp_index]

    def next_target_xy_after(
        self, wp_index: int,
    ) -> Optional[Tuple[float, float]]:
        """Return the (x, y) of the waypoint that would FOLLOW
        `wp_index` in the patrol's traversal order, or None if no
        successor exists (open path at the end).

        Used to compute the rotate-to-face-next heading at the moment
        of arrival — we need to know where we're going next before
        the runner advances.
        """
        n = self.n
        if n == 0:
            return None
        if not self.patrol.loop:
            if wp_index >= n - 1:
                return None
            nxt = self.patrol.waypoints[wp_index + 1]
            return (nxt.x_m, nxt.y_m)
        # Closed loop: next is (wp_index + 1) % n. For the lap-closing
        # leg (current == 0 with legs_completed > 0 about to terminate),
        # we still report wp[1] as "next" — main_window's terminal
        # check happens before we'd need to rotate further.
        nxt = self.patrol.waypoints[(wp_index + 1) % n]
        return (nxt.x_m, nxt.y_m)

    def on_arrived(self) -> Tuple[Optional[int], bool]:
        """Record arrival at the current target. Return the new
        `(next_wp_index, lap_completed)`. `next_wp_index` is None when
        the patrol is now terminal — caller transitions the mission to
        ARRIVED.

        Mutates `wp_index` / `lap_index` / `_legs_completed` in place.
        """
        prior = self.wp_index
        self._legs_completed += 1
        n = self.n
        if n == 0:
            return (None, False)
        p = self.patrol

        if not p.loop:
            if prior >= n - 1:
                return (None, False)
            self.wp_index = prior + 1
            return (self.wp_index, False)

        # loop=True. Lap closure: arrived at wp[0] after at least one
        # prior leg. The initial leg (from robot start position to
        # wp[0]) is leg #1 and does NOT close a lap.
        if self._legs_completed > 1 and prior == 0:
            self.lap_index += 1
            if p.laps is not None and self.lap_index >= p.laps:
                return (None, True)  # terminal, with lap_complete
            # Start the next lap. wp[0] → wp[1] → ...
            self.wp_index = 1 if n > 1 else 0
            return (self.wp_index, True)

        # Mid-lap progression. Closing-the-loop leg is when prior is
        # the last waypoint and we have more laps to go (or unlimited).
        if prior >= n - 1:
            self.wp_index = 0
            return (0, False)
        self.wp_index = prior + 1
        return (self.wp_index, False)


# ── Waypoint → costmap snap ─────────────────────────────────────────


@dataclass
class SnapResult:
    """Outcome of `snap_to_accessible`. `snapped_xy` is the world point
    the planner / follower should aim at; equal to `original_xy` when
    no relocation was needed. `distance_m` is the displacement
    (0 if no snap), useful for trace events and for filtering noise.
    """
    snapped_xy: Tuple[float, float]
    original_xy: Tuple[float, float]
    cost_at_original: float
    cost_at_snapped: float
    distance_m: float
    snapped: bool

    @property
    def needs_snap(self) -> bool:
        return self.snapped


def snap_to_accessible(
    costmap: Any,
    xy: Tuple[float, float],
    *,
    radius_m: float = 1.0,
    cost_threshold: Optional[float] = None,
) -> Optional[SnapResult]:
    """Find the nearest accessible cell to `xy` in `costmap` and return
    a `SnapResult`. "Accessible" = not lethal AND cost <
    `cost_threshold` (default: `costmap.config.halo_max / 2`, so cells
    deep in inflation halo are skipped).

    Search order:
      1. If the original cell is already accessible, return it
         unchanged with `snapped=False`.
      2. Otherwise, scan expanding Chebyshev rings outward. Within each
         ring, pick the lowest-cost acceptable cell. Stop at the first
         ring with an acceptable cell.
      3. If no acceptable cell exists within `radius_m`, return None.

    Duck-typed on the costmap: needs `.lethal` / `.cost` arrays, a
    `.meta` dict with `resolution_m` / `origin_x_m` / `origin_y_m`, and
    a `.config.halo_max` (for the default threshold). Returns None if
    `xy` is outside the grid entirely.
    """
    try:
        res = float(costmap.meta["resolution_m"])
        ox = float(costmap.meta["origin_x_m"])
        oy = float(costmap.meta["origin_y_m"])
    except (KeyError, AttributeError, TypeError):
        return None
    lethal = costmap.lethal
    cost = costmap.cost
    nx, ny = lethal.shape

    if cost_threshold is None:
        try:
            halo_max = float(costmap.config.halo_max)
        except AttributeError:
            halo_max = 100.0
        cost_threshold = halo_max / 2.0

    x_w, y_w = float(xy[0]), float(xy[1])
    i0 = int(math.floor((x_w - ox) / res + 1e-9))
    j0 = int(math.floor((y_w - oy) / res + 1e-9))
    if not (0 <= i0 < nx and 0 <= j0 < ny):
        return None

    cost_original = float(cost[i0, j0])

    def acceptable(i: int, j: int) -> bool:
        if not (0 <= i < nx and 0 <= j < ny):
            return False
        if bool(lethal[i, j]):
            return False
        return float(cost[i, j]) < cost_threshold

    def cell_to_world(i: int, j: int) -> Tuple[float, float]:
        return (ox + (i + 0.5) * res, oy + (j + 0.5) * res)

    if acceptable(i0, j0):
        return SnapResult(
            snapped_xy=(x_w, y_w),
            original_xy=(x_w, y_w),
            cost_at_original=cost_original,
            cost_at_snapped=cost_original,
            distance_m=0.0,
            snapped=False,
        )

    radius_cells = max(1, int(math.ceil(radius_m / res)))
    # Expand Chebyshev rings. Within a ring, pick the lowest-cost
    # acceptable cell (tie-breaks toward "more centered"). On a real
    # tie of cost, the (di, dj) iteration order picks deterministically.
    for r in range(1, radius_cells + 1):
        best: Optional[Tuple[int, int]] = None
        best_cost = float("inf")
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if max(abs(di), abs(dj)) != r:
                    continue
                ii, jj = i0 + di, j0 + dj
                if not acceptable(ii, jj):
                    continue
                c = float(cost[ii, jj])
                if c < best_cost:
                    best_cost = c
                    best = (ii, jj)
        if best is not None:
            sx, sy = cell_to_world(*best)
            return SnapResult(
                snapped_xy=(sx, sy),
                original_xy=(x_w, y_w),
                cost_at_original=cost_original,
                cost_at_snapped=best_cost,
                distance_m=math.hypot(sx - x_w, sy - y_w),
                snapped=True,
            )
    return None
