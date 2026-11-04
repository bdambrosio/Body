"""Task layer above the hierarchical drive (explore Phase D, docs/explore_phase_d.md).

A ``Task`` decides where to go next; the ``TaskRunner`` drives each ``Leg``
with a fresh one-waypoint ``HierarchicalDrive`` (route planning, path
streaming, watchdog, recovery ladder, pose-doubt suspend all as for a patrol)
and hands the outcome back. Patrol keeps its own Go path; ``ExploreTask`` is
the first task.

No Qt here: main_window ticks the runner from its drive timer and supplies the
callables in ``ExploreContext``.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np

from desktop.nav.explore import (
    ExploreConfig, ExploreMap, ExplorePlanner, investigate_spot, reachable_distances,
)
from desktop.nav.hierarchical_drive import HierState

logger = logging.getLogger(__name__)

Pt = Tuple[float, float]

ARRIVED, UNREACHABLE, FAILED, PREEMPTED = "arrived", "unreachable", "failed", "preempted"


@dataclass
class Leg:
    xy: Pt
    purpose: str                        # viewpoint | investigate | home
    arrival_tol_m: float
    final_heading: Optional[float] = None
    dwell_s: float = 0.0
    give_up_after: Optional[int] = 1    # ladder cycles before abandoning; None = never
    ref: Any = None                     # task's own handle (viewpoint index, change id)


@dataclass
class LegOutcome:
    result: str                         # arrived | unreachable | failed | preempted
    elapsed_s: float


class Task(Protocol):
    name: str

    def next_leg(self, now: float) -> Optional[Leg]: ...
    def on_leg_done(self, leg: Leg, outcome: LegOutcome, now: float) -> None: ...
    def preempt(self, now: float, leg: Leg) -> bool: ...
    def status(self) -> Dict[str, Any]: ...
    def summary(self) -> Dict[str, Any]: ...


class TaskRunner:
    """Drives a task leg by leg. ``start_leg(leg)`` builds and starts the drive
    (main_window: one-waypoint patrol, recovery, pose doubt) and returns it.
    The operator's Stop (drive → IDLE) ends the task."""

    def __init__(self, task: Task, start_leg: Callable[[Leg], Any],
                 on_finished: Optional[Callable[[str, Dict[str, Any]], None]] = None) -> None:
        self.task = task
        self._start_leg = start_leg
        self._on_finished = on_finished
        self.drive: Any = None
        self.leg: Optional[Leg] = None
        self._leg_t0 = 0.0
        self._dwell_until: Optional[float] = None
        self.finished: Optional[str] = None      # done | stopped | error

    def start(self, now: float) -> None:
        logger.info("task %s: start", self.task.name)
        self._next(now)

    def _next(self, now: float) -> None:
        leg = self.task.next_leg(now)
        if leg is None:
            self._finish("done")
            return
        self.leg, self._leg_t0, self._dwell_until = leg, now, None
        logger.info("task %s: leg %s to (%.2f, %.2f)%s", self.task.name, leg.purpose,
                    leg.xy[0], leg.xy[1],
                    f" facing {math.degrees(leg.final_heading):.0f}°"
                    if leg.final_heading is not None else "")
        self.drive = self._start_leg(leg)

    def _done(self, result: str, now: float) -> None:
        leg = self.leg
        self.task.on_leg_done(leg, LegOutcome(result, now - self._leg_t0), now)
        self._next(now)

    def _finish(self, why: str) -> None:
        if self.finished:
            return
        self.finished = why
        summ = self.task.summary()
        logger.info("task %s: finished (%s) %s", self.task.name, why, summ.get("headline", ""))
        if self._on_finished is not None:
            self._on_finished(why, summ)

    def stop(self) -> None:
        if self.drive is not None and not self.finished:
            self.drive.stop()
        self._finish("stopped")

    def tick(self, now: float) -> None:
        if self.finished or self.drive is None:
            return
        st = self.drive.state()
        if st == HierState.ARRIVED:
            if self.leg.dwell_s > 0:
                if self._dwell_until is None:
                    self._dwell_until = now + self.leg.dwell_s
                if now < self._dwell_until:
                    return
            self._done(ARRIVED, now)
        elif st == HierState.FAILED:
            reason = getattr(self.drive, "fail_reason", lambda: None)()
            self._done(UNREACHABLE if reason == "abandoned" else FAILED, now)
        elif st == HierState.IDLE:
            self._finish("stopped")                  # operator Stop
        elif self.task.preempt(now, self.leg):
            self.drive.stop()
            self._done(PREEMPTED, now)


@dataclass
class ExploreContext:
    """What ExploreTask needs from the running nav (all thread-safe reads)."""
    costmap: Callable[[], Any]                    # live planning costmap
    robot_xy: Callable[[], Optional[Pt]]
    seen: Callable[[], Optional[np.ndarray]]      # bool (nx, ny): session grid observed
    # True: still blocked there; False: seen free; None: not observed.
    occupied_near: Callable[[Pt, float], Optional[bool]]
    home_xy: Pt
    home_heading: Optional[float] = None          # world rad; face it on the home leg
    on_investigated: Callable[[int, str, Dict[str, Any]], None] = lambda cid, o, d: None


@dataclass
class _Pending:
    change_id: int
    xy: Pt
    first_view: Optional[Pt]
    verdict: Optional[str]
    gone: bool = False
    label: Optional[str] = None


class ExploreTask:
    """Greedy next-best-view over the mapped house + investigate legs for new
    changes; returns home at the end (docs/explore_phase_d.md)."""

    name = "explore"
    INVESTIGATE_VERDICTS = ("novel", "flagged")

    def __init__(self, emap: ExploreMap, ctx: ExploreContext,
                 cfg: Optional[ExploreConfig] = None) -> None:
        self.cfg = cfg or emap.cfg
        self.emap = emap
        self.ctx = ctx
        self.planner = ExplorePlanner(emap)
        self._lock = threading.Lock()
        self._t0: Optional[float] = None
        self._queue: List[_Pending] = []
        self._changes: Dict[int, _Pending] = {}
        self._investigated: Dict[int, str] = {}
        self._phase = "explore"                   # explore | home | done
        self._end_reason = ""
        self._legs: List[Dict[str, Any]] = []
        self._counts: Dict[str, int] = {}
        self._last_now: Optional[float] = None

    # ── change reports (reporter listener; scan thread) ──────────────
    def on_change(self, rec: Dict[str, Any]) -> None:
        cid = rec.get("change_id")
        if cid is None:
            return
        with self._lock:
            if rec.get("type") == "label":
                p = self._changes.get(cid)
                if p is not None:
                    p.label = rec.get("label")
                return
            if rec.get("type") != "change":
                return
            kind, status, verdict = rec.get("kind"), rec.get("status"), rec.get("verdict")
            key = f"{kind}/{status}" + (f"/{verdict}" if verdict else "")
            self._counts[key] = self._counts.get(key, 0) + 1
            if status == "gone":
                p = self._changes.get(cid)
                if p is not None:
                    p.gone = True
                return
            if kind != "new" or status != "appeared":
                return
            if verdict is not None and verdict not in self.INVESTIGATE_VERDICTS:
                return
            robot = rec.get("robot")
            p = _Pending(int(cid), (float(rec["x"]), float(rec["y"])),
                         (float(robot[0]), float(robot[1])) if robot else None, verdict)
            self._changes[p.change_id] = p
            self._queue.append(p)

    # ── Task ─────────────────────────────────────────────────────────
    def _over_budget(self, now: float) -> bool:
        return self._t0 is not None and now - self._t0 >= self.cfg.budget_min * 60.0

    def _dist(self) -> Optional[np.ndarray]:
        cm = self.ctx.costmap()
        xy = self.ctx.robot_xy()
        if cm is None or xy is None:
            return None
        return reachable_distances(cm.lethal, getattr(cm, "unknown", None), cm.meta, xy)

    def _home_leg(self, why: str) -> Leg:
        self._phase = "home"
        self._end_reason = why
        logger.info("explore: %s — going home", why)
        return Leg(self.ctx.home_xy, "home", self.cfg.home_arrival_tol_m,
                   final_heading=self.ctx.home_heading, give_up_after=None)

    def next_leg(self, now: float) -> Optional[Leg]:
        if self._t0 is None:
            self._t0 = now
        self._last_now = now
        if self._phase == "done":
            return None
        if self._phase == "home":
            self._phase = "done"
            return None
        if self._over_budget(now):
            return self._home_leg(f"time budget ({self.cfg.budget_min:.0f} min) spent")
        dist = self._dist()
        if dist is None:
            return self._home_leg("no costmap / pose")
        leg = self._next_investigation(dist)
        if leg is not None:
            return leg
        cov = self.planner.coverage(self.ctx.seen())
        if cov >= self.cfg.target_coverage:
            return self._home_leg(f"coverage {cov:.0%} reached")
        ch = self.planner.choose(self.ctx.seen(), dist)
        if ch is None:
            return self._home_leg(f"nothing left worth a look (coverage {cov:.0%})")
        logger.info("explore: viewpoint %d at (%.2f, %.2f): gain %d cells, %.1f m away, "
                    "coverage %.0f%%", ch.index, ch.xy[0], ch.xy[1], ch.gain, ch.travel_m,
                    100 * cov)
        return Leg(ch.xy, "viewpoint", self.cfg.viewpoint_arrival_tol_m,
                   give_up_after=self.cfg.give_up_after_cycles, ref=ch.index)

    def _next_investigation(self, dist: np.ndarray) -> Optional[Leg]:
        with self._lock:
            if len(self._investigated) >= self.cfg.max_investigations:
                self._queue.clear()
                return None
            while self._queue:
                # nearest first
                self._queue.sort(key=lambda p: _finite(dist, self.emap.meta, p.xy))
                p = self._queue.pop(0)
                if p.change_id in self._investigated:
                    continue
                if p.gone:
                    self._investigated[p.change_id] = "gone"
                    self.ctx.on_investigated(p.change_id, "gone", {"before_visit": True})
                    continue
                spot = investigate_spot(self.emap.drive, self.emap.meta, dist, p.xy,
                                        p.first_view, self.cfg)
                if spot is None:
                    self._investigated[p.change_id] = "unreachable"
                    self.ctx.on_investigated(p.change_id, "unreachable", {"why": "no spot"})
                    continue
                logger.info("explore: investigate change #%d at (%.2f, %.2f) from "
                            "(%.2f, %.2f)%s", p.change_id, p.xy[0], p.xy[1], spot.xy[0],
                            spot.xy[1], "" if spot.new_angle else " (same angle)")
                return Leg(spot.xy, "investigate", self.cfg.investigate_arrival_tol_m,
                           final_heading=spot.heading, dwell_s=self.cfg.investigate_dwell_s,
                           give_up_after=self.cfg.give_up_after_cycles, ref=p.change_id)
        return None

    def on_leg_done(self, leg: Leg, outcome: LegOutcome, now: float) -> None:
        self._last_now = now
        self._legs.append({"purpose": leg.purpose, "xy": [round(leg.xy[0], 2), round(leg.xy[1], 2)],
                           "result": outcome.result, "s": round(outcome.elapsed_s, 1)})
        if leg.purpose == "viewpoint":
            if outcome.result == ARRIVED:
                self.planner.mark_visited(leg.ref)
            elif outcome.result in (UNREACHABLE, FAILED):
                self.planner.mark_abandoned(leg.ref)
        elif leg.purpose == "investigate":
            cid = leg.ref
            with self._lock:
                p = self._changes.get(cid)
            if outcome.result != ARRIVED:
                res = "unreachable" if outcome.result != PREEMPTED else "preempted"
                detail: Dict[str, Any] = {"leg": outcome.result}
            elif p is not None and p.gone:
                res, detail = "gone", {}
            else:
                occ = self.ctx.occupied_near(p.xy if p else leg.xy, 0.15)
                res = {True: "confirmed", False: "dismissed", None: "unseen"}[occ]
                detail = {}
            if p is not None and p.label:
                detail["label"] = p.label
            with self._lock:
                self._investigated[cid] = res
            self.ctx.on_investigated(cid, res, detail)
        elif leg.purpose == "home":
            self._phase = "done"

    def preempt(self, now: float, leg: Leg) -> bool:
        return leg.purpose != "home" and self._over_budget(now)

    def status(self) -> Dict[str, Any]:
        return {"phase": self._phase, "coverage": self.planner.coverage(self.ctx.seen()),
                "visited": len(self.planner.visited), "investigated": len(self._investigated),
                "queued": len(self._queue)}

    def summary(self) -> Dict[str, Any]:
        cov = self.planner.coverage(self.ctx.seen())
        with self._lock:
            inv = dict(self._investigated)
            changes = {cid: {"xy": [round(p.xy[0], 2), round(p.xy[1], 2)], "label": p.label,
                             "verdict": p.verdict, "investigated": inv.get(cid)}
                       for cid, p in self._changes.items()}
            counts = dict(self._counts)
        outcomes: Dict[str, int] = {}
        for v in inv.values():
            outcomes[v] = outcomes.get(v, 0) + 1
        return {
            "headline": (f"coverage {cov:.0%}, {len(self.planner.visited)} viewpoints, "
                         f"{len(self.planner.abandoned)} abandoned, investigations {outcomes}; "
                         f"{self._end_reason}"),
            "coverage": round(cov, 3),
            "end_reason": self._end_reason,
            "duration_s": (round(self._last_now - self._t0, 1)
                           if self._t0 is not None and self._last_now is not None else None),
            "viewpoints_visited": len(self.planner.visited),
            "viewpoints_abandoned": len(self.planner.abandoned),
            "legs": list(self._legs),
            "change_counts": counts,
            "new_changes": changes,
            "investigations": outcomes,
        }


def _finite(dist: np.ndarray, meta: dict, xy: Pt) -> float:
    r = float(meta["resolution_m"])
    i = int(math.floor((xy[0] - float(meta["origin_x_m"])) / r))
    j = int(math.floor((xy[1] - float(meta["origin_y_m"])) / r))
    if 0 <= i < dist.shape[0] and 0 <= j < dist.shape[1] and math.isfinite(dist[i, j]):
        return float(dist[i, j])
    return 1e9
