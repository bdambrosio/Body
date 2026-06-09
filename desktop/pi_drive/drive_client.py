"""Drive command/status client for the Tier-3 debug console.

Owns a Zenoh session separate from the chassis StubController (decoupled;
two sessions to the same router is fine). Publishes ``body/drive/goto``,
subscribes ``body/drive/status`` + ``body/odom``. Converts an operator's
body-frame click into an odom-frame goal using the latest odom pose, so
the goal stays fixed in the world as the robot moves (the same rule the
Pi uses — see docs/drive_tier3_spec.md).
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
from typing import Any, Dict, Optional, Tuple

from desktop.chassis.transport import open_session

logger = logging.getLogger(__name__)


class DriveClient:
    def __init__(self, router: str, *, trace_path: Optional[str] = None):
        self._router = router
        self._session: Optional[Any] = None
        self._subs: list[Any] = []
        self._lock = threading.Lock()
        self._status: Optional[Dict[str, Any]] = None
        self._odom: Optional[Dict[str, Any]] = None
        self._scan: Optional[Dict[str, Any]] = None
        # Seed from wall-clock so cmd_ids strictly exceed any previous
        # session's: the Pi's Tier-3 keeps its last cmd_id across our restarts
        # and rejects any goto with a lower id as superseded. A per-launch
        # counter from 0 would be silently dropped until it climbed back past
        # the Pi's remembered value. Deciseconds, not seconds: a re-pick-heavy
        # session can issue more than one command per second on average, ending
        # with cmd_id > time(); a prompt restart would then seed LOWER and every
        # goto would be silently rejected. Outrunning 10 ids/s is implausible.
        self._cmd_id = int(time.time() * 10.0)
        # Optional post-hoc trace: one JSON object per received status,
        # stamped with the desktop wall-clock arrival time. Mirrors the
        # nav JSONL tracing pattern so a leg can be reviewed offline.
        self._trace = None
        if trace_path:
            try:
                self._trace = open(trace_path, "a", encoding="utf-8")
                logger.info("drive trace → %s", trace_path)
            except OSError:
                logger.exception("could not open trace file %s", trace_path)

    # ── Lifecycle ────────────────────────────────────────────────────

    def connect(self) -> Tuple[bool, Optional[str]]:
        if self._session is not None:
            return True, None
        try:
            self._session = open_session(self._router)
            self._subs.append(
                self._session.declare_subscriber("body/drive/status", self._on_status)
            )
            self._subs.append(
                self._session.declare_subscriber("body/odom", self._on_odom)
            )
            self._subs.append(
                self._session.declare_subscriber("body/lidar/scan", self._on_scan)
            )
        except Exception as e:
            logger.exception("drive client connect failed")
            return False, f"{type(e).__name__}: {e}"
        return True, None

    def shutdown(self) -> None:
        for s in self._subs:
            try:
                s.undeclare()
            except Exception:
                pass
        self._subs.clear()
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None
        if self._trace is not None:
            try:
                self._trace.close()
            except OSError:
                pass
            self._trace = None

    # ── Subscriptions ────────────────────────────────────────────────

    def _on_status(self, sample: Any) -> None:
        try:
            obj = json.loads(sample.payload.to_string())
        except Exception:
            return
        if isinstance(obj, dict):
            with self._lock:
                self._status = obj
                if self._trace is not None:
                    try:
                        self._trace.write(
                            json.dumps({"recv_ts": time.time(), **obj}) + "\n"
                        )
                        self._trace.flush()
                    except (OSError, TypeError):
                        pass

    def _on_odom(self, sample: Any) -> None:
        try:
            obj = json.loads(sample.payload.to_string())
        except Exception:
            return
        if isinstance(obj, dict):
            with self._lock:
                self._odom = obj

    def _on_scan(self, sample: Any) -> None:
        try:
            obj = json.loads(sample.payload.to_string())
        except Exception:
            return
        if isinstance(obj, dict):
            with self._lock:
                self._scan = obj

    def latest_status(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return dict(self._status) if self._status is not None else None

    def latest_scan(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._scan

    def odom_pose(self) -> Optional[Tuple[float, float, float]]:
        with self._lock:
            o = self._odom
            if o is None:
                return None
            try:
                return (float(o["x"]), float(o["y"]), float(o["theta"]))
            except (KeyError, TypeError, ValueError):
                return None

    # ── Commands ─────────────────────────────────────────────────────

    def send_goto_from_body(
        self,
        bx: float,
        by: float,
        *,
        final_heading_rad: Optional[float] = None,
        arrival_tol_m: Optional[float] = None,
        v_max: Optional[float] = None,
    ) -> Optional[int]:
        """Convert a body-frame click to an odom goal and publish it.
        Returns the cmd_id, or None if no odom/connection yet."""
        pose = self.odom_pose()
        if pose is None or self._session is None:
            return None
        c, s = math.cos(pose[2]), math.sin(pose[2])
        ox = pose[0] + bx * c - by * s
        oy = pose[1] + bx * s + by * c
        with self._lock:
            self._cmd_id += 1
            cid = self._cmd_id
        msg: Dict[str, Any] = {
            "ts": time.time(), "cmd_id": cid, "frame": "odom",
            "x_m": ox, "y_m": oy, "kind": "goto",
        }
        if final_heading_rad is not None:
            msg["final_heading_rad"] = final_heading_rad
        if arrival_tol_m is not None:
            msg["arrival_tol_m"] = arrival_tol_m
        if v_max is not None:
            msg["v_max"] = v_max
        self._session.put("body/drive/goto", json.dumps(msg))
        return cid

    def cancel(self) -> None:
        if self._session is None:
            return
        with self._lock:
            self._cmd_id += 1
            cid = self._cmd_id
        self._session.put(
            "body/drive/goto",
            json.dumps({"ts": time.time(), "cmd_id": cid, "kind": "cancel"}),
        )
