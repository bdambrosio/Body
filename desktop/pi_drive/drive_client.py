"""Drive command/status client for the Tier-3 debug console.

Owns a Zenoh session separate from the chassis StubController (decoupled;
two sessions to the same router is fine). Publishes ``body/drive/goto``,
subscribes ``body/drive/status`` + ``body/odom``. Converts an operator's
body-frame click into an odom-frame goal using the latest odom pose, so
the goal stays fixed in the world as the robot moves (the same rule the
Pi uses — see docs/drive_tier3_spec.md). When the caller knows the instant
the body-frame point was computed for (Tier-2: the scan ts), it is sent as
``frame="body"`` + ``anchor_ts`` instead and Tier-3 anchors it from its own
odom ring buffer at that instant (time-consistent; X6).

Delivery (X5): goto/cancel go out with control QoS (BLOCK, high priority).
Every command is tracked until a ``body/drive/status`` carrying its cmd_id
arrives; if none does within ``ACK_TIMEOUT_S`` it is re-published verbatim
(same cmd_id), at most ``MAX_RESENDS`` times, after which ``ack_failed``
reports it. A ``REJECTED`` status (cmd_id at/below the Pi's watermark) — or
any status with a higher cmd_id — re-seeds the counter past it so the next
command is accepted.
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
from typing import Any, Dict, Optional, Tuple

from desktop.chassis.transport import control_put, open_session
from desktop.perception.depth_stop_logger import depth_fired

logger = logging.getLogger(__name__)


GOTO_TOPIC = "body/drive/goto"
DEPTH_MEMORY_TOPIC = "body/drive/depth_memory"


class DriveClient:
    ACK_TIMEOUT_S = 1.0
    MAX_RESENDS = 3

    def __init__(self, router: str, *, trace_path: Optional[str] = None):
        self._router = router
        self._session: Optional[Any] = None
        self._subs: list[Any] = []
        self._lock = threading.Lock()
        self._status: Optional[Dict[str, Any]] = None
        # Rising edge of a depth stop / would-block, latched off the stream
        # so a stop shorter than a UI redraw still gets logged (11-03).
        self._depth_on = False
        self._depth_event: Optional[Dict[str, Any]] = None
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
        # Highest cmd_id seen on body/drive/status (any state) and the Pi's
        # reported watermark (REJECTED). The next id is always above both, so
        # a second client (or a Pi that outran our time seed) can't leave our
        # gotos silently superseded (X5).
        self._max_seen_cmd_id = 0
        # Ack tracking for the newest command: (cmd_id, payload_json,
        # last_send_mono, resends). Only the newest is tracked — a newer
        # command supersedes the older on the Pi anyway.
        self._pending: Optional[Tuple[int, str, float, int]] = None
        self._acked: set = set()
        self._ack_failed: set = set()
        self._ack_stop = threading.Event()
        self._ack_thread: Optional[threading.Thread] = None
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
        self._ack_stop.clear()
        self._ack_thread = threading.Thread(
            target=self._ack_loop, name="drive-client-ack", daemon=True)
        self._ack_thread.start()
        return True, None

    def is_connected(self) -> bool:
        return self._session is not None

    def shutdown(self) -> None:
        self._ack_stop.set()
        if self._ack_thread is not None and self._ack_thread.is_alive():
            self._ack_thread.join(timeout=1.0)
        self._ack_thread = None
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
                self._note_status_locked(obj)
                on = depth_fired(obj) is not None
                if on and not self._depth_on and self._depth_event is None:
                    self._depth_event = obj
                self._depth_on = on
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

    def _note_status_locked(self, obj: Dict[str, Any]) -> None:
        """Ack bookkeeping + cmd_id re-seeding from one status (lock held)."""
        try:
            cid = int(obj.get("cmd_id") or 0)
        except (TypeError, ValueError):
            return
        state = obj.get("state")
        if state != "REJECTED":
            self._max_seen_cmd_id = max(self._max_seen_cmd_id, cid)
        wm = obj.get("watermark")
        if wm is not None:
            try:
                self._max_seen_cmd_id = max(self._max_seen_cmd_id, int(wm))
            except (TypeError, ValueError):
                pass
        p = self._pending
        if p is not None and (cid == p[0] or (cid > p[0] and state != "REJECTED")):
            # Matching status (incl. REJECTED: the Pi answered — the caller
            # handles the rejection), or a newer accepted command from another
            # client superseded ours. Either way stop resending.
            self._acked.add(p[0])
            self._pending = None

    def _next_cmd_id_locked(self) -> int:
        self._cmd_id = max(self._cmd_id, self._max_seen_cmd_id) + 1
        return self._cmd_id

    def _publish_tracked(self, cid: int, msg: Dict[str, Any]) -> None:
        payload = json.dumps(msg)
        with self._lock:
            self._pending = (cid, payload, time.monotonic(), 0)
            # Bound the bookkeeping sets; old ids are irrelevant.
            if len(self._acked) > 256:
                self._acked = {c for c in self._acked if c > cid - 128}
            if len(self._ack_failed) > 256:
                self._ack_failed = {c for c in self._ack_failed if c > cid - 128}
        assert self._session is not None
        control_put(self._session, GOTO_TOPIC, payload)

    def pump_resends(self, now_mono: Optional[float] = None) -> None:
        """Resend the pending command if unacknowledged for ACK_TIMEOUT_S;
        after MAX_RESENDS mark it ack-failed. Called by the ack thread
        (public for tests)."""
        now = time.monotonic() if now_mono is None else now_mono
        resend: Optional[str] = None
        with self._lock:
            p = self._pending
            if p is None or now - p[2] < self.ACK_TIMEOUT_S:
                return
            cid, payload, _t, n = p
            if n >= self.MAX_RESENDS:
                logger.warning("drive: cmd_id=%d never acknowledged after %d resends",
                               cid, n)
                self._ack_failed.add(cid)
                self._pending = None
                return
            self._pending = (cid, payload, now, n + 1)
            resend = payload
            logger.info("drive: no status for cmd_id=%d in %.1fs — resend %d/%d",
                        cid, self.ACK_TIMEOUT_S, n + 1, self.MAX_RESENDS)
        session = self._session
        if session is not None and resend is not None:
            try:
                control_put(session, GOTO_TOPIC, resend)
            except Exception:
                logger.exception("drive: resend failed")

    def _ack_loop(self) -> None:
        while not self._ack_stop.wait(0.1):
            try:
                self.pump_resends()
            except Exception:
                logger.exception("drive: ack loop error")

    def ack_failed(self, cmd_id: int) -> bool:
        """True once ``cmd_id`` exhausted its resends with no matching status."""
        with self._lock:
            return cmd_id in self._ack_failed

    def is_acked(self, cmd_id: int) -> bool:
        with self._lock:
            return cmd_id in self._acked

    def supports_path(self) -> bool:
        """True when the Pi's Tier-3 advertises ``caps: ["path"]`` (goto
        kind "path", nav redesign step 3) on its latest status."""
        with self._lock:
            st = self._status
            return bool(isinstance(st, dict) and "path" in (st.get("caps") or ()))

    def send_path(
        self, path_body: list, anchor_ts: float, *,
        cmd_id: Optional[int] = None, path_seq: int = 1, route_id: int = 1,
        end_is_goal: bool = False, ttl_s: Optional[float] = None,
        arrival_tol_m: Optional[float] = None, v_max: Optional[float] = None,
        final_heading_rad: Optional[float] = None,
    ) -> Optional[int]:
        """Publish a path goal (``kind: "path"``) in the robot body frame at
        ``anchor_ts`` (Pi odom ts the frame refers to). ``cmd_id`` None → a
        new goal (new id, ack-tracked + resent like a goto); otherwise a
        refresh of that goal (``path_seq`` must increase) — published once,
        the next refresh is its retry. Returns the cmd_id, or None if not
        connected."""
        if self._session is None or len(path_body) < 2:
            return None
        new = cmd_id is None
        if new:
            with self._lock:
                cmd_id = self._next_cmd_id_locked()
        msg: Dict[str, Any] = {
            "ts": time.time(), "cmd_id": int(cmd_id), "kind": "path", "frame": "body",
            "anchor_ts": float(anchor_ts), "path_seq": int(path_seq),
            "route_id": int(route_id), "end_is_goal": bool(end_is_goal),
            "path_xy": [[round(float(x), 3), round(float(y), 3)] for x, y in path_body],
        }
        if ttl_s is not None:
            msg["ttl_s"] = float(ttl_s)
        if arrival_tol_m is not None:
            msg["arrival_tol_m"] = float(arrival_tol_m)
        if v_max is not None:
            msg["v_max"] = float(v_max)
        if final_heading_rad is not None:
            msg["final_heading_rad"] = float(final_heading_rad)
        if new:
            self._publish_tracked(cmd_id, msg)
        else:
            control_put(self._session, GOTO_TOPIC, json.dumps(msg))
        return cmd_id

    def pop_depth_event(self) -> Optional[Dict[str, Any]]:
        """The first depth-stop / would-block status since the last call
        (its rising edge), or None. Clears the latch."""
        with self._lock:
            ev, self._depth_event = self._depth_event, None
            return dict(ev) if ev is not None else None

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
        omega_max: Optional[float] = None,
        anchor_ts: Optional[float] = None,
    ) -> Optional[int]:
        """Publish a body-frame goal. Returns the cmd_id, or None if no
        odom/connection yet.

        With ``anchor_ts`` (Pi wall-clock instant the body frame refers to —
        Tier-2 passes the scan ts) the goal goes out as ``frame="body"`` and
        Tier-3 anchors it with its own odom interpolated at that instant.
        Without it (operator click) it is converted here to an odom goal with
        the latest odom pose."""
        pose = self.odom_pose()
        if pose is None or self._session is None:
            return None
        with self._lock:
            cid = self._next_cmd_id_locked()
        if anchor_ts is not None:
            msg: Dict[str, Any] = {
                "ts": time.time(), "cmd_id": cid, "frame": "body",
                "anchor_ts": float(anchor_ts),
                "x_m": float(bx), "y_m": float(by), "kind": "goto",
            }
        else:
            c, s = math.cos(pose[2]), math.sin(pose[2])
            ox = pose[0] + bx * c - by * s
            oy = pose[1] + bx * s + by * c
            msg = {
                "ts": time.time(), "cmd_id": cid, "frame": "odom",
                "x_m": ox, "y_m": oy, "kind": "goto",
            }
        if final_heading_rad is not None:
            msg["final_heading_rad"] = final_heading_rad
        if arrival_tol_m is not None:
            msg["arrival_tol_m"] = arrival_tol_m
        if v_max is not None:
            msg["v_max"] = v_max
        if omega_max is not None:
            msg["omega_max"] = omega_max
        self._publish_tracked(cid, msg)
        return cid

    def reset_depth_memory(self, reason: str) -> bool:
        """Tell Tier-3 to forget its depth memory (idempotent). Sent twice,
        ~0.2 s apart, so one lost put on the WiFi link doesn't leave stale
        cells around a robot that was just carried or relocated."""
        if self._session is None:
            return False
        payload = json.dumps({"ts": time.time(), "action": "reset", "reason": reason})
        session = self._session
        control_put(session, DEPTH_MEMORY_TOPIC, payload)

        def _again() -> None:
            try:
                control_put(session, DEPTH_MEMORY_TOPIC, payload)
            except Exception:                   # session closed meanwhile
                pass
        t = threading.Timer(0.2, _again)
        t.daemon = True
        t.start()
        return True

    def cancel(self) -> None:
        if self._session is None:
            return
        with self._lock:
            cid = self._next_cmd_id_locked()
        # Tracked + resent like a goto: a lost cancel leaves Tier-3 chasing a
        # goal the desktop believes revoked (E1/X3).
        self._publish_tracked(
            cid, {"ts": time.time(), "cmd_id": cid, "kind": "cancel"})
