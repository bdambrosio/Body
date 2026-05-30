"""Tier-2 debug console — main window.

Exposes the Tier-2 step in isolation: click a target on the body-frame view,
watch Tier-2 pick a sub-goal (bearing ray + annotated dot), see the exact
goto sent to Tier-3 and the status it returns, and an Events panel that
surfaces anomalies (swept_block, cmd_id_mismatch, e-stop, staleness). All the
decision/event logic lives in the pure `Tier2Session`; this window only
rasterizes the scan, renders, and writes the JSONL trace.

Reuses the chassis StubController (heartbeat — required for any motion) and the
DriveClient (goto/status/odom/scan). Keeps live_command OFF: Tier-3 owns
cmd_vel. Run with QT_QPA_PLATFORM=xcb on Wayland.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional, Tuple

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QGroupBox, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMainWindow, QPushButton, QVBoxLayout, QWidget,
)

from body.lib.local_drive_core import body_to_odom
from body.lib.scan_raster import ScanRasterConfig, rasterize_scan
from desktop.world_map.map_views import SharedMapView, WorldDriveableView

from .drive_client import DriveClient
from .local_map_view import BodyLocalMapView
from .tier2_session import Tier2Session, Tier2SessionConfig

logger = logging.getLogger(__name__)

_LEVEL_COLOR = {"info": QColor(180, 200, 220), "warn": QColor(230, 200, 90),
                "error": QColor(235, 110, 110)}


class Tier2Window(QMainWindow):
    def __init__(self, controller, drive: DriveClient, *,
                 localizer=None,
                 trace_path: Optional[str] = None, redraw_hz: float = 10.0):
        super().__init__()
        self.setWindowTitle("Body — Tier-2 Debug Console")
        self.controller = controller
        self.drive = drive
        self.localizer = localizer       # PF; None → body-only mode
        self._raster = ScanRasterConfig()
        self.session = Tier2Session(drive, Tier2SessionConfig())

        self._view = BodyLocalMapView(self)
        self._view.set_click_callback(self._on_map_click)

        # World map (map mode only): right-click = world Tier-2 target,
        # left-click = relocate-at (assert true position).
        self._world_view = None
        if self.localizer is not None:
            self._shared = SharedMapView()
            self._world_view = WorldDriveableView(
                stale_s=2.0, shared=self._shared)
            self._shared.set_goal_callback(self._on_world_target)
            self._shared.set_locate_callback(self._on_world_locate)

        self._conn_lbl = QLabel("conn: —")
        self._pose_lbl = QLabel("pose: —")
        self._t2_lbl = QLabel("tier2: —")
        self._t2_lbl.setWordWrap(True)
        self._t3_lbl = QLabel("tier3: —")
        self._t3_lbl.setWordWrap(True)
        self._estop_lbl = QLabel("estop: —")

        self._tol_spin = QDoubleSpinBox()
        self._tol_spin.setRange(0.03, 1.0); self._tol_spin.setSingleStep(0.01)
        self._tol_spin.setValue(0.15); self._tol_spin.setSuffix(" m tol")
        self._vmax_spin = QDoubleSpinBox()
        self._vmax_spin.setRange(0.02, 0.5); self._vmax_spin.setSingleStep(0.01)
        self._vmax_spin.setValue(0.18); self._vmax_spin.setSuffix(" m/s")

        self._drive_chk = QCheckBox("Drive (send gotos)")
        self._drive_chk.toggled.connect(self.session.set_drive)

        # Armed "Set location": while checked, a LEFT-click on the world map is
        # a relocate_at (assert true position); unchecked, left-click pans.
        # map_views only routes left-clicks to _on_world_locate when locate_mode
        # is on, so this toggle is required for left-click to do anything.
        self._locate_chk = QCheckBox("Set location (left-click)")
        if self.localizer is not None:
            self._locate_chk.toggled.connect(self._shared.set_locate_mode)

        connect_btn = QPushButton("Connect"); connect_btn.clicked.connect(self._connect)
        clear_btn = QPushButton("Clear target"); clear_btn.clicked.connect(self._clear)
        stop_btn = QPushButton("STOP / Cancel"); stop_btn.clicked.connect(self._stop)

        self._events = QListWidget()

        panel = QGroupBox("tier-2 debug")
        pl = QVBoxLayout(panel)
        pl.addWidget(self._conn_lbl)
        if self.localizer is not None:
            pl.addWidget(self._pose_lbl)
        pl.addWidget(self._t2_lbl)
        pl.addWidget(self._t3_lbl)
        pl.addWidget(self._estop_lbl)
        row = QHBoxLayout(); row.addWidget(self._tol_spin); row.addWidget(self._vmax_spin)
        pl.addLayout(row)
        pl.addWidget(self._drive_chk)
        if self.localizer is not None:
            pl.addWidget(self._locate_chk)
        brow = QHBoxLayout(); brow.addWidget(connect_btn); brow.addWidget(clear_btn)
        pl.addLayout(brow)
        if self.localizer is not None:
            reloc_btn = QPushButton("Relocate"); reloc_btn.clicked.connect(self._on_relocate)
            pl.addWidget(reloc_btn)
        pl.addWidget(stop_btn)
        pl.addWidget(QLabel(
            "right-click world map = target; left-click = set location"
            if self.localizer is not None else "click the map to set a target"))
        pl.addWidget(QLabel("events:"))
        pl.addWidget(self._events, stretch=1)

        central = QWidget()
        lay = QHBoxLayout(central)
        if self._world_view is not None:
            lay.addWidget(self._world_view, stretch=1)
        lay.addWidget(self._view, stretch=1)
        lay.addWidget(panel)
        self.setCentralWidget(central)

        # JSONL trace (mirror DriveClient's simple writer).
        self._trace = None
        if trace_path:
            try:
                self._trace = open(trace_path, "a", encoding="utf-8")
                self._trace.write(json.dumps({
                    "kind": "header", "ts": time.time(),
                    "raster": vars(self._raster),
                }) + "\n")
                self._trace.flush()
                logger.info("tier2 trace → %s", trace_path)
            except OSError:
                logger.exception("could not open trace %s", trace_path)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(int(1000 / max(1.0, redraw_hz)))
        self._connect()

    # ── Connection / controls ────────────────────────────────────────

    def _connect(self) -> None:
        ok1, err1 = self.controller.connect()
        ok2, err2 = self.drive.connect()
        self._conn_lbl.setText(
            "conn: connected" if (ok1 and ok2)
            else f"conn: FAILED {err1 or ''} {err2 or ''}".strip())

    def _clear(self) -> None:
        self.session.clear_target()
        self._view.set_overlay(None, None, None, 0.0)
        if self._world_view is not None:
            self._shared.set_goal(None)
            self._shared.set_lookahead(None)

    def _stop(self) -> None:
        self._drive_chk.setChecked(False)
        self.session.clear_target()
        try:
            self.controller.stop_all()
        except Exception:
            pass

    def _on_map_click(self, bx: float, by: float) -> None:
        # Body-view click sets the target only in body-only mode; in map mode
        # the world map (right-click) is the sole target input.
        if self.localizer is not None:
            return
        pose = self.drive.odom_pose()
        if pose is None:
            self._push_event("warn", "no_odom", "click ignored — no odom yet")
            return
        self._apply_tunables()
        self.session.set_target_from_body(bx, by, pose)

    def _on_world_target(self, wx: float, wy: float) -> None:
        """Right-click on the world map = the true Tier-2 world target."""
        self._apply_tunables()
        self.session.set_target_point(wx, wy)
        self._shared.set_goal((wx, wy))
        self._push_event("info", "target", f"world ({wx:+.2f}, {wy:+.2f})")

    def _on_world_locate(self, wx: float, wy: float) -> None:
        """Left-click (while 'Set location' armed) = assert true position; PF
        recovers heading by scan-match. One-shot: disarm after the click."""
        res = self.localizer.request_relocate_at(wx, wy)
        ok = bool(res.get("success"))
        self._push_event("info" if ok else "warn", "relocate_at",
                         f"({wx:+.2f},{wy:+.2f}) {'ok' if ok else res.get('reason','failed')}")
        self._locate_chk.setChecked(False)

    def _on_relocate(self) -> None:
        res = self.localizer.request_relocate()
        ok = bool(res.get("success"))
        self._push_event("info" if ok else "warn", "relocate",
                         "ok" if ok else str(res.get("reason", "failed")))

    def _apply_tunables(self) -> None:
        self.session.set_tunables(
            subgoal_arrival_tol_m=self._tol_spin.value(),
            sub_v_max=self._vmax_spin.value())

    # ── Tick ─────────────────────────────────────────────────────────

    def _tick(self) -> None:
        now = time.time()
        scan = self.drive.latest_scan()
        grid = meta = None
        scan_age: Optional[float] = None
        if scan is not None:
            ts = float(scan.get("ts", 0.0))
            scan_age = (now - ts) if ts else None
            if scan.get("ranges"):
                grid, meta = rasterize_scan(
                    scan.get("ranges"), float(scan.get("angle_min", 0.0)),
                    float(scan.get("angle_increment", 0.0)), self._raster)
        status = self.drive.latest_status()
        e_stop, hb_ok = self._chassis_flags()

        # Pose frame must match the target frame: PF world pose in map mode
        # (target is world), odom pose in body-only mode (target is odom).
        if self.localizer is not None:
            pose = self._update_world_view(now)
        else:
            pose = self.drive.odom_pose()

        tick = self.session.tick(
            now, pose=pose, grid=grid, meta=meta, scan_age_s=scan_age,
            tier3_status=status, e_stop_active=e_stop, heartbeat_ok=hb_ok)

        self._render(tick, grid, meta, status, e_stop, pose)
        self._trace_write(tick)

    def _update_world_view(self, now: float):
        """Refresh the world map from the PF; return the world pose (or None)."""
        lp = self.localizer.pose_source.latest_pose()
        pose = lp[0] if lp is not None else None
        snap = self.localizer.snapshot_for_ui()
        if snap is not None and self._world_view is not None:
            self._world_view.update_map(
                snap["driveable"], snap["meta"], now, pose=pose,
                pose_history=self.localizer.pose_trail(),
                bounds_ij=snap.get("bounds_ij"))
        if pose is not None:
            self._pose_lbl.setText(
                f"pose: x={pose[0]:+.2f} y={pose[1]:+.2f} θ={pose[2]*57.3:+.0f}°")
        else:
            self._pose_lbl.setText("pose: — (not localized — Relocate)")
        return pose

    def _chassis_flags(self) -> Tuple[bool, bool]:
        st = self.controller.state
        with st.lock:
            status = st.status
            motor = st.motor_state
            connected = bool(getattr(st, "connected", True))
        e_stop = False
        if isinstance(motor, dict) and "e_stop_active" in motor:
            e_stop = bool(motor["e_stop_active"])
        elif isinstance(status, dict) and "e_stop_active" in status:
            e_stop = bool(status["e_stop_active"])
        return e_stop, connected

    def _render(self, tick, grid, meta, status, e_stop, pose) -> None:
        # Tier-3's serviced goal (blue) + the Tier-2 overlay (target/ray/sub-goal).
        goal_body = None
        if isinstance(status, dict):
            gb = status.get("goal_body_xy")
            if isinstance(gb, list) and len(gb) == 2:
                goal_body = (float(gb[0]), float(gb[1]))
        self._view.update_data(grid, meta, goal_body, "")

        d = tick.decision
        self._view.set_overlay(
            tick.target_body, d.body_xy if d else None,
            d.bearing_rad if d else None, d.free_dist_m if d else 0.0)

        # World map: project the body sub-goal back to world (display only).
        if self._world_view is not None:
            sub_w = (body_to_odom(d.body_xy, pose)
                     if (d and d.body_xy and pose is not None) else None)
            self._shared.set_lookahead(sub_w)

        if not tick.has_target:
            self._t2_lbl.setText("tier2: (click a target)")
        elif d is None:
            self._t2_lbl.setText(f"tier2: target d={tick.target_dist_m:.2f}m — no decision")
        else:
            tags = ("ok" if d.ok else d.reason)
            extra = " capped" if d.capped_at_target else (" backoff" if d.backoff_applied else "")
            sub = f"({d.body_xy[0]:+.2f},{d.body_xy[1]:+.2f})" if d.body_xy else "—"
            self._t2_lbl.setText(
                f"tier2: {tags}{extra}  bearing={d.bearing_rad*57.3:+.0f}° "
                f"off={d.bearing_offset_rad*57.3:+.0f}°  "
                f"free={d.free_dist_m:.2f}/{d.max_dist_m:.2f}m  sub={sub}")

        if isinstance(status, dict):
            self._t3_lbl.setText(
                f"tier3: {status.get('state','—')} ‹{status.get('mode','')}› "
                f"cmd={status.get('cmd_id')} d={float(status.get('dist_remaining_m',0)):.2f} "
                f"v={float(status.get('v_mps',0)):+.2f} ω={float(status.get('omega_radps',0)):+.2f}"
                + (f" [{status['blocked_reason']}]" if status.get("blocked_reason") else ""))
        self._estop_lbl.setText("estop: " + ("E-STOP" if e_stop else "clear"))

        for ev in tick.events:
            self._push_event(ev.level, ev.code, ev.detail)

    def _push_event(self, level: str, code: str, detail: str) -> None:
        ts = time.strftime("%H:%M:%S")
        item = QListWidgetItem(f"{ts}  {code}  {detail}".rstrip())
        item.setForeground(_LEVEL_COLOR.get(level, QColor(200, 200, 200)))
        self._events.addItem(item)
        while self._events.count() > 200:
            self._events.takeItem(0)
        self._events.scrollToBottom()

    def _trace_write(self, tick) -> None:
        if self._trace is None:
            return
        try:
            self._trace.write(json.dumps(tick.as_dict()) + "\n")
            self._trace.flush()
        except (OSError, TypeError):
            pass

    def closeEvent(self, event) -> None:
        fns = [self.drive.shutdown, self.controller.shutdown]
        if self.localizer is not None:
            fns.append(self.localizer.shutdown)
        for fn in fns:
            try:
                fn()
            except Exception:
                pass
        if self._trace is not None:
            try:
                self._trace.close()
            except OSError:
                pass
        super().closeEvent(event)
