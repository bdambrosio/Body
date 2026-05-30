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

from body.lib.scan_raster import ScanRasterConfig, rasterize_scan

from .drive_client import DriveClient
from .local_map_view import BodyLocalMapView
from .tier2_session import Tier2Session, Tier2SessionConfig

logger = logging.getLogger(__name__)

_LEVEL_COLOR = {"info": QColor(180, 200, 220), "warn": QColor(230, 200, 90),
                "error": QColor(235, 110, 110)}


class Tier2Window(QMainWindow):
    def __init__(self, controller, drive: DriveClient, *,
                 trace_path: Optional[str] = None, redraw_hz: float = 10.0):
        super().__init__()
        self.setWindowTitle("Body — Tier-2 Debug Console")
        self.controller = controller
        self.drive = drive
        self._raster = ScanRasterConfig()
        self.session = Tier2Session(drive, Tier2SessionConfig())

        self._view = BodyLocalMapView(self)
        self._view.set_click_callback(self._on_map_click)

        self._conn_lbl = QLabel("conn: —")
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

        connect_btn = QPushButton("Connect"); connect_btn.clicked.connect(self._connect)
        clear_btn = QPushButton("Clear target"); clear_btn.clicked.connect(self._clear)
        stop_btn = QPushButton("STOP / Cancel"); stop_btn.clicked.connect(self._stop)

        self._events = QListWidget()

        panel = QGroupBox("tier-2 debug")
        pl = QVBoxLayout(panel)
        pl.addWidget(self._conn_lbl)
        pl.addWidget(self._t2_lbl)
        pl.addWidget(self._t3_lbl)
        pl.addWidget(self._estop_lbl)
        row = QHBoxLayout(); row.addWidget(self._tol_spin); row.addWidget(self._vmax_spin)
        pl.addLayout(row)
        pl.addWidget(self._drive_chk)
        brow = QHBoxLayout(); brow.addWidget(connect_btn); brow.addWidget(clear_btn)
        pl.addLayout(brow)
        pl.addWidget(stop_btn)
        pl.addWidget(QLabel("click the map to set a target"))
        pl.addWidget(QLabel("events:"))
        pl.addWidget(self._events, stretch=1)

        central = QWidget()
        lay = QHBoxLayout(central)
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

    def _stop(self) -> None:
        self._drive_chk.setChecked(False)
        self.session.clear_target()
        try:
            self.controller.stop_all()
        except Exception:
            pass

    def _on_map_click(self, bx: float, by: float) -> None:
        pose = self.drive.odom_pose()
        if pose is None:
            self._push_event("warn", "no_odom", "click ignored — no odom yet")
            return
        self.session.set_tunables(
            subgoal_arrival_tol_m=self._tol_spin.value(),
            sub_v_max=self._vmax_spin.value())
        self.session.set_target_from_body(bx, by, pose)

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
        odom_pose = self.drive.odom_pose()
        status = self.drive.latest_status()
        e_stop, hb_ok = self._chassis_flags()

        tick = self.session.tick(
            now, odom_pose=odom_pose, grid=grid, meta=meta, scan_age_s=scan_age,
            tier3_status=status, e_stop_active=e_stop, heartbeat_ok=hb_ok)

        self._render(tick, grid, meta, status, e_stop)
        self._trace_write(tick)

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

    def _render(self, tick, grid, meta, status, e_stop) -> None:
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
        for fn in (self.drive.shutdown, self.controller.shutdown):
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
