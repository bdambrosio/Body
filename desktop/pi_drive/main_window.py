"""Tier-3 drive debug console — main window.

Reuses the chassis StubController for the Zenoh connection, the always-on
heartbeat (the watchdog e-stops without it, so it's required for *any*
motion, including Pi-initiated drives), and live local_map ingestion.
Keeps live_command OFF so the desktop never publishes cmd_vel — the Pi's
Tier-3 owns cmd_vel while a goal is active (see docs/drive_tier3_spec.md).
"""
from __future__ import annotations

import logging
import math
import time
from typing import Optional, Tuple

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QDoubleSpinBox, QGroupBox, QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QVBoxLayout, QWidget,
)

from body.lib import drive_config, zenoh_helpers
from body.lib.scan_raster import rasterize_scan

from .drive_client import DriveClient
from .local_map_view import BodyLocalMapView

logger = logging.getLogger(__name__)


class PiDriveWindow(QMainWindow):
    def __init__(self, controller, drive: DriveClient, *, redraw_hz: float = 10.0):
        super().__init__()
        self.setWindowTitle("Body — Tier-3 Drive Console")
        self.controller = controller
        self.drive = drive

        self._view = BodyLocalMapView(self)
        self._view.set_click_callback(self._on_map_click)
        # Render exactly what Tier-3 sees: the live scan rasterized with the
        # same pure code AND the same config.json params as body.local_drive
        # (shared body.lib.drive_config builder → no divergence).
        self._raster = drive_config.scan_raster_config(zenoh_helpers.load_body_config())

        self._conn_lbl = QLabel("conn: —")
        self._state_lbl = QLabel("drive: —")
        self._goal_lbl = QLabel("goal: —")
        # Bring-up telemetry: is the robot actually responding to cmd_vel?
        self._estop_lbl = QLabel("estop: —")
        self._odom_lbl = QLabel("odom: —")
        self._motor_lbl = QLabel("motor: —")
        # For the odom "moving?" indicator (position delta between redraws).
        self._last_odom_xy: Optional[Tuple[float, float]] = None
        self._last_odom_t: Optional[float] = None

        self._tol_spin = QDoubleSpinBox()
        self._tol_spin.setRange(0.03, 1.0)
        self._tol_spin.setSingleStep(0.01)
        self._tol_spin.setValue(0.12)
        self._tol_spin.setSuffix(" m tol")

        self._vmax_spin = QDoubleSpinBox()
        self._vmax_spin.setRange(0.02, 0.5)
        self._vmax_spin.setSingleStep(0.01)
        self._vmax_spin.setValue(0.18)
        self._vmax_spin.setSuffix(" m/s")

        connect_btn = QPushButton("Connect")
        connect_btn.clicked.connect(self._connect)
        stop_btn = QPushButton("STOP / Cancel")
        stop_btn.clicked.connect(self._stop)

        panel = QGroupBox("drive")
        pl = QVBoxLayout(panel)
        pl.addWidget(self._conn_lbl)
        pl.addWidget(self._state_lbl)
        pl.addWidget(self._goal_lbl)
        row = QHBoxLayout()
        row.addWidget(self._tol_spin)
        row.addWidget(self._vmax_spin)
        pl.addLayout(row)
        pl.addWidget(connect_btn)
        pl.addWidget(stop_btn)
        pl.addWidget(QLabel("click the map to set a goal"))
        pl.addSpacing(8)
        pl.addWidget(self._estop_lbl)
        pl.addWidget(self._odom_lbl)
        pl.addWidget(self._motor_lbl)
        pl.addStretch(1)

        central = QWidget()
        lay = QHBoxLayout(central)
        lay.addWidget(self._view, stretch=1)
        lay.addWidget(panel)
        self.setCentralWidget(central)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(int(1000 / max(1.0, redraw_hz)))

        self._connect()

    # ── Connection ───────────────────────────────────────────────────

    def _connect(self) -> None:
        ok1, err1 = self.controller.connect()
        ok2, err2 = self.drive.connect()
        if ok1 and ok2:
            self._conn_lbl.setText("conn: connected")
        else:
            self._conn_lbl.setText(f"conn: FAILED {err1 or ''} {err2 or ''}".strip())

    # ── Map click → goal ─────────────────────────────────────────────

    def _on_map_click(self, bx: float, by: float) -> None:
        cid = self.drive.send_goto_from_body(
            bx, by,
            arrival_tol_m=self._tol_spin.value(),
            v_max=self._vmax_spin.value(),
        )
        if cid is None:
            self._goal_lbl.setText("goal: (no odom yet — cannot place)")
        else:
            self._goal_lbl.setText(f"goal: sent #{cid} body=({bx:+.2f}, {by:+.2f})")

    def _stop(self) -> None:
        self.drive.cancel()
        try:
            self.controller.stop_all()
        except Exception:
            pass

    # ── Redraw ───────────────────────────────────────────────────────

    def _tick(self) -> None:
        scan = self.drive.latest_scan()
        drive = None
        meta = None
        if scan is not None and scan.get("ranges"):
            drive, meta = rasterize_scan(
                scan.get("ranges"), float(scan.get("angle_min", 0.0)),
                float(scan.get("angle_increment", 0.0)), self._raster,
            )
        status = self.drive.latest_status()
        goal_body: Optional[Tuple[float, float]] = None
        state_text = ""
        if status is not None:
            gb = status.get("goal_body_xy")
            if isinstance(gb, list) and len(gb) == 2:
                goal_body = (float(gb[0]), float(gb[1]))
            state = status.get("state", "—")
            reason = status.get("blocked_reason")
            mode = status.get("mode")
            state_text = (
                f"{state}  d={float(status.get('dist_remaining_m', 0.0)):.2f}m "
                f"v={float(status.get('v_mps', 0.0)):+.2f} "
                f"ω={float(status.get('omega_radps', 0.0)):+.2f}"
                + (f"  ‹{mode}›" if mode else "")
                + (f"  [{reason}]" if reason else "")
            )
            self._state_lbl.setText(f"drive: {state_text}")
        self._view.update_data(drive, meta, goal_body, state_text)
        self._update_telemetry()

    def _update_telemetry(self) -> None:
        """e-stop / odom / motor readout — answers 'is the robot actually
        responding to cmd_vel?' during bring-up."""
        st = self.controller.state
        with st.lock:
            status = st.status
            motor = st.motor_state
            odom = st.odom

        estop = None
        if isinstance(motor, dict) and "e_stop_active" in motor:
            estop = bool(motor["e_stop_active"])
        elif isinstance(status, dict) and "e_stop_active" in status:
            estop = bool(status["e_stop_active"])
        timeout = bool(motor.get("cmd_timeout_active")) if isinstance(motor, dict) else False
        stall = bool(motor.get("stall_detected")) if isinstance(motor, dict) else False
        flags = []
        if estop:
            flags.append("E-STOP")
        if timeout:
            flags.append("cmd_timeout")
        if stall:
            flags.append("stall")
        self._estop_lbl.setText(
            "estop: " + (" ".join(flags) if flags else ("clear" if estop is not None else "—"))
        )

        if isinstance(odom, dict):
            try:
                x, y, th = float(odom["x"]), float(odom["y"]), float(odom["theta"])
            except (KeyError, TypeError, ValueError):
                x = y = th = float("nan")
            now = time.monotonic()
            moving = "—"
            if self._last_odom_xy is not None and self._last_odom_t is not None:
                d = math.hypot(x - self._last_odom_xy[0], y - self._last_odom_xy[1])
                dt = max(1e-3, now - self._last_odom_t)
                moving = f"MOVING {d / dt:.2f} m/s" if d > 0.005 else "still"
            self._last_odom_xy = (x, y)
            self._last_odom_t = now
            src = str(odom.get("source", "?"))
            self._odom_lbl.setText(
                f"odom: x={x:+.2f} y={y:+.2f} θ={math.degrees(th):+.0f}°  [{moving}]  ({src})"
            )
        else:
            self._odom_lbl.setText("odom: — (no body/odom)")

        if isinstance(motor, dict):
            self._motor_lbl.setText(
                f"motor: L={motor.get('left_pwm', 0):+.2f}/{motor.get('left_dir', '?')} "
                f"R={motor.get('right_pwm', 0):+.2f}/{motor.get('right_dir', '?')}"
            )
        else:
            self._motor_lbl.setText("motor: — (no body/motor_state)")

    def closeEvent(self, event) -> None:
        try:
            self.drive.shutdown()
        except Exception:
            pass
        try:
            self.controller.shutdown()
        except Exception:
            pass
        super().closeEvent(event)
