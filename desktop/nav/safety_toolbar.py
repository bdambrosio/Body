"""Persistent safety-strip toolbar for the nav shell.

Hosts the controls an operator must always have one click away:
connection, ALL-STOP, Live-command toggle, and diagnostic pills
(connection, heartbeat, e-stop). Teleop sliders and camera feeds live
in toggleable docks so the map can dominate screen real estate during
mission work.

Owns both chassis + fuser connection management because nav always
runs them as a pair.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QToolBar, QWidget,
)

from desktop.chassis.controller import StubController
from desktop.world_map.controller import FuserController

logger = logging.getLogger(__name__)


class _Pill(QLabel):
    """Small colored chip — reused pattern from chassis.ui_qt."""

    def __init__(self, text: str, parent: Optional[QWidget] = None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFont(QFont("Monospace", 8))
        self.setMinimumWidth(60)
        self.set_state(False)

    def set_state(self, ok: bool) -> None:
        bg = "#2c7a2c" if ok else "#5a2a2a"
        fg = "white" if ok else "#bbb"
        self.setStyleSheet(
            f"background:{bg};color:{fg};"
            f"padding:1px 6px;border-radius:4px;"
        )


class SafetyToolbar(QToolBar):
    """Always-visible top strip for safety + connection controls."""

    # Heartbeat staleness threshold — 2× the default 5 Hz period.
    # If the desktop publisher hasn't bumped heartbeat_seq for this
    # long, treat the pill as stale/red.
    HB_FRESH_S = 0.4

    def __init__(
        self,
        chassis: StubController,
        fuser: FuserController,
        *,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__("Safety", parent)
        self.chassis = chassis
        self.fuser = fuser
        self._last_hb_seq: int = -1
        self._last_hb_change_wall: float = 0.0

        self.setMovable(False)
        self.setFloatable(False)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)

        self._build()
        self._wire()

    # ── Build ────────────────────────────────────────────────────────

    def _build(self) -> None:
        host = QWidget(self)
        row = QHBoxLayout(host)
        row.setContentsMargins(6, 4, 6, 4)
        row.setSpacing(8)

        row.addWidget(QLabel("Router:"))
        self._router_edit = QLineEdit(self.chassis.config.router)
        self._router_edit.setMinimumWidth(220)
        row.addWidget(self._router_edit)

        self._connect_btn = QPushButton("Connect")
        row.addWidget(self._connect_btn)

        row.addSpacing(16)

        self._pill_conn = _Pill("conn")
        self._pill_hb = _Pill("hb")
        self._pill_estop = _Pill("no-stop")
        for p in (self._pill_conn, self._pill_hb, self._pill_estop):
            row.addWidget(p)

        row.addStretch(1)

        self._live_box = QCheckBox("Live cmd")
        self._live_box.setEnabled(False)
        self._live_box.setToolTip(
            "Arms the chassis publisher to emit cmd_vel/cmd_direct. "
            "Values come from the teleop dock or nav mission logic."
        )
        row.addWidget(self._live_box)

        self._stop_btn = QPushButton("ALL STOP")
        self._stop_btn.setStyleSheet(
            "QPushButton{background:#aa2222;color:white;"
            "font-weight:bold;padding:4px 14px;}"
        )
        self._stop_btn.setToolTip(
            "Zero both cmd_vel and cmd_direct, drop Live command, "
            "supersede any stored command on the Pi."
        )
        row.addWidget(self._stop_btn)

        self.addWidget(host)

    def _wire(self) -> None:
        self._connect_btn.clicked.connect(self._on_connect_clicked)
        self._live_box.toggled.connect(self._on_live_toggled)
        self._stop_btn.clicked.connect(self._on_stop_clicked)

    # ── Slots ────────────────────────────────────────────────────────

    def _on_connect_clicked(self) -> None:
        if self._both_connected():
            self._disconnect_both()
            return

        endpoint = self._router_edit.text().strip() or self.chassis.config.router
        self.chassis.config.router = endpoint
        self.fuser.config.router = endpoint

        QApplication.processEvents()
        errors = []
        for name, ctrl in (("fuser", self.fuser), ("chassis", self.chassis)):
            ok, err = ctrl.connect()
            if not ok:
                errors.append(f"{name}: {err}")
        if errors:
            logger.warning("connect: %s", "; ".join(errors))

    def _disconnect_both(self) -> None:
        # chassis first — it supersedes commands on the way out.
        try:
            self.chassis.disconnect()
        except Exception:
            logger.exception("chassis disconnect raised")
        try:
            self.fuser.disconnect()
        except Exception:
            logger.exception("fuser disconnect raised")
        # stop_all is inside chassis.disconnect; reflect in UI now so
        # the Live checkbox doesn't linger checked for a tick.
        blk = self._live_box.blockSignals(True)
        self._live_box.setChecked(False)
        self._live_box.blockSignals(blk)

    def _on_live_toggled(self, on: bool) -> None:
        self.chassis.set_live_command(bool(on))

    def _on_stop_clicked(self) -> None:
        try:
            self.chassis.stop_all()
        except Exception:
            logger.exception("ALL-STOP raised")
        # stop_all drops live_command on the chassis state; mirror.
        blk = self._live_box.blockSignals(True)
        self._live_box.setChecked(False)
        self._live_box.blockSignals(blk)

    # ── Helpers ──────────────────────────────────────────────────────

    def _fuser_connected(self) -> bool:
        # FuserController exposes no public flag; session presence is
        # the best proxy without adding API surface.
        return getattr(self.fuser, "_session", None) is not None

    def _both_connected(self) -> bool:
        with self.chassis.state.lock:
            chassis_connected = self.chassis.state.connected
        return chassis_connected and self._fuser_connected()

    # ── Tick: called from main window redraw timer ──────────────────

    def refresh(self) -> None:
        s = self.chassis.state
        with s.lock:
            chassis_connected = s.connected
            live = s.live_command
            status = s.status
            motor = s.motor_state
            hb_seq = s.heartbeat_seq
        fuser_connected = self._fuser_connected()
        both_connected = chassis_connected and fuser_connected

        self._connect_btn.setText("Disconnect" if both_connected else "Connect")
        self._router_edit.setEnabled(not both_connected)
        self._live_box.setEnabled(chassis_connected)

        # Sync Live checkbox with authoritative state without firing.
        if self._live_box.isChecked() != live:
            blk = self._live_box.blockSignals(True)
            self._live_box.setChecked(live)
            self._live_box.blockSignals(blk)

        self._pill_conn.set_state(both_connected)

        now = time.time()
        if hb_seq != self._last_hb_seq:
            self._last_hb_seq = hb_seq
            self._last_hb_change_wall = now
        hb_ok = (
            chassis_connected
            and self._last_hb_change_wall > 0
            and (now - self._last_hb_change_wall) < self.HB_FRESH_S
        )
        self._pill_hb.set_state(hb_ok)

        estop_active = False
        if isinstance(motor, dict):
            estop_active = bool(motor.get("e_stop_active", False))
        elif isinstance(status, dict):
            estop_active = bool(status.get("e_stop_active", False))
        self._pill_estop.set_state(not estop_active)
