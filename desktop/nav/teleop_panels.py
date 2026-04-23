"""Teleop dock group for the nav shell.

Three tabified QDockWidgets on the right-hand side:
- CmdVelDock (new, nav-only): linear/angular spinboxes + Apply — the
  twist equivalent of the chassis main-window cmd_row.
- MotorTestDock (reused from chassis): per-wheel cmd_direct bring-up.
- SweepDock (reused from chassis): sweep-360 calibration mission.

Exposed as a single TeleopPanels manager so nav's View menu can toggle
the whole group with one action. Wiring (signals → chassis controller)
mirrors the chassis BodyStubWindow so behavior matches what you get
running `python -m desktop.chassis` standalone.
"""
from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDockWidget, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel,
    QMainWindow, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from desktop.chassis.config import StubConfig
from desktop.chassis.controller import StubController
from desktop.chassis.sweep_dock import SweepDock
from desktop.chassis.ui_qt import MotorTestDock

logger = logging.getLogger(__name__)


def _make_scrollable(dock: QDockWidget) -> None:
    """Re-wrap dock.widget() inside a QScrollArea so the dock content
    can scroll vertically instead of forcing the main window taller.

    No-op if the dock has no body widget yet."""
    body = dock.widget()
    if body is None:
        return
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    scroll.setWidget(body)
    dock.setWidget(scroll)


class CmdVelDock(QDockWidget):
    """Twist input: linear (m/s) + angular (rad/s) + Apply.

    Apply writes into controller.last_cmd_vel; the chassis publisher
    thread emits that value at cmd_vel_hz iff Live command is on.
    """

    apply_requested = pyqtSignal(float, float)  # (linear, angular)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Drive (cmd_vel)", parent)
        self.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.BottomDockWidgetArea
        )

        body = QWidget(self)
        v = QVBoxLayout(body)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(6)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("linear (m/s):"))
        self.linear_box = QDoubleSpinBox()
        self.linear_box.setRange(-1.0, 1.0)
        self.linear_box.setSingleStep(0.05)
        self.linear_box.setDecimals(2)
        self.linear_box.setValue(0.3)
        row1.addWidget(self.linear_box)
        row1.addStretch(1)
        v.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("angular (rad/s):"))
        self.angular_box = QDoubleSpinBox()
        self.angular_box.setRange(-2.0, 2.0)
        self.angular_box.setSingleStep(0.1)
        self.angular_box.setDecimals(2)
        self.angular_box.setValue(0.2)
        row2.addWidget(self.angular_box)
        row2.addStretch(1)
        v.addLayout(row2)

        row3 = QHBoxLayout()
        self.apply_btn = QPushButton("Apply cmd_vel")
        self.apply_btn.setToolTip(
            "Set the stored cmd_vel. Values are published at cmd_vel_hz "
            "only while Live cmd (safety toolbar) is on."
        )
        row3.addWidget(self.apply_btn)
        row3.addStretch(1)
        v.addLayout(row3)
        v.addStretch(1)

        self.setWidget(body)
        self.apply_btn.clicked.connect(self._emit_apply)

    def _emit_apply(self) -> None:
        self.apply_requested.emit(
            float(self.linear_box.value()),
            float(self.angular_box.value()),
        )

    def zero_and_disable(self) -> None:
        """Used when ALL-STOP fires or sweep mission starts."""
        self.linear_box.setValue(0.0)
        self.angular_box.setValue(0.0)


class TeleopPanels:
    """Owns the three teleop docks, their wiring, and the group toggle.

    Not a QWidget itself — just a coordinator. Construct with the chassis
    controller + config; call `attach_to(main_window)` to install docks.
    """

    def __init__(
        self,
        chassis: StubController,
        chassis_config: StubConfig,
    ) -> None:
        self.chassis = chassis
        self.chassis_config = chassis_config

        self.cmd_vel_dock = CmdVelDock()
        self.motor_dock = MotorTestDock(
            max_wheel_default=chassis_config.max_wheel_vel_default,
            timeout_ms_default=chassis_config.cmd_vel_timeout_ms,
        )
        self.sweep_dock = SweepDock(chassis)
        # MotorTestDock and SweepDock stack a lot vertically (pad + pills
        # + readouts); without scroll-wrapping, their content min height
        # forces the whole QMainWindow min height up when the user opens
        # the teleop panels. Wrap body widgets in a QScrollArea so they
        # scroll inside a small dock rather than growing the window.
        _make_scrollable(self.motor_dock)
        _make_scrollable(self.sweep_dock)

        self._installed: bool = False
        self._wire_signals()

    # ── Install into parent QMainWindow ──────────────────────────────

    def attach_to(self, window: QMainWindow) -> None:
        """Add the three docks to `window`, tabify them, default hide."""
        area = Qt.DockWidgetArea.RightDockWidgetArea
        window.addDockWidget(area, self.cmd_vel_dock)
        window.addDockWidget(area, self.motor_dock)
        window.addDockWidget(area, self.sweep_dock)
        window.tabifyDockWidget(self.cmd_vel_dock, self.motor_dock)
        window.tabifyDockWidget(self.cmd_vel_dock, self.sweep_dock)
        self.set_visible(False)
        self.cmd_vel_dock.raise_()
        self._installed = True

    # ── Group visibility ─────────────────────────────────────────────

    def set_visible(self, visible: bool) -> None:
        for d in (self.cmd_vel_dock, self.motor_dock, self.sweep_dock):
            d.setVisible(visible)

    def is_visible(self) -> bool:
        return any(
            d.isVisible() for d in
            (self.cmd_vel_dock, self.motor_dock, self.sweep_dock)
        )

    # ── Wiring ───────────────────────────────────────────────────────

    def _wire_signals(self) -> None:
        self.cmd_vel_dock.apply_requested.connect(self._on_cmd_vel_apply)
        self.motor_dock.mode_change_requested.connect(
            self._on_motor_mode_change
        )
        self.motor_dock.cmd_direct_changed.connect(
            self._on_cmd_direct_changed
        )
        self.motor_dock.stop_requested.connect(self._on_motor_stop)
        self.sweep_dock.mission_active.connect(self._on_sweep_active)

    def _on_cmd_vel_apply(self, linear: float, angular: float) -> None:
        self.chassis.set_cmd_vel(linear, angular)

    def _on_motor_mode_change(self, mode: str) -> None:
        # Mirror chassis BodyStubWindow._on_motor_mode_change: engage ON
        # switches mode to cmd_direct + arms Live; engage OFF stops all
        # and reverts to cmd_vel. The safety-toolbar Live checkbox picks
        # up the live state change on its next refresh tick.
        self.chassis.config.cmd_vel_timeout_ms = self.motor_dock.timeout_ms()
        if mode == "cmd_direct":
            self.chassis.set_cmd_mode("cmd_direct")
            self.chassis.set_live_command(True)
        else:
            self.chassis.stop_all()
            self.chassis.set_cmd_mode("cmd_vel")

    def _on_cmd_direct_changed(self, left: float, right: float) -> None:
        self.chassis.set_cmd_direct(left, right)

    def _on_motor_stop(self) -> None:
        self.chassis.stop_all()
        self.cmd_vel_dock.zero_and_disable()

    def _on_sweep_active(self, active: bool) -> None:
        """Lock out competing commanders while sweep owns cmd_vel.

        Safety toolbar's ALL-STOP and Live checkbox remain live — the
        operator must always be able to panic-stop a mission.
        """
        self.cmd_vel_dock.apply_btn.setEnabled(not active)
        self.motor_dock.engage_btn.setEnabled(not active)
        if active and self.motor_dock.engage_btn.isChecked():
            # Running sweep while direct-mode engaged would race; force
            # disengage (same pattern as chassis).
            self.motor_dock.engage_btn.setChecked(False)

    # ── Per-tick update ──────────────────────────────────────────────

    def update_state(self, snap: dict, now: float) -> None:
        """Called from the nav main window's redraw tick."""
        self.motor_dock.update_state(snap, now)


def build_chassis_snapshot(chassis: StubController) -> dict:
    """Assemble the dict shape MotorTestDock.update_state expects.

    Mirrors chassis BodyStubWindow._snapshot — only the keys motor_dock
    reads are populated; other keys (depth, rgb, lidar) would be wasted
    work here and are omitted.
    """
    s = chassis.state
    with s.lock:
        return dict(
            connected=s.connected, live=s.live_command,
            status=s.status, status_ts=s.status_ts,
            odom=s.odom, odom_ts=s.odom_ts,
            motor=s.motor_state, motor_ts=s.motor_ts,
        )
