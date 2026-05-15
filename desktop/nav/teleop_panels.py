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

from PyQt6.QtCore import Qt, QSignalBlocker, pyqtSignal
from PyQt6.QtWidgets import (
    QDockWidget, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel,
    QMainWindow, QScrollArea, QVBoxLayout, QWidget,
)

from desktop.chassis.config import StubConfig
from desktop.chassis.controller import StubController
from desktop.chassis.sweep_dock import SweepDock
from desktop.chassis.ui_qt import DifferentialPad, MotorTestDock

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


class TwistPad(DifferentialPad):
    """Skidpad that emits cmd_vel twist instead of per-wheel velocities.

    Pad ny → linear m/s (up = forward). Pad nx → -angular rad/s so
    pushing right on the pad turns the robot right (ROS REP 103 has
    angular.z positive = CCW, i.e. left turn). Release-to-zero and
    expo curve are inherited from DifferentialPad.
    """

    twist_changed = pyqtSignal(float, float)  # (linear, angular)

    def __init__(
        self,
        max_linear: float,
        max_angular: float,
        parent: Optional[QWidget] = None,
    ):
        # Parent stores max_wheel; we don't use it, but pass a sane
        # number so inherited paint code has a value if it ever runs
        # before our subclass is fully initialized.
        super().__init__(max_linear, parent)
        self._max_linear = float(max_linear)
        self._max_angular = float(max_angular)

    def set_max_linear(self, v: float) -> None:
        self._max_linear = float(v)
        self._emit()

    def set_max_angular(self, v: float) -> None:
        self._max_angular = float(v)
        self._emit()

    def current_twist(self) -> tuple[float, float]:
        nx, ny = self._curved()
        linear = ny * self._max_linear
        angular = -nx * self._max_angular
        return linear, angular

    def _emit(self) -> None:
        linear, angular = self.current_twist()
        self.twist_changed.emit(linear, angular)

    def _readout_text(self) -> str:
        lin, ang = self.current_twist()
        return f"lin {lin:+.2f} m/s   ang {ang:+.2f} rad/s"


class CmdVelDock(QDockWidget):
    """Twist drive surface: skidpad → body/cmd_vel.

    Release-to-zero (dead-man style). The pad's twist_changed signal
    writes into controller.last_cmd_vel on every position change; the
    chassis publisher thread emits that value at cmd_vel_hz iff Live
    cmd is on. Use this for normal driving and SLAM mapping runs —
    cmd_vel keeps inputs inside the kinematic envelope that odometry
    assumes.
    """

    # (linear_mps, angular_rps)
    twist_changed = pyqtSignal(float, float)

    DEFAULT_MAX_LINEAR = 0.3
    DEFAULT_MAX_ANGULAR = 0.8

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

        limits_row = QHBoxLayout()
        limits_row.addWidget(QLabel("Max linear (m/s):"))
        self.max_linear_box = QDoubleSpinBox()
        self.max_linear_box.setRange(0.05, 1.0)
        self.max_linear_box.setSingleStep(0.05)
        self.max_linear_box.setDecimals(2)
        self.max_linear_box.setValue(self.DEFAULT_MAX_LINEAR)
        limits_row.addWidget(self.max_linear_box)
        limits_row.addSpacing(12)
        limits_row.addWidget(QLabel("Max angular (rad/s):"))
        self.max_angular_box = QDoubleSpinBox()
        self.max_angular_box.setRange(0.05, 3.0)
        self.max_angular_box.setSingleStep(0.1)
        self.max_angular_box.setDecimals(2)
        self.max_angular_box.setValue(self.DEFAULT_MAX_ANGULAR)
        limits_row.addWidget(self.max_angular_box)
        limits_row.addStretch(1)
        v.addLayout(limits_row)

        self.pad = TwistPad(self.DEFAULT_MAX_LINEAR, self.DEFAULT_MAX_ANGULAR)
        pad_row = QHBoxLayout()
        pad_row.addStretch(1)
        pad_row.addWidget(self.pad)
        pad_row.addStretch(1)
        v.addLayout(pad_row, 1)

        hint = QLabel(
            "Hold-to-drive. Requires Live cmd (safety toolbar) to be on."
        )
        hint.setStyleSheet("color:#888;")
        hint.setWordWrap(True)
        v.addWidget(hint)

        self.setWidget(body)

        self.pad.twist_changed.connect(self.twist_changed)
        self.max_linear_box.valueChanged.connect(self.pad.set_max_linear)
        self.max_angular_box.valueChanged.connect(self.pad.set_max_angular)

    def zero_and_disable(self) -> None:
        """Used when ALL-STOP fires or sweep mission starts."""
        self.pad.recenter()


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
        self.cmd_vel_dock.twist_changed.connect(self._on_cmd_vel_twist)
        self.motor_dock.mode_change_requested.connect(
            self._on_motor_mode_change
        )
        self.motor_dock.cmd_direct_changed.connect(
            self._on_cmd_direct_changed
        )
        self.motor_dock.stop_requested.connect(self._on_motor_stop)
        self.sweep_dock.mission_active.connect(self._on_sweep_active)

    def _on_cmd_vel_twist(self, linear: float, angular: float) -> None:
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

        Block signals on the pad and engage button when forcibly
        resetting their UI state. recenter() emits twist_changed(0, 0)
        → _on_cmd_vel_twist → chassis.set_cmd_vel(0, 0); unchecking
        engage_btn emits mode_change_requested("cmd_vel") →
        chassis.stop_all(). Both race against the sweep worker's
        per-step set_cmd_vel(0, ω) + set_live_command(True), and
        mission_active fires on every state change (PRECHECK,
        ROTATING, SETTLING, ...) — when the GUI wins those races, the
        worker's command is clobbered back to zero and the bot sits
        still for the whole sweep. Signal-blocking preserves the
        visual reset without the controller-state side effects.
        """
        self.cmd_vel_dock.pad.setEnabled(not active)
        self.motor_dock.engage_btn.setEnabled(not active)
        if active:
            with QSignalBlocker(self.cmd_vel_dock.pad):
                self.cmd_vel_dock.pad.recenter()
        if active and self.motor_dock.engage_btn.isChecked():
            with QSignalBlocker(self.motor_dock.engage_btn):
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
