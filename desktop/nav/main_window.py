"""Main window for the nav operator UI.

Hosts both controllers in one Qt process. Central widget = the two
world-map views (height + driveable). A persistent SafetyToolbar owns
Connect/ALL-STOP/Live/pills (steps 3 and 4 will add teleop + camera
docks). A small Map toolbar carries world_map-specific actions (reset
today; goto/explore/etc. later).
"""
from __future__ import annotations

import logging
import math
import time

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QMainWindow, QToolBar,
    QVBoxLayout, QWidget,
)
from PyQt6.QtGui import QAction

from desktop.chassis.config import StubConfig
from desktop.chassis.controller import StubController
from desktop.world_map.config import FuserConfig
from desktop.world_map.controller import FuserController
from desktop.world_map.map_views import WorldDriveableView, WorldHeightView

from .camera_panels import CameraPanels, build_camera_snapshot
from .safety_toolbar import SafetyToolbar
from .teleop_panels import TeleopPanels, build_chassis_snapshot

logger = logging.getLogger(__name__)


class NavMainWindow(QMainWindow):
    def __init__(
        self,
        fuser: FuserController,
        fuser_config: FuserConfig,
        chassis: StubController,
        chassis_config: StubConfig,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Body Nav")
        self.fuser = fuser
        self.fuser_config = fuser_config
        self.chassis = chassis
        self.chassis_config = chassis_config

        self._build_toolbars()
        self._build_ui()
        self._build_docks()
        self._build_menu()
        self._build_timer()

    # ── Layout ───────────────────────────────────────────────────────

    def _build_toolbars(self) -> None:
        self._safety_toolbar = SafetyToolbar(self.chassis, self.fuser, parent=self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self._safety_toolbar)

        # Map toolbar: world_map-scoped actions. Starts with Reset-world;
        # grows as nav primitives (goto/explore/...) come online.
        self._map_toolbar = QToolBar("Map", self)
        self._map_toolbar.setMovable(False)
        self._map_toolbar.setFloatable(False)
        self._map_toolbar.setContextMenuPolicy(
            Qt.ContextMenuPolicy.PreventContextMenu
        )
        reset_act = QAction("Reset world", self)
        reset_act.setToolTip(
            "Clear the accumulated world map and re-anchor to the "
            "robot's current pose."
        )
        reset_act.triggered.connect(
            lambda: self.fuser.request_reset(reason="ui_reset")
        )
        self._map_toolbar.addAction(reset_act)
        self.addToolBarBreak(Qt.ToolBarArea.TopToolBarArea)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self._map_toolbar)

    def _build_ui(self) -> None:
        central = QWidget(self)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        maps = QHBoxLayout()
        self._height_view = WorldHeightView(stale_s=self.fuser_config.map_stale_s)
        self._drive_view = WorldDriveableView(stale_s=self.fuser_config.map_stale_s)
        maps.addWidget(self._height_view, stretch=1)
        maps.addWidget(self._drive_view, stretch=1)
        outer.addLayout(maps, stretch=1)

        # Bottom status strip: fuser (pose + rates + cells + session) on
        # the left, chassis text summary on the right. The safety pills
        # at the top handle *gate* state (conn/hb/estop); this strip is
        # for values the pills can't convey (ages, counts, session id).
        bot = QHBoxLayout()
        self._pose_lbl = QLabel("pose: —")
        self._rates_lbl = QLabel("rates: —")
        self._cells_lbl = QLabel("cells: —")
        self._session_lbl = QLabel("session: —")
        self._chassis_lbl = QLabel("chassis: —")
        self._notes_lbl = QLabel("")
        self._notes_lbl.setStyleSheet("color: #e8a; font-weight: bold;")
        for w in (self._pose_lbl, self._rates_lbl,
                  self._cells_lbl, self._session_lbl, self._chassis_lbl):
            w.setStyleSheet("color: #ccc;")
            bot.addWidget(w)
        bot.addWidget(self._notes_lbl, stretch=1)
        outer.addLayout(bot)

        self.setCentralWidget(central)
        self.resize(1000, 600)

    def _build_docks(self) -> None:
        # By default QMainWindow gives the bottom corners to the Bottom
        # dock area, which would let the Cameras dock span the full
        # width and squeeze Teleop out of the lower-right. Hand the
        # right corners to the Right area so Teleop occupies the entire
        # right column regardless of what's docked at the bottom.
        self.setCorner(
            Qt.Corner.TopRightCorner, Qt.DockWidgetArea.RightDockWidgetArea,
        )
        self.setCorner(
            Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.RightDockWidgetArea,
        )
        self._teleop = TeleopPanels(self.chassis, self.chassis_config)
        self._teleop.attach_to(self)
        self._cameras = CameraPanels(self.chassis)
        self._cameras.attach_to(self)

    def _build_menu(self) -> None:
        view_menu = self.menuBar().addMenu("&View")
        self._teleop_action = QAction("Teleop panels", self)
        self._teleop_action.setCheckable(True)
        self._teleop_action.setChecked(self._teleop.is_visible())
        self._teleop_action.setShortcut("Ctrl+T")
        self._teleop_action.triggered.connect(self._teleop.set_visible)
        view_menu.addAction(self._teleop_action)

        self._camera_action = QAction("Camera panels", self)
        self._camera_action.setCheckable(True)
        self._camera_action.setChecked(self._cameras.is_visible())
        self._camera_action.setShortcut("Ctrl+Shift+C")
        self._camera_action.triggered.connect(self._cameras.set_visible)
        view_menu.addAction(self._camera_action)

    def _build_timer(self) -> None:
        period_ms = int(1000.0 / max(1.0, self.fuser_config.ui_redraw_hz))
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setInterval(period_ms)
        self._redraw_timer.timeout.connect(self._on_redraw_tick)
        self._redraw_timer.start()

    # ── Tick ─────────────────────────────────────────────────────────

    def _on_redraw_tick(self) -> None:
        self._safety_toolbar.refresh()
        self._refresh_fuser_panel()
        self._refresh_chassis_panel()
        # Dock groups only consume ticks when visible. Each group owns
        # its own snapshot builder so nav doesn't have to know which
        # state fields either group reads.
        if self._teleop.is_visible():
            self._teleop.update_state(
                build_chassis_snapshot(self.chassis), time.time(),
            )
        if self._cameras.is_visible():
            self._cameras.update_state(build_camera_snapshot(self.chassis))
        # Keep View menu checkmarks in sync if the user closed a dock
        # via its titlebar X rather than via the menu action.
        if self._teleop_action.isChecked() != self._teleop.is_visible():
            self._teleop_action.setChecked(self._teleop.is_visible())
        if self._camera_action.isChecked() != self._cameras.is_visible():
            self._camera_action.setChecked(self._cameras.is_visible())

    def _refresh_fuser_panel(self) -> None:
        snap = self.fuser.snapshot_for_ui()
        latest = self.fuser.pose_source.latest_pose()
        pose = latest[0] if latest is not None else None
        ts = time.time()
        if snap is not None:
            self._height_view.update_map(
                snap["grid"], snap["meta"], ts, pose=pose,
            )
            self._drive_view.update_map(
                snap["driveable"], snap["meta"], ts, pose=pose,
            )
        else:
            self._height_view.update_map(None, None, 0.0, pose=pose)
            self._drive_view.update_map(None, None, 0.0, pose=pose)

        st = self.fuser.status_summary()
        if pose is not None:
            self._pose_lbl.setText(
                f"pose: x={pose[0]:+.2f} y={pose[1]:+.2f} "
                f"θ={math.degrees(pose[2]):+.1f}°"
            )
        else:
            self._pose_lbl.setText("pose: (no odom)")

        rates = st["rates"]
        ages = st["ages"]
        parts = []
        for name, key in (("lm", "local_map"), ("od", "odom")):
            hz = rates.get(key)
            age = ages.get(key)
            hz_s = f"{hz:.1f}Hz" if hz is not None else "—"
            age_s = f"{age:.2f}s" if age is not None else "—"
            parts.append(f"{name} {hz_s}/{age_s}")
        self._rates_lbl.setText("rates: " + "  ".join(parts))

        self._cells_lbl.setText(
            f"cells: obs={st['cells_observed']} trav={st['cells_traversed']}"
        )
        self._session_lbl.setText(
            f"session: {st['session_id'][:8]}  ({st['pose_source']})"
        )
        self._notes_lbl.setText(st.get("notes") or "")

    def _refresh_chassis_panel(self) -> None:
        """Text summary with values the pills can't convey (status age,
        heartbeat seq). Gate colors live on the safety toolbar.
        """
        s = self.chassis.state
        with s.lock:
            connected = s.connected
            status_ts = s.status_ts
            hb_seq = s.heartbeat_seq
        if not connected:
            self._chassis_lbl.setText("chassis: disconnected")
            self._chassis_lbl.setStyleSheet("color: #a88;")
            return
        age_s = (
            f"{time.time() - status_ts:.1f}s" if status_ts > 0 else "—"
        )
        self._chassis_lbl.setText(
            f"chassis: status/{age_s}  hb#{hb_seq}"
        )
        self._chassis_lbl.setStyleSheet("color: #ccc;")

    # ── Lifecycle ────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        try:
            self._redraw_timer.stop()
        except Exception:
            pass
        super().closeEvent(event)


def run_app(
    fuser: FuserController,
    fuser_config: FuserConfig,
    chassis: StubController,
    chassis_config: StubConfig,
) -> int:
    app = QApplication.instance() or QApplication([])
    win = NavMainWindow(fuser, fuser_config, chassis, chassis_config)
    win.show()
    return app.exec()
