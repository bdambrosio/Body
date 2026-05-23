"""Qt UI for manual mapping sessions — nav-style teleop + camera feeds."""

from __future__ import annotations

import logging
import math
import time

from PyQt6.QtCore import Qt, QTimer, QUrl
from PyQt6.QtGui import QAction, QDesktopServices
from PyQt6.QtWidgets import (
    QApplication, QDockWidget, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QSplitter, QToolBar, QVBoxLayout, QWidget, QSizePolicy,
)

from desktop.chassis.config import StubConfig
from desktop.chassis.controller import StubController
from desktop.mapping.controller import MappingConfig, MappingController
from desktop.nav.camera_panels import CameraPanels, build_camera_snapshot
from desktop.nav.safety_toolbar import SafetyToolbar
from desktop.nav.teleop_panels import TeleopPanels, build_chassis_snapshot
from desktop.world_map.map_views import SharedMapView, WorldDriveableView

logger = logging.getLogger(__name__)


class MappingMainWindow(QMainWindow):
    def __init__(
        self,
        controller: MappingController,
        config: MappingConfig,
        chassis: StubController,
        chassis_config: StubConfig,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Body Mapping")
        self.controller = controller
        self.config = config
        self.chassis = chassis
        self.chassis_config = chassis_config

        self._build_toolbars()
        self._build_ui()
        self._build_docks()
        self._build_menu()
        self._build_timers()

    def _build_toolbars(self) -> None:
        self._safety_toolbar = SafetyToolbar(
            self.chassis, self.controller, parent=self,
        )
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self._safety_toolbar)

        self._map_toolbar = QToolBar("Map", self)
        self._map_toolbar.setMovable(False)
        self._map_toolbar.setFloatable(False)
        self._map_toolbar.setContextMenuPolicy(
            Qt.ContextMenuPolicy.PreventContextMenu,
        )

        reset_act = QAction("Reset map", self)
        reset_act.setToolTip(
            "Clear the occupancy grid and start a fresh mapping session.",
        )
        reset_act.triggered.connect(
            lambda: self.controller.request_reset(reason="ui_reset"),
        )
        self._map_toolbar.addAction(reset_act)

        save_act = QAction("Save map", self)
        save_act.setToolTip(
            "Export reference_map.npz and a snapshot bundle under "
            "~/Body/sessions/<session_id>/.",
        )
        save_act.triggered.connect(self._on_save)
        self._map_toolbar.addAction(save_act)

        fit_act = QAction("Fit maps", self)
        fit_act.setToolTip(
            "Reset map zoom/pan to auto-fit the populated region.",
        )
        fit_act.triggered.connect(self._on_fit_maps)
        self._map_toolbar.addAction(fit_act)

        self._stream_rgb_act = QAction("Stream RGB", self)
        self._stream_rgb_act.setCheckable(True)
        self._stream_rgb_act.setChecked(False)
        self._stream_rgb_act.setToolTip(
            "Toggle low-rate (2 Hz) streaming of OAK-D RGB into the "
            "feed pane — useful when the robot is out of sight.",
        )
        self._stream_rgb_act.toggled.connect(self._on_toggle_stream_rgb)
        self._map_toolbar.addAction(self._stream_rgb_act)

        self.addToolBarBreak(Qt.ToolBarArea.TopToolBarArea)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self._map_toolbar)

    def _build_ui(self) -> None:
        central = QWidget(self)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        self._v_splitter = QSplitter(Qt.Orientation.Vertical, central)
        self._v_splitter.setChildrenCollapsible(True)

        maps_widget = QWidget()
        maps = QHBoxLayout(maps_widget)
        maps.setContentsMargins(0, 0, 0, 0)
        self._shared_view = SharedMapView()
        self._drive_view = WorldDriveableView(
            stale_s=self.config.map_stale_s,
            shared=self._shared_view,
        )
        maps.addWidget(self._drive_view, stretch=1)
        self._v_splitter.addWidget(maps_widget)

        self._cameras = CameraPanels(self.chassis)
        self._v_splitter.addWidget(self._cameras.feeds_widget)

        self._v_splitter.setStretchFactor(0, 3)
        self._v_splitter.setStretchFactor(1, 1)
        self._splitter_balanced = False

        outer.addWidget(self._v_splitter, stretch=1)

        small = self.font()
        small.setPointSize(max(7, small.pointSize() - 1))
        self._pose_lbl = self._mk_status_label("pose: —", 220, small)
        self._heading_lbl = self._mk_status_label("heading: —", 180, small)
        self._cells_lbl = self._mk_status_label("cells: —", 180, small)
        self._session_lbl = self._mk_status_label("session: —", 210, small)
        self._chassis_lbl = self._mk_status_label("chassis: —", 200, small)
        self._notes_lbl = QLabel("")
        self._notes_lbl.setStyleSheet("color: #e8a; font-weight: bold;")
        self._notes_lbl.setFont(small)

        status = QVBoxLayout()
        status.setContentsMargins(0, 0, 0, 0)
        status.setSpacing(2)
        row1 = QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)
        row1.setSpacing(8)
        for w in (self._pose_lbl, self._heading_lbl, self._cells_lbl, self._session_lbl):
            row1.addWidget(w)
        row1.addStretch(1)
        status.addLayout(row1)
        row2 = QHBoxLayout()
        row2.setContentsMargins(0, 0, 0, 0)
        row2.setSpacing(8)
        row2.addWidget(self._chassis_lbl)
        row2.addStretch(1)
        status.addLayout(row2)
        row3 = QHBoxLayout()
        row3.setContentsMargins(0, 0, 0, 0)
        row3.addWidget(self._notes_lbl, stretch=1)
        status.addLayout(row3)
        outer.addLayout(status)

        self.setCentralWidget(central)
        self.resize(960, 880)

    def _mk_status_label(
        self, initial_text: str, width_px: int, font,
    ) -> QLabel:
        lbl = QLabel(initial_text)
        lbl.setStyleSheet("color: #ccc;")
        lbl.setFont(font)
        lbl.setFixedWidth(width_px)
        lbl.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred,
        )
        return lbl

    def _build_docks(self) -> None:
        self._teleop = TeleopPanels(self.chassis, self.chassis_config)
        self._teleop.attach_to(self)
        self._teleop.set_visible(True)

        self._vision_dock = QDockWidget("Vision", self)
        self._vision_dock.setWidget(self._cameras.vision_widget)
        self.addDockWidget(
            Qt.DockWidgetArea.LeftDockWidgetArea, self._vision_dock,
        )

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

        self._vision_action = QAction("Vision panel", self)
        self._vision_action.setCheckable(True)
        self._vision_action.setChecked(self._vision_dock.isVisible())
        self._vision_action.setShortcut("Ctrl+Shift+V")
        self._vision_action.triggered.connect(self._vision_dock.setVisible)
        view_menu.addAction(self._vision_action)

        view_menu.addSeparator()

        self._grid_action = QAction("Map grid (1 m)", self)
        self._grid_action.setCheckable(True)
        self._grid_action.setChecked(self._shared_view.show_grid())
        self._grid_action.setShortcut("Ctrl+G")
        self._grid_action.toggled.connect(self._shared_view.set_show_grid)
        view_menu.addAction(self._grid_action)

        self._rings_action = QAction("Range rings (1/2/5 m)", self)
        self._rings_action.setCheckable(True)
        self._rings_action.setChecked(self._shared_view.show_range_rings())
        self._rings_action.setShortcut("Ctrl+R")
        self._rings_action.toggled.connect(
            self._shared_view.set_show_range_rings,
        )
        view_menu.addAction(self._rings_action)

    def _build_timers(self) -> None:
        period_ms = int(1000.0 / max(1.0, self.config.ui_redraw_hz))
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setInterval(period_ms)
        self._redraw_timer.timeout.connect(self._on_redraw_tick)
        self._redraw_timer.start()

        self._stream_rgb_timer = QTimer(self)
        self._stream_rgb_timer.setInterval(500)
        self._stream_rgb_timer.timeout.connect(self._on_stream_rgb_tick)

    def _on_redraw_tick(self) -> None:
        self._safety_toolbar.refresh()
        self._refresh_map_panel()
        self._refresh_status_panel()
        self._refresh_chassis_panel()

        if self._teleop.is_visible():
            self._teleop.update_state(
                build_chassis_snapshot(self.chassis), time.time(),
            )
        if self._cameras.is_visible():
            cam_snap = build_camera_snapshot(self.chassis)
            cam_snap["streaming_on"] = self._stream_rgb_act.isChecked()
            self._cameras.update_state(cam_snap)

        if self._teleop_action.isChecked() != self._teleop.is_visible():
            self._teleop_action.setChecked(self._teleop.is_visible())
        if self._camera_action.isChecked() != self._cameras.is_visible():
            self._camera_action.setChecked(self._cameras.is_visible())
        if self._vision_action.isChecked() != self._vision_dock.isVisible():
            self._vision_action.setChecked(self._vision_dock.isVisible())

    def _refresh_map_panel(self) -> None:
        snap = self.controller.snapshot_for_ui()
        pose = self.controller.pose_tracker.pose()
        trail = self.controller.pose_trail()
        ts = time.time()
        if snap is None:
            self._drive_view.update_map(
                None, None, 0.0, pose=pose,
                pose_history=trail, bounds_ij=None,
            )
            return
        self._drive_view.update_map(
            snap.get("driveable"),
            snap.get("meta"),
            ts,
            pose=pose,
            pose_history=trail,
            bounds_ij=snap.get("bounds_ij"),
        )

    def _refresh_status_panel(self) -> None:
        st = self.controller.status_summary()
        pose = st.get("pose")
        if pose is not None:
            self._pose_lbl.setText(
                f"pose: ({pose[0]:+.2f}, {pose[1]:+.2f}, "
                f"{pose[2]:+.2f})"
            )
        else:
            self._pose_lbl.setText("pose: —")

        imu_settled = st.get("imu_settled")
        heading_src = st.get("heading_source") or "none"
        if pose is not None and imu_settled:
            deg = math.degrees(pose[2])
            tag = "imu" if heading_src == "imu" else "enc"
            self._heading_lbl.setText(f"heading: {tag} {deg:+5.1f}°")
        elif not imu_settled:
            self._heading_lbl.setText("heading: wait imu")
        else:
            self._heading_lbl.setText("heading: —")

        snap = self.controller.snapshot_for_ui()
        bounds = snap.get("bounds_ij") if snap else None
        if bounds is not None:
            i0, i1, j0, j1 = bounds
            cells = (i1 - i0 + 1) * (j1 - j0 + 1)
            self._cells_lbl.setText(f"cells: {cells:,}")
        else:
            self._cells_lbl.setText("cells: —")

        sid = st.get("session_id") or "—"
        self._session_lbl.setText(f"session: {sid[:8]}")

    def _refresh_chassis_panel(self) -> None:
        s = self.chassis.state
        with s.lock:
            connected = s.connected
            status_ts = s.status_ts
            hb_seq = s.heartbeat_seq
        if not connected:
            self._chassis_lbl.setText("chassis: disconnected")
            self._chassis_lbl.setStyleSheet("color: #a88;")
            return
        if status_ts > 0:
            age_s = f"{time.time() - status_ts:>4.1f}"
        else:
            age_s = "  — "
        self._chassis_lbl.setText(
            f"chassis: {age_s}s  #{hb_seq % 10000:04d}",
        )
        self._chassis_lbl.setStyleSheet("color: #ccc;")

    def _on_save(self) -> None:
        try:
            path = self.controller.save_snapshot_bundle()
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))
            return
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("Saved")
        msg.setText(f"Map saved to:\n{path}")
        open_btn = msg.addButton("Open folder", QMessageBox.ButtonRole.ActionRole)
        msg.addButton(QMessageBox.StandardButton.Ok)
        msg.exec()
        if msg.clickedButton() is open_btn:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _on_fit_maps(self) -> None:
        self._shared_view.reset_view()

    def _on_toggle_stream_rgb(self, checked: bool) -> None:
        if checked:
            self._stream_rgb_timer.start()
        else:
            self._stream_rgb_timer.stop()

    def _on_stream_rgb_tick(self) -> None:
        self.chassis.request_rgb_streaming()

    def _balance_splitter_once(self) -> None:
        if self._splitter_balanced:
            return
        v_total = self._v_splitter.height()
        if v_total <= 0:
            return
        maps_h = int(v_total * 0.75)
        feeds_h = max(0, v_total - maps_h)
        self._v_splitter.setSizes([maps_h, feeds_h])
        self._splitter_balanced = True

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._splitter_balanced:
            QTimer.singleShot(0, self._balance_splitter_once)

    def closeEvent(self, event) -> None:
        try:
            self._redraw_timer.stop()
            self._stream_rgb_timer.stop()
        except Exception:
            pass
        super().closeEvent(event)


def run_mapping_app(
    controller: MappingController,
    config: MappingConfig,
    chassis: StubController,
    chassis_config: StubConfig,
) -> int:
    app = QApplication.instance() or QApplication([])
    win = MappingMainWindow(controller, config, chassis, chassis_config)
    win.show()
    return app.exec()
