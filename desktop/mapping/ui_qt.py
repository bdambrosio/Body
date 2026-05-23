"""Minimal Qt UI for mapping sessions."""

from __future__ import annotations

import logging

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QVBoxLayout, QWidget, QMessageBox,
)

from desktop.mapping.controller import MappingConfig, MappingController
from desktop.world_map.map_views import WorldDriveableView

logger = logging.getLogger(__name__)


class MappingWindow(QMainWindow):
    def __init__(self, controller: MappingController, config: MappingConfig):
        super().__init__()
        self.setWindowTitle("Body Mapping")
        self.controller = controller
        self.config = config
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._redraw)
        self._timer.start(200)

    def _build_ui(self) -> None:
        central = QWidget(self)
        layout = QVBoxLayout(central)
        top = QHBoxLayout()
        self._connect_btn = QPushButton("Connect")
        self._connect_btn.clicked.connect(self._on_connect)
        top.addWidget(self._connect_btn)
        self._reset_btn = QPushButton("Reset map")
        self._reset_btn.clicked.connect(
            lambda: self.controller.request_reset(reason="ui_reset"),
        )
        top.addWidget(self._reset_btn)
        self._save_btn = QPushButton("Save map")
        self._save_btn.clicked.connect(self._on_save)
        top.addWidget(self._save_btn)
        self._status = QLabel("disconnected")
        top.addWidget(self._status, stretch=1)
        layout.addLayout(top)
        self._drive_view = WorldDriveableView(stale_s=2.0)
        layout.addWidget(self._drive_view, stretch=1)
        self.setCentralWidget(central)

    def _on_connect(self) -> None:
        if self.controller.connected:
            self.controller.shutdown()
            self._connect_btn.setText("Connect")
            self._status.setText("disconnected")
            return
        ok, err = self.controller.connect()
        if ok:
            self._connect_btn.setText("Disconnect")
            self._status.setText("connected")
        else:
            QMessageBox.warning(self, "Connect failed", err or "unknown")

    def _on_save(self) -> None:
        try:
            path = self.controller.save_snapshot_bundle()
            QMessageBox.information(self, "Saved", f"Map saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))

    def _redraw(self) -> None:
        snap = self.controller.snapshot_for_ui()
        if snap is None:
            return
        import time
        meta = snap.get("meta") or {}
        self._drive_view.update_map(
            snap.get("driveable"),
            meta,
            time.time(),
            pose=self.controller.pose_tracker.pose(),
            bounds_ij=snap.get("bounds_ij"),
        )
        st = self.controller.status_summary()
        pose = st.get("pose")
        if pose:
            self._status.setText(
                f"pose=({pose[0]:+.2f}, {pose[1]:+.2f}) session={st.get('session_id')}"
            )


def run_mapping_app(controller: MappingController, config: MappingConfig) -> int:
    app = QApplication.instance() or QApplication([])
    win = MappingWindow(controller, config)
    win.resize(900, 700)
    win.show()
    return app.exec()
