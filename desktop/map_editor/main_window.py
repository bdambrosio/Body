"""Map-editor main window.

Loads a `reference_map.npz`, renders the driveable layer, and lets the
operator paint Wall / Free / Unknown with a disk brush, then Save /
Save As. Undo is stroke-grouped. The editor never fuses — the brush is
the only writer of the map.

Phase 2 (optional, when a `--router` is supplied): a Connect action
brings up a read-only live link (MCL pose + lidar scan). The live scan
is drawn over the map so the operator can correct it against ground
truth. Relocate / Set-location seat the pose. None of this edits the
map; the brush remains the only writer.
"""
from __future__ import annotations

import logging
import math
import os
import time
from typing import List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QActionGroup
from PyQt6.QtWidgets import (
    QApplication, QFileDialog, QLabel, QMainWindow, QMessageBox, QSpinBox,
    QToolBar, QWidget,
)

from desktop.world_map.map_views import SharedMapView

from . import editor_map as em
from .editable_map_view import EditableMapView
from .live_overlay import body_xy_to_world, pose_compose, pose_relative

logger = logging.getLogger(__name__)

_UNDO_DEPTH = 25
_SCAN_MAX_RANGE_M = 12.0


class MapEditorWindow(QMainWindow):
    def __init__(self, map_path: Optional[str] = None, *,
                 router: Optional[str] = None, pf_device: str = "auto",
                 pf_particles: int = 5000) -> None:
        super().__init__()
        self.setWindowTitle("World Map Editor")
        self.resize(1100, 900)

        self._emap: Optional[em.EditorMap] = None
        self._path: Optional[str] = None
        self._dirty: bool = False
        self._brush_kind: str = em.WALL
        self._undo: List[np.ndarray] = []
        self._drive_cache: Optional[np.ndarray] = None

        # Live (Phase 2) state — only wired when a router is given.
        self._router = router
        self._pf_device = pf_device
        self._pf_particles = pf_particles
        self._link = None  # LiveLink, created on Connect
        self._live_pose: Optional[Tuple[float, float, float]] = None
        # Manual-align overlay pose (world). When set, the overlay is
        # drawn at this operator-aligned pose, dead-reckoned by odom
        # only (no scan-match creep). None → use the MCL pose.
        self._overlay_pose: Optional[Tuple[float, float, float]] = None
        self._odom_anchor: Optional[Tuple[float, float, float]] = None

        self._shared = SharedMapView()
        self._view = EditableMapView(shared=self._shared)
        self.setCentralWidget(self._view)
        self._view.paintAtWorld.connect(self._on_paint_at)
        self._view.strokeStarted.connect(self._on_stroke_started)
        self._view.alignDragWorld.connect(self._on_align_drag)
        self._shared.set_locate_callback(self._on_locate)

        self._build_toolbar()
        self._status = self.statusBar()
        # Persistent numeric pose readout (right side of the status bar).
        self._pose_lbl = QLabel("")
        self._pose_lbl.setStyleSheet("color:#9cf; font-family:monospace;")
        self._status.addPermanentWidget(self._pose_lbl)

        # Live redraw timer (5 Hz). Inert until connected.
        self._live_timer = QTimer(self)
        self._live_timer.setInterval(200)
        self._live_timer.timeout.connect(self._live_tick)

        if map_path:
            self._load(map_path)
        else:
            self._refresh_actions()

    # ── Toolbar ─────────────────────────────────────────────────────

    def _build_toolbar(self) -> None:
        tb = QToolBar("main")
        tb.setMovable(False)
        self.addToolBar(tb)

        self._act_load = tb.addAction("Load…", self._on_load)
        self._act_save = tb.addAction("Save", self._on_save)
        self._act_save_as = tb.addAction("Save As…", self._on_save_as)
        tb.addSeparator()

        self._act_edit = tb.addAction("Edit")
        self._act_edit.setCheckable(True)
        self._act_edit.toggled.connect(self._view.set_paint_mode)
        tb.addSeparator()

        # Brush palette as an exclusive, checkable action group.
        grp = QActionGroup(self)
        grp.setExclusive(True)
        for kind, label in ((em.WALL, "Wall"), (em.FREE, "Free"),
                            (em.UNKNOWN, "Unknown")):
            a = tb.addAction(label)
            a.setCheckable(True)
            a.triggered.connect(lambda _c=False, k=kind: self._set_kind(k))
            grp.addAction(a)
            if kind == em.WALL:
                a.setChecked(True)
        tb.addSeparator()

        tb.addWidget(QLabel(" Brush "))
        self._brush_spin = QSpinBox()
        self._brush_spin.setRange(0, 40)
        self._brush_spin.setValue(2)
        self._brush_spin.setSuffix(" cells")
        self._brush_spin.setToolTip("Brush radius in grid cells (0 = single cell).")
        tb.addWidget(self._brush_spin)
        tb.addSeparator()

        self._act_undo = tb.addAction("Undo", self._on_undo)
        self._act_undo.setShortcut("Ctrl+Z")

        # Live (Phase 2) controls — only when a router is configured.
        self._act_connect = None
        self._act_relocate = None
        self._act_locate = None
        if self._router:
            tb.addSeparator()
            # NB: drive Connect from `toggled` (passes the checked bool),
            # not addAction's `triggered` (which calls the slot with no
            # args → _on_connect would lose its `want`).
            self._act_connect = tb.addAction("Connect")
            self._act_connect.setCheckable(True)
            self._act_connect.toggled.connect(self._on_connect)
            self._act_relocate = tb.addAction("Relocate", self._on_relocate)
            self._act_relocate.setEnabled(False)
            self._act_locate = tb.addAction("Set location")
            self._act_locate.setCheckable(True)
            self._act_locate.setEnabled(False)
            self._act_locate.toggled.connect(self._on_locate_armed)
            tb.addSeparator()
            # Manual align: drag the scan onto trusted walls, rotate with
            # the buttons. Dead-reckoned by odom (no scan-match creep).
            self._act_align = tb.addAction("Align scan")
            self._act_align.setCheckable(True)
            self._act_align.setEnabled(False)
            self._act_align.toggled.connect(self._on_align_toggled)
            self._act_rot_ccw = tb.addAction("⟲")
            self._act_rot_ccw.setToolTip("Rotate scan +1° (key: ,)")
            self._act_rot_ccw.setShortcut(",")
            self._act_rot_ccw.triggered.connect(lambda: self._on_rotate(+1))
            self._act_rot_cw = tb.addAction("⟳")
            self._act_rot_cw.setToolTip("Rotate scan −1° (key: .)")
            self._act_rot_cw.setShortcut(".")
            self._act_rot_cw.triggered.connect(lambda: self._on_rotate(-1))
            for a in (self._act_rot_ccw, self._act_rot_cw):
                a.setEnabled(False)

    def _set_kind(self, kind: str) -> None:
        self._brush_kind = kind

    # ── Load / Save ─────────────────────────────────────────────────

    def _on_load(self) -> None:
        if not self._confirm_discard():
            return
        start = (os.path.dirname(self._path) if self._path
                 else os.path.expanduser("~/Body/maps"))
        path, _ = QFileDialog.getOpenFileName(
            self, "Load reference map", start,
            "Reference map (reference_map.npz);;NumPy (*.npz)")
        if path:
            self._load(path)

    def _load(self, path: str) -> None:
        try:
            self._emap = em.load_npz(path)
        except Exception as e:  # noqa: BLE001 — surface to operator
            logger.exception("load failed")
            QMessageBox.critical(self, "Load failed",
                                 f"{type(e).__name__}: {e}")
            return
        self._path = path
        self._dirty = False
        self._undo.clear()
        nx, ny = self._emap.shape
        logger.info("loaded %s (%dx%d res=%.3f)", path, nx, ny,
                    self._emap.resolution_m)
        self._rerender(fit=True)
        self._refresh_actions()
        self._update_title()

    def _on_save(self) -> None:
        if self._emap is None or self._path is None:
            return self._on_save_as()
        self._write(self._path, backup=True)

    def _on_save_as(self) -> None:
        if self._emap is None:
            return
        start = self._path or os.path.expanduser("~/Body/maps/reference_map.npz")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save reference map as", start, "Reference map (*.npz)")
        if not path:
            return
        if not path.endswith(".npz"):
            path += ".npz"
        self._write(path, backup=True)
        self._path = path

    def _write(self, path: str, *, backup: bool) -> None:
        try:
            em.save_npz(self._emap, path, backup=backup)
        except Exception as e:  # noqa: BLE001
            logger.exception("save failed")
            QMessageBox.critical(self, "Save failed",
                                 f"{type(e).__name__}: {e}")
            return
        self._dirty = False
        self._update_title()
        self._status.showMessage(f"Saved {path}", 4000)

    # ── Paint ───────────────────────────────────────────────────────

    def _on_stroke_started(self) -> None:
        if self._emap is None:
            return
        self._undo.append(self._emap.snapshot_occ())
        if len(self._undo) > _UNDO_DEPTH:
            self._undo.pop(0)
        self._act_undo.setEnabled(True)

    def _on_paint_at(self, x_w: float, y_w: float) -> None:
        if self._emap is None:
            return
        i, j = self._emap.world_to_cell(x_w, y_w)
        if not self._emap.in_bounds(i, j):
            return
        ii, jj = self._emap.brush_cells(i, j, self._brush_spin.value())
        self._emap.paint(ii, jj, self._brush_kind)
        self._dirty = True
        self._rerender(fit=False)
        self._update_title()

    def _on_undo(self) -> None:
        if self._emap is None or not self._undo:
            return
        self._emap.restore_occ(self._undo.pop())
        self._dirty = True
        self._rerender(fit=False)
        self._update_title()
        self._act_undo.setEnabled(bool(self._undo))

    # ── Render / state ──────────────────────────────────────────────

    def _rerender(self, *, fit: bool) -> None:
        if self._emap is None:
            return
        if fit:
            self._shared.reset_view()
        # Cache the driveable render so the live tick can refresh the
        # pose marker without recomputing it every frame.
        self._drive_cache = self._emap.driveable_grid()
        self._view.update_map(
            self._drive_cache, self._emap.meta, ts=time.time(),
            pose=self._live_pose, bounds_ij=self._emap.bounds_ij(),
        )

    def _refresh_actions(self) -> None:
        have = self._emap is not None
        for a in (self._act_save, self._act_save_as, self._act_edit):
            a.setEnabled(have)
        self._act_undo.setEnabled(have and bool(self._undo))

    # ── Live overlay (Phase 2) ──────────────────────────────────────

    def _on_connect(self, want: bool) -> None:
        if want:
            if self._emap is None or self._path is None:
                QMessageBox.warning(self, "No map",
                                    "Load a reference map before connecting.")
                self._act_connect.setChecked(False)
                return
            from .live_overlay import LiveLink
            self._link = LiveLink(
                self._router, self._path, pf_device=self._pf_device,
                pf_particles=self._pf_particles)
            self._status.showMessage("Connecting…")
            QApplication.processEvents()
            ok, err = self._link.connect()
            if not ok:
                QMessageBox.critical(self, "Connect failed", str(err))
                self._link = None
                self._act_connect.setChecked(False)
                self._status.showMessage("Connect failed", 4000)
                return
            self._act_connect.setText("Disconnect")
            self._act_relocate.setEnabled(True)
            self._act_locate.setEnabled(True)
            self._act_align.setEnabled(True)
            self._act_rot_ccw.setEnabled(True)
            self._act_rot_cw.setEnabled(True)
            self._live_timer.start()
            self._status.showMessage("Connected — Relocate to seat the pose.")
        else:
            self._teardown_link()

    def _teardown_link(self) -> None:
        self._live_timer.stop()
        if self._link is not None:
            self._link.disconnect()
            self._link = None
        self._live_pose = None
        self._overlay_pose = None
        self._odom_anchor = None
        self._view.set_scan_points(None)
        self._pose_lbl.setText("")
        if self._act_connect is not None:
            self._act_connect.setText("Connect")
            self._act_connect.setChecked(False)
            self._act_relocate.setEnabled(False)
            self._act_locate.setChecked(False)
            self._act_locate.setEnabled(False)
            self._act_align.setChecked(False)
            self._act_align.setEnabled(False)
            self._act_rot_ccw.setEnabled(False)
            self._act_rot_cw.setEnabled(False)
            self._view.set_align_mode(False)
        self._rerender(fit=False)

    def _on_relocate(self) -> None:
        if self._link is None:
            return
        res = self._link.relocate()
        ok = bool(res.get("success"))
        self._status.showMessage(
            f"Relocate {'ok' if ok else 'failed'}: {res}", 5000)

    def _on_locate_armed(self, on: bool) -> None:
        # Set-location uses a left-click; turn off paint mode so the
        # click relocates instead of painting.
        if on and self._act_edit.isChecked():
            self._act_edit.setChecked(False)
        self._shared.set_locate_mode(on)

    def _on_locate(self, x_w: float, y_w: float) -> None:
        """Left-click while Set-location is armed → relocate-at."""
        if self._link is None:
            return False
        res = self._link.relocate_at(x_w, y_w)
        ok = bool(res.get("success"))
        self._status.showMessage(
            f"Set location ({x_w:.2f}, {y_w:.2f}) "
            f"{'ok' if ok else 'failed'}: {res}", 5000)
        if self._act_locate is not None:
            self._act_locate.setChecked(False)  # one-shot
        return True

    # ── Manual align ────────────────────────────────────────────────

    def _on_align_toggled(self, on: bool) -> None:
        if self._link is None:
            return
        if on:
            if self._act_edit.isChecked():
                self._act_edit.setChecked(False)
            base = self._live_pose or self._link.latest_pose()
            if base is None:
                self._status.showMessage(
                    "No pose yet — wait for the overlay before aligning.", 4000)
                self._act_align.setChecked(False)
                return
            self._overlay_pose = (float(base[0]), float(base[1]), float(base[2]))
            self._odom_anchor = self._link.odom_pose()
            self._view.set_align_mode(True)
            self._view.setFocus()  # so arrow-key nudge reaches the view
            self._status.showMessage(
                "Align: drag or arrow-keys (Shift=coarse) to move the scan "
                "onto a trusted wall; ⟲/⟳ (, .) to rotate. Odom dead-reckoned.")
        else:
            self._overlay_pose = None
            self._odom_anchor = None
            self._view.set_align_mode(False)

    def _on_align_drag(self, dx_w: float, dy_w: float) -> None:
        if self._overlay_pose is None:
            return
        x, y, th = self._overlay_pose
        self._overlay_pose = (x + dx_w, y + dy_w, th)

    def _on_rotate(self, sign: int) -> None:
        if self._overlay_pose is None:
            self._status.showMessage(
                "Turn on Align scan first to rotate the overlay.", 3000)
            return
        x, y, th = self._overlay_pose
        self._overlay_pose = (x, y, th + sign * math.radians(1.0))

    def _live_tick(self) -> None:
        if self._link is None or self._emap is None:
            return
        od = self._link.odom_pose()
        if self._overlay_pose is not None:
            # Manual-aligned pose, dead-reckoned by odom only (no
            # scan-match creep): advance by the odom delta since the
            # last tick, then re-anchor.
            if self._odom_anchor is not None and od is not None:
                self._overlay_pose = pose_compose(
                    self._overlay_pose, pose_relative(self._odom_anchor, od))
                self._odom_anchor = od
            pose = self._overlay_pose
            body = self._link.scan_body_xy(max_range_m=_SCAN_MAX_RANGE_M)
            world = body_xy_to_world(body, pose) if body is not None else None
            mode = "align"
        else:
            pose = self._link.latest_pose()
            world = self._link.latest_scan_world(max_range_m=_SCAN_MAX_RANGE_M)
            mode = "mcl"
        self._live_pose = pose
        self._view.set_scan_points(world)
        if self._drive_cache is not None:
            self._view.update_map(
                self._drive_cache, self._emap.meta, ts=time.time(),
                pose=pose, bounds_ij=self._emap.bounds_ij(),
            )
        age = self._link.scan_age_s(time.time())
        age_s = "scan —" if age is None else f"scan {age:.1f}s"
        n = 0 if world is None else len(world)
        self._status.showMessage(f"Live[{mode}] · {age_s} · {n} pts")
        if pose is None:
            self._pose_lbl.setText("pose: —")
        else:
            self._pose_lbl.setText(
                f"[{mode}] x={pose[0]:+.2f} y={pose[1]:+.2f} "
                f"θ={math.degrees(pose[2]):+6.1f}°")

    def _update_title(self) -> None:
        name = os.path.basename(self._path) if self._path else "(no map)"
        star = " *" if self._dirty else ""
        self.setWindowTitle(f"World Map Editor — {name}{star}")

    def _confirm_discard(self) -> bool:
        if not self._dirty:
            return True
        r = QMessageBox.question(
            self, "Discard changes?",
            "The current map has unsaved edits. Discard them?")
        return r == QMessageBox.StandardButton.Yes

    def closeEvent(self, event) -> None:  # noqa: N802 — Qt override
        if self._confirm_discard():
            self._teardown_link()
            event.accept()
        else:
            event.ignore()


def run(map_path: Optional[str] = None, *, router: Optional[str] = None,
        pf_device: str = "auto", pf_particles: int = 5000) -> int:
    app = QApplication.instance() or QApplication([])
    win = MapEditorWindow(map_path=map_path, router=router,
                          pf_device=pf_device, pf_particles=pf_particles)
    win.show()
    return app.exec()
