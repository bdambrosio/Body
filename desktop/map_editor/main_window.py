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
from collections import deque
from dataclasses import replace
from typing import List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QActionGroup
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QLabel, QMainWindow, QMessageBox,
    QSpinBox, QToolBar, QWidget,
)

from desktop.localization.checkpoint_matcher import (
    CheckpointMatchConfig,
    CheckpointMatcher,
    crop_disk,
)
from desktop.localization.checkpoints import (
    checkpoints_from_metadata,
    upsert_checkpoint,
    write_checkpoints_to_metadata,
)
from desktop.localization.raycast_match import RaycastConfig, score_pose
from desktop.world_map.map_views import SharedMapView

from . import editor_map as em
from .editable_map_view import EditableMapView
from .live_overlay import body_xy_to_world, pose_compose, pose_relative

logger = logging.getLogger(__name__)

_UNDO_DEPTH = 25
_SCAN_MAX_RANGE_M = 12.0
# Recognize re-stamps observed occupancy within this radius of the asserted
# pose, rebuilt from the last few odom-stitched scans (Phase 1 of the
# topological-localization plan; see docs/topological_localization_design.md).
_RECOGNIZE_RADIUS_M = 2.0

# Which edit layer each brush kind writes.
_KIND_LAYER = {
    em.WALL: "occ", em.FREE: "occ", em.UNKNOWN: "occ",
    em.NOGO: "nogo", em.ERASE_NOGO: "nogo",
}


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
        self._active_layer: str = "occ"  # "occ" | "nogo"
        self._undo: List = []  # stack of (log_odds, nogo) snapshots
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
        # Latest world-frame live-scan endpoints (what the cyan dots show).
        self._live_scan_world: Optional[np.ndarray] = None
        # Recent (world_scan, pose) snapshots in the *asserted* (align) frame,
        # for Recognize's odom-stitched multi-scan re-stamp. Filled only in
        # align mode; cleared on any frame discontinuity (re-assert / rotate).
        self._scan_ring: deque = deque(maxlen=16)

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
        # Live checkpoint-recognition readout (nearest cp + inlier/short).
        self._match_lbl = QLabel("")
        self._match_lbl.setStyleSheet("color:#888; font-family:monospace;")
        self._status.addPermanentWidget(self._match_lbl)

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

        # Edit-layer selector: Occupancy (perception → localization +
        # planning) vs No-go (policy → planning only).
        layer_grp = QActionGroup(self)
        layer_grp.setExclusive(True)
        self._act_layer_occ = tb.addAction("Occupancy")
        self._act_layer_occ.setCheckable(True)
        self._act_layer_occ.setChecked(True)
        self._act_layer_occ.triggered.connect(
            lambda _c=False: self._set_layer("occ"))
        layer_grp.addAction(self._act_layer_occ)
        self._act_layer_nogo = tb.addAction("No-go")
        self._act_layer_nogo.setCheckable(True)
        self._act_layer_nogo.triggered.connect(
            lambda _c=False: self._set_layer("nogo"))
        layer_grp.addAction(self._act_layer_nogo)
        tb.addSeparator()

        # Brush palette — one exclusive group spanning both layers; only
        # the active layer's kinds are visible (see _set_layer).
        grp = QActionGroup(self)
        grp.setExclusive(True)
        self._brush_actions = {}
        for kind, label in ((em.WALL, "Wall"), (em.FREE, "Free"),
                            (em.UNKNOWN, "Unknown"),
                            (em.NOGO, "Paint"), (em.ERASE_NOGO, "Erase")):
            a = tb.addAction(label)
            a.setCheckable(True)
            a.triggered.connect(lambda _c=False, k=kind: self._set_kind(k))
            a.setVisible(_KIND_LAYER[kind] == "occ")  # start on occupancy
            grp.addAction(a)
            self._brush_actions[kind] = a
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
        tb.addSeparator()

        # Checkpoint management — works offline (no live link needed).
        # The combo mirrors the rings drawn on the map; Delete removes the
        # selected checkpoint from the map metadata (Save persists it).
        tb.addWidget(QLabel(" Checkpoint "))
        self._cp_combo = QComboBox()
        self._cp_combo.setToolTip(
            "Checkpoints stored in this map (Recognize adds them).")
        tb.addWidget(self._cp_combo)
        self._act_del_cp = tb.addAction("Delete cp", self._on_delete_checkpoint)
        self._act_del_cp.setToolTip(
            "Delete the selected checkpoint from the map metadata. The healed "
            "occupancy stays — only the match anchor is removed. Save to "
            "persist.")

        # Live (Phase 2) controls — only when a router is configured.
        self._act_connect = None
        self._act_relocate = None
        self._act_locate = None
        self._act_recognize = None
        self._act_test_match = None
        if self._router:
            # Live controls live on a SECOND toolbar row so the top row
            # doesn't overflow the window width and bury buttons in the
            # ">>" overflow menu.
            self.addToolBarBreak()
            tb2 = QToolBar("live")
            tb2.setMovable(False)
            self.addToolBar(tb2)
            # NB: drive Connect from `toggled` (passes the checked bool),
            # not addAction's `triggered` (which calls the slot with no
            # args → _on_connect would lose its `want`).
            self._act_connect = tb2.addAction("Connect")
            self._act_connect.setCheckable(True)
            self._act_connect.toggled.connect(self._on_connect)
            self._act_relocate = tb2.addAction("Relocate", self._on_relocate)
            self._act_relocate.setEnabled(False)
            self._act_locate = tb2.addAction("Set location")
            self._act_locate.setCheckable(True)
            self._act_locate.setEnabled(False)
            self._act_locate.toggled.connect(self._on_locate_armed)
            tb2.addSeparator()
            # Manual align: drag the scan onto trusted walls, rotate with
            # the buttons. Dead-reckoned by odom (no scan-match creep).
            self._act_align = tb2.addAction("Align scan")
            self._act_align.setCheckable(True)
            self._act_align.setEnabled(False)
            self._act_align.toggled.connect(self._on_align_toggled)
            self._act_rot_ccw = tb2.addAction("⟲")
            self._act_rot_ccw.setToolTip("Rotate scan +1° (key: ,)")
            self._act_rot_ccw.setShortcut(",")
            self._act_rot_ccw.triggered.connect(lambda: self._on_rotate(+1))
            self._act_rot_cw = tb2.addAction("⟳")
            self._act_rot_cw.setToolTip("Rotate scan −1° (key: .)")
            self._act_rot_cw.setShortcut(".")
            self._act_rot_cw.triggered.connect(lambda: self._on_rotate(-1))
            for a in (self._act_rot_ccw, self._act_rot_cw):
                a.setEnabled(False)
            tb2.addSeparator()
            # Recognize: heal the map locally so the asserted pose scores best
            # here (replace observed occupancy within a radius from a few
            # odom-stitched scans). Requires an asserted pose (Align/Set loc).
            self._act_recognize = tb2.addAction("Recognize")
            self._act_recognize.setToolTip(
                f"Heal the map here: replace occupancy within "
                f"{_RECOGNIZE_RADIUS_M:.0f} m of the asserted pose (Align scan / "
                "Set location) from the live scan, so this pose scores best "
                "here. Edits the localization map.")
            self._act_recognize.setEnabled(False)
            self._act_recognize.triggered.connect(self._on_recognize)
            # Validate recognition without wiring nav: run the matcher at the
            # current pose and report the correction it would apply.
            self._act_test_match = tb2.addAction("Test match")
            self._act_test_match.setToolTip(
                "Run the checkpoint matcher at the current pose against the "
                "live scan and report the correction it would apply.")
            self._act_test_match.setEnabled(False)
            self._act_test_match.triggered.connect(self._on_test_match)

    def _set_kind(self, kind: str) -> None:
        self._brush_kind = kind

    def _set_layer(self, layer: str) -> None:
        """Switch the active edit layer; show that layer's brushes and
        select its default kind (Wall / no-go Paint)."""
        self._active_layer = layer
        default = em.WALL if layer == "occ" else em.NOGO
        for kind, a in self._brush_actions.items():
            a.setVisible(_KIND_LAYER[kind] == layer)
        self._brush_actions[default].setChecked(True)
        self._set_kind(default)
        if layer == "nogo":
            self._status.showMessage(
                "No-go layer (orange): paint keep-out zones. Planning only "
                "— does not affect localization.", 5000)
        else:
            self._status.showMessage(
                "Occupancy layer: Wall / Free / Unknown.", 3000)

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
        self._undo.append(self._emap.snapshot_state())
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
        self._emap.restore_state(self._undo.pop())
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
        self._push_nogo_overlay()
        self._refresh_checkpoint_markers()

    def _refresh_checkpoint_markers(self) -> None:
        """Draw the saved checkpoints (rings + ids) on the map and mirror
        them into the toolbar combo."""
        if self._emap is None:
            return
        cps = checkpoints_from_metadata(self._emap.metadata)
        self._shared.set_checkpoints(
            [(c.x_m, c.y_m, c.radius_m, c.id) for c in cps])
        # Rebuild the combo only when the set actually changed — this runs
        # on every rerender (paint strokes included) and a no-op rebuild
        # would clobber the operator's selection.
        labels = [f"{c.id}  ({c.x_m:+.2f}, {c.y_m:+.2f})" for c in cps]
        current = [self._cp_combo.itemText(i)
                   for i in range(self._cp_combo.count())]
        if labels != current:
            selected = self._cp_combo.currentData()
            self._cp_combo.clear()
            for c, lbl in zip(cps, labels):
                self._cp_combo.addItem(lbl, c.id)
            if selected is not None:
                idx = self._cp_combo.findData(selected)
                if idx >= 0:
                    self._cp_combo.setCurrentIndex(idx)
        enable = bool(cps)
        self._cp_combo.setEnabled(enable)
        self._act_del_cp.setEnabled(enable)

    def _on_delete_checkpoint(self) -> None:
        """Remove the combo-selected checkpoint from the map metadata."""
        if self._emap is None:
            return
        cp_id = self._cp_combo.currentData()
        if cp_id is None:
            return
        r = QMessageBox.question(
            self, "Delete checkpoint?",
            f"Delete {cp_id} from this map?\n\nThe healed occupancy stays — "
            "only the match anchor is removed. Save to persist.")
        if r != QMessageBox.StandardButton.Yes:
            return
        cps = [c for c in checkpoints_from_metadata(self._emap.metadata)
               if c.id != cp_id]
        write_checkpoints_to_metadata(self._emap.metadata, cps)
        self._dirty = True
        self._refresh_checkpoint_markers()
        self._update_title()
        self._status.showMessage(f"Deleted {cp_id} — Save to persist.", 5000)

    def _push_nogo_overlay(self) -> None:
        """Hand the keep-out cell centers (world frame) to the view for the
        orange overlay, or clear it when the mask is empty."""
        if self._emap is None:
            return
        mask = self._emap.nogo
        res = self._emap.resolution_m
        if mask is None or not mask.any():
            self._view.set_nogo_cells(None, res)
            return
        ii, jj = np.where(mask)
        cx = self._emap.origin_x_m + (ii + 0.5) * res
        cy = self._emap.origin_y_m + (jj + 0.5) * res
        self._view.set_nogo_cells(np.column_stack([cx, cy]), res)

    def _refresh_actions(self) -> None:
        have = self._emap is not None
        for a in (self._act_save, self._act_save_as, self._act_edit):
            a.setEnabled(have)
        self._act_undo.setEnabled(have and bool(self._undo))
        if not have:
            self._cp_combo.clear()
            self._cp_combo.setEnabled(False)
            self._act_del_cp.setEnabled(False)

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
            self._act_recognize.setEnabled(True)
            self._act_test_match.setEnabled(True)
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
        self._live_scan_world = None
        self._scan_ring.clear()
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
            self._act_recognize.setEnabled(False)
            self._act_test_match.setEnabled(False)
            self._match_lbl.setText("")
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
        self._scan_ring.clear()   # believed pose moved
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
            self._scan_ring.clear()   # new asserted frame
            self._view.set_align_mode(True)
            self._view.setFocus()  # so arrow-key nudge reaches the view
            self._status.showMessage(
                "Align: drag or arrow-keys (Shift=coarse) to move the scan "
                "onto a trusted wall; ⟲/⟳ (, .) to rotate. Odom dead-reckoned.")
        else:
            self._overlay_pose = None
            self._odom_anchor = None
            self._scan_ring.clear()
            self._view.set_align_mode(False)

    def _on_align_drag(self, dx_w: float, dy_w: float) -> None:
        if self._overlay_pose is None:
            return
        x, y, th = self._overlay_pose
        self._overlay_pose = (x + dx_w, y + dy_w, th)
        self._scan_ring.clear()   # pose jumped → old-frame scans are stale

    def _on_rotate(self, sign: int) -> None:
        if self._overlay_pose is None:
            self._status.showMessage(
                "Turn on Align scan first to rotate the overlay.", 3000)
            return
        x, y, th = self._overlay_pose
        self._overlay_pose = (x, y, th + sign * math.radians(1.0))
        self._scan_ring.clear()   # pose jumped → old-frame scans are stale

    def _recent_distinct_scans(
        self, max_scans: int = 3, min_step_m: float = 0.03,
    ) -> List[Tuple[np.ndarray, Tuple[float, float, float]]]:
        """Up to `max_scans` recent (world_scan, pose) from the ring, newest
        first, spaced ≥ min_step_m in xy so they give distinct viewpoints. A
        stationary bot yields just the latest."""
        out: List[Tuple[np.ndarray, Tuple[float, float, float]]] = []
        last_xy: Optional[Tuple[float, float]] = None
        for world, pose in reversed(self._scan_ring):
            if last_xy is not None and math.hypot(
                pose[0] - last_xy[0], pose[1] - last_xy[1]) < min_step_m:
                continue
            out.append((world, pose))
            last_xy = (pose[0], pose[1])
            if len(out) >= max_scans:
                break
        return out

    def _on_recognize(self) -> None:
        """Heal the map locally so the asserted pose scores best here:
        replace observed occupancy within `_RECOGNIZE_RADIUS_M` of the
        asserted pose, rebuilt from the last few odom-stitched scans.
        Requires an asserted pose (Align scan / Set location). Edits the
        localization map — confirms first."""
        if self._emap is None or self._link is None:
            return
        if self._overlay_pose is None:
            self._status.showMessage(
                "Recognize needs an asserted pose — turn on Align scan and "
                "line the scan up to the real walls first.", 5000)
            return
        scans = self._recent_distinct_scans()
        if not scans:
            self._status.showMessage("No live scan to recognize yet.", 3000)
            return
        pose = self._overlay_pose
        r = QMessageBox.question(
            self, "Recognize here?",
            f"Replace occupancy within {_RECOGNIZE_RADIUS_M:.1f} m of the "
            f"asserted pose, rebuilt from {len(scans)} scan(s), so this becomes "
            "the best-scoring pose here?\n\nThis edits the LOCALIZATION map — "
            "verify the scan matches the real room first.")
        if r != QMessageBox.StandardButton.Yes:
            return
        self._undo.append(self._emap.snapshot_state())
        if len(self._undo) > _UNDO_DEPTH:
            self._undo.pop(0)
        self._act_undo.setEnabled(True)
        n = self._emap.restamp_from_scans(
            [(w, (p[0], p[1])) for (w, p) in scans],
            center_xy=(pose[0], pose[1]),
            radius_m=_RECOGNIZE_RADIUS_M,
        )
        # Persist this spot as an LPR checkpoint (add, or correct a nearby one).
        cps = checkpoints_from_metadata(self._emap.metadata)
        cps, cp = upsert_checkpoint(
            cps, pose, _RECOGNIZE_RADIUS_M, created_ts=time.time())
        write_checkpoints_to_metadata(self._emap.metadata, cps)
        self._dirty = True
        self._rerender(fit=False)
        self._update_title()
        self._status.showMessage(
            f"Recognized {cp.id}: re-stamped {n} cell(s) within "
            f"{_RECOGNIZE_RADIUS_M:.1f} m from {len(scans)} scan(s).", 5000)

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
        self._live_scan_world = world
        # In align mode the overlay is odom-dead-reckoned from the asserted
        # anchor, so consecutive scans share one consistent world frame —
        # exactly the odom-stitched set Recognize wants. (MCL-mode poses can
        # jump, so don't collect those.)
        if mode == "align" and world is not None and pose is not None:
            self._scan_ring.append((world, pose))
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
        self._update_checkpoint_readout(pose)

    def _scan_angles_ranges(self):
        """Live scan as (angles_rad, ranges_m) in the body frame, or None."""
        body = self._link.scan_body_xy(max_range_m=_SCAN_MAX_RANGE_M)
        if body is None or len(body) == 0:
            return None
        return np.arctan2(body[:, 1], body[:, 0]), np.hypot(body[:, 0], body[:, 1])

    def _update_checkpoint_readout(self, pose) -> None:
        """Score the live scan at the current pose against the nearest
        checkpoint patch — a cheap, continuous "is this spot recognized?"
        signal (inlier high ✓ → the healed map fits here)."""
        if self._emap is None or self._link is None or pose is None:
            self._match_lbl.setText("")
            return
        cps = checkpoints_from_metadata(self._emap.metadata)
        if not cps:
            self._match_lbl.setText("")
            return
        near = min(cps, key=lambda c: math.hypot(c.x_m - pose[0], c.y_m - pose[1]))
        if math.hypot(near.x_m - pose[0], near.y_m - pose[1]) > near.radius_m:
            self._match_lbl.setText(f"cp: none ≤{near.radius_m:.0f}m")
            self._match_lbl.setStyleSheet("color:#888; font-family:monospace;")
            return
        ar = self._scan_angles_ranges()
        if ar is None:
            self._match_lbl.setText("")
            return
        occ = self._emap.log_odds > 0.0
        sub, sox, soy = crop_disk(
            occ, self._emap.origin_x_m, self._emap.origin_y_m,
            self._emap.resolution_m, (near.x_m, near.y_m), near.radius_m)
        if sub.size == 0:
            self._match_lbl.setText("")
            return
        rc = replace(RaycastConfig(),
                     max_range_m=min(RaycastConfig().max_range_m, near.radius_m))
        s = score_pose(sub, sox, soy, self._emap.resolution_m, pose, ar[0], ar[1], rc)
        ok = s.inlier_frac >= 0.6 and s.short_frac <= 0.25
        self._match_lbl.setText(
            f"{near.id} {'✓' if ok else '·'} in={s.inlier_frac:.2f} sh={s.short_frac:.2f}")
        self._match_lbl.setStyleSheet(
            f"color:{'#8f8' if ok else '#fb6'}; font-family:monospace;")

    def _on_test_match(self) -> None:
        """Full checkpoint match at the current pose — reports the correction
        the runtime localizer would apply (validates recognition end-to-end)."""
        if self._emap is None or self._link is None:
            return
        pose = self._live_pose
        cps = checkpoints_from_metadata(self._emap.metadata)
        if not cps or pose is None:
            self._status.showMessage("No checkpoints / no pose to test.", 4000)
            return
        ar = self._scan_angles_ranges()
        if ar is None:
            self._status.showMessage("No live scan to test.", 4000)
            return
        occ = self._emap.log_odds > 0.0
        matcher = CheckpointMatcher(
            occ, self._emap.origin_x_m, self._emap.origin_y_m,
            self._emap.resolution_m, cps, CheckpointMatchConfig())
        m = matcher.match(pose, ar[0], ar[1])
        if m is None:
            self._status.showMessage(
                "Test match: NO checkpoint accepted here (none near, or the "
                "scan doesn't fit the healed map).", 7000)
            return
        dx, dy = m.pose[0] - pose[0], m.pose[1] - pose[1]
        dth = math.degrees(math.atan2(
            math.sin(m.pose[2] - pose[2]), math.cos(m.pose[2] - pose[2])))
        self._status.showMessage(
            f"Test match: {m.checkpoint_id} inlier={m.inlier_frac:.2f} "
            f"short={m.short_frac:.2f} → correction "
            f"dx={dx:+.2f} dy={dy:+.2f} dθ={dth:+.1f}°", 8000)

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
