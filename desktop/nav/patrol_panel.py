"""Patrol dock for the nav shell.

Compact dock to define / load / save / clear the active patrol. The
canonical patrol lives on `SharedMapView` (rendered on the maps); this
panel reads it on `refresh()` and writes it back via `set_patrol()`
when the operator changes something. There's deliberately no separate
in-panel patrol model — one source of truth simplifies the round-
tripping with the map's right-click append.

UI surface (v1):

    Name:    [QLineEdit, editable when no mission]
    Loop:    [☑]    Laps: [QSpinBox, 0=unlimited]
    Session: <8-char authored sid> ●  match / mismatch chip
    Status:  <N waypoints>    [Edit on map: ☐]

    [ New ] [ Load ▼ ] [ Save ] [ Save As… ] [ Delete ]
    [ Clear all ] [ Remove last ]

While the mission is active, all controls except Edit-on-map are
disabled — patrols are immutable mid-run (per the locked design).
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QDockWidget, QHBoxLayout, QInputDialog, QLabel,
    QLineEdit, QMainWindow, QMenu, QMessageBox, QPushButton,
    QSpinBox, QVBoxLayout, QWidget,
)

from desktop.world_map.map_views import SharedMapView

from . import patrol as patrol_mod

logger = logging.getLogger(__name__)


class PatrolDock(QDockWidget):
    """Single dock — name + loop/laps + buttons. No waypoint table:
    operator inspects waypoints on the map (numbered pins). For v1
    that's adequate; a future table can land here when reorder /
    per-row edit becomes a felt need.
    """

    edit_mode_toggled = pyqtSignal(bool)

    def __init__(
        self,
        shared_view: SharedMapView,
        *,
        get_live_session_id,
        parent: Optional[QMainWindow] = None,
    ) -> None:
        super().__init__("Patrol", parent)
        self._shared = shared_view
        self._get_live_session_id = get_live_session_id  # callable () -> str
        self._mission_active: bool = False
        self._build()
        self._wire()
        # Initial sync — populate from whatever's currently on the
        # shared view (None / empty in fresh launches).
        self.refresh()

    # ── Build ────────────────────────────────────────────────────────

    def _build(self) -> None:
        host = QWidget(self)
        col = QVBoxLayout(host)
        col.setContentsMargins(8, 6, 8, 6)
        col.setSpacing(4)

        # Row 1: Name
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("(unsaved)")
        name_row.addWidget(self._name_edit, stretch=1)
        col.addLayout(name_row)

        # Row 2: Loop + Laps
        ll_row = QHBoxLayout()
        self._loop_box = QCheckBox("Loop")
        ll_row.addWidget(self._loop_box)
        ll_row.addSpacing(8)
        ll_row.addWidget(QLabel("Laps:"))
        self._laps_spin = QSpinBox()
        self._laps_spin.setRange(0, 9999)
        self._laps_spin.setValue(1)
        self._laps_spin.setToolTip(
            "Number of full circuits to drive. 0 = unlimited "
            "(operator cancels)."
        )
        ll_row.addWidget(self._laps_spin)
        ll_row.addStretch(1)
        col.addLayout(ll_row)

        # Row 3: session match hint + waypoint count
        self._session_lbl = QLabel("session: —")
        self._session_lbl.setStyleSheet("color: #ccc;")
        col.addWidget(self._session_lbl)

        self._status_lbl = QLabel("no patrol")
        self._status_lbl.setStyleSheet("color: #ccc;")
        col.addWidget(self._status_lbl)

        # Row 4: Edit-on-map toggle
        em_row = QHBoxLayout()
        self._edit_box = QCheckBox("Edit on map (right-click appends)")
        self._edit_box.setToolTip(
            "While on, right-click on any map appends a waypoint to "
            "the active patrol instead of setting a single goal."
        )
        em_row.addWidget(self._edit_box)
        em_row.addStretch(1)
        col.addLayout(em_row)

        # Row 5: library buttons (New, Load, Save, Save As, Delete)
        lib_row = QHBoxLayout()
        self._new_btn = QPushButton("New")
        self._load_btn = QPushButton("Load ▼")
        self._save_btn = QPushButton("Save")
        self._save_as_btn = QPushButton("Save As…")
        self._delete_btn = QPushButton("Delete")
        for b in (self._new_btn, self._load_btn, self._save_btn,
                  self._save_as_btn, self._delete_btn):
            lib_row.addWidget(b)
        col.addLayout(lib_row)

        # Row 6: edit buttons (Clear all, Remove last)
        ed_row = QHBoxLayout()
        self._clear_btn = QPushButton("Clear all")
        self._remove_last_btn = QPushButton("Remove last")
        ed_row.addWidget(self._clear_btn)
        ed_row.addWidget(self._remove_last_btn)
        ed_row.addStretch(1)
        col.addLayout(ed_row)

        self.setWidget(host)

    def _wire(self) -> None:
        self._name_edit.editingFinished.connect(self._on_name_edited)
        self._loop_box.toggled.connect(self._on_loop_toggled)
        self._laps_spin.valueChanged.connect(self._on_laps_changed)
        self._edit_box.toggled.connect(self._on_edit_mode_toggled)
        self._new_btn.clicked.connect(self._on_new)
        self._load_btn.clicked.connect(self._on_load)
        self._save_btn.clicked.connect(self._on_save)
        self._save_as_btn.clicked.connect(self._on_save_as)
        self._delete_btn.clicked.connect(self._on_delete)
        self._clear_btn.clicked.connect(self._on_clear_all)
        self._remove_last_btn.clicked.connect(self._on_remove_last)

    # ── Lifecycle / external API ─────────────────────────────────────

    def attach_to(self, window: QMainWindow) -> None:
        window.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self)
        self.setVisible(False)

    def set_visible(self, visible: bool) -> None:
        self.setVisible(visible)

    def is_visible(self) -> bool:
        return self.isVisible()

    def set_mission_active(self, active: bool) -> None:
        """Called by main_window when the mission goes active/terminal.
        Locks editing controls while the patrol is being driven."""
        if active == self._mission_active:
            return
        self._mission_active = active
        # Edit-on-map stays available even while running so the
        # operator can see what's checked — but mutating actions lock.
        for w in (
            self._name_edit, self._loop_box, self._laps_spin,
            self._new_btn, self._load_btn, self._save_btn,
            self._save_as_btn, self._delete_btn,
            self._clear_btn, self._remove_last_btn,
        ):
            w.setEnabled(not active)
        if active:
            # Mid-mission edits to the map shouldn't be possible
            # either — turn off the right-click-append mode and
            # uncheck the box.
            if self._edit_box.isChecked():
                self._edit_box.setChecked(False)
            self._edit_box.setEnabled(False)
        else:
            self._edit_box.setEnabled(True)

    def refresh(self) -> None:
        """Sync widgets to the current shared-view patrol. Cheap;
        called from the redraw tick. Idempotent — controls only get
        block-signals updates when the value actually differs."""
        p = self._shared.patrol()
        # Name
        cur_name = self._name_edit.text()
        target_name = "" if p is None else (p.name or "")
        if cur_name != target_name and not self._name_edit.hasFocus():
            blk = self._name_edit.blockSignals(True)
            self._name_edit.setText(target_name)
            self._name_edit.blockSignals(blk)
        # Loop / laps
        target_loop = bool(p.loop) if p else True
        if self._loop_box.isChecked() != target_loop:
            blk = self._loop_box.blockSignals(True)
            self._loop_box.setChecked(target_loop)
            self._loop_box.blockSignals(blk)
        target_laps = 0 if (p is None or p.laps is None) else int(p.laps)
        if self._laps_spin.value() != target_laps:
            blk = self._laps_spin.blockSignals(True)
            self._laps_spin.setValue(target_laps)
            self._laps_spin.blockSignals(blk)
        # Session match
        live_sid = self._get_live_session_id() or ""
        if p is None:
            self._session_lbl.setText("session: —")
            self._session_lbl.setStyleSheet("color: #ccc;")
        else:
            authored = (p.session_id or "")
            authored_short = authored[:8] if authored else "—"
            live_short = live_sid[:8] if live_sid else "—"
            if authored and live_sid and authored == live_sid:
                self._session_lbl.setText(
                    f"session: {authored_short} ✓ matches"
                )
                self._session_lbl.setStyleSheet("color: #8f8;")
            elif authored:
                self._session_lbl.setText(
                    f"session: {authored_short} ≠ live {live_short}"
                )
                self._session_lbl.setStyleSheet("color: #ec8;")
            else:
                self._session_lbl.setText(
                    f"session: (unset) — live {live_short}"
                )
                self._session_lbl.setStyleSheet("color: #ec8;")
        # Waypoint count
        if p is None:
            self._status_lbl.setText("no patrol — click New to start")
        else:
            n = len(p.waypoints)
            self._status_lbl.setText(
                f"{n} waypoint{'s' if n != 1 else ''}"
                + (" (loop)" if p.loop else " (open)")
            )

    # ── Slots ────────────────────────────────────────────────────────

    def _on_name_edited(self) -> None:
        p = self._shared.patrol()
        if p is None:
            return
        new_name = self._name_edit.text().strip()
        if new_name and new_name != p.name:
            p.name = new_name
            self._shared.set_patrol(p)

    def _on_loop_toggled(self, on: bool) -> None:
        p = self._shared.patrol()
        if p is None:
            return
        if p.loop != bool(on):
            p.loop = bool(on)
            self._shared.set_patrol(p)

    def _on_laps_changed(self, value: int) -> None:
        p = self._shared.patrol()
        if p is None:
            return
        new_laps = None if value <= 0 else int(value)
        if p.laps != new_laps:
            p.laps = new_laps
            self._shared.set_patrol(p)

    def _on_edit_mode_toggled(self, on: bool) -> None:
        self._shared.set_patrol_edit_mode(bool(on))
        self.edit_mode_toggled.emit(bool(on))

    def _on_new(self) -> None:
        # Confirm if discarding an unsaved patrol with waypoints.
        cur = self._shared.patrol()
        if cur is not None and len(cur.waypoints) > 0:
            ok = QMessageBox.question(
                self, "Discard current patrol?",
                f"Replace the active patrol ({len(cur.waypoints)} "
                f"waypoints) with a new empty one?",
            )
            if ok != QMessageBox.StandardButton.Yes:
                return
        live_sid = self._get_live_session_id() or ""
        p = patrol_mod.new_empty(session_id=live_sid)
        self._shared.set_patrol(p)
        # Convenience: enable edit-on-map so the operator can start
        # placing pins immediately.
        if not self._edit_box.isChecked():
            self._edit_box.setChecked(True)

    def _on_load(self) -> None:
        names = patrol_mod.list_library()
        if not names:
            QMessageBox.information(
                self, "No patrols saved",
                f"No patrols found in {patrol_mod.library_dir()}.",
            )
            return
        menu = QMenu(self)
        for n in names:
            menu.addAction(n, lambda checked=False, name=n: self._do_load(name))
        # Anchor under the Load button.
        global_pos = self._load_btn.mapToGlobal(
            self._load_btn.rect().bottomLeft()
        )
        menu.exec(global_pos)

    def _do_load(self, name: str) -> None:
        try:
            p = patrol_mod.load_from_library(name)
        except Exception as e:
            logger.exception("patrol load failed")
            QMessageBox.warning(
                self, "Load failed",
                f"Could not load patrol '{name}':\n{type(e).__name__}: {e}",
            )
            return
        self._shared.set_patrol(p)
        live_sid = self._get_live_session_id() or ""
        if p.session_id and live_sid and p.session_id != live_sid:
            QMessageBox.information(
                self, "Session mismatch",
                f"Patrol '{p.name}' was authored against session "
                f"{p.session_id[:8]}, but the live session is "
                f"{live_sid[:8]}. The patrol's waypoint coordinates "
                f"will be applied as-is; use Re-localize first if "
                f"the world frames may have shifted.",
            )

    def _on_save(self) -> None:
        p = self._shared.patrol()
        if p is None:
            QMessageBox.information(
                self, "Nothing to save", "No active patrol to save.",
            )
            return
        if not p.waypoints:
            QMessageBox.warning(
                self, "Empty patrol",
                "Add at least one waypoint before saving.",
            )
            return
        try:
            path = patrol_mod.save_to_library(p)
        except Exception as e:
            logger.exception("patrol save failed")
            QMessageBox.warning(
                self, "Save failed",
                f"Could not save patrol:\n{type(e).__name__}: {e}",
            )
            return
        QMessageBox.information(
            self, "Patrol saved",
            f"Wrote {path}",
        )

    def _on_save_as(self) -> None:
        p = self._shared.patrol()
        if p is None:
            return
        suggested = p.name or "patrol"
        new_name, ok = QInputDialog.getText(
            self, "Save patrol as", "Patrol name:", text=suggested,
        )
        if not ok or not new_name.strip():
            return
        p.name = new_name.strip()
        self._shared.set_patrol(p)
        self._on_save()

    def _on_delete(self) -> None:
        names = patrol_mod.list_library()
        if not names:
            QMessageBox.information(
                self, "Nothing to delete", "Library is empty.",
            )
            return
        name, ok = QInputDialog.getItem(
            self, "Delete patrol",
            f"Pick a patrol to delete from {patrol_mod.library_dir()}:",
            names, 0, False,
        )
        if not ok or not name:
            return
        confirm = QMessageBox.question(
            self, "Delete patrol",
            f"Delete '{name}' from the library? This cannot be undone.",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        if patrol_mod.delete_from_library(name):
            QMessageBox.information(
                self, "Deleted", f"Removed '{name}' from library.",
            )
        else:
            QMessageBox.warning(
                self, "Not found", f"'{name}' was already gone.",
            )

    def _on_clear_all(self) -> None:
        p = self._shared.patrol()
        if p is None:
            return
        if not p.waypoints:
            return
        confirm = QMessageBox.question(
            self, "Clear all waypoints?",
            f"Remove all {len(p.waypoints)} waypoints from the "
            f"active patrol?",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        p.clear()
        self._shared.set_patrol(p)

    def _on_remove_last(self) -> None:
        p = self._shared.patrol()
        if p is None or not p.waypoints:
            return
        p.remove_last()
        self._shared.set_patrol(p)
