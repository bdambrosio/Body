"""Detections dock + end-of-run review dialog (see desktop/nav/detections.py).

* ``DetectionsDock`` — what the robot noticed, newest first, with the VLM
  crop as a thumbnail. Click a row: the map centres on it and the crop shows
  below. Fed on the GUI thread by main_window (``ingest``).
* ``ReviewDialog`` — the run's findings plus the change store's places, with
  Approve (background: stop announcing) / Reject (always announce) / Forget.
  Replaces ``scripts/change_store.py`` for the review step.
"""
from __future__ import annotations

import datetime
import logging
import os
from typing import Any, Callable, Dict, List, Optional

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QDialog, QDockWidget, QHBoxLayout, QHeaderView, QLabel,
    QListWidget, QListWidgetItem, QMainWindow, QPushButton, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from desktop.world_map.map_views import SharedMapView

from .detections import INTERESTING_LABELS, Detection, DetectionModel

logger = logging.getLogger(__name__)

THUMB = QSize(64, 64)
_KIND_COLORS = {"new": QColor(255, 170, 40), "moving": QColor(60, 220, 255),
                "missing": QColor(200, 120, 255)}


def _pixmap(path: Optional[str]) -> Optional[QPixmap]:
    if not path or not os.path.exists(path):
        return None
    pm = QPixmap(path)
    return None if pm.isNull() else pm


def _placeholder(kind: str) -> QPixmap:
    """No photo: a small dot in the pin colour (same as the map)."""
    from PyQt6.QtGui import QPainter
    pm = QPixmap(THUMB)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(_KIND_COLORS.get(kind, QColor(160, 160, 160)))
    p.drawEllipse(THUMB.width() // 2 - 7, THUMB.height() // 2 - 7, 14, 14)
    p.end()
    return pm


def _fill_list(lst: QListWidget, rows: List[Detection], selected: Optional[int]) -> None:
    lst.blockSignals(True)
    lst.clear()
    for d in rows:
        pm = _pixmap(d.jpeg_path) or _placeholder(d.kind)
        it = QListWidgetItem(QIcon(pm.scaled(THUMB, Qt.AspectRatioMode.KeepAspectRatio,
                                             Qt.TransformationMode.SmoothTransformation)),
                             f"{d.title()}\n{d.detail()}")
        it.setData(Qt.ItemDataRole.UserRole, d.change_id)
        if d.what:
            it.setToolTip(d.what)
        if not d.present:
            it.setForeground(QColor(140, 140, 140))
        if d.alert:
            f = it.font()
            f.setBold(True)
            it.setFont(f)
        lst.addItem(it)
        if d.change_id == selected:
            it.setSelected(True)
            lst.setCurrentItem(it)
    lst.blockSignals(False)


def _make_list(parent: QWidget) -> QListWidget:
    lst = QListWidget(parent)
    lst.setIconSize(THUMB)
    lst.setUniformItemSizes(False)
    lst.setWordWrap(True)
    lst.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    return lst


class DetectionsDock(QDockWidget):
    review_requested = pyqtSignal()

    def __init__(self, model: DetectionModel, shared_view: SharedMapView,
                 parent: Optional[QMainWindow] = None) -> None:
        super().__init__("Detections", parent)
        self._model = model
        self._shared = shared_view
        self._shown_version = -1
        host = QWidget(self)
        v = QVBoxLayout(host)
        v.setContentsMargins(4, 4, 4, 4)
        top = QHBoxLayout()
        self._counts = QLabel("nothing yet", host)
        self._all = QCheckBox("Show all", host)
        self._all.setToolTip("Also show places the change store knows recur, and "
                             "'missing' walls (mostly grazing lidar beams)")
        top.addWidget(self._counts, 1)
        top.addWidget(self._all)
        v.addLayout(top)
        split = QSplitter(Qt.Orientation.Vertical, host)
        self._list = _make_list(split)
        self._preview = QLabel("select a detection", split)
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumHeight(120)
        self._preview.setWordWrap(True)
        split.addWidget(self._list)
        split.addWidget(self._preview)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        v.addWidget(split, 1)
        btns = QHBoxLayout()
        self._review = QPushButton("Review places…", host)
        self._review.setToolTip("Approve / reject recurring places in the change store")
        self._clear = QPushButton("Clear", host)
        btns.addWidget(self._review)
        btns.addStretch(1)
        btns.addWidget(self._clear)
        v.addLayout(btns)
        self.setWidget(host)
        self._all.toggled.connect(lambda _on: self.refresh(force=True))
        self._list.itemSelectionChanged.connect(self._on_select)
        self._review.clicked.connect(self.review_requested.emit)
        self._clear.clicked.connect(self._on_clear)

    def attach_to(self, window: QMainWindow) -> None:
        window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self)

    def refresh(self, force: bool = False) -> None:
        if not force and self._model.version == self._shown_version:
            return
        self._shown_version = self._model.version
        show_all = self._all.isChecked()
        rows = self._model.rows(show_all=show_all)
        _fill_list(self._list, rows, self._shared.selected_detection())
        self._shared.set_detections(rows)
        c = self._model.counts()
        self._counts.setText(f"{c['notable']} notable ({c['present']} present) "
                             f"of {c['total']} reports")
        self._show_preview(self._model.get(self._shared.selected_detection() or -1))

    def _on_select(self) -> None:
        items = self._list.selectedItems()
        if not items:
            return
        cid = int(items[0].data(Qt.ItemDataRole.UserRole))
        d = self._model.get(cid)
        self._shared.set_selected_detection(cid)
        if d is not None:
            self._shared.center_on(d.x, d.y)
        self._show_preview(d)

    def _show_preview(self, d: Optional[Detection]) -> None:
        if d is None:
            self._preview.setPixmap(QPixmap())
            self._preview.setText("select a detection")
            return
        pm = _pixmap(d.jpeg_path)
        if pm is None:
            self._preview.setPixmap(QPixmap())
            self._preview.setText(f"{d.title()}\n{d.detail()}\n(no photo — never in "
                                  "the camera's view while present)")
            return
        self._preview.setText("")
        self._preview.setPixmap(pm.scaled(
            self._preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))
        self._preview.setToolTip(f"{d.title()} — {d.what or ''}")

    def _on_clear(self) -> None:
        self._model.clear()
        self._shared.set_selected_detection(None)
        self.refresh(force=True)


class ReviewDialog(QDialog):
    """Run findings (top) + change-store places (bottom)."""

    COLS = ("id", "state", "kind", "x", "y", "size", "patrols", "reports", "label",
            "last seen", "this run")

    def __init__(self, model: DetectionModel, store: Any, shared_view: SharedMapView, *,
                 headline: Optional[str] = None, since: float = 0.0,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Explore review" if headline else "Review places")
        self.resize(900, 720)
        self._model = model
        self._store = store
        self._shared = shared_view
        v = QVBoxLayout(self)
        if headline:
            h = QLabel(headline, self)
            h.setWordWrap(True)
            f = h.font()
            f.setPointSizeF(f.pointSizeF() * 1.2)
            f.setBold(True)
            h.setFont(f)
            v.addWidget(h)
        split = QSplitter(Qt.Orientation.Vertical, self)
        # Findings
        fbox = QWidget(split)
        fv = QVBoxLayout(fbox)
        fv.setContentsMargins(0, 0, 0, 0)
        rows = model.rows(since=since)
        fv.addWidget(QLabel(f"Findings this run: {len(rows)}"
                            + ("" if rows else " — nothing new or moving worth a look"), fbox))
        frow = QHBoxLayout()
        self._findings = _make_list(fbox)
        self._photo = QLabel(fbox)
        self._photo.setMinimumWidth(260)
        self._photo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        frow.addWidget(self._findings, 2)
        frow.addWidget(self._photo, 1)
        fv.addLayout(frow, 1)
        _fill_list(self._findings, rows, None)
        self._findings.itemSelectionChanged.connect(self._on_finding)
        split.addWidget(fbox)
        # Places
        pbox = QWidget(split)
        pv = QVBoxLayout(pbox)
        pv.setContentsMargins(0, 0, 0, 0)
        ptop = QHBoxLayout()
        self._places_label = QLabel(pbox)
        self._show_all = QCheckBox("Include background / rejected", pbox)
        ptop.addWidget(self._places_label, 1)
        ptop.addWidget(self._show_all)
        pv.addLayout(ptop)
        self._table = QTableWidget(0, len(self.COLS), pbox)
        self._table.setHorizontalHeaderLabels(self.COLS)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._table.verticalHeader().setVisible(False)
        # No click-to-sort: the fill order (proposed first, most patrols
        # first) is the review order.
        pv.addWidget(self._table, 1)
        btns = QHBoxLayout()
        self._approve = QPushButton("Approve (background)", pbox)
        self._approve.setToolTip("This place routinely differs from the map (a door, chair "
                                 "legs): stop announcing reports here")
        self._reject = QPushButton("Reject (always report)", pbox)
        self._forget = QPushButton("Forget", pbox)
        self._forget.setToolTip("Drop the entry; it is learned again from scratch")
        close = QPushButton("Close", pbox)
        for b in (self._approve, self._reject, self._forget):
            btns.addWidget(b)
        btns.addStretch(1)
        btns.addWidget(close)
        pv.addLayout(btns)
        split.addWidget(pbox)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        v.addWidget(split, 1)
        self._show_all.toggled.connect(lambda _on: self._fill_places())
        self._table.itemSelectionChanged.connect(self._on_place)
        self._approve.clicked.connect(lambda: self._act("background"))
        self._reject.clicked.connect(lambda: self._act("rejected"))
        self._forget.clicked.connect(lambda: self._act(None))
        close.clicked.connect(self.accept)
        if store is None:
            for b in (self._approve, self._reject, self._forget, self._show_all):
                b.setEnabled(False)
        self._fill_places()

    # ── findings ────────────────────────────────────────────────────
    def _on_finding(self) -> None:
        items = self._findings.selectedItems()
        if not items:
            return
        d = self._model.get(int(items[0].data(Qt.ItemDataRole.UserRole)))
        if d is None:
            return
        self._shared.set_selected_detection(d.change_id)
        self._shared.center_on(d.x, d.y)
        self._set_photo(d)

    def _set_photo(self, d: Optional[Detection]) -> None:
        pm = _pixmap(d.jpeg_path) if d is not None else None
        if pm is None:
            self._photo.setPixmap(QPixmap())
            self._photo.setText("(no photo)" if d is not None else "")
            return
        self._photo.setText("")
        self._photo.setPixmap(pm.scaled(QSize(260, 320), Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.SmoothTransformation))

    # ── places ──────────────────────────────────────────────────────
    def _fill_places(self) -> None:
        self._table.setRowCount(0)
        if self._store is None:
            self._places_label.setText("No change store (nav started without a map path).")
            return
        states = ("proposed", "candidate") + (
            ("background", "rejected") if self._show_all.isChecked() else ())
        order = {s: i for i, s in enumerate(("proposed", "candidate", "background", "rejected"))}
        entries = sorted((e for e in list(self._store.entries.values()) if e.state in states),
                         key=lambda e: (order.get(e.state, 9), -len(e.sessions), e.eid))
        seen = self._model.by_store_entry()
        for e in entries:
            r = self._table.rowCount()
            self._table.insertRow(r)
            d = seen.get(e.eid)
            last = datetime.datetime.fromtimestamp(e.last_seen).strftime("%m-%d %H:%M") \
                if e.last_seen else ""
            vals = (e.eid, e.state, e.kind, round(e.x, 2), round(e.y, 2), round(e.extent_m, 2),
                    len(e.sessions), e.n_reports, e.label or "", last,
                    f"#{d.change_id}" + (" 📷" if d and d.jpeg_path else "") if d else "")
            for c, val in enumerate(vals):
                it = QTableWidgetItem()
                it.setData(Qt.ItemDataRole.DisplayRole, val)
                if c == 0:
                    it.setData(Qt.ItemDataRole.UserRole, e.eid)
                if e.state == "proposed":
                    it.setBackground(QColor(255, 236, 190))
                    it.setForeground(QColor(20, 20, 20))
                self._table.setItem(r, c, it)
        s = self._store.summary()
        self._places_label.setText(
            "Places that differ from the map — " + ", ".join(f"{v} {k}" for k, v in sorted(s.items()))
            + ".  Proposed = seen in enough patrols to suggest approving.")

    def _selected_eids(self) -> List[int]:
        rows = {i.row() for i in self._table.selectedItems()}
        out = []
        for r in sorted(rows):
            it = self._table.item(r, 0)
            if it is not None:
                out.append(int(it.data(Qt.ItemDataRole.UserRole)))
        return out

    def _on_place(self) -> None:
        eids = self._selected_eids()
        if not eids or self._store is None:
            return
        e = self._store.entries.get(eids[0])
        if e is None:
            return
        d = self._model.by_store_entry().get(e.eid)
        if d is not None:
            self._shared.set_selected_detection(d.change_id)
        self._shared.center_on(e.x, e.y)
        self._set_photo(d)

    def _act(self, state: Optional[str]) -> None:
        eids = self._selected_eids()
        if not eids or self._store is None:
            return
        if state == "background":
            risky = [e for e in (self._store.entries.get(i) for i in eids)
                     if e is not None and e.label in INTERESTING_LABELS]
            if risky:
                from PyQt6.QtWidgets import QMessageBox
                names = ", ".join(f"#{e.eid} ({e.label})" for e in risky)
                if QMessageBox.question(
                        self, "Approve as background?",
                        f"{names} looked like something that matters. Approving hides "
                        "future reports of the same size at that place.\n\nApprove anyway?"
                ) != QMessageBox.StandardButton.Yes:
                    return
        for eid in eids:
            ok = self._store.forget(eid) if state is None else self._store.set_state(eid, state)
            logger.info("change store: entry %d -> %s (%s)", eid, state or "forgotten",
                        "ok" if ok else "missing")
        self._fill_places()
