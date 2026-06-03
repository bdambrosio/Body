"""Standalone Handoff Inspector window.

Subscribes the three tier-handoff record topics (``drive/handoff/t{1,2,3}``)
and renders, for each, exactly what that tier is about to hand the next one
down. Per-tier Arm + Continue drive the breakpoints via ``drive/handoff/ctrl``;
the producers (desktop ``HierarchicalDrive`` for T1/T2, Pi ``local_drive`` for
T3) hold and single-step. Fully decoupled — this process only consumes records
and emits control.

The zenoh subscriber callbacks fire on zenoh threads; they only stash the
latest record under a lock. A QTimer renders on the Qt thread (the same
poll-latest pattern as the rest of the desktop UI).
"""
from __future__ import annotations

import math
import threading
import time
from typing import Any, Dict, Tuple

import numpy as np
from PyQt6.QtWidgets import (
    QCheckBox, QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF

from body.lib import zenoh_helpers
from body.lib.handoff_gate import CTRL_KEY, RECORD_PREFIX
from desktop.pi_drive.local_map_view import BodyLocalMapView

_DEFAULT_META = {"resolution_m": 0.08, "origin_x_m": -2.5, "origin_y_m": -2.5,
                 "nx": 64, "ny": 64}
_TITLES = {1: "HO-1  Tier-1 → Tier-2", 2: "HO-2  Tier-2 → Tier-3",
           3: "HO-3  Tier-3 → motors"}


# ── pure helpers (unit-tested without Qt) ────────────────────────────

def grid_and_meta(rec: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Decode the (optional) body-frame scan grid. When the record carried no
    grid (lean record / breakpoint disarmed), synthesize an all-unknown
    placeholder so the robot + overlay still render on an empty body frame."""
    meta = rec.get("meta") or _DEFAULT_META
    rows = rec.get("grid")
    if rows is not None:
        return np.array(rows, dtype=np.int8), meta
    n, m = int(meta.get("nx", 64)), int(meta.get("ny", 64))
    return np.full((n, m), -1, dtype=np.int8), meta


def format_record(rec: Dict[str, Any]) -> str:
    """A compact per-field readout of a handoff record (tier-specific)."""
    t = rec.get("tier")
    if t == 1:
        p = rec.get("pose", [0, 0, 0])
        wp = rec.get("wp", [0, 0])
        return (f"pose=({p[0]:.2f},{p[1]:.2f},{math.degrees(p[2]):.0f}°)\n"
                f"wp[{rec.get('wp_index')}/{rec.get('wp_total')}]="
                f"({wp[0]:.2f},{wp[1]:.2f})  "
                f"{'TERMINAL' if rec.get('terminal') else 'pass-through'}\n"
                f"bearing={math.degrees(rec.get('bearing_rad', 0)):.0f}°  "
                f"dist={rec.get('wp_dist_m', 0):.2f}  "
                f"tol={rec.get('arrival_tol_m', 0):.2f}")
    if t == 2:
        sg = rec.get("subgoal_body", [0, 0])
        return (f"src={rec.get('src')}  free={rec.get('free_dist_m', 0):.2f}\n"
                f"sub_body=({sg[0]:.2f},{sg[1]:.2f})  "
                f"bearing={math.degrees(rec.get('bearing_rad', 0)):.0f}°\n"
                f"tol={rec.get('arrival_tol_m', 0):.2f}  "
                f"cmd_id={rec.get('cmd_id', '—')}")
    if t == 3:
        g = rec.get("goal_body", [0, 0])
        return (f"plan={rec.get('plan_reason')}  "
                f"swept_blocked={rec.get('swept_blocked')}\n"
                f"v={rec.get('v_mps', 0):.3f}  ω={rec.get('omega_radps', 0):+.3f}\n"
                f"goal_body=({g[0]:.2f},{g[1]:.2f})  "
                f"cmd_id={rec.get('cmd_id', '—')}")
    return ""


# ── widgets ──────────────────────────────────────────────────────────

class WorldMapView(QWidget):
    """World-frame reference map (the Tier-1 global layer): the static map as a
    QImage cropped to its occupied bbox, plus the live robot pose + current
    waypoint + bearing from each HO-1 record."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 320)
        self._img = None                 # QImage of the cropped occupied region
        self._extent = None              # (x0, y0, x1, y1) world metres
        self._pose = None
        self._wp = None
        self._route = None               # full Tier-1 route [[x,y],...]
        self._wp_index = 0

    def set_map(self, drive, meta) -> None:
        g = np.asarray(drive)
        occ = g != -1
        if not occ.any():
            self._img = None
            self.update()
            return
        ii, jj = np.where(occ)
        imin, imax, jmin, jmax = int(ii.min()), int(ii.max()), int(jj.min()), int(jj.max())
        res = float(meta["resolution_m"])
        ox, oy = float(meta["origin_x_m"]), float(meta["origin_y_m"])
        crop = g[imin:imax + 1, jmin:jmax + 1]            # [i=x, j=y]
        gt = np.flipud(crop.T)                            # rows = y (top high), cols = x
        h, w = gt.shape
        rgb = np.empty((h, w, 3), dtype=np.uint8)
        rgb[...] = (18, 18, 18)                           # unknown / bg
        rgb[gt == 1] = (46, 92, 56)                       # clear
        rgb[gt == 0] = (200, 80, 80)                      # blocked (walls)
        self._img = QImage(rgb.tobytes(), w, h, 3 * w,
                           QImage.Format.Format_RGB888).copy()
        self._extent = (ox + imin * res, oy + jmin * res,
                        ox + (imax + 1) * res, oy + (jmax + 1) * res)
        self.update()

    def update_overlay(self, pose, wp, route=None, wp_index=0) -> None:
        self._pose = pose
        self._wp = wp
        self._route = route
        self._wp_index = int(wp_index)
        self.update()

    def _fit(self):
        if self._extent is None:
            return None
        x0, y0, x1, y1 = self._extent
        wm, hm = x1 - x0, y1 - y0
        if wm <= 0 or hm <= 0:
            return None
        m = 8
        ww, wh = self.width() - 2 * m, self.height() - 2 * m
        ppm = min(ww / wm, wh / hm)
        ox_px = m + (ww - wm * ppm) / 2
        oy_px = m + (wh - hm * ppm) / 2
        return ppm, ox_px, oy_px

    def _w2p(self, fit, wx, wy):
        ppm, ox_px, oy_px = fit
        x0, _y0, _x1, y1 = self._extent
        return ox_px + (wx - x0) * ppm, oy_px + (y1 - wy) * ppm   # flip y

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.fillRect(0, 0, self.width(), self.height(), QColor(12, 12, 12))
        if self._img is None:
            p.setPen(QColor(160, 160, 160))
            p.drawText(10, 18, "no map (pass --map)")
            return
        fit = self._fit()
        if fit is None:
            return
        ppm, ox_px, oy_px = fit
        x0, y0, x1, y1 = self._extent
        p.drawImage(QRectF(ox_px, oy_px, (x1 - x0) * ppm, (y1 - y0) * ppm), self._img)
        # Full Tier-1 route: visited segments dim, upcoming bright; a dot per
        # sub-waypoint. (Tier-1 builds this whole route; Tier-2 only gets the
        # current waypoint, drawn as the yellow crosshair below.)
        if self._route and len(self._route) >= 2:
            pts = [QPointF(*self._w2p(fit, w[0], w[1])) for w in self._route]
            for i, (a, b) in enumerate(zip(pts, pts[1:])):
                upcoming = i + 1 >= self._wp_index
                p.setPen(QPen(QColor(90, 200, 120) if upcoming
                              else QColor(90, 90, 90), 1.4))
                p.drawLine(a, b)
            for i, pt in enumerate(pts):
                p.setBrush(QColor(120, 210, 140) if i >= self._wp_index
                           else QColor(110, 110, 110))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawEllipse(pt, 2.2, 2.2)
        if self._pose is None:
            return
        rx, ry = self._w2p(fit, self._pose[0], self._pose[1])
        if self._wp is not None:
            wx, wy = self._w2p(fit, self._wp[0], self._wp[1])
            p.setPen(QPen(QColor(120, 200, 255, 160), 1, Qt.PenStyle.DashLine))
            p.drawLine(int(rx), int(ry), int(wx), int(wy))
            p.setPen(QPen(QColor(255, 220, 80), 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QPointF(wx, wy), 6, 6)
            p.drawLine(int(wx - 8), int(wy), int(wx + 8), int(wy))
            p.drawLine(int(wx), int(wy - 8), int(wx), int(wy + 8))
        self._draw_robot(p, rx, ry, float(self._pose[2]))

    def _draw_robot(self, p, rx, ry, th) -> None:
        c, s = math.cos(th), math.sin(th)

        def pt(fwd, left):                 # body (fwd, left) → screen (y flipped)
            wx, wy = fwd * c - left * s, fwd * s + left * c
            return QPointF(rx + wx, ry - wy)

        tri = QPolygonF([pt(11, 0), pt(-6, 6), pt(-6, -6)])
        p.setPen(QPen(QColor(255, 255, 255), 1))
        p.setBrush(QColor(255, 255, 255))
        p.drawPolygon(tri)


class HandoffPanel(QWidget):
    """One tier column: Arm + Continue, a body-frame view, and a readout."""

    def __init__(self, tier: int, on_arm, on_continue, world_map=None, parent=None):
        super().__init__(parent)
        self._tier = tier
        v = QVBoxLayout(self)
        head = QHBoxLayout()
        head.addWidget(QLabel(f"<b>{_TITLES[tier]}</b>"))
        self._arm = QCheckBox("Arm")
        self._arm.toggled.connect(lambda c: on_arm(tier, c))
        head.addWidget(self._arm)
        self._cont = QPushButton("Continue ▶")
        self._cont.clicked.connect(lambda: on_continue(tier))
        head.addWidget(self._cont)
        head.addStretch(1)
        v.addLayout(head)
        self._status = QLabel("waiting…")
        v.addWidget(self._status)
        # Tier-1 is a world-frame handoff → show the global reference map (when
        # one was loaded). Tier-2/Tier-3 are body-frame local maps.
        self._world = tier == 1 and world_map is not None
        if self._world:
            self._view = WorldMapView()
            self._view.set_map(world_map[0], world_map[1])
        else:
            self._view = BodyLocalMapView()
        v.addWidget(self._view, 1)
        self._readout = QLabel("—")
        self._readout.setStyleSheet("font-family: monospace;")
        self._readout.setWordWrap(True)
        v.addWidget(self._readout)

    def set_arm(self, checked: bool) -> None:
        self._arm.setChecked(checked)        # toggled signal emits the ctrl

    def update_from(self, rec, age_s: float) -> None:
        if rec is None:
            self._status.setText("waiting for a record…")
            return
        armed = self._arm.isChecked()
        self._status.setText(
            f"seq={rec.get('seq', '—')}  age={age_s:.1f}s"
            + ("   ⏸ ARMED (holding)" if armed else ""))
        if self._world:                       # Tier-1 world map (global layer)
            pose = rec.get("pose")
            wp = rec.get("wp")
            self._view.update_overlay(
                tuple(pose) if pose else None,
                tuple(wp) if wp else None,
                route=rec.get("route"), wp_index=rec.get("wp_index", 0))
            self._readout.setText(format_record(rec))
            return
        grid, meta = grid_and_meta(rec)
        t = self._tier
        if t == 1:
            brg = rec.get("bearing_rad", 0.0)
            d = rec.get("wp_dist_m", 1.0)
            self._view.update_data(grid, meta, None)
            self._view.set_planned_path(None)
            self._view.set_overlay((d * math.cos(brg), d * math.sin(brg)),
                                   None, brg, 0.0)
        elif t == 2:
            sg = tuple(rec.get("subgoal_body", (0.0, 0.0)))
            tb = rec.get("target_body")
            brg = rec.get("bearing_rad", 0.0)
            # goal_body=None: the sub-goal is drawn once, as the orange overlay
            # dot (not also as a blue goal-circle).
            self._view.update_data(grid, meta, None)
            self._view.set_planned_path(None)
            self._view.set_overlay(tuple(tb) if tb else None, sg, brg,
                                   rec.get("free_dist_m", 0.0))
        else:  # t == 3
            g = tuple(rec.get("goal_body", (0.0, 0.0)))
            self._view.update_data(grid, meta, g)
            self._view.set_overlay(None, None, None, 0.0)
            self._view.set_planned_path(rec.get("path_body"))
        self._readout.setText(format_record(rec))


class HandoffInspectorWindow(QMainWindow):
    def __init__(self, session, *, reference_map=None,
                 record_prefix: str = RECORD_PREFIX, ctrl_key: str = CTRL_KEY):
        super().__init__()
        self.setWindowTitle("Tier Handoff Inspector")
        self._session = session
        self._ctrl_key = ctrl_key
        self._lock = threading.Lock()
        self._latest: Dict[int, Any] = {1: None, 2: None, 3: None}
        self._recv: Dict[int, float] = {1: 0.0, 2: 0.0, 3: 0.0}
        self._subs = [
            zenoh_helpers.declare_subscriber_json(
                session, f"{record_prefix}/t{t}", self._make_handler(t))
            for t in (1, 2, 3)
        ]

        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        bar = QHBoxLayout()
        arm_all = QPushButton("Arm all")
        arm_all.clicked.connect(self._arm_all)
        run_free = QPushButton("Run free (disarm all)")
        run_free.clicked.connect(self._run_free)
        bar.addWidget(arm_all)
        bar.addWidget(run_free)
        bar.addStretch(1)
        outer.addLayout(bar)
        row = QHBoxLayout()
        outer.addLayout(row, 1)
        self._panels: Dict[int, HandoffPanel] = {}
        for t in (1, 2, 3):
            self._panels[t] = HandoffPanel(
                t, self._send_arm, self._send_continue,
                world_map=reference_map if t == 1 else None)
            row.addWidget(self._panels[t], 1)

        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

    # ── records in ──────────────────────────────────────────────────
    def _make_handler(self, tier: int):
        def handler(_key, msg):
            with self._lock:
                self._latest[tier] = msg
                self._recv[tier] = time.monotonic()
        return handler

    # ── control out ─────────────────────────────────────────────────
    def _send(self, tier: int, action: str) -> None:
        zenoh_helpers.publish_json(self._session, self._ctrl_key,
                                   {"tier": tier, "action": action})

    def _send_arm(self, tier: int, checked: bool) -> None:
        self._send(tier, "arm" if checked else "disarm")

    def _send_continue(self, tier: int) -> None:
        self._send(tier, "continue")

    def _arm_all(self) -> None:
        for panel in self._panels.values():
            panel.set_arm(True)

    def _run_free(self) -> None:
        for panel in self._panels.values():
            panel.set_arm(False)

    # ── render ──────────────────────────────────────────────────────
    def _refresh(self) -> None:
        now = time.monotonic()
        with self._lock:
            latest = dict(self._latest)
            recv = dict(self._recv)
        for t, panel in self._panels.items():
            rec = latest[t]
            panel.update_from(rec, (now - recv[t]) if rec is not None else 0.0)

    def closeEvent(self, event) -> None:
        try:
            self._timer.stop()
        except Exception:
            pass
        super().closeEvent(event)
