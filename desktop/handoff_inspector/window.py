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
from PyQt6.QtCore import QTimer

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

class HandoffPanel(QWidget):
    """One tier column: Arm + Continue, a body-frame view, and a readout."""

    def __init__(self, tier: int, on_arm, on_continue, parent=None):
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
            self._view.update_data(grid, meta, sg)
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
    def __init__(self, session, *, record_prefix: str = RECORD_PREFIX,
                 ctrl_key: str = CTRL_KEY):
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
            self._panels[t] = HandoffPanel(t, self._send_arm, self._send_continue)
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
