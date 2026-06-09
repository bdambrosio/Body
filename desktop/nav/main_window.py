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
import os
import shutil
import time
from typing import Any, Dict, Optional

import numpy as np

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication, QDockWidget, QFileDialog, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QSplitter, QToolBar, QVBoxLayout, QWidget,
)
from PyQt6.QtGui import QAction, QDesktopServices
from PyQt6.QtCore import QUrl

from desktop.chassis.config import StubConfig
from desktop.chassis.controller import StubController
from desktop.localization.config import LocalizationConfig
from desktop.localization.controller import LocalizationController
from desktop.world_map.costmap import CostmapConfig, build_costmap
from desktop.world_map.map_views import (
    SharedMapView, WorldCostmapView, WorldDriveableView, WorldHeightView,
)
from .health import LivenessWatcher
from .hierarchical_drive import (
    HierarchicalDrive, HierConfig, HierState, PFPoseProvider,
)
from . import patrol as patrol_mod
from .patrol import PatrolRunner
from .patrol_expand import ExpandConfig, expand_patrol
from .patrol_panel import PatrolDock
from .planner import AStarConfig, PlanResult, plan_path
from .pose_health import PoseHealthMonitor
from .tracing import CAT_PLAN, LEVEL_WARN, Tracer

from .camera_panels import CameraPanels, build_camera_snapshot
from .safety_toolbar import SafetyToolbar
from .teleop_panels import TeleopPanels, build_chassis_snapshot

from body.lib.local_drive_core import body_to_odom
from desktop.localization.checkpoint_localizer import (
    CheckpointLocalizer,
    CheckpointPoseProvider,
)
from desktop.localization.checkpoint_matcher import (
    CheckpointMatchConfig,
    CheckpointMatcher,
)
from desktop.localization.checkpoints import checkpoints_from_metadata
from desktop.nav.slam.scan_matcher import lidar_scan_to_xy
from desktop.pi_drive.drive_client import DriveClient
from desktop.chassis.transport import open_session
from body.lib.handoff_gate import HandoffGate

# Tight, odom-primed search window for the runtime checkpoint re-anchor (the
# drift between throttled re-anchors is small, so the window can be small →
# fast enough to run inline). Tunable.
_RUNTIME_CP_CFG = CheckpointMatchConfig(
    xy_half_m=0.15, xy_step_m=0.06,
    theta_half_rad=math.radians(9.0), theta_step_rad=math.radians(3.0),
)

logger = logging.getLogger(__name__)


class NavMainWindow(QMainWindow):
    def __init__(
        self,
        fuser: LocalizationController,
        fuser_config: LocalizationConfig,
        chassis: StubController,
        chassis_config: StubConfig,
        *,
        use_checkpoint_pose: bool = False,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Body Nav")
        self.fuser = fuser
        self.fuser_config = fuser_config
        self.chassis = chassis
        self.chassis_config = chassis_config
        # EXPERIMENTAL: hierarchical drive localizes via odom dead-reckon +
        # checkpoint re-anchor instead of the PF posterior. Off by default;
        # read at each Go. The active localizer is kept for a status readout.
        self._use_checkpoint_pose: bool = bool(use_checkpoint_pose)
        self._cp_localizer: Optional[CheckpointLocalizer] = None

        # Hierarchical drive (Tier-1/Tier-2/Tier-3) is the production drive
        # path — Go/Stop always route through it. The DriveClient (own zenoh
        # session, body/drive/goto + status + scan) is opened lazily on the
        # first Go. Initialized before _build_toolbars so the ALL-STOP
        # callback can reference it safely.
        self._hier_drive: Optional[HierarchicalDrive] = None
        self._drive_client: Optional[DriveClient] = None
        # Handoff inspector seam: a dedicated zenoh session carrying the HO-1/
        # HO-2 records + arm/continue, opened lazily on the first Go.
        self._handoff_session: Optional[Any] = None
        self._handoff_gate: Optional[HandoffGate] = None

        self._build_toolbars()
        self._build_ui()
        self._build_docks()
        self._build_menu()
        self._build_timer()
        self._show_map_checkpoints()

    def _show_map_checkpoints(self) -> None:
        """Draw the loaded map's LPR checkpoints (purple rings) on the maps."""
        try:
            cps = checkpoints_from_metadata(self.fuser.reference_map.metadata)
            self._shared_view.set_checkpoints(
                [(c.x_m, c.y_m, c.radius_m, c.id) for c in cps])
        except Exception:
            logger.exception("failed to load map checkpoints for overlay")

    # ── Layout ───────────────────────────────────────────────────────

    def _build_toolbars(self) -> None:
        self._safety_toolbar = SafetyToolbar(self.chassis, self.fuser, parent=self)
        # ALL-STOP must cancel an in-flight hierarchical-drive goto too.
        self._safety_toolbar.set_stop_callback(self._on_all_stop)
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

        relocate_act = QAction("Re-localize", self)
        relocate_act.setToolTip(
            "Snap the robot pose to a wide global scan-match against "
            "the current world map. Use after the steady-state matcher "
            "has diverged (huge SLAM drift); the map is kept."
        )
        relocate_act.triggered.connect(self._on_relocate)
        self._map_toolbar.addAction(relocate_act)

        save_act = QAction("Save snapshot", self)
        save_act.setToolTip(
            "Write a self-contained snapshot bundle (layers.npz, "
            "PNGs, summary.json) for offline inspection. Path is "
            "shown on completion; default ~/Body/sessions/<sid>/."
        )
        save_act.triggered.connect(self._on_save_snapshot)
        self._map_toolbar.addAction(save_act)

        load_act = QAction("Load snapshot", self)
        load_act.setToolTip(
            "Restore a saved layers.npz into the world grid as a "
            "prior. Vote decay continues normally — re-observed "
            "cells stay confident, unobserved cells gradually fade "
            "toward the floor."
        )
        load_act.triggered.connect(self._on_load_snapshot)
        self._map_toolbar.addAction(load_act)

        fit_act = QAction("Fit maps", self)
        fit_act.setToolTip(
            "Reset map zoom/pan to auto-fit the populated region. "
            "(Shortcut: double-click a map.)"
        )
        fit_act.triggered.connect(self._on_fit_maps)
        self._map_toolbar.addAction(fit_act)

        clear_goal_act = QAction("Clear goal", self)
        clear_goal_act.setToolTip(
            "Remove the current goal pin and any planned path. "
            "Right-click a map to set a new goal."
        )
        clear_goal_act.triggered.connect(self._on_clear_goal)
        self._map_toolbar.addAction(clear_goal_act)

        self._go_act = QAction("Go", self)
        self._go_act.setToolTip(
            "Drive the loaded patrol with hierarchical drive: each leg routes "
            "toward a live-observed sub-goal via the Pi's Tier-3 loop "
            "(body/drive/goto). Place waypoints (Patrol edit) first. Tier-3 "
            "owns cmd_vel — leave Live cmd OFF; nav keeps the heartbeat alive."
        )
        self._go_act.triggered.connect(self._on_go)
        self._map_toolbar.addAction(self._go_act)

        self._cancel_act = QAction("Stop", self)
        self._cancel_act.setToolTip(
            "Stop the hierarchical drive and cancel the in-flight Tier-3 goto."
        )
        self._cancel_act.triggered.connect(self._on_cancel)
        self._map_toolbar.addAction(self._cancel_act)

        self._resume_act = QAction("Resume", self)
        self._resume_act.setToolTip(
            "Resume a paused hierarchical drive: a SUSPENDED connectivity hold "
            "(stale pose), or a BLOCKED pause whose retry window ran out. The "
            "bot does NOT restart on its own — click this to re-acquire and "
            "continue."
        )
        self._resume_act.setEnabled(False)
        self._resume_act.triggered.connect(self._on_resume_hier)
        self._map_toolbar.addAction(self._resume_act)

        self._patrol_edit_act = QAction("Patrol edit", self)
        self._patrol_edit_act.setCheckable(True)
        self._patrol_edit_act.setChecked(False)
        self._patrol_edit_act.setToolTip(
            "Toggle right-click semantics: while on, right-clicking a "
            "map appends a waypoint to the active patrol (creating one "
            "if none is loaded). While off, right-click sets the single "
            "goal as before."
        )
        self._patrol_edit_act.toggled.connect(self._on_patrol_edit_action)
        self._map_toolbar.addAction(self._patrol_edit_act)

        self._locate_act = QAction("Set location", self)
        self._locate_act.setCheckable(True)
        self._locate_act.setChecked(False)
        self._locate_act.setToolTip(
            "Manual relocalize override: while on, LEFT-click a map to "
            "assert the robot's true (x, y) there. The localizer keeps "
            "that point and recovers heading via a full 360° scan-match. "
            "Use when Re-localize snaps to the wrong place. One-shot — "
            "the mode turns off after a click."
        )
        self._locate_act.toggled.connect(self._on_locate_action)
        self._map_toolbar.addAction(self._locate_act)

        self._stream_rgb_act = QAction("Stream RGB", self)
        self._stream_rgb_act.setCheckable(True)
        self._stream_rgb_act.setChecked(False)
        self._stream_rgb_act.setToolTip(
            "Toggle low-rate (2 Hz) streaming of OAK-D RGB into the "
            "feed pane — useful when the robot is out of sight. "
            "Default off; on-demand Request RGB still works."
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

        # Vertical stack: [ maps | feeds | vision ]. Pulling vision out
        # of a right-hand column lets the patrol dock + maps fit on a
        # 1440-wide laptop screen without horizontal overflow. Each row
        # is independently collapsible via splitter handles.
        self._v_splitter = QSplitter(Qt.Orientation.Vertical, central)
        self._v_splitter.setChildrenCollapsible(True)

        maps_widget = QWidget()
        maps = QHBoxLayout(maps_widget)
        maps.setContentsMargins(0, 0, 0, 0)
        # Shared view state so all map panels pan/zoom together and
        # share the grid + range-ring toggles.
        self._shared_view = SharedMapView()
        self._height_view = WorldHeightView(
            stale_s=self.fuser_config.map_stale_s,
            shared=self._shared_view,
        )
        self._drive_view = WorldDriveableView(
            stale_s=self.fuser_config.map_stale_s,
            shared=self._shared_view,
        )
        self._costmap_view = WorldCostmapView(
            stale_s=self.fuser_config.map_stale_s,
            shared=self._shared_view,
        )
        maps.addWidget(self._height_view, stretch=1)
        maps.addWidget(self._drive_view, stretch=1)
        maps.addWidget(self._costmap_view, stretch=1)
        self._costmap_config = CostmapConfig(
            footprint_radius_m=self.fuser_config.footprint_radius_m,
        )
        self._astar_config = AStarConfig()
        self._last_plan: Optional[PlanResult] = None
        self._last_costmap = None  # cached for replanning when goal changes
        self._shared_view.set_goal_callback(self._on_goal_requested)
        self._shared_view.set_locate_callback(self._on_locate_requested)
        # Tracing: JSONL ring + edge-triggered emits (LivenessWatcher,
        # plan-edge events). Pose sampler + auto-snapshot callback are
        # attached so events can stamp a pose / write a bundle. See
        # `tracing.py` and `health.py`.
        self._tracer = Tracer()
        self._tracer.attach_pose_sampler(self._sample_pose_for_trace)
        self._tracer.attach_snapshot_cb(self._auto_snapshot_for_trace)
        self._liveness = LivenessWatcher(
            self._tracer, fuser=self.fuser, chassis=self.chassis,
        )
        # Edge-trigger state for plan tracing. None = uninitialized (first
        # observation establishes baseline without emitting).
        self._last_plan_ok: Optional[bool] = None
        # Pose-health divergence detector: fed each redraw from the scan
        # matcher so the rolling quality window is warm. Telemetry only —
        # reset on relocate; no auto-action is wired to it.
        self._pose_health = PoseHealthMonitor()
        # Right-click append target for patrol-edit mode. Wired to
        # SharedMapView in `_build_ui` (next block).
        self._shared_view.set_patrol_append_callback(
            self._on_patrol_append_requested
        )
        self._v_splitter.addWidget(maps_widget)

        self._cameras = CameraPanels(self.chassis)
        self._v_splitter.addWidget(self._cameras.feeds_widget)
        # Vision is now a left-dock-area dock (stacks vertically with
        # Patrol) — see _build_docks. Removed from the central splitter
        # so chat doesn't compete with maps + feeds for vertical room.

        # Maps dominates; feeds is a thin strip below it.
        self._v_splitter.setStretchFactor(0, 3)
        self._v_splitter.setStretchFactor(1, 1)
        self._splitter_balanced = False

        outer.addWidget(self._v_splitter, stretch=1)

        # Bottom status block: 3 narrow rows of fixed-width labels +
        # a notes row. Fixed widths and Ignored size policies keep the
        # central widget's width-hint stable across redraws — without
        # this, per-tick text variation (grace countdown, drift, plan
        # ms, hb seq) pumps the window width and produces visible
        # jitter under XWayland/Mutter. Any label whose CONTENT may
        # vary in width MUST live in this block with a fixed width.
        small = self.font()
        small.setPointSize(max(7, small.pointSize() - 1))
        self._pose_lbl = self._mk_status_label("pose: —", 220, small)
        self._rates_lbl = self._mk_status_label("rates: —", 250, small)
        self._cells_lbl = self._mk_status_label("cells: —", 180, small)
        self._session_lbl = self._mk_status_label("session: —", 210, small)
        self._slam_lbl = self._mk_status_label("slam: —", 360, small)
        self._plan_lbl = self._mk_status_label("plan: —", 220, small)
        self._follow_lbl = self._mk_status_label("follow: —", 360, small)
        self._chassis_lbl = self._mk_status_label("chassis: —", 200, small)
        self._notes_lbl = QLabel("")
        self._notes_lbl.setStyleSheet("color: #e8a; font-weight: bold;")
        self._notes_lbl.setFont(small)

        status = QVBoxLayout()
        status.setContentsMargins(0, 0, 0, 0)
        status.setSpacing(2)
        # Row 1: data freshness — pose / rates / cells / session.
        row1 = QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)
        row1.setSpacing(8)
        for w in (self._pose_lbl, self._rates_lbl,
                  self._cells_lbl, self._session_lbl):
            row1.addWidget(w)
        row1.addStretch(1)
        status.addLayout(row1)
        # Row 2: control state — slam / plan / follow / chassis.
        row2 = QHBoxLayout()
        row2.setContentsMargins(0, 0, 0, 0)
        row2.setSpacing(8)
        for w in (self._slam_lbl, self._plan_lbl,
                  self._follow_lbl, self._chassis_lbl):
            row2.addWidget(w)
        row2.addStretch(1)
        status.addLayout(row2)
        # Row 3: notes — stretches to the available width.
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
        """Build a status-strip label that caps at width_px but can
        shrink, with an Ignored horizontal size policy so per-tick text
        changes don't pump the layout's width-hint. width_px is the
        maximum the label will need at any point in its lifetime — see
        _refresh_fuser_panel for the width-stable formatters that keep
        content within this.

        Capping with setMaximumWidth (not setFixedWidth) + Ignored policy
        is deliberate: setFixedWidth pins min=max, which sums across the
        status rows into a ~1160 px hard floor on the window width and
        stops the operator shrinking the window to fit smaller screens.
        Ignored policy gives the same jitter immunity without the floor.
        """
        from PyQt6.QtWidgets import QSizePolicy
        lbl = QLabel(initial_text)
        lbl.setStyleSheet("color: #ccc;")
        lbl.setFont(font)
        lbl.setMaximumWidth(width_px)
        lbl.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred,
        )
        return lbl

    def _build_docks(self) -> None:
        # Camera feeds live in the central splitter (built in _build_ui).
        # Teleop, Patrol, and Vision are dock-area panels.
        self._teleop = TeleopPanels(self.chassis, self.chassis_config)
        self._teleop.attach_to(self)
        # Patrol dock: lives on the left dock area to avoid stealing
        # space from the teleop column on the right.
        self._patrol_dock = PatrolDock(
            self._shared_view,
            get_live_session_id=lambda: self.fuser.grid.session_id,
            parent=self,
        )
        self._patrol_dock.attach_to(self)
        self._patrol_dock.edit_mode_toggled.connect(self._on_patrol_edit_toggled)
        # Vision dock: same left dock area as Patrol so they share the
        # vertical column. Qt stacks dock widgets added to the same
        # area; user can drag handles to rebalance. Visible by default
        # (parity with the previous central-splitter placement).
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

        self._patrol_action = QAction("Patrol panel", self)
        self._patrol_action.setCheckable(True)
        self._patrol_action.setChecked(self._patrol_dock.is_visible())
        self._patrol_action.setShortcut("Ctrl+P")
        self._patrol_action.triggered.connect(self._patrol_dock.set_visible)
        view_menu.addAction(self._patrol_action)

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
            self._shared_view.set_show_range_rings
        )
        view_menu.addAction(self._rings_action)

        self._scan_overlay_action = QAction("Lidar scan overlay", self)
        self._scan_overlay_action.setCheckable(True)
        self._scan_overlay_action.setChecked(
            self._shared_view.show_scan_overlay()
        )
        self._scan_overlay_action.setShortcut("Ctrl+L")
        self._scan_overlay_action.setToolTip(
            "Overlay the live lidar scan, transformed into the believed "
            "pose, on the maps. If the pose match is right the scan lands "
            "on the walls; if not, the mismatch is visible."
        )
        self._scan_overlay_action.toggled.connect(
            self._shared_view.set_show_scan_overlay
        )
        view_menu.addAction(self._scan_overlay_action)

        self._cp_pose_action = QAction("Checkpoint pose (experimental)", self)
        self._cp_pose_action.setCheckable(True)
        self._cp_pose_action.setChecked(self._use_checkpoint_pose)
        self._cp_pose_action.setToolTip(
            "Hierarchical drive localizes via odom dead-reckon + checkpoint "
            "re-anchor instead of the PF posterior. Applied at the next Go; "
            "needs checkpoints (Recognize) in the map."
        )
        self._cp_pose_action.toggled.connect(self._on_toggle_checkpoint_pose)
        view_menu.addAction(self._cp_pose_action)

    def _build_timer(self) -> None:
        period_ms = int(1000.0 / max(1.0, self.fuser_config.ui_redraw_hz))
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setInterval(period_ms)
        self._redraw_timer.timeout.connect(self._on_redraw_tick)
        self._redraw_timer.start()

        # Hierarchical-drive loop on its own fast timer, decoupled from the
        # (5 Hz, render-bound) redraw tick. The Tier-2 re-pick latency lives
        # here; capture showed the ARRIVED->next-goto gap was ~all desktop-side
        # (2 render ticks), so ticking at 20 Hz off the render path cuts it and
        # removes the variance from heavy frames. tick() is inert when not
        # driving, so a free-running timer is cheap.
        self._hier_timer = QTimer(self)
        self._hier_timer.setInterval(50)   # 20 Hz
        self._hier_timer.timeout.connect(self._on_hier_tick)
        self._hier_timer.start()

        # Streaming-RGB timer (off by default). The toggle action
        # starts/stops it. Rate is fixed at 2 Hz for v1; cheap to
        # adjust later if testing reveals a different sweet spot.
        self._stream_rgb_hz = 2.0
        self._stream_rgb_timer = QTimer(self)
        self._stream_rgb_timer.setInterval(
            int(1000.0 / self._stream_rgb_hz)
        )
        self._stream_rgb_timer.timeout.connect(self._on_stream_rgb_tick)
        # Not started — _on_toggle_stream_rgb starts it on demand.

    # ── Tick ─────────────────────────────────────────────────────────

    def _on_hier_tick(self) -> None:
        # Fast, render-decoupled hierarchical-drive step (see _build_timer).
        if self._hier_drive is not None:
            self._hier_drive.tick(time.time())

    def _on_redraw_tick(self) -> None:
        self._safety_toolbar.refresh()
        # Liveness watcher self-throttles to its own 1 Hz cadence;
        # cheap to call on every redraw.
        self._liveness.tick()
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
            cam_snap = build_camera_snapshot(self.chassis)
            cam_snap["streaming_on"] = self._stream_rgb_act.isChecked()
            self._cameras.update_state(cam_snap)
        # Keep View menu checkmarks in sync if the user closed a dock
        # via its titlebar X rather than via the menu action.
        if self._teleop_action.isChecked() != self._teleop.is_visible():
            self._teleop_action.setChecked(self._teleop.is_visible())
        if self._camera_action.isChecked() != self._cameras.is_visible():
            self._camera_action.setChecked(self._cameras.is_visible())
        if self._vision_action.isChecked() != self._vision_dock.isVisible():
            self._vision_action.setChecked(self._vision_dock.isVisible())
        if self._patrol_action.isChecked() != self._patrol_dock.is_visible():
            self._patrol_action.setChecked(self._patrol_dock.is_visible())
        # Sync the patrol dock's widgets with the current shared-view
        # state (session-match hint, waypoint count). Cheap.
        if self._patrol_dock.is_visible():
            self._patrol_dock.refresh()

    def _update_scan_match_overlay(self, st: dict) -> None:
        sm = st.get("scan_match") or {}
        if sm.get("valid"):
            self._shared_view.set_scan_match_overlay(sm)
        else:
            self._shared_view.set_scan_match_overlay(None)

    def _update_scan_overlay(self, pose) -> None:
        """Transform the latest lidar scan into the believed pose (world
        frame) and hand it to the shared map view, so the operator can see
        whether the scan lands on the mapped walls — i.e. whether the pose
        match is correct. Cleared when no scan/pose is available."""
        if pose is None:
            self._shared_view.set_scan_overlay(None)
            return
        with self.chassis.state.lock:
            scan = self.chassis.state.lidar_scan
        ranges = scan.get("ranges") if isinstance(scan, dict) else None
        if not isinstance(ranges, list) or not ranges:
            self._shared_view.set_scan_overlay(None)
            return
        angle_min = float(scan.get("angle_min", 0.0))
        angle_inc = float(scan.get("angle_increment", 0.0))
        if angle_inc == 0.0:
            self._shared_view.set_scan_overlay(None)
            return
        n = len(ranges)
        angles = angle_min + np.arange(n, dtype=np.float64) * angle_inc
        r = np.asarray(
            [v if isinstance(v, (int, float)) else np.nan for v in ranges],
            dtype=np.float64,
        )
        pts_body = lidar_scan_to_xy(r, angles)   # (M, 2); invalid dropped
        if pts_body.shape[0] == 0:
            self._shared_view.set_scan_overlay(None)
            return
        # Body → world by the believed pose (same transform the matcher uses).
        px, py, pth = float(pose[0]), float(pose[1]), float(pose[2])
        c, s = math.cos(pth), math.sin(pth)
        bx, by = pts_body[:, 0], pts_body[:, 1]
        wx = px + bx * c - by * s
        wy = py + bx * s + by * c
        self._shared_view.set_scan_overlay(np.stack([wx, wy], axis=-1))

    def _refresh_fuser_panel(self) -> None:
        snap = self.fuser.snapshot_for_ui()
        latest = self.fuser.pose_source.latest_pose()
        pose = latest[0] if latest is not None else None
        trail = self.fuser.pose_trail()
        # Live lidar scan, transformed into the believed pose, over the map.
        self._update_scan_overlay(pose)
        # Pull status_summary up here so pose_age is available for the
        # mission tick. ages["odom"] is local-arrival-time age — what
        # we actually want for freshness (Pi clock skew doesn't matter).
        st = self.fuser.status_summary()
        ages = st["ages"] or {}
        pose_age: Optional[float] = ages.get("odom")
        ts = time.time()
        cm = None
        if snap is not None:
            self._height_view.update_map(
                snap["grid"], snap["meta"], ts, pose=pose,
                pose_history=trail, bounds_ij=snap.get("bounds_ij"),
            )
            self._drive_view.update_map(
                snap["driveable"], snap["meta"], ts, pose=pose,
                pose_history=trail, bounds_ij=snap.get("bounds_ij"),
            )
            try:
                cm = build_costmap(snap, self._costmap_config, pose=pose)
            except Exception:
                logger.exception("costmap build failed; skipping panel update")
                cm = None
            self._costmap_view.update_map(
                cm, snap["meta"], ts, pose=pose,
                pose_history=trail, bounds_ij=snap.get("bounds_ij"),
            )
            self._update_scan_match_overlay(st)
            # Cache for replanning when goal changes; if a goal pin is
            # set, replan against the freshly-built costmap so the path
            # preview keeps up as the map fills in.
            self._last_costmap = cm
            if cm is not None and self._shared_view.goal() is not None:
                self._replan(cm, pose)
        else:
            self._height_view.update_map(
                None, None, 0.0, pose=pose,
                pose_history=trail, bounds_ij=None,
            )
            self._drive_view.update_map(
                None, None, 0.0, pose=pose,
                pose_history=trail, bounds_ij=None,
            )
            self._costmap_view.update_map(
                None, None, 0.0, pose=pose,
                pose_history=trail, bounds_ij=None,
            )
            self._update_scan_match_overlay(st)

        # Feed the pose-health monitor every tick so the rolling window
        # is warm. `scan_obs_run` dedupes repeat reads of the same match.
        try:
            ms = self.fuser.pose_source.match_summary()
            self._pose_health.ingest(
                ms, time.time(), seq=int(ms.get("scan_obs_run", 0)),
            )
        except Exception:
            logger.exception("pose-health ingest failed; skipping")

        # The hierarchical drive runs its own loop (Tier-3 owns cmd_vel via
        # body/drive/goto); tick() runs on _hier_timer (fast, decoupled).
        # Here we only refresh the overlay at the render rate.
        if self._hier_drive is not None:
            self._refresh_hier_overlay(pose)

        # All status labels below use width-stable formats: every
        # numeric field has a fixed min-width via `:>N.Mf` / `:>Nd` so
        # 3.20 and 12.34 render the same width, and "—" placeholders
        # are padded to the same. This is what stops the bottom block
        # from pumping the window's preferred width every redraw.
        if pose is not None:
            self._pose_lbl.setText(
                f"pose: x={pose[0]:>+6.2f} y={pose[1]:>+6.2f} "
                f"θ={math.degrees(pose[2]):>+6.1f}°"
            )
        else:
            self._pose_lbl.setText("pose: (no odom)")

        rates = st["rates"]
        ages = st["ages"]

        def _hz(v: Optional[float]) -> str:
            return f"{v:>4.1f}" if v is not None else "  — "

        def _age(v: Optional[float]) -> str:
            return f"{v:>4.2f}" if v is not None else "  — "

        self._rates_lbl.setText(
            f"rates: lm {_hz(rates.get('local_map'))}Hz/"
            f"{_age(ages.get('local_map'))}s  "
            f"od {_hz(rates.get('odom'))}Hz/{_age(ages.get('odom'))}s"
        )

        self._cells_lbl.setText(
            f"cells: obs={st['cells_observed']:>5d} "
            f"trav={st['cells_traversed']:>5d}"
        )

        # SLAM health: pose-unavailable streak (≥10 = sticky note set)
        # plus cumulative scan-match correction since session reset.
        # Labelled "corr=" (matches PoseSource.correction_summary API)
        # not "drift=" — for ImuPlusScanMatchPose this *is* cumulative
        # snap distance, but for ParticleFilterPoseSource it's the sum
        # of per-match |argmax − prior|, of which the filter only
        # applies a fractional Bayesian reweight to its posterior. The
        # label has to be honest for both.
        unavail = int(st.get("pose_unavail_streak") or 0)
        corr = st.get("correction_summary") or {}
        corr_m = float(corr.get("total_m") or 0.0)
        corr_deg = math.degrees(float(corr.get("total_rad") or 0.0))
        n_corr = int(corr.get("n_applied") or 0)
        sm = st.get("scan_match") or {}
        sm_hint = ""
        if sm.get("valid"):
            sm_hint = (
                f"  sm Δ={float(sm.get('shift_m', 0.0)):>4.2f}m/"
                f"{float(sm.get('improvement', 0.0)):>4.0f}"
            )
        self._slam_lbl.setText(
            f"slam: lost={unavail:>2d}  "
            f"corr={corr_m:>5.2f}m/{corr_deg:>+5.0f}°  "
            f"n={n_corr:>3d}{sm_hint}"
        )
        if unavail >= 10:
            self._slam_lbl.setStyleSheet("color: #e8a;")
        elif unavail > 0:
            self._slam_lbl.setStyleSheet("color: #ec8;")
        else:
            self._slam_lbl.setStyleSheet("color: #ccc;")

        # Plan status: "—" with no goal; compact one-shot otherwise.
        goal = self._shared_view.goal()
        if goal is None:
            self._plan_lbl.setText("plan: —")
            self._plan_lbl.setStyleSheet("color: #ccc;")
        else:
            plan = self._last_plan
            if plan is None:
                self._plan_lbl.setText("plan: pending")
                self._plan_lbl.setStyleSheet("color: #cc8;")
            elif plan.ok:
                self._plan_lbl.setText(
                    f"plan: {plan.distance_m:>5.2f}m / "
                    f"{plan.elapsed_ms:>4.0f}ms"
                )
                self._plan_lbl.setStyleSheet("color: #8cf;")
            else:
                # Truncate failure msg so the label width never bursts.
                self._plan_lbl.setText(f"plan: {plan.msg[:14]}")
                self._plan_lbl.setStyleSheet("color: #e8a;")

        # Drive status label + Go/Stop/Resume enable mirror the
        # hierarchical-drive state.
        hd = self._hier_drive
        running = hd is not None and hd.state() not in (
            HierState.IDLE, HierState.ARRIVED, HierState.FAILED,
        )
        self._go_act.setEnabled(not running)
        self._cancel_act.setEnabled(running)
        self._resume_act.setEnabled(hd is not None and hd.can_resume())
        if hd is None:
            self._follow_lbl.setText("hier: —")
            self._follow_lbl.setStyleSheet("color: #ccc;")
        else:
            # Held at an inspector breakpoint reads as "running" (the drive
            # IS active, just paused) — make that obvious so a paused drive
            # isn't mistaken for a stopped/finished one. HO-1/HO-2 hold on
            # the desktop (held_tier); HO-3 holds on the Pi (status mode).
            held = hd.held_tier()
            if held is None and self._drive_client is not None:
                drive_st = self._drive_client.latest_status()
                if drive_st is not None and drive_st.get("mode") == "held":
                    held = 3
            if held is not None:
                self._follow_lbl.setText(
                    f"⏸ PAUSED @ HO-{held} — breakpoint (Run free in inspector)")
                self._follow_lbl.setStyleSheet("color: #fd0; font-weight: bold;")
            else:
                br = hd.block_reason()
                txt = f"hier: {hd.state().value}" + (f"  {br[:16]}" if br else "")
                # Checkpoint-pose: show the last re-anchor (cp id + inlier) so
                # the operator sees it locking onto checkpoints while driving.
                if self._cp_localizer is not None:
                    lm = self._cp_localizer.last_match
                    txt += (f"  ⚓{lm.checkpoint_id} {lm.inlier_frac:.2f}"
                            if lm is not None else "  ⚓cp?")
                self._follow_lbl.setText(txt)
                self._follow_lbl.setStyleSheet(
                    "color: #fb4;" if hd.is_suspended() else
                    "color: #e8a;" if hd.state() in (HierState.BLOCKED, HierState.FAILED)
                    else "color: #8cf;"
                )

        # Session id is fixed-8; pose-source label can be "odom" or
        # "imu+scan_match" — clamp so the label width is bounded.
        src = st.get("pose_source") or "—"
        if len(src) > 14:
            src = src[:13] + "…"
        self._session_lbl.setText(
            f"session: {st['session_id'][:8]} ({src})"
        )
        self._notes_lbl.setText(st.get("notes") or "")

    # ── Relocate helper ─────────────────────────────────────────────

    def _run_relocate(self, *, reason: str) -> dict:
        """MCL relocate (global scan-match) + patrol/goal frame shift."""
        self.chassis.set_cmd_vel(0.0, 0.0)
        result = dict(self.fuser.request_relocate(reason=reason))
        if result.get("success"):
            result["shift_count"] = self._apply_relocate_to_patrol(result)
            # Fresh fix — drop stale low-quality samples so the health
            # monitor judges the new pose from scratch.
            self._pose_health.reset()
        else:
            result["shift_count"] = 0
        return result

    def _refresh_chassis_panel(self) -> None:
        """Text summary with values the pills can't convey (status age,
        heartbeat seq). Gate colors live on the safety toolbar.

        Width-stable formatters: status age is `{:>4.1f}s` so 0.1 and
        12.3 share width; hb seq is rendered as the last 4 digits so it
        doesn't grow without bound across a long session.
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
        if status_ts > 0:
            age_s = f"{time.time() - status_ts:>4.1f}"
        else:
            age_s = "  — "
        self._chassis_lbl.setText(
            f"chassis: {age_s}s  #{hb_seq % 10000:04d}"
        )
        self._chassis_lbl.setStyleSheet("color: #ccc;")

    # ── Planning ─────────────────────────────────────────────────────

    def _on_goal_requested(self, x_w: float, y_w: float) -> None:
        """Right-click in any map view → set goal here, plan now."""
        self._shared_view.set_goal((x_w, y_w))
        cm = self._last_costmap
        latest = self.fuser.pose_source.latest_pose()
        pose = latest[0] if latest is not None else None
        if cm is None or pose is None:
            self._shared_view.set_planned_path([])
            self._notes_lbl.setText(
                "goal set; waiting for map + pose before planning"
            )
            return
        self._replan(cm, pose)

    def _replan(self, costmap, pose) -> None:
        goal = self._shared_view.goal()
        if goal is None or costmap is None or pose is None:
            return
        result = plan_path(
            costmap,
            start_world=(pose[0], pose[1]),
            goal_world=goal,
            config=self._astar_config,
        )
        self._last_plan = result
        if result.ok:
            self._shared_view.set_planned_path(result.waypoints_world)
        else:
            self._shared_view.set_planned_path([])
        # Edge-triggered plan tracing: emit on fail→ok and ok→fail
        # transitions only. Every-tick plan.ok would flood the file.
        # The first observed result establishes the baseline silently.
        prev_ok = self._last_plan_ok
        cur_ok = bool(result.ok)
        if prev_ok is None:
            self._last_plan_ok = cur_ok
        elif cur_ok != prev_ok:
            self._last_plan_ok = cur_ok
            if cur_ok:
                self._tracer.emit(
                    CAT_PLAN, "ok",
                    {
                        "distance_m": result.distance_m,
                        "expansions": result.n_expansions,
                        "elapsed_ms": result.elapsed_ms,
                        "goal": [goal[0], goal[1]],
                    },
                )
            else:
                self._tracer.emit(
                    CAT_PLAN, "fail",
                    {
                        "msg": result.msg,
                        "expansions": result.n_expansions,
                        "elapsed_ms": result.elapsed_ms,
                        "goal": [goal[0], goal[1]],
                    },
                    level=LEVEL_WARN,
                )

    def _on_clear_goal(self) -> None:
        self._shared_view.set_goal(None)
        self._shared_view.set_planned_path([])
        self._last_plan = None

    def _on_locate_action(self, checked: bool) -> None:
        """Toolbar toggle → arm left-click 'set location' on the maps."""
        self._shared_view.set_locate_mode(bool(checked))
        if checked:
            self._notes_lbl.setText(
                "Set-location armed: left-click the robot's true position."
            )

    def _on_locate_requested(self, x_w: float, y_w: float) -> None:
        """Left-click in locate mode → assert true (x, y), recover yaw."""
        # One-shot: disarm immediately without re-entering the handler.
        if self._locate_act.isChecked():
            blk = self._locate_act.blockSignals(True)
            self._locate_act.setChecked(False)
            self._locate_act.blockSignals(blk)
        self._shared_view.set_locate_mode(False)

        # relocate_at rewrites the pose frame, like relocate — stop motion.
        # (A running hierarchical drive re-picks on its own: the relocate
        # bumps the provider's discrete correction_seq.)
        self.chassis.set_cmd_vel(0.0, 0.0)

        result = dict(self.fuser.request_relocate_at(x_w, y_w, reason="ui_locate"))
        if result.get("success"):
            result["shift_count"] = self._apply_relocate_to_patrol(result)
            shift = int(result.get("shift_count", 0))
            yaw_deg = math.degrees(float(result["best_pose"][2]))
            QMessageBox.information(
                self, "Set location",
                f"Placed robot at ({x_w:+.2f}, {y_w:+.2f}) m; recovered "
                f"heading {yaw_deg:+.1f}° (scan-match improvement "
                f"{result.get('improvement', 0.0):.0f} over "
                f"{result.get('evidence_cells', 0)} evidence cells)."
                + (
                    f"\n\nShifted {shift} waypoint(s) / goal pin to match "
                    f"the new frame." if shift > 0 else ""
                ),
            )
        else:
            QMessageBox.warning(
                self, "Set location failed",
                f"reason: {result.get('reason', 'unknown')}\n"
                + "\n".join(
                    f"{k}: {v}" for k, v in result.items()
                    if k not in ("success", "reason")
                ),
            )

    def _try_checkpoint_relocate(self) -> bool:
        """Recognize which checkpoint we're at and seat the pose there —
        robust on a metrically-loose map (where the global scan-match snaps to
        the wrong self-similar basin). Returns True if a checkpoint matched."""
        matcher = self._checkpoint_matcher(CheckpointMatchConfig())
        if matcher is None:
            return False
        ar = self.fuser.pose_source.latest_scan_polar()
        if ar is None:
            return False
        latest = self.fuser.pose_source.latest_pose()
        yaw = float(latest[0][2]) if latest is not None else None
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            m = matcher.relocalize(
                ar[0], ar[1], yaw_hint=yaw,
                xy_half_m=0.6, theta_half_rad=math.radians(60.0))
        finally:
            QApplication.restoreOverrideCursor()
        if m is None:
            return False
        # Seat the PF pose at the recognized spot (relocate_at recovers yaw on
        # the locally-healed patch); re-seed an active checkpoint localizer.
        self.fuser.request_relocate_at(m.pose[0], m.pose[1], reason="ui_relocate_cp")
        if self._cp_localizer is not None:
            odom = self._cp_odom()
            if odom is not None:
                self._cp_localizer.seed(m.pose, odom)
        QMessageBox.information(
            self, "Re-localize",
            f"Recognized checkpoint {m.checkpoint_id} "
            f"(inlier {m.inlier_frac:.2f}) — pose seated there.")
        return True

    def _on_relocate(self) -> None:
        # Prefer checkpoint recognition (robust on a metrically-loose map);
        # fall back to the global scan-match only when no checkpoint matches.
        # Zero cmd_vel first — a relocate rewrites the world offset. (A
        # running hierarchical drive re-picks on its own: the relocate bumps
        # the provider's discrete correction_seq.)
        self.chassis.set_cmd_vel(0.0, 0.0)
        if self._try_checkpoint_relocate():
            return
        result = self._run_relocate(reason="ui_relocate")
        if result.get("success"):
            shift = int(result.get("shift_count", 0))
            detail = ""
            if "improvement" in result and "evidence_cells" in result:
                detail = (
                    f"(improvement {result['improvement']:.0f} over "
                    f"{result['evidence_cells']} evidence cells)."
                )
            elif result.get("method") == "mcl":
                n_particles = result.get("particle_count")
                if n_particles is not None:
                    detail = f"(MCL snap, {n_particles} particles)."
                else:
                    detail = "(MCL particle snap)."
            QMessageBox.information(
                self, "Re-localize",
                f"Snapped pose by "
                f"dx={result['dx']:+.2f} m, dy={result['dy']:+.2f} m, "
                f"dθ={math.degrees(result['dtheta']):+.1f}° "
                f"{detail}"
                + (
                    f"\n\nShifted {shift} waypoint(s) / goal pin to "
                    f"match the new frame." if shift > 0 else ""
                ),
            )
        else:
            QMessageBox.warning(
                self, "Re-localize failed",
                f"reason: {result.get('reason', 'unknown')}\n"
                + "\n".join(
                    f"{k}: {v}" for k, v in result.items()
                    if k not in ("success", "reason")
                ),
            )

    def _apply_relocate_to_patrol(self, result: Dict[str, Any]) -> int:
        """Transform the active patrol's waypoints and the single goal
        pin by the SE(2) given by `result["prior_pose"]` +
        (dx, dy, dtheta). Saved Patrol on disk is NOT mutated — the
        operator persists via Save if they want the new coords.

        Returns the count of points transformed (waypoints + 1 for the
        goal if present).
        """
        prior = result.get("prior_pose")
        if not prior or len(prior) < 3:
            return 0
        dx = float(result.get("dx", 0.0))
        dy = float(result.get("dy", 0.0))
        dtheta = float(result.get("dtheta", 0.0))
        bot_old = (float(prior[0]), float(prior[1]))
        bot_new = (bot_old[0] + dx, bot_old[1] + dy)
        cos_d = math.cos(dtheta)
        sin_d = math.sin(dtheta)

        def transform(xy):
            rx = xy[0] - bot_old[0]
            ry = xy[1] - bot_old[1]
            return (
                bot_new[0] + cos_d * rx - sin_d * ry,
                bot_new[1] + sin_d * rx + cos_d * ry,
            )

        n_transformed = 0
        wp_changes: list = []

        patrol = self._shared_view.patrol()
        if patrol is not None:
            for i, wp in enumerate(patrol.waypoints):
                old_xy = (wp.x_m, wp.y_m)
                new_xy = transform(old_xy)
                wp.x_m = new_xy[0]
                wp.y_m = new_xy[1]
                wp_changes.append({
                    "wp_index": i,
                    "old_xy": [old_xy[0], old_xy[1]],
                    "new_xy": [new_xy[0], new_xy[1]],
                })
                n_transformed += 1
            # Re-set on shared view to fire notify so map re-renders.
            self._shared_view.set_patrol(patrol)

        goal = self._shared_view.goal()
        goal_change = None
        if goal is not None:
            new_goal = transform(goal)
            self._shared_view.set_goal(new_goal)
            goal_change = {
                "old_xy": [goal[0], goal[1]],
                "new_xy": [new_goal[0], new_goal[1]],
            }
            n_transformed += 1

        if n_transformed > 0:
            try:
                self._tracer.emit(
                    "patrol", "world_relocated",
                    {
                        "prior_pose": [
                            float(prior[0]), float(prior[1]), float(prior[2]),
                        ],
                        "dx": dx,
                        "dy": dy,
                        "dtheta": dtheta,
                        "n_waypoints": (
                            len(patrol.waypoints) if patrol is not None else 0
                        ),
                        "waypoints": wp_changes,
                        "goal": goal_change,
                    },
                )
            except Exception:
                logger.exception("world_relocated emit raised")

        return n_transformed

    def _on_cancel(self) -> None:
        """Operator-initiated stop: cancel the in-flight Tier-3 goto and
        drop the drive. Live cmd is left as-is so the operator can drive
        manually without a second click."""
        self._stop_hier_drive()

    # ── Hierarchical drive ───────────────────────────────────────────

    def _new_drive_trace_path(self) -> Optional[str]:
        """A per-session JSONL path for the Tier-3 status trace (every
        body/drive/status: state, plan_reason, path_body, goal, v/omega,
        blocked_reason) — for offline root-cause of a drive leg."""
        try:
            d = os.path.expanduser("~/body-logs")
            os.makedirs(d, exist_ok=True)
            return os.path.join(d, f"drive-trace-{int(time.time())}.jsonl")
        except OSError:
            logger.exception("could not prepare drive-trace dir")
            return None

    def _ensure_drive_client(self) -> bool:
        """Lazily open the DriveClient session (own zenoh session for
        body/drive/goto + status + scan) on the first Go. Returns True once
        a connected client is available; warns and returns False otherwise.
        Closed in closeEvent."""
        if self._drive_client is None:
            self._drive_client = DriveClient(
                self.chassis_config.router,
                trace_path=self._new_drive_trace_path())
        ok, err = self._drive_client.connect()
        if not ok:
            QMessageBox.warning(
                self, "Hierarchical drive",
                f"Drive client connect failed:\n{err}",
            )
            return False
        return True

    def _ensure_handoff_gate(self) -> Optional[HandoffGate]:
        """Lazily open the handoff-inspector session + gate (HO-1/HO-2 records
        + arm/continue over zenoh). Tolerant: on failure the drive simply runs
        with no sink (NullHandoffSink), i.e. exactly as before."""
        if self._handoff_gate is not None:
            return self._handoff_gate
        try:
            self._handoff_session = open_session(self.chassis_config.router)
            self._handoff_gate = HandoffGate(self._handoff_session, tiers=(1, 2))
        except Exception:
            logger.exception("handoff gate session failed; running without inspector")
            self._handoff_session = None
            self._handoff_gate = None
        return self._handoff_gate

    def _on_go(self) -> None:
        """Start hierarchical drive over the loaded patrol. Live cmd stays
        OFF — Tier-3 owns cmd_vel; nav only keeps the heartbeat alive."""
        patrol = self._shared_view.patrol()
        if patrol is None or len(patrol.waypoints) == 0:
            QMessageBox.information(
                self, "Hierarchical drive",
                "Place patrol waypoints first (Patrol edit), then review the "
                "route on the map before Go.",
            )
            return
        if not self._ensure_drive_client():
            return
        with self.chassis.state.lock:
            connected = self.chassis.state.connected
        if not connected:
            QMessageBox.warning(
                self, "Hierarchical drive", "Chassis disconnected — connect first.",
            )
            return
        if self._drive_client.odom_pose() is None or self._drive_client.latest_scan() is None:
            QMessageBox.warning(
                self, "Hierarchical drive",
                "No odom/scan from the Pi yet — wait for telemetry.",
            )
            return
        # Tier-1 global expansion: route each high-level segment with A* on the
        # reference-map costmap and densify into Tier-3-executable sub-waypoints,
        # so the drive follows corridors around dead-ends instead of beelining.
        if self._last_costmap is None:
            QMessageBox.warning(
                self, "Hierarchical drive",
                "No costmap yet — wait for the map to render before Go.",
            )
            return
        # Route the lead-in (start pose → first marker) too, so the start isn't
        # a greedy beeline. Best-effort: a missing lead-in falls back to today's
        # behavior, kept separate from the patrol so lap accounting is unchanged.
        latest = self.fuser.pose_source.latest_pose()
        start_xy = (latest[0][0], latest[0][1]) if latest is not None else None
        exp = expand_patrol(patrol, self._last_costmap, ExpandConfig(),
                            start_xy=start_xy)
        if not exp.ok:
            seg = exp.failed_segment
            QMessageBox.warning(
                self, "Hierarchical drive",
                f"Could not route the patrol (segment {seg}): {exp.reason}\n"
                "Adjust that waypoint or add an intermediate one and retry.",
            )
            return
        exec_patrol = exp.patrol
        lead_in = exp.lead_in or []
        # Preview the dense routed path on the maps (cyan polyline) — lead-in
        # (minus its shared last point) + markers.
        self._shared_view.set_planned_path(
            list(lead_in[:-1]) + [(w.x_m, w.y_m) for w in exec_patrol.waypoints]
        )
        logger.info("hier: expanded %d waypoints -> %d sub-waypoints (+%d lead-in)",
                    len(patrol.waypoints), len(exec_patrol.waypoints),
                    max(0, len(lead_in) - 1))
        runner = PatrolRunner(exec_patrol)
        provider = self._make_pose_provider()
        self._hier_drive = HierarchicalDrive(
            runner, provider, self._drive_client, HierConfig(),
            sink=self._ensure_handoff_gate(), lead_in=lead_in,
        )
        self._shared_view.set_patrol_active_wp_index(0)
        self._hier_drive.start()

    # ── Pose provider (PF posterior, or checkpoint re-anchor) ────────

    def _make_pose_provider(self):
        """The PoseProvider the hierarchical drive runs on. Default: the PF
        posterior (`PFPoseProvider`). With checkpoint-pose enabled (and the
        map has checkpoints): odom dead-reckon + checkpoint re-anchor."""
        self._cp_localizer = None
        if not self._use_checkpoint_pose:
            return PFPoseProvider(self.fuser)
        matcher = self._checkpoint_matcher(_RUNTIME_CP_CFG)
        if matcher is None:
            QMessageBox.warning(
                self, "Checkpoint pose",
                "No checkpoints in this map — using the PF pose instead.\n"
                "Recognize some spots in the map editor first.")
            return PFPoseProvider(self.fuser)
        self._cp_localizer = CheckpointLocalizer(matcher, reanchor_min_interval_s=0.5)
        logger.info("hier: checkpoint-pose enabled")
        return CheckpointPoseProvider(
            self._cp_localizer,
            odom_fn=self._cp_odom,
            scan_fn=self._cp_scan,
            seed_fn=self._cp_seed,
            age_fn=self.fuser.pose_source.odom_age_s,
        )

    def _checkpoint_matcher(self, cfg: CheckpointMatchConfig):
        """A CheckpointMatcher over the loaded map + its checkpoints, or None
        when the map has none."""
        rm = self.fuser.reference_map
        cps = checkpoints_from_metadata(rm.metadata)
        if not cps:
            return None
        occ = rm.occupancy_log_odds > 0.0
        return CheckpointMatcher(
            occ, rm.origin_x_m, rm.origin_y_m, rm.resolution_m, cps, cfg)

    def _cp_odom(self):
        return self._drive_client.odom_pose() if self._drive_client else None

    def _cp_seed(self):
        latest = self.fuser.pose_source.latest_pose()
        return latest[0] if latest is not None else None

    def _cp_scan(self):
        if self._drive_client is None:
            return None
        scan = self._drive_client.latest_scan()
        if not scan:
            return None
        ranges = scan.get("ranges")
        if not isinstance(ranges, list) or not ranges:
            return None
        amin = float(scan.get("angle_min", 0.0))
        ainc = float(scan.get("angle_increment", 0.0))
        if ainc == 0.0:
            return None
        n = len(ranges)
        angles = amin + np.arange(n, dtype=np.float64) * ainc
        r = np.asarray(
            [v if isinstance(v, (int, float)) else np.nan for v in ranges],
            dtype=np.float64)
        return angles, r

    def _on_toggle_checkpoint_pose(self, on: bool) -> None:
        """Flip the pose source for the *next* Go (a running drive keeps its
        provider until restarted)."""
        self._use_checkpoint_pose = bool(on)

    def _on_resume_hier(self) -> None:
        """Operator resume of a paused hier drive (SUSPENDED connectivity hold
        or a BLOCKED pause whose retry window ran out)."""
        hd = self._hier_drive
        if hd is not None:
            hd.request_resume()

    def _stop_hier_drive(self) -> None:
        if self._hier_drive is not None:
            self._hier_drive.stop()
            self._hier_drive = None
        self._cp_localizer = None
        self._shared_view.set_lookahead(None)
        self._shared_view.set_planned_path([])   # clear the routed-path preview
        self._cameras.set_lidar_overlay(None, None)

    def _on_all_stop(self) -> None:
        """ALL-STOP hook: cancel any in-flight hierarchical-drive goto."""
        self._stop_hier_drive()

    def _refresh_hier_overlay(self, pose) -> None:
        """Render the live Tier-2 sub-goal + next waypoint (display only).

        World maps get the sub-goal as a lookahead marker; the lidar plot
        gets both the sub-goal and the next waypoint in BODY frame, so the
        world->body translation is visible against the live scan."""
        hd = self._hier_drive
        sub = hd.current_subgoal_body() if hd else None
        wp_world = hd.current_waypoint_world() if hd else None
        self._shared_view.set_lookahead(
            body_to_odom(sub, pose) if (sub is not None and pose is not None) else None
        )
        wp_body = None
        if wp_world is not None and pose is not None:
            dx, dy = wp_world[0] - pose[0], wp_world[1] - pose[1]
            c, s = math.cos(-pose[2]), math.sin(-pose[2])
            wp_body = (dx * c - dy * s, dx * s + dy * c)
        self._cameras.set_lidar_overlay(wp_body, sub)

    # ── Toolbar handlers ─────────────────────────────────────────────

    def _on_load_snapshot(self) -> None:
        """Pick a layers.npz via file dialog and restore it into the
        live world grid. Cancelling any active mission first so we
        don't drive against a freshly-replaced map."""
        # Default to the conventional sessions directory; fall back
        # to home if it doesn't exist yet.
        default_dir = os.path.expanduser("~/Body/sessions")
        if not os.path.isdir(default_dir):
            default_dir = os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load snapshot — pick a layers.npz",
            default_dir,
            "World snapshots (layers.npz)",
        )
        if not path:
            return
        # Stop a running drive so we don't drive on a freshly-replaced map.
        self._stop_hier_drive()
        try:
            summary = self.fuser.load_snapshot(path)
        except Exception as e:
            logger.exception("load_snapshot failed")
            QMessageBox.warning(
                self, "Load snapshot failed",
                f"Could not load snapshot:\n{type(e).__name__}: {e}",
            )
            return
        # If the bundle contains a patrols.json, load it into the
        # shared view so the operator gets the patrol back alongside
        # the world layers. Best-effort — a malformed sidecar
        # shouldn't block the layer reload.
        patrol_info = ""
        patrol_path = os.path.join(os.path.dirname(path), "patrols.json")
        if os.path.isfile(patrol_path):
            try:
                p = patrol_mod.load_from_file(patrol_path)
                self._shared_view.set_patrol(p)
                self._shared_view.set_patrol_active_wp_index(0)
                patrol_info = (
                    f"\n\npatrol loaded: {p.name} "
                    f"({len(p.waypoints)} waypoints)"
                )
            except Exception:
                logger.exception("patrols.json load from bundle failed")
                patrol_info = "\n\npatrols.json found but failed to load"
        QMessageBox.information(
            self, "Snapshot loaded",
            f"Loaded {summary['cells_observed']} cells from\n"
            f"{path}\n\n"
            f"loaded session: {summary['loaded_session_id'][:8]}\n"
            f"live session:   {summary['current_session_id'][:8]}"
            + patrol_info
        )

    def _on_save_snapshot(self) -> None:
        try:
            out_dir = self._save_bundle_with_trace()
        except Exception as e:
            logger.exception("snapshot bundle write failed")
            QMessageBox.warning(
                self, "Snapshot failed",
                f"Could not write snapshot bundle:\n{type(e).__name__}: {e}",
            )
            return
        # Non-modal toast; let the operator click through to the
        # directory if they want to inspect it.
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle("Snapshot saved")
        box.setText(f"Snapshot bundle written:\n{out_dir}")
        open_btn = box.addButton("Open folder", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Ok)
        box.exec()
        if box.clickedButton() is open_btn:
            QDesktopServices.openUrl(QUrl.fromLocalFile(out_dir))

    # ── Trace integration helpers ────────────────────────────────────

    def _save_bundle_with_trace(self) -> str:
        """Write a snapshot bundle and, if a trace is currently open,
        copy the active trace.jsonl into the bundle directory. Also
        embeds the active patrol (when one is loaded) as
        `patrols.json`, so a reload of the bundle restores the patrol
        alongside the world layers. Returns the bundle path. Raises
        on bundle-write errors; trace/patrol side files are best-
        effort.
        """
        out_dir = self.fuser.save_snapshot_bundle()
        trace_path = self._tracer.current_path()
        if trace_path:
            try:
                shutil.copy(
                    trace_path, os.path.join(out_dir, "trace.jsonl"),
                )
            except Exception:
                logger.exception(
                    "trace copy into snapshot bundle failed"
                )
        patrol = self._shared_view.patrol()
        if patrol is not None and len(patrol.waypoints) > 0:
            try:
                patrol_mod.write_to_file(
                    patrol, os.path.join(out_dir, "patrols.json"),
                )
            except Exception:
                logger.exception(
                    "patrol embed into snapshot bundle failed"
                )
        return out_dir

    def _sample_pose_for_trace(self):
        """Pose sampler registered with the Tracer. Returns the latest
        world-frame pose or None when none is available."""
        try:
            latest = self.fuser.pose_source.latest_pose()
        except Exception:
            return None
        if latest is None:
            return None
        return latest[0]

    def _auto_snapshot_for_trace(self, event_code: str) -> Optional[str]:
        """Tracer-invoked auto-snapshot. Writes a full bundle (including
        the active trace prefix) and returns the bundle path so it can
        be stamped into the triggering event's data.
        """
        try:
            return self._save_bundle_with_trace()
        except Exception:
            logger.exception(
                f"auto-snapshot for {event_code} raised; skipping"
            )
            return None

    def _on_fit_maps(self) -> None:
        # Single call: shared view propagates to all attached panels.
        self._shared_view.reset_view()

    def _on_toggle_stream_rgb(self, checked: bool) -> None:
        if checked:
            self._stream_rgb_timer.start()
        else:
            self._stream_rgb_timer.stop()

    # ── Patrol UI handlers ───────────────────────────────────────────

    def _on_patrol_edit_action(self, checked: bool) -> None:
        """Toolbar toggle → propagate to shared view + dock checkbox.
        The dock has its own checkbox; we keep both in sync via the
        shared-view boolean so either control reflects reality."""
        self._shared_view.set_patrol_edit_mode(bool(checked))
        # Keep the dock checkbox visually in sync without firing its
        # toggle signal (which would re-enter this handler).
        if self._patrol_dock._edit_box.isChecked() != bool(checked):
            blk = self._patrol_dock._edit_box.blockSignals(True)
            self._patrol_dock._edit_box.setChecked(bool(checked))
            self._patrol_dock._edit_box.blockSignals(blk)

    def _on_patrol_edit_toggled(self, on: bool) -> None:
        """Dock checkbox → propagate to toolbar action."""
        if self._patrol_edit_act.isChecked() != bool(on):
            blk = self._patrol_edit_act.blockSignals(True)
            self._patrol_edit_act.setChecked(bool(on))
            self._patrol_edit_act.blockSignals(blk)

    def _on_patrol_append_requested(self, x_w: float, y_w: float) -> None:
        """Right-click append handler — wired into the SharedMapView.
        Creates an empty patrol if none is loaded (so the operator can
        start placing pins without an explicit New click), then appends
        the world point.
        """
        hd = self._hier_drive
        if hd is not None and hd.state() not in (
            HierState.IDLE, HierState.ARRIVED, HierState.FAILED,
        ):
            # Edit-mid-drive is locked by design.
            return
        p = self._shared_view.patrol()
        if p is None:
            live_sid = self.fuser.grid.session_id
            p = patrol_mod.new_empty(session_id=live_sid)
        p.append(x_w, y_w)
        self._shared_view.set_patrol(p)

    def _on_stream_rgb_tick(self) -> None:
        # Streaming-mode capture: in-flight gating in the controller
        # keeps a slow Pi from accruing a backlog. If the chassis is
        # disconnected, request_rgb_streaming() returns None and the
        # tick is a no-op.
        self.chassis.request_rgb_streaming()

    # ── Lifecycle ────────────────────────────────────────────────────

    def _balance_splitter_once(self) -> None:
        # Applied via QTimer.singleShot(0, …) from showEvent: by the
        # time this fires, the splitter has a real size from the first
        # layout pass. Idempotent so spurious re-fires don't stomp on
        # a user's drag.
        if self._splitter_balanced:
            return
        v_total = self._v_splitter.height()
        if v_total <= 0:
            return
        # Maps gets ~75% of the column, feeds the remaining ~25%.
        # Vision lives in the left dock area now (alongside Patrol).
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
        except Exception:
            pass
        # Flush + close any open trace file so the last events aren't
        # left in an unclosed handle (line-buffering covers writes, but
        # close releases the fd cleanly for any tail consumer).
        try:
            self._tracer.close()
        except Exception:
            pass
        # Stop any in-flight hierarchical drive and close the DriveClient
        # session (opened lazily on the first Go).
        try:
            self._stop_hier_drive()
            if self._drive_client is not None:
                self._drive_client.shutdown()
                self._drive_client = None
            if self._handoff_session is not None:
                self._handoff_session.close()
                self._handoff_session = None
                self._handoff_gate = None
        except Exception:
            pass
        super().closeEvent(event)


def run_app(
    fuser: LocalizationController,
    fuser_config: LocalizationConfig,
    chassis: StubController,
    chassis_config: StubConfig,
    *,
    use_checkpoint_pose: bool = False,
) -> int:
    app = QApplication.instance() or QApplication([])
    win = NavMainWindow(fuser, fuser_config, chassis, chassis_config,
                        use_checkpoint_pose=use_checkpoint_pose)
    win.show()
    return app.exec()
