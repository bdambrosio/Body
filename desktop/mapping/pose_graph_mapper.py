"""Pose-graph SLAM mapper — scan match, loop closure, optimized occupancy."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from desktop.fusion.ekf_pose_tracker import EkfPoseTracker
from desktop.fusion.load_slam_config import (
    FusionNoiseConfig,
    SlamConfig,
    loop_closure_matcher_config_from_slam,
    scan_matcher_config_from_slam,
)
from desktop.mapping.graph.pose_graph import (
    PoseGraphEdge,
    odom_information,
    optimize_pose_graph,
    relative_pose,
)
from desktop.mapping.occupancy_builder import OccupancyBuilder
from desktop.nav.slam.scan_matcher import ScanMatcher, lidar_scan_to_xy
from desktop.nav.slam.types import Pose2D
from desktop.reference_map.reference_map import ReferenceMap

PoseTuple = Tuple[float, float, float]


@dataclass
class GraphNode:
    node_id: int
    ts: float
    x: float
    y: float
    theta: float
    ranges: np.ndarray
    angles: np.ndarray


@dataclass
class PoseGraphMapper:
    """slam_toolbox-style pose graph: local scan match + loop closure."""

    slam_config: SlamConfig
    fusion_config: FusionNoiseConfig
    ekf: EkfPoseTracker
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _nodes: List[GraphNode] = field(default_factory=list, repr=False)
    _edges: List[PoseGraphEdge] = field(default_factory=list, repr=False)
    _display_builder: Optional[OccupancyBuilder] = field(default=None, repr=False)
    _matcher: ScanMatcher = field(default=None, repr=False)  # type: ignore[assignment]
    _loop_matcher: ScanMatcher = field(default=None, repr=False)  # type: ignore[assignment]
    _last_match_mono: float = field(default=0.0, repr=False)
    _trajectory: Deque[Tuple[float, float, float, float]] = field(
        default_factory=lambda: deque(maxlen=4096),
        repr=False,
    )
    _last_loop_closure_ts: float = field(default=0.0, repr=False)
    _last_match_improvement: float = field(default=0.0, repr=False)
    _last_match_accepted: bool = field(default=False, repr=False)
    _optimize_count: int = field(default=0, repr=False)
    _loop_closure_count: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if self._matcher is None:
            self._matcher = ScanMatcher(scan_matcher_config_from_slam(self.slam_config))
        if self._loop_matcher is None:
            self._loop_matcher = ScanMatcher(
                loop_closure_matcher_config_from_slam(self.slam_config),
            )
        self._display_builder = OccupancyBuilder(
            extent_m=self.slam_config.extent_m,
            resolution_m=self.slam_config.resolution_m,
        )

    def reset(self) -> None:
        with self._lock:
            self._nodes.clear()
            self._edges.clear()
            self._trajectory.clear()
            self._last_match_mono = 0.0
            self._last_loop_closure_ts = 0.0
            self._last_match_improvement = 0.0
            self._last_match_accepted = False
            self._optimize_count = 0
            self._loop_closure_count = 0
            self._display_builder = OccupancyBuilder(
                extent_m=self.slam_config.extent_m,
                resolution_m=self.slam_config.resolution_m,
            )

    def add_scan(
        self,
        ts: float,
        ranges_m: np.ndarray,
        angles_rad: np.ndarray,
    ) -> bool:
        """Rate-limited scan ingestion; returns True if a node was added."""
        now_mono = time.monotonic()
        hz = max(0.5, self.slam_config.match_hz)
        if now_mono - self._last_match_mono < 1.0 / hz:
            return False
        self._last_match_mono = now_mono

        if not self.ekf.is_ready():
            return False
        prior = self.ekf.pose_at(ts)
        if prior is None:
            return False

        scan_xy = lidar_scan_to_xy(ranges_m, angles_rad)
        prior_pose = Pose2D(prior[0], prior[1], prior[2])

        with self._lock:
            if not self._nodes:
                self._add_node(ts, prior, ranges_m, angles_rad)
                self._integrate_display_node(self._nodes[-1])
                self._trajectory.append((ts, prior[0], prior[1], prior[2]))
                return True

            submap, evidence_count = self._match_evidence()
            if evidence_count >= self.slam_config.min_submap_evidence_cells:
                result = self._matcher.search(
                    scan_xy,
                    prior_pose,
                    submap,
                    self._display_builder.origin_x_m,
                    self._display_builder.origin_y_m,
                    self._display_builder.resolution_m,
                )
                corrected = (
                    (result.pose.x, result.pose.y, result.pose.theta)
                    if result.accepted
                    else prior
                )
                self._last_match_improvement = float(result.improvement)
                self._last_match_accepted = bool(result.accepted)
            else:
                corrected = prior
                self._last_match_improvement = 0.0
                self._last_match_accepted = False

            prev = self._nodes[-1]
            rel = relative_pose(
                (prev.x, prev.y, prev.theta),
                corrected,
            )
            info = odom_information(
                rel[0],
                rel[2],
                alpha_trans=self.fusion_config.alpha_trans_per_m,
                alpha_rot_m=self.fusion_config.alpha_rot_per_m,
                alpha_rot_rad=self.fusion_config.alpha_rot_per_rad,
                match_response=max(0.5, self._last_match_improvement / 20.0),
            )
            new_id = self._add_node(ts, corrected, ranges_m, angles_rad)
            self._edges.append(
                PoseGraphEdge(
                    from_id=prev.node_id,
                    to_id=new_id,
                    measurement=rel,
                    information=info,
                ),
            )
            self._try_loop_closure(new_id, scan_xy, corrected)
            did_optimize = False
            if (
                len(self._nodes) % max(1, self.slam_config.graph_optimize_every_n_nodes) == 0
                or self._loop_closure_count > self._optimize_count
            ):
                self._optimize_graph()
                did_optimize = True
            if did_optimize:
                self._rebuild_display_map()
            else:
                self._integrate_display_node(self._nodes[-1])
            self._trajectory.append((ts, corrected[0], corrected[1], corrected[2]))
            return True

    def display_pose(self) -> Optional[PoseTuple]:
        with self._lock:
            if not self._nodes:
                return self.ekf.pose()
            n = self._nodes[-1]
            return (n.x, n.y, n.theta)

    def trajectory(self) -> List[Tuple[float, float, float]]:
        with self._lock:
            return [(x, y, th) for (_t, x, y, th) in self._trajectory]

    def snapshot_for_ui(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self._display_builder is None:
                return None
            return self._display_builder.snapshot_for_ui()

    def to_reference_map(
        self,
        *,
        session_id: str = "",
        metadata: Optional[dict] = None,
    ) -> ReferenceMap:
        with self._lock:
            self._optimize_graph()
            self._rebuild_display_map()
            assert self._display_builder is not None
            traj = None
            if self._trajectory:
                traj = np.array(list(self._trajectory), dtype=np.float64)
            return self._display_builder.to_reference_map(
                session_id=session_id,
                trajectory=traj,
                metadata=metadata or {"mapping_version": 2, "slam": "pose_graph"},
            )

    def diagnostics(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "node_count": len(self._nodes),
                "edge_count": len(self._edges),
                "loop_closure_count": self._loop_closure_count,
                "optimize_count": self._optimize_count,
                "last_match_improvement": self._last_match_improvement,
                "last_match_accepted": self._last_match_accepted,
                "last_loop_closure_ts": self._last_loop_closure_ts,
            }

    def _add_node(
        self,
        ts: float,
        pose: PoseTuple,
        ranges: np.ndarray,
        angles: np.ndarray,
    ) -> int:
        node_id = len(self._nodes)
        self._nodes.append(
            GraphNode(
                node_id=node_id,
                ts=ts,
                x=pose[0],
                y=pose[1],
                theta=pose[2],
                ranges=ranges.copy(),
                angles=angles.copy(),
            ),
        )
        return node_id

    def _match_evidence(self) -> Tuple[np.ndarray, int]:
        """Occupancy evidence for scan match — reuses the display grid."""
        assert self._display_builder is not None
        evidence = np.maximum(self._display_builder.log_odds, 0.0).astype(np.float32)
        count = int((evidence > 0.05).sum())
        return evidence, count

    def _try_loop_closure(
        self,
        new_id: int,
        scan_xy: np.ndarray,
        corrected: PoseTuple,
    ) -> None:
        if len(self._nodes) < 5:
            return
        new_node = self._nodes[new_id]
        radius = self.slam_config.loop_closure_search_radius_m
        candidates: List[int] = []
        for node in self._nodes[:-3]:
            if math.hypot(node.x - corrected[0], node.y - corrected[1]) <= radius:
                candidates.append(node.node_id)
        if not candidates:
            return

        sub, evidence_count = self._match_evidence()
        if evidence_count < self.slam_config.min_submap_evidence_cells:
            return

        prior_pose = Pose2D(corrected[0], corrected[1], corrected[2])
        result = self._loop_matcher.search(
            scan_xy,
            prior_pose,
            sub,
            self._display_builder.origin_x_m,  # type: ignore[union-attr]
            self._display_builder.origin_y_m,  # type: ignore[union-attr]
            self._display_builder.resolution_m,  # type: ignore[union-attr]
        )
        if not result.accepted:
            return
        norm_score = result.score / max(1.0, float(scan_xy.shape[0]))
        if norm_score < self.slam_config.loop_closure_min_response:
            return

        best_id = min(
            candidates,
            key=lambda i: math.hypot(
                self._nodes[i].x - result.pose.x,
                self._nodes[i].y - result.pose.y,
            ),
        )
        anchor = self._nodes[best_id]
        rel_new = relative_pose(
            (anchor.x, anchor.y, anchor.theta),
            (result.pose.x, result.pose.y, result.pose.theta),
        )
        info = odom_information(
            rel_new[0],
            rel_new[2],
            alpha_trans=self.fusion_config.alpha_trans_per_m,
            alpha_rot_m=self.fusion_config.alpha_rot_per_m,
            alpha_rot_rad=self.fusion_config.alpha_rot_per_rad,
            match_response=max(1.0, result.improvement / 15.0),
        )
        self._edges.append(
            PoseGraphEdge(
                from_id=best_id,
                to_id=new_id,
                measurement=rel_new,
                information=info,
            ),
        )
        self._loop_closure_count += 1
        self._last_loop_closure_ts = new_node.ts

    def _optimize_graph(self) -> None:
        if len(self._nodes) < 2:
            return
        poses = np.array(
            [[n.x, n.y, n.theta] for n in self._nodes],
            dtype=np.float64,
        )
        optimize_pose_graph(poses, self._edges, anchor_id=0)
        for i, node in enumerate(self._nodes):
            node.x = float(poses[i, 0])
            node.y = float(poses[i, 1])
            node.theta = float(poses[i, 2])
        self._optimize_count += 1

    def _integrate_display_node(self, node: GraphNode) -> None:
        assert self._display_builder is not None
        self._display_builder.integrate_scan(
            node.ranges,
            node.angles,
            (node.x, node.y, node.theta),
        )

    def _rebuild_display_map(self) -> None:
        self._display_builder = OccupancyBuilder(
            extent_m=self.slam_config.extent_m,
            resolution_m=self.slam_config.resolution_m,
        )
        for node in self._nodes:
            self._integrate_display_node(node)
