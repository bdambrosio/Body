"""Tests for PoseGraphMapper and pose-graph optimizer."""

from __future__ import annotations

import math
import unittest
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from desktop.fusion.load_slam_config import FusionNoiseConfig, SlamConfig
from desktop.mapping.graph.pose_graph import (
    PoseGraphEdge,
    optimize_pose_graph,
    relative_pose,
)
from desktop.mapping.pose_graph_mapper import PoseGraphMapper
from desktop.world_map.pose_source import Pose

PoseTuple = Tuple[float, float, float]


class _MockEkf:
    """Minimal EKF stand-in for synthetic SLAM tests."""

    def __init__(self, trajectory: List[Tuple[float, PoseTuple]]) -> None:
        self._trajectory = trajectory
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    def pose_at(self, ts: float) -> Optional[Pose]:
        best: Optional[Tuple[float, PoseTuple]] = None
        for t, pose in self._trajectory:
            if best is None or abs(t - ts) < abs(best[0] - ts):
                best = (t, pose)
        return best[1] if best is not None else None

    def diagnostics(self) -> Dict[str, Any]:
        return {"imu_settled": True, "heading_source": "imu"}


def _corridor_scan(y_pose: float, *, wall_x: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """Single wall parallel to y-axis at x=wall_x in world frame."""
    n = 36
    angles = np.linspace(-math.pi / 2, math.pi / 2, n)
    ranges = []
    for a in angles:
        cos_a = math.cos(a)
        sin_a = math.sin(a)
        if abs(cos_a) < 1e-3:
            ranges.append(np.nan)
            continue
        t = (wall_x - 0.0) / cos_a
        wy = y_pose + t * sin_a
        if t > 0 and abs(wy) < 0.5:
            ranges.append(t)
        else:
            ranges.append(np.nan)
    return np.asarray(ranges, dtype=np.float64), angles


class TestPoseGraphOptimizer(unittest.TestCase):
    def test_closes_square_loop(self) -> None:
        poses = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, math.pi / 2],
                [0.0, 1.0, math.pi],
                [0.05, 0.05, -math.pi / 2],
            ],
            dtype=np.float64,
        )
        edges: List[PoseGraphEdge] = []
        for i in range(4):
            j = i + 1
            edges.append(
                PoseGraphEdge(
                    from_id=i,
                    to_id=j,
                    measurement=relative_pose(tuple(poses[i]), tuple(poses[j])),
                    information=np.diag([100.0, 100.0, 100.0]),
                ),
            )
        edges.append(
            PoseGraphEdge(
                from_id=0,
                to_id=4,
                measurement=relative_pose(tuple(poses[0]), tuple(poses[4])),
                information=np.diag([50.0, 50.0, 50.0]),
            ),
        )
        optimize_pose_graph(poses, edges, anchor_id=0)
        self.assertLess(abs(poses[4, 0]), 0.08)
        self.assertLess(abs(poses[4, 1]), 0.08)


class TestPoseGraphMapper(unittest.TestCase):
    def test_corridor_out_and_back_single_wall(self) -> None:
        slam = SlamConfig(
            match_hz=100.0,
            graph_optimize_every_n_nodes=3,
            submap_node_count=30,
            min_submap_evidence_cells=10,
            min_improvement=0.0,
            loop_closure_search_radius_m=3.0,
            loop_closure_min_response=0.01,
        )
        fusion = FusionNoiseConfig()
        traj: List[Tuple[float, PoseTuple]] = []
        ts = 0.0
        for y in np.linspace(0.0, 3.0, 8):
            traj.append((ts, (0.0, float(y), 0.0)))
            ts += 0.5
        for y in np.linspace(3.0, 0.0, 8):
            traj.append((ts, (0.0, float(y), math.pi)))
            ts += 0.5

        ekf = _MockEkf(traj)
        mapper = PoseGraphMapper(slam_config=slam, fusion_config=fusion, ekf=ekf)  # type: ignore[arg-type]

        for t, (_x, y, _th) in traj:
            mapper._last_match_mono = 0.0
            ranges, angles = _corridor_scan(y)
            mapper.add_scan(t, ranges, angles)

        ref = mapper.to_reference_map(session_id="test")
        occ = ref.driveable_int8()
        wall_cols = np.where(np.any(occ == 0, axis=0))[0]
        self.assertGreater(len(wall_cols), 0)
        self.assertGreater(mapper.diagnostics()["node_count"], 4)


if __name__ == "__main__":
    unittest.main()
