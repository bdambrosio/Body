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

    def _mapper(self, traj):
        slam = SlamConfig(match_hz=100.0, min_submap_evidence_cells=10**9,
                          graph_optimize_every_n_nodes=1000)
        return PoseGraphMapper(slam_config=slam, fusion_config=FusionNoiseConfig(),
                               ekf=_MockEkf(traj))  # type: ignore[arg-type]

    def test_prior_composes_last_node_with_ekf_delta(self) -> None:
        # S3: once the graph has corrected a node, the next prior must carry
        # that correction forward (last node ⊕ EKF delta), not snap back to
        # the absolute EKF pose.
        traj = [(0.0, (0.0, 0.0, 0.0)), (1.0, (1.0, 0.0, 0.0))]
        mapper = self._mapper(traj)
        ranges, angles = _corridor_scan(0.0)
        self.assertTrue(mapper.add_scan(0.0, ranges, angles))
        node0 = mapper._nodes[0]
        node0.x, node0.y, node0.theta = 0.0, 0.4, math.pi / 2   # graph correction
        mapper._last_match_mono = 0.0
        self.assertTrue(mapper.add_scan(1.0, ranges, angles))
        n1 = mapper._nodes[1]
        # Sparse submap → no match → node at the prior: 1 m along node0's
        # (corrected) heading.
        self.assertAlmostEqual(n1.x, 0.0, places=6)
        self.assertAlmostEqual(n1.y, 1.4, places=6)
        self.assertAlmostEqual(n1.theta, math.pi / 2, places=6)

    def test_rate_slot_not_consumed_on_failed_lookup(self) -> None:
        # S9: a scan past the EKF tail must not burn the rate slot.
        class _Ekf(_MockEkf):
            def pose_at(self, ts):
                return None if ts > 5.0 else (0.0, 0.0, 0.0)
        slam = SlamConfig(match_hz=0.5, min_submap_evidence_cells=10**9)
        mapper = PoseGraphMapper(slam_config=slam, fusion_config=FusionNoiseConfig(),
                                 ekf=_Ekf([]))  # type: ignore[arg-type]
        ranges, angles = _corridor_scan(0.0)
        self.assertFalse(mapper.add_scan(9.0, ranges, angles))   # lookup fails
        self.assertTrue(mapper.add_scan(1.0, ranges, angles))    # slot still free


if __name__ == "__main__":
    unittest.main()


class TestKeyframeGate(unittest.TestCase):
    """Mapping plan Stage 0: no node without motion; edge matches rejected;
    export without closures skips the optimiser."""

    def _mapper(self, traj, **kw):
        kw.setdefault("match_hz", 100.0)
        slam = SlamConfig(min_submap_evidence_cells=10**9, **kw)
        return PoseGraphMapper(slam_config=slam, fusion_config=FusionNoiseConfig(),
                               ekf=_MockEkf(traj))  # type: ignore[arg-type]

    def test_stationary_scans_add_no_nodes(self):
        traj = [(float(t), (0.0, 0.0, 0.0)) for t in range(10)] + [(10.0, (0.5, 0.0, 0.0))]
        mapper = self._mapper(traj)
        ranges, angles = _corridor_scan(0.0)
        self.assertTrue(mapper.add_scan(0.0, ranges, angles))
        for t in range(1, 10):
            mapper._last_match_mono = 0.0
            self.assertFalse(mapper.add_scan(float(t), ranges, angles))
        mapper._last_match_mono = 0.0
        self.assertTrue(mapper.add_scan(10.0, ranges, angles))
        self.assertEqual(mapper.diagnostics()["node_count"], 2)
        self.assertEqual(mapper.diagnostics()["keyframes_skipped"], 9)

    def test_rotation_alone_is_a_keyframe(self):
        traj = [(0.0, (0.0, 0.0, 0.0)), (1.0, (0.0, 0.0, math.radians(6.0)))]
        mapper = self._mapper(traj)
        ranges, angles = _corridor_scan(0.0)
        self.assertTrue(mapper.add_scan(0.0, ranges, angles))
        mapper._last_match_mono = 0.0
        self.assertTrue(mapper.add_scan(1.0, ranges, angles))

    def test_replay_clock(self):
        traj = [(0.0, (0.0, 0.0, 0.0)), (0.1, (0.5, 0.0, 0.0)), (0.2, (1.0, 0.0, 0.0))]
        mapper = self._mapper(traj, match_hz=2.0)
        clock = {"t": 0.0}
        mapper._clock = lambda: clock["t"]
        ranges, angles = _corridor_scan(0.0)
        self.assertTrue(mapper.add_scan(0.0, ranges, angles))
        clock["t"] = 0.1
        self.assertFalse(mapper.add_scan(0.1, ranges, angles))   # < 1/hz on the scan clock
        clock["t"] = 0.6
        self.assertTrue(mapper.add_scan(0.2, ranges, angles))

    def test_export_without_closures_skips_optimiser(self):
        traj = [(0.0, (0.0, 0.0, 0.0)), (1.0, (0.5, 0.0, 0.0))]
        mapper = self._mapper(traj)
        ranges, angles = _corridor_scan(0.0)
        mapper.add_scan(0.0, ranges, angles)
        mapper._last_match_mono = 0.0
        mapper.add_scan(1.0, ranges, angles)
        ref = mapper.to_reference_map(session_id="t")
        self.assertEqual(mapper.diagnostics()["optimize_count"], 0)
        self.assertEqual(ref.trajectory.shape, (2, 4))

    def test_window_edge_match_rejected(self):
        from types import SimpleNamespace
        from desktop.nav.slam.types import Pose2D
        traj = [(0.0, (0.0, 0.0, 0.0)), (1.0, (0.5, 0.0, 0.0))]
        slam = SlamConfig(match_hz=100.0, min_submap_evidence_cells=0)
        mapper = PoseGraphMapper(slam_config=slam, fusion_config=FusionNoiseConfig(),
                                 ekf=_MockEkf(traj))  # type: ignore[arg-type]

        class _EdgeMatcher:
            def search(self, *a, **k):
                return SimpleNamespace(pose=Pose2D(9.0, 9.0, 0.0), improvement=50.0,
                                       score=100.0, score_prior=50.0, accepted=True,
                                       search_exhausted=True)
        mapper._matcher = _EdgeMatcher()
        ranges, angles = _corridor_scan(0.0)
        mapper.add_scan(0.0, ranges, angles)
        mapper._last_match_mono = 0.0
        mapper.add_scan(1.0, ranges, angles)
        n1 = mapper._nodes[1]
        self.assertAlmostEqual(n1.x, 0.5, places=6)       # prior kept, not 9.0
        self.assertEqual(mapper.diagnostics()["edge_rejects"], 1)

    def test_edge_match_recentred_then_accepted(self):
        """A correction bigger than the window: the first search ends on
        the edge, the re-centred one settles inside → accepted."""
        from types import SimpleNamespace
        from desktop.nav.slam.types import Pose2D
        traj = [(0.0, (0.0, 0.0, 0.0)), (1.0, (0.5, 0.0, 0.0))]
        slam = SlamConfig(match_hz=100.0, min_submap_evidence_cells=0)
        mapper = PoseGraphMapper(slam_config=slam, fusion_config=FusionNoiseConfig(),
                                 ekf=_MockEkf(traj))  # type: ignore[arg-type]
        calls = []

        class _Matcher:
            def search(self, pts, prior, *a, **k):
                calls.append((prior.x, prior.y, prior.theta))
                edge = len(calls) == 1
                return SimpleNamespace(
                    pose=Pose2D(prior.x, prior.y, prior.theta + (0.14 if edge else 0.12)),
                    improvement=50.0, score=100.0, score_prior=50.0, accepted=True,
                    search_exhausted=edge)
        mapper._matcher = _Matcher()
        ranges, angles = _corridor_scan(0.0)
        mapper.add_scan(0.0, ranges, angles)
        mapper._last_match_mono = 0.0
        mapper.add_scan(1.0, ranges, angles)
        self.assertEqual(len(calls), 2)
        self.assertAlmostEqual(calls[1][2], 0.14, places=6)       # re-centred on the edge pose
        self.assertAlmostEqual(mapper._nodes[1].theta, 0.26, places=6)
        self.assertEqual(mapper.diagnostics()["edge_rejects"], 0)
