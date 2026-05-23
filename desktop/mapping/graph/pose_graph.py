"""SE(2) pose-graph optimizer for mapping."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

Pose3 = Tuple[float, float, float]


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def relative_pose(p_i: Pose3, p_j: Pose3) -> Pose3:
    """Pose of j expressed in i's frame."""
    xi, yi, ti = p_i
    xj, yj, tj = p_j
    dx = xj - xi
    dy = yj - yi
    c = math.cos(ti)
    s = math.sin(ti)
    return (c * dx + s * dy, -s * dx + c * dy, _wrap(tj - ti))


def compose_pose(p_i: Pose3, rel: Pose3) -> Pose3:
    xi, yi, ti = p_i
    dx, dy, dt = rel
    c = math.cos(ti)
    s = math.sin(ti)
    return (xi + c * dx - s * dy, yi + s * dx + c * dy, _wrap(ti + dt))


@dataclass
class PoseGraphEdge:
    from_id: int
    to_id: int
    measurement: Pose3
    information: np.ndarray  # 3×3


def _edge_jacobian(p_i: Pose3, p_j: Pose3) -> Tuple[np.ndarray, np.ndarray]:
    """Jacobians of relative_pose w.r.t. p_i and p_j (3×3 each)."""
    xi, yi, ti = p_i
    xj, yj, _tj = p_j
    dx_w = xj - xi
    dy_w = yj - yi
    c = math.cos(ti)
    s = math.sin(ti)

    dr_dxi = np.array(
        [[-c, -s, -s * dx_w + c * dy_w], [s, -c, -c * dx_w - s * dy_w], [0.0, 0.0, -1.0]],
        dtype=np.float64,
    )
    dr_dxj = np.array(
        [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return dr_dxi, dr_dxj


def optimize_pose_graph(
    poses: np.ndarray,
    edges: List[PoseGraphEdge],
    *,
    anchor_id: int = 0,
    max_iterations: int = 20,
    damping: float = 1e-6,
) -> np.ndarray:
    """Gauss-Newton pose-graph optimization on SE(2).

    ``poses`` is (N, 3) modified in place and returned.
    Node ``anchor_id`` is fixed.
    """
    n = poses.shape[0]
    if n <= 1 or not edges:
        return poses

    for _ in range(max_iterations):
        H = np.zeros((3 * n, 3 * n), dtype=np.float64)
        b = np.zeros(3 * n, dtype=np.float64)
        total_err = 0.0

        for edge in edges:
            i = edge.from_id
            j = edge.to_id
            if i < 0 or j < 0 or i >= n or j >= n:
                continue
            p_i = tuple(poses[i])
            p_j = tuple(poses[j])
            pred = relative_pose(p_i, p_j)
            meas = edge.measurement
            err = np.array(
                [
                    meas[0] - pred[0],
                    meas[1] - pred[1],
                    _wrap(meas[2] - pred[2]),
                ],
                dtype=np.float64,
            )
            total_err += float(err @ err)
            J_i, J_j = _edge_jacobian(p_i, p_j)
            Omega = edge.information
            Ji_t = J_i.T @ Omega
            Jj_t = J_j.T @ Omega

            ii = slice(3 * i, 3 * i + 3)
            jj = slice(3 * j, 3 * j + 3)
            H[ii, ii] += Ji_t @ J_i
            H[ii, jj] += Ji_t @ J_j
            H[jj, ii] += Jj_t @ J_i
            H[jj, jj] += Jj_t @ J_j
            b[ii] += Ji_t @ err
            b[jj] += Jj_t @ err

        if total_err < 1e-10:
            break

        free = [k for k in range(n) if k != anchor_id]
        if not free:
            break
        idx = np.array([3 * k + d for k in free for d in range(3)], dtype=int)
        H_sub = H[np.ix_(idx, idx)] + damping * np.eye(len(idx))
        b_sub = b[idx]
        try:
            delta = np.linalg.solve(H_sub, b_sub)
        except np.linalg.LinAlgError:
            break

        if float(np.max(np.abs(delta))) < 1e-5:
            break

        for k, node in enumerate(free):
            poses[node, 0] += delta[3 * k + 0]
            poses[node, 1] += delta[3 * k + 1]
            poses[node, 2] = _wrap(poses[node, 2] + delta[3 * k + 2])

    return poses


def odom_information(
    ds: float,
    dtheta: float,
    *,
    alpha_trans: float,
    alpha_rot_m: float,
    alpha_rot_rad: float,
    match_response: float = 1.0,
) -> np.ndarray:
    """Build 3×3 information matrix for an odometry edge."""
    sigma_trans = alpha_trans * abs(ds) + alpha_rot_m * abs(dtheta)
    sigma_rot = alpha_rot_m * abs(ds) + alpha_rot_rad * abs(dtheta)
    sigma_trans = max(sigma_trans, 0.02)
    sigma_rot = max(sigma_rot, 0.01)
    scale = max(0.5, min(2.0, match_response))
    return np.diag(
        [
            scale / (sigma_trans ** 2),
            scale / (sigma_trans ** 2),
            scale / (sigma_rot ** 2),
        ],
    ).astype(np.float64)
