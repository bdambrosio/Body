"""Differential-drive kinematics (twist ↔ wheel speeds, odometry integration)."""

from __future__ import annotations

import math
from typing import NamedTuple


class Pose(NamedTuple):
    x: float
    y: float
    theta: float


def twist_to_wheel_velocities(linear_ms: float, angular_rads: float, wheel_base_m: float) -> tuple[float, float]:
    """Return (left_mps, right_mps) at wheel contact; positive = forward."""
    if wheel_base_m <= 0.0:
        return linear_ms, linear_ms
    v_left = linear_ms - (angular_rads * wheel_base_m / 2.0)
    v_right = linear_ms + (angular_rads * wheel_base_m / 2.0)
    return v_left, v_right


def pwm_from_velocity(velocity_ms: float, max_wheel_vel_ms: float) -> tuple[float, str]:
    """Map signed wheel velocity to duty [-1,1] and direction label."""
    if max_wheel_vel_ms <= 0.0:
        return 0.0, "fwd"
    pwm = velocity_ms / max_wheel_vel_ms
    if pwm > 1.0:
        pwm = 1.0
    elif pwm < -1.0:
        pwm = -1.0
    direction = "fwd" if pwm >= 0 else "rev"
    return abs(pwm), direction


def integrate_odometry(
    pose: Pose,
    left_delta_m: float,
    right_delta_m: float,
    wheel_base_m: float,
) -> Pose:
    """Dead-reckon from left/right arc length over one step."""
    if wheel_base_m <= 0.0:
        d_center = (left_delta_m + right_delta_m) / 2.0
        return Pose(pose.x + d_center * math.cos(pose.theta), pose.y + d_center * math.sin(pose.theta), pose.theta)

    d_center = (left_delta_m + right_delta_m) / 2.0
    d_theta = (right_delta_m - left_delta_m) / wheel_base_m
    theta_mid = pose.theta + d_theta / 2.0
    x = pose.x + d_center * math.cos(theta_mid)
    y = pose.y + d_center * math.sin(theta_mid)
    theta = math.atan2(math.sin(pose.theta + d_theta), math.cos(pose.theta + d_theta))
    return Pose(x, y, theta)


def ticks_to_delta_m(delta_ticks: int, wheel_radius_m: float, ticks_per_rev: int) -> float:
    if ticks_per_rev <= 0:
        return 0.0
    return (delta_ticks / float(ticks_per_rev)) * (2.0 * math.pi * wheel_radius_m)
