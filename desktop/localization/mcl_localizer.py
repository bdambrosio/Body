"""Monte Carlo localization against a frozen ReferenceMap."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from desktop.nav.slam.scan_matcher import lidar_scan_to_xy
from desktop.reference_map.reference_map import ReferenceMap
from desktop.world_map.particle_filter_pose import (
    ParticleFilterConfig,
    ParticleFilterPose,
)


@dataclass
class MCLConfig:
    scan_min_range_m: float = 0.15
    scan_max_range_m: float = 5.0
    scan_stride: int = 2
    beam_log_eps: float = 1e-4
    relocate_spray_xy_m: float = 3.0
    relocate_spray_theta_rad: float = math.pi
    relocate_seed_sigma_xy_m: float = 0.5
    relocate_seed_sigma_theta_rad: float = math.radians(20.0)


class MCLLocalizer:
    """Single-filter MCL with likelihood-field beam model."""

    def __init__(
        self,
        reference_map: ReferenceMap,
        *,
        pf_config: Optional[ParticleFilterConfig] = None,
        config: Optional[MCLConfig] = None,
    ) -> None:
        self._map = reference_map
        self._pf = ParticleFilterPose(pf_config)
        self._config = config or MCLConfig()
        self._likelihood_t: Optional[torch.Tensor] = None
        self._refresh_map_tensors()

    @property
    def filter(self) -> ParticleFilterPose:
        return self._pf

    @property
    def reference_map(self) -> ReferenceMap:
        return self._map

    def set_reference_map(self, reference_map: ReferenceMap) -> None:
        self._map = reference_map
        self._refresh_map_tensors()

    def _refresh_map_tensors(self) -> None:
        cfg = self._pf.cfg
        field = self._map.likelihood_field.astype(np.float32)
        self._likelihood_t = torch.as_tensor(
            field, device=cfg.device, dtype=torch.float32,
        )
        self._origin_x = float(self._map.origin_x_m)
        self._origin_y = float(self._map.origin_y_m)
        self._res = float(self._map.resolution_m)
        self._nx = int(self._map.nx)
        self._ny = int(self._map.ny)

    def seed_at(
        self,
        x: float,
        y: float,
        theta: float,
        *,
        sigma_xy_m: Optional[float] = None,
        sigma_theta_rad: Optional[float] = None,
    ) -> None:
        self._pf.seed_at(
            x, y, theta,
            sigma_xy_m=sigma_xy_m,
            sigma_theta_rad=sigma_theta_rad,
        )

    def predict(self, delta_s: float, delta_theta: float) -> None:
        self._pf.predict(delta_s, delta_theta)

    def observe_imu_yaw(
        self, world_yaw: float, sigma_rad: Optional[float] = None,
    ) -> None:
        self._pf.observe_imu_yaw(world_yaw, sigma_rad=sigma_rad)

    def maybe_resample(self) -> bool:
        return self._pf.maybe_resample()

    def posterior_mean(self) -> Tuple[float, float, float]:
        return self._pf.posterior_mean()

    def posterior_cov(self) -> np.ndarray:
        return self._pf.posterior_cov().detach().cpu().numpy()

    def observe_scan_ranges(
        self,
        ranges_m: np.ndarray,
        angles_rad: np.ndarray,
    ) -> None:
        """Likelihood-field beam model update."""
        points = lidar_scan_to_xy(ranges_m, angles_rad)
        if points.shape[0] < 5:
            return
        cfg = self._config
        r = np.hypot(points[:, 0], points[:, 1])
        mask = (r >= cfg.scan_min_range_m) & (r <= cfg.scan_max_range_m)
        points = points[mask]
        if cfg.scan_stride > 1:
            points = points[:: cfg.scan_stride]
        if points.shape[0] < 5:
            return
        self._observe_points_body(points)

    def _observe_points_body(self, points_xy: np.ndarray) -> None:
        assert self._likelihood_t is not None
        self._pf._require_seeded()
        assert self._pf._state is not None and self._pf._log_w is not None

        pf_cfg = self._pf.cfg
        pts = torch.as_tensor(
            points_xy, device=pf_cfg.device, dtype=pf_cfg.state_dtype,
        )
        state = self._pf._state
        P = state.shape[0]
        n_pts = pts.shape[0]

        cos_t = torch.cos(state[:, 2])
        sin_t = torch.sin(state[:, 2])
        # (P, N) world endpoints
        wx = (
            state[:, 0].unsqueeze(1)
            + cos_t.unsqueeze(1) * pts[:, 0]
            - sin_t.unsqueeze(1) * pts[:, 1]
        )
        wy = (
            state[:, 1].unsqueeze(1)
            + sin_t.unsqueeze(1) * pts[:, 0]
            + cos_t.unsqueeze(1) * pts[:, 1]
        )

        ix = torch.floor(
            (wx - self._origin_x) / self._res + 1e-9,
        ).to(torch.int64)
        iy = torch.floor(
            (wy - self._origin_y) / self._res + 1e-9,
        ).to(torch.int64)
        in_bounds = (
            (ix >= 0) & (ix < self._nx) & (iy >= 0) & (iy < self._ny)
        )
        ix_safe = ix.clamp(0, self._nx - 1)
        iy_safe = iy.clamp(0, self._ny - 1)
        # Gather likelihood at each (particle, point).
        lik = self._likelihood_t[ix_safe, iy_safe]
        lik = torch.where(in_bounds, lik, torch.zeros_like(lik))
        lik = lik.clamp(min=self._config.beam_log_eps)
        # Mean log-likelihood across rays per particle.
        log_lik = torch.log(lik).mean(dim=1)
        self._pf._log_w = self._pf._log_w + log_lik.to(pf_cfg.weight_dtype)

    def spray_particles(
        self,
        center_x: float,
        center_y: float,
        center_theta: float,
        *,
        sigma_xy_m: Optional[float] = None,
        sigma_theta_rad: Optional[float] = None,
    ) -> None:
        sx = (
            sigma_xy_m
            if sigma_xy_m is not None
            else self._config.relocate_seed_sigma_xy_m
        )
        st = (
            sigma_theta_rad
            if sigma_theta_rad is not None
            else self._config.relocate_seed_sigma_theta_rad
        )
        self._pf.seed_at(center_x, center_y, center_theta, sigma_xy_m=sx, sigma_theta_rad=st)

    def n_eff(self) -> float:
        return self._pf.n_eff()
