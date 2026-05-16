#!/usr/bin/env python3
"""Phase 1 demo: visualize the scan-matcher likelihood field.

For each of three hand-built scenes (corridor, open room, symmetric
room), generate a synthetic evidence grid + a scan taken from inside it,
run ``ScanMatcher.search(return_field=True)``, and plot 2D marginals
of the resulting score field:

  * Top row: scene layout (walls in grey, true robot in red).
  * Middle row: (dx, dy) marginal of the field at argmax dθ. Bright =
    high correlation. The hand-built priors are intentionally a few cm
    off truth so the peak sits inside the window but not at center.
  * Bottom row: dθ profile through argmax (dx, dy).

The third scene is the diagnostic: a symmetric room. Phase 0 §4
predicted the matcher's likelihood-field representation would expose
the 180°-flip basin instead of argmaxing it away. The dθ profile
should show two peaks ~π apart, with comparable amplitude.

Usage:
    PYTHONPATH=. python3 scripts/phase1_likelihood_field_demo.py \\
        [--out PATH]  [--show]

Outputs a single PNG (default ``phase1_likelihood_field.png`` next to
the script) and prints summary statistics to stdout.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

from desktop.nav.slam.scan_matcher import ScanMatcher, ScanMatcherConfig
from desktop.nav.slam.types import Pose2D


RESOLUTION_M = 0.04  # matches local_map / world_map config


@dataclass
class Scene:
    name: str
    extent_x_m: float
    extent_y_m: float
    truth_pose: Pose2D
    prior_pose: Pose2D
    build_walls: Callable[[np.ndarray], None]  # mutates evidence in place

    def origin(self) -> Tuple[float, float]:
        return (-self.extent_x_m / 2.0, -self.extent_y_m / 2.0)

    def grid_shape(self) -> Tuple[int, int]:
        return (
            int(round(self.extent_x_m / RESOLUTION_M)),
            int(round(self.extent_y_m / RESOLUTION_M)),
        )

    def build_evidence(self) -> np.ndarray:
        ev = np.zeros(self.grid_shape(), dtype=np.float32)
        self.build_walls(ev)
        return ev

    def evidence_world_points(self, ev: np.ndarray) -> np.ndarray:
        """All cells in `ev` with positive weight, returned as world (x, y)."""
        ox, oy = self.origin()
        ii, jj = np.where(ev > 0)
        x = ox + (ii + 0.5) * RESOLUTION_M
        y = oy + (jj + 0.5) * RESOLUTION_M
        return np.stack([x, y], axis=-1)


def _box_perimeter(ev: np.ndarray, weight: float = 10.0) -> None:
    ev[0, :] = weight
    ev[-1, :] = weight
    ev[:, 0] = weight
    ev[:, -1] = weight


def _scene_corridor() -> Scene:
    # Long thin corridor: 8 m × 1.2 m. Translational uncertainty along
    # the length is high; lateral is well-constrained.
    return Scene(
        name="corridor",
        extent_x_m=8.0,
        extent_y_m=1.2,
        truth_pose=Pose2D(x=0.0, y=0.0, theta=0.0),
        prior_pose=Pose2D(x=0.06, y=-0.04, theta=math.radians(2.0)),
        build_walls=_box_perimeter,
    )


def _scene_open_room() -> Scene:
    # Plain rectangular room — both axes well-constrained.
    return Scene(
        name="open_room",
        extent_x_m=6.0,
        extent_y_m=4.0,
        truth_pose=Pose2D(x=0.0, y=0.0, theta=0.0),
        prior_pose=Pose2D(x=0.08, y=-0.06, theta=math.radians(3.0)),
        build_walls=_box_perimeter,
    )


def _scene_symmetric_room() -> Scene:
    # Square room — 180°-flip basin should be visible in dθ profile.
    return Scene(
        name="symmetric_room",
        extent_x_m=4.0,
        extent_y_m=4.0,
        truth_pose=Pose2D(x=0.0, y=0.0, theta=0.0),
        # Prior intentionally biased away from the 180° flip so the
        # matcher's argmax picks the correct orientation. The field
        # should still show evidence of the basin at ±π.
        prior_pose=Pose2D(x=0.04, y=-0.02, theta=math.radians(2.0)),
        build_walls=_box_perimeter,
    )


def _scan_from_truth(
    ev: np.ndarray, scene: Scene, max_range_m: float = 5.0,
) -> np.ndarray:
    """Take a 360° lidar sample from the truth pose — every wall cell that
    is within range becomes one returned point, expressed in body frame.

    Cheap synthetic substitute for a ray-cast: skips occlusion modeling,
    which is OK because the rooms here are convex and the truth pose is
    inside. For non-convex scenes you would want a real ray-cast."""
    world_pts = scene.evidence_world_points(ev)
    dx = world_pts[:, 0] - scene.truth_pose.x
    dy = world_pts[:, 1] - scene.truth_pose.y
    r = np.hypot(dx, dy)
    mask = (r > 0.05) & (r <= max_range_m)
    dx, dy = dx[mask], dy[mask]
    # Rotate into body frame: R(-theta).
    th = -scene.truth_pose.theta
    c, s = math.cos(th), math.sin(th)
    bx = c * dx - s * dy
    by = s * dx + c * dy
    return np.stack([bx, by], axis=-1)


def _wider_search_config() -> ScanMatcherConfig:
    # Wider theta window so the symmetric-room flip basin is inside the
    # search bounds at all. Default ±8° wouldn't see it.
    return ScanMatcherConfig(
        xy_half_m=0.30,
        theta_half_rad=math.radians(180.0),
        xy_step_m=RESOLUTION_M,
        theta_step_rad=math.radians(3.0),
        min_improvement=5.0,
    )


def _narrow_search_config() -> ScanMatcherConfig:
    # Standard runtime window — what the matcher actually uses in fuser
    # threads. Good for the corridor / open-room subplots.
    return ScanMatcherConfig(
        xy_half_m=0.30,
        theta_half_rad=math.radians(8.0),
        xy_step_m=RESOLUTION_M,
        theta_step_rad=math.radians(1.0),
        min_improvement=5.0,
    )


def _plot_scene(
    fig, gs_col, scene: Scene, ev: np.ndarray, result, narrow: bool,
):
    import matplotlib.pyplot as plt

    sf = result.score_field
    ox, oy = scene.origin()

    # ── Row 0: scene layout ──────────────────────────────────────
    ax0 = fig.add_subplot(gs_col[0])
    ax0.set_title(f"{scene.name}", fontsize=10)
    ax0.imshow(
        ev.T, origin="lower",
        extent=[ox, ox + scene.extent_x_m, oy, oy + scene.extent_y_m],
        cmap="Greys", vmin=0, vmax=10.0,
    )
    ax0.plot(
        scene.truth_pose.x, scene.truth_pose.y, "ro", markersize=4,
        label="truth",
    )
    # Draw heading arrow.
    L = 0.3
    ax0.arrow(
        scene.truth_pose.x, scene.truth_pose.y,
        L * math.cos(scene.truth_pose.theta),
        L * math.sin(scene.truth_pose.theta),
        head_width=0.08, color="r",
    )
    ax0.plot(
        scene.prior_pose.x, scene.prior_pose.y, "b+", markersize=6,
        label="prior",
    )
    ax0.set_xlabel("x (m)")
    ax0.set_ylabel("y (m)")
    ax0.set_aspect("equal")
    ax0.legend(fontsize=7, loc="lower left")

    # ── Row 1: (dx, dy) marginal at argmax dθ ───────────────────
    ax1 = fig.add_subplot(gs_col[1])
    flat_idx = int(np.argmax(sf.field))
    ix_max, iy_max, ith_max = np.unravel_index(flat_idx, sf.field.shape)
    slab = sf.field[:, :, ith_max]  # (Nx, Ny)
    im = ax1.imshow(
        slab.T, origin="lower",
        extent=[
            float(sf.dx_axis[0]), float(sf.dx_axis[-1]),
            float(sf.dy_axis[0]), float(sf.dy_axis[-1]),
        ],
        cmap="viridis", aspect="equal",
    )
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    ax1.set_title(
        f"(dx, dy) slice @ dθ = {math.degrees(float(sf.dth_axis[ith_max])):+.1f}°",
        fontsize=9,
    )
    ax1.plot(0.0, 0.0, "kx", markersize=6, label="prior")
    ax1.plot(
        float(sf.dx_axis[ix_max]), float(sf.dy_axis[iy_max]),
        "r*", markersize=10, label="argmax",
    )
    ax1.set_xlabel("dx (m)")
    ax1.set_ylabel("dy (m)")
    ax1.legend(fontsize=7, loc="lower left")

    # ── Row 2: dθ profile through argmax (dx, dy) ───────────────
    ax2 = fig.add_subplot(gs_col[2])
    profile = sf.field[ix_max, iy_max, :]
    ax2.plot(np.degrees(sf.dth_axis), profile, "-o", markersize=3)
    ax2.axvline(
        math.degrees(float(sf.dth_axis[ith_max])),
        color="r", linestyle="--", linewidth=1, label="argmax dθ",
    )
    ax2.set_xlabel("dθ (deg)")
    ax2.set_ylabel("score")
    ax2.set_title(
        f"dθ profile @ (dx, dy) = "
        f"({float(sf.dx_axis[ix_max]):+.02f}, {float(sf.dy_axis[iy_max]):+.02f})",
        fontsize=9,
    )
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7)

    # ── Summary stats to stdout ─────────────────────────────────
    field_max = float(sf.field.max())
    field_argmax_pose = (
        scene.prior_pose.x + float(sf.dx_axis[ix_max]),
        scene.prior_pose.y + float(sf.dy_axis[iy_max]),
        scene.prior_pose.theta + float(sf.dth_axis[ith_max]),
    )
    print(f"[{scene.name}]")
    print(f"  prior          : {scene.prior_pose.as_tuple()}")
    print(f"  truth          : {scene.truth_pose.as_tuple()}")
    print(f"  argmax pose    : {field_argmax_pose}")
    print(f"  field shape    : {tuple(sf.field.shape)}  "
          f"(narrow={narrow})")
    print(f"  field max/mean : {field_max:.1f} / {float(sf.field.mean()):.2f}")
    print(f"  result.score   : {result.score:.1f}  prior {result.score_prior:.1f}")
    print(f"  improvement    : {result.improvement:.1f}  accepted={result.accepted}")
    # For the symmetric room: how strong is the secondary peak?
    if not narrow:
        # Average over (dx, dy) → marginal over dθ.
        dth_marginal = sf.field.mean(axis=(0, 1))
        peak_dth = float(sf.dth_axis[int(np.argmax(dth_marginal))])
        # Find the largest value at least 90° from the peak.
        away_mask = np.abs(sf.dth_axis - peak_dth) > math.radians(90.0)
        if away_mask.any():
            sec = float(dth_marginal[away_mask].max())
            best = float(dth_marginal.max())
            ratio = sec / best if best > 0 else 0.0
            print(f"  secondary/peak θ-marginal ratio: {ratio:.3f}  "
                  f"(>0.5 ≈ symmetric basin)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parent.parent / "phase1_likelihood_field.png",
        help="Where to save the figure. Default: project root.",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Also pop up the figure interactively (requires display).",
    )
    args = parser.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenes = [
        (_scene_corridor(), _narrow_search_config(), True),
        (_scene_open_room(), _narrow_search_config(), True),
        # The symmetric scene needs the wide θ window or the flip basin
        # is literally off the field.
        (_scene_symmetric_room(), _wider_search_config(), False),
    ]

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(nrows=3, ncols=len(scenes), hspace=0.45, wspace=0.35)

    for col, (scene, cfg, narrow) in enumerate(scenes):
        matcher = ScanMatcher(cfg)
        ev = scene.build_evidence()
        scan_body = _scan_from_truth(ev, scene)
        ox, oy = scene.origin()
        result = matcher.search(
            scan_body, scene.prior_pose, ev,
            ox, oy, RESOLUTION_M, return_field=True,
        )
        col_specs = [gs[r, col] for r in range(3)]
        _plot_scene(fig, col_specs, scene, ev, result, narrow)

    fig.suptitle(
        "Phase 1 — scan-matcher likelihood field, three synthetic scenes",
        fontsize=12,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"\nWrote {args.out}")

    if args.show:
        plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
