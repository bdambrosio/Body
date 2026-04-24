# SLAM Promotion — Pi-side contract

**Project:** Body
**Date:** 2026-04-24
**Scope:** What the Pi side needs to deliver for desktop SLAM to be promoted from shadow mode to a production pose source (`ImuPlusScanMatchPose` replacing `OdomPose` in `desktop/world_map/controller.py`). Companion to `docs/imu_integration_spec.md` (IMU wire contract) and `docs/world_map_spec.md` (fuser architecture).

---

## 1. Division of labor

Desktop owns the **pose estimate** once SLAM is enabled. Specifically, `ImuPlusScanMatchPose` consumes three Pi-published topics and produces the corrected world-frame pose that the world-map fuser anchors each `body/local_map` frame against:

- `body/imu` — yaw (from fused quaternion)
- `body/odom` — translation (from wheel encoders)
- `body/lidar/scan` — periodic position + yaw corrections via scan-match against the accumulated `WorldGrid`

The Pi does **not** need to fuse IMU into odom. `body/odom.theta` stays encoder-derived; the desktop substitutes IMU yaw in the pose combiner. That keeps the Pi simpler and lets desktop swap its pose model without Pi changes.

## 2. What must be true before flipping the `--slam` flag

### 2.1 BNO085 calibrated and in `rotation_vector` mode

- `body/imu.fusion.mode == "rotation_vector"` — absolute yaw. Bounded drift relative to magnetic north.
- `body/imu.fusion.accuracy_rad` stable and small — `< ~0.06` rad (3.4°) once warm. The desktop `ImuYawTracker` already refuses to answer queries until `min_settle_samples` consecutive readings fall below `settle_accuracy_rad`, so startup transients are handled; steady-state accuracy is what matters for map quality.
- `body/imu.fusion.mag_status == "calibrated"` during normal driving. If the magnetometer gets disturbed by motor current and drops to `uncalibrated`, the scan-matcher has to close more drift but won't fail.

`game_rotation_vector` mode still works — yaw is relative (starts at zero at boot, drifts ~0.5–1°/min) — but the absolute-heading advantage is gone. Fine for testing, less good for multi-room or long-run mapping.

### 2.2 `body/odom.source == "wheel_encoders"`

- Encoders wired and reading correctly (right-encoder sign fix in commit `88bad74` is required).
- PID closed-loop tracking is on (commit `9b8ee75` + `5b61985`), so commanded vs. actual wheel velocity agree within a few % at steady state.
- `left_ticks` / `right_ticks` monotonically increase forward, decrement reverse. (Assumed; the desktop doesn't currently reconstruct from ticks, but the scan-match fallback path might want to.)

If the Pi falls back to `source == "commanded_vel_playback"`, desktop should **disable SLAM** and run odom-only — scan-matching against a drifting encoder-less pose will produce worse results than plain odom.

### 2.3 `body/lidar/scan` healthy

- Publishing at roughly `10 Hz` (LD19 default), `scan_time_ms` ≈ 100.
- `angle_min`, `angle_max`, `angle_increment` set consistently frame-to-frame; desktop caches these at first scan.
- `ranges` entries are meters or `None` for invalid returns. `range_min` / `range_max` clip what the matcher considers.
- Scan is in the **lidar frame**, not body frame; any pose offset between lidar and robot center is not currently modeled on desktop. If the lidar isn't at the rotation center, mapping will show a small radial offset but still close loops.

### 2.4 Clock

All three topics share `ts` semantics (seconds since epoch, Pi system clock). The scan-matcher asks `ImuYawTracker.yaw_at(scan.ts)` and `OdomPose.pose_at(scan.ts)`; both return `None` if the scan timestamp is outside the sample buffer. Realistic skew tolerance is <20 ms between `body/imu` and `body/lidar/scan` arrivals. No action needed unless you see pose-resolution misses in the fuser log (`pose_unavailable` notes) correlated with large clock jumps.

## 3. Topic rates and what desktop does with them

| Topic                | Rate      | Desktop consumer                             |
|----------------------|-----------|----------------------------------------------|
| `body/imu`           | 100 Hz    | `ImuYawTracker` (desktop)                    |
| `body/odom`          | 20–50 Hz  | `OdomPose` inside `ImuPlusScanMatchPose`     |
| `body/lidar/scan`    | 10 Hz     | `ScanMatcher` (rate-limited to ~2 Hz)        |
| `body/local_map`     | ~5 Hz     | Fuser — unchanged from current world_map     |

Desktop is content with these rates. If any of them falls meaningfully below (IMU <50 Hz, odom <10 Hz, scan <5 Hz) the scan-match acceptance rate will degrade and the fuser log will show more `pose_unavailable` streaks.

## 4. What is explicitly **not** required Pi-side

These are deliberately **not** Pi-side work, to keep the contract minimal:

- **No IMU-into-odom fusion on Pi.** Desktop does the combining.
- **No scan-matching on Pi.** Pi stays a pure sensor/actuator node.
- **No pose correction feedback Pi→desktop.** The desktop pose is internal to the fuser.
- **No landmark / feature extraction on Pi.** Raw scan ranges are all desktop needs.
- **No `ts` alignment between topics on Pi.** Desktop handles interpolation / bracketing.
- **No BNO085 offset calibration on Pi** beyond what the SH-2 fusion firmware already does. If the IMU is mounted with a non-trivial yaw offset relative to the robot's forward direction, record it as a constant in the Pi publisher (or in a shared config) so desktop can rotate into body frame — but a one-time measured offset is enough; no live calibration loop is needed.

## 5. Hand-off checklist

Before the user flips `--slam` on nav:

1. Drive a short loop with `python -m desktop.nav --shadow-slam --router tcp/<pi>:7447`.
2. Grep the shadow log for these numbers per scan-match attempt:
   - `accepted` rate > ~50% of non-`search_exhausted` attempts
   - median `improvement` > `min_improvement` threshold by at least 2×
   - `search_exhausted` rate < ~20%
   - `imu_settled == true` for ≥ 95% of the post-warmup window
3. If `mag_status` flaps during the drive, consider re-calibrating or using `game_rotation_vector` and budgeting for slower-loop drift.
4. If all four numbers look right, the `ImuPlusScanMatchPose` promotion PR can merge with the feature flag on by default.

## 6. Roll-back

SLAM is a `FuserConfig` flag. If live mapping quality regresses vs. odom-only, nav can be relaunched without `--slam` and the fuser reverts to `OdomPose` with zero Pi-side changes.
