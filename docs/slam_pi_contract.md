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
- `body/imu.fusion.accuracy_rad` stable and small — `< ~0.06` rad (3.4°) once warm. The desktop `ImuYawTracker` refuses to answer queries until `min_settle_samples` consecutive readings fall below `settle_accuracy_rad` (default `0.06`), so startup transients are handled; steady-state accuracy is what matters for map quality.
- `body/imu.fusion.calibration_status` — integer `0..3` per the SH-2 spec (0 = unreliable, 3 = high). Published when available. Desktop consumes it only as a diagnostic today; a `< 3` value during driving predicts slower correction and more noise, but is not a hard gate.

#### Threshold coupling — important

Three thresholds on Pi and desktop interact. They must be ordered so Pi doesn't declare "settled" at an accuracy it will then flap away from:

```
  imu.calibration_stable_threshold_rad    ≤    imu.mag_accuracy_fallback_rad    ≤    desktop settle_accuracy_rad
```

Concretely, for SLAM promotion:

- `imu.calibration_stable_threshold_rad` ≤ `0.06` (matches desktop settle).
- `imu.mag_accuracy_fallback_rad` ≤ `0.087` (Pi's existing default) is OK but should be ≥ the stable threshold, or Pi will settle and fall back on the very next sample.
- `imu.mag_accuracy_fallback_count` of `20` at `100 Hz` = 0.2 s before flap — acceptable.

Current `config.json` has `calibration_stable_threshold_rad: 0.175`, which is too loose for SLAM: Pi will start publishing at accuracy well above the fallback threshold and drop out of `rotation_vector` within 20 samples. **Tighten this to `0.06`** before relying on `--slam`.

`game_rotation_vector` mode still works — yaw is relative (starts at zero at boot, drifts ~0.5–1°/min) — but the absolute-heading advantage is gone. See §2.2 for the operating differences. Indoor magnetometer disturbance from motor current, ferrous structure, etc. often forces this mode; treat it as the normal indoor case, not the degraded one.

### 2.2 Operating in `game_rotation_vector` mode

When `fusion_mode: "game_rotation_vector"` (current `config.json` default after Pi-side review), several things differ from RV mode:

**Pi side (no config change needed):** SH-2 firmware doesn't compute a dynamic accuracy estimate in GAME_RV. The Pi publishes `accuracy_rad` as the **constant** value of `imu.game_rotation_vector_accuracy_rad` (currently `0.175`) on every sample. The Pi-side settle gate (`calibration_stable_threshold_rad: 0.175`) matches that constant, so settling is purely time-gated by `settle_time_s: 2.0`. This is correct.

**Desktop side (small change required for SLAM promotion):** `ImuYawTracker.DEFAULT_SETTLE_ACCURACY_RAD = 0.06` would reject every GAME_RV sample (`0.175 > 0.06`) and never settle. Two options for the SLAM promotion PR:

1. *(preferred)* Time-based gate when `fusion_mode == "game_rotation_vector"`: require `min_settle_samples` consecutive samples received plus a small wall-clock buffer (e.g. `≥ 0.2 s`). Skip the accuracy comparison entirely — there's nothing dynamic to compare against in this mode.
2. Bump desktop `settle_accuracy_rad` to `≥ 0.18`, document that the gate is degenerate in GAME_RV, and rely on `min_settle_samples` to keep startup transients out.

**Settle-time budget — how long is too long?** The cost of waiting for settle is per *meter driven during settle*, not per second elapsed. Pi already waits 2 s before publishing; desktop adds ~0.2 s for buffer. If the user holds the robot still until they press Reset world, total cost is zero. Padding settle further is wasted time unless the user is driving during it (encoder-only yaw drift accumulates and contaminates the early map).

**Reset-world yaw rebind.** GAME_RV yaw is "whatever the BNO085 happened to read at boot," not magnetic north. To pin the world frame to the user-defined orientation regardless of GAME_RV's boot heading, `ImuPlusScanMatchPose` should capture the IMU yaw at `Reset world` time and store it as an offset (analogous to how `OdomPose.rebind_world_to_current()` already does for translation). The handoff from encoder yaw to IMU yaw at settle is then seamless — no pose jump.

**Drift-correction load.** In GAME_RV, scan-match against the world grid is the *only* mechanism keeping yaw bounded over the long term. RV mode had magnetometer as a safety net; GAME_RV does not. Practical implications:

- Acceptance-rate target tightens: > 70% of non-`search_exhausted` attempts (vs. > 50% in RV).
- Long featureless straight runs are the failure mode — locally the geometry is rotationally ambiguous along the corridor axis, and there's no magnetometer to anchor. Mitigation: avoid long featureless straights, or add a Manhattan-World yaw prior (~150 lines, post-promotion follow-up; uses lidar to detect dominant orthogonal wall directions).
- Cross-session map persistence is not free: each boot defines its own arbitrary world frame. Reloading a saved map requires either (a) the user aligning the robot to a floor mark whose pose is known in the saved map (manual entry of x/y/yaw on Reset), or (b) a one-time AprilTag mounted in view of the OAK-D RGB at boot (pose-from-tag, automatic). See §7 for floor-mark options.

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

1. **Pi config audit.** Confirm `config.json` has:
   - `imu.fusion_mode` matches the mode SLAM will operate in (`rotation_vector` if magnetometer is reliable, `game_rotation_vector` otherwise).
   - For RV: `imu.calibration_stable_threshold_rad ≤ 0.06`, `mag_accuracy_fallback_rad` between that and `0.087`, BNO085 DCD saved after a figure-8.
   - For GAME_RV: `calibration_stable_threshold_rad` matches `game_rotation_vector_accuracy_rad` (both `0.175` is fine — it's a constant in this mode, settle is time-gated). No DCD save required.
2. **Shadow drive.** `python -m desktop.nav --shadow-slam --router tcp/<pi>:7447`, run a short loop.
3. Grep the shadow log for these numbers per scan-match attempt:
   - `accepted` rate > ~50% (RV mode) / > ~70% (GAME_RV mode) of non-`search_exhausted` attempts.
   - median `improvement` > `min_improvement` threshold by at least 2×.
   - `search_exhausted` rate < ~20%.
   - `imu_settled == true` for ≥ 95% of the post-warmup window.
4. **For RV mode only:** watch `body/imu.fusion.mode` during the drive. If it flaps from `rotation_vector` to `game_rotation_vector` at any point, fix thresholds first or accept that you're effectively running GAME_RV. (For GAME_RV-by-config, this check doesn't apply.)
5. If Pi config is clean and all log numbers look right, the `ImuPlusScanMatchPose` promotion PR can merge with the feature flag on by default.

## 6. Roll-back

SLAM is a `FuserConfig` flag. If live mapping quality regresses vs. odom-only, nav can be relaunched without `--slam` and the fuser reverts to `OdomPose` with zero Pi-side changes.

## 7. Starting-pose anchors (floor marks)

Two unrelated reasons to want explicit starting-pose definition:

- *Within a session:* user wants the world frame oriented to a meaningful direction (e.g. "+x = down the hallway"), not just "wherever the robot was facing at Reset."
- *Across sessions:* if a saved map is reloaded, each boot's `OdomPose` and GAME_RV yaw start from arbitrary zero — the robot has no idea where it is in the saved map.

All three options below are desktop-only; no Pi changes.

### 7.1 Manual aligned-then-press (recommended starting point)

Mark a known pose on the floor (e.g. tape cross with an arrow). User aligns the robot to the mark, then clicks `Reset world` with optional input fields for `(x, y, θ)` of that mark in the saved-map frame.

The current `Reset world` action implicitly assumes the mark is at `(0, 0, 0)`. Adding three numeric fields (defaulted to zero, so the existing UX is unchanged) makes it work for arbitrary saved-map-relative starts.

**Cost:** ~40 lines (UI + plumbing into `OdomPose.rebind_world_to_current` and the IMU yaw rebind described in §2.2). Critical for cross-session resume.

### 7.2 AprilTag fiducial via OAK-D RGB

Print one AprilTag, mount it on a wall the OAK-D can see at boot. Tag pose-in-image gives full `(x, y, yaw)` in the world frame the tag was originally registered against. Robot self-localizes at boot with no human alignment.

**Cost:** ~200 lines using the `apriltag` Python library, plus a one-time "register this tag at this pose in the saved map" UI. Worthwhile once saved-map reload is in regular use.

### 7.3 Lidar landmark alignment

User clicks a known geometric feature (corner, doorway) on the loaded map view and says "the robot is here, facing that." Constrains scan-match initial guess from the click; runs a wider-than-normal search to converge.

**Cost:** ~150 lines. Best when no fiducial hardware is acceptable and the environment has distinctive geometry. Lower priority than 7.1.

### Which to do when

- For session-anchored mapping today, no work needed — current `Reset world` is enough as long as the user accepts "+x = current heading."
- Once map persistence (save/load grid) lands, do 7.1 same PR.
- 7.2 and 7.3 only if the manual workflow is too cumbersome for actual use.
