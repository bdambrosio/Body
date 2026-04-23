# BNO085 IMU integration — consumer-side contract

**Project:** Body
**Date:** 2026-04-23
**Hardware:** BNO085 (CEVA SH-2 firmware) on Pi i2c.
**Scope:** what the desktop SLAM pipeline expects from Pi-side IMU integration. Replaces the OAK-D-Lite onboard IMU which is absent on this unit (early-Kickstarter hardware).

---

## 1. Role of the IMU in the pose pipeline

The BNO085 is the **primary source of robot orientation**. On-chip Bosch fusion delivers a driftless-in-pitch/roll quaternion with either:

- **Rotation Vector** (mag + accel + gyro): absolute yaw reference, bounded yaw drift. Preferred when the magnetometer is usable.
- **Game Rotation Vector** (accel + gyro only): relative yaw — no absolute heading, slow yaw drift (~0.5–1°/min typical). Use this when the motor current disturbs the magnetometer enough to contaminate fusion.

Which of the two reports is driving fusion is tagged in the published message (see §2). Downstream consumers adapt: absolute-yaw fusion lets the scan-matcher use the IMU yaw as a near-zero-width prior; relative-yaw fusion means the scan-matcher has to close the drift, but the drift is slow enough that it's fine at lidar rate.

The IMU also provides:

- **Gyroscope** (raw rates, rad/s, body frame) — useful for angular velocity and short-horizon propagation between fusion updates.
- **Linear acceleration** (gravity-compensated, body frame) — nice-to-have for motion-detection heuristics. Double-integration for translation is **not** a consumer expectation; noise integrates to garbage over seconds. Translation comes from scan-matching, not the IMU.

## 2. Wire contract — `body/imu`

Proposed rename: publish to **`body/imu`** (new topic) instead of `body/oakd/imu`. The latter is no longer OAK-D-attached and the name is misleading. Existing subscribers (`watchdog.py`, `chassis/state.py`, `chassis/config.py::Topics.oakd_imu`) flip to the new topic in the same change.

Schema extends the existing `oakd_imu_report` shape — everything already there, plus fusion metadata:

```json
{
  "ts": 1234567890.123,
  "accel":       { "x": 0.0, "y": 0.0, "z": 9.81 },
  "gyro":        { "x": 0.0, "y": 0.0, "z": 0.0 },
  "linear_accel":{ "x": 0.0, "y": 0.0, "z": 0.0 },
  "orientation": { "w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0 },
  "fusion": {
    "mode": "rotation_vector",
    "accuracy_rad": 0.035,
    "mag_status": "calibrated",
    "gyro_status": "calibrated",
    "accel_status": "calibrated"
  }
}
```

Field notes:

| Field | Required? | Notes |
|---|---|---|
| `ts` | yes | Sensor sample time, not publish time. SH-2 timestamps are fine; just subtract boot-epoch offset to wall time. |
| `accel` | yes | Raw accel, m/s², body frame. Include gravity. |
| `gyro` | yes | Raw rates, rad/s, body frame. |
| `linear_accel` | optional | Gravity-removed accel from the BNO085 report. Skip if you don't enable it. |
| `orientation` | yes | Fused quaternion, **wxyz** (note: BNO085 native is ijkr ≈ xyzw — convert). |
| `fusion.mode` | yes | `"rotation_vector"` \| `"game_rotation_vector"` \| `"raw"`. Tells consumer whether yaw is absolute or drift-bounded-only. |
| `fusion.accuracy_rad` | yes | BNO085 per-report accuracy estimate. Consumer uses this as orientation σ. |
| `fusion.*_status` | optional | SH-2 calibration status. Diagnostic. |

Backwards compat: keep publishing the existing `body/oakd/imu` topic for one cycle with the same payload shape, or flip in a single atomic change. Prefer the atomic change — fewer topics = less confusion.

## 3. Frame convention

Body frame: **x-forward, y-left, z-up** (right-handed). This matches the lidar, the local_map, and the fuser. The BNO085 has to be mounted or software-rotated so its reported frame matches this convention. If mounting forces a non-standard orientation:

- Apply the static rotation **on the Pi side**, before publishing. Don't export axis confusion to desktop.
- Document the mount orientation in a comment in whichever Pi module owns the BNO085 (probably a new `body/imu_driver.py`).

If mount rotation is applied, both accel and gyro and the quaternion must be rotated consistently.

## 4. Publish rate

**Target:** 100 Hz. Quaternion updates at this rate give the scan-matcher a rotation prior with <10 ms of gyro propagation between IMU sample and lidar sample (scan at ~10 Hz).

Raw gyro can come off the chip faster (400 Hz on the BNO085). Aggregating down for the wire is fine — the SH-2 report rate can be configured independently per report. Don't publish at 400 Hz; pointless bandwidth.

## 5. Startup + calibration

- **Boot-time settle**: BNO085 auto-calibrates gyro bias at rest over ~1–2 s after power-up. Let the robot stand still for ≥ 2 s after `imu_driver` starts before trusting `orientation`. Publish a `status = "calibrating"` flag or equivalent until the SH-2 accuracy report crosses a threshold.
- **Mag calibration** (if Rotation Vector mode used): requires a figure-8 motion to complete. Document whether this has been done on first boot; the BNO085 persists calibration to flash. Sensitive to motor current — perform calibration with motors de-energized.
- **Mag interference test**: run motors at typical drive current near the IMU mount point. Watch `fusion.accuracy_rad` and `mag_status`. If accuracy exceeds ~5° during commanded motion, the mag is contaminated; switch to Game Rotation Vector and accept the slow yaw drift.

## 6. What consumers will do with this

- **Scan-matcher** (desktop, future `desktop/nav/slam/`): yaw from `orientation` becomes the rotation prior. Search window ≈ 4 × `fusion.accuracy_rad`, or ±2° if that's smaller.
- **World-map fuser pose source**: new `ImuPlusScanMatchPose` implementation reads this topic for rotation, blends with scan-match for translation.
- **Chassis UI**: no change. The existing chassis state still reads the same fields from the new topic name.

## 7. What consumers will NOT do

- Integrate `gyro` themselves to get yaw — the BNO085's onboard fusion does this better than we can.
- Double-integrate `accel` for translation — see §1.
- Trust `orientation` before the boot-time settle completes.

## 8. Acceptance test

When `body/imu` first goes live:

1. **Stationary robot, 30 s**: `gyro.z` mean < 0.005 rad/s; quaternion yaw drift < 1° over 30 s in Game mode, < 0.5° in Rotation Vector mode.
2. **90° hand-rotation in place**: published quaternion yaw changes by 90° ± 2°. Sign matches CCW = positive (right-hand rule around z-up).
3. **Commanded motor spin, 5 s at moderate duty**: `fusion.accuracy_rad` stays below 3° in Game mode; in Rotation Vector mode, compare `mag_status` before vs during — if it degrades materially, this is the signal to switch fusion modes (§5).

Passing (1) + (2) is the minimum bar to flip consumers. (3) determines Rotation Vector vs Game mode — doesn't block integration, but determines search-window tightness downstream.
