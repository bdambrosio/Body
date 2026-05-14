# Pi change request — motor: bump min_drive_pwm to clear static-friction breakaway

**Date:** 2026-05-14
**Audience:** Pi-side developer (Bruce, via Cursor Remote-SSH)
**Scope:** one-line `config.json` edit. No code change. No schema / topic changes.
**Status:** request — desktop has characterized the symptom via Sweep-360.

---

## Problem

Sweep-360 commanded 360° of in-place rotation (6 steps × 60° at 20 °/s) and the
robot physically rotated only ~40°. The SweepDock status panel showed:

```
state: done
step: 6/6
yaw_accum:  +0.00°
yaw_sources: lidar=+0.00°  imu=—  cmd=+60.00°
last fused: +0.00°  (lidar conf=0.70)
```

The mission ran to completion — no abort, no stale-data fail. Lidar scan-match
reported 0° per step at 0.70 confidence, which means the scans before and after
each step looked nearly identical. The most parsimonious explanation: the wheels
barely moved. At commanded angular rate 0.349 rad/s with `max_wheel_vel_ms = 0.9`,
the feed-forward duty is

```
ff = v_wheel / max_v = (omega * wheel_base/2) / max_v
   = (0.349 * 0.095) / 0.9
   = 0.037   (3.7% duty)
```

The PI loop snaps that up to `min_drive_pwm = 0.10` (currently configured), but
even 10% duty appears to be below the breakaway threshold on this floor — wheels
sit static until the integrator winds up over multiple seconds.

## Why it's a config-only fix

`body/motor_controller.py` already implements dead-zone compensation:

```python
# WheelPI.step(), motor_controller.py:55-57
pwm = ff + self.kp * err + self.ki * self.integ
if self.min_drive_pwm > 0.0 and abs(pwm) < self.min_drive_pwm:
    pwm = math.copysign(self.min_drive_pwm, v_cmd)
```

`min_drive_pwm` is read from `motor.min_drive_pwm` in `config.json` (line 100 of
`motor_controller.py`), currently set to **0.10**. The infrastructure is correct;
the value is just too low for this floor + tire + battery combination. The
21-day-old memory `project_body_motor_deadzone` estimated breakaway at 15–25%
duty — consistent with the symptom.

This is a **calibration** task, not a code task. No PR review of `motor_controller.py`
needed.

## The change

In `config.json`, the `motor` block:

```diff
   "motor": {
     ...
     "velocity_loop_enabled": true,
     "velocity_kp": 0.8,
     "velocity_ki": 2.0,
     "velocity_integ_limit": 0.5,
-    "min_drive_pwm": 0.10
+    "min_drive_pwm": 0.18
   },
```

Test value 0.18 chosen as the midpoint of the prior 15–25% estimate. We'll
ratchet up or down based on the next Sweep-360 run.

**No other field changes.** In particular do NOT touch:

- `velocity_kp`, `velocity_ki`, `velocity_integ_limit` — the PI gains are
  reasonable and changing them in the same step would conflate causes.
- `max_wheel_vel_ms` — that's the wheel velocity ceiling; unrelated.
- `min_drive_pwm` of 0.0 as a "safety" default — that disables the dead-zone
  snap entirely, which is the opposite of what we want.

## Verification

1. After the config change, restart the Pi stack (`body/launcher.py`).
2. On the desktop, with `min_drive_pwm = 0.18` active on the Pi, re-run
   Sweep-360 with the same parameters Bruce used earlier
   (`step_deg=60, total_deg=360, rate=20 dps`).
3. Watch the SweepDock status:
   - `yaw_accum` should now be much closer to 360° if breakaway was the
     dominant issue. If it climbs to ~300–340°, the bot is moving but some
     slip remains — expected on hardwood, and a separate problem.
   - `lidar` per step should report non-trivial values (15–60° range) with
     confidence still ≥ 0.35.
4. Inspect `body/motor_state` (50 Hz) during a slow rotation step. The
   per-tick `left_pwm` / `right_pwm` should sit at ≥ 0.18 in magnitude
   throughout the step, not at 0.10 anymore.

## If 0.18 still under-rotates

Two possibilities, in order of likelihood:

1. **Breakaway is higher than 0.18.** Ratchet to 0.22, then 0.25. Stop when
   either (a) the bot rotates close to commanded, or (b) startup behavior
   becomes jerky/aggressive (the snap is mostly intended to overcome static
   friction, not to bypass the loop).

2. **Wheels move but slip on the floor.** This is invisible to the PI loop
   because the encoders are on the motor shaft, not the wheel — they'll
   happily report v_meas = v_cmd while the wheel skates. Diagnose by
   driving 1 m straight via teleop **without `--slam`** on the desktop
   (raw odom display) and comparing to a tape measure. If encoder reports
   1.0 m but tape says 0.7 m, slip is the dominant issue and the next
   move is mechanical (tires, weight) not config.

## Out of scope for this change

- IMU integration into Sweep-360 (desktop-side; `sweep_mission.py:345` has
  a stale `imu_deg = None  # current hardware has no IMU` comment we'll fix
  separately).
- Closed-loop rotation via IMU yaw (desktop-side; bigger work).
- "Live cmd dropped" mission-fail demotion (desktop-side; shipping now in
  the same desktop commit as this spec).

## Commit

```
motor: bump min_drive_pwm 0.10 → 0.18 for breakaway

Sweep-360 commanded 360° and physically rotated ~40° at 20 dps, with lidar
scan-match reporting 0° per step at conf 0.70 (i.e. scans nearly identical
between steps). Breakaway estimate from the deadzone memory is 15-25% duty;
0.18 is the test value. PI loop and dead-zone snap infrastructure unchanged.
See docs/motor_pwm_calibration_pi_change.md for diagnostic detail and the
fallback plan if 0.18 isn't enough.
```
