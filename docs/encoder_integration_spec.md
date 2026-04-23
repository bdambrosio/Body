# Wheel encoder integration — consumer-side contract

**Project:** Body
**Date:** 2026-04-23
**Scope:** what the desktop SLAM / fuser pipeline expects from Pi-side wheel encoder integration. Hardware wiring is covered in `motor_controller_spec.md` §2–3.

---

## 1. Role of encoders in the pose pipeline

Encoders are **not the primary pose source**. This robot is a two-wheel differential drive with a passive caster on carpet/hard-floor mix; even perfect quadrature counts feed a kinematic model that assumes no slip, a known wheelbase, a rotation axis at the wheel midpoint, and a frictionless caster — none of which hold tightly enough on this platform to trust integrated encoder odometry as ground truth.

In the desktop pipeline:

- **Rotation** comes from the BNO085 IMU (see `imu_integration_spec.md`).
- **Translation (xy)** comes from lidar scan-matching, seeded by a short-horizon prior from encoders + IMU.
- **Encoders specifically** provide three signals, in decreasing order of importance:
  1. **Stall detection**: commanded PWM above the breakaway threshold but near-zero wheel speed for ≥ stall window → raise `motor_state.stall_detected`.
  2. **Slip / sanity**: compare integrated encoder translation against scan-match translation. When they disagree by more than the combined noise floor, mark the fused pose update as `low_confidence` (cell votes downweighted).
  3. **Short-horizon translation prior**: seed the scan-match xy search window around the encoder-derived displacement since the last successful match. A bad seed is OK (scan-match will still find the correct alignment within its window); what we gain is a smaller search window and faster convergence.

Consumers that currently read `source = "commanded_vel_playback"` (the v1 fuser) will continue to work when encoders go live; the pose just gets more trustworthy.

## 2. Wire contract — `body/odom`

No schema changes. Populate the existing fields per `body/lib/schemas.py` with real values:

| Field | Semantics when `source = "wheel_encoders"` |
|---|---|
| `ts` | Sample time of the tick read, not publish time. Within 5 ms of the actual read. |
| `x`, `y`, `theta` | Integrated pose using `diff_drive.integrate_odometry`. Consumer treats this as a **noisy** prior, not ground truth. |
| `vx`, `vtheta` | Derived from tick deltas / `dt_ms`. Zero on cycles where both ticks were zero. |
| `left_ticks`, `right_ticks` | Cumulative signed counts since driver start. Must be signed (quadrature direction); an unsigned counter flipping at reverse is a wire-contract bug. |
| `dt_ms` | Time between this read and the previous one, milliseconds. |
| `source` | `"wheel_encoders"` once ticks are live. |

Critical: counts are **cumulative and monotonic in the direction of rotation** — when the wheel reverses, counts decrement. Consumers diff successive values and never assume sign.

## 3. Sampling expectations

- **Rate**: match the existing motor_controller inner loop (currently ~50 Hz). Don't publish at kHz just because you can; 50 Hz is enough for scan-match seeding.
- **Acquisition**: prefer hardware counter (Pi 5 RP1 has two, but under Linux they're hard to reach; `lgpio`/`pigpio` edge-callback counting is the realistic path). Polling GPIO at kHz from Python is lossy and will miss edges at high wheel speeds.
- **Missed edges**: a quadrature A/B decoder that drops an edge produces a single-count error, not a stuck counter. That's acceptable. What's **not** acceptable is stuck counts on reverse direction (symptom: unsigned counter, bug per §2).

## 4. Calibration expectations

The kinematic model has two per-robot constants the SLAM consumer does **not** need to know precisely, but does need to be stable:

- `wheel_radius_m` — tick-to-meter conversion. Drift over session OK; bulk-changing between runs means consumer will see scale jumps in encoder xy.
- `wheel_base_m` — tick-differential-to-theta conversion. Same stability requirement.

A future UMBmark-style calibration run (2 m straight, 2 m square) can refine these. **v1 does not require calibration** — the scan-matcher absorbs both biases via its bounded xy search. Calibration just tightens the search window and improves scan-match convergence speed.

## 5. What consumers will NOT do

- Treat `x`, `y`, `theta` as globally correct.
- Accumulate encoder pose over long horizons (> 1 minute) without scan-match corrections.
- Use encoders to detect direction-of-travel ambiguities; that's the IMU's job.

## 6. Acceptance test

When `source = "wheel_encoders"` first goes live, these three behaviors must hold:

1. **Stationary robot**: `left_ticks` and `right_ticks` remain constant (±1 tick jitter from quadrature dither is OK).
2. **Forward command for 1 s at 0.2 m/s**: both counters increment by ≈ same number of ticks in the same sign direction. Discrepancy < 20% left-vs-right (gross sanity; UMBmark tightens later).
3. **Reverse command**: counters decrement.

If any of these fails, the tick path is broken and consumer treats `source` as `"wheel_encoders"` but the data as garbage — better to leave `source = "commanded_vel_playback"` and ship the fix first.
