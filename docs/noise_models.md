# Noise Models — Phase 0 Calibration

**Status:** Awaiting data collection. Tools landed; experiments + numbers pending.

This document records the empirically-measured sensor noise priors that
the Bayesian SLAM redesign
(`docs/bayesian_localization_redesign.md`) consumes. Phase 0 of that
plan is "Foundation: noise-model calibration data."

We need three sets of numbers:

  A. **IMU yaw drift** — drift rate (rad/s) + per-sample σ (rad).
     Consumed as `σ²_IMU(t) ≈ σ²_sample + (drift_rate · t)²` by the
     filter's IMU observation likelihood.
  B. **Encoder translation noise** — coefficient `α_1` such that
     `σ_trans ≈ α_1 · |Δs|`. Consumed by the motion-model proposal
     distribution per odom step.
  C. **Encoder rotation noise** — coefficient `α_4` such that
     `σ_rot ≈ α_4 · |Δθ|`. Same.

Cross-terms α_2 (translation σ from rotation) and α_3 (rotation σ from
translation) are zero for now; we can add if combined-motion drives
show they matter.

## Phase 0 status

- [x] Plan locked (this doc + parent redesign doc).
- [x] Analysis tooling shipped (`scripts/phase0_imu_stationary.py`,
      `scripts/phase0_odom_drive.py`).
- [x] Experiment A run (2026-05-15).
- [x] Experiment B run (3 drives at 1/3/5 m, 2026-05-15).
- [ ] Experiment C run (≥3 rotations at different magnitudes).
- [ ] Numbers below partly filled in (A + B).
- [ ] Bruce review.

## Pi-side changes

**None required for Phase 0.** All experiments use existing
`body/odom`, `body/imu`, `body/lidar/scan` topics that the Pi already
publishes. No `body/*` code changes; no resync. The recorder and
analysis run entirely on the desktop.

## Experimental procedure

In every experiment: keep the desktop Zenoh router endpoint consistent
with the Pi (`tcp/<pi-ip>:7447`). Wait for IMU `settled` log line
(~2 s after boot) before running. Replace `<ip>` and timestamps below.

### Experiment A — IMU stationary drift (~10 min)

```bash
PYTHONPATH=. python3 scripts/record_body_topics.py \
    --router tcp/<pi-ip>:7447 \
    --topics body/imu \
    --out ~/body-logs/phase0-imu-stationary-$(date +%Y%m%d-%H%M).jsonl
# Wait at least 5 minutes — longer is better for tight drift-rate estimate.
# Do NOT touch the bot. Avoid foot traffic that could vibrate the floor.
# Ctrl-C to stop.
```

Goal: characterize the IMU yaw drift while motionless. game_rotation_vector
mode is known to drift ~0.5–1°/min; we measure ours specifically.

### Experiment B — Encoder translation noise (~5 min per run)

Setup: place bot at a marked start, tape-measure 3 m of clear floor,
mark the end. Repeat at 3 distances (e.g. 1, 3, 5 m) so we can fit α_1.

```bash
PYTHONPATH=. python3 scripts/record_body_topics.py \
    --router tcp/<pi-ip>:7447 \
    --topics body/odom body/imu \
    --out ~/body-logs/phase0-trans-3m-$(date +%Y%m%d-%H%M).jsonl &
RECORDER=$!
# Drive bot straight at modest speed (the nav twist pad's straight-up
# command). When the bot reaches the marked endpoint, stop it.
kill $RECORDER
```

For each run, record the **actual** distance the bot ended up at
(measure the bot's center to the start mark with a tape). Note it in
the analysis invocation below.

### Experiment C — Encoder rotation noise (~5 min per run)

Setup: bot at a known orientation. Rotate in place by a known commanded
amount. The IMU is treated as ground truth for short-window rotation
(it's much better than the encoder over single drives — drift only matters
at multi-minute scales we measured in Experiment A).

```bash
PYTHONPATH=. python3 scripts/record_body_topics.py \
    --router tcp/<pi-ip>:7447 \
    --topics body/odom body/imu \
    --out ~/body-logs/phase0-rot-360-$(date +%Y%m%d-%H%M).jsonl &
RECORDER=$!
# Use the sweep mission for clean repeatable rotation, OR drive a
# pure-rotation cmd_vel through the twist pad. Rotate at least 360°.
# Stop bot. Repeat at varied total angles (90, 180, 360, 720).
kill $RECORDER
```

## Analysis

After collecting logs:

```bash
# A. IMU stationary
PYTHONPATH=. python3 scripts/phase0_imu_stationary.py \
    ~/body-logs/phase0-imu-stationary-*.jsonl --plot

# B. Each translation run
PYTHONPATH=. python3 scripts/phase0_odom_drive.py \
    ~/body-logs/phase0-trans-3m-*.jsonl \
    --mode translation --measured-distance-m 3.0

# C. Each rotation run
PYTHONPATH=. python3 scripts/phase0_odom_drive.py \
    ~/body-logs/phase0-rot-360-*.jsonl \
    --mode rotation
```

Each invocation prints a short summary including the candidate noise-
model coefficients. With ≥3 runs of B and ≥3 of C, fit α_1 and α_4 by
inspection (does the error grow linearly with distance/angle? if not,
does it grow with the *square root* — random walk? — or is there a
constant offset suggesting a calibration bias to fix first?).

Record findings below.

## Measured values

### IMU drift (Experiment A)

Recording: `~/body-logs/phase0-imu-stationary-20260515-1857.jsonl`.

- Window duration: 421.6 s (~7 min).
- Sample rate: 99.1 Hz (matches Pi-side `imu.publish_hz = 100`).
- Samples used: 41,779.
- **Drift rate: -0.012 deg/min (-3.42e-6 rad/s).**
- Total drift across window: -0.083 deg (effectively below detection
  floor — call this an *upper bound* on drift, not a measurement).
- **Per-sample σ: 0.071 deg (1.23 mrad).**
- Per-√s σ (assumes independent samples — conservative upper bound;
  the SH-2 fusion filter induces sample-to-sample correlation):
  0.70 deg/√s (1.23e-2 rad/√s).

**Interpretation:** BNO085 in game_rotation_vector mode is unusually
quiet on this bot. The IMU contributes essentially zero orientation
error over any drive duration we care about (< 1 deg in a multi-minute
drive even at the upper bound). In the particle filter, treat the IMU
yaw observation as a tight constraint; the dominant orientation
uncertainty will come from encoder rotation slip (Experiment C),
not IMU drift.

### Encoder translation (Experiment B)

Three drives at nominal 1/3/5 m, twist-pad straight forward, ~0.1–0.15 m/s.
Recordings in `~/body-logs/phase0/trans-{1,3,5}m-20260515-*.jsonl`.

| Run | Measured (m) | Encoder (m) | Abs err (m) | Frac err | α_1 point | Encoder Δθ | IMU Δθ | Enc−IMU |
|---|---|---|---|---|---|---|---|---|
| B1 | 1.00 | 0.9883 | 0.0117 | 1.17% | 0.0117 | +1.64° | +0.41° | +1.23° |
| B2 | 2.98 | 3.1125 | 0.1325 | 4.45% | 0.0445 | -0.29° | -3.40° | +3.11° |
| B3 | 5.02 | 4.9096 | 0.1104 | 2.20% | 0.0220 | +10.26° | +6.08° | +4.18° |

**Fit / chosen α_1: 0.04** (slightly above mean+1σ of the three samples;
conservative but not absurd).

**Cross-term α_3 observed: 0.017 rad/m (~1°/m of translation).**
Encoder θ over-reports rotation during translation in all three runs,
consistently. Per-run estimates 1.23/1.04/0.84 °/m — tight enough that
this is not noise; it's a real cross-coupling term that the motion
model must include from the start. The standard Thrun-Burgard-Fox
diff-drive noise model has α_3 as exactly this (rotation σ per meter
of translation).

Notes:
- No calibration bias: error signs flip across runs (under, over,
  under), so it's slip noise, not wheel-radius miscalibration.
- The bot does NOT drive perfectly straight: IMU reports 0.4–6° of
  rotation during each "straight" drive. Probably differential motor
  output, floor camber, or slight pad-input bias from the operator.
  Doesn't affect α_1 fit (chord-vs-arc correction for 10° is 0.1%).
- The encoder consistently sees MORE rotation than the IMU during
  translation — never less. Sign of differential wheel slip favoring
  one side.

### Encoder rotation (Experiment C)

_Pending._

| Run | IMU Δθ (deg) | Encoder Δθ (deg) | Abs err (deg) | Frac err | α_4 point |
|---|---|---|---|---|---|
|   |   |   |   |   |   |

Fit / chosen α_4: `<>`.
Notes (slip rate, direction asymmetry?): _pending._

## Output: filter priors

Once the table above is filled, the particle filter will be initialized
with:

```python
# Motion model (per odom step Δs, Δθ):
SIGMA_TRANSLATION_PER_M = <α_1>
SIGMA_ROTATION_PER_RAD  = <α_4>

# IMU yaw observation:
IMU_SIGMA_PER_SAMPLE_RAD = <σ_sample>
IMU_DRIFT_RATE_RAD_PER_S = <drift_rate>
```

These go into `desktop/world_map/particle_filter_pose.py` when Phase 2
starts.

## Out of scope for Phase 0

- **Scan-match likelihood landscape** — deferred until we have a clean
  SLAM session recording with diverse scene types. The
  `scripts/phase0_scan_likelihood_*.py` will land in a follow-up.
- **Cross-noise terms α_2, α_3** — measured only if combined-motion
  drives reveal they matter.
- **Battery / floor-surface dependence** — keep one set of conditions
  for the initial fit; document any session-to-session variation we
  observe later.
