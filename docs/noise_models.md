# Noise Models — Phase 0 Calibration

**Status:** Phase 0 complete (2026-05-15). All five priors locked.
Ready to drive Phase 1 (likelihood-field scan matcher) and Phase 2
(particle filter motion model).

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
- [x] Experiment C run (3 rotations 180/360/720°, 2026-05-15).
- [x] Calibration fix applied (`motor.wheel_base_m: 0.190 → 0.181`,
      2026-05-15 — Pi-side resync + motor_controller restart done).
- [x] Post-fix sanity check (single 360° rotation): bias collapsed
      from -5.58% to -0.61%, α_4 verified.
- [x] All numbers locked.
- [x] Bruce review (informal, in-conversation).

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

Three in-place rotations via SweepDock with `step_deg == total_deg`
(one continuous rotation), rate = 15 dps, direction = ccw. Recordings
in `~/body-logs/phase0/rot-{180,360,720}-20260515-*.jsonl`.

**Pre-fix data (config had `wheel_base_m: 0.190 m`):**

| Run | IMU Δθ | Encoder Δθ | Abs err | Frac err | α_4 point |
|---|---|---|---|---|---|
| C1 | +175.31° | +168.32° | -7.00° | **-3.99%** | 0.040 |
| C2 | +354.87° | +335.06° | -19.81° | **-5.58%** | 0.056 |
| C3 | +716.26° | +677.91° | -38.35° | **-5.35%** | 0.054 |

**Critical finding: ~5% systematic bias, same sign across all runs.**
Not noise — calibration. The diff-drive odometry computes
`Δθ_enc = (d_right - d_left) / wheel_base_m`, so if `wheel_base_m` is
too large by factor `k`, encoder Δθ is too small by exactly `1/k`.
Observed `Δθ_enc / Δθ_true ∈ [0.946, 0.960]` → true wheel_base is
~0.181 m (0.181 / 0.190 = 0.953).

This is the same root cause as the rotation overshoot we calibrated
out with the sweep_mission coast model earlier this session: too-large
`wheel_base_m` also makes the Pi command too-large wheel velocities
for a requested ω → bot rotates faster than commanded. One root, two
symptoms.

Translation (Experiment B) shows NO systematic bias because translation
uses `wheel_radius_m` (not wheel_base), which is apparently correct.

**Fix applied:** `body/config.json` motor.wheel_base_m → 0.181 m
(2026-05-15). Pi-side change; requires resync + `motor_controller.py`
restart. After restart, sweep coast model may also want re-fitting
since commanded ω will now produce matching actual ω; the current
coefficients were tuned on top of the 5% over-rotation bias.

**Post-fix verification run (2026-05-15 19:49, C2′):**

| Run | IMU Δθ | Encoder Δθ | Abs err | Frac err | α_4 point |
|---|---|---|---|---|---|
| C2′ post-fix | +355.78° | +353.61° | -2.17° | **-0.61%** | **0.0061** |

Bias collapsed 9× (from -5.58% → -0.61%). The residual 0.61% is
within the noise floor for a single in-place rotation run. Direction
still consistent (encoder slightly *under*-reports), so wheel_base
could in principle nudge lower (0.180 m?) to bisect the bias to 0,
but the residual is already small enough that calling it noise is
defensible.

**α_4 locked at 0.01** — a touch conservative vs the 0.006 point
estimate. Leaves headroom for the direction-asymmetry we haven't yet
measured (only ccw runs) and for slip variance under different floor
conditions.

Notes:
- Direction asymmetry not tested (all 3 runs were ccw). Worth a
  single cw run later to confirm symmetric behavior; if asymmetric,
  the bot has a side-slip preference that the motion model should
  capture as a yaw bias term.
- For the lidar scan-match prior side of the architecture, the bias
  matters: a 5% under-reporting encoder Δθ over a 30 m drive
  accumulates ~5–10° of pose error vs reality, which is precisely
  the amount that flips scan-match in symmetric rooms. Fixing this
  may visibly improve the existing scan-match-as-prior pipeline
  *before* the particle filter lands.

## Output: filter priors

Once the table above is filled, the particle filter will be initialized
with:

```python
# Motion model (per odom step Δs, Δθ):
SIGMA_TRANSLATION_PER_M       = 0.04        # α_1
SIGMA_ROTATION_PER_M_OF_TRANS = 0.017       # α_3 (rad/m)
SIGMA_ROTATION_PER_RAD        = 0.01        # α_4 (post wheel_base
                                            # recal; verified by C2′
                                            # at 0.6% residual error)

# IMU yaw observation (BNO085 game_rotation_vector):
IMU_SIGMA_PER_SAMPLE_RAD = 1.23e-3          # 0.07 deg
IMU_DRIFT_RATE_RAD_PER_S = 3.42e-6          # upper bound; treat as 0
                                            # for most filter purposes
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
