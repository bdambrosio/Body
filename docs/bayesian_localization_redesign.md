# Bayesian Localization & Mapping Redesign

**Status:** Draft plan, 2026-05-15. Multi-session work, no code changes yet.
**Owner:** Bruce + Claude.
**Why this exists:** the current pose stack is a chain of deterministic
point-estimate overwrites masquerading as fusion. The result is brittle
SLAM that loses pose on any non-trivial drive and can't recover without
operator intervention. This document is the plan to replace it with a
proper probabilistic stack that uses all three of our sensors and the
desktop GPU.

---

## 1. Current state — what we have and why it breaks

### 1.1 Pi side (sensor producers, no global localization)

| Layer | Responsibility | Output |
|---|---|---|
| `body/motor_controller.py:352` | `diff_drive.integrate_odometry` on encoder ticks | `body/odom`: (x, y, θ) in odom frame, 50 Hz |
| `body/imu_driver.py` | BNO085 quaternion + accuracy_rad | `body/imu`: quaternion, 100 Hz (game_rotation_vector mode) |
| `body/lidar_driver.py` | Raw RPLidar | `body/lidar/scan`: ranges + angles, ~5 Hz |
| `body/oakd_driver.py` | OAK-D depth + on-demand RGB | `body/depth/oakd`, `body/oakd/rgb` |
| `body/local_map.py` | Lidar + depth fused in body frame using current `body/odom` | `body/map/local_2p5d`: 2.5 D grid, 5 Hz |

Pi has no localizer. Good. Keeps the heavy fusion off the bot.

### 1.2 Desktop side (everything else)

| Layer | What it does |
|---|---|
| `pose_source.py::OdomPose` | Stores raw odom + single (off_x, off_y, off_theta) world transform |
| `nav/slam/imu_yaw.py::ImuYawTracker` | Unwraps IMU yaw, exposes `yaw_at(ts)` |
| `world_map/imu_scan_pose.py::ImuPlusScanMatchPose` | The "fuser": world pose = encoder position + IMU yaw, *replacing* encoder θ. Periodic scan-match via `_on_scan`; on accept, rewrites OdomPose's offsets + yaw_offset. Slew via hard clamps, not uncertainty. |
| `nav/slam/scan_matcher.py::ScanMatcher` | Grid-search over (Δx, Δy, Δθ); returns single best pose + scalar improvement. Throws away the likelihood landscape. |
| `world_map/world_grid.py::WorldGrid` | Accumulates local_2p5d into world frame using `latest_pose()`. `fuse_local_map` + `stamp_traversal`. |
| `world_map/costmap.py::build_costmap` | World grid → lethal/halo/cost with traversal protection |
| `world_map/controller.py::FuserController:514,544` | Drives the grid + pose updates |
| `nav/planner.py` | A* over costmap |
| `nav/follower.py` | Pure-pursuit, reads same fuser pose |
| `nav/safety.py` | Body-frame local_map check (recently rewritten, drift-immune) |

### 1.3 Why this design fails — Bayesian critique

1. **No state distribution anywhere.** Every layer carries a point
   estimate. No covariance, no particle set, no Σ propagated forward.
   When sources disagree, the system picks via hard-coded preference
   and clamp limits, not via principled weighting.

2. **Scan-match discards likelihood.** `ScanMatcher.search` returns
   one (x, y, θ) and a scalar `improvement`. The full likelihood field
   over (Δx, Δy, Δθ) — including multi-modal peaks, ridge-along-corridor
   shapes, the 180°-flip basin in symmetric rooms — is computed and
   thrown away. Hence the `low_improvement` failure when the correct
   pose IS the local optimum and there's nothing better to say.

3. **"Fusion" is sequential overwrite.** `imu_scan_pose._apply_correction`
   rewrites OdomPose's offsets so subsequent queries match the corrected
   pose. Two inconsistent corrections (moving person, reflection, 180°
   decoy) silently replace each other. No memory of prior evidence.

4. **Slew clamps are anti-correlation filters, not Kalman gates.**
   `max_translation_correction_m=0.30` rejects a legitimate 0.5 m
   correction the same way it rejects a bogus 0.5 m flip. Innovation
   gating with proper Σ would discriminate based on direction and
   geometry; we can't because we have no Σ.

5. **IMU yaw treated as truth, not measurement.** `ImuPlusScanMatchPose.pose_at`
   reads `imu_yaw - yaw_offset` as the θ output, bypassing encoder θ.
   Scan-match disagreements re-pin `yaw_offset`. No probabilistic combination
   of IMU drift rate (~0.5°/min in game_rotation_vector) with scan-match
   confidence (varies 0.1–0.7).

6. **Map fuses with no pose uncertainty.** `WorldGrid.fuse_local_map`
   uses `latest_pose()`. Wrong pose → observations land in wrong cells →
   grid corrupts → next scan-match has bad anchor → pose drifts further.
   Closed feedback loop with no uncertainty representation to dampen it.

7. **No particle filter, not because we tried, but because the
   architecture is point-estimate end-to-end.** Adding particles on top
   of OdomPose-with-overwrites would be particles watching a deterministic
   process. The fuser has to be rebuilt around posterior propagation.

---

## 2. Target architecture

**Principle:** every layer carries a distribution. Observations update
the distribution via Bayes' rule. The point estimate falls out as
posterior mean (or MAP) for downstream consumers, but the distribution
is the primary object.

### 2.1 Sensor fusion node — particle filter on desktop GPU

- State per particle: $(x, y, \theta)$.
- Population: 5,000–10,000 particles. 12 GB on RTX 5060 Ti is overkill.
- Initial spread: cluster around start pose with Σ₀ chosen by use case
  (small for "resume" / known start, large for global re-localization).

**Predict (50 Hz, on every odom message):**
- Per particle, sample noisy odom increment from
  $\Delta_i \sim \mathcal{N}(\Delta_\text{odom}, \Sigma_\text{odom}(\Delta_\text{odom}))$
- $\Sigma_\text{odom}$ grows with translation magnitude and angular speed
  (slip term). Calibrated empirically — needs real noise data.
- IMU yaw delta treated as a *relative-yaw observation* in the same step,
  not as truth replacing θ. Σ_IMU from BNO085's `accuracy_rad` field
  (or constant in game_rotation_vector mode).

**Update (asynchronous, weights only):**
- **Lidar scan** (5 Hz): per-particle likelihood via the scan-match
  score field at each particle's pose. GPU-parallel — embarrassingly so.
  Replaces the current grid-search-argmax with full Bayesian update.
- **AprilTag detection** (opportunistic, *sparse*): when a tag is in
  frame, multivariate Gaussian likelihood centered at the
  inverse-projected pose. Tight Σ. Posterior collapses on these events,
  but **the system must converge without them.**
- **Visual place recognition** (1–2 Hz on RGB): top-k matched memorized
  frames contribute a global-location prior. Soft signal; smooths.

**Resample** on effective sample size drop (KLD-sampling or threshold).
Inject randomness on high-motion ticks to keep diversity.

**Output:** posterior mean, covariance, and full sample for diagnostics.

### 2.2 Mapping — pose-uncertainty-aware

Start with **best-particle mapping** (FastSLAM-style): each `local_map`
update uses the highest-weight particle's pose. Periodically check the
top-N particles haven't diverged; if they have, snapshot multiple maps.

Future option (more correct, heavier): per-particle Gaussian smear of
observations across cells based on pose covariance, so uncertain pose
updates a *region* of cells with reduced weight.

### 2.3 Scan matcher — returns likelihood field, not argmax

Modify `ScanMatcher.search` to return the full $(N_x \times N_y \times N_\theta)$
score grid normalized to a likelihood. Particle filter queries this
field at each particle's pose. Multi-modal peaks are preserved; filter
resolves them with other evidence.

### 2.4 RGB observation streams

**Tier A: AprilTags (opportunistic, do not architect around).** When
in frame, near-zero-Σ pose anchor. Sparse distribution — bot may go
minutes without seeing a tag. Architecture must work in their absence.
Plumb as one observation type among many.

**Tier B: Visual place recognition (the real RGB workhorse).** Build a
feature bank (NetVLAD, DBoW2, DINOv2 + cosine — TBD) from training
drives. At inference, query each RGB frame for top-k similar memorized
frames; their tagged poses contribute a global-location likelihood. The
5060 Ti makes this trivially fast. **This is the primary defense against
the symmetric-room flip and accumulated drift; AprilTags are bonus.**

**Tier C: Tight VIO (OpenVINS-class).** Only after A and B are paying off.

### 2.5 Persistent maps + global re-localization

- Save world grid + RGB feature bank + AprilTag world poses on shutdown.
- Load on boot. Run particle filter with broad initial spread; scan-match
  likelihood + RGB place recognition converge over the first few seconds.
- The bot starts knowing the house.

### 2.6 Architecture changes that fall out

- `OdomPose.update` becomes one observation stream, not the trunk.
- `_apply_correction` and slew clamps disappear. Filter handles "is this
  innovation believable" via likelihood naturally.
- `low_improvement` errors disappear — multi-modal posteriors are valid.
- `traversal_protection` becomes unnecessary — filter naturally
  decreases occupancy uncertainty in driven cells.
- `relocate` button maps to "spray initial particles broadly, let filter
  converge."

### 2.7 Sanity / safety layer (keep as-is)

Body-frame local_map forward-arc safety check (this session) stays.
Drift-immune, orthogonal to global localization, correct.

---

## 3. Phased implementation plan

Each phase is a coherent, testable, mergeable brick. Estimated session
counts assume Bruce-paced work (focused but not full-time).

### Phase 0 — Foundation. **(1 session)**

**Goal:** noise-model calibration data.

- Collect odom drift data: drive straight 5 m, measure end-pose error.
  Repeat for rotation-only. This gives the variance scaling for
  $\Sigma_\text{odom}$.
- Collect IMU yaw drift data: bot stationary 5 minutes, log yaw vs
  time. Gives game_rotation_vector drift rate empirically.
- Collect scan-match likelihood landscape on a few representative
  scans (corridor, open room, symmetric room). Verify the multi-modal
  cases match our intuition.

**Deliverable:** `docs/noise_models.md` with measured Σ values and
the data behind them.

**Validation:** Bruce reviews — these become the priors for everything
downstream. Wrong noise models → bad filter.

**Dependencies:** none.

**Open questions:** what's the cleanest way to log these? Probably
extending nav tracing (per `project_body_nav_tracing_2026_05_12`).

---

### Phase 1 — Likelihood-field scan matcher. **(1–2 sessions)**

**Goal:** scan matcher returns full score field, not just argmax.

- Modify `nav/slam/scan_matcher.py::ScanMatcher.search` to optionally
  return the $(N_x \times N_y \times N_\theta)$ score grid.
- Add a `likelihood_at(x, y, θ, score_field)` lookup utility.
- Backward-compatible: argmax interface still returns the same
  PoseEstimate; new interface returns the full field.

**Deliverable:** updated `ScanMatcher`, unit tests, demo script that
plots likelihood field for a hand-picked scan.

**Validation:** likelihood field shows the expected peaks for a
known-symmetric scan. Unit tests pass. Shadow-mode comparison: argmax
result unchanged from current behavior.

**Dependencies:** Phase 0 nice-to-have but not blocking.

**Open questions:** normalization. Raw scores aren't probabilities;
need to decide softmax temperature or similar. Probably calibrate
against Phase 0 likelihood landscapes.

---

### Phase 2 — Particle filter pose source (CPU). **(2–3 sessions)**

**Goal:** drop-in replacement for `ImuPlusScanMatchPose` that runs a
particle filter and reports posterior mean/cov via the same
`PoseSource` interface.

- New class `world_map/particle_filter_pose.py::ParticleFilterPose`.
- Predict from odom + IMU yaw observation.
- Update from scan likelihood field (Phase 1 output).
- 1,000–2,000 particles on CPU first. GPU port in Phase 4.
- Resampling: KLD-adaptive or stratified, TBD.
- Exposed via `--pf` flag in `desktop/nav/__main__.py`, parallel to
  current `--slam`. Easy A/B comparison.

**Deliverable:** new pose source, integration with FuserController,
A/B mode in nav launcher.

**Validation:**
- Bench against logged data from prior sessions. Replay a drive, compare
  particle filter trajectory to current scan-match trajectory. Should be
  comparable or better, especially through rotations and the symmetric-
  room flip cases.
- Confirm posterior covariance reflects uncertainty (large during
  rotation, small after good scan match, etc.).
- No regressions in best-case scenarios.

**Dependencies:** Phase 1 (likelihood field), Phase 0 noise models.

**Open questions:**
- Proposal distribution: how aggressive should slip modeling be?
- Resampling threshold: ESS / N_eff ratio.
- Initial particle spread for various "where am I" cases.

---

### Phase 3 — AprilTag observation (opportunistic). **(1 session)**

**Goal:** when a tag is in frame, particle filter posterior tightens
to anchor pose. Bot operates fine in absence of tags.

- Subscribe to OAK-D AprilTag detection topic (or wire up detection
  via DepthAI Python API if Pi side doesn't publish yet — Pi-side
  change may be needed).
- Calibration file `config/apriltag_poses.yaml`: tag_id → (x_w, y_w,
  θ_w, ±Σ).
- On detection: compute the implied bot pose from the relative tag
  pose + known world pose; apply as Gaussian likelihood observation
  to the filter.
- **Critical:** filter must converge without tags. Test by running a
  drive in a tag-free area and confirming it tracks reasonably from
  odom + lidar + IMU alone.

**Deliverable:** AprilTag observation stream, calibration loader,
documentation of how to add tags.

**Validation:**
- Place 1–2 tags. Drive past one. Confirm posterior tightens visibly
  (covariance display in nav UI).
- Drive in a tag-free area for 30 s. Confirm filter doesn't blow up.

**Dependencies:** Phase 2 (filter to absorb observations).

**Open questions:**
- Pi-side AprilTag detection vs desktop-side. OAK-D's neural compute
  could run it onboard; bandwidth-cheap.
- Tag size, mounting heights, lighting requirements.

---

### Phase 4 — GPU port. **(1–2 sessions)**

**Goal:** particle filter likelihood evaluation runs on RTX 5060 Ti.

- Port the per-particle scan likelihood evaluation to CUDA via
  PyTorch (probably) or CuPy (lighter dep). 5,000–10,000 particles per
  scan in <5 ms target.
- Keep CPU fallback for non-GPU desktops.
- The rest of the filter (predict, resample) stays on CPU — fast
  enough at scale.

**Deliverable:** GPU likelihood kernel, CPU/GPU mode switch.

**Validation:**
- 10k particles per 5 Hz scan, <50% of one GPU's time.
- Bit-exact agreement with CPU version (modulo floating-point order).

**Dependencies:** Phase 2 working on CPU.

**Open questions:**
- PyTorch vs CuPy vs raw CUDA. PyTorch is the team's existing
  competency, has solid kernel-fusion. Probably the right call.

---

### Phase 5 — Pose-aware mapping. **(1–2 sessions)**

**Goal:** `WorldGrid` updates use best-particle pose (or per-particle
Gaussian smear as future option).

- Modify `world_map/controller.py::FuserController` to pass the
  best-particle pose (highest weight) to `fuse_local_map` and
  `stamp_traversal`.
- Add divergence check: if top-N particles disagree on pose by more
  than X cm or θ, log and consider snapshotting multiple map versions.
- Remove `traversal_protection` overrides — replace with proper
  per-cell occupancy uncertainty from the filter.

**Deliverable:** rebuilt mapping integration, removed legacy traversal
protection.

**Validation:**
- Map quality on the same drive: should be visibly cleaner than current
  best-effort grid because pose is more accurate.
- No regressions on costmap-based planning.

**Dependencies:** Phase 2 (particles + best-particle pose available).

**Open questions:**
- Multi-map storage when particles disagree. Probably defer to a
  Phase 5.5 if it becomes a real problem.

---

### Phase 6 — Visual place recognition. **(2–3 sessions)**

**Goal:** RGB-driven global-location prior. The primary defense
against accumulated drift; replaces the role we briefly considered
giving to AprilTags.

- During training drives, store (RGB feature vector, tagged pose) in
  a feature bank.
- At inference, each RGB frame queries top-k similar features; their
  poses contribute a soft likelihood to the filter.
- Choice of feature: NetVLAD (well-established), DINOv2 + cosine sim
  (modern, no fine-tune needed, runs on GPU), DBoW2 (lighter, classic).
  Probably DINOv2 first — easy to set up, paper-quality results out
  of the box.
- RGB request rate: 1–2 Hz to keep bandwidth manageable, gated on bot
  motion or filter uncertainty being high.

**Deliverable:** feature bank builder, runtime VPR observation stream.

**Validation:**
- Drive a loop, return to start. Without VPR, drift visible. With VPR,
  posterior should snap back near start at the loop closure.
- Tag-free environment.

**Dependencies:** Phase 4 (GPU available), Phase 2 (filter to absorb).

**Open questions:**
- Feature bank size limits (memory + query speed).
- How to handle illumination changes between training and inference
  drives.
- Online learning: should the bank update during normal operation?

---

### Phase 7 — Persistent maps + warm boot. **(1 session)**

**Goal:** bot starts knowing the house.

- Save world grid + feature bank + AprilTag config on shutdown.
- Load on boot.
- Global re-localization at startup: broad particle spread, converge
  over first few seconds using scan + VPR.

**Deliverable:** persistence + warm-boot path.

**Validation:**
- Restart nav between sessions. Bot localizes within 5–10 seconds.
- Plan goals before driving (already-known map).

**Dependencies:** Phases 2, 6.

---

### Phase 8 — Cleanup. **(1 session)**

- Remove `_apply_correction`, slew clamps, `relocate` button (or rewrite
  it as "spray particles" instead of "find single best pose").
- Remove `OdomPose`'s "world transform" hack.
- Documentation pass: this doc + updated specs in `docs/`.

---

## 4. Out-of-scope / explicit non-goals

- **ROS 2 / Nav2 migration.** Decided in the brainstorm we keep the
  Zenoh + custom-Python stack; this redesign improves the *fusion* layer,
  not the framework.
- **Replacing the planner.** A* + costmap is fine. The reason planning
  felt broken was bad pose; fix pose, planner will be fine.
- **Replacing the follower.** Same logic.
- **Replacing local_map fusion on Pi.** It's clean: body-frame, no global
  pose. Keep.
- **AprilTags as primary localization.** Architecture works without them;
  they're a bonus observation.

---

## 5. Risks + things that might force re-scoping

1. **Compute budget on Pi for AprilTag detection.** If the Pi can't
   handle it, detect on desktop using streamed RGB. Bandwidth hit but
   workable.
2. **Pi → desktop bandwidth.** RGB at 1–2 Hz is fine; if VPR wants 5 Hz,
   may need to reconsider. JPEG compression is already in place.
3. **Filter divergence in featureless areas.** Long hallways, plain
   walls — scan-match has a likelihood ridge. The filter will represent
   this correctly (spread along the corridor axis), but the bot needs
   to know "I am uncertain along this axis" and behave accordingly. Nav
   needs to consume covariance, not just mean.
4. **VPR robustness to illumination / furniture changes.** A robust
   feature (DINOv2) helps; full retraining is a fallback.
5. **GPU availability.** If the 5060 Ti is busy with other workloads,
   CPU path needs to remain viable. Phase 2 deliberately ships CPU
   first; GPU is throughput optimization, not correctness.

---

## 6. What I (Claude) need from Bruce

- **Noise-model calibration data** (Phase 0). I can describe the
  protocol but you'll run the bot and we'll do the math together.
- **Sanity check on the proposal distribution** for the particle
  filter — odom slip is the gnarliest term.
- **Review of resampling strategy** — KLD-adaptive vs stratified vs
  systematic. I'll pick a default; you'll tell me if it's right.
- **Feature/VPR taste** — NetVLAD vs DINOv2 vs DBoW2 is a judgment
  call balancing modernity, robustness, and compute cost.

## 7. What I can drive alone

- All plumbing (subscriber wiring, integration with FuserController,
  UI exposure).
- Scan-matcher likelihood-field refactor (Phase 1).
- Particle filter core (Phase 2 CPU implementation).
- GPU port (Phase 4).
- AprilTag observation stream (Phase 3).
- Persistence layer (Phase 7).
- Cleanup (Phase 8).

---

## 8. Session-resumption notes for future Claudes

This is multi-session work. Each session should:

1. Re-read this document, especially the "current phase" status (track
   in this file as we go — add a "**Phase N status**" line at the end of
   each phase as it completes).
2. Re-read `MEMORY.md` for any new project context.
3. Confirm phase boundaries with Bruce before diving in.
4. Update this doc with what was done and what's open.

If the design diverges from this plan, **update this doc**, don't
just ship the divergence.

---

## 9. Status log

- **2026-05-15:** Document drafted. Phase 0 pending.
