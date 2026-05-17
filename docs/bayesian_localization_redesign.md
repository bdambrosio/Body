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

### Phase 0 — Foundation. **(1 session) — DONE 2026-05-15**

**Goal:** noise-model calibration data.

Numbers locked, recorded in `docs/noise_models.md`:

| Parameter | Value | Source |
|---|---|---|
| α_1 (trans σ / m) | 0.04 | Experiment B (3 runs at 1/3/5 m) |
| α_3 (rot σ / m of trans) | 0.017 rad/m | Experiment B cross-term |
| α_4 (rot σ / rad) | 0.01 | Experiment C2′ post-fix |
| σ_IMU per sample | 1.23 mrad (0.07°) | Experiment A (7 min stationary) |
| IMU drift rate | ≈ 0 (-0.012 °/min upper bound) | Experiment A |

**Two big findings beyond the numbers themselves:**

1. **`wheel_base_m` was miscalibrated by 5% (0.190 → 0.181 m).** Same
   root cause as the rotation overshoot the sweep coast model was
   patching this session. Calibration > workarounds — the coast model
   coefficients are now stale and should be re-fit (the bot will be
   rotating closer to commanded ω now). Not a Phase 1 concern but
   flagged in §10.

2. **BNO085 IMU yaw is much quieter than assumed.** game_rotation_vector
   drift is ~50× below the textbook 0.5–1°/min estimate; per-sample
   σ is sub-tenth-of-a-degree. **Architecture implication:** the IMU
   is a near-rigid yaw constraint, not a drift source. The dominant
   yaw uncertainty in the filter will come from encoder slip (α_3, α_4),
   not from the IMU. The IMU yaw observation can use a very tight Σ.

**Things Phase 0 deliberately did NOT measure** (deferred, see §10):
- Direction asymmetry (cw vs ccw — only ccw was tested)
- Cross-term α_2 (translation σ from rotation — pure rotation tests
  didn't analyze residual translation drift)
- Floor-surface dependence (one set of conditions)
- Scan-match likelihood landscape characterization — deferred until
  Phase 1 has the score field plumbing in place, then this becomes
  a natural validation step inside Phase 1.

---

### Phase 1 — Likelihood-field scan matcher. **(1–2 sessions)**

**Goal:** scan matcher returns full score field, not just argmax.

- Modify `nav/slam/scan_matcher.py::ScanMatcher.search` to optionally
  return the $(N_x \times N_y \times N_\theta)$ score grid alongside
  (or instead of) the argmax `PoseEstimate`.
- Add a `likelihood_at(x, y, θ, score_field)` lookup utility — the
  filter will call this once per particle to compute the per-particle
  observation likelihood.
- Backward-compatible: existing `search()` argmax interface still
  returns the same `PoseEstimate` so `ImuPlusScanMatchPose` keeps
  working unchanged. New optional output (kwarg `return_field=True`
  or similar) exposes the field.

**Deliverable:** updated `ScanMatcher`, unit tests, demo script that
plots likelihood field for a hand-picked scan (corridor, open room,
symmetric room — provides the Phase 0 §"deferred scan-match landscape"
characterization as a side benefit).

**Validation:**
- Likelihood field shows expected peaks for known-symmetric scan
  (the 180°-flip basin is visible and quantifiable, not just argmaxed
  away).
- Argmax-mode regression: existing scan-match tests + a shadow-mode
  run against captured session data shows the argmax result is
  identical to the current implementation, bit-for-bit.

**Dependencies:** Phase 0 priors are useful but not blocking — we
need σ_IMU to figure out reasonable score-→-likelihood normalization
later, but the field itself is unit-agnostic.

**Open questions:**
- **Normalization.** Raw scores from `_score_at` aren't probabilities.
  Options to convert: (a) softmax with a temperature chosen from the
  observed score-spread distribution at calibrated locations, (b)
  Gaussian fit around the peak — return mean + Σ, (c) log-likelihood
  proportional to score and let the particle filter normalize via
  resampling. Probably (c) for the first pass — particle weight
  ratios only need *relative* likelihood; absolute normalization is
  only needed for divergence diagnostics. Document this choice in
  the Phase 1 PR.
- **Memory.** Full $(N_x \times N_y \times N_\theta)$ grid at current
  config is ~40×40×72 floats ≈ 460 KB per call; fine for one scan/sec.
  At 10 Hz it's 4.6 MB/s allocation churn — would want a preallocated
  buffer. Probably fine to defer.
- **Coordinate frame.** Current `search()` takes a `prior_pose` and
  returns absolute world poses. The field should be indexed by
  *delta from prior* so callers can shift it under a different prior
  without recomputing. Easy change but worth being deliberate.

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
- **2026-05-15:** Phase 0 kickoff. Analysis tooling shipped:
  `scripts/phase0_imu_stationary.py` and `scripts/phase0_odom_drive.py`.
  Procedure + result template at `docs/noise_models.md`. No Pi-side
  changes. Awaiting Bruce's experiment runs A/B/C.
- **2026-05-15 (later):** Phase 0 complete. All five priors locked
  (α_1 = 0.04, α_3 = 0.017 rad/m, α_4 = 0.01, σ_IMU_sample = 1.23 mrad,
  drift ≈ 0). Side-effect: wheel_base_m calibration bug found and
  fixed (0.190 → 0.181 m); same root cause as the sweep rotation
  overshoot. Ready for Phase 1.
- **2026-05-16:** Phase 1 complete. `ScanMatcher.search` gained a
  `return_field=True` kwarg; result carries an optional
  `ScoreField(field, dx_axis, dy_axis, dth_axis)` of dtype float32,
  indexed by delta-from-prior. `likelihood_at(dx, dy, dth, field)`
  helper does trilinear interp. Argmax-path is bit-for-bit unchanged
  when the flag is off (regression test asserts this). Demo script
  `scripts/phase1_likelihood_field_demo.py` plots corridor / open-room
  / symmetric-room scenes — the symmetric (4×4 m) square exposes a
  4-peak dθ marginal at 0°/±90°/±180°, exactly the basin a point
  estimate would silently collapse. Phase 1 open-question decisions
  for the PR record: (a) normalization deferred — raw correlation
  scores returned as log-likelihood up to additive constant, particle
  filter (Phase 2) normalizes via importance weights; (b) coord frame
  indexed by delta-from-prior in world frame (lets a filter share one
  field across particles whose priors sit inside the window); (c)
  preallocated-buffer micro-optimization deferred — current ~6 KB
  field allocation per scan is fine at 1–10 Hz.
- **2026-05-16 (Phase 4):** Particle filter GPU support landed.
  `ParticleFilterPose` already had `cfg.device` plumbed; we just
  needed: (a) launcher flags `--pf-device {auto,cpu,cuda}` and
  `--pf-particles N`, with `auto` resolving to cuda if available;
  (b) plumb-through to `FuserConfig.pf_device` / `pf_n_particles`
  → `FuserController` (particle branch) → `ParticleFilterPoseSource`.
  Shadow driver (`--pf-shadow`) picks up the same device + particle
  count so it benches the same config production runs with.
  Scan_matcher.search stays on CPU — the numpy vectorization (commit
  5447182) made it 22 ms, no longer a bottleneck worth porting.
  Pre-launch microbench on Bruce's workstation (RTX PRO 6000
  Blackwell Max-Q): 1k particles CPU 0.08 ms vs GPU 0.12 ms
  (overhead loses), 10k CPU 0.49 ms vs GPU 0.13 ms (4× win), 100k
  CPU 4.03 ms vs GPU 0.13 ms (30× win). GPU stays flat to 100k.
  VRAM: torch+CUDA fixed overhead ~700 MB – 1 GB, particle state +
  transients add < 50 MB even at 100k particles. Plenty of room for
  Phase 6's DINOv2-B (~1 GB) on the same device. 3 new
  `@skipUnless(cuda)` tests in test_particle_filter_pose.py covering
  state-lives-on-cuda, end-to-end pipeline, cov_at returns
  (3, 3) on cuda. 114 desktop tests passing (111 + 3 cuda when
  available). Phase 4 closes the original redesign plan's GPU
  port goal at the scope worth doing — scan_matcher CUDA kernel
  port would be a future "if needed" item.

- **2026-05-16 (end of day):** Phases 0–3, 5 (partial), 5.5 Variant A,
  and 8 (cutover prep) all landed in one session. `--pf` is the
  production pose source for live autonomous nav; particle filter has
  driven an "ARRIVED on goal" mission start-to-finish (run 13:21 →
  follow:ARRIVED goal=0.17 m) and a 99 s manual drive with 99.7%
  predict success, 0 teleports, 0 catastrophic divergences. Doorway
  scrape failure mode partly addressed: planner now requires ≥1 cell
  clearance from any lethal cell, σ-aware vote weighting reduces
  phantom-obstacle planting from high-σ scans. Two doorjamb collisions
  remain instructive: in both cases, the planner was producing
  zero-clearance paths and the costmap was showing phantom narrowing
  on one side. After the clearance + σ-aware patches plus the
  operational WiFi fix (single-AP `GL-MT3000-bee` on wlan1, wlan0
  disabled — see `deploy/NETWORK.md`), the session ended with "solid
  wifi, no hiccups." Total: 111 desktop tests, all passing. Remaining
  Phases: 4 (GPU port, throughput optimization), 5 full (per-cell
  occupancy uncertainty — Variant A is the cheap stand-in), 6 (VPR,
  the real drift cure), 7 (persistent maps), 8 (removing
  ImuPlusScanMatchPose). None are blocking; the system is operational
  on the particle filter.

- **2026-05-16 (evening):** Phase 5.5 Variant A — σ-aware vote
  weighting in WorldGrid. Cheaper alternative to the full per-cell
  occupancy uncertainty rewrite; addresses the same root cause (poor
  poses planting phantom obstacles in the grid) at ~zero CPU cost.
  Each scan's per-vote contribution to the grid is now scaled by
  ``(σ_nominal / σ_now)²`` where σ_now is the filter's posterior
  std_xy from cov_at(). Tight pose (σ ≤ 2 cm nominal) → full weight
  (legacy behaviour). 2× nominal → 0.25× weight. 4× nominal → 1/16×
  weight. Floor at 0.05 prevents transient catastrophic σ from
  silencing the grid entirely.
  Motivated by the 13:21 doorway scrape (filter ran into the doorjamb;
  costmap showed phantom narrowing on the right by 8–24 cm). 4×
  higher corr/match than earlier short drives suggested the filter
  was working harder than usual to correct itself; those uncertain
  moments planted phantom votes.
  Threading: FuserController calls cov_at(cap_ts) per fusion step,
  passes pose_weight_scale to fuse_local_map. Point-estimate sources
  return cov_at=None → scale stays 1.0, no behaviour change. 10 new
  tests covering the helper and the fuse path; 107 total desktop.
  Live validation: re-run the same drive and look at whether the
  doorway maps with less phantom narrowing.

- **2026-05-16 (later still^2):** Phase 5 — pose-aware mapping — partial.
  Two of three Phase 5 deliverables landed; the third (replacing
  `traversal_protection` with per-cell occupancy uncertainty) is a
  bigger architectural change deferred to a follow-up.
  - Added `PoseSource.best_pose_at(ts)` to the abstract interface,
    default-delegating to `pose_at` so the legacy sources need no
    changes. `ParticleFilterPoseSource.best_pose_at` returns
    `posterior_mode()` — the highest-weight particle, sharper than
    the mean and less prone to smearing walls during multi-modal
    phases.
  - `FuserController._fusion_loop` now calls `best_pose_at` (was
    `pose_at`) when no Pi-stamped odom anchor is present — the world
    grid sees the MAP particle. The Pi-anchor branch still uses
    `to_world` since that's a deterministic frame transform.
  - `FuserController._traversal_loop` also uses `best_pose_at` for
    `stamp_traversal`. The pose-trail breadcrumb keeps using the
    posterior mean (smoother for the operator UI).
  - Divergence warning: per-tick `cov_at` query → log warning at
    most once per 5 s when σ_xy > 0.20 m or σ_θ > 15°. Point-estimate
    sources return `cov_at = None` and skip the check silently. Plan
    §3 Phase 5 "consider snapshotting multiple map versions" remains
    a future option (Phase 5.5 territory).
  - 97 desktop tests (2 new for best_pose_at), all passing.
  - Live validation pending: re-run `--pf` and check that map
    quality is at least no worse than the 13:21 ARRIVED-on-goal
    screenshot. Phase 8 cleanup (removing `ImuPlusScanMatchPose`)
    waits for one more session of confidence.

- **2026-05-16 (much later):** Phase 8 cutover *prep* complete (this is
  the structural promotion before flipping the production switch on
  live data). New `ParticleFilterPoseSource` in
  `desktop/world_map/particle_filter_pose_source.py` implements the
  full `PoseSource` interface (`pose_at`, `latest_pose`,
  `rebind_world_to_current`, `to_world`, `cov_at`, `connect`/`disconnect`).
  Owns its zenoh subscriptions to `body/imu` and `body/lidar/scan`;
  consumes `body/odom` via `PoseSource.update()` driven by
  `FuserController._on_odom`. World-offset bookkeeping mirrors
  `OdomPose` (off_x, off_y, off_theta captured at seed/rebind) so the
  Pi's odom-frame anchor on `body/map/local_2p5d` messages converts
  correctly via `to_world`. `FuserController` grew a tri-state
  `pose_source_type: "odom" | "slam" | "particle"`; `slam_enabled`
  retained as a back-compat alias. Nav launcher: new `--pf` flag,
  mutually exclusive with `--slam`. 11 new tests (95 total desktop),
  all passing. Phase 5 (WorldGrid consumes filter pose for grid
  building) is the next session — for now, `--pf` runs the filter as
  the source-of-truth pose but the grid still uses `pose_at` the
  same way it always has. Live validation: pending Bruce running
  `python -m desktop.nav --pf ...` and watching for map smearing
  vs the existing `--slam` baseline.

- **2026-05-16 (later still):** Phase 3 plumbing complete (battery-
  recharge session). Detector + calibration loader + observer +
  launcher flag, all unit-tested (18 new tests, 84 total desktop).
  Detection runs desktop-side via pupil-apriltags (plan §5 risk #1
  predicted this); tag36h11 default. Observer subscribes to
  body/oakd/rgb; optionally drives capture_rgb at a configurable
  rate via body/oakd/config (default 1 Hz). For each known tag in
  the calibration YAML, computes the implied bot world pose via
  `T_world_body = T_world_tag · T_cam_tag⁻¹ · T_body_cam⁻¹`, then
  applies `pf.observe_xy_world(x, y, σ_xy)` and `pf.observe_imu_yaw(θ, σ_θ)`
  to the same filter the scan-likelihood update writes to. Trace
  records of type `apriltag_obs` land in the same JSONL the scan_obs
  records do, so legacy/filter/tag streams are all aligned by ts for
  offline analysis. CLI: `--apriltag-config config/apriltag_poses.yaml`
  + `--apriltag-request-hz 1.0`. Example calibration at
  `config/apriltag_poses.yaml.example` with documentation of frame
  conventions (world ENU, body x-forward, camera OpenCV). Validation
  per plan: filter must still converge without tags — that's the
  default code path when `--apriltag-config` is omitted. Live
  validation pending Bruce printing/mounting a tag and recording a
  trace; the plumbing is testable end-to-end via the mocked-detector
  integration test (`test_apriltag.TestObserverFlow`).

- **2026-05-16 (later):** Phase 2 complete on CPU. Four sub-commits:
  (2.1) `desktop/world_map/particle_filter_pose.py` — `ParticleFilterPose`
  with motion model (Phase 0 α priors, σ floor for diversity), IMU yaw
  Gaussian observation, posterior mean. (2.2) `update_from_scan_likelihood`
  + vectorized trilinear interp into the Phase 1 score field — einops
  `rearrange` + `reduce` over a (P, 2, 2, 2) corner cube; auto-temperature
  defaults to `max(field.std(), 1.0)` so a 1σ score gap = 1 nat of
  log-weight (Phase 1 open-question (a) resolved). (2.3) Systematic
  low-variance resampling, N_eff < N/2 gating, 3×3 posterior cov via
  `torch.einsum`, `FilterDiagnostics` snapshot. (2.4) `shadow_pf_driver.py`
  attaches to FuserController's session; `--pf-shadow PATH` flag in
  `desktop/nav/__main__.py` writes one JSONL record per scan tick
  comparing legacy pose to filter posterior. No production flip.
  Test count: 24 filter unit + 6 driver integration + 33 pre-existing
  = 63 desktop tests passing. torch + einops + matplotlib added to
  `desktop/requirements.txt` (commit 6cf4bd4). Phase 2 plan §6 ask
  "sanity check on the proposal distribution" still pending — Bruce
  to review the (Δs, Δθ) Gaussian-additive model in
  particle_filter_pose.py::predict against α priors. Ready for live
  Pi shadow-mode trace capture and offline analysis before Phase 3.

---

## 10. Known loose ends (not blocking, track + revisit)

Things we've noticed but deliberately deferred. None blocks the next
phase; capture here so they don't get lost.

- **Sweep coast model is stale.** `desktop/chassis/sweep_mission.py`
  has `ROTATE_COAST_LINEAR_DEG_PER_DPS = 0.224` and
  `ROTATE_COAST_QUADRATIC_DEG_PER_DPS2 = 0.0128`, both fitted on top
  of the 5% wheel_base over-rotation bias. Post-fix the bot will be
  rotating closer to commanded ω, so these coefficients now
  over-anticipate coast and sweeps will land slightly *under*
  step_deg. Acceptable for now; re-fit when convenient (one
  calibration run at 30 dps + 15 dps).
- **Encoder rotation bias still slightly negative (-0.61%) post-fix.**
  wheel_base 0.181 m bisects the bias close to zero but not exactly.
  Could nudge to 0.180 to bisect tighter. Marginal; defer unless we
  see a reason. With one post-fix sample we can't distinguish
  remaining-bias from slip noise anyway.
- **Direction asymmetry untested.** Phase 0 C runs were all ccw.
  Worth one cw run at 360° to confirm; if asymmetric, filter's
  motion model wants a directional bias term (encoder Δθ × 0.95 vs
  × 1.05 depending on sign of ω).
- **α_2 not measured.** Cross-term: translation σ from rotation.
  Would require a pure-rotation run analyzed for residual encoder
  (x, y) drift. Not currently emitted by `phase0_odom_drive.py`.
  Easy 10-line addition; do it next time we need C-style data.
- **Floor-surface dependence undocumented.** Phase 0 calibration was
  one set of conditions. If the bot operates on multiple surfaces
  (wood floor vs rug vs tile), α_1 and α_4 will vary. Worth a
  quick re-run when conditions change significantly.
- **min_drive_pwm trim** (0.18 → 0.16, Bruce-side commit 0b10018
  on 2026-05-15). Not yet validated with a fresh translation/rotation
  noise pass. Likely fine — lower static kick is gentler — but if
  α_1 drifts up in later sessions, the trimmed kick could be a
  contributor.
- **AprilTag observation stream (Phase 3) intentionally minimal.** The
  current architecture doesn't depend on tags. They're a Phase 3
  opportunistic observation; spatial distribution will be sparse. Don't
  re-architect around them.
- **scan_match coordinate frame question.** Phase 1 likely surfaces
  the question: is the score field indexed by absolute world pose or
  by Δfrom-prior? Δfrom-prior is more useful for the filter (each
  particle has its own prior). Defer concrete answer to Phase 1 PR.

These get re-evaluated at the start of every phase. If something here
turns out to be blocking, promote it to the active phase's checklist.
