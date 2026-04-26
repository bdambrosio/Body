# Nav stack test plan

Status: Phase 0 of the agreed phased plan (memory: `project_body_nav_phased_plan_2026_04_26.md`). This document is the foundation against which subsequent phases are validated. Each phase ends with a real-world pass on the relevant subset.

The first autonomous round-trip drive (2026-04-25) validated the core path: pose → costmap → A* → pure-pursuit → cmd_vel → forward-arc safety. This plan covers what we don't yet exercise.

## Test categories

### A. Unit / dry-run (no hardware)

These run against synthetic grids and synthesized poses. No Pi, no chassis. Purpose is to lock invariants we don't want to debug live.

- **A1. Mission state transitions**
  - IDLE → FOLLOWING → ARRIVED on linear path with synthetic pose advancing toward goal.
  - FOLLOWING → PAUSED("no_pose") when synthetic pose stream is paused; resumes on fresh pose.
  - FOLLOWING → PAUSED("no_path") when costmap is mutated to block the path mid-mission.
  - PAUSED → RECOVERING(action) when escalation policy fires; recovery action runs to completion.
  - RECOVERING → FOLLOWING on success; → FAILED after retry budget exhausted.

- **A2. Replan invariants**
  - Replan on every tick when goal exists. Atomic path swap: follower never sees a half-updated path.
  - REPLAN_FAILED event emitted exactly when planner returns `not ok`.
  - Path identity (object id or hash) only changes on actual replan, not on every tick (avoid log churn).

- **A3. Recovery classifier**
  - Goal in unknown → classifies as `goal_in_unknown`.
  - Goal inside lethal halo → classifies as `goal_in_lethal_halo`.
  - Robot boxed in by lethal cells → classifies as `boxed_in`.
  - Path exists but routing fails (start unreachable) → classifies as `start_unreachable`.

- **A4. Primitive primitives**
  - 360-rotate: integrates yaw to ≥ 2π and stops; cancellable mid-rotation; respects pose-loss (pauses if pose ages out).
  - Back-up: integrates distance to N meters and stops; aborts on rear-arc lethal hit; respects pose-loss.

- **A5. Streaming RGB plumbing**
  - With streaming on, frames update at the configured rate (±20%); pending-id correlation does not gate streaming-mode replies.
  - With streaming off, on-demand Request RGB still works.
  - Streaming on + on-demand button pressed → operator gets the on-demand reply, streaming continues without losing frames.

### B. Hardware-in-loop, controlled

Run on the live robot in an empty room with the operator within reach of ALL-STOP. Each scenario has a clear pass/fail bar.

- **B1. Pose-loss injection.** Operator covers the LIDAR (or otherwise causes scan-match to lose lock once SLAM is promoted; for v1 odom-only, simulate by killing odom publisher temporarily). Mission must enter PAUSED("no_pose") within `pose_age_threshold_s` and zero cmd_vel. Resume must trigger automatically when odom returns. Pass: zero cmd_vel within 1.5× threshold; resume within 1 tick of fresh pose.

- **B2. Mid-mission costmap change.** Operator drives toward a goal across an open area; mid-mission, operator (or a second person) places an obstacle on the path. Mission must replan around it before the safety arc trips, OR the safety arc trips and a replan within 1 s routes around. Pass: no contact, mission completes or fails cleanly; if it fails, recovery must have been attempted.

- **B3. Replan trigger frequency.** With a quasi-static map, log path identity every tick for 60 s. Pass: path object changes only when the underlying costmap changes (not every tick); count under ~5 path changes for a static scene.

- **B4. Motor stall.** Operator wedges a chassis wheel against a baseboard while the mission is FOLLOWING. We don't have stall detection in v1, so this validates the *failure mode* — operator must use ALL-STOP. Future test pass: stall detected; mission transitions to FAILED("motor_stall"). Today pass: ALL-STOP zeros cmd_vel cleanly; no driver-level wedging or escape.

- **B5. Rug tilt.** Drive across a rug or threshold known to cause depth-pitch confusion. Pass: phantom obstacles (if any) clear within 5 s of stable surface re-acquisition; mission either continues unaffected or pauses and recovers — must not commit to a phantom-induced detour that drives into a real wall.

- **B6. Low-light.** Drive same path in normal light, then dimmed (single 40W bulb in a 4×4 m room). Pass: scan-match (when promoted) keeps lock or fails gracefully; depth coverage quantified via `depth valid_frac` in status strip; RGB capture still produces a usable frame for the streaming feed.

- **B7. Narrow doorway.** Drive through a doorway with `doorway_width - footprint_diameter` margin between 0.10 m and 0.25 m (tight but feasible). Pass: completes or fails cleanly. Failure should be PAUSED("no_path") with goal-in-lethal-halo or boxed-in classification, not a wall-strike.

- **B8. Phantom obstacles — glass / dark / low contrast.** Drive a path that passes near (a) a glass door, (b) a dark recess, (c) a low-contrast skirting board. Pass: planner does not commit the robot to drive *through* any of these as if free; if one becomes the only "free" route, mission pauses with goal-in-unknown.

- **B9. Multi-waypoint long path.** *(Phase 4 — runs once waypoints land.)* Place 4+ waypoints across two rooms; verify ARRIVED-at-N → continue, replan inheritance per segment, recovery per segment.

- **B10. Explore — empty / cluttered / multi-room.** *(Phase 5 — runs once explore lands.)* Three subscenarios; verify frontier exhaustion in (a), graceful narrow-frontier handling in (b), correct doorway traversal between rooms in (c).

### C. Streaming RGB UX

- **C1. Toolbar toggle.** Streaming on/off via toolbar; default off; persists across the session. Pass: toggling on shows ≥ 1 Hz updates; toggling off freezes the last frame; on-demand still works in either state.

- **C2. Bandwidth + Pi load.** With streaming on at the chosen rate (default 2 Hz), measure (a) Pi CPU on the oakd loop, (b) Zenoh throughput on `body/oakd/rgb`. Pass: Pi CPU stays under whatever headroom we agree with the Pi side (default budget: + 5 percentage points over baseline); throughput under 200 KB/s at 320×240 JPEG q=85.

- **C3. Out-of-sight piloting.** Drive the robot manually out of the operator's line of sight using only the streaming feed (no autonomous mission). Pass: operator can return the robot to the start without touching it.

## Pass-fail recording

For real-world tests (B*, C*), record per run:

- Date + git commit short SHA + `v_max` + `vote_capacity` + key tunables
- Outcome (pass / fail / abort)
- One-sentence note (especially for fails)
- Snapshot saved at the moment of the most interesting tick (failures, near-misses)

Snapshots already write to `~/Body/sessions/<sid>/snap_<ts>/`; reuse that path.

## Test fixtures we don't have yet

These would speed up A-class tests but are not blockers — note them so we don't forget when one becomes a pain point:

- Synthetic Costmap factory (programmable lethal/unknown/free regions). A small builder class in a test helper module; not a full sim environment.
- Synthetic PoseSource (deterministic time-driven sequence). Useful for follower + mission unit tests.
- Headless main-window tick driver. Subset of `_on_redraw_tick` that runs without Qt for pure logic tests.

When the third one starts to feel needed, lift the tick into a non-Qt method and have Qt call it; resist the urge to mock Qt.
