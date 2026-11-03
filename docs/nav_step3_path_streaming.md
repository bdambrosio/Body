# Nav redesign step 3 — path streaming to Tier-3 (plan)

*2026-11-03. Draft for review. Parent: `docs/nav_redesign.md`.*

## What it buys (measured, honest)

The 11-03 laps on `Hallway_loop` (step 1 live) set the baseline:

| | 10:44 lap 1 |
|---|---|
| path driven / time | 28 m / 173 s → **0.162 m/s** mean (cap `v_max` 0.18) |
| time near-stopped (< 0.05 m/s, incl. turning in place) | 11 % |
| `goto`s per lap (each a new Tier-3 goal) | ~30 |
| re-picks per lap | 27 sub-goal ARRIVED, 11 bearing drift, 3 pose correction |

So **lap time is not the reason**: at most ~10 % comes back from removing
the stop at each sub-goal. The reasons are structural:

1. **Two planners disagree.** The desktop plans the route; Tier-3 then
   re-plans its own local A* to one point 1.5 m out and may take a
   different way round an obstacle. "Blocked" still means three different
   things in three places.
2. **Churn resets Tier-3.** Every re-pick is a new `cmd_id`, which resets
   Tier-3's path hysteresis, endpoint latch, progress clock and deadline.
   The 09:51 failure showed 35 drift re-picks in one leg; with no
   continuous goal nobody can measure progress along the route.
3. **The special cases can't go until this lands.** Bearing-drift re-pick,
   detour re-pick, fan-out, carrot skip and `arrived_no_motion` all exist
   to repair the point hand-off. Step 5 (a route-progress watchdog) wants
   one continuous goal to measure.

If those don't justify a Pi contract change right now, the alternative is
to do step 5 first on the current hand-off (it can measure progress along
the desktop route) and come back to this.

## Contract change (Pi): `body/drive/goto` `kind: "path"`

New fields; everything else as in `docs/drive_tier3_spec.md`.

| field | type | meaning |
|---|---|---|
| `kind` | `"path"` | new kind; `goto`/`cancel`/`stop` unchanged |
| `cmd_id` | int | one id for the **whole route leg sequence** (until terminal, block-out, or cancel) |
| `path_seq` | int | increments per refresh. Same `cmd_id` + higher `path_seq` = **update in place** (no Tier-3 reset); lower/equal = ignored (resend) |
| `frame`, `anchor_ts` | | `"body"` + the scan ts the polyline was computed in; anchored with the Pi's odom ring exactly like a body goal today |
| `path_xy` | [[x,y],…] | polyline, robot-first, ~0.10 m spacing, ≤ 40 points (≤ ~4 m; Tier-3's raster is ±2.5 m) |
| `end_is_goal` | bool | true: the end is the terminal → `ARRIVED` within `arrival_tol_m`. false: the end is just the horizon — never stop there on purpose |
| `ttl_s` | float | refresh deadline (default 1.0). See staleness |
| `v_max`, `arrival_tol_m`, `final_heading_rad` | | as today |

Status additions: `path_seq` (the one being followed), `path_progress_m`
(arc length done along the current path), `mode` adds `rejoin`, `starved`;
`blocked_reason` adds `path_blocked` (with `depth_hit_body` / a new
`block_body_xy` = where), `path_stale`. Status also carries
`caps: ["path"]` so the desktop knows the Pi supports it.

Watermark semantics: unchanged for `cmd_id`. A refresh is not a resend:
the arbiter compares `(cmd_id, path_seq)`.

## Tier-3 behaviour in path mode (`body/lib/tier3_controller.py`)

Kept as is: rasterization, the one obstacle model, pure pursuit, swept
veto, depth veto, IMU yaw correction, rotate-in-place hysteresis, odom-reset
and e-stop handling.

* **Follow the given path, not a point.** The path (odom frame, anchored at
  `anchor_ts`) replaces the A\* goal. Pure pursuit on it, lookahead as today.
* **Local avoidance with rejoin.** Each tick the costmap checks the path
  ahead within the raster. Clear → follow it. Obstructed → local A\* to a
  **rejoin point**: the first clear path point past the obstruction (within
  the raster and within `max_rejoin_deviation_m`, proposed 0.5 m, of the
  path). `mode=rejoin` while on the detour. No rejoin point →
  `BLOCKED:path_blocked` with `block_body_xy`.
* **Refresh = update, not restart.** A higher `path_seq` swaps the polyline
  and keeps progress, rotation budget, latches and the depth tracker. It
  also **clears a non-terminal BLOCKED**: the desktop answers a block with a
  replanned path, and that is the resume.
* **Staleness.** No refresh for `ttl_s` → cap speed to `v_min` and keep
  following (`mode=starved`). No refresh for 2 × `ttl_s` → stop,
  `BLOCKED:path_stale`. This covers what heartbeat doesn't: the desktop link
  alive but its planner wedged (today's CUDA failure).
* **Ends.** `end_is_goal=false`: slow toward the end, stop at it only if no
  refresh has extended it (then `starved`). `end_is_goal=true`: today's
  arrival + optional final heading.
* **Progress / bounds.** `no_progress` measured on arc length along the path
  (detours count). The 30 s per-goal deadline does not apply in path mode
  (a leg can be minutes); staleness + no_progress + the desktop's route
  watchdog (step 5) bound it. Rotation budget unchanged.

## Desktop (`desktop/nav/hierarchical_drive.py`)

* **New state `FOLLOWING_PATH`** replaces SELECT_SUBGOAL ↔ DRIVING_SUBGOAL
  when a costmap provider exists **and** the Pi reports `caps` with `path`.
  Otherwise today's sub-goal hops remain (fallback, debug consoles, teleop
  click).
* **The streamed path runs through the via-points.** Plan robot → via\_k
  (relaxed) → via\_k+1, concatenated, so the path doesn't end at every
  via; `end_is_goal` only for the terminal. Passing a via (world pose
  within its radius) advances the carrot; the next refresh just extends the
  path.
* **Two rates.** Global A\* on a worker thread at ~1 Hz, and immediately on
  a block, a pose correction or a via advance. It currently runs on the
  UI thread at each SELECT, 50–200 ms. Streaming at ~4 Hz re-expresses the
  current world path in the body frame of the newest scan: MCL pose at that
  scan's ts from the pose buffer, `anchor_ts` = that ts, so the world↔odom
  offset cancels as it does today.
* **Blocks.** `path_blocked`/`depth_block` → stamp `block_body_xy` /
  `depth_hit_body` into the planning map (the existing `_note_block_hits`),
  replan, stream the new path (same `cmd_id`, next `path_seq`). The 10 s
  retry window and the 11-01 escalation ladder stay until step 5 replaces
  the window with route progress.
* **Retired in path mode** (kept for the fallback): bearing-drift re-pick,
  sub-goal ARRIVED re-pick, detour re-pick, fan-out, carrot skip,
  `arrived_no_motion`, `leg_timeout` per sub-goal.
* Behind a flag: `HierConfig.path_mode` (default **off**), nav `--path-mode`.

## Build order (each leaves the robot drivable)

1. **Pi, path follower in the controller.** Contract fields, arbiter
   `(cmd_id, path_seq)`, follow / rejoin / starved / stale / ends, status
   fields + `caps`. Closed-loop sims in `body/test_tier3_controller.py`:
   straight and curved follow; refresh mid-path (no reset, no slowdown at
   the boundary); stale → `v_min` → stop; obstruction → rejoin; walled →
   `path_blocked` with location; refresh clears the block; terminal
   arrival + heading; odom reset and e-stop drop the path.
2. **Pi deploy**, path mode unused; laps on sub-goals as a regression check.
3. **Desktop.** Worker-thread route planning (also helps today), streamer,
   `FOLLOWING_PATH`, block → replan, `caps` fallback. Sim tests with a fake
   IO: refresh rate, no re-picks, via continuity, block/replan/resume,
   fallback when `caps` is missing, link loss → `path_stale` handling.
4. **Robot A/B** on `Hallway_loop`: 2 laps sub-goal mode, 2 laps
   `--path-mode`. Compare lap time, near-stopped fraction, Tier-3 goals and
   blocks per lap, interventions. Keep path mode only if it is at least as
   good on every one.
5. Make it the default; delete the retired special cases (separate commit).

## Risks

* **Rejoin authority.** Letting Tier-3 detour 0.5 m off the route is a
  judgement call: too strict and it blocks on things a small swerve
  avoids; too loose and it reintroduces the two-planner disagreement.
* **Frame consistency.** The path must be expressed with the MCL pose *at*
  `anchor_ts`; the pose buffer has that (the 11-03 odom backlog that broke
  it is fixed in 9f7898f). If the ts is outside the buffer, skip that
  refresh rather than send a mis-anchored path.
* **Link gaps.** At 4 Hz a 1 s WiFi gap is 4 lost refreshes → `starved`, not
  a stop. Over 2 s → `path_stale`, and the existing recovery handles the
  link.
* **Raster mismatch.** The Pi plans on 8 cm cells and the desktop on 5 cm;
  a path the desktop thinks clear can clip the Pi's inflation at a doorway.
  Rejoin absorbs small cases; align the footprints (open since 10-31) if
  doorways block.

## Decisions for the owner

1. Do step 3 now, or step 5 first? (I'd do 3: it is what lets 5 and the
   cleanup be simple.)
2. Rejoin deviation limit: 0.5 m proposed.
3. Staleness: `ttl_s` 1.0 s → `v_min`; 2.0 s → stop. OK?

## As built — Pi side (build step 1, 2026-11-03)

Owner decisions: step 3 now; stray limit 0.5 m; staleness 1 s → `v_min`,
2 s → stop, **monitored** so the limits can be re-tuned after test patrols.

Changes from the draft above, found in the closed-loop sims:

* **`route_id`, not every refresh, is the resume.** Refreshes arrive every
  ~250 ms, so "a refresh clears a block" would retry blindly 4×/s. Only a
  new `route_id` (the sender re-planned) resets progress / rotation budget
  and clears a terminal block.
* **Every refresh re-references progress without restarting its timer.**
  Otherwise a path extended at the horizon reads as "remaining went up" and
  trips `no_progress`; a robot that truly isn't moving still trips it.
* **Side latch for detours.** An obstacle centred on the path made the
  rejoin A* alternate sides every tick (turn-in-place until
  `rotate_timeout`). Keeping the old detour (point-mode-style hysteresis)
  let the robot drift inside the padded footprint → endless `swept_block`.
  Instead each tick plans fresh but with the other side of the obstructed
  stretch closed off once a side was chosen.
* **Rejoin lead 0.5 m.** Aiming the detour at the first clear path point
  forced a sharp turn back that pure pursuit cut into the obstacle
  (`swept_block` beside the post); aiming 0.5 m further merges gently.
* A refresh whose `anchor_ts` is outside the odom ring is skipped (counted)
  rather than anchored at the latest odom.
* Staleness monitoring: the Pi prints per path goal `refreshes / max_gap /
  starved / stale_blocks / skipped_refreshes` when it ends, and each
  `path_stale` as it happens; status carries `path_age_s` every tick.

Sim results (corridor 1.2 m, posts on a 4 m path): centred 8 cm post and
off-centre 8 / 15 cm posts → arrive, clearance ≥ 0.14 m (footprint 0.11),
stray ≤ 0.32 m. A 15 cm post centred (0.45 m gaps ≈ the padded footprint)
→ clean `path_blocked`; point mode ends `boxed_in` beside it.

## As built — desktop side (build step 3, 2026-11-03)

* `nav --path-mode` → `HierConfig.path_mode`. Used only while the Pi's
  status carries `caps: ["path"]`; otherwise (and with no costmap) the
  sub-goal hops run unchanged. Default off until the robot A/B.
* No new state: in path mode SELECT sends a path goal (new `cmd_id` if none
  is live, else a refresh with the next `route_id` — a via advance), and
  DRIVING streams refreshes at `path_stream_hz` 4 Hz from the latest pose
  (`world_pose_stamped`: `anchor_ts` = the Pi odom ts that pose refers to),
  quietly re-plans the global route every `path_replan_s` 1 s (and at once
  on a discrete pose correction), and skips the sub-goal re-pick, bearing
  drift, detour and pose-correction re-picks.
* Streamed path: from the robot's projection on the global route, every
  0.10 m, ≤ 4 m; `end_is_goal` only for the terminal when not truncated.
  The path ends at the current via; the drive advances the via (pass test)
  before Tier-3 reaches the end, and the next refresh carries the next leg
  (`route_id`+1, same goal) — so no concatenation through the via.
* Blocks: every Tier-3 BLOCKED/FAULT goes through the existing
  `_enter_blocked` (which cancels the goal, X3) → retry window → escalation
  ladder; a retry is a new path goal. A `path_blocked` point is stamped
  lethal for planning for `path_block_stamp_s` 20 s (lidar-seen, so not
  gated like depth hits).
* Worker-thread planning: **not done** — measured global A* on the real map
  at 1–22 ms (13.8 m leg) + costmap build ~25 ms, so a 1 Hz re-plan on the
  UI thread is cheap. `max_replan_ms` is in the stats line to check.
* Staleness monitoring (desktop): one line per via and at stop —
  `hier: path stream (...): sends goals routes max_send_gap max_path_age
  starved stale blocks replans max_replan` — plus a log line when Tier-3
  reports `starved` and a warning on `path_stale`.
