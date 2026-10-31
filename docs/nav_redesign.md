# Navigation redesign — one map, one planner, soft via-points

*2026-10-31. Agreed direction after a day of patch-on-patch stalls.*

## Goal

Drive a known route through a lived-in house indefinitely, without a human,
while furniture moves, people walk through and the floor glares. Measured by
**route progress per minute** and **interventions per hour**, not by whether
any individual sub-goal was "arrived".

## Why the Tier-1/2/3 shape could not meet it

The waypoint that sat 20 cm inside a cabinet's inflation is the whole problem
in miniature: the fix ("move it 30 cm") is obvious to anyone who can see the
route *and* the obstacle at once, and **no component could see both**.
Tier-1 owned the route with no map; Tier-2 saw one scan but only knew a
bearing; Tier-3 saw obstacles but only knew one body-frame point. The only
move left to any of them was to refuse: BLOCKED → retry → pause. Every
special case added on 10-31 (fan-out, detour re-pick, no-motion arrival,
carrot skip) was one tier guessing at what another tier knew.

Structural consequences:

1. Three world models, three definitions of "blocked" (Tier-2 march grid,
   Tier-3 costmap, depth memory) — a goal "clear" to one is "snapped" to the
   next; half the stalls were boundary disagreements.
2. Hard targets: waypoints and sub-goals were exact points; `ARRIVED` was a
   stop. A vertex inside an inflation is unreachable by definition.
3. Recovery = repeat, then wait for a human. No route-level progress metric.
4. Persistent obstacle knowledge lived in Tier-3 (30 s, odom frame, wiped on
   restart) while the desktop's world grid was used only for display and
   localization.

## Target shape

**One map, one planner, one frame.** The desktop's world grid (lidar +
2.5D local map + traversal history) is the planning map; depth-memory cells
and VLM semantics land on it as evidence with per-source confidence and
decay. Everything is in the world frame via the PF pose; the Pi gets
geometry in odom through the existing anchoring.

**The route is a corridor, not a polyline.** Authored markers are
*via-points* with a radius. The planner's goal is "a free cell within the
via-point's radius"; a via-point whose disk is blocked is relaxed to the
nearest free cell or skipped with a note. The terminal has a tighter radius
and is never skipped.

**Tier-3 follows a path; it does not choose destinations.** The desktop
plans globally and hands Tier-3 the next stretch of path. Tier-3 keeps pure
pursuit, the swept veto, the depth veto and a seconds-long safety memory.
"Blocked" from Tier-3 means one thing — "this path is no longer drivable,
here is where" — and the only response is: put the obstacle on the map,
replan.

**Recovery is planning, not procedures.** Blocked → map update → replan.
Boxed in → plan from the nearest free cell / back up along the last path.
Stall = no route-arc-length progress for T seconds, with a fixed escalation:
widen the corridor → relax the via → skip it → back up → human. E-stop and
pose-lost stay the only true special cases.

## Steps

| # | Step | Pi change | Status |
|---|------|-----------|--------|
| 1 | Global planner on the live world costmap each leg; via-points relaxed/skipped; Tier-3 gets the furthest live-visible point along the path (`desktop/nav/route_planner.py`). Depth-veto hits and Tier-3's depth memory are stamped into the plan. The one-shot `expand_patrol` at Go is retired. | none | **done 10-31** |
| 1b | Planning map = reference-map prior + session grid, unknown routable at a cost; frontier fallback when a via is sealed off; path vs goal masks; footprint 0.11; `plan-fail-*.npz` dumps. | none | **done 10-31** |
| 2 | Via-point `radius_m` / `must_visit` in the patrol format (`Waypoint`), honoured by the planner (relax within radius, skip, or block for must-visit). Editor UI for them: pending (edit the JSON for now). | none | **format done 10-31**; editor pending |
| 3 | Path streaming to Tier-3 (`goto` kind `path`: a polyline, refreshed at 2–5 Hz) — no per-sub-goal stop-and-go. | yes | |
| 4 | Depth evidence becomes a world-map source (confidence + decay) once the veto's false-positive rate is measured low (blob gates + path corridor shipped 10-31, observe-only; `scripts/depth_denoise_score.py`). The Pi depth memory is retired. | yes | gates done; map source pending |
| 5 | Route-progress watchdog + escalation ladder replace the retry window and the operator pause. First rung shipped 10-31: an intermediate via that stays blocked for the window is skipped before any pause; terminal / must-visit still pause. Remaining: progress-based stall timer, back-up primitive, retire the 10 s window. | none | partial |

Each step is testable in the existing sims and leaves the robot drivable.
Result so far (15:05–15:13 on step 1 + observe-only depth): three clean
laps, one resume per lap for the fixed `leg_timeout`.

## Depth veto (observe-only) — what the logged frames showed

28 would-block frames in one session (none a contact): the raw rule (≥ 8
pixels in the commanded-arc corridor) fires on 17 offline; the blob gates
(connected floor blob ≥ 6 cm longest dimension, ≥ 6 cm high somewhere, ≥
0.25 m away) remove every small one and keep 9 that are 13–36 cm wide and
20–24 cm tall — real objects *beside* the path that the arc projected
0.8 m ahead swept through while turning. Hence the second fix: the veto
scores along Tier-3's planned path with the footprint half-width, max
range 0.45 m. `act` stays off until the next sessions' would-block rate is
measured. A persistence gate (same odom cells across ≥ 3 frames with ≥
10 cm of motion) is the remaining idea, for the glare class.

## Step 1 contract (as built)

* `HierarchicalDrive(..., costmap_provider=lambda: live Costmap)`. Every
  SELECT: `plan_route(costmap, pose, carrot)` → A* (`desktop/nav/planner.py`)
  from the robot to the carrot **relaxed** within `RouteConfig.via_radius_m`
  (0.5 m) to the nearest free cell; `via_unreachable` → intermediate carrot
  skipped (`ADVANCE`), terminal → `BLOCKED terminal_unreachable`; `no_route`
  → `BLOCKED no_route` (retries, the map may fill in).
* Obstacles Tier-3 knows that the map may not: `depth_mem_body` (every
  status) and the `depth_hit_body` of a depth_block in the last 20 s are
  stamped lethal for the plan.
* Sub-goal = `visible_subgoal_along_path`: the furthest point along the path
  (≤ `horizon_m` 1.5 m) that the live scan raster, footprint-inflated with
  the depth memory stamped, shows clear; backed off 4 cm. Below
  `min_subgoal_m` (0.2 m) → `BLOCKED path_blocked_live` (retry), or the
  carrot is passed if it is within `blocked_carrot_skip_m`.
* Reaching the relaxed point counts as reaching the via.
* No costmap → the previous straight-bearing Tier-2 march (+ fan) remains
  as the fallback, unchanged.
* The map preview shows the live global path (`current_route_world()`).

## What survives from the 10-31 work

All sensing: the floor calibration (`scripts/depth_floor_calib.py`), the
depth veto, the memory's confirm/clear logic (to be relocated to the map),
the A* endpoint detour gate, the swept check, the drive client. The
orchestration special cases (fan-out, detour re-pick, carrot skip) stay as
fallbacks until steps 3–5 make them unreachable, then go.
