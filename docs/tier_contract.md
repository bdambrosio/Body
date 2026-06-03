# Hierarchical nav: the Tier 1 / 2 / 3 contract

Three tiers turn a topological route into safe motion. **Core principle: only a
coarse *direction* crosses from the global/world map into the metric drive loop.
The actual point the robot drives toward is *selected by Tier-3 from its own
reachable set*, so "the sub-goal is reachable" holds by construction.**

```
Tier 1 (topological)      Tier 2 (direction)             Tier 3 (local A*, on Pi)
ordered waypoints   ──►   project waypoint to a    ──►   inflate scan → SELECT the
(world frame)             body-frame point along         reachable frontier cell
   ▲                      the bearing (no geometry)      toward it → A* → follow,
   │                          │  body/drive/goto          owns cmd_vel + swept-veto
   └── advance on  ◄──────────┴───────────────────────◄── body/drive/status (+ path)
       PF-pose arrival
```

**Tier-3's footprint-inflated local A\* (`body/lib/local_planner.py`) is the
single authority for local feasibility, routing *and goal selection*.** Tier-2
hands a *direction* (a body-frame point toward the waypoint); Tier-3 snaps that
onto the nearest cell it can actually reach (footprint A\* on the live scan),
then routes there. This is what makes "Tier-2 sets a goal Tier-3 can reach" true
*by construction* — selection and routing use **one** footprint model on **one**
grid. Production wiring is `desktop/nav/hierarchical_drive.py`.

> History: Tier-2 used to *pre-select* the goal (furthest-clear point + ±75° fan)
> on a separate desktop scan raster. That used a **point-clear** model while
> Tier-3 routes with a **footprint** model — two models, two grids — so Tier-2
> could (and did) hand points Tier-3 couldn't reach, stalling at corners. Goal
> selection now lives in Tier-3 (see Invariant **I3**).

## Frames

- **world** — the PF/map frame. Waypoints are stored here.
- **odom** — the Pi's drifting wheel/IMU-integrated frame. `body/drive/goto`
  goals are in odom so they stay fixed as the robot moves.
- **body** — robot-centric: `+x` forward, `+y` left, origin at the robot.

## Invariants (the contract at each interface)

| # | Interface | Invariant | Enforced by | Status |
|---|-----------|-----------|-------------|--------|
| **I1** | pose seam `world_pose()` | map-frame pose, or `None` when stale/unavailable; **no metric-accuracy guarantee** | `PFPoseProvider` / `CheckpointPoseProvider`; drive treats `None` as suspend/align | holds |
| **I2** | Tier-1 → Tier-2 | next **world-frame** waypoint + `waypoint_tol_m`; **arrival judged by PF pose**, not Tier-3; Tier-1 owns ordering. A waypoint may be *topologically* right but its straight-line bearing blocked — that is legal **iff I3 absorbs it** | `PatrolRunner`, `hierarchical_drive` | holds |
| **I3** | Tier-2 → Tier-3 | **the executed sub-goal is footprint-reachable** — because Tier-3 *selects* it from its own reachable set toward the Tier-2 direction (it never has to honor an unreachable point) | `plan_local` (reachable-frontier snap), `astar_toward` | holds *by construction* |
| **I4** | Tier-3 → Tier-2 | `ARRIVED` for one tick → `IDLE` (consumers treat `IDLE@cmd_id` as done); `BLOCKED` carries a reason; status services a specific `cmd_id` | `local_drive.publish_status`, `hierarchical_drive._tick_driving` | holds |
| **I5** | Tier-3 → motor | Tier-3 owns `body/cmd_vel` **only while a goal is active**; desktop **heartbeat still required**; the **motor 500 ms cmd-timeout + watchdog e-stop are supreme** | `local_drive.publish_cmd`, `motor_controller`, `watchdog` | holds |
| **I6** | frames | waypoints in world; **goto goals in odom** (fixed as the robot moves); body→odom via the *live* odom at send time (world↔odom cancels) | `drive_client.send_goto_from_body`, `local_drive.on_goto` (rejects non-`odom`) | holds; *caveat:* a checkpoint re-anchor steps the map-frame yaw → steps the bearing (see Coherence) |
| **I7** | cmd_id | strictly increasing; higher supersedes; **Pi rejects a lower id as stale**; wall-clock seed survives desktop restart | `DriveClient`, `local_drive.on_goto` | holds |

## Tier 1 → Tier 2  (I2)

Tier 1 hands the **next destination** (world-frame waypoint) + arrival semantics.
Tier 2's only world-frame dependency is converting it to a **body bearing +
distance**:

```
bearing = bearing_to_waypoint(rx, ry, r_yaw, wx, wy) = wrap(atan2(wy-ry, wx-rx) - r_yaw)
dist    = hypot(wx-rx, wy-ry)
```

This is the **pose-yaw seam**: a wrong heading estimate (`r_yaw`) sends the
bearing off by that error. (The Tier-2 debug console bypasses the seam with a
body-frame target.)

## Tier 2 → Tier 3  (I3, `body/drive/goto`)

Tier 2 **projects only**: a body-frame point at `min(dist, horizon)` along
`bearing` — the waypoint direction, clamped to the local horizon. **No clearance
check, no fan, no routing.** `DriveClient.send_goto_from_body` rotates it to odom
via the live odom pose (so world↔odom cancels).

Goto fields: `cmd_id` (I7), `frame="odom"`, `x_m`, `y_m`, `arrival_tol_m`,
`v_max`, `kind` (goto|cancel|stop). Tier 3 requires the desktop **heartbeat**
even though the desktop keeps `live_command` OFF (Tier 3 owns `body/cmd_vel`).

**Reachability (I3) is Tier-3's job:** given the requested point, `plan_local`
runs footprint A\* (`astar_toward`); if the point is reachable it routes there,
otherwise it routes to the **reachable cell closest to the point** (rounding the
corner). It returns `no_path` **only** when genuinely boxed in (no reachable cell
makes progress) — a real dead-end for Tier-1 / the operator, not a corner.

## Tier 3 → Tier 2  (I4, `body/drive/status`)

Tier 3 (`body/local_drive.py`) builds a footprint-inflated, clearance-graded
costmap (`body/lib/local_costmap.py`) from the live scan, **selects + routes**
with A\* (`local_planner.plan_local` → `astar_toward`), follows the path with
pure-pursuit (`steer_to_body_point`), and keeps the **swept-footprint veto only
as a last-resort stop** (`drive_safety`, effective radius == the A\* footprint).
Re-plans every tick. Reports:

- `state` — IDLE | DRIVING | ARRIVED | BLOCKED | CANCELED | FAULT. (ARRIVED is
  published for a *single* tick, then drops the goal → IDLE.)
- `blocked_reason` — `no_path` | `boxed_in` | `swept_block` | `no_progress` |
  `odom_stale` | `no_scan`.
- `mode` — `follow` | `plan`.
- `path_body_xy`, `goal_body_xy`, `dist_remaining_m`, `v_mps`, `omega_radps`,
  serviced `cmd_id`.

Tier 2 re-picks toward the same waypoint on ARRIVED/IDLE; on BLOCKED it retries a
few times then pauses. See `docs/drive_tier3_spec.md` for the full Tier-3 spec.

## Coherence notes

- **I3 is now true by construction** (one footprint model, one grid, one
  authority). Before the goal-selection move it was *asserted but false* — the
  source of the corner stall.
- **The pose-yaw / checkpoint-anchor seam (I6 caveat):** the bearing is computed
  from the map-frame `world_pose` yaw; a checkpoint re-anchor steps that yaw
  between ticks, stepping the bearing. Per-tick consistent, but the step
  propagates to the sub-goal — watch for re-anchor-induced bearing jitter; smooth
  the re-anchor if it manifests as drive chatter.
- **Two rasterizations of one scan:** only Tier-3 rasterizes the scan now
  (`body/local_drive.py`); the desktop no longer needs the scan for Tier-2. This
  removes the desktop↔Pi grid mismatch entirely.

## Shared inputs

`body/odom` (pose), `body/lidar/scan` (the obstacle substrate — **not** the fused
`local_map`, which lags while moving), and the PF/checkpoint world pose
(desktop-internal, production only).
