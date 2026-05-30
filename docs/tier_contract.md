# Hierarchical nav: the Tier 1 / 2 / 3 contract

Three tiers turn a topological route into safe motion. **Core principle: only
a coarse *direction* crosses from the global/world map into the metric drive
loop; every point the robot actually drives toward is observed live in the
lidar scan.**

```
Tier 1 (topological)      Tier 2 (visibility)            Tier 3 (reactive, on Pi)
ordered waypoints   ──►   bearing → furthest live    ──►  pure-pursuit + local
(world frame)             free point (body frame)         plan + swept-footprint
   ▲                          │  body/drive/goto             veto, owns cmd_vel
   └── advance on  ◄──────────┴───────────────────────◄──  body/drive/status
       PF-pose arrival
```

Production wiring lives in `desktop/nav/hierarchical_drive.py`
(`HierarchicalDrive`). The Tier-2 step itself is the pure
`plan_tier2()` in `body/lib/tier2_subgoal.py`, shared with the Tier-2 debug
console (`desktop/pi_drive --tier2`).

## Frames

- **world** — the PF/map frame. Waypoints are stored here.
- **odom** — the Pi's drifting wheel/IMU-integrated frame. `body/drive/goto`
  goals are in odom so they stay fixed as the robot moves.
- **body** — robot-centric: `+x` forward, `+y` left, origin at the robot. The
  lidar scan grid (`body.lib.scan_raster`) and Tier-2's reasoning live here.

## Tier 1 → Tier 2  (in-process)

Tier 1 hands Tier 2 the **next destination** (a world-frame waypoint) plus
arrival semantics (`waypoint_tol_m`). Tier 1 owns ordering/advance
(`PatrolRunner`); **arrival at a waypoint is judged by the PF world pose**, not
by Tier 3's status.

Tier 2's *only* world-frame dependency is converting that destination to a
**body bearing + distance**:

```
bearing = bearing_to_waypoint(rx, ry, r_yaw, wx, wy)   # = wrap(atan2(wy-ry, wx-rx) - r_yaw)
dist    = hypot(wx-rx, wy-ry)
```

This is the **PF-yaw seam**: a wrong robot heading estimate (`r_yaw`) sends the
bearing — and therefore the robot — off by that error, even when the waypoint
is in the clear. (The Tier-2 debug console deliberately bypasses this seam by
taking a *body-frame* target directly, so Tier-2 can be debugged without PF.)

## Tier 2 → Tier 3  (`body/drive/goto`, `schemas.drive_goto`)

Tier 2 ray-marches the live body-frame scan grid along `bearing`, **capped at
`dist`** (never aim past the waypoint), and picks the furthest free point
(`furthest_free_point` / `plan_tier2` → `Tier2Decision`):

- clear all the way to the target → the sub-goal **is** the target (no backoff),
- blocked/unknown/horizon first → back off `Tier2Config.backoff_m` from it.

The sub-goal stays **body-frame** until `DriveClient.send_goto_from_body` rotates
it to **odom** using the live odom pose — so the world↔odom yaw difference
cancels and Tier 2 never re-touches world/odom math.

Goto fields: `cmd_id` (**monotonic; higher supersedes; the Pi rejects a lower
id as stale** — `DriveClient` seeds `cmd_id` from wall-clock so it survives
desktop restarts), `frame="odom"`, `x_m`, `y_m`, `arrival_tol_m`, `v_max`,
`kind` (goto|cancel|stop). Tier 3 requires the desktop **heartbeat** even though
the desktop keeps `live_command` OFF (Tier 3 owns `body/cmd_vel`).

## Tier 3 → Tier 2  (`body/drive/status`, `schemas.drive_status`)

Tier 3 (`body/local_drive.py`) runs pure-pursuit + a local planner (centering,
governor, fan, gap-seek) and a **swept-footprint safety veto** over its *own*
rasterized scan. It reports every tick:

- `state` — IDLE | DRIVING | ARRIVED | BLOCKED | CANCELED | FAULT. (ARRIVED is
  published for a *single* tick, then it drops the goal → IDLE; consumers must
  treat IDLE-for-our-cmd_id as "sub-goal done".)
- `mode` — pursue | center | nudge | seek | rotate | blocked.
- `blocked_reason` — swept_block | no_progress | odom_stale | no_scan.
- `goal_body_xy`, `dist_remaining_m`, `v_mps`, `omega_radps`, and the
  **serviced `cmd_id`** (compare against the last sent id to detect a stale-id
  collision).

Tier 2 re-picks toward the same waypoint on ARRIVED/IDLE; on BLOCKED it retries
a few times then pauses (consecutive-block counter). See
`docs/drive_tier3_spec.md` for the full Tier-3 spec and arbitration rules.

## Shared inputs

`body/odom` (pose), `body/lidar/scan` (the obstacle substrate — **not** the
fused `local_map`, which lags while moving), and the PF world pose
(desktop-internal, production only).
