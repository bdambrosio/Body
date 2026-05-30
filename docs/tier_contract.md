# Hierarchical nav: the Tier 1 / 2 / 3 contract

Three tiers turn a topological route into safe motion. **Core principle: only
a coarse *direction* crosses from the global/world map into the metric drive
loop; every point the robot actually drives toward is observed live in the
lidar scan.**

```
Tier 1 (topological)      Tier 2 (projection)            Tier 3 (local A*, on Pi)
ordered waypoints   ──►   clamp waypoint onto      ──►   inflate → A* → follow
(world frame)             the local map (body)           path + swept-veto, owns cmd_vel
   ▲                          │  body/drive/goto             body/drive/status (+ path)
   └── advance on  ◄──────────┴───────────────────────◄──  (state, path_body_xy)
       PF-pose arrival
```

**The local A\* (`body/lib/local_planner.py`) is the single authority for local
feasibility/routing** — it inflates the body-frame scan grid by the robot
footprint and finds a path the body can actually follow, or reports no path.
Tier-2 no longer does geometry; it just projects the waypoint onto the local
map. This makes "Tier-2 sets a waypoint Tier-3 can reach" hold by construction
(one footprint model, one planner). Production wiring lives in
`desktop/nav/hierarchical_drive.py`; the Tier-2 projection is `plan_tier2()` in
`body/lib/tier2_subgoal.py`, shared with the debug console
(`desktop/pi_drive --tier2`).

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

Tier 2 **projects** the waypoint onto the local map (`plan_tier2`): clamp `dist`
to the scan horizon along `bearing` (never aim past the waypoint), then nudge
off any not-clear cell onto the nearest clear one (Tier-3's A* does the real
footprint snap). The result stays **body-frame** until
`DriveClient.send_goto_from_body` rotates it to **odom** via the live odom pose
— so world↔odom cancels and Tier 2 never re-touches odom math. Tier-2 does *no*
routing; the Pi A* routes.

Goto fields: `cmd_id` (**monotonic; higher supersedes; the Pi rejects a lower
id as stale** — `DriveClient` seeds `cmd_id` from wall-clock so it survives
desktop restarts), `frame="odom"`, `x_m`, `y_m`, `arrival_tol_m`, `v_max`,
`kind` (goto|cancel|stop). Tier 3 requires the desktop **heartbeat** even though
the desktop keeps `live_command` OFF (Tier 3 owns `body/cmd_vel`).

## Tier 3 → Tier 2  (`body/drive/status`, `schemas.drive_status`)

Tier 3 (`body/local_drive.py`) builds a footprint-inflated, clearance-graded
costmap (`body/lib/local_costmap.py`) from the live scan, runs **A\***
(`body/lib/astar.py` via `local_planner.plan_local`) robot→goal, follows the
path with pure-pursuit (`steer_to_body_point` toward a lookahead), and keeps the
**swept-footprint veto only as a last-resort stop** (`drive_safety`, sized ≤ the
A* footprint). Re-plans every tick. It reports:

- `state` — IDLE | DRIVING | ARRIVED | BLOCKED | CANCELED | FAULT. (ARRIVED is
  published for a *single* tick, then it drops the goal → IDLE; consumers treat
  IDLE-for-our-cmd_id as "done".)
- `blocked_reason` — `no_path` | `goal_unreachable` | `start_blocked` |
  `goal_out_of_map` | `swept_block` | `no_progress` | `odom_stale` | `no_scan`.
- `mode` — `follow` | `plan`.
- `path_body_xy` — the local A* path (body frame, downsampled) for the operator
  UI to render.
- `goal_body_xy`, `dist_remaining_m`, `v_mps`, `omega_radps`, serviced `cmd_id`.

Tier 2 re-picks toward the same waypoint on ARRIVED/IDLE; on BLOCKED it retries
a few times then pauses. See `docs/drive_tier3_spec.md` for the full Tier-3
spec and arbitration rules.

## Shared inputs

`body/odom` (pose), `body/lidar/scan` (the obstacle substrate — **not** the
fused `local_map`, which lags while moving), and the PF world pose
(desktop-internal, production only).
