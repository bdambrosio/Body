# Tier-3 reactive drive (Pi-side) — interface spec

Tier-3 is the lowest tier of the hierarchical navigator (see
`docs/tier_contract.md`): given **one subgoal**, drive there on the live
**rasterized lidar scan** (not the fused `local_map`, which lags while
moving), avoiding what is seen, with no dependence on the global map or
particle-filter pose. It runs **on the Pi** (`body.local_drive`) so the
reactive loop sits next to the freshest scan with no network round-trip.

Production sender is the desktop hierarchical drive
(`desktop/nav/hierarchical_drive.py`); the `desktop.pi_drive` debug consoles
emit the identical `body/drive/goto` command.

## Topics

### `body/drive/goto` (sender → Pi) — `schemas.drive_goto`
| field | type | meaning |
|---|---|---|
| `ts` | float | send time (wall clock) |
| `cmd_id` | int | monotonic; a higher id supersedes a lower one. A **cancel advances the watermark** to its own id, so a delayed duplicate goto cannot re-arm a revoked goal |
| `frame` | str | `"odom"` (only value in v1; others rejected) |
| `x_m`, `y_m` | float | goal point in the odom frame |
| `final_heading_rad` | float? | face this on arrival; omit = don't care |
| `arrival_tol_m` | float? | per-command override of config |
| `v_max` | float? | per-command speed override |
| `kind` | str | `"goto"` \| `"cancel"` \| `"stop"` |

**Why odom frame.** A body-frame point is stale by the time the Pi acts;
a world/global point would reintroduce the dependence on the global map
we are trying to escape. The sender converts a body-frame point to odom
using the live odom pose at send time (`DriveClient.send_goto_from_body`),
so a constant world↔odom offset cancels. The Pi then tracks the fixed odom
point as the robot moves, using its own odom — only a *coarse direction*
ever crosses from any higher tier into the metric drive loop.

**IMU yaw correction.** Wheel odom is blind to externally-forced rotation
(a floor ridge kicking the chassis, wheel slip), which would rotate the
goal's body-frame bearing by exactly the missed angle. The heading used for
the goal transform is therefore wheel θ plus the IMU-vs-wheel yaw divergence
accumulated since the goal started (`ImuYawCorrector`, fed by `body/imu`).
The baseline resets per goal so long-term IMU drift never enters, and the
published `body/odom` contract is untouched. No IMU (or a stale one) →
zero correction, wheel-only behavior.

### `body/drive/status` (Pi → sender) — `schemas.drive_status`
Published every control tick.
| field | type | meaning |
|---|---|---|
| `cmd_id` | int | the goal being serviced (0 = none) |
| `state` | str | `IDLE`\|`DRIVING`\|`ARRIVED`\|`BLOCKED`\|`CANCELED`\|`FAULT` |

`ARRIVED` is published for one tick, then the goal drops → `IDLE`.
`CANCELED` is published for one tick after `cancel`/`stop` clears an **active**
goal (under the revoked goal's `cmd_id`), then `IDLE`. Senders that treat
`IDLE@cmd_id` as "sub-goal done" must handle `CANCELED` as revoke, not success.
| `goal_body_xy` | [float,float]? | active goal in the *live* body frame (for display) |
| `dist_remaining_m` | float | range to goal |
| `v_mps`, `omega_radps` | float | commanded velocity this tick |
| `blocked_reason` | str? | `no_path`\|`boxed_in`\|`swept_block`\|`depth_block`\|`no_progress`\|`deadline`\|`no_scan` (FAULT carries `odom_stale`) |
| `mode` | str? | `follow`\|`plan`\|`realign`\|`held` |
| `path_body_xy` | list? | the A\* path being followed (display/inspector) |
| `build` | str? | Pi git sha — the desktop flags a stale deploy |

## cmd_vel ownership (arbitration)

Exactly one producer of `body/cmd_vel` at a time:

- While a `goto` is active, **Tier-3 owns `body/cmd_vel`.** The desktop
  keeps `StubController.live_command` **off**, so it publishes no cmd_vel
  and cannot fight the Pi. The desktop still publishes **heartbeat**
  (always, while connected) — the watchdog e-stops without it, so heartbeat
  is required for *any* motion including Pi-initiated drives.
- On `cancel`/`stop` (or goal drop), Tier-3 **publishes a zero cmd_vel
  immediately** — the motors never coast on the last command waiting for
  the 500 ms cmd timeout.
- Manual teleop (live_command on) is mutually exclusive with an active
  goto. The Pi watchdog / e-stop / motor timeout remain supreme over both.

## Behaviour

Each control tick (10 Hz):

1. Rasterize the latest lidar scan into a body-frame int8 grid
   (`body/lib/scan_raster.py`).
2. Build the footprint-inflated, clearance-graded costmap and run **local
   A\*** to (or toward) the goal — `plan_local` → `astar_toward`
   (`body/lib/local_costmap.py`, `local_planner.py`). The A\* is the single
   local authority: if the goal cell is unreachable it routes to the
   reachable cell closest to it; `no_path` only when genuinely boxed in.
3. Pure-pursuit follow: steer to a lookahead point on the path
   (`steer_to_body_point`, rotate-in-place hysteresis for large bearings).
4. **Swept-footprint veto as last resort** (`body/lib/drive_safety.py`,
   effective radius == the A\* footprint): a new obstacle on the commanded
   arc between replans → re-aim in place toward the lookahead
   (`mode=realign`, bounded by `swept_realign_timeout_s`), else stop +
   `BLOCKED:swept_block`. The veto only stops, never steers.
5. **Depth near-field veto** (`body/lib/depth_veto.py`, config
   `local_drive.depth_veto`): while translating, if fresh `body/oakd/depth`
   shows enough obstacle-slab hits in a short forward envelope (default
   ~0.08–0.80 m, footprint half-width), stop immediately with
   `BLOCKED:depth_block` (no realign — head-on soft obstacles rarely clear
   by yaw). Fail-open when depth is missing/stale/rotating fast so a dead
   OAK does not freeze the robot; lidar still owns planning. This is **not**
   consumption of `body/map/local_2p5d` — the fused map still lags while
   moving and stays off the planner path.

Terminal conditions:

- Within `arrival_tol_m` → `ARRIVED` for one tick (after the optional
  `final_heading_rad` rotate), then the goal drops → `IDLE`.
- No goal progress while translating for `no_progress_timeout_s` →
  `BLOCKED:no_progress`.
- **Per-goal deadline** `goal_deadline_s` (default 30 s, 0 disables) →
  stop + `BLOCKED:deadline`. This catches rotate/drive dithers the
  translation-gated no-progress watchdog can't see; the goal stays active
  (so the status doesn't decay to IDLE, which senders read as success)
  until a superseding goto or cancel clears it.
- Stale odom/scan (0.5 s) → `FAULT:odom_stale` / `BLOCKED:no_scan`, motors
  zeroed.

Config lives under `config.json` → `local_drive`. The planner/raster
sections are built through `body/lib/drive_config.py` — the **same builders
the desktop uses** to model Tier-3 (contract I8 in `docs/tier_contract.md`).
