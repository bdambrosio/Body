"""Nav / hierarchical-drive test plan

Production path: `HierarchicalDrive` (desktop) → `body/drive/goto` →
`body.local_drive` (Pi Tier-3). The old reactive `Mission` /
FOLLOWING/PAUSED stack was removed (see `desktop/CLEANUP.md`).

## A. Unit / dry-run (no hardware)

Run from the Body repo root with `PYTHONPATH=.`:

```bash
python -m unittest desktop.nav.test_hierarchical_drive \
  desktop.nav.test_patrol_expand desktop.nav.test_planner_clearance \
  desktop.nav.test_pose_health
python -m unittest body.test_tier2_subgoal body.test_local_planner \
  body.test_astar body.test_drive_config body.test_local_drive_core
```

Covered today (non-exhaustive):

- Hier state machine: ALIGNING → SELECT → DRIVING → ADVANCE / ARRIVED
- Sub-goal IDLE/ARRIVED re-pick; BLOCKED retry window + operator resume
- SUSPENDED on pose loss (no auto-resume); CANCELED → FAILED
- Clear-run vs blind (no scan); clear-run fail → BLOCKED (not blind)
- Lead-in then patrol; passed-vertex advance; send_failed cancels
- Patrol expand + lead-in A*; Tier-2 `plan_tier2` == clear-run

## B. Hardware-in-loop (operator within reach of ALL-STOP)

- **B1. Pose loss.** Cover lidar / drop link mid-drive → SUSPENDED, motors
  stop (Tier-3 cancel). Pose return must **not** auto-resume; Resume then Go
  path works.
- **B2. Mid-leg obstacle.** Person steps into path → Tier-3 BLOCKED /
  swept_block; hier retries then pauses for Resume.
- **B3. Cancel / Stop.** Stop while driving → one-tick `CANCELED` on Pi,
  hier IDLE; robot does not re-pick as if ARRIVED.
- **B4. Relocate while driving.** Re-localize / Set location stops hier and
  cancels goto; authored patrol shifts; press Go to re-expand and continue.
- **B5. Narrow doorway / corner.** Clear-run + Tier-3 footprint A* round
  corners without stalling on unreachable pre-selected goals.
- **B6. Multi-waypoint loop.** 4+ markers, one lap; terminal uses passed-test
  (does not stop a leg short).

## C. Streaming RGB / teleop

Unchanged operator checks: streaming toggle, bandwidth, out-of-sight teleop.
Not part of the hierarchical drive loop (Tier-3 owns cmd_vel while goto active;
keep live_command OFF during Go).

## Recording

For B* runs: date, git SHA, key tunables (`v_max`, horizon), pass/fail,
one-line note, snapshot under `~/Body/sessions/<sid>/snap_<ts>/`.
"""
