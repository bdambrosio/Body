# EKF + Pose-Graph SLAM — Progress Note

**Last updated:** 2026-05-22  
**Status:** First live mapping session after cutover; map quality improved; **desktop lag unresolved**

See [slam_map_architecture.md](slam_map_architecture.md) for the target architecture and operator checklist.

---

## What landed (single cutover)

The plan in `.cursor/plans/ekf_and_graph_slam_*.plan.md` was implemented on the desktop side. Pi topics unchanged (`body/odom`, `body/imu`, `body/lidar/scan`).

| Layer | Location | Notes |
|-------|----------|-------|
| Config | `config.json` → `fusion`, `slam` | Phase 0 noise defaults; scan-match / loop-closure tuning |
| Config loader | `desktop/fusion/load_slam_config.py` | Shared by mapping and nav |
| EKF | `desktop/fusion/ekf_pose_tracker.py` | IMU owns θ; encoder contributes forward `ds` only |
| Pose graph | `desktop/mapping/pose_graph_mapper.py`, `desktop/mapping/graph/pose_graph.py` | Submap scan match, loop closure, SE(2) optimize, occupancy from optimized nodes |
| Mapping wire-up | `desktop/mapping/controller.py`, `ui_qt.py` | Replaced direct `MappingPoseTracker` + live `integrate_scan` path |
| Nav wire-up | `desktop/localization/mcl_pose_source.py` | MCL predict uses EKF motion deltas |

Unit tests pass for config loader, EKF, pose-graph optimizer, and synthetic corridor mapping. The legacy `MappingPoseTracker` module and its tests were removed (2026-05-27).

---

## First live session (2026-05-22)

**Positive:** Map appears **much more stable** than the pre-cutover mapping sessions.

- Walls render as relatively sharp red boundaries without obvious doubling.
- Green free space and red obstacles stay aligned with the driven path.
- White pose arrow sits at the tip of the yellow trail (same optimized-graph frame).
- No obvious rotation “fan” artifact in the session captured so far.

**Not yet verified on robot:**

- Full hallway out-and-back loop closure (single wall on return leg).
- Save → nav with `--map` and MCL convergence on return.
- Long-session graph growth and memory/CPU profile.

---

## Open issue: desktop response time / lag

During the first live mapping drive, **desktop UI and teleop reaction time were extremely slow**. Operator experience felt CPU-bound: map updates and chassis interaction lagged noticeably behind robot motion.

**Suspected hot paths (not profiled yet):**

1. **Pose-graph work on the Zenoh scan callback thread** — per accepted scan @ `slam.match_hz` (default 2 Hz): submap rebuild, correlation scan match, optional loop-closure search (wider window), Gauss-Newton optimize every N nodes, full display occupancy rebuild from all nodes.
2. **Full map rebuild** — `_rebuild_display_map()` re-ray-casts every graph node after each optimize; cost grows with node count.
3. **Scan matcher** — vectorized but still tens of ms per search; loop closure runs a second wide search.
4. **UI redraw** — `ui_redraw_hz` (default 5 Hz) plus map panel refresh on every grid update callback.

**Investigation TODO (next session):**

- [ ] Profile mapping session: `cProfile` or `py-spy` on `desktop.mapping` during teleop; identify top frames.
- [ ] Measure scan-callback wall time vs `slam.match_hz`; log submap build / match / optimize / rebuild separately.
- [ ] Consider moving SLAM off the subscriber thread (worker queue + back-pressure).
- [ ] Incremental occupancy update instead of full rebuild after every optimize.
- [ ] Cap submap node count / graph optimize frequency if node count is large.
- [ ] Confirm UI timer and chassis heartbeat are not starved (compare monotonic lag on status strip vs Pi odom timestamps).

**Workaround until fixed:** Lower `slam.match_hz`, increase `graph_optimize_every_n_nodes`, or map in shorter segments. Not validated as sufficient.

---

## Config knobs relevant to performance

From `config.json` → `slam`:

| Key | Default | Effect |
|-----|---------|--------|
| `match_hz` | 2.0 | Scan processing rate cap |
| `graph_optimize_every_n_nodes` | 5 | How often SPA runs |
| `submap_node_count` | 20 | Submap size for local match |
| `scan_match_xy_half_m` / `scan_match_theta_half_deg` | 0.30 m / 8° | Local search volume |
| `loop_closure_search_radius_m` | 2.0 | Triggers extra wide match |

Mapping UI also uses `MappingConfig.ui_redraw_hz` (default 5 Hz) in `desktop/mapping/__main__.py`.

---

## Next steps

1. **Profile and fix desktop lag** (highest priority for operator usability).
2. Complete manual hallway checklist in [slam_map_architecture.md](slam_map_architecture.md).
3. Tune loop-closure gates if false positives appear in symmetric corridors.
4. ~~Optional: remove or thin `MappingPoseTracker` once nav-on-saved-map is confirmed.~~ Done — module removed 2026-05-27.

---

## Related docs

- [slam_map_architecture.md](slam_map_architecture.md) — pipeline overview and verification checklist
- [noise_models.md](noise_models.md) — Phase 0 fusion noise priors
- Plan (do not edit): `.cursor/plans/ekf_and_graph_slam_d690f712.plan.md`
