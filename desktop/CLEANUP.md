# Desktop cleanup — executed 2026-06-01

Reachability analysis from the three current nav-stack apps
(`desktop.nav`, `desktop.map_editor`, `desktop.pi_drive`) plus a dangling-import
gate. **29 files removed.** Decision: retire the `world_map` fuser app and the
`chassis` standalone launcher.

## Removed (29 files)

**world_map fuser app + fuser-only internals:**
`world_map/__main__.py`, `ui_qt.py`, `controller.py`, `config.py`,
`imu_scan_pose.py`, `particle_filter_pose_source.py`, `shadow_pf_driver.py`,
`snapshot.py`, `apriltag_calibration.py`, `apriltag_detector.py`,
`apriltag_observer.py`, and the whole `world_map/vpr/` dir
(anchor/bank/calibration_sweep/extractor/shadow_driver + `__init__`).

**chassis standalone launcher:** `chassis/__main__.py` (only the launcher —
`chassis/ui_qt.py` + the config/controller/state/sweep_mission library STAY;
nav reuses them).

**misc dead:** `nav/slam/shadow_driver.py`.

**tests of removed modules:** `world_map/test_apriltag.py`,
`test_imu_scan_pose.py`, `test_particle_filter_pose_source.py`,
`test_shadow_pf_driver.py`, `test_pose_weight_scale.py`, and `world_map/vpr/test_*`.

## IMPORTANT correction (why the first pass was wrong)

The initial Tier A/B list wrongly flagged **`vision_service.py`** and
**`utils/json_utils.py`** as dead. They are **LIVE** and were KEPT: `nav`
reuses chassis GUI widgets (`nav/camera_panels.py` + `nav/teleop_panels.py` →
`chassis/ui_qt.py`), which lazily `import vision_service` / `from utils.json_utils
import …` by bare name (resolved because the apps put `desktop/` on `sys.path`).
The first tracer only modeled `Body/` on the path, not `Body/desktop/`, so it
missed bare-name imports. Lesson for future reachability passes: model **both**
`Body/` and `Body/desktop/` as import roots.

## Still-present non-nav-app entry points (intentional)

- `desktop.mapping` — the **map builder** GUI (produces `reference_map.npz` the
  editor edits). Not retired. `mapping/__main__.py` + `mapping/ui_qt.py` are
  only reachable via mapping itself, but the package is a required pipeline tool.

## Verification

- byte-compile of the full remaining tree: clean
- no kept file imports any removed module (static dangling-import gate)
- no dynamic/`importlib` refs to removed modules
- live standalone tests pass (`nav.test_patrol_expand`,
  `world_map.test_particle_filter_pose`, `nav.test_pose_health`)
- smoke-import of every live module on the delete boundary: clean
