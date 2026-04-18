# Local 2.5D map (`body/map/local_2p5d`)

**Process:** `python -m body.local_map` (see `body/launcher.py` when `local_map.enabled` is true).

## Purpose

Egocentric **max height above ground** grid in the **body frame** (+x forward, +y left, +z up). Each cell stores the highest obstacle sample from:

- **`body/lidar/scan`** — horizontal slice at `lidar_z_body_m` (default from `lidar.height_above_ground_m`).
- **`body/oakd/depth`** with `format: depth_uint16_mm` — stereo points unprojected with approximate intrinsics and `depth_*` extrinsics.

This matches the navigation intent: lidar sees a **torso-height ring**; depth fills **near-ground** structure in the camera wedge. Fusion rule is **per-cell maximum** of valid samples (v1; no temporal decay).

## Configuration (`config.json` → `local_map`)

| Key | Meaning |
|-----|--------|
| `enabled` | If false, `local_map` exits immediately (launcher still starts the module; use launcher edit or systemd to omit). |
| `publish_hz` | Fusion/publish rate (default **2 Hz** in code and sample `config.json` — large JSON payloads; increase only if needed). |
| `resolution_m` | Square cell size (m). |
| `extent_*_m` | Rectangle around robot: forward/back/left/right from body origin. |
| `ground_z_body_m` | Ground plane z (usually 0). Samples at or below are ignored. |
| `lidar_x_body_m`, `lidar_y_body_m`, `lidar_yaw_rad` | Lidar origin and yaw offset (rad) for scan angles (0 = forward per spec, CCW). |
| `lidar_z_body_m` | Optional override; else `lidar.height_above_ground_m`. |
| `depth_x_body_m`, `depth_y_body_m`, `depth_z_body_m` | Camera optical center in body; z defaults to `oakd.depth_camera_height_above_ground_m`. |
| `depth_yaw_rad`, `depth_pitch_rad`, `depth_roll_rad` | Euler (Z–Y–X) after fixed OpenCV→body axis fix (see code). |
| `depth_hfov_deg`, `depth_vfov_deg` | Approximate pinhole FOV for resized depth (`oakd.depth_out_width` × `height`). |

Tune **`lidar_yaw_rad`** / **`depth_*`** to match your CAD mount; defaults assume sensors on the body origin.

## Wire message (`schemas.local_map_2p5d`)

- `frame`: `"body"`.
- `kind`: `"max_height_grid"`.
- `origin_x_m`, `origin_y_m`: grid corner (minimum x, minimum y).
- `nx`, `ny`: cell counts; index **i** along +x, **j** along +y.
- `max_height_m`: `list` length `nx`, each row length `ny`, entries `null` or height in meters above `ground_z_body_m`.
- `sources`: optional `lidar_ts`, `depth_ts` of inputs used.

## Extrinsics you still owe the stack

Horizontal **offsets** (lidar vs body origin, camera vs body), **yaw** alignment between lidar 0° and body +x, and **camera pitch/roll** if the OAK is not level. Heights alone are not enough; fill the `*_x_body_m`, `*_y_body_m`, and Euler fields from CAD or calibration.

## Related

- [body_project_spec.md](../body_project_spec.md) — `body/lidar/scan`, `body/oakd/depth`.
