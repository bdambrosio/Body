# Body

Differential-drive robot software in two halves that share one repo:

- **`body/`** — Pi-side runtime. Independent Python processes on a Raspberry Pi (target) communicate over [Zenoh](https://zenoh.io/) using JSON messages.
- **`desktop/`** — Operator-side stack (laptop / workstation). `chassis` is a low-level monitoring + manual command UI; `mapping` builds a reference map, `localization` + `nav` drive against it (hierarchical Tier-1/2/3), and `map_editor` cleans the map. (`world_map` is the legacy online fuser, kept for comparison.) All connect to the Pi over the same Zenoh router.

The contract between the two halves — and with any external agent (Jill / Cognitive Workbench) — is defined in [docs/body_project_spec.md](docs/body_project_spec.md).

## Requirements

- Python 3.11+
- `eclipse-zenoh` and, for `oakd_driver`, `depthai` (see [requirements.txt](requirements.txt)); Linux udev rules for Movidius (`03e7`) are required to open the OAK from a non-root user.
- **OAK-D-Lite IMU:** retail units usually include a BNO IMU; DepthAI may require `oakd.imu_enable_firmware_update: true` (default in [config.json](config.json)) on first use. Some **Kickstarter OAK-D-Lite** boards have **no IMU** ([Luxonis docs](https://docs.luxonis.com/software-v3/depthai/depthai-components/nodes/imu/)) — set **`imu_hardware_present`: false** to run `oakd_driver` with synthetic `body/oakd/imu` so the launcher does not crash.
- A Zenoh **router** (`zenohd`) reachable by every Body process and every client (`desktop.chassis`, `desktop.world_map`, or Jill). On the robot, run the router on the Pi and listen on TCP **7447** (see [Configuration](#configuration)).
- Desktop side (`desktop/`) needs `PyQt6` + `requests`; see [desktop/requirements.txt](desktop/requirements.txt). Install only on machines that will run the operator UI — no need on the Pi.

## Install (once per machine)

Use the **repository root** (the directory that contains `config.json` and the `body/` package—not the inner `body/` folder alone):

```bash
cd /path/to/Body
python3 -m venv .venv --system-site-packages
.venv/bin/pip install -r requirements.txt
export PYTHONPATH="$(pwd)"
```

**Raspberry Pi + `motor.gpio_enabled`:** install **`python3-lgpio`** with apt (`sudo apt install python3-lgpio`). That package lives under the system interpreter’s `dist-packages`. A venv created **without** `--system-site-packages` cannot import `lgpio`, and `motor_controller` will fail at startup. Use **`--system-site-packages`** as above, or edit `.venv/pyvenv.cfg` and set `include-system-site-packages = true`, then retry.

**Raspberry Pi 5 PWM sysfs (non-root):** `motor_controller` uses RP1 hardware PWM via `/sys/class/pwm/...` (see [docs/motor_controller_spec.md](docs/motor_controller_spec.md) §4.6). Those attribute files are `root:root 0644` by default, so the launcher fails with `PermissionError` on `pwmchipN/pwmK/period` when run as a regular user. Install the shipped udev rule so members of `gpio` can write them:

```bash
sudo cp deploy/99-pwm.rules /etc/udev/rules.d/99-pwm.rules
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=pwm
```

No reboot required. Verify with `ls -l /sys/class/pwm/pwmchip0/` — group should be `gpio`, mode `g+rw`. Ensure the launch user is in `gpio` (`groups`; add with `sudo usermod -aG gpio $USER` and re-login if not).

Use the same `PYTHONPATH` for `launcher` and any `python -m body.*` command. The launcher also sets `PYTHONPATH` for child processes.

**Desktop install (laptop / workstation only):**

Use a separate venv from the Pi-side `.venv`. Pi and desktop have non-overlapping needs (depthai/lgpio are Pi-only; PyQt6 is desktop-only), and the Pi venv typically uses `--system-site-packages` for lgpio while the desktop one does not.

```bash
cd /path/to/Body
python3 -m venv desktop/.venv
desktop/.venv/bin/pip install -r desktop/requirements.txt
export PYTHONPATH="$(pwd)"
```

`PYTHONPATH` must point at the **repo root** (not `desktop/`) so both `body.*` and `desktop.*` packages import. No `--system-site-packages` needed on the desktop side.

## Configuration

| Item | Purpose |
|------|---------|
| [config.json](config.json) | Zenoh `connect_endpoints`, motor/lidar/oakd/watchdog tuning. |
| `ZENOH_CONNECT` | Optional override: single endpoint, e.g. `tcp/192.168.1.50:7447`. Replaces `zenoh.connect_endpoints` for all processes. |

Router on the Pi (matches the spec): listen on `0.0.0.0:7447` so peers on the LAN can connect. Example `zenohd` config fragment:

```json
{
  "mode": "router",
  "listen": { "endpoints": ["tcp/0.0.0.0:7447"] }
}
```

Processes on the Pi should connect to **`tcp/127.0.0.1:7447`** (default in `config.json`). A laptop running `desktop.chassis` or `desktop.world_map` uses **`tcp/<pi-ip>:7447`** via `ZENOH_CONNECT` or edited `connect_endpoints`.

### Starting `zenohd` (router)

Body expects a **router** already running before you start `body.launcher` or any desktop client.

**`zenohd` is not installed by `pip` or your `.venv`.** The Python package `eclipse-zenoh` is only the client library. If the shell says `zenohd: command not found`, install the router binary below (or add it to your `PATH`).

1. **Install the router binary** on the machine that runs the router (usually the Pi). Pick one:
   - Official options: [Zenoh installation](https://zenoh.io/docs/getting-started/installation/).
   - **Raspberry Pi 5 (64-bit):** use the **aarch64 Linux standalone** archive from [eclipse-zenoh/zenoh releases](https://github.com/eclipse-zenoh/zenoh/releases). Unpack so `zenohd` and the bundled `*.so` plugins stay in the **same directory** (the archive layout is flat). Example (adjust `ZV` to match your `eclipse-zenoh` major.minor, e.g. `1.9.0`):

```bash
ZV=1.9.0
curl -sLO "https://github.com/eclipse-zenoh/zenoh/releases/download/${ZV}/zenoh-${ZV}-aarch64-unknown-linux-gnu-standalone.zip"
mkdir -p "$HOME/zenoh/${ZV}"
unzip -o "zenoh-${ZV}-aarch64-unknown-linux-gnu-standalone.zip" -d "$HOME/zenoh/${ZV}"
```

2. **Config:** This repo includes [deploy/zenohd-router.json](deploy/zenohd-router.json) — listens on **TCP `0.0.0.0:7447`**.

3. **Run** from the directory that contains `zenohd` (foreground; use `tmux` / `systemd` for production). Example if Body lives at `~/Body`:

```bash
"$HOME/zenoh/1.9.0/zenohd" -c "$HOME/Body/deploy/zenohd-router.json"
```

To put `zenohd` on your `PATH`, copy **both** `zenohd` and the `libzenoh_plugin_*.so` files into the same target directory (e.g. `$HOME/zenoh/1.9.0` already does), then:

```bash
export PATH="$HOME/zenoh/1.9.0:$PATH"
zenohd -c "$HOME/Body/deploy/zenohd-router.json"
```

If your `zenohd` build only accepts JSON5 configs, copy `zenohd-router.json` to `zenohd-router.json5` and pass that path.

4. **Check:** With `zenohd` running, start `body.launcher` on the Pi; processes should connect to `tcp/127.0.0.1:7447` per [config.json](config.json).

## Operation overview

```mermaid
flowchart LR
  subgraph pi [Pi]
    R[zenohd]
    L[body.launcher]
    R --- L
  end
  subgraph clients [Desktop clients optional]
    C[desktop.chassis]
    W[desktop.world_map]
    J[Jill bridge]
  end
  C --> R
  W --> R
  J --> R
```

1. Start **`zenohd`** on the Pi (or your dev box for all-local tests).
2. Start **`body.launcher`** on the Pi (motor, lidar, oakd, watchdog processes).
3. Optionally start **`desktop.chassis`** (or a Jill-side bridge) on a laptop so **`body/heartbeat`** and **`body/cmd_vel`** are published. Without heartbeats, the watchdog will treat the robot as not under command and can trigger **`body/emergency_stop`**.
4. Optionally start **`desktop.world_map`** on the same or another laptop to fuse `body/map/local_2p5d` + `body/odom` into a continuous world map (see [docs/world_map_spec.md](docs/world_map_spec.md)). Consumer-only; safe to run alongside `chassis`.

## Running the stack (`body.launcher`)

On the **Pi** (after `zenohd` is up):

```bash
cd /path/to/Body
export PYTHONPATH="$(pwd)"
.venv/bin/python -m body.launcher
```

Startup order: `watchdog` → `motor_controller` → `lidar_driver` → `oakd_driver`. Logs are prefixed by process name.

**Stop:** `Ctrl+C` or `SIGTERM` to the launcher; it sends `SIGTERM` to children, waits, then `SIGKILL` if needed.

**Restarts:** If a child exits unexpectedly, the launcher restarts it with exponential backoff (capped at 30 s).

**Deploy tip:** If errors reference old line numbers or missing symbols (e.g. `XLinkOut` on DepthAI v3), the Pi’s `~/Body` tree is behind your main repo—`git pull` or rsync the updated `body/` tree, then restart the launcher.

**Watchdog:** Until something publishes **`body/heartbeat`** (e.g. `desktop.chassis` with Live cmd enabled), the watchdog may emit **`body/emergency_stop`** (`heartbeat_timeout`). That is expected; start a desktop client when you want the stack to see a live operator.

### Running under systemd

For mapping runs, install the service units so SSH drops do not stop the Pi-side runtime:

```bash
sudo cp deploy/zenohd.service deploy/body-launcher.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now zenohd.service body-launcher.service
```

Check status and logs:

```bash
systemctl status zenohd.service body-launcher.service
journalctl -u body-launcher.service -f
```

The units assume Body lives at `/home/bruce/Body`, the launcher venv is `/home/bruce/Body/.venv`, and `zenohd` is installed at `/home/bruce/zenoh/1.9.0/zenohd`.

## Standalone mode (no Jill)

**Standalone** means: Body processes on the Pi, and **you** provide heartbeat and velocity commands using `desktop.chassis` — no Cognitive Workbench / agent needs to run.

### On the Pi

1. Start `zenohd`.
2. Start `body.launcher` as above.

### On a laptop / workstation (robot’s router on LAN)

```bash
cd /path/to/Body
export PYTHONPATH="$(pwd)"
export ZENOH_CONNECT=tcp/192.168.1.50:7447
desktop/.venv/bin/python -m desktop.chassis
```

(Replace the address with your Pi’s IP or hostname; flags `--router`, `--heartbeat-hz`, `--map-stale-s`, `-v` are available — see `python -m desktop.chassis --help`.)

`chassis` opens a PyQt6 window with docks for camera, lidar, local map, motor test, vision, and a sweep-360 calibration mission. Heartbeat + `cmd_vel` are published while the **Live cmd** checkbox is on; toggle off to release motion authority without quitting.

### Optional: world_map fuser

```bash
desktop/.venv/bin/python -m desktop.world_map
```

Consumer-only — does not publish heartbeat or cmd_vel. Safe to run alongside `chassis`. See [docs/world_map_spec.md](docs/world_map_spec.md).

**Motion authority:** Do **not** run `chassis` (with Live cmd on) and another publisher (e.g. Jill) both commanding `body/cmd_vel` at the same time; the motor side effectively sees interleaved commands.

## Integration expectations (Jill / other agents)

Any desktop agent that embodies this robot should:

- Publish **`body/heartbeat`** at ≥ **2 Hz** while the robot is expected to accept motion.
- Publish **`body/cmd_vel`** often enough to satisfy the message **`timeout_ms`** (default **500 ms** in the spec) while moving or holding speed.
- Subscribe to `body/odom`, `body/lidar/scan`, `body/oakd/*`, `body/status`, `body/motor_state`, etc., as needed.

After a heartbeat fault, recovery follows **§5.10** in [docs/body_project_spec.md](docs/body_project_spec.md) (heartbeat back and a new `cmd_vel` path as implemented on the Pi).

## Smoke check (optional)

With the stack running, subscribe to `body/**` with Zenoh tooling (e.g. `zenoh-python` examples) and confirm traffic: `body/odom`, `body/lidar/scan`, `body/oakd/imu`, `body/status`, and—when `chassis` or Jill is active—`body/heartbeat` and `body/cmd_vel`.

## Navigation UI (`desktop.nav`)

Map-and-localize stack: build a reference map once, then navigate with MCL against the frozen occupancy grid.

Use the **desktop venv** ([Install (desktop)](#install-once-per-machine)): `desktop/.venv/bin/python`, with `PYTHONPATH` set to the repo root.

```bash
# 1. Mapping session (teleop drive, save reference_map.npz)
desktop/.venv/bin/python -m desktop.mapping --router tcp/PI_IP:7447

# 2. Navigation (requires --map)
desktop/.venv/bin/python -m desktop.nav --router tcp/PI_IP:7447 \
  --map ~/Body/maps/<session>/map_*/reference_map.npz \
  --relocate-on-load
```

Run tests: `desktop/.venv/bin/python -m unittest discover -s desktop/reference_map -p 'test_*.py'` (same pattern for `desktop/localization`, `desktop/mapping`).

Legacy online fusion (`desktop.world_map` fuser, `--pf`, `--slam`) remains in tree for comparison but is no longer used by `desktop.nav`. See [docs/slam_map_architecture.md](docs/slam_map_architecture.md).

### Hierarchical drive (Tier 1 / 2 / 3)

Autonomous local driving uses a three-tier hierarchy so only a coarse *direction* crosses from the (noisy, topological) world map into the metric loop — every point the robot actually drives toward is observed live in the lidar scan:

- **Tier 1** (desktop) — ordered world-frame waypoints (patrols).
- **Tier 2** (desktop) — projects the next waypoint onto the live local map (a body bearing + a reachable sub-goal).
- **Tier 3** (Pi, `body.local_drive`) — the single local-routing authority: inflates the live scan into a costmap, runs local A\*, and follows the path with pure-pursuit; owns `body/cmd_vel`.

See [docs/tier_contract.md](docs/tier_contract.md) and [docs/drive_tier3_spec.md](docs/drive_tier3_spec.md); production wiring is `desktop/nav/hierarchical_drive.py`. Debug the Pi-side drive in isolation with the **pi_drive** console:

```bash
# Tier-3 only — click a body-frame goal:
desktop/.venv/bin/python -m desktop.pi_drive --router tcp/PI_IP:7447
# Tier-2 against a saved map — world target → projection → Pi A*:
desktop/.venv/bin/python -m desktop.pi_drive --tier2 \
  --load-map ~/Body/maps/<session>/map_*/reference_map.npz \
  --router tcp/PI_IP:7447
```

## Map editing (`desktop.map_editor`)

Clean a mapping-run `reference_map.npz` so MCL localizes better. Loads an existing map, lets you paint **Wall / Free / Unknown** occupancy with a disk brush — toggle **Edit** on first (left-drag paints, middle/right-drag pans) — with Undo and Save / Save As. Save regenerates the MCL `likelihood_field` + `distance_field` from the edited occupancy and backs up the original to `.bak`. The editor never fuses — the brush is the only writer.

```bash
QT_QPA_PLATFORM=xcb desktop/.venv/bin/python -m desktop.map_editor \
  --map ~/Body/maps/<session>/map_*/reference_map.npz
```

Optional **live overlay**: add `--router tcp/PI_IP:7447`, then **Connect** to a running bot for a read-only lidar overlay (MCL pose + live scan drawn over the map; never fuses). **Relocate** / **Set location** seat the pose, so you can drive to a spot, see the live scan against the map, and paint the correction. The bot can be powered on *after* the editor starts.

## Network (Pi side)

The Pi runs Body services as `body-launcher.service`. The Pi's WiFi should be on a dedicated single-AP network for stable zenoh — see [deploy/NETWORK.md](deploy/NETWORK.md) for the GL.iNet MT3000 setup that's been validated.

## Layout

- [body/](body/) — Pi-side package: `launcher`, drivers (`motor_controller`, `lidar_driver`, `oakd_driver`, `watchdog`, `imu_driver`), `local_map`, `local_drive` (Tier-3 reactive drive), `lib/` (`zenoh_helpers`, `schemas`, `diff_drive`, `host_metrics`, and the pure drive cores `astar`, `local_costmap`, `local_planner`, `scan_raster`, `drive_safety`, `tier2_subgoal`).
- [desktop/](desktop/) — operator-side packages: [`chassis`](desktop/chassis), [`mapping`](desktop/mapping) (reference map builder), [`reference_map`](desktop/reference_map) (frozen-map I/O + likelihood/distance fields), [`localization`](desktop/localization) (MCL), [`nav`](desktop/nav) (hierarchical drive UI), [`pi_drive`](desktop/pi_drive) (Tier-2/3 debug consoles), [`map_editor`](desktop/map_editor) (reference-map editor), [`world_map`](desktop/world_map) (legacy online fuser), `vision_service.py`, `utils/`.
- [docs/](docs/) — specs and design docs, including [tier_contract.md](docs/tier_contract.md) + [drive_tier3_spec.md](docs/drive_tier3_spec.md) (hierarchical drive), [bayesian_localization_redesign.md](docs/bayesian_localization_redesign.md) (Phase 0–8 plan and status log), and [noise_models.md](docs/noise_models.md) (Phase 0 motion-model calibration).
- [scripts/](scripts/) — calibration + analysis tools (`phase0_*.py`, `phase1_likelihood_field_demo.py`, `record_body_topics.py`).
- [deploy/](deploy/) — ops files (`zenohd-router.json`, `body-launcher.service`, `99-pwm.rules`, `NETWORK.md`).

## License

See [LICENSE](LICENSE).
