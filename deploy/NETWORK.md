# Body — Network setup (Pi side)

How the Raspberry Pi's WiFi should be configured for stable zenoh communication with the operator desktop. Validated 2026-05-16 after a session that lost minutes of operating time to mesh-roam-induced connectivity gaps.

## Topology

- **Pi** runs `body-launcher.service` and publishes the `body/**` zenoh topics.
- **Desktop** runs `desktop.nav` (or `desktop.mapping` / `desktop.map_editor` / `desktop.pi_drive`) and connects to the zenoh router. Standalone `desktop.chassis` / `desktop.world_map` apps were removed 2026-06 — see `desktop/CLEANUP.md`.
- **Bot router** is a [GL.iNet GL-MT3000 (Beryl AX)](https://www.gl-inet.com/products/gl-mt3000/) travel router carried with the bot. It exposes:
  - SSID `GL-MT3000-bee` (2.4 GHz) — **use this for the Pi**
  - SSID `GL-MT3000-bee-5G` (5 GHz) — **avoid for the Pi**; less range through walls
- Bot subnet is **192.168.8.0/24** with the router at **192.168.8.1**.
- The Pi gets a DHCP lease somewhere in 192.168.8.x; the desktop is reachable at **192.168.8.60** in our setup.

## Pi WiFi interfaces

The Pi has two WiFi interfaces:

- **`wlan0`** — built-in Broadcom (CYW43455 on Pi 4/5). The `brcmf` driver is unreliable for our use:
  - `bgscan: Failed to enable signal strength monitoring` (per-boot warning)
  - Periodic `brcmf_p2p_send_action_frame: Unknown Frame: category 0xa, action 0x8` errors
  - Aggressive roaming when surrounded by a multi-AP mesh
  - Roams to busy 5 GHz APs that occasionally fail authentication (10 s recovery)

- **`wlan1`** — USB WiFi dongle dedicated to the bot's MT3000 AP. Stable, single-AP, no roaming.

**Run the bot on `wlan1`. Disable `wlan0`.** The performance / reliability gap is large enough to make this non-negotiable.

## Setup (one time per Pi)

### Disable `wlan0` (built-in Broadcom)

```bash
# Soft-disable now and on every future reboot
sudo nmcli connection delete "BruceJane" 2>/dev/null || true
sudo nmcli device set wlan0 managed no

# Persist via NetworkManager config
sudo tee /etc/NetworkManager/conf.d/10-disable-wlan0.conf >/dev/null <<'EOF'
[device-wlan0]
match-device=interface-name:wlan0
managed=false
EOF

sudo systemctl reload NetworkManager
```

After `nmcli device status`, `wlan0` should report `unmanaged`.

### Connect `wlan1` to `GL-MT3000-bee` (2.4 GHz)

```bash
# Find the 2.4 GHz BSSID. Should show a single line near 2400-2480 MHz.
sudo nmcli device wifi rescan ifname wlan1
sleep 3
nmcli -f IN-USE,BSSID,SSID,FREQ,SIGNAL device wifi list ifname wlan1 | grep -i bee

# Connect (replace YOUR_PASSWORD). Explicit ifname binds to wlan1.
sudo nmcli device wifi connect "GL-MT3000-bee" \
    password 'YOUR_PASSWORD' \
    ifname wlan1

# Lock interface + set high priority so this is always the preferred connection
sudo nmcli connection modify "GL-MT3000-bee" \
    connection.interface-name wlan1 \
    connection.autoconnect yes \
    connection.autoconnect-priority 100 \
    802-11-wireless.band bg
```

If you later see a second AP advertising the same SSID (mesh extender added), pin the BSSID as well:

```bash
sudo nmcli connection modify "GL-MT3000-bee" 802-11-wireless.bssid AA:BB:CC:DD:EE:FF
```

### Verify

```bash
nmcli device status
# Expect:
#   wlan1  wifi   connected   GL-MT3000-bee
#   wlan0  wifi   unmanaged   --

ip addr show wlan1                # inet 192.168.8.XX/24
ping -c 3 192.168.8.1             # gateway (MT3000)
ping -c 3 192.168.8.60            # desktop

sudo systemctl restart body-launcher
sudo journalctl -u body-launcher -f --since="30 sec ago"
```

Healthy startup journalctl should show drivers coming up (`motor_controller`, `lidar_driver`, `imu_driver`, `oakd_driver`, `local_map`) and **no `Trying to associate with` / `WNM: Preferred List Available` / `RRM: Ignoring radio measurement request`** lines mid-session. If any of those appear, something's wrong — wlan0 is back, or there's a second AP with the same SSID.

## Time sync (chrony)

The Pi 5 has no RTC battery. With the stock `systemd-timesyncd` it boots with a stale clock and can **step** its wall clock minutes into a session. Every `body/**` message carries a wall-clock `ts` that the desktop compares against its own clock (EKF, buffers, scan↔pose interpolation), so skew and mid-session steps cause silent misbehaviour. (Intra-process timeouts on the Pi — command expiry, watchdog heartbeat, stall detection, launcher liveness — use `time.monotonic()` and are immune to steps.)

One time per Pi:

```bash
sudo apt install chrony                      # replaces systemd-timesyncd
sudo cp ~/Body/deploy/chrony-body.conf /etc/chrony/conf.d/body.conf
sudo systemctl enable --now chrony chrony-wait.service
sudo systemctl daemon-reload                 # body-launcher.service is ordered After=time-sync.target
chronyc sources -v; chronyc tracking         # expect the desktop (192.168.8.60) selected, offset < a few ms
```

On the **desktop**, serve time to the bot subnet even without internet (chrony on the desktop):

```
allow 192.168.8.0/24
local stratum 10
```

`chrony-wait.service` holds `time-sync.target` (and therefore `body-launcher`) until the clock is synchronised, up to its own timeout (about 3 min by default on Debian). If the desktop is off at boot the launcher starts after that timeout with an unsynchronised clock; once the desktop appears chrony **slews** (no step after the first 3 updates, `makestep 0.5 3`). If you prefer not to wait at boot, `systemctl disable chrony-wait.service` — the launcher then starts immediately and only the ordering is lost.

## Zenoh transport on the Pi

- Every Pi process connects to the local `zenohd` router (`config.json` → `zenoh.connect_endpoints`, default `tcp/127.0.0.1:7447`); the desktop connects to that router over WiFi.
- `body-launcher.service` sets `ZENOH_MULTICAST_SCOUTING=0`: with explicit endpoints, multicast scouting only creates extra peer-to-peer links that bypass the router and multiply WiFi traffic.
- Optional: uncomment `Environment=ZENOH_MODE=client` in the service so every Pi process talks strictly through the router. Not enabled by default because in client mode a process exits at start-up if `zenohd` is not reachable yet (the launcher restarts it with backoff, so this is recoverable, just noisier). These env vars are read by `body/lib/zenoh_helpers.load_body_config()`; they are Pi-only so the shared `config.json` stays valid for desktop tools.
- Control topics (`body/cmd_vel`, `body/cmd_direct`, `body/drive/goto`, `body/drive/status`, `body/emergency_stop`, `body/estop_rearm`, `body/heartbeat`) are published by `zenoh_helpers.publish_json` with `CongestionControl.BLOCK`, `Priority.INTERACTIVE_HIGH` and `express=True`, so they are never dropped under congestion and are not queued behind depth/map/scan traffic. Disable with `zenoh.control_qos_enabled: false`.

## Motor failsafe

Kernel (sysfs) PWM state outlives the process that set it. Three layers force the motor PWM channels to 0/disabled:

1. `motor_controller` disables PWM first in its `finally` (before closing zenoh) and handles SIGTERM/SIGINT/SIGHUP.
2. The launcher zeroes PWM via sysfs (`motor_gpio.failsafe_zero_pwm`) whenever `motor_controller` exits for any reason, and immediately when it is found hung (`body/odom` stale) before killing it.
3. `ExecStopPost=+… python -m body.lib.motor_gpio --zero-pwm` zeroes PWM whenever the service stops, including after a launcher crash or SIGKILL. `prepare-pwm-sysfs.sh` also starts from duty 0 / disabled.

Manual: `sudo PYTHONPATH=~/Body ~/Body/.venv/bin/python -m body.lib.motor_gpio --zero-pwm`.

## Symptoms this fix addresses

Before (logged 2026-05-16, journalctl excerpt):
```
15:10:32  Trying to associate with 60:83:e7:71:d1:eb (5 GHz)
15:10:42  Authentication with 60:83:e7:71:d1:eb timed out      ← 10 s blocked
15:10:45  ASSOC-REJECT bssid=60:83:e7:71:d1:ef status_code=16
15:10:46  Trying to associate with 60:83:e7:71:d1:ea (2.4 GHz) ← fallback
15:10:56  Connection to 60:83:e7:71:d1:eb completed (5 GHz)     ← 14 s total

…and earlier, after a 5-minute roam:
15:06:37  motor_controller: emergency_stop reason='heartbeat_timeout'
15:06:40  motor_controller: emergency_stop
15:06:45  motor_controller: emergency_stop
```

The 14-second WiFi reconvergence was visible from the operator side as "lost connection with bot, recovered after about 15 s." The heartbeat-timeout e-stops fired when retransmits during the roam ate into the chassis's 2-second watchdog window.

After the wlan1-only configuration above: zero roams in a 20-minute session, no heartbeat timeouts attributable to network gaps.

## Notes

- This is a Pi-side OS configuration, not part of the body code. Re-applies after every fresh Pi flash.
- If the MT3000 router changes its BSSID (firmware update, reset), update the pin via `nmcli connection modify`.
- Eth0 on the Pi can be used for stationary bench testing if you want to skip WiFi entirely — set up a USB Ethernet adapter and connect both Pi and desktop to the same wired subnet. The connection profile stays the same; just the interface changes.
