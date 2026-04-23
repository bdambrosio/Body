# imu_driver.py — Informal Specification

**Project:** Body (robot body software stack)
**Date:** 2026-04-23
**Hardware:** BNO085 9-DOF IMU breakout (CEVA SH-2 fused firmware), Raspberry Pi 5
**Related:** `imu_integration_spec.md` (consumer-side contract), `body_project_spec.md` §5.6 (wire message shape)

This spec is authoritative for the **producer** side — what runs on the Pi, what wires go where, and how the driver behaves. The consumer contract (`imu_integration_spec.md`) defines what desktop SLAM expects on the wire; this spec defines how the Pi satisfies that contract.

---

## 1. Role

The BNO085 is the **primary source of robot orientation**. Desktop SLAM (`desktop/nav/slam/imu_yaw.py`) consumes yaw from its on-chip fused quaternion. The Pi side only wraps i²c I/O and publishes; it does not integrate gyro or filter accel — the SH-2 firmware already does that better than we could.

This unit's OAK-D-Lite is early-Kickstarter hardware without an onboard IMU; the BNO085 replaces that role entirely. When this driver ships, `oakd_driver.py` stops publishing `body/oakd/imu` (see §10).

---

## 2. BNO085 Module Pinout

Assumed breakout: SparkFun-style 10-pin 0.1" header exposing all SH-2 interface modes (I²C / SPI / UART). See attached schematic for the reference design; equivalent Adafruit/third-party boards use the same pinout.

### Header J6 (1×10, 0.1" pitch)

| Pin | Name           | Description                                                          |
|-----|----------------|----------------------------------------------------------------------|
| 1   | 3V3            | Supply, 2.4–3.6 V. Use Pi 3.3 V rail.                                |
| 2   | GND            | Ground (shared with Pi).                                             |
| 3   | SCL / SCK / RX | I²C clock in I²C mode; SPI clock in SPI; UART RX in UART.            |
| 4   | SDA / MISO / TX| I²C data in I²C mode; SPI MISO in SPI; UART TX in UART.              |
| 5   | ADDR / MOSI    | I²C address select (see §3); SPI MOSI in SPI mode.                   |
| 6   | CS             | SPI chip select. Unused in I²C. Leave open.                          |
| 7   | INT            | Active-low interrupt — asserts when the SH-2 FIFO has a report ready.|
| 8   | RST            | Active-low reset. On-board 10 kΩ pull-up keeps it de-asserted if unconnected. |
| 9   | PS1            | Interface select bit 1. On-board 10 kΩ pull-down → I²C by default.   |
| 10  | PS0 / WAKE     | Interface select bit 0 / host wake. On-board pull-down → I²C default.|

### Interface mode (PS1:PS0)

| PS1 | PS0 | Mode      |
|-----|-----|-----------|
| 0   | 0   | I²C (default — use this) |
| 0   | 1   | UART-RVC  |
| 1   | 0   | UART      |
| 1   | 1   | SPI       |

We use I²C. PS0 and PS1 stay unconnected (pulled low on the board).

### I²C pull-ups on the module

The breakout has 2.2 kΩ pull-ups on SDA and SCL to 3.3 V, enabled by solder jumper **SJ1**. Leave SJ1 closed (factory default). With this board on the bus, do **not** add external pull-ups on the Pi side. If a second I²C peripheral later lands on the same bus with its own pull-ups, verify combined pull-up strength stays ≥ 1.6 kΩ and ≤ 10 kΩ.

---

## 3. I²C Address

- **Default (ADDR open):** `0x4B`
- **ADDR shorted to 3V3 (jumper closed):** `0x4A`

Leave the jumper open. We will use **`0x4B`**. If a future second BNO085 is added, strap the second board to `0x4A`.

Verify on first power-up:

```
sudo apt install i2c-tools
sudo i2cdetect -y 1
# expect "4b" in the matrix
```

---

## 4. Raspberry Pi 5 GPIO Allocation

This spec extends `motor_controller_spec.md` §3. Existing pins are preserved; only the IMU pins are new.

### Pi 5 — full pin allocation after IMU wiring

| Function        | BCM GPIO | Physical Pin | Notes |
|-----------------|----------|--------------|-------|
| Motor L PWM     | 12       | 32           | Hardware PWM0 CH0 (existing) |
| Motor L DIR     | 5        | 29           | Digital out (existing) |
| Motor R PWM     | 13       | 33           | Hardware PWM0 CH1 (existing) |
| Motor R DIR     | 6        | 31           | Digital out (existing) |
| Encoder L ch A  | 23       | 16           | `gpio_claim_alert`, pull-up (existing) |
| Encoder L ch B  | 24       | 18           | `gpio_claim_alert`, pull-up (existing) |
| Encoder R ch A  | 27       | 13           | `gpio_claim_alert`, pull-up (existing) |
| Encoder R ch B  | 22       | 15           | `gpio_claim_alert`, pull-up (existing) |
| **IMU SDA**     | **2**    | **3**        | I²C1 data. Hardware I²C, no claim needed. |
| **IMU SCL**     | **3**    | **5**        | I²C1 clock. Hardware I²C, no claim needed. |
| **IMU INT**     | **17**   | **11**       | Input, no pull (BNO085 actively drives). Edge-alert on falling. |
| **IMU RST**     | **25**   | **22**       | Output, default HIGH (de-asserted). Pulse LOW ≥ 10 µs to reset. |

**Rationale for 17 and 25:**
- GPIO 17 (pin 11) is free of kernel-reserved functions and physically near the i²c pins for short wiring.
- GPIO 25 (pin 22) is on the same header side as the motor-driver wiring, free, and not bound to SPI/I²C/UART aliases.
- Neither pin conflicts with the motor controller's existing assignments or with `pwmchip2` on the RP1.

### Wiring Summary

```
Pi 5                        BNO085 Module (J6)
────                        ──────────────────
3.3V   (header pin 1)  ──── 3V3  (J6 pin 1)
GND    (header pin 6)  ──── GND  (J6 pin 2)
GPIO 3 / SCL (pin 5)   ──── SCL  (J6 pin 3, labeled SCL/SCK/RX)
GPIO 2 / SDA (pin 3)   ──── SDA  (J6 pin 4, labeled SDA/MISO/TX)
GPIO 17      (pin 11)  ──── INT  (J6 pin 7)
GPIO 25      (pin 22)  ──── RST  (J6 pin 8)
                            ADDR  (J6 pin 5) — leave open → 0x4B
                            CS    (J6 pin 6) — leave open
                            PS1   (J6 pin 9)  — leave open → I²C mode
                            PS0   (J6 pin 10) — leave open → I²C mode
```

Any Pi GND pin is equivalent to header pin 6. Cross-check pin order on the physical breakout's silkscreen before wiring — third-party clones occasionally reorder J6.

---

## 5. Mounting and Frame

Body frame convention (shared with lidar, local_map, fuser): **x-forward, y-left, z-up** (right-handed).

Mount the BNO085 so its silkscreen axes match the body frame, or document a static rotation. The driver applies the mount rotation **on the Pi side** before publishing — consumers receive body-frame values only (see `imu_integration_spec.md` §3).

- Physically place the IMU away from the motor driver and motor wiring — several centimeters is fine. The MDD10A and unshielded motor wires emit magnetic noise that contaminates the magnetometer. Worst-case noise determines whether we run Rotation Vector (mag-fused, absolute yaw) or Game Rotation Vector (gyro+accel only, relative yaw). See §7.
- Orient the module flat and level. Pitch/roll drift calibration is automatic but benefits from a known gravity reference at startup.

---

## 6. Software — imu_driver.py

New process `body/imu_driver.py`. Mirrors the structure of `motor_controller.py` and `oakd_driver.py`: loads `config.json`, opens Zenoh, opens i²c, runs a loop, publishes `body/imu`, exits cleanly on SIGTERM/SIGINT.

### 6.1 Responsibility

Sole owner of the BNO085 i²c transaction. Reads SH-2 FIFO reports, assembles a single JSON message per fusion tick, publishes to `body/imu`. No interpretation.

### 6.2 Zenoh Interface

**Publishes:**
- `body/imu` — fused IMU report. Schema per `imu_integration_spec.md` §2.

**Subscribes:** none (driver is read-only relative to the robot).

### 6.3 Loop

- Initialize i²c (`/dev/i2c-1`) at 400 kHz.
- Drive RST low for ≥ 10 µs, release, wait ≥ 100 ms for SH-2 boot.
- Enable SH-2 reports at target rates:
  - Rotation Vector **or** Game Rotation Vector: 100 Hz
  - Accelerometer: 100 Hz
  - Gyroscope: 400 Hz (aggregated down to 100 Hz for the wire — take the latest sample at publish time, or average)
  - Linear Acceleration: optional, 100 Hz (see config)
- Loop:
  1. Block on INT falling edge (or poll the status register every 2 ms if INT is not wired).
  2. Read all pending reports from the SH-2 FIFO (multiple report types may arrive per interrupt).
  3. Cache the most recent of each report type.
  4. If the cache has at least an orientation **and** an accel **and** a gyro newer than the last publish, assemble and publish.
  5. Rate-limit publishes to `publish_hz` (default 100).

**Timestamps:** use the SH-2 per-report timestamp converted to wall time (subtract `time.monotonic() → time.time()` offset captured at boot). Not the publish time.

### 6.4 Fusion mode selection

Driver supports both fusion modes (consumer spec §1). Mode is configured by `motor`-style key (`imu.fusion_mode`), but the driver always reports whichever mode actually produced the latest quaternion in the message payload (`fusion.mode`). Mismatch (e.g. mag calibration lost mid-run) is surfaced, not hidden.

Proposed fallback policy (config-driven):

```
imu.fusion_mode = "rotation_vector"          # preferred — absolute yaw
imu.fusion_fallback = "game_rotation_vector" # used if mag accuracy > threshold
imu.mag_accuracy_fallback_rad = 0.087        # ~5°; see §7
```

If `rotation_vector` accuracy exceeds the threshold for ≥ N consecutive reports, the driver re-enables `game_rotation_vector`, notes the switch in a stdout log line, and tags subsequent messages accordingly. No automatic switch back — re-enabling Rotation Vector requires a driver restart (keeps behavior deterministic across a session).

### 6.5 Boot-time settle

The BNO085 auto-calibrates gyro bias at rest in the first ~1–2 s. During that window the SH-2 accuracy field is high. The driver:

- Holds off publishing until `fusion.accuracy_rad` first drops below `imu.calibration_stable_threshold_rad` (default 0.087 rad ≈ 5°), **and** at least `imu.settle_time_s` (default 2.0) of wall time has elapsed since driver start.
- While settling, the driver publishes a one-shot stdout line at 1 Hz showing current accuracy so operators know to keep the robot stationary.
- Optionally (§9), publishes a `calibrating`/`ready` status signal — TBD whether this rides on `body/status` or a new `body/imu_status` topic. v1: print-only.

### 6.6 Configuration (`config.json` → `imu` section)

Suggested keys (implementer may trim on first pass):

```json
"imu": {
  "enabled": true,
  "i2c_bus": 1,
  "i2c_address": 75,                       // 0x4B
  "reset_gpio": 25,
  "interrupt_gpio": 17,
  "publish_hz": 100,
  "fusion_mode": "rotation_vector",
  "fusion_fallback": "game_rotation_vector",
  "mag_accuracy_fallback_rad": 0.087,
  "calibration_stable_threshold_rad": 0.087,
  "settle_time_s": 2.0,
  "linear_accel_enabled": false,
  "mount_rotation_euler_deg": [0.0, 0.0, 0.0]  // roll, pitch, yaw applied before publish
}
```

### 6.7 Error handling

- **i²c open failure / NACK on probe at `0x4B`**: print a single diagnostic line (address used, bus, suggested fixes — check `i2cdetect -y 1`, check SJ1 pull-ups, check wiring), then exit nonzero so `watchdog` can detect and surface via `body/status`. Do not silently publish stale data.
- **INT never fires for > 500 ms after enabling reports**: fall back to polling once, log, then treat as hardware failure if still dark. Same exit path.
- **SH-2 report decode error** (bad checksum, unexpected length): skip the report, increment an internal counter. If error rate > 10% for > 1 s, log + restart the SH-2 (RST pulse) before giving up.
- **Mid-run i²c bus error**: attempt one RST pulse + re-enable; if still failing, exit.

### 6.8 Shutdown

On SIGTERM/SIGINT: stop enabling reports, release Zenoh session, close i²c, drive RST low (optional, just for tidiness), exit zero.

---

## 7. Magnetometer Interference Test

Before trusting Rotation Vector mode under load, run this once after first integration and again any time the chassis layout changes:

1. Boot with the robot stationary. Wait for `fusion.accuracy_rad` to drop below threshold.
2. Log `fusion.accuracy_rad` and `mag_status` at 1 Hz for 10 s.
3. Command `cmd_direct left=0.3 right=0.3` (or full-speed, duty-limited) for 10 s straight-line.
4. Compare `fusion.accuracy_rad` median during motion vs stationary.

- Median accuracy stays < `mag_accuracy_fallback_rad` under load → keep `rotation_vector`.
- Accuracy inflates beyond threshold → set `imu.fusion_mode = "game_rotation_vector"` and accept ~0.5–1°/min yaw drift (which scan-matching can absorb; see `encoder_integration_spec.md` §1 and `imu_integration_spec.md` §5).

Record the result in a build-log note — this is per-robot configuration.

---

## 8. Magnetometer Calibration (Rotation Vector only)

Required once per robot (SH-2 persists to its flash):

1. Power up with motors de-energized (no commanded motion, nothing driving PWM).
2. Hold the chassis away from large metal or nearby magnets.
3. Slowly rotate the whole chassis through a figure-8 motion for ≥ 30 s, covering all three axes.
4. Watch `fusion.mag_status` — it progresses `unreliable → low → medium → high` as points accumulate. Once `high`, calibration is saved.
5. Reboot the IMU (RST pulse or power cycle) and verify `mag_status` returns to `high` or `medium` — confirms persistence.

If the robot is relocated to a new magnetic environment (different room, different metal-content floor), consider recalibrating.

---

## 9. Dependencies (Python)

Two viable library paths; implementer picks after the hardware arrives and one is validated:

1. **Adafruit `adafruit-circuitpython-bno08x`** via Blinka on the Pi.
   - Pros: widely documented, Python-native, handles SH-2 decode.
   - Cons: Blinka layer on Pi 5 is usable but heavier; depends on `adafruit_blinka`, `adafruit_bus_device`.
   - Install: `pip install adafruit-circuitpython-bno08x adafruit-blinka`.

2. **Hillcrest / CEVA `sh2` C driver with thin Python binding** (custom or via `pi-bno085`-style packages on PyPI).
   - Pros: closer to the metal, smaller dependency graph.
   - Cons: one-time integration cost, fewer examples.

**Recommendation:** start with option 1. Swap to option 2 only if Blinka blocks us or we need sub-100 Hz jitter that Python-in-the-loop cannot deliver — neither is expected.

Add to `requirements.txt` when implemented. Keep the Pi-side venv's `--system-site-packages` setting (required for `lgpio`; see `README.md`).

---

## 10. Rollout Phases

1. **Pre-wire:** spec merged (this file), consumer spec merged (`imu_integration_spec.md`), desktop tests green.
2. **Wire hardware:** 6 wires total (3V3, GND, SDA, SCL, INT, RST). `i2cdetect` shows `0x4b`.
3. **Bring up `imu_driver.py`:** publish `body/imu` at 100 Hz with real fusion data. Verify with `scripts/record_body_topics.py` and manual `z2-tools` subscribe.
4. **Consumer switchover (atomic commit):**
   - Delete IMU publish path in `body/oakd_driver.py` (this unit has no OAK-D IMU).
   - Remove `body/oakd/imu` from `watchdog.monitored_topics` in `config.json`.
   - Add `body/imu` to monitored topics; add `imu_driver` to watchdog's managed processes (parallel to `motor_controller`, `lidar_driver`, `oakd_driver`).
   - Rename `schemas.oakd_imu_report` → `schemas.imu_report`; extend with `fusion` block; update `body_project_spec.md` §5.6 if it still references the old shape.
   - Update `desktop/chassis/config.py::Topics.oakd_imu` → `Topics.imu` (or just rename the string — `ImuReading.from_payload` already accepts either topic).
5. **Run interference test (§7)** and pick the final `fusion_mode` for this chassis.
6. **Run magnetometer calibration (§8)** if `rotation_vector` is kept.
7. **Acceptance tests** per `imu_integration_spec.md` §8 — must all pass before declaring the IMU subsystem live.

---

## 11. Open Questions

- **Mount location:** picking a spot that's far enough from the MDD10A + motor leads that we can stay in Rotation Vector mode would save us noticeable yaw drift over a long session. Physical layout TBD.
- **Calibrating status on `body/status`:** nice to have for the UI (so Jill can grey out headings during settle), but not on critical path. Revisit after v1 ships.
- **Linear-accel usefulness:** spec consumers do not ask for `linear_accel`. Leave disabled (`linear_accel_enabled: false`) on first boot; enable if a future heuristic needs it.
- **Double-IMU future:** the `0x4A` / `0x4B` address split exists; if we ever want a second IMU for redundancy or body-vs-head separation, the second strap lands on `0x4A`. Not a v1 concern.
