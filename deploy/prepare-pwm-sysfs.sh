#!/bin/sh
# Export the two motor PWM channels and make them group-writable (gpio).
# Chip discovery mirrors body/lib/motor_gpio.py::_find_pwmchip: explicit arg,
# else pwmchip2 (Pi 5 RP1 PWM0 on recent kernels), else the first chip with
# npwm >= 4. Never hard-code pwmchip0 — the index changes between kernels.
set -eu

pick_chip() {
    if [ -n "${1:-}" ]; then
        echo "$1"; return 0
    fi
    if [ -d /sys/class/pwm/pwmchip2 ]; then
        echo /sys/class/pwm/pwmchip2; return 0
    fi
    for c in /sys/class/pwm/pwmchip*; do
        [ -d "$c" ] || continue
        n="$(cat "$c/npwm" 2>/dev/null || echo 0)"
        if [ "$n" -ge 4 ]; then
            echo "$c"; return 0
        fi
    done
    return 1
}

if ! chip="$(pick_chip "${1:-}")"; then
    echo "prepare-pwm-sysfs: no suitable pwmchip found under /sys/class/pwm" >&2
    exit 1
fi
echo "prepare-pwm-sysfs: using $chip"

for ch in 0 1; do
    if [ ! -d "$chip/pwm$ch" ]; then
        echo "$ch" > "$chip/export" 2>/dev/null || true
    fi
done

for _ in 1 2 3 4 5; do
    if [ -d "$chip/pwm0" ] && [ -d "$chip/pwm1" ]; then
        break
    fi
    sleep 0.1
done

# Start from a known-safe state: duty 0, disabled (a previous run may have died
# with PWM enabled).
for ch in 0 1; do
    if [ -d "$chip/pwm$ch" ]; then
        echo 0 > "$chip/pwm$ch/duty_cycle" 2>/dev/null || true
        echo 0 > "$chip/pwm$ch/enable" 2>/dev/null || true
    fi
done

chown -R root:gpio "$chip"
chmod -R g+rwX "$chip"
