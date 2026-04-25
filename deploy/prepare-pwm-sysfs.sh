#!/bin/sh
set -eu

chip="${1:-/sys/class/pwm/pwmchip0}"

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

chown -R root:gpio "$chip"
chmod -R g+rwX "$chip"
