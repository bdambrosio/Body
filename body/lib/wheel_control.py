"""Pure motor-control logic for motor_controller (no GPIO, no Zenoh — unit-testable).

* :class:`WheelPI` — per-wheel velocity loop: Coulomb (kinetic + velocity) feed-forward,
  ramped static-friction kick with regime hysteresis, PI with anti-windup.
* :func:`parse_cmd_vel` / :func:`parse_cmd_direct` — validate incoming command dicts.
* :class:`CommandArbiter` — per-topic monotonic receipt-time arbitration with expiry.
* :class:`EStopLatch` — motor-side e-stop latch with explicit re-arm.

See docs/motor_controller_spec.md §4.7–§4.9 and §6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# Integrator never sees a dt larger than this (a loop hiccup must not dump a
# big chunk of error into the integral in one tick).
MAX_INTEGRATION_DT_S = 0.1

CMD_TIMEOUT_MIN_MS = 50.0


@dataclass
class WheelPI:
    """Per-wheel velocity PI with Coulomb feed-forward and a ramped static kick.

    Output (signed duty, before |.|/direction split)::

        ff   = sign(v_cmd) * (kinetic_pwm + |v_cmd| * (1 - kinetic_pwm) / max_v)
        duty = ff + kp * err + ki * integ          (err = v_cmd - v_meas)

    With ``kinetic_pwm == 0`` the feed-forward is the legacy ``v_cmd / max_v``.

    Friction regime (measured speed *along the commanded direction*, with
    hysteresis and ``regime_confirm_ticks`` debounce so a single noisy tick
    can't flip it):

      * static  → moving when ``s >= kinetic_threshold_ms``
      * moving  → static when ``s <  static_threshold_ms``

    In the static regime the duty is floored at ``min_drive_pwm`` (breakaway),
    ramped in over ``kick_ramp_ticks`` to avoid a torque step; if the wheel is
    still stuck after the ramp the floor keeps rising at
    ``kick_escalation_per_s`` (bounded by ``integ_max_duty``). The kick ramp
    restarts only when the regime is confirmed static again.

    Anti-windup:
      * integrator reset on a zero command and on a command sign change;
      * integration frozen while the kick floor is still ramping and holding
        the output, and while the output is saturated in the error direction;
      * integral authority capped: ``|ki * integ| <= integ_max_duty`` (and the
        legacy state clamp ``|integ| <= integ_limit``).
    """

    kp: float
    ki: float
    integ_limit: float
    min_drive_pwm: float  # static (breakaway) kick floor
    kinetic_threshold_ms: float = 0.02
    kick_ramp_ticks: int = 5
    kinetic_pwm: float = 0.0  # Coulomb feed-forward offset (kinetic friction duty)
    static_threshold_ms: float | None = None  # default: kinetic_threshold_ms / 2
    regime_confirm_ticks: int = 2
    integ_max_duty: float = 0.2
    kick_escalation_per_s: float = 0.0  # floor growth while still stuck after the ramp
    integ: float = 0.0
    kick_tick_count: int = 0
    moving: bool = False
    _escalation: float = 0.0
    _regime_count: int = 0
    _last_sign: float = 0.0

    def reset(self) -> None:
        self.integ = 0.0
        self.kick_tick_count = 0
        self._escalation = 0.0
        self.moving = False
        self._regime_count = 0
        self._last_sign = 0.0

    def _integ_clamp(self) -> float:
        lim = abs(self.integ_limit)
        if self.ki > 0.0 and self.integ_max_duty >= 0.0:
            lim = min(lim, self.integ_max_duty / self.ki)
        return lim

    def _update_regime(self, s_along: float) -> None:
        thr_hi = self.kinetic_threshold_ms
        thr_lo = (
            self.static_threshold_ms
            if self.static_threshold_ms is not None
            else 0.5 * thr_hi
        )
        n = max(1, int(self.regime_confirm_ticks))
        if self.moving:
            if s_along < thr_lo:
                self._regime_count += 1
                if self._regime_count >= n:
                    self.moving = False
                    self._regime_count = 0
                    self.kick_tick_count = 0
                    self._escalation = 0.0
            else:
                self._regime_count = 0
        else:
            if s_along >= thr_hi:
                self._regime_count += 1
                if self._regime_count >= n:
                    self.moving = True
                    self._regime_count = 0
            else:
                self._regime_count = 0

    def step(
        self, v_cmd: float, v_meas: float, dt: float, max_v: float
    ) -> tuple[float, str]:
        if max_v <= 0.0 or abs(v_cmd) < 1e-6 or not math.isfinite(v_cmd):
            self.reset()
            return 0.0, "fwd"
        if not math.isfinite(v_meas):
            v_meas = 0.0
        sign = 1.0 if v_cmd > 0.0 else -1.0
        if self._last_sign != 0.0 and sign != self._last_sign:
            # Command reversal: old-sign windup would fight the new direction.
            self.integ = 0.0
            self.kick_tick_count = 0
            self._escalation = 0.0
            self.moving = False
            self._regime_count = 0
        self._last_sign = sign
        dt = min(max(0.0, dt), MAX_INTEGRATION_DT_S)

        self._update_regime(v_meas * sign)

        kin = min(max(0.0, self.kinetic_pwm), 1.0)
        ff = sign * (kin + abs(v_cmd) * (1.0 - kin) / max_v)
        err = v_cmd - v_meas
        lim = self._integ_clamp()
        integ_cand = min(lim, max(-lim, self.integ + err * dt))
        pwm = ff + self.kp * err + self.ki * integ_cand

        floor_holding = False
        if not self.moving and self.min_drive_pwm > 0.0:
            self.kick_tick_count += 1
            t_frac = min(1.0, self.kick_tick_count / max(1, self.kick_ramp_ticks))
            if t_frac >= 1.0 and self.kick_escalation_per_s > 0.0:
                # Ramp done and still stuck (breakaway above min_drive_pwm, e.g.
                # low battery / carpet): keep raising the floor, bounded.
                self._escalation = min(
                    max(0.0, self.integ_max_duty),
                    self._escalation + self.kick_escalation_per_s * dt,
                )
            floor = self.min_drive_pwm * t_frac + self._escalation
            if pwm * sign < floor:
                pwm = sign * floor
                # The floor (not the integrator) does the breakaway work while it
                # is ramping or escalating — integrating then would only wind up
                # and lurch at breakaway. Without escalation, once the ramp is
                # complete the (capped) integrator is what pushes beyond it.
                floor_holding = t_frac < 1.0 or self.kick_escalation_per_s > 0.0

        saturated = abs(pwm) > 1.0 and pwm * err > 0.0
        if not floor_holding and not saturated:
            self.integ = integ_cand

        if pwm > 1.0:
            pwm = 1.0
        elif pwm < -1.0:
            pwm = -1.0
        return abs(pwm), "fwd" if pwm >= 0 else "rev"


class CommandError(ValueError):
    """Raised by the command parsers for a malformed message."""


def _finite_number(msg: dict[str, Any], key: str) -> float:
    if key not in msg:
        raise CommandError(f"missing {key!r}")
    v = msg[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise CommandError(f"{key!r} is not a number ({v!r})")
    f = float(v)
    if not math.isfinite(f):
        raise CommandError(f"{key!r} is not finite ({v!r})")
    return f


def _timeout_ms(msg: dict[str, Any], default_ms: float) -> float:
    if msg.get("timeout_ms") is None:
        return float(default_ms)
    return _finite_number(msg, "timeout_ms")


def parse_cmd_vel(msg: dict[str, Any], default_timeout_ms: float = 500.0) -> tuple[float, float, float]:
    """Return ``(linear, angular, timeout_ms)`` or raise :class:`CommandError`."""
    return (
        _finite_number(msg, "linear"),
        _finite_number(msg, "angular"),
        _timeout_ms(msg, default_timeout_ms),
    )


def parse_cmd_direct(msg: dict[str, Any], default_timeout_ms: float = 500.0) -> tuple[float, float, float]:
    """Return ``(left, right, timeout_ms)`` or raise :class:`CommandError`."""
    return (
        _finite_number(msg, "left"),
        _finite_number(msg, "right"),
        _timeout_ms(msg, default_timeout_ms),
    )


def clamp_timeout_ms(timeout_ms: float, max_ms: float) -> float:
    hi = max(CMD_TIMEOUT_MIN_MS, float(max_ms))
    return min(hi, max(CMD_TIMEOUT_MIN_MS, float(timeout_ms)))


@dataclass
class StoredCommand:
    kind: str  # "twist" | "direct"
    a: float  # linear | left
    b: float  # angular | right
    recv_mono: float
    timeout_s: float

    def expired(self, now_mono: float) -> bool:
        return (now_mono - self.recv_mono) > self.timeout_s


class CommandArbiter:
    """Per-topic receipt-time arbitration (Pi monotonic clock; sender ``ts`` ignored).

    Each topic keeps only its latest command, stamped with its local receipt
    time and its own clamped timeout. The active command is the most recently
    *received* one that has not expired; expired commands are dropped.
    """

    def __init__(self, max_timeout_ms: float = 1000.0) -> None:
        self.max_timeout_ms = float(max_timeout_ms)
        self._cmds: dict[str, StoredCommand] = {}

    def store(self, kind: str, a: float, b: float, timeout_ms: float, now_mono: float) -> StoredCommand:
        cmd = StoredCommand(
            kind, float(a), float(b), now_mono,
            clamp_timeout_ms(timeout_ms, self.max_timeout_ms) / 1000.0,
        )
        self._cmds[kind] = cmd
        return cmd

    def select(self, now_mono: float) -> StoredCommand | None:
        for k in [k for k, c in self._cmds.items() if c.expired(now_mono)]:
            del self._cmds[k]
        if not self._cmds:
            return None
        return max(self._cmds.values(), key=lambda c: c.recv_mono)

    def clear(self) -> None:
        self._cmds.clear()


@dataclass
class EStopLatch:
    """Motor-side e-stop state.

    * ``body/emergency_stop`` latches.
    * ``body/status.e_stop_active`` is the watchdog's own latch, mirrored as a level.
    * The motor latch clears only on ``body/estop_rearm`` while the heartbeat is
      ok, or when the watchdog's status goes true → false (the watchdog only
      clears after accepting a re-arm). Motion commands never clear it.
    """

    latched: bool = False
    status_e_stop: bool = False

    @property
    def active(self) -> bool:
        return self.latched or self.status_e_stop

    def on_emergency_stop(self) -> None:
        self.latched = True

    def on_status(self, e_stop_active: bool) -> None:
        was = self.status_e_stop
        self.status_e_stop = bool(e_stop_active)
        if was and not self.status_e_stop:
            self.latched = False

    def on_rearm(self, heartbeat_ok: bool) -> bool:
        """Returns True if the motor latch was cleared."""
        if self.latched and heartbeat_ok:
            self.latched = False
            return True
        return False
