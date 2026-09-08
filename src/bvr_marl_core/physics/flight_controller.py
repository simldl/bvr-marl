"""
Inner-loop lift-vector autopilot.

The outer loop (an RL policy, or a behaviour tree going through the same action
processor) commands a *lift vector*: a load factor ``n`` and a bank angle
``phi``. This module is what actually flies that command.

Why this layer exists
---------------------
The kinematic point-mass form used by :mod:`bvr_marl_core.physics.flying_objects`

    chi_dot   = (g / V) * n * sin(phi) / cos(gamma)
    gamma_dot = (g / V) * (n * cos(phi) - cos(gamma))

says nothing about how long it takes to *get* to a bank angle. Evaluated once
per one-second control tick, it lets an agent hold +70 deg of bank on one tick
and -70 deg on the next, paying nothing for the reversal. That is unflyable, and
because it is free, an optimizer will find it and sit on it. The resulting
ground tracks are the characteristic chain of disconnected, kinked segments.

So bank and load factor are held here as genuine *states*, slewed toward their
commands under a roll-rate and a g-onset limit, and the guidance equations are
evaluated against the bank the aircraft has actually achieved -- never against
the bank it was told to reach. The loop runs at a sub-step well below the
control interval so a hard turn is integrated as an arc rather than as one long
straight chord.

Reference form: the standard 3-DOF point-mass equations, e.g. Vinh, *Flight
Mechanics of High-Performance Aircraft* (1993), ch. 2; Stevens & Lewis,
*Aircraft Control and Simulation*, 2nd ed., sec. 2.5. The cascade structure
(outer command -> rate-limited inner loop -> airframe) mirrors the low-level
controller in the AeroBench F-16 benchmark (Heidlauf et al., ARCH 2018), at the
fidelity our point-mass model supports.
"""

from __future__ import annotations

import math

#: Guard on cos(gamma) in the guidance denominators. At the 60 deg flight-path
#: limit cos(gamma) = 0.5, so this only ever binds if the pitch cap is raised
#: toward the vertical, where the flat-Earth turn equation degenerates anyway.
_MIN_COS_GAMMA = 0.1

#: Longest integration step the inner loop will take, in seconds.
#:
#: Chosen from measured integration error, not from taste. Flying a sustained
#: 70 deg / 5 g turn for 60 s and comparing against a 200-step-per-tick
#: reference, the cross-track error at a 1 s control tick is:
#:
#:     steps/tick    error      cost (ms per 4v4 sim tick)
#:         1          240 m          1.82
#:         2          247 m          2.08
#:         4           96 m          2.67
#:        10           74 m          4.00
#:        20           42 m            --
#:
#: One step per tick puts the error at 240 m, which is the same order as the
#: 250 m missile lethal radius -- i.e. large enough to change intercept
#: outcomes, not merely to look jagged. Four steps cuts that to 96 m for a 47%
#: tick cost; ten steps buys only a further 22 m for another 50%. So 0.25 s is
#: the knee of the curve. Override per-instance where a study needs more.
DEFAULT_MAX_SUBSTEP_S = 0.25


class LiftVectorAutopilot:
    """Flies an (n, phi) command with rate-limited bank and load-factor states."""

    def __init__(self, parent, max_substep_s: float = DEFAULT_MAX_SUBSTEP_S):
        self.parent = parent
        self.max_substep_s = float(max_substep_s)

        # Achieved states. These persist across ticks -- that persistence is the
        # whole point, it is what makes repositioning the lift vector cost time.
        self.phi_deg = 0.0
        self.n = 1.0

        # Commanded lift vector, refreshed by the outer loop each control tick.
        self.n_cmd = 1.0
        self.phi_cmd_deg = 0.0
        self.s_factor = 1.0
        self.c_factor = 1.0

        #: False until an outer loop actually commands a lift vector. While
        #: false the legacy desired-heading path (AWACS, validation harnesses,
        #: scripted manoeuvres) keeps its previous behaviour untouched.
        self.active = False

    # -- outer-loop interface -------------------------------------------------

    def set_lift_vector(
        self,
        n_cmd: float,
        phi_cmd_deg: float,
        s_factor: float = 1.0,
        c_factor: float = 1.0,
    ) -> None:
        """Accept a lift-vector command for the coming control tick."""
        self.n_cmd = float(n_cmd)
        self.phi_cmd_deg = float(phi_cmd_deg)
        self.s_factor = float(s_factor)
        self.c_factor = float(c_factor)
        self.active = True

    def reset(self) -> None:
        """Return to wings-level 1 g. MUST be called on episode reset."""
        self.phi_deg = 0.0
        self.n = 1.0
        self.n_cmd = 1.0
        self.phi_cmd_deg = 0.0
        self.s_factor = 1.0
        self.c_factor = 1.0
        self.active = False

    # -- inner loop -----------------------------------------------------------

    def substep_count(self, tick_secs: float) -> int:
        """Number of integration steps to take across one control tick."""
        if tick_secs <= self.max_substep_s:
            return 1
        return int(math.ceil(tick_secs / self.max_substep_s))

    def advance(self, dt_s: float) -> tuple[float, float, float]:
        """Advance the inner loop one sub-step.

        Slews the achieved bank and load factor toward their commands, then
        evaluates the point-mass guidance equations against the *achieved*
        attitude.

        Returns:
            (desired_yaw_deg, desired_pitch_deg, omega_max_deg_s) -- the
            attitude targets for this sub-step and the turn-rate cap that
            produced them. Handing the cap back out lets the airframe layer
            reuse this exact number instead of recomputing a looser one, so the
            command is rate-limited once and in one place.
        """
        m = self.parent
        physics = m.physics
        v = max(float(m.speed), 1e-3)
        g = physics.g
        alt = m.position.alt

        # 1) Roll state: a rate-limited slew toward the commanded bank. This is
        #    the term that makes a lift-vector reversal cost time.
        phi_max = self._phi_limit()
        phi_cmd = max(-phi_max, min(phi_max, self.phi_cmd_deg))
        roll_rate_max = physics.max_roll_rate_deg_s * self.s_factor * self.c_factor
        self.phi_deg = _slew(self.phi_deg, phi_cmd, roll_rate_max * dt_s)

        # 2) Load-factor state: a rate-limited slew under the g-onset limit.
        n_max = self._n_limit()
        n_cmd = max(-n_max, min(n_max, self.n_cmd))
        self.n = _slew(self.n, n_cmd, physics.n_rate_max_g_per_s * dt_s)

        # 3) Guidance from the ACHIEVED lift vector, retaining the flight-path
        #    angle terms that the flat-Earth simplification drops. At the 60 deg
        #    pitch limit the cos(gamma) terms are a factor-of-two effect.
        gamma = math.radians(m.pitch_deg)
        cos_gamma = max(math.cos(gamma), _MIN_COS_GAMMA)
        phi_rad = math.radians(self.phi_deg)

        chi_dot = (g / v) * self.n * math.sin(phi_rad) / cos_gamma
        gamma_dot = (g / v) * (self.n * math.cos(phi_rad) - cos_gamma)

        # 4) Single rate authority. The envelope-aware caps are computed here
        #    and nowhere else; the airframe layer is handed omega_max so it
        #    follows rather than re-limiting against a different envelope.
        omega_max = self._omega_limit(v, alt)
        q_max = physics.max_pitch_rate_deg_s * self.s_factor * self.c_factor

        chi_dot_deg = _clip(math.degrees(chi_dot), omega_max)
        gamma_dot_deg = _clip(math.degrees(gamma_dot), q_max)

        # 5) Integrate the attitude targets for this sub-step.
        desired_yaw = m.yaw_deg + chi_dot_deg * dt_s
        desired_pitch = m.pitch_deg + gamma_dot_deg * dt_s

        pitch_limit = self._pitch_limit(alt, v)
        desired_pitch = max(-pitch_limit, min(pitch_limit, desired_pitch))
        # Mirror the airframe's ground-proximity floor so the command never
        # fights the blocker below it.
        floor = self._pitch_floor(alt)
        if floor is not None:
            desired_pitch = max(desired_pitch, floor)

        # Publish the achieved load factor for the induced-drag term. Using the
        # achieved value rather than the command is what makes an unloaded
        # push-over actually cheap and a sustained pull actually expensive.
        physics.n_external = abs(self.n)

        return desired_yaw, desired_pitch, omega_max

    # -- envelope helpers -----------------------------------------------------

    def _phi_limit(self) -> float:
        base = getattr(self.parent.physics, "phi_max_deg", 80.0)
        return max(0.0, base * self.s_factor * self.c_factor)

    def _n_limit(self) -> float:
        n_max = self.parent.physics.n_max
        return max(1.0, min(n_max, self.s_factor * self.c_factor * n_max))

    def _omega_limit(self, v: float, alt: float) -> float:
        physics = self.parent.physics
        compute = getattr(physics, "compute_instantaneous_turn_rate", None)
        if callable(compute):
            omega = compute(v, alt)
        else:
            omega = getattr(physics, "max_turn_rate_deg_s", 30.0)
        if not isinstance(omega, (int, float)):
            omega = 30.0
        return max(0.0, float(omega) * self.s_factor * self.c_factor)

    def _pitch_limit(self, alt: float, v: float) -> float:
        physics = self.parent.physics
        getter = getattr(physics, "get_pitch_limit_deg", None)
        if callable(getter):
            limit = getter(self.parent.position, v)
            if isinstance(limit, (int, float)):
                return float(limit)
        return 60.0

    def _pitch_floor(self, alt: float) -> float | None:
        physics = self.parent.physics
        getter = getattr(physics, "_ground_proximity_pitch_floor", None)
        if callable(getter):
            floor = getter(alt)
            if isinstance(floor, (int, float)):
                return float(floor)
        return None


def _slew(current: float, target: float, max_delta: float) -> float:
    """Move ``current`` toward ``target`` by at most ``max_delta``."""
    delta = target - current
    if abs(delta) <= max_delta:
        return target
    return current + (max_delta if delta > 0 else -max_delta)


def _clip(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))
