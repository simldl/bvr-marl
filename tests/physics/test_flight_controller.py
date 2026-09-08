#!/usr/bin/env python3
"""
Tests for the inner-loop lift-vector autopilot.

These pin the behaviour the autopilot exists to provide: bank and load factor
are states with rate limits, guidance reads the achieved attitude rather than
the command, the control tick is integrated as an arc, and the turn-rate cap has
exactly one owner.
"""

import math

import pytest

from bvr_marl_core.aircraft.control.movement_control import AircraftControlSystem
from bvr_marl_core.physics.aircraft import AircraftPhysics
from bvr_marl_core.physics.flight_controller import (
    DEFAULT_MAX_SUBSTEP_S,
    LiftVectorAutopilot,
)
from bvr_marl_core.simulator.core.helpers import Position


class _Jet:
    """Minimal airframe carrier: real physics, real position, no sim coupling."""

    def __init__(self, speed=300.0, alt=8000.0, yaw=0.0, pitch=0.0):
        self.physics = AircraftPhysics(
            AircraftPhysics.Params(mass_kg=16000.0, reference_area_m2=28.0)
        )
        self.position = Position(0.0, 0.0, alt)
        self.yaw_deg = yaw
        self.pitch_deg = pitch
        self.roll_deg = 0.0
        self.speed = speed
        self.min_speed_mps = 80.0
        self.max_speed_mps = 680.0
        self.min_alt_m = 0.0
        self.max_alt_m = 20000.0
        self.map_limits = _Limits()
        self.keep_inside_boundary = True
        self.boundary_violation_active = False
        self.boundary_violation_countdown = 0
        self.removal_reason = None
        self.fuel = None
        self.is_mortally_hit = False
        self.control = AircraftControlSystem(self)


class _Limits:
    bottom_lat = -80.0
    top_lat = 80.0
    left_lon = -180.0
    right_lon = 180.0


# ============================================================================
# Bank is a state with a rate limit
# ============================================================================


def test_bank_reversal_cannot_happen_within_one_tick():
    """A lift-vector reversal must cost time, not be free between two ticks."""
    jet = _Jet()
    jet.control.set_lift_vector(n_cmd=4.0, phi_cmd_deg=60.0)
    jet.control.update_movement(1.0)
    bank_after_right = jet.control.autopilot.phi_deg
    assert bank_after_right == pytest.approx(60.0, abs=1e-6)

    # Now command the mirror-image bank. One tick is 1.0 s at 150 deg/s, so the
    # airframe can traverse at most 150 deg -- it reaches -60, but only because
    # 120 deg of travel fits. Shrink the tick and it must fall short.
    jet.control.set_lift_vector(n_cmd=4.0, phi_cmd_deg=-60.0)
    jet.control.update_movement(0.5)
    bank_after_reversal = jet.control.autopilot.phi_deg
    assert bank_after_reversal > -60.0, "reversal completed instantly; roll rate not enforced"
    assert bank_after_reversal == pytest.approx(60.0 - 150.0 * 0.5, abs=1e-6)


def test_roll_rate_limit_is_honoured_per_substep():
    """Bank slews at no more than max_roll_rate_deg_s, measured across the tick."""
    jet = _Jet()
    jet.physics.max_roll_rate_deg_s = 90.0
    jet.control.set_lift_vector(n_cmd=1.0, phi_cmd_deg=80.0)
    jet.control.update_movement(0.5)
    assert jet.control.autopilot.phi_deg == pytest.approx(45.0, abs=1e-6)


def test_bank_persists_across_ticks():
    """Achieved bank is carried between control ticks, not rebuilt from zero."""
    jet = _Jet()
    jet.control.set_lift_vector(n_cmd=3.0, phi_cmd_deg=45.0)
    jet.control.update_movement(0.1)
    first = jet.control.autopilot.phi_deg
    jet.control.update_movement(0.1)
    second = jet.control.autopilot.phi_deg
    assert 0.0 < first < second <= 45.0


def test_reported_roll_is_the_achieved_bank_not_an_inversion():
    """Published roll equals the autopilot's state, so it cannot drift from it."""
    jet = _Jet()
    jet.control.set_lift_vector(n_cmd=5.0, phi_cmd_deg=55.0)
    jet.control.update_movement(1.0)
    assert jet.roll_deg == pytest.approx(jet.control.autopilot.phi_deg, abs=1e-9)
    assert jet.roll_deg == pytest.approx(55.0, abs=1e-6)


def test_g_onset_rate_limit_is_enforced():
    """Load factor ramps under n_rate_max_g_per_s rather than stepping."""
    jet = _Jet()
    jet.physics.n_rate_max_g_per_s = 4.0
    jet.control.set_lift_vector(n_cmd=9.0, phi_cmd_deg=0.0)
    jet.control.update_movement(0.5)
    # Starts at 1 g, 0.5 s at 4 g/s -> 3 g.
    assert jet.control.autopilot.n == pytest.approx(3.0, abs=1e-6)


# ============================================================================
# Guidance reads the achieved attitude
# ============================================================================


def test_turn_rate_follows_achieved_bank_not_commanded_bank():
    """With bank still ramping, heading change is far below the commanded value."""
    ramping = _Jet()
    ramping.control.set_lift_vector(n_cmd=4.0, phi_cmd_deg=60.0)
    ramping.control.update_movement(0.1)

    settled = _Jet()
    settled.control.autopilot.phi_deg = 60.0
    settled.control.autopilot.n = 4.0
    settled.control.set_lift_vector(n_cmd=4.0, phi_cmd_deg=60.0)
    settled.control.update_movement(0.1)

    assert abs(settled.yaw_deg) > abs(ramping.yaw_deg) > 0.0


def test_cos_gamma_terms_are_present_in_guidance():
    """In a steep climb the flight-path terms must change the guidance output.

    Flat-Earth guidance drops cos(gamma): chi_dot = (g/V) n sin(phi) and
    gamma_dot = (g/V)(n cos(phi) - 1). At gamma = 60 deg, cos(gamma) = 0.5, so
    the correct forms differ by a factor of two in turn rate and by half a g in
    the gravity term. Compare a climbing jet against a level one at identical
    bank and load factor.
    """
    level = _Jet(pitch=0.0)
    climbing = _Jet(pitch=60.0)
    for jet in (level, climbing):
        jet.control.autopilot.phi_deg = 45.0
        jet.control.autopilot.n = 3.0
        jet.control.set_lift_vector(n_cmd=3.0, phi_cmd_deg=45.0)

    yaw_level, _, _ = level.control.autopilot.advance(0.05)
    yaw_climb, _, _ = climbing.control.autopilot.advance(0.05)

    # 1/cos(60 deg) = 2.0: the climbing jet's heading rate must be double.
    rate_level = yaw_level - level.yaw_deg
    rate_climb = yaw_climb - climbing.yaw_deg
    assert rate_climb == pytest.approx(2.0 * rate_level, rel=1e-6)


def test_gamma_dot_subtracts_cos_gamma_not_unity():
    """Wings-level at 1 g in a climb, the flight path must still bend upward.

    Held at 45 deg of climb with wings level and 1 g:
        correct form  n cos(phi) - cos(gamma) = 1 - 0.707 = +0.293  -> pitches up
        dropped form  n cos(phi) - 1          = 1 - 1     =  0      -> no change
    45 deg is used rather than 60 so the 60 deg flight-path cap does not bind
    and mask the effect.
    """
    jet = _Jet(pitch=45.0)
    jet.control.autopilot.phi_deg = 0.0
    jet.control.autopilot.n = 1.0
    jet.control.set_lift_vector(n_cmd=1.0, phi_cmd_deg=0.0)

    _, pitch_cmd, _ = jet.control.autopilot.advance(0.05)
    assert pitch_cmd > jet.pitch_deg


# ============================================================================
# Sub-stepping
# ============================================================================


def test_control_tick_is_integrated_in_substeps():
    """A one-second tick is broken into steps of at most max_substep_s."""
    autopilot = LiftVectorAutopilot(parent=None, max_substep_s=0.1)
    assert autopilot.substep_count(1.0) == 10
    assert autopilot.substep_count(0.05) == 1
    assert autopilot.substep_count(0.25) == 3


def test_default_substep_gives_four_steps_at_the_campaign_tick():
    """Every campaign config runs tick_secs=1.0; that must sub-divide."""
    autopilot = LiftVectorAutopilot(parent=None)
    assert autopilot.max_substep_s == DEFAULT_MAX_SUBSTEP_S
    assert autopilot.substep_count(1.0) == 4


def test_substepping_flies_an_arc_not_a_chord():
    """Integrating the turn in sub-steps must not land where one big step does.

    A jet holding bank for a full second sweeps an arc. Taking that second as a
    single straight segment places it measurably off the arc; the sub-stepped
    path is the one that curves.
    """
    substepped = _Jet()
    substepped.control.autopilot.phi_deg = 70.0
    substepped.control.autopilot.n = 5.0
    substepped.control.set_lift_vector(n_cmd=5.0, phi_cmd_deg=70.0)
    substepped.control.update_movement(1.0)

    single = _Jet()
    single.control.autopilot.phi_deg = 70.0
    single.control.autopilot.n = 5.0
    single.control.set_lift_vector(n_cmd=5.0, phi_cmd_deg=70.0)
    single.control.autopilot.max_substep_s = 10.0  # forces one step for the tick
    single.control.update_movement(1.0)

    dlat = substepped.position.lat - single.position.lat
    dlon = substepped.position.lon - single.position.lon
    separation_deg = math.hypot(dlat, dlon)
    assert separation_deg > 0.0, "sub-stepping made no difference to the path"


def test_substepping_does_not_change_total_elapsed_time():
    """Sub-steps partition the tick; they must not add or lose simulated time."""
    jet = _Jet()
    jet.control.set_lift_vector(n_cmd=2.0, phi_cmd_deg=20.0)
    before = jet.physics._elapsed_time_s
    jet.control.update_movement(1.0)
    assert jet.physics._elapsed_time_s == pytest.approx(before + 1.0, abs=1e-9)


# ============================================================================
# One rate authority
# ============================================================================


def test_supplied_turn_rate_cap_replaces_the_recomputed_one():
    """update_yaw_deg follows the autopilot's cap instead of its own."""
    jet = _Jet()
    pos = jet.position
    # Ask for a 90 deg heading change in 1 s under a 5 deg/s externally set cap.
    new_yaw, _ = jet.physics.update_yaw_deg(pos, 0.0, 90.0, jet.speed, 1.0, omega_max_deg_s=5.0)
    assert new_yaw == pytest.approx(5.0, abs=1e-6)

    # Without the override the physics-derived (much looser) cap applies.
    unbounded, _ = jet.physics.update_yaw_deg(pos, 0.0, 90.0, jet.speed, 1.0)
    assert unbounded > 5.0


def test_autopilot_command_is_not_limited_twice():
    """The airframe reproduces the autopilot's heading target exactly.

    If the airframe re-limited against its own envelope, the achieved heading
    would fall short of what guidance asked for.
    """
    jet = _Jet()
    jet.control.autopilot.phi_deg = 70.0
    jet.control.autopilot.n = 6.0
    jet.control.set_lift_vector(n_cmd=6.0, phi_cmd_deg=70.0)

    desired_yaw, _, _ = jet.control.autopilot.advance(0.1)
    jet.control.set_lift_vector(n_cmd=6.0, phi_cmd_deg=70.0)

    fresh = _Jet()
    fresh.control.autopilot.phi_deg = 70.0
    fresh.control.autopilot.n = 6.0
    fresh.control.set_lift_vector(n_cmd=6.0, phi_cmd_deg=70.0)
    fresh.control.autopilot.max_substep_s = 0.1
    fresh.control.update_movement(0.1)

    assert fresh.yaw_deg == pytest.approx(desired_yaw, abs=1e-6)


def test_pitch_rate_limit_has_a_single_source():
    """Both consumers read physics.max_pitch_rate_deg_s, not divergent defaults."""
    from bvr_marl_core.rl.environment.spaces.action_space.processors import (
        LiftVectorProcessor,
    )

    jet = _Jet()
    assert hasattr(jet.physics, "max_pitch_rate_deg_s")
    jet.physics.max_pitch_rate_deg_s = 13.0

    proc = LiftVectorProcessor()
    assert proc._get_max_pitch_rate(jet, 1.0, 1.0) == pytest.approx(13.0)

    # The airframe limiter must agree with that same number.
    pitched = jet.physics.update_pitch_deg(0.0, 90.0, jet.speed, 1.0, jet.position)
    assert pitched == pytest.approx(13.0, abs=1e-6)


# ============================================================================
# The legacy path stays untouched
# ============================================================================


def test_autopilot_is_dormant_until_a_lift_vector_is_commanded():
    """Scripted desired-heading callers keep their previous behaviour."""
    jet = _Jet()
    assert jet.control.autopilot.active is False

    jet.control.set_yaw_deg(30.0)
    jet.control.update_movement(1.0)

    # Legacy path: roll is still inferred from the achieved turn rate, and the
    # autopilot never touched the aircraft.
    assert jet.control.autopilot.active is False
    assert jet.control.autopilot.phi_deg == 0.0
    assert jet.yaw_deg != 0.0


def test_reset_returns_the_inner_loop_to_wings_level():
    """Episode reset must not carry a previous episode's bank into a new spawn."""
    jet = _Jet()
    jet.control.set_lift_vector(n_cmd=6.0, phi_cmd_deg=60.0)
    jet.control.update_movement(1.0)
    assert jet.control.autopilot.phi_deg != 0.0

    jet.control.autopilot.reset()
    assert jet.control.autopilot.phi_deg == 0.0
    assert jet.control.autopilot.n == 1.0
    assert jet.control.autopilot.active is False
