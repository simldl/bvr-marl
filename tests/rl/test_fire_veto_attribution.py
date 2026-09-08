"""`vetoed_missile_wasted` must say WHICH gate rejected the launch.

Collapsed into one counter, `wasted` says only that a permissible trigger press died
on geometry. As measured it sat at 15-50 per episode with zero
launches for 474 iterations: enough to rule out target selection (`no_target` was
exactly 0.000) and nothing else. FOV, range and lock have unrelated fixes -- pointing,
closure, and the sensor chain -- so the split is the difference between "the shots are
being eaten" and knowing what to change.

The three are CONDITIONS, not a partition: a press taken out of FOV and beyond range
increments both. Each subtotal is therefore bounded by `wasted` rather than summing to
it, which is what these tests pin down.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from bvr_marl_core.aircraft.systems import fire_veto
from bvr_marl_core.aircraft.systems.fire_feasibility import FireGates
from bvr_marl_core.aircraft.systems.fire_veto import (
    TEAM_A_WASTED_VETO_KEYS,
    VETO_CATEGORY_FOV,
    VETO_CATEGORY_LOCK,
    VETO_CATEGORY_RANGE,
    WASTED_VETO_CATEGORIES,
    wasted_categories_from_gates,
    wasted_category_key,
)


def _gates(**overrides) -> FireGates:
    """A fully-satisfied gate set, so each test perturbs exactly one condition."""
    defaults = dict(
        inventory_ok=True,
        radar_lock=True,
        datalink_lock=False,
        target_in_fov=True,
        gimbal_ok=True,
        radar_range_ok=True,
        cooldown_ok=True,
        target_not_saturated=True,
    )
    defaults.update(overrides)
    return FireGates(**defaults)


def test_the_three_gates_are_reported_independently():
    assert wasted_categories_from_gates(_gates(target_in_fov=False)) == {
        VETO_CATEGORY_FOV: 1,
        VETO_CATEGORY_RANGE: 0,
        VETO_CATEGORY_LOCK: 0,
    }
    assert wasted_categories_from_gates(_gates(radar_range_ok=False)) == {
        VETO_CATEGORY_FOV: 0,
        VETO_CATEGORY_RANGE: 1,
        VETO_CATEGORY_LOCK: 0,
    }
    assert wasted_categories_from_gates(_gates(radar_lock=False)) == {
        VETO_CATEGORY_FOV: 0,
        VETO_CATEGORY_RANGE: 0,
        VETO_CATEGORY_LOCK: 1,
    }


def test_simultaneous_failures_are_all_reported():
    """A partition would have to pick a winner and hide the other gate.

    Fixing the reported gate would then move the number nowhere, which is the exact
    debugging loop this split exists to break.
    """
    categories = wasted_categories_from_gates(_gates(target_in_fov=False, radar_range_ok=False))

    assert categories[VETO_CATEGORY_FOV] == 1
    assert categories[VETO_CATEGORY_RANGE] == 1


def test_the_gimbal_limit_counts_as_a_pointing_failure():
    """Both `outside FOV` and `outside gimbal` are fixed by pointing differently."""
    assert wasted_categories_from_gates(_gates(gimbal_ok=False))[VETO_CATEGORY_FOV] == 1


def test_a_datalink_cued_shot_is_not_counted_as_out_of_range():
    """That shot is legal beyond own radar range -- counting it as `range` would

    send the reader after a closure problem that does not exist."""
    categories = wasted_categories_from_gates(_gates(radar_range_ok=False, datalink_lock=True))

    assert categories[VETO_CATEGORY_RANGE] == 0
    assert categories[VETO_CATEGORY_LOCK] == 0


def test_missing_gates_yield_zeros_rather_than_raising():
    """An attribution failure must never drop the press from the collapsed counter."""
    assert wasted_categories_from_gates(None) == {c: 0 for c in WASTED_VETO_CATEGORIES}


def test_subtotals_are_bounded_by_the_collapsed_wasted_counter():
    from bvr_marl_core.rl.environment.gym.gym_components.state_tracker import StateTracker

    tracker = StateTracker()
    tracker.initialize_agent("agent_0")

    # Two presses: one out of FOV only, one out of FOV and beyond range.
    for gates in (_gates(target_in_fov=False), _gates(target_in_fov=False, radar_range_ok=False)):
        tracker.update_diagnostic_metrics(
            "agent_0",
            valid_shots=0,
            vetoed_shots=1,
            lock_ok=False,
            fov_ok=False,
            wasted_shots=1,
            wasted_shots_by_category=wasted_categories_from_gates(gates),
        )

    per_gate = tracker.episode_vetoed_missile_wasted_by_category["agent_0"]
    wasted = tracker.episode_vetoed_missile_wasted["agent_0"]

    assert wasted == 2
    assert per_gate[VETO_CATEGORY_FOV] == 2
    assert per_gate[VETO_CATEGORY_RANGE] == 1
    assert per_gate[VETO_CATEGORY_LOCK] == 0
    assert all(count <= wasted for count in per_gate.values())


def test_per_gate_counts_do_not_leak_across_episodes():
    """The collapsed `no_target` counter shipped with exactly this bug once."""
    from bvr_marl_core.rl.environment.gym.gym_components.state_tracker import StateTracker

    tracker = StateTracker()
    tracker.initialize_agent("agent_0")
    tracker.update_diagnostic_metrics(
        "agent_0",
        valid_shots=0,
        vetoed_shots=1,
        lock_ok=False,
        fov_ok=False,
        wasted_shots=1,
        wasted_shots_by_category=wasted_categories_from_gates(_gates(target_in_fov=False)),
    )

    tracker.reset()

    assert tracker.episode_vetoed_missile_wasted_by_category == {}


def test_the_per_step_state_keys_are_initialised_and_reset():
    from bvr_marl_core.rl.environment.spaces.action_space.utils.debug_info import (
        DebugInfoCollector,
    )

    state = DebugInfoCollector().init_training_signals()
    for category in WASTED_VETO_CATEGORIES:
        assert state[wasted_category_key(category)] == 0

    state[wasted_category_key(VETO_CATEGORY_FOV)] = 1
    DebugInfoCollector().reset_step_counters(state)
    assert state[wasted_category_key(VETO_CATEGORY_FOV)] == 0


def test_training_signals_expose_every_gate():
    from bvr_marl_core.rl.environment.spaces.action_space.utils.debug_info import (
        DebugInfoCollector,
    )

    collector = DebugInfoCollector()
    state = collector.init_training_signals()
    state[wasted_category_key(VETO_CATEGORY_LOCK)] = 1

    signals = collector.get_training_signals(state)

    assert signals[fire_veto.wasted_signal_key(VETO_CATEGORY_LOCK)] == 1
    assert signals[fire_veto.wasted_signal_key(VETO_CATEGORY_FOV)] == 0


def test_the_firing_handler_attributes_the_gates_that_were_unmet():
    """The hop from the step's gate evaluation to the per-step state key.

    This is ADDING_A_METRIC.md's still-open trap 3: everything downstream can be
    correct while this link quietly never runs, and it presents as a zero column.
    """
    from bvr_marl_core.rl.environment.spaces.action_space.weapon_firing import (
        WeaponFiringHandler,
    )

    class _Weapons:
        def fire_missile(self, sim, target, missile_cls):
            return None, "not_in_fov(target=bandit-1)", {"has_lock": False, "in_fov": False}

    class _Unit:
        id = "A0"
        missile_types = [object]

        def __init__(self):
            self.weapons = _Weapons()

    handler = WeaponFiringHandler.__new__(WeaponFiringHandler)
    handler.missile_auto = None
    handler.simulator = None

    state: dict = {}
    handler._fire_missile(
        _Unit(), object(), state, "A0", _gates(target_in_fov=False, radar_lock=False)
    )

    assert state["vetoed_missile_wasted_this_step"] == 1
    assert state[wasted_category_key(VETO_CATEGORY_FOV)] == 1
    assert state[wasted_category_key(VETO_CATEGORY_LOCK)] == 1
    assert state[wasted_category_key(VETO_CATEGORY_RANGE)] == 0


def test_the_per_gate_totals_reach_the_episode_info_dict():
    """ADDING_A_METRIC.md step 2: the key has to actually land in `info`."""
    from bvr_marl_core.rl.environment.gym.gym_components.state_tracker import StateTracker
    from bvr_marl_core.rl.environment.gym.gym_components.termination import TerminationChecker

    tracker = StateTracker()
    tracker.initialize_agent("A0")
    tracker.initialize_agent("B0")
    tracker.update_diagnostic_metrics(
        "A0",
        valid_shots=0,
        vetoed_shots=1,
        lock_ok=False,
        fov_ok=False,
        wasted_shots=1,
        wasted_shots_by_category=wasted_categories_from_gates(_gates(radar_range_ok=False)),
    )

    checker = TerminationChecker(simulator=SimpleNamespace(active_units={}), config=None)
    checker.config = SimpleNamespace(agent_ids=["A0"], opponent_ids=["B0"])

    info = checker.compute_episode_info(
        end_reasons=[],
        agents_alive=True,
        opponents_alive=True,
        current_step=1,
        state_tracker=tracker,
        agent_to_unit_id={},
    )

    assert info["team_a_vetoed_missile_wasted_range"] == 1
    assert info["team_a_vetoed_missile_wasted_fov"] == 0
    assert info["team_b_vetoed_missile_wasted_range"] == 0
    assert all(info[key] <= info["team_a_vetoed_missile_wasted"] for key in TEAM_A_WASTED_VETO_KEYS)


def test_exported_header_keys_cover_every_gate():
    """The behavior callback reserves exactly these columns (ADDING_A_METRIC.md trap 2)."""
    assert WASTED_VETO_CATEGORIES == ("fov", "range", "lock")
    assert len(TEAM_A_WASTED_VETO_KEYS) == len(WASTED_VETO_CATEGORIES)
    for category in WASTED_VETO_CATEGORIES:
        assert f"team_a_vetoed_missile_wasted_{category}" in TEAM_A_WASTED_VETO_KEYS


def test_the_range_gate_uses_the_missile_envelope_not_just_radar_range():
    """A target inside radar range but beyond the missile's reach is out of range.

    A measured run held `wasted_range` at exactly 0.00 for 307 iterations while the agent
    stood off at ~75 km: the gate was radar range alone (95% of rated), which a
    long-range radar satisfies from far outside any viable launch.
    """
    assert _gates(radar_range_ok=True, weapon_range_ok=False).launch_range_ok is False
    assert wasted_categories_from_gates(_gates(weapon_range_ok=False))[VETO_CATEGORY_RANGE] == 1


def test_a_datalink_lock_relaxes_radar_range_but_never_missile_range():
    """Someone else can hold the track; nobody can extend the missile's reach."""
    beyond_radar = _gates(radar_range_ok=False, datalink_lock=True, weapon_range_ok=True)
    beyond_missile = _gates(radar_range_ok=False, datalink_lock=True, weapon_range_ok=False)

    assert beyond_radar.launch_range_ok is True
    assert beyond_missile.launch_range_ok is False


def test_an_unscorable_envelope_does_not_silently_block_the_shot():
    """No WEZ -> permissive. A DLZ hiccup must not become an invisible no-fire."""
    from bvr_marl_core.aircraft.systems.fire_feasibility import _weapon_range_ok

    class _NoWez:
        wez = None

    assert _weapon_range_ok(_NoWez(), object(), 50_000.0) is True


def test_the_weapon_range_gate_reads_the_dlz_ceiling():
    from bvr_marl_core.aircraft.systems.fire_feasibility import _weapon_range_ok

    class _Dlz:
        r_pi_m = 40_000.0

    class _Wez:
        def compute_dlz(self, target):
            return _Dlz()

    class _Shooter:
        wez = _Wez()

    inside = _Shooter()
    outside = _Shooter()
    assert _weapon_range_ok(inside, object(), 30_000.0) is True
    assert _weapon_range_ok(outside, object(), 75_000.0) is False


def test_the_energy_model_uses_the_airframes_own_induced_drag_factor():
    """The Ps model must not fly a different aeroplane than the integrator.

    Production `AircraftPhysics` stores the induced-drag factor as `K_ind` and
    exposes neither `oswald_e` nor `aspect_ratio`, so every real aircraft fell
    through to the 0.05 "absent or mocked" fallback -- a generic high-aspect-ratio
    value, against a Eurofighter's actual 0.1617. Since the term enters as k*CL**2 it
    is negligible at 1 g and dominant in a hard turn, which is where it was found:
    the Ps model reported +35 to +71 m/s available while the aircraft realised -22 to
    -120 and bled 250 -> 84 m/s.
    """
    from bvr_marl_core.physics.constraints.envelope_calculator import EnvelopeCalculator

    class _Physics:
        K_ind = 0.1617

    class _Unit:
        physics = _Physics()

    assert EnvelopeCalculator().compute_induced_drag_factor(_Unit()) == 0.1617


def test_induced_drag_falls_back_only_when_the_airframe_says_nothing():
    from bvr_marl_core.physics.constraints.envelope_calculator import EnvelopeCalculator

    calc = EnvelopeCalculator()

    class _Geometry:
        oswald_e = 0.82
        aspect_ratio = 7.5

    class _GeometryUnit:
        physics = _Geometry()

    class _Empty:
        pass

    class _EmptyUnit:
        physics = _Empty()

    # No K_ind but real geometry -> derive it.
    assert calc.compute_induced_drag_factor(_GeometryUnit()) == pytest.approx(
        1.0 / (math.pi * 0.82 * 7.5)
    )
    # Nothing at all (mocks) -> the documented constant.
    assert calc.compute_induced_drag_factor(_EmptyUnit()) == 0.05


def test_per_agent_action_state_does_not_survive_an_episode():
    """The contact-slot registry must not carry ghosts into the next episode.

    `ContactSlotRegistry` expires a coasting contact when
    `now - last_seen > coast_timeout`, where `now` is `sim.elapsed_time_s`. That
    clock RESETS to 0 each episode, so a contact last seen at t=700 is evaluated as
    `0 - 700 = -700` and never expires. Measured on a REUSED env instance with a
    trained checkpoint, occupied slots grew 1.00 -> 1.91 -> 2.06 -> 3.04 -> 4.57
    across successive episodes while the radar held ~1.3 live tracks, and because
    the target axis bins over OCCUPIED slots a policy emitting a fixed 0.6 then
    designates a stale identity the radar cannot have locked. lock_rate collapsed
    0.950 -> 0.279 -> 0.025 in lockstep -- the exact signature seen in training, and
    invisible to any single-episode probe.
    """
    from bvr_marl_core.rl.environment.spaces.action_space.base_processor import (
        ActionProcessorBase,
    )

    processor = ActionProcessorBase.__new__(ActionProcessorBase)
    processor.agent_states = {0: {"contact_slot_registry": object(), "v_bar": 250.0}}

    processor.reset()

    assert processor.agent_states == {}


def test_a_registry_whose_clock_went_backwards_drops_its_ghosts():
    """Defence in depth for any caller that reuses a registry across episodes."""
    from bvr_marl_core.domain.tactical_contact import ContactSlotRegistry

    class _Contact:
        """Minimal stand-in: the registry only reads these four fields."""

        def __init__(self, track_id):
            self.track_id = track_id
            self.engageable = True
            self.suspect_deception = False
            self.is_missile = False

    def _contact(track_id):
        return _Contact(track_id)

    registry = ContactSlotRegistry(max_slots=8, coast_timeout_s=10.0)
    registry.update([_contact("old-1"), _contact("old-2")], time_s=700.0)
    assert sum(c is not None for c in registry.update([], time_s=701.0)) == 2

    # New episode: the clock restarts, so the previous identities must go.
    slots = registry.update([_contact("new-1")], time_s=0.0)

    occupied = [c for c in slots if c is not None]
    assert len(occupied) == 1
    assert occupied[0].track_id == "new-1"


def test_seeker_tracks_are_reassociated_to_the_cued_target_in_the_endgame():
    """`track_only` must not discard the missile's OWN seeker.

    The policy exists to stop a weapon wandering onto a different target, and it
    enforced that by identity: `track.track_id == designated_id`. But the seeker
    numbers its tracks in its own namespace, so on acquisition it reports a
    different id than the contact the weapon was cued against, and the filter threw
    away the only sensor that could still see the target.

    Traced on a stationary-target intercept: the seeker held designated id 1000001 out to
    6.7 km, flipped to its own id 8 at 5.3 km, and the provider committed nothing
    after that. At a 206 m closest approach the shot scored as 5.75 s stale and
    unlocked, giving P_trk 0.349 and Pk 0.277 on what was a good intercept.
    """
    from bvr_marl_core.missiles.guidance.target_provider import GuidanceTargetProvider
    from bvr_marl_core.simulator.core.units import Position

    provider = GuidanceTargetProvider.__new__(GuidanceTargetProvider)
    provider.last_confirmed_target_pos = Position(lat=0.0, lon=0.0, alt=10_000.0)

    class _Missile:
        position = Position(lat=0.0, lon=0.0, alt=10_000.0)

    provider.missile = _Missile()

    class _Track:
        def __init__(self, track_id, north_m):
            self.track_id = track_id
            self.reference_frame = None
            # ENU east/north/up relative to the reference position.
            self.state = (0.0, float(north_m), 0.0, 0.0, 0.0, 0.0)

    near = _Track("seeker-local-8", 300.0)
    far = _Track("someone-else", 40_000.0)

    matched = provider._reassociate_seeker_tracks([far, near])
    assert matched == [near], "the track in the predicted basket is the cued target"

    # A weapon that has genuinely lost its target must coast, not grab whatever it
    # can see: everything outside the gate is still rejected.
    assert provider._reassociate_seeker_tracks([far]) == []
