"""Support assets are invisible to hostile sensors, not merely protected from fire.

`is_non_engageable` only refuses the launch, and a refused launch still costs everything
upstream of it: the unit is swept, its RCS and SNR are computed, it becomes a detection, a
track, a contact, and it OCCUPIES A TARGET SLOT.

Measured over 10 matched pairs (same weights, same spawn, only AWACS RCS differing),
blinding the AWACS raised real shot opportunities **1.75x** and did so in **10 of 10**
episodes -- while the selector addressed one directly on only **2.6%** of steps. The damage
was slot occupancy, not mis-selection, so refusing the shot could never have fixed it.

Dropping them at enumeration also measured **~20% faster ticks** on the 4v4 stage
(102.4 -> 82.0 ms/tick, radar candidates per call 9.96 -> 8.01).
"""

from __future__ import annotations

from types import SimpleNamespace

from bvr_marl_core.radar.units.aircraft import radar_target_candidates


def _sim(*units):
    return SimpleNamespace(active_units={u.id: u for u in units})


def _u(uid, group, **kw):
    return SimpleNamespace(id=uid, group=group, **kw)


def test_sensor_invisible_units_are_dropped_at_enumeration():
    """Not filtered from detections later -- never enumerated in the first place.

    That is what makes it a performance fix as well as a behavioural one: everything
    downstream of this list is per-candidate work.
    """
    me = _u(1, "blue")
    bandit = _u(2, "red")
    awacs = _u(3, "red", is_sensor_invisible=True)
    got = radar_target_candidates(_sim(me, bandit, awacs), me)
    assert [u.id for u in got] == [2]


def test_friendlies_and_self_are_still_excluded():
    me = _u(1, "blue")
    wingman = _u(2, "blue")
    bandit = _u(3, "red")
    got = radar_target_candidates(_sim(me, wingman, bandit), me)
    assert [u.id for u in got] == [3]


def test_a_visible_but_protected_unit_is_still_swept():
    """The two rules are independent: ROE is about being shot at, not about being seen."""
    me = _u(1, "blue")
    protected = _u(2, "red", is_non_engageable=True, is_sensor_invisible=False)
    got = radar_target_candidates(_sim(me, protected), me)
    assert [u.id for u in got] == [2], "ROE must not silently imply invisibility"


def test_missing_flag_defaults_to_visible():
    """A unit that never heard of the flag keeps the old behaviour."""
    me = _u(1, "blue")
    plain = _u(2, "red")
    assert [u.id for u in radar_target_candidates(_sim(me, plain), me)] == [2]


def test_awacs_is_sensor_invisible_by_default():
    from bvr_marl_core.registry import get_aircraft_class
    from bvr_marl_core.simulator.core.helpers import Position

    limits = SimpleNamespace(
        left_lon=-5, right_lon=5, bottom_lat=-5, top_lat=5, min_alt=0, max_alt=20_000
    )
    awacs = get_aircraft_class("AWACS")(
        Position(0.0, 0.0, 10_000.0), 90.0, 200.0, "red", limits, 0.0, 20_000.0
    )
    assert awacs.is_sensor_invisible is True
    # Still protected by ROE as well; invisibility did not replace that rule.
    assert awacs.is_non_engageable is True


def test_a_fighter_is_visible():
    from bvr_marl_core.registry import get_aircraft_class
    from bvr_marl_core.simulator.core.helpers import Position

    limits = SimpleNamespace(
        left_lon=-5, right_lon=5, bottom_lat=-5, top_lat=5, min_alt=0, max_alt=20_000
    )
    jet = get_aircraft_class("Eurofighter")(
        Position(0.0, 0.0, 9_000.0), 90.0, 280.0, "red", limits, 0.0, 20_000.0
    )
    assert getattr(jet, "is_sensor_invisible", False) is False


def test_invisibility_is_not_an_rcs_trick():
    """The platform stays physically large; it is excluded, not faked small.

    Zeroing RCS would be a different (and wrong) model: it would still cost the whole
    per-candidate sensor chain, would still occupy a target slot once a false alarm or a
    datalink track appeared, and would make a 150-tonne airframe undetectable to
    *friendly* sensors too. The exclusion is at enumeration and hostile-only.
    """
    from bvr_marl_core.registry import get_aircraft_class

    cfg = get_aircraft_class("AWACS").Config()
    assert cfg.rcs == 100.0, "AWACS must keep a large RCS; invisibility is not rcs=0"


def test_sensor_invisible_is_configurable_per_scenario():
    """`awacs_sensor_invisible: false` must produce a visible AWACS."""
    from bvr_marl_core.rl.environment.gym.gym_components.config import AWACSConfigData

    assert AWACSConfigData().awacs_sensor_invisible is True
    assert AWACSConfigData(awacs_sensor_invisible=False).awacs_sensor_invisible is False


def test_scenario_yaml_can_turn_invisibility_off():
    from bvr_marl_core.rl.environment.gym.gym_components.config import (
        AWACSConfigData,
    )

    # Mirrors the dict shape `awacs_cfg` is read from in config.py.
    awacs_cfg = {"awacs_sensor_invisible": False}
    built = AWACSConfigData(
        awacs_sensor_invisible=bool(awacs_cfg.get("awacs_sensor_invisible", True))
    )
    assert built.awacs_sensor_invisible is False


def test_the_rule_is_hostile_only():
    """A team must never be blinded to its OWN support asset."""
    from bvr_marl_core.domain.sensing_visibility import is_sensor_invisible_to

    friendly_awacs = _u(9, "blue", is_sensor_invisible=True)
    hostile_awacs = _u(10, "red", is_sensor_invisible=True)
    assert is_sensor_invisible_to(friendly_awacs, "blue") is False
    assert is_sensor_invisible_to(hostile_awacs, "blue") is True


def test_sensible_hostiles_filters_group_and_invisibility():
    from bvr_marl_core.domain.sensing_visibility import sensible_hostiles

    units = [
        _u(1, "blue"),
        _u(2, "blue", is_sensor_invisible=True),
        _u(3, "red"),
        _u(4, "red", is_sensor_invisible=True),
    ]
    assert [u.id for u in sensible_hostiles(units, "blue")] == [3]


def test_rwr_and_seeker_paths_share_the_radar_definition():
    """All four sensing paths must agree, or an "invisible" unit reappears in one.

    Before this, the RWR/passive branch enumerated units independently: an AWACS is a
    250 km emitter and one-way RWR range is a multiple of that, so an AWACS hidden from
    radar was still the single loudest thing on every RWR.
    """
    from bvr_marl_core.aircraft.systems import sensor as sensor_mod
    from bvr_marl_core.domain import sensing_visibility
    from bvr_marl_core.missiles import missile as missile_mod
    from bvr_marl_core.missiles.guidance import target_provider
    from bvr_marl_core.radar.units import aircraft as radar_units

    assert sensor_mod.is_sensor_invisible_to is sensing_visibility.is_sensor_invisible_to
    assert missile_mod.is_sensor_invisible_to is sensing_visibility.is_sensor_invisible_to
    assert target_provider.is_sensor_invisible_to is sensing_visibility.is_sensor_invisible_to
    assert radar_units.sensible_hostiles is sensing_visibility.sensible_hostiles
