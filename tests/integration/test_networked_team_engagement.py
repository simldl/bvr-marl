"""A networked team must resolve its opponents and be able to kill them.

Regression cover for a failure where every sensor-limited shot missed. The shared
network picture fused a whole formation into one track, that track's report lineage
therefore named several aircraft, contact attribution refused to resolve it, so no
weapon-truth association was ever registered -- and collision detection skips a
weapon it cannot attribute. Missiles flew correct intercepts and passed through
their targets for the entire episode.
"""

import math

import pytest

from bvr_marl_core.aircraft.types.eurofighter import Eurofighter
from bvr_marl_core.domain.tactical_contact import TacticalContact
from bvr_marl_core.missiles.fox3.amraam import AIM120_AMRAAM
from bvr_marl_core.simulator.core.helpers import Position
from bvr_marl_core.simulator.simulator import Simulator


class _MapLimits:
    bottom_lat = -5.0
    top_lat = 5.0
    left_lon = -5.0
    right_lon = 5.0
    min_alt = 0.0
    max_alt = 20_000.0


def _fighter(lat, lon, yaw, group):
    return Eurofighter(
        Position(lat, lon, 9_000.0),
        yaw_deg=yaw,
        speed_mps=280.0,
        group=group,
        map_limits=_MapLimits(),
        min_alt_m=0.0,
        max_alt_m=20_000.0,
    )


def _two_v_two(seed, separation_deg=0.05, range_km=30.0):
    sim = Simulator(tick_secs=0.5, random_seed=seed)
    sim.record_traces = False
    east = (range_km * 1_000.0) / (111_000.0 * math.cos(0.0))
    blues = [
        _fighter(separation_deg, 0.0, 90.0, "blue"),
        _fighter(-separation_deg, 0.0, 90.0, "blue"),
    ]
    reds = [
        _fighter(separation_deg, east, 270.0, "red"),
        _fighter(-separation_deg, east, 270.0, "red"),
    ]
    for unit in (*blues, *reds):
        sim.add_unit(unit)
    return sim, blues, reds


def test_networked_team_resolves_each_opponent_as_its_own_contact():
    sim, blues, reds = _two_v_two(seed=11)
    for _ in range(12):
        sim.do_tick()

    for blue in blues:
        engageable = [track for track in blue.radar.cached_tracks or () if track.engageable]
        assert len(engageable) == len(reds), (
            "the shared picture collapsed the opposing formation into one contact"
        )

    attributed = {
        sim.evaluator_truth_id_for_contact(blue.id, track.track_id)
        for blue in blues
        for track in blue.radar.cached_tracks or ()
    }
    assert attributed == {red.id for red in reds}


@pytest.mark.slow
def test_sensor_limited_shots_from_a_networked_team_can_kill():
    sim, blues, reds = _two_v_two(seed=11)
    fired = {}

    for _ in range(400):
        sim.do_tick()
        for blue in blues:
            if blue.id in fired or blue.id not in sim.active_units:
                continue
            engageable = [
                track
                for track in getattr(blue.sensor, "sensor_tracks", ()) or ()
                if track.engageable and not track.suspect_deception
            ]
            if not engageable:
                continue
            contact = TacticalContact.from_track_snapshot(engageable[0])
            missile, _veto, _diagnostics = blue.weapons.fire_missile_at_contact(
                sim, contact, AIM120_AMRAAM
            )
            if missile is not None:
                fired[blue.id] = missile
                assert sim.evaluator_target_for_weapon(missile) is not None, (
                    "an operational shot was launched with no evaluator attribution, "
                    "so collision detection will never run for it"
                )
        if fired and all(missile.id not in sim.active_units for missile in fired.values()):
            break

    assert len(fired) == len(blues)
    assert not sim.diagnostic_counters["missing_evaluator_target"]
    assert not sim.diagnostic_counters["missile_terminal_association_missing"]

    terminal = [
        event.record for event in sim.events if type(event).__name__ == "MissileTerminalEvent"
    ]
    assert len(terminal) == len(fired), "every shot must reach a terminal resolution"
    assert all(record["miss_distance_m"] < 500.0 for record in terminal)
    assert {record["target_id"] for record in terminal} <= {red.id for red in reds}


@pytest.mark.slow
def test_shots_at_a_tight_maneuvering_formation_still_detonate():
    """A close, breaking formation must not defeat weapon attribution.

    When two opponents fly tight enough that the network picture fuses them into one
    track, that track's report lineage names both aircraft, so vote-based attribution
    ties and yields nothing. The fail-closed path then scored every such shot against
    nobody: collision detection was skipped and the missiles flew straight through
    their targets ("fly through, no detonation"). Attribution must instead resolve the
    fused shot to one real aircraft by geometry so it detonates.
    """
    sim, blues, reds = _two_v_two(seed=11, separation_deg=0.02, range_km=26.0)
    fired = {}
    red_ids = {red.id for red in reds}

    for step in range(400):
        # Both opponents break hard once the merge develops -- the classic evasive
        # turn that previously coincided with the attribution failure.
        if step > 40:
            for red in reds:
                if red.id in sim.active_units:
                    red.desired_yaw_deg = 180.0
        sim.do_tick()
        for blue_index, blue in enumerate(blues):
            if blue.id in fired or blue.id not in sim.active_units:
                continue
            engageable = [
                track
                for track in getattr(blue.sensor, "sensor_tracks", ()) or ()
                if track.engageable and not track.suspect_deception
            ]
            if not engageable:
                continue
            # Each shooter picks a distinct track when the picture offers more than one.
            contact = TacticalContact.from_track_snapshot(
                engageable[min(blue_index, len(engageable) - 1)]
            )
            missile, _veto, _diagnostics = blue.weapons.fire_missile_at_contact(
                sim, contact, AIM120_AMRAAM
            )
            if missile is not None:
                fired[blue.id] = missile
                assert sim.evaluator_target_for_weapon(missile) is not None
        if fired and all(missile.id not in sim.active_units for missile in fired.values()):
            break

    assert len(fired) == len(blues)
    # The bug's signature: an operational shot with no attributed victim.
    assert not sim.diagnostic_counters["missing_evaluator_target"]

    terminal = [
        event.record for event in sim.events if type(event).__name__ == "MissileTerminalEvent"
    ]
    assert len(terminal) == len(fired), "every shot must detonate, not fly through"
    # Every detonation must resolve to a real opposing aircraft -- never a friendly,
    # never a phantom, and never nothing (the fly-through the fix removes).
    victims = [record["target_id"] for record in terminal]
    assert victims, "no missile reached a terminal detonation"
    assert set(victims) <= red_ids
