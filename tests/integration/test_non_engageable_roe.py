"""Units the scenario protects must survive the sensor-limited path too.

``is_non_engageable`` was enforced only on the oracle fire paths. A track's
``engageable`` flag means "confirmed and not deception" and carries no rules of
engagement, so a sensor-limited agent could select, fire at, and kill an AWACS.

An AWACS is now ``is_sensor_invisible`` as well, which is the stronger protection and the
one that also saves the radar work -- but the two rules are independent, and this file
tests the ROE one. Its subject therefore has to be made visible explicitly; otherwise the
test would pass for the wrong reason (nothing to shoot at) and stop guarding the veto.
"""

from types import SimpleNamespace

import pytest

from bvr_marl_core.domain.tactical_contact import TacticalContact
from bvr_marl_core.missiles.fox3.amraam import AIM120_AMRAAM
from bvr_marl_core.registry import get_aircraft_class
from bvr_marl_core.simulator.core.helpers import Position
from bvr_marl_core.simulator.simulator import Simulator

_MAP = SimpleNamespace(
    left_lon=-5, right_lon=5, bottom_lat=-5, top_lat=5, min_alt=0, max_alt=20_000
)


def _fighter(lat, lon, yaw, group):
    return get_aircraft_class("Eurofighter")(
        Position(lat, lon, 9_000.0), yaw, 280.0, group, _MAP, 0.0, 20_000.0
    )


def _awacs(lat, lon, group):
    return get_aircraft_class("AWACS")(
        Position(lat, lon, 10_000.0), 90.0, 200.0, group, _MAP, 0.0, 20_000.0
    )


def _contacts_for(shooter):
    return [
        TacticalContact.from_track_snapshot(track)
        for track in getattr(shooter.sensor, "sensor_tracks", ()) or ()
        if track.engageable and not track.suspect_deception
    ]


@pytest.mark.slow
def test_sensor_limited_launch_is_vetoed_against_a_protected_unit():
    sim = Simulator(tick_secs=0.5, random_seed=3)
    sim.record_traces = False
    shooter = _fighter(0.0, 0.0, 90.0, "blue")
    awacs = _awacs(0.0, 30_000.0 / 111_000.0, "red")
    assert awacs.is_non_engageable is True
    # This test is about the ROE veto, which only has anything to veto if the target can
    # be SEEN. An AWACS is sensor-invisible by default, so it never becomes a contact and
    # the veto path is unreachable through it. Making this one visible keeps the subject
    # under test -- a protected-but-detectable unit -- and keeps the two rules separate:
    # invisibility is about being seen, ROE about being shot at.
    awacs.is_sensor_invisible = False
    sim.add_unit(shooter)
    sim.add_unit(awacs)

    vetoes = []
    for _ in range(30):
        sim.do_tick()
        for contact in _contacts_for(shooter):
            missile, veto, _ = shooter.weapons.fire_missile_at_contact(sim, contact, AIM120_AMRAAM)
            assert missile is None, "a weapon was committed to a protected unit"
            vetoes.append(veto)

    assert vetoes, "precondition: the AWACS was never held as an engageable contact"
    assert "contact_non_engageable_roe" in vetoes


def test_collision_resolution_never_kills_a_protected_unit():
    """Backstop: even a weapon already attributed to it must not resolve a hit."""
    sim = Simulator(tick_secs=0.5, random_seed=3)
    sim.record_traces = False
    protected = SimpleNamespace(
        id=2,
        is_non_engageable=True,
        is_destroyed=False,
        is_mortally_hit=False,
        position=Position(0.0, 0.0, 9_000.0),
    )
    missile = SimpleNamespace(
        id=10,
        is_missile=True,
        weapon_track=object(),
        position=Position(0.0, 0.0, 9_000.0),
    )
    sim.active_units[2] = protected
    sim.active_units[10] = missile
    sim._weapon_truth_associations[10] = 2

    hits = []
    sim.ccd.on_hit = lambda *args, **kwargs: hits.append(args)
    sim.ccd.post_update_ccd([missile], 0.5, sim)

    assert hits == []
    assert sim.diagnostic_counters["hit_suppressed_non_engageable"] == 1
