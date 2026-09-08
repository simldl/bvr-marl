"""A weapon fired at an ambiguous contact must stay resolvable once it clears up."""

from types import SimpleNamespace

from bvr_marl_core.simulator.core.helpers import Position
from bvr_marl_core.simulator.simulator import Simulator


def _sim_with_units(*unit_ids):
    sim = Simulator(random_seed=7)
    for unit_id in unit_ids:
        sim.active_units[unit_id] = SimpleNamespace(id=unit_id)
    return sim


def test_weapon_is_attributed_once_the_picture_disambiguates_after_launch():
    """Attribution can be unavailable at launch and arrive later.

    Registering only at launch left such a weapon permanently unattributed, and an
    unattributed weapon is skipped by collision detection -- it flies a correct
    trajectory and can never hit anything.
    """
    sim = _sim_with_units(1, 2, 10)
    missile = SimpleNamespace(id=10)

    sim.register_weapon_contact_association(missile.id, sensor_id=1, contact_id="track-a")
    assert sim.evaluator_target_for_weapon(missile) is None

    sim.register_sensor_report_truth_association(1, 5, 2)
    sim.refresh_contact_truth_associations(1, {"track-a": ((1, 5),)})

    assert sim.evaluator_target_for_weapon(missile) is sim.active_units[2]


def test_resolved_weapon_attribution_does_not_drift_afterwards():
    """A weapon already in flight keeps the aircraft it was first resolved against."""
    sim = _sim_with_units(1, 2, 3, 10)
    missile = SimpleNamespace(id=10)
    sim.register_weapon_contact_association(missile.id, sensor_id=1, contact_id="track-a")

    sim.register_sensor_report_truth_association(1, 5, 2)
    sim.refresh_contact_truth_associations(1, {"track-a": ((1, 5),)})
    assert sim.evaluator_target_for_weapon(missile) is sim.active_units[2]

    # The contact later re-associates onto a different aircraft.
    sim.register_sensor_report_truth_association(1, 6, 3)
    sim.register_sensor_report_truth_association(1, 7, 3)
    sim.refresh_contact_truth_associations(1, {"track-a": ((1, 6), (1, 7))})

    assert sim.evaluator_target_for_weapon(missile) is sim.active_units[2]


def test_launch_time_truth_association_still_wins():
    sim = _sim_with_units(1, 2, 3, 10)
    missile = SimpleNamespace(id=10)
    sim.register_weapon_truth_association(missile.id, 2)
    sim.register_weapon_contact_association(missile.id, sensor_id=1, contact_id="track-a")

    sim.register_sensor_report_truth_association(1, 5, 3)
    sim.refresh_contact_truth_associations(1, {"track-a": ((1, 5),)})

    assert sim.evaluator_target_for_weapon(missile) is sim.active_units[2]


def test_contact_association_is_dropped_with_the_weapon():
    sim = _sim_with_units(1, 2, 10)
    missile = SimpleNamespace(id=10)
    sim.register_weapon_contact_association(missile.id, sensor_id=1, contact_id="track-a")

    sim.remove_unit(10)

    assert 10 not in sim._weapon_contact_associations


def test_unattributable_weapon_resolves_to_nothing_rather_than_a_wrong_unit():
    sim = _sim_with_units(1, 2, 10)
    missile = SimpleNamespace(id=10)
    sim.register_weapon_contact_association(missile.id, sensor_id=1, contact_id="track-a")

    # Evenly split evidence: the contact stays ambiguous.
    sim.register_sensor_report_truth_association(1, 5, 2)
    sim.register_sensor_report_truth_association(1, 6, 99)
    sim.refresh_contact_truth_associations(1, {"track-a": ((1, 5), (1, 6))})

    assert sim.evaluator_target_for_weapon(missile) is None


def test_ambiguous_launch_resolves_from_current_lineage_after_contact_reissue():
    sim = _sim_with_units(1, 2, 3, 10)
    sim.register_sensor_report_truth_association(1, 5, 2)
    sim.register_sensor_report_truth_association(1, 6, 3)
    sim.register_weapon_contact_association(
        10,
        sensor_id=1,
        contact_id="track-a",
        report_lineage=((1, 5), (1, 6)),
    )
    sim.refresh_contact_truth_associations(1, {"track-a": ((1, 5), (1, 6))})
    missile = SimpleNamespace(
        id=10,
        weapon_track=SimpleNamespace(snapshot=SimpleNamespace(report_lineage=((1, 5), (1, 6)))),
    )
    # The launch lineage ties between units 2 and 3, so vote-based attribution cannot
    # name one. A real weapon in flight no longer fails closed (which wasted the shot
    # and flew it through its target); the geometric fallback resolves it to one of
    # the two live aircraft its own lineage was built from -- never a non-candidate.
    ambiguous = sim.evaluator_target_for_weapon(missile)
    assert ambiguous in (sim.active_units[2], sim.active_units[3])

    sim.register_sensor_report_truth_association(1, 7, 2)
    sim.refresh_contact_truth_associations(1, {"track-b": ((1, 7),)})
    missile.weapon_track.snapshot.report_lineage = ((1, 7),)

    assert sim.evaluator_target_for_weapon(missile) is sim.active_units[2]


def test_fused_track_majority_does_not_latch_wrong_formation_member():
    """Guidance geometry must beat a transient report majority before terminal.

    This is the live-view failure mode that an exact-tie fallback does not cover:
    the fused track has enough reports to vote for aircraft 2, but its continuing
    operational hypothesis is following aircraft 3. Collision detection must check
    aircraft 3 or the missile will pass through it.
    """
    sim = Simulator(random_seed=7)
    missile = SimpleNamespace(
        id=10,
        group="blue",
        position=Position(0.0, 0.0, 8_000.0),
        weapon_track=SimpleNamespace(
            snapshot=SimpleNamespace(report_lineage=((1, 5), (1, 6), (1, 7)))
        ),
        target_provider=SimpleNamespace(get_guidance_target=lambda: Position(0.0, 0.10, 8_000.0)),
    )
    majority = SimpleNamespace(
        id=2,
        group="red",
        position=Position(0.05, 0.10, 8_000.0),
    )
    followed = SimpleNamespace(
        id=3,
        group="red",
        position=Position(0.0, 0.10, 8_000.0),
    )
    sim.active_units = {2: majority, 3: followed, 10: missile}
    sim.register_sensor_report_truth_association(1, 5, 2)
    sim.register_sensor_report_truth_association(1, 6, 2)
    sim.register_sensor_report_truth_association(1, 7, 3)
    sim.register_weapon_contact_association(
        missile.id,
        sensor_id=1,
        contact_id="fused-track",
        report_lineage=((1, 5), (1, 6), (1, 7)),
    )

    assert sim._dominant_truth_id(missile.weapon_track.snapshot.report_lineage) == 2
    assert sim.evaluator_target_for_weapon(missile) is followed
    assert missile.id not in sim._weapon_truth_associations
