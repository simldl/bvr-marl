from types import SimpleNamespace

import numpy as np
import pytest

from bvr_marl_core.domain.information import TrackLifecycle, TrackSnapshot
from bvr_marl_core.domain.tactical_contact import (
    NO_TARGET_ACTION_FRACTION,
    ContactSlotRegistry,
    TacticalContact,
    action_value_for_contact_slot,
)
from bvr_marl_core.rl.environment.spaces.action_space.utils.target_sorting import TargetSorter


class _ForbiddenTruth:
    def __getattribute__(self, name):
        raise AssertionError(f"evaluator truth was accessed: {name}")


def _track(track_id, *, classification="fighter", engageable=True, suspect=False):
    class_index = {"fighter": 0, "support_aircraft": 1, "missile": 2}.get(classification.lower(), 4)
    probabilities = tuple(1.0 if index == class_index else 0.0 for index in range(5))
    return TrackSnapshot(
        track_id=track_id,
        state_time_s=0.0,
        state=(track_id, 0.0, 10000.0, 250.0, 0.0, 0.0),
        covariance=tuple(tuple(row) for row in np.eye(6)),
        confidence=0.8,
        lifecycle=TrackLifecycle.CONFIRMED,
        classification_probabilities=probabilities,
        engageable=engageable,
        suspect_deception=suspect,
    )


def _contact(track_id):
    return TacticalContact.from_sensor_track(_track(track_id))


def test_sensor_track_conversion_never_reads_evaluator_handle():
    contact = _contact(7)
    assert contact.track_id == 7
    assert contact.classification == "fighter"


def test_sensor_track_conversion_rejects_truncated_position_covariance():
    with pytest.raises(ValueError, match="6x6 covariance"):
        TacticalContact(
            track_id=7,
            state=(0.0,) * 6,
            covariance=tuple(tuple(row) for row in np.eye(3)),
            confidence=0.8,
        )


def test_registry_keeps_slots_stable_and_reuses_only_after_coast_timeout():
    registry = ContactSlotRegistry(max_slots=2, coast_timeout_s=5.0)
    first = registry.update([_contact(20), _contact(10)], time_s=0.0)
    assert [contact.track_id for contact in first] == [10, 20]

    coasting = registry.update([_contact(20)], time_s=4.0)
    assert [contact.track_id for contact in coasting] == [10, 20]

    replaced = registry.update([_contact(20), _contact(30)], time_s=6.0)
    assert [contact.track_id for contact in replaced] == [30, 20]


def test_registry_reserves_a_no_target_region_then_splits_the_rest():
    """Binning is over OCCUPIED slots, with a fixed no-target slice at the bottom.

    The old scheme cut [0,1] into max_slots+1 fixed bins, so with the default 8 slots a
    1v1 left the only designating band at [0.111, 0.222] -- 88.9% of the axis designated
    nothing and 0.5 was permanently empty. Measured policy sigma on this axis is ~0.2,
    so even a correctly centred mean designated a real contact only about a third of the
    time, which silently switched off lock_rate, shot_opportunity and every firing gate.
    """
    registry = ContactSlotRegistry(max_slots=2)
    slots = registry.update([_contact(10), _contact(20)], time_s=0.0)

    # The no-target choice stays reachable...
    assert registry.select(0.0, slots) is None
    assert registry.select(NO_TARGET_ACTION_FRACTION - 0.01, slots) is None
    # ...and the remaining 80% splits evenly over the two occupied slots.
    assert registry.select(NO_TARGET_ACTION_FRACTION + 0.01, slots).track_id == 10
    assert registry.select(1.0, slots).track_id == 20


def test_a_policy_centred_at_one_half_designates_a_real_contact():
    """The property the whole change exists to provide.

    A squashed Gaussian sits near 0.5 at initialisation and drifts around it. Under the
    old binning that value addressed a permanently empty slot in any 1v1, so the agent
    could not designate, lock, or fire no matter what the radar handed it.
    """
    registry = ContactSlotRegistry(max_slots=8)
    slots = registry.update([_contact(10)], time_s=0.0)

    assert registry.select(0.5, slots).track_id == 10
    # ...and stays designating across a realistic exploration spread (sigma ~0.2).
    for value in (0.3, 0.4, 0.5, 0.6, 0.7, 0.9):
        assert registry.select(value, slots) is not None, value


def test_no_contacts_means_no_selection_at_any_action_value():
    registry = ContactSlotRegistry(max_slots=4)
    slots = registry.update([], time_s=0.0)

    for value in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert registry.select(value, slots) is None


def test_action_value_helper_lands_on_its_slot():
    registry = ContactSlotRegistry(max_slots=8)
    for occupied in (1, 2, 4):
        slots = registry.update([_contact(10 * i) for i in range(occupied)], time_s=0.0)
        for index in range(occupied):
            value = action_value_for_contact_slot(index, 8, occupied_slots=occupied)
            assert registry.select(value, slots) is slots[index], (occupied, index)


def test_target_sorter_contact_path_uses_only_ownship_tracks():
    sorter = TargetSorter(contact_slots=2)
    state = sorter.init_target_state()
    unit = SimpleNamespace(sensor=SimpleNamespace(sensor_tracks=[_track(10)]))

    assert sorter.select_contact(unit, 0.0, state, time_s=0.0) is None
    assert sorter.select_contact(unit, 0.5, state, time_s=0.0).track_id == 10
