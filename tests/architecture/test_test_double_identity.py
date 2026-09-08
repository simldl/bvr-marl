"""Test doubles must not be able to fabricate weapon identity.

``unittest.mock.Mock`` answers every attribute access with a truthy Mock. Production
code decides what a weapon *is* from exactly such accesses -- does it carry a weapon
track, what contact was it committed to, which weapons does this aircraft have in the
air. A bare Mock therefore reads as an operational weapon carrying a launch contact,
so tests silently exercise the wrong branch and report failures that say nothing
about the code under test.

These tests pin the behaviour of the ``tests.mocks.identity`` helpers that exist to
prevent that, so the trap cannot quietly reopen.
"""

from unittest.mock import Mock

from bvr_marl_core.rl.environment.spaces.action_space.automation import MissileAutomation
from bvr_marl_core.simulator.simulator import Simulator
from tests.mocks.identity import declare_unit_identity, declare_weapon_identity


def test_identity_probes_do_not_accept_a_fabricated_answer():
    """A weapon double that declares nothing must classify as nothing.

    ``_missile_target_id`` decides the saturation namespace from what the weapon says
    it is. It types-checks rather than truth-checks precisely so a Mock's fabricated
    ``weapon_track`` cannot promote a legacy weapon into contact space.
    """
    assert Simulator._missile_target_id(Mock()) is None


def test_a_declared_weapon_double_is_classified_in_unit_space():
    weapon = declare_weapon_identity(Mock(), designated_target_id=42)

    assert Simulator._missile_target_id(weapon) == ("unit", 42)


def test_a_declared_weapon_double_reports_no_weapon_track():
    weapon = declare_weapon_identity(Mock())

    assert weapon.weapon_track is None
    assert weapon.launch_contact_id is None


def test_a_declared_unit_double_owns_a_real_weapon_list():
    unit = declare_unit_identity(Mock())

    # A bare Mock returns a non-iterable here and breaks every caller that walks it.
    assert MissileAutomation.weapons_in_flight(unit, {}) == []


def test_declared_unit_weapons_are_filtered_to_those_still_airborne():
    airborne = declare_weapon_identity(Mock(), designated_target_id=1)
    airborne.id = 10
    spent = declare_weapon_identity(Mock(), designated_target_id=1)
    spent.id = 11
    unit = declare_unit_identity(Mock(), missiles=[airborne, spent])

    assert MissileAutomation.weapons_in_flight(unit, {10: object()}) == [airborne]


def test_declared_unit_is_engageable_by_default():
    unit = declare_unit_identity(Mock())

    assert unit.is_non_engageable is False
    assert unit.is_missile is False
