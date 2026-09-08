from types import SimpleNamespace

import numpy as np
import pytest

from bvr_marl_core.aircraft.systems.observation_helper import ObservationHelper
from bvr_marl_core.domain.tactical_contact import TacticalContact
from bvr_marl_core.simulator.core.helpers import Position
from bvr_marl_core.simulator.core.units import Velocity


def _contact():
    return TacticalContact(
        track_id="track-7",
        state=(10_000.0, 40_000.0, 1_000.0, -100.0, -150.0, 0.0),
        covariance=tuple(tuple(row) for row in np.eye(6)),
        confidence=0.9,
        classification="fighter",
    )


def _aircraft():
    dlz = SimpleNamespace(
        r_min_m=1_000.0,
        r_tr_m=50_000.0,
        r_pi_m=70_000.0,
        r_aero_m=90_000.0,
        r_nez_in_m=1_000.0,
        r_nez_out_m=35_000.0,
    )
    estimate = SimpleNamespace(nominal=dlz, closing_speed_mps=300.0)
    wez = SimpleNamespace(
        compute_dlz_from_track=lambda state, covariance: estimate,
        compute_dlz=lambda target: (_ for _ in ()).throw(AssertionError("truth DLZ used")),
        sqi_from_estimate=lambda range_m, closing_mps, bands: 0.61,
        zone_for_range=lambda range_m, bands: "R2",
        nez_visible=lambda range_m, bands, show_in: True,
    )
    sensor = SimpleNamespace(get_locked_targets=lambda: {"track-7"})
    weapons = SimpleNamespace(
        remaining_missiles=2,
        is_contact_in_fov=lambda contact: True,
        check_fire_feasibility=lambda target: (_ for _ in ()).throw(
            AssertionError("truth fire-feasibility used")
        ),
    )
    return SimpleNamespace(
        id=1,
        position=Position(50.0, 8.0, 5_000.0),
        yaw_deg=0.0,
        speed=250.0,
        velocity=Velocity(0.0, 250.0, 0.0),
        radar=SimpleNamespace(h_fov_deg=90.0, max_range_m=100_000.0),
        sensor=sensor,
        weapons=weapons,
        wez=wez,
        remaining_missiles=2,
    )


def test_contact_dlz_sqi_and_fire_support_never_use_truth_target():
    helper = ObservationHelper(_aircraft())
    contact = _contact()

    envelope = helper.get_dlz_nez_features(contact)
    feasibility = helper.get_fire_feasibility(contact)
    lock = helper.get_lock_quality(contact)

    assert envelope["valid"] is True
    assert envelope["sqi"] == pytest.approx(0.61)
    assert feasibility["can_fire"] is True
    assert feasibility["remaining_missiles"] == 2
    assert lock["has_lock"] is True
    assert 0.0 < lock["lock_strength"] <= contact.confidence
