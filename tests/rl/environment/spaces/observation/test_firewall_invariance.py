"""Information-firewall invariance tests (paper P0 #2 / item 7 + missing #6).

- Observation invariance under hidden-truth perturbation: changing world truth that
  is not part of any sensor report must leave the sensor-limited observation unchanged.
- Entity-ID permutation invariance: reordering the tracker's track list must not
  change the produced token set.
- No live handle: sensor-limited observation arrays are plain floats, never entities.
- Guard tripwire: an oracle-style truth resolution inside a sensor-limited scope raises.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from bvr_marl_core.domain.truth_access_guard import (
    TruthAccessViolation,
    forbidden_truth_access,
    resolve_truth_unit,
)
from bvr_marl_core.rl.environment.spaces.observation.constants import d_EF
from bvr_marl_core.rl.environment.spaces.observation.enemy_info_builder import EnemyInfoBuilder
from tests.helpers.track_snapshot import track_snapshot


class _Config:
    em_slots = 2
    ef_slots = 2

    def __init__(self, mode="sensor_limited"):
        self.information_mode = mode


def _wez():
    dlz = SimpleNamespace(
        r_min_m=2_000.0,
        r_tr_m=35_000.0,
        r_pi_m=55_000.0,
        r_aero_m=70_000.0,
        r_nez_in_m=2_000.0,
        r_nez_out_m=20_000.0,
    )
    estimate = SimpleNamespace(nominal=dlz, closing_speed_mps=350.0)
    return SimpleNamespace(
        compute_dlz_from_track=lambda state, covariance: estimate,
        zone_for_range=lambda distance, bands: "R2",
        sqi_from_estimate=lambda distance, closing, bands: 0.6,
    )


def _track(tid, x):
    return track_snapshot(
        tid,
        state=(x, 2_000.0, 500.0, -200.0, 0.0, 0.0),
        covariance=np.eye(6) * 100.0,
        confidence=0.75,
        classification="unknown",
    )


def _builder(tracks, mode="sensor_limited", truth_units=None):
    sensor = SimpleNamespace(
        sensor_tracks=list(tracks),
        get_locked_targets=lambda: set(),
        bda_confirmed=set(),
        get_nez_features=lambda sim, tid: {},
    )
    ownship = SimpleNamespace(id="blue", yaw_deg=0.0, sensor=sensor, wez=_wez())
    units = {"blue": ownship}
    if truth_units:
        units.update(truth_units)
    sim = SimpleNamespace(active_units=units)
    cfg = _Config(mode)
    if mode == "oracle":
        cfg.oracle_use_reason = "diagnostic"
    return EnemyInfoBuilder(sim, cfg), sim


def _ef_tokens(builder):
    return builder.build("blue")[1]


def test_observation_invariant_to_hidden_truth_perturbation():
    tracks = [_track(41, 10_000.0)]
    truth = SimpleNamespace(
        id=41, position=SimpleNamespace(lat=1.0, lon=2.0, alt=3.0), speed=250.0, is_missile=False
    )
    b1, _ = _builder(tracks, truth_units={41: truth})
    before = _ef_tokens(b1)
    # Perturb hidden world truth that is not part of any sensor report.
    truth.position = SimpleNamespace(lat=9.0, lon=9.0, alt=9.0)
    truth.speed = 600.0
    after = _ef_tokens(b1)
    assert np.array_equal(before, after)


def test_entity_id_permutation_invariance():
    a, b = _track(41, 10_000.0), _track(42, 30_000.0)
    fwd, _ = _builder([a, b])
    rev, _ = _builder([b, a])
    tokens_fwd = _ef_tokens(fwd)
    tokens_rev = _ef_tokens(rev)
    # The tracker's list order must not change the produced token *set*.
    fwd_sorted = tokens_fwd[np.argsort(tokens_fwd[:, 0])]
    rev_sorted = tokens_rev[np.argsort(tokens_rev[:, 0])]
    assert np.allclose(fwd_sorted, rev_sorted)


def test_no_live_handle_in_observation():
    b, _ = _builder([_track(41, 10_000.0)])
    tokens = _ef_tokens(b)
    assert tokens.dtype.kind == "f"  # plain floats, no object handles
    assert tokens.shape[1] == d_EF


def test_guard_tripwire_fires_on_oracle_resolution_in_sensor_limited_scope():
    # Oracle mode resolves track identities to truth; doing so inside a
    # sensor-limited (forbidden) scope must raise via the guard.
    b, _ = _builder([_track(41, 10_000.0)], mode="oracle", truth_units={})
    with forbidden_truth_access("sensor_limited_observation"), pytest.raises(TruthAccessViolation):
        b.build("blue")


def test_guard_silent_in_normal_modes():
    # Neither a normal sensor-limited build nor a normal oracle build raises.
    sl, _ = _builder([_track(41, 10_000.0)], mode="sensor_limited")
    assert _ef_tokens(sl).shape == (2, d_EF)
    orc, _ = _builder([_track(41, 10_000.0)], mode="oracle", truth_units={})
    assert orc.build("blue")[1].shape == (2, d_EF)


def _valid(tokens):
    # Rows whose folded-in validity mask (last column) is set.
    return tokens[tokens[:, -1] > 0.5]


def test_sensor_limited_classification_ignores_truth_unit_flags():
    """Firewall audit (item 4): the oracle truth handle is isolated from the
    sensor-limited path -- classification and engageability come from the TRACK, never
    from the (legacy-tuple) truth unit."""
    track = _track(41, 10_000.0)  # classification "unknown": non-missile, non-support
    # A truth unit whose flags, if consulted, would reclassify/drop the token.
    adversarial = SimpleNamespace(
        id=41,
        position=SimpleNamespace(lat=1.0, lon=2.0, alt=3.0),
        speed=250.0,
        is_missile=True,
        is_support_asset=True,
        is_non_engageable=True,
    )
    sl, _ = _builder([track], mode="sensor_limited", truth_units={41: adversarial})
    em, ef = sl.build("blue")
    # Sensor-limited: the track drives a single engageable enemy-fighter token, and it
    # is NOT reclassified as a missile despite the truth unit's is_missile flag.
    assert _valid(ef).shape[0] == 1
    assert _valid(em).shape[0] == 0

    # Oracle contrast: consulting the truth unit's is_non_engageable flag drops it.
    orc, _ = _builder([track], mode="oracle", truth_units={41: adversarial})
    oem, oef = orc.build("blue")
    assert _valid(oef).shape[0] == 0


def test_pipeline_staging_wraps_every_stage_in_the_trip_wire():
    """Firewall audit (item 3): the ObservationBuilder wraps the whole multi-stage
    ``_build`` in the forbidden-truth trip-wire under sensor-limited mode, so a truth
    resolution added to ANY stage (own/friendly/enemy/warning/passive) fails loudly."""
    from bvr_marl_core.rl.environment.spaces.observation.builder import ObservationBuilder

    ob = ObservationBuilder.__new__(ObservationBuilder)
    ob.simulator = SimpleNamespace(active_units={})
    # Stand in for a sub-builder stage that attempts a track-id -> truth resolution.
    ob._build = lambda agent_id: resolve_truth_unit(ob.simulator, 41, reason="stage_leak")

    ob._sensor_limited = True
    with pytest.raises(TruthAccessViolation):
        ob.build("blue")

    # Outside sensor-limited mode the same access is permitted (no wrapper): no raise.
    ob._sensor_limited = False
    assert ob.build("blue") is None
