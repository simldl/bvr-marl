"""Unit tests for the decomposed kill-probability model (Phase 4)."""

import math
from types import SimpleNamespace

import pytest

from bvr_marl_core.simulator.core.effectiveness import (
    FuzeModel,
    KillProbabilityModel,
    TerminalTrackQualityModel,
    VulnerabilityModel,
    WarheadModel,
)


def _missile(**kw):
    base = {"hit_probability": 0.85, "lethal_radius_m": 100.0}
    base.update(kw)
    m = SimpleNamespace(**base)
    # A clean-intercept baseline carries a fresh terminal lock, so the (now active)
    # P_trk term is 1.0 and these tests isolate the warhead/fuze/vulnerability terms.
    m.radar = SimpleNamespace(get_locked_target=lambda: 1)
    m.target_provider = SimpleNamespace(
        last_confirmed_track_age_s=0.0,
        has_fresh_track=lambda: True,
        has_coastable_track=lambda: False,
    )
    return m


class TestSubmodelDefaults:
    def test_fuze_defaults_to_one_and_reads_reliability(self):
        assert FuzeModel().probability(_missile()) == pytest.approx(1.0)
        assert FuzeModel().probability(_missile(fuze_reliability=0.9)) == pytest.approx(0.9)

    def test_vulnerability_neutral_and_track_unity_under_clean_lock(self):
        # Vulnerability is still a neutral scaffold; P_trk is active but reaches
        # unity for a fresh, locked, low-uncertainty terminal track.
        assert VulnerabilityModel().probability(_missile(), None, 0.0) == pytest.approx(1.0)
        assert TerminalTrackQualityModel().probability(_missile()) == pytest.approx(1.0)

    def test_warhead_gaussian_falloff(self):
        wh = WarheadModel()
        m = _missile(hit_probability=0.8, lethal_radius_m=100.0)
        assert wh.probability(m, None) == pytest.approx(1.0)
        assert wh.probability(m, 0.0) == pytest.approx(1.0)
        assert wh.probability(m, 100.0) == pytest.approx(math.exp(-1.0))

    def test_warhead_flat_when_no_lethal_radius(self):
        m = _missile(hit_probability=0.8, lethal_radius_m=0.0)
        assert WarheadModel().probability(m, 250.0) == pytest.approx(1.0)

    def test_warhead_effectiveness_overrides_hit_probability(self):
        m = _missile(hit_probability=0.8, warhead_effectiveness=0.6, lethal_radius_m=100.0)
        assert WarheadModel().probability(m, 0.0) == pytest.approx(0.6)


class TestKillProbabilityModel:
    def test_default_ignores_legacy_base_probability(self):
        model = KillProbabilityModel()
        m = _missile(hit_probability=0.8, lethal_radius_m=100.0)
        for d in (None, 0.0, 50.0, 150.0, 400.0):
            expected = 1.0 if d is None else math.exp(-((d / 100.0) ** 2))
            pk, comp = model.compute(m, None, d)
            assert pk == pytest.approx(expected)
            # All non-warhead terms are neutral; Pk is carried by the warhead.
            assert comp["p_int"] == comp["p_fuze"] == comp["p_vul"] == comp["p_trk"] == 1.0
            assert comp["p_wh"] == pytest.approx(expected)

    def test_guidance_reliability_is_not_multiplied_after_intercept(self):
        m = _missile(
            hit_probability=0.8,
            lethal_radius_m=100.0,
            warhead_effectiveness=0.9,
            fuze_reliability=0.95,
            guidance_reliability=0.9,
        )
        pk, comp = KillProbabilityModel().compute(m, None, 0.0)
        assert pk == pytest.approx(0.9 * 0.95)
        assert comp["p_int"] == pytest.approx(1.0)
        assert comp["p_fuze"] == pytest.approx(0.95)
        assert comp["p_wh"] == pytest.approx(0.9)

    def test_pk_is_clamped_to_unit_interval(self):
        # Pathological over-unity inputs must not produce Pk > 1.
        m = _missile(hit_probability=2.0, lethal_radius_m=0.0, guidance_reliability=2.0)
        pk, _ = KillProbabilityModel().compute(m, None, 0.0)
        assert 0.0 <= pk <= 1.0
