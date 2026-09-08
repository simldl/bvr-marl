"""Terminal tracking-quality submodel (Phase 7)."""

import math
from types import SimpleNamespace

import pytest

from bvr_marl_core.simulator.core.effectiveness import TerminalTrackQualityModel


def _missile(*, locked=True, fresh=True, age=0.0, uncertainty=None):
    radar = SimpleNamespace(get_locked_target=lambda: 123 if locked else None)
    provider = SimpleNamespace(
        last_confirmed_track_age_s=age,
        has_fresh_track=lambda: fresh,
        has_coastable_track=lambda: False,
    )
    m = SimpleNamespace(radar=radar, target_provider=provider)
    if uncertainty is not None:
        m.terminal_track_uncertainty = uncertainty
    return m


class TestTerminalTrackQualityDefaults:
    def test_neutral_under_clean_lock(self):
        # A fresh, locked, low-uncertainty track is effectively unpenalised.
        assert TerminalTrackQualityModel().probability(_missile()) == pytest.approx(1.0)

    def test_no_lock_penalty_by_default(self):
        # P_trk is active by default: a shot with no usable terminal track keeps
        # only ``no_lock_factor`` (0.6) of its clean-hit lethality.
        m = _missile(locked=False, fresh=False)
        assert TerminalTrackQualityModel().probability(m) == pytest.approx(0.6)

    def test_coastable_track_counts_as_terminal_lock(self):
        radar = SimpleNamespace(get_locked_target=lambda: None)
        provider = SimpleNamespace(
            last_confirmed_track_age_s=0.5,
            has_fresh_track=lambda: False,
            has_coastable_track=lambda: True,
        )
        m = SimpleNamespace(radar=radar, target_provider=provider)
        # Isolate the lock semantics from the age term: a coastable track still
        # counts as a terminal lock (m_lock = 1), so no_lock_factor never applies.
        model = TerminalTrackQualityModel(no_lock_factor=0.0, lambda_age=0.0)
        assert model.probability(m) == pytest.approx(1.0)


class TestTerminalTrackQualityEnabled:
    def test_age_decay_reduces_p_trk(self):
        model = TerminalTrackQualityModel(lambda_age=0.5)
        m = _missile(age=2.0)
        assert model.probability(m) == pytest.approx(math.exp(-1.0))

    def test_uncertainty_decay_reduces_p_trk(self):
        model = TerminalTrackQualityModel(lambda_cov=2.0)
        m = _missile(uncertainty=0.5)
        assert model.probability(m) == pytest.approx(math.exp(-1.0))

    def test_no_lock_penalty_when_enabled(self):
        model = TerminalTrackQualityModel(no_lock_factor=0.6)
        m = _missile(locked=False, fresh=False)
        assert model.probability(m) == pytest.approx(0.6)

    def test_never_confirmed_track_drives_to_zero_with_age_decay(self):
        model = TerminalTrackQualityModel(lambda_age=1.0)
        m = _missile(locked=False, fresh=False, age=float("inf"))
        # age capped at 30 s -> exp(-30) ~ 0, and clamped to [0,1].
        assert model.probability(m) == pytest.approx(0.0, abs=1e-9)
