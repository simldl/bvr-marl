"""Per-missile effectiveness parameters (Phase 5).

Confirms each Fox-3 type exposes distinct effectiveness submodel parameters,
that they mirror into :class:`MissileParameters` (used by NEZ/DLZ), and that the
kill model consumes them.
"""

import pytest

from bvr_marl_core.missiles.fox3.amraam import AIM120_AMRAAM
from bvr_marl_core.missiles.fox3.k77m import K77M
from bvr_marl_core.missiles.fox3.meteor import Meteor
from bvr_marl_core.missiles.fox3.r37m import R37M
from bvr_marl_core.missiles.fox3.r77_1 import R77_1
from bvr_marl_core.missiles.missile_parameters import MissileParameters
from bvr_marl_core.simulator.core.hit_event_helpers import kill_probability

EXPECTED = {
    AIM120_AMRAAM: dict(
        warhead_effectiveness=1.0,
        fuze_reliability=0.95,
        guidance_reliability=0.95,
        seeker_reliability=0.95,
        datalink_reliability=0.95,
    ),
    Meteor: dict(
        warhead_effectiveness=1.0,
        fuze_reliability=0.96,
        guidance_reliability=0.96,
        seeker_reliability=0.96,
        datalink_reliability=0.97,
    ),
    R77_1: dict(
        warhead_effectiveness=1.0,
        fuze_reliability=0.96,
        guidance_reliability=0.96,
        seeker_reliability=0.94,
        datalink_reliability=0.92,
    ),
    R37M: dict(
        warhead_effectiveness=1.0,
        fuze_reliability=0.96,
        guidance_reliability=0.94,
        seeker_reliability=0.92,
        datalink_reliability=0.90,
    ),
}


@pytest.mark.parametrize("missile_cls,expected", list(EXPECTED.items()))
def test_params_mirror_into_missile_parameters(missile_cls, expected):
    # from_missile_class instantiates the real missile and reads its attributes,
    # so a correct mirror also proves Missile.__init__ set them from config.
    params = MissileParameters.from_missile_class(missile_cls)
    for key, value in expected.items():
        assert getattr(params, key) == pytest.approx(value), f"{missile_cls.__name__}.{key}"


def test_missiles_are_differentiated():
    """The types must not all collapse to identical effectiveness."""
    reliabilities = {
        cls.__name__: (
            exp["fuze_reliability"],
            exp["seeker_reliability"],
            exp["datalink_reliability"],
        )
        for cls, exp in EXPECTED.items()
    }
    assert len(set(reliabilities.values())) > 1


def test_kill_model_consumes_mirrored_params():
    """Pk for a MissileParameters reflects the decomposed terms.

    Guidance quality has already manifested in CPA and only fuze reliability
    reduces a zero-distance geometric intercept.
    """
    params = MissileParameters.from_missile_class(AIM120_AMRAAM)
    expected = 0.95
    assert kill_probability(params, 0.0) == pytest.approx(expected)


@pytest.mark.parametrize(
    "missile_cls",
    [AIM120_AMRAAM, Meteor, K77M, R77_1, R37M],
)
def test_fox3_seeker_envelope_reaches_legacy_fighter_at_100km(missile_cls):
    params = MissileParameters.from_missile_class(missile_cls)
    assert params.radar.max_range_m >= 100_000.0


@pytest.mark.parametrize(
    "missile_cls",
    [AIM120_AMRAAM, Meteor, K77M, R77_1, R37M],
)
def test_clean_intercept_pk_is_at_least_75_percent(missile_cls):
    params = MissileParameters.from_missile_class(missile_cls)
    assert kill_probability(params, 0.0) >= 0.75


@pytest.mark.parametrize(
    "missile_cls",
    [AIM120_AMRAAM, Meteor, K77M, R77_1, R37M],
)
def test_fox3_lethal_radius_is_warhead_efold_below_fuze(missile_cls):
    # Lethal radius (the warhead Gaussian e-fold, 250 m) is deliberately DECOUPLED
    # from the wider proximity-fuze/CCD detonation radius (~500 m): a fuze-triggering
    # but off-boresight near-miss degrades Pk instead of counting as a clean kill.
    params = MissileParameters.from_missile_class(missile_cls)
    assert params.lethal_radius_m == pytest.approx(250.0)
    assert params.warhead_effectiveness == pytest.approx(1.0)
