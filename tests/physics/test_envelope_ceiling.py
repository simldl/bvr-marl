"""The service ceiling must degrade control authority, not remove it.

``c_factor`` multiplies ``n_max``, ``phi_max_deg`` and the energy envelope, so a
``c_factor`` of 0 does not model "hard to manoeuvre up here" -- it takes the
controls away. That is an absorbing state, because getting back down needs the
authority it just removed.

Observed before the floor existed: a BT fighter climbed to exactly 18 000 m and
froze -- altitude 18000.0, speed 405, yaw 74, roll 0.0, pitch 0.0, unchanged for
200+ steps -- while its tree correctly commanded a full-deflection bank onto a
270 deg recovery heading. It flew straight out of the map, and a removal-reason
tally over three BT-vs-BT episodes came back 7 of 7 ``boundary_violation``.
"""

import pytest

from bvr_marl_core.physics.constraints.envelope_calculator import EnvelopeCalculator


class _Physics:
    def __init__(self, service_ceiling=18_000.0):
        self.service_ceiling = service_ceiling


class _Unit:
    def __init__(self, service_ceiling=18_000.0):
        self.physics = _Physics(service_ceiling)


@pytest.mark.parametrize("alt", [18_000.0, 18_500.0, 25_000.0])
def test_ceiling_factor_never_reaches_zero(alt):
    """At and above the ceiling the aircraft must retain some authority."""
    calc = EnvelopeCalculator()

    c = calc._compute_ceiling_factor(_Unit(), alt)

    assert c >= calc.CEILING_FACTOR_FLOOR > 0.0, (
        f"c_factor {c} at {alt} m leaves the aircraft with no bank, no load factor "
        "and no way to descend"
    )


def test_ceiling_factor_still_falls_with_altitude():
    """The floor must not flatten the gradient over the normal flight envelope."""
    calc = EnvelopeCalculator()
    unit = _Unit()

    low = calc._compute_ceiling_factor(unit, 5_000.0)
    mid = calc._compute_ceiling_factor(unit, 12_000.0)

    assert low > mid > calc.CEILING_FACTOR_FLOOR
    assert low == pytest.approx(1.0 - (5_000.0 / 18_000.0) ** 2)


def test_ceiling_floor_is_configurable():
    calc = EnvelopeCalculator(ceiling_factor_floor=0.05)

    assert calc._compute_ceiling_factor(_Unit(), 20_000.0) == pytest.approx(0.05)
