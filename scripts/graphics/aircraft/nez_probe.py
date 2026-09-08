"""Drive the production DLZ/SQI models for the engagement-aid figures.

Every number these figures plot comes from ``aircraft/core/nez.py`` rather than
from a formula copied into the plotting layer. That copy is what this module
exists to remove: the previous figure scripts reimplemented the zone edges with
their own fractions (0.60/0.88/1.04 against a static base range, where
production uses 0.55/0.80 of the span to a *computed* ``r_aero``) and drew the
aspect dependence as a cosine, so both drifted silently as the model changed and
neither showed the aspect, altitude or time-of-flight terms at all.

The aircraft stubs below carry only the attributes ``NoEscapeZoneCalculator``
reads. They exist because building a full simulator aircraft to evaluate a
closed-form zone would be slower and no more faithful -- the model under test is
reached exactly as production reaches it.
"""

from __future__ import annotations

from bvr_marl_core.aircraft.core.nez import NoEscapeZoneCalculator, clear_dlz_cache
from bvr_marl_core.missiles.fox3.amraam import AIM120_AMRAAM
from bvr_marl_core.missiles.fox3.meteor import Meteor
from bvr_marl_core.missiles.missile_parameters import MissileParameters
from bvr_marl_core.simulator.core.helpers import Position

#: Degrees of latitude per kilometre, for placing a target at a chosen range.
DEG_PER_KM = 1.0 / 111.0

#: Radar cross-section of the reference target (m^2). A non-stealth legacy
#: fighter; the calibration requirement in the book is quoted against this.
REFERENCE_TARGET_RCS_M2 = 5.0


def amraam_params() -> MissileParameters:
    return MissileParameters.from_missile_class(AIM120_AMRAAM, fox_type=3)


def meteor_params() -> MissileParameters:
    return MissileParameters.from_missile_class(Meteor, fox_type=3)


class _Platform:
    """Minimal stand-in exposing only what the zone model reads."""

    def __init__(self, lat, lon, alt_m, yaw_deg, speed_mps, ident, params=None, rcs=None):
        self.position = Position(lat, lon, alt_m)
        self.yaw_deg = float(yaw_deg)
        self.pitch_deg = 0.0
        self.roll_deg = 0.0
        self.speed = float(speed_mps)
        self.id = self.name = ident
        self.group = "BLUE"
        if rcs is not None:
            self.rcs = float(rcs)
        if params is not None:
            self.missile_params = {params.name: params}


def shooter(alt_m=10_000.0, speed_mps=300.0, params=None) -> _Platform:
    """Ownship at the origin, heading north."""
    return _Platform(0.0, 0.0, alt_m, 0.0, speed_mps, "own", params=params or amraam_params())


def target(
    range_km=60.0,
    alt_m=10_000.0,
    speed_mps=250.0,
    yaw_deg=180.0,
    rcs=REFERENCE_TARGET_RCS_M2,
) -> _Platform:
    """Target due north of the shooter.

    ``yaw_deg`` 180 closes head-on, 0 runs away, 90 beams. The zone model reads
    the target's velocity component along the line of sight, so heading and speed
    together set the aspect term.
    """
    return _Platform(range_km * DEG_PER_KM, 0.0, alt_m, yaw_deg, speed_mps, "tgt", rcs=rcs)


def dlz_for(own: _Platform, tgt: _Platform):
    """Production DLZ. The per-tick cache is cleared so each sample is independent."""
    clear_dlz_cache()
    return NoEscapeZoneCalculator(own).compute_dlz(tgt)


def dlz_at(
    *,
    own_alt_m=10_000.0,
    own_speed_mps=300.0,
    tgt_alt_m=10_000.0,
    tgt_speed_mps=250.0,
    tgt_yaw_deg=180.0,
    range_km=60.0,
    params=None,
    rcs=REFERENCE_TARGET_RCS_M2,
):
    """Convenience wrapper returning the DLZ for one named geometry."""
    return dlz_for(
        shooter(own_alt_m, own_speed_mps, params),
        target(range_km, tgt_alt_m, tgt_speed_mps, tgt_yaw_deg, rcs),
    )


def sqi_at(slant_range_km: float, **geometry) -> float:
    """Production shot-quality index at a given range for one geometry."""
    params = geometry.pop("params", None)
    own = shooter(
        geometry.get("own_alt_m", 10_000.0),
        geometry.get("own_speed_mps", 300.0),
        params,
    )
    tgt = target(
        slant_range_km,
        geometry.get("tgt_alt_m", 10_000.0),
        geometry.get("tgt_speed_mps", 250.0),
        geometry.get("tgt_yaw_deg", 180.0),
        geometry.get("rcs", REFERENCE_TARGET_RCS_M2),
    )
    clear_dlz_cache()
    calc = NoEscapeZoneCalculator(own)
    return calc.sqi(own, tgt)
