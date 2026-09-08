"""Shared SQI (shot-quality index) accessor for the aircraft SQI graphics.

Calls ``NoEscapeZoneCalculator.sqi()`` directly. It previously reimplemented a
logistic in ``a0 + a_d*d + a_vc*vc + a_th*cos + a_rho*rho`` form and kept the
coefficients "in sync" by hand -- but production replaced that logistic with a
piecewise zone model precisely because the logistic still returned ~0.35 far
beyond maximum range, i.e. it scored a kinematically impossible shot as
half-viable. The figures were therefore plotting a superseded model.

Range parameters are read from the missile classes through
``MissileParameters.from_missile_class`` rather than by grepping their source
text for a literal, which previously picked up the seeker's radar range
(150 km) in place of the weapon's kinematic maximum (160 km).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nez_probe import (  # noqa: E402
    REFERENCE_TARGET_RCS_M2,
    amraam_params,
    dlz_at,
    meteor_params,
    sqi_at,
)

__all__ = [
    "REFERENCE_TARGET_RCS_M2",
    "amraam_params",
    "compute_sqi",
    "dlz_at",
    "get_amraam_params",
    "get_meteor_base_range_km",
    "meteor_params",
    "sqi_at",
]


def get_amraam_params() -> tuple[float, float]:
    """Return ``(max_range_km, min_range_km)`` for the AMRAAM from the model."""
    p = amraam_params()
    return p.max_range_m / 1000.0, p.min_range_m / 1000.0


def get_meteor_base_range_km() -> float:
    """Return the Meteor kinematic maximum range in km, from the model."""
    return meteor_params().max_range_m / 1000.0


def compute_sqi(
    range_km: float,
    *,
    tgt_speed_mps: float = 250.0,
    tgt_yaw_deg: float = 180.0,
    own_alt_m: float = 10_000.0,
    own_speed_mps: float = 300.0,
    tgt_alt_m: float = 10_000.0,
) -> float:
    """Production shot-quality index for one geometry.

    ``tgt_yaw_deg`` is the target's heading: 180 closes head-on, 0 runs away,
    90 beams. Together with ``tgt_speed_mps`` this sets the aspect term, which
    is what the zone model actually reads.
    """
    return sqi_at(
        range_km,
        tgt_speed_mps=tgt_speed_mps,
        tgt_yaw_deg=tgt_yaw_deg,
        own_alt_m=own_alt_m,
        own_speed_mps=own_speed_mps,
        tgt_alt_m=tgt_alt_m,
    )
