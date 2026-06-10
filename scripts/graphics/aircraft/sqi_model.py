"""Shared SQI (shot quality index) model used by the aircraft SQI graphics.

The logistic model mirrors ``aircraft/core/nez.py:sqi()`` in ``bvr_marl_core``.
AMRAAM range is read from the missile class source when available, with a
hard-coded fallback so the scripts run even if the import path changes.
"""

from __future__ import annotations

import numpy as np

# Logistic SQI coefficients (kept in sync with bvr_marl_core nez.sqi()).
_A0 = -1.4  # baseline threshold
_A_D = 3.0  # distance / range factor
_A_VC = 1.2  # closure-rate factor (normalised by 400 m/s)
_A_TH = 0.8  # aspect-angle factor (cosine of relative heading)
_A_RHO = 0.25  # altitude / density factor


def get_amraam_params():
    """Return (base_range_km, min_range_km) for the AMRAAM, reading the source.

    Falls back to (150.0, 1.5) if the missile class cannot be imported.
    """
    base_range_km, min_range_km = 150.0, 1.5
    try:
        import inspect

        from bvr_marl_core.missiles.fox3.amraam import AIM120_AMRAAM

        source = inspect.getsource(AIM120_AMRAAM.__init__)
        if "150_000" in source:
            base_range_km = 150.0
        elif "40_000" in source:
            base_range_km = 40.0
        print(f"[OK] AMRAAM base range from source: {base_range_km} km")
    except Exception as exc:  # pragma: no cover - defensive fallback
        print(f"[WARN] Using hard-coded AMRAAM values: {exc}")
    return base_range_km, min_range_km


def get_meteor_base_range_km():
    """Return the Meteor base range in km, reading the source with fallback."""
    base_range_km = 200.0
    try:
        import inspect

        from bvr_marl_core.missiles.fox3.meteor import Meteor

        source = inspect.getsource(Meteor.__init__)
        if "200_000" in source:
            base_range_km = 200.0
        elif "50_000" in source:
            base_range_km = 50.0
        print(f"[OK] Meteor base range from source: {base_range_km} km")
    except Exception as exc:  # pragma: no cover - defensive fallback
        print(f"[WARN] Using hard-coded Meteor values: {exc}")
    return base_range_km


def compute_sqi(distance_normalized, closure_rate_norm, aspect_cos, rho_ratio=1.0):
    """Compute SQI in [0, 1] from the logistic model.

    Args:
        distance_normalized: Range factor (0 = max range, 1 = min range).
        closure_rate_norm: Closure rate / 400 m/s, clipped to [-1, 1].
        aspect_cos: Cosine of the relative heading angle.
        rho_ratio: Altitude density ratio (1.0 at sea level).
    """
    x = (
        _A0
        + _A_D * distance_normalized
        + _A_VC * closure_rate_norm
        + _A_TH * aspect_cos
        + _A_RHO * (rho_ratio - 1)
    )
    return 1.0 / (1.0 + np.exp(-x))
