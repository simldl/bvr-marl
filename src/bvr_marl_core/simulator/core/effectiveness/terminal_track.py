"""Terminal tracking-quality submodel (P_trk)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from bvr_marl_core.simulator.core.effectiveness.common import clamp01

# When a track has never been confirmed (age is +inf), treat it as a large but
# finite age so an enabled age-decay drives P_trk toward zero without overflow.
_MAX_TRACK_AGE_S = 30.0


def _track_age_s(provider: Any | None) -> float:
    """Age of the last confirmed track (s); 0 if unknown, capped if never seen."""
    if provider is None:
        return 0.0
    try:
        age = float(getattr(provider, "last_confirmed_track_age_s", 0.0))
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(age):
        return _MAX_TRACK_AGE_S
    return max(0.0, age)


def _track_uncertainty(missile: Any) -> float:
    """Normalized terminal track uncertainty in [0, 1] (0 = perfect, unknown)."""
    val = getattr(missile, "terminal_track_uncertainty", None)
    if val is None:
        return 0.0
    try:
        return clamp01(float(val))
    except (TypeError, ValueError):
        return 0.0


def _has_terminal_lock(radar: Any | None, provider: Any | None) -> bool:
    """True if the missile has a valid terminal lock or a usable (fresh/coasting)
    track at the endgame.

    A context with no target provider has no live seeker/track state to judge —
    e.g. envelope/DLZ planning on a static ``MissileParameters`` — so it is treated
    as a nominal lock rather than penalised. A live missile always carries a target
    provider; only then can a genuinely lost track apply the no-lock factor."""
    if provider is None:
        return True
    if radar is not None and hasattr(radar, "get_locked_target"):
        try:
            if radar.get_locked_target() is not None:
                return True
        except (AttributeError, TypeError, ValueError, KeyError, IndexError, ZeroDivisionError):
            pass
    for meth in ("has_fresh_track", "has_coastable_track"):
        fn = getattr(provider, meth, None)
        if callable(fn):
            try:
                if fn():
                    return True
            except (AttributeError, TypeError, ValueError, KeyError, IndexError, ZeroDivisionError):
                pass
    return False


@dataclass
class TerminalTrackQualityModel:
    """P_trk — terminal seeker/track quality at the endgame.

        P_trk = exp(-lambda_age * t_age) * exp(-lambda_cov * C_trk) * m_lock

    where ``t_age`` is the age of the last valid track, ``C_trk`` a normalized
    track-uncertainty measure, and ``m_lock`` reflects whether a valid terminal
    lock/track exists. Inputs are read live from the missile's radar and target
    provider.

    Terminal-track quality is enabled: a shot guided on a fresh, low-uncertainty,
    locked track keeps P_trk ~ 1.0, while a shot coasting on a stale, uncertain, or
    lost track has its kill probability reduced. This makes estimator quality causally
    affect weapon employment and adds realism (coasting shots are less reliable).
    Chosen so good shots
    (``t_age`` ~ 0 s, ``C_trk`` ~ 0, locked) are effectively unpenalised:
      * ``lambda_age = 0.08`` /s  -> P_trk 0.79 at 3 s stale, 0.45 at 10 s, 0.09 at 30 s;
      * ``lambda_cov = 0.7``      -> P_trk 0.70 at C_trk=0.5, 0.50 at C_trk=1.0;
      * ``no_lock_factor = 0.6``  -> a shot with no usable terminal track keeps 60%.
    """

    lambda_age: float = 0.08
    lambda_cov: float = 0.7
    no_lock_factor: float = 0.6

    def probability(self, missile: Any) -> float:
        provider = getattr(missile, "target_provider", None)
        radar = getattr(missile, "radar", None)

        t_age = _track_age_s(provider)
        c_trk = _track_uncertainty(missile)
        m_lock = 1.0 if _has_terminal_lock(radar, provider) else clamp01(self.no_lock_factor)

        p_trk = math.exp(-self.lambda_age * t_age) * math.exp(-self.lambda_cov * c_trk) * m_lock
        return clamp01(p_trk)
