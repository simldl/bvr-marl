"""The one definition of what a track-confidence number means.

``TrackSnapshot.confidence`` is produced in exactly one place
(``radar.tracking.helpers.track_manager.calculate_confidence``) but was consumed
against **eight different hand-picked thresholds**, none of which referenced the
others and several of which were unreachable in practice. Four sat at 0.80-0.85 --
above the tracker's practical ceiling at BVR range, because the accuracy factor is
range-normalised and floors at 0.55 -- so they were silently acting as "never":
engageability, weapons-quality classification, an observation feature and a shaping
term were all gated on a precision the tracker does not claim to deliver at 75 km.

The bands below are calibrated against the MEASURED confidence distribution after the
recency fix (see ``CONFIDENCE_RECENCY_TAU_UPDATES``), not chosen for roundness. A
mature, consistently-tracked, accurate contact at BVR range peaks around 0.78.

Import these rather than writing a number. A threshold that is not one of these is a
statement that this call site knows something the tracker does not.
"""

from __future__ import annotations

# A contact exists but the estimate is not yet worth acting on.
TRACK_CONFIDENCE_TENTATIVE = 0.20
"""Below this the track is noise or a single unconfirmed hit."""

TRACK_CONFIDENCE_PROBABLE = 0.40
"""Good enough to manoeuvre against and to include in the picture."""

TRACK_CONFIDENCE_COMMIT = 0.45
"""Good enough to commit an intercept. Deliberately below FIRM: committing is
reversible (a controller re-evaluates every tick and can abort), so it does not need
weapons-grade confidence."""

TRACK_CONFIDENCE_SHOOT = 0.55
"""Good enough to release a weapon. Above COMMIT because a launch is not reversible,
but below FIRM because waiting for FIRM at BVR range means never shooting."""

TRACK_CONFIDENCE_FIRM = 0.60
"""Weapons-quality track. Reachable by a mature, well-held contact; NOT a precision
claim about absolute position, which degrades with range by construction."""

CONFIDENCE_BANDS: tuple[tuple[str, float], ...] = (
    ("FIRM", TRACK_CONFIDENCE_FIRM),
    ("PROBABLE", TRACK_CONFIDENCE_PROBABLE),
    ("TENTATIVE", TRACK_CONFIDENCE_TENTATIVE),
)


def confidence_band(confidence: float) -> str:
    """Name the band a confidence value falls in. ``UNKNOWN`` below TENTATIVE."""
    value = float(confidence)
    for name, floor in CONFIDENCE_BANDS:
        if value >= floor:
            return name
    return "UNKNOWN"
