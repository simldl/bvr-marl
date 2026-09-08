"""Factories for the authoritative tracker contract used by tests."""

from __future__ import annotations

import numpy as np

from bvr_marl_core.domain.classification import CLASSIFICATION_LABELS
from bvr_marl_core.domain.information import FrameReference, TrackLifecycle, TrackSnapshot


def track_snapshot(
    track_id=1,
    *,
    state=(1_000.0, 2_000.0, 500.0, 100.0, 50.0, 10.0),
    covariance=None,
    classification="fighter",
    confidence=0.9,
    age_s=0.0,
    lifetime_s=10.0,
    engageable=True,
    suspect_deception=False,
    reference=(45.0, 2.0, 8_000.0),
    source_ids=(),
    report_lineage=(),
) -> TrackSnapshot:
    normalized = str(classification).lower()
    aliases = {"aircraft": "fighter", "support": "support_aircraft"}
    normalized = aliases.get(normalized, normalized)
    class_index = (
        CLASSIFICATION_LABELS.index(normalized)
        if normalized in CLASSIFICATION_LABELS
        else CLASSIFICATION_LABELS.index("unknown")
    )
    probabilities = tuple(
        1.0 if index == class_index else 0.0 for index in range(len(CLASSIFICATION_LABELS))
    )
    frame = None if reference is None else FrameReference(*reference)
    covariance = np.eye(6) if covariance is None else covariance
    return TrackSnapshot(
        track_id=track_id,
        state_time_s=float(age_s),
        state=tuple(state),
        covariance=tuple(tuple(row) for row in covariance),
        confidence=confidence,
        lifecycle=TrackLifecycle.CONFIRMED,
        classification_probabilities=probabilities,
        source_ids=tuple(source_ids),
        report_lineage=tuple(report_lineage),
        age_s=age_s,
        lifetime_s=lifetime_s,
        engageable=engageable,
        reference_frame=frame,
        suspect_deception=suspect_deception,
    )
