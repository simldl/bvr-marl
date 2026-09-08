"""Truth-free tactical contacts and stable policy slots."""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

from bvr_marl_core.domain.classification import CLASSIFICATION_LABELS
from bvr_marl_core.domain.information import (
    ReportLineage,
    TrackSnapshot,
    canonical_report_lineage,
)


@dataclass(frozen=True, slots=True)
class TacticalContact:
    """Operational contact exported to controllers without an entity handle."""

    track_id: Hashable
    state: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    confidence: float
    classification: str = "unknown"
    classification_probabilities: tuple[float, ...] | None = None
    classification_entropy_nats: float | None = None
    effective_classification_evidence: float = 0.0
    age_s: float = 0.0
    engageable: bool = True
    suspect_deception: bool = False
    source_ids: tuple[int | str, ...] = ()
    report_lineage: ReportLineage = ()

    def __post_init__(self) -> None:
        state = tuple(float(value) for value in self.state)
        if len(state) != 6 or not all(math.isfinite(value) for value in state):
            raise ValueError("A tactical contact requires six finite state values.")
        covariance = tuple(tuple(float(value) for value in row) for row in self.covariance)
        if len(covariance) != 6 or any(len(row) != 6 for row in covariance):
            raise ValueError("A tactical contact requires a 6x6 covariance matrix.")
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "confidence", min(1.0, max(0.0, float(self.confidence))))
        object.__setattr__(self, "age_s", max(0.0, float(self.age_s)))
        object.__setattr__(
            self,
            "effective_classification_evidence",
            max(0.0, float(self.effective_classification_evidence)),
        )
        object.__setattr__(self, "classification", str(self.classification).lower())
        object.__setattr__(self, "source_ids", tuple(self.source_ids))
        object.__setattr__(self, "report_lineage", canonical_report_lineage(self.report_lineage))
        if self.classification_probabilities is not None:
            probabilities = tuple(float(value) for value in self.classification_probabilities)
            if len(probabilities) != len(CLASSIFICATION_LABELS):
                raise ValueError("Tactical contact classification belief has the wrong schema.")
            if any(value < 0.0 for value in probabilities) or not math.isclose(
                sum(probabilities), 1.0, abs_tol=1e-6
            ):
                raise ValueError("Classification probabilities must be nonnegative and sum to 1.")
            object.__setattr__(self, "classification_probabilities", probabilities)
            entropy = -sum(value * math.log(max(value, 1e-12)) for value in probabilities)
            object.__setattr__(self, "classification_entropy_nats", entropy)

    @property
    def is_missile(self) -> bool:
        return "missile" in self.classification

    @classmethod
    def from_track_snapshot(cls, snapshot: TrackSnapshot) -> TacticalContact:
        """Convert the authoritative immutable tracker output without truth access."""
        return cls(
            track_id=snapshot.track_id,
            state=snapshot.state,
            covariance=snapshot.covariance,
            confidence=snapshot.confidence,
            classification=snapshot.classification,
            classification_probabilities=snapshot.classification_probabilities,
            classification_entropy_nats=snapshot.classification_entropy_nats,
            effective_classification_evidence=snapshot.effective_classification_evidence,
            age_s=snapshot.age_s,
            engageable=snapshot.engageable,
            suspect_deception=snapshot.suspect_deception,
            source_ids=snapshot.source_ids,
            report_lineage=snapshot.report_lineage,
        )

    # Compatibility spelling for external callers; the accepted contract is
    # deliberately still TrackSnapshot, never the retired positional tuple.
    from_sensor_track = from_track_snapshot


class ContactSlotRegistry:
    """Keep track identities in stable policy slots while contacts coast."""

    def __init__(self, max_slots: int, coast_timeout_s: float = 10.0):
        if max_slots <= 0:
            raise ValueError("max_slots must be positive.")
        self.max_slots = int(max_slots)
        self.coast_timeout_s = max(0.0, float(coast_timeout_s))
        self._track_to_slot: dict[Hashable, int] = {}
        self._contacts: dict[Hashable, TacticalContact] = {}
        self._last_seen_s: dict[Hashable, float] = {}

    def update(
        self, contacts: Sequence[TacticalContact], time_s: float
    ) -> tuple[TacticalContact | None, ...]:
        """Update visible contacts, retaining absent identities for a bounded coast."""
        now = float(time_s)
        # A clock that went BACKWARDS means this registry is being reused across an
        # episode boundary. The expiry below is a forward difference, so stale
        # identities would then never age out: `now - last_seen` goes negative and
        # can never exceed the timeout. Belt-and-braces with the processor-level
        # reset, because a registry that silently accumulates ghosts corrupts target
        # selection rather than failing loudly.
        if self._last_seen_s and now < min(self._last_seen_s.values()):
            self._track_to_slot.clear()
            self._contacts.clear()
            self._last_seen_s.clear()
        visible = {
            contact.track_id: contact
            for contact in contacts
            if contact.engageable and not contact.suspect_deception and not contact.is_missile
        }
        for track_id, contact in visible.items():
            self._contacts[track_id] = contact
            self._last_seen_s[track_id] = now

        expired = [
            track_id
            for track_id, last_seen in self._last_seen_s.items()
            if track_id not in visible and now - last_seen > self.coast_timeout_s
        ]
        for track_id in expired:
            self._track_to_slot.pop(track_id, None)
            self._contacts.pop(track_id, None)
            self._last_seen_s.pop(track_id, None)

        occupied = set(self._track_to_slot.values())
        free_slots = [slot for slot in range(self.max_slots) if slot not in occupied]
        unassigned = sorted(
            (track_id for track_id in visible if track_id not in self._track_to_slot),
            key=str,
        )
        for track_id, slot in zip(unassigned, free_slots, strict=False):
            self._track_to_slot[track_id] = slot

        slots: list[TacticalContact | None] = [None] * self.max_slots
        for track_id, slot in self._track_to_slot.items():
            slots[slot] = self._contacts[track_id]
        return tuple(slots)

    def select(
        self, action_value: float, slots: Sequence[TacticalContact | None]
    ) -> TacticalContact | None:
        """Map [0,1] to a no-target region followed by the OCCUPIED contact slots.

        Binning is over the contacts that actually exist, not over ``max_slots``. The
        previous scheme cut [0,1] into ``max_slots + 1`` fixed bins -- 9 bins of 0.111 at
        the default slot count -- and in a 1v1 only ONE of them was ever occupied. That
        left the single designating band at [0.111, 0.222]: **88.9% of the action range
        designated nothing**, and 0.5 sat deep in permanently-empty territory.

        Measured policy sigma on this axis is ~0.2, nearly twice the old bin width, so
        even a perfectly centred mean designated a real contact only about a third of the
        time. Since `lock_rate`, `shot_opportunity` and every firing gate are evaluated
        against the DESIGNATED contact, that one axis silently switched all of them off:
        v21 stage 1 recorded lock_rate 0.954 -> 0.000 while engageable tracks held flat
        at 0.37-0.44 and ~46 trigger pulls per episode were vetoed.

        Now the no-target choice keeps a fixed slice at the bottom and the rest of the
        axis is split evenly among the occupied slots, so a policy centred at 0.5 always
        designates a real contact and stays there under realistic exploration noise.

        The cost is deliberate: slot identity is no longer pinned to a fixed sub-range as
        contacts come and go. The registry's own slot assignment and stickiness still
        keep identity stable step to step; what changes is that an EMPTY slot no longer
        consumes part of the action range it cannot use.
        """
        occupied = [contact for contact in slots if contact is not None]
        if not occupied:
            return None

        value = min(1.0, max(0.0, float(action_value)))
        if value < NO_TARGET_ACTION_FRACTION:
            return None

        span = (1.0 - NO_TARGET_ACTION_FRACTION) / len(occupied)
        index = int((value - NO_TARGET_ACTION_FRACTION) / span)
        return occupied[min(index, len(occupied) - 1)]


NO_TARGET_ACTION_FRACTION = 0.2
"""Bottom slice of the target-selection axis reserved for the explicit no-target choice.

Kept small on purpose. The policy must be able to decline a shot, but the previous
layout spent 8 of 9 bins on slots that are empty in any 1v1, which made declining the
overwhelming default. At 0.2 the no-target choice stays reachable (a 0.2-wide band
against a policy sigma of ~0.2) while a Gaussian centred at 0.5 designates a real
contact.
"""


DEFAULT_CONTACT_SLOTS = 8
"""Slot count assumed by :class:`ContactSlotRegistry` and its callers by default."""


PINNED_ACTION_LOG_STD = -6.0
"""Log-std that makes an AUTOMATED (non-policy) action axis effectively deterministic.

Lives here, next to the binning that dictates it: :meth:`ContactSlotRegistry.select`
quantises the target-selection axis into bins of width ``1/(max_slots + 1)`` = 0.111 at
the default slot count, so a pinned axis needs a spread far below that. The previous
value used by the policy heads, -2.0, gives sigma 0.1353 -- WIDER than the bin. Even
with the mean correctly set by :func:`action_value_for_contact_slot`, only 31.9% of
sampled steps then designated the real contact; 34.0% fell in the no-target bin and
34.1% on slots that are empty in any 1v1. Shots on those steps are vetoed
``no_target_selected``, which is what made every BVR agent passive.

-6.0 gives sigma 0.00248, putting the bin edge 22 sigma away, while staying far enough
from zero that the Gaussian log-density stays finite and well-scaled in float32.
"""


def entropy_of_pinned_axes(count: int, log_std: float = PINNED_ACTION_LOG_STD) -> float:
    """Differential entropy contributed by ``count`` pinned Gaussian action axes.

    A pinned axis is a constant, not a learned parameter, so this term has zero gradient
    and does not affect training -- but it DOES land in the entropy that RLlib reports,
    which is summed over the full action vector. At ``log_std`` -6.0 each pinned axis
    contributes -4.58, so five of them drag the reported entropy down by 22.9 and a
    healthy policy reads as "entropy -19.2". Subtracting this restores a number that
    means what a reader expects: the entropy of the axes the policy actually controls.
    """
    return count * (0.5 * math.log(2.0 * math.pi * math.e) + float(log_std))


def action_value_for_contact_slot(
    slot_index: int,
    max_slots: int = DEFAULT_CONTACT_SLOTS,
    occupied_slots: int | None = None,
) -> float:
    """Return an action value that :meth:`ContactSlotRegistry.select` maps to ``slot_index``.

    The inverse of ``select``'s binning, returning the CENTRE of the target bin so that
    small perturbations (exploration noise, a slightly different slot count) still land
    on the intended slot. Callers that need to pin or initialize the target-selection
    action must use this instead of hand-picking a number: the naive "neutral" value 0.5
    maps to slot 3 under the default 8 slots, which is permanently empty in any 1v1
    engagement, so every shot taken against it is vetoed ``no_target_selected``.

    ``slot_index`` is 0-based over contact slots; pass ``-1`` for the explicit
    no-target bin.
    """
    if int(slot_index) < 0:
        return 0.0
    occupied = max(1, int(occupied_slots if occupied_slots is not None else 1))
    index = min(int(slot_index), occupied - 1)
    span = (1.0 - NO_TARGET_ACTION_FRACTION) / occupied
    return NO_TARGET_ACTION_FRACTION + (index + 0.5) * span
