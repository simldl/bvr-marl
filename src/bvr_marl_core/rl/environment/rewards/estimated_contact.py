"""Unit-like reward view constructed exclusively from an operational track."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from bvr_marl_core.domain.tactical_contact import TacticalContact
from bvr_marl_core.radar.core.utils import enu_to_geodetic
from bvr_marl_core.rl.environment.rewards.information import RewardInformationClass
from bvr_marl_core.simulator.core.helpers import Position
from bvr_marl_core.simulator.core.units import Velocity


@dataclass(frozen=True, slots=True)
class EstimatedContact:
    """Compatibility facade for shaping code that expects unit kinematics.

    Every field is reconstructed from a :class:`TacticalContact`; the facade
    deliberately has no simulator handle, damage state, weapon inventory, or
    target object. ``operational_contact`` remains available to covariance-aware
    DLZ/SQI consumers so they need not treat this estimate as ground truth.
    """

    id: object
    position: Position
    yaw_deg: float
    pitch_deg: float
    speed: float
    velocity: Velocity
    confidence: float
    classification: str
    operational_contact: TacticalContact
    information_class: RewardInformationClass = RewardInformationClass.OBSERVATION_ONLY

    @property
    def is_missile(self) -> bool:
        return self.operational_contact.is_missile


def estimated_contact_from_track(ownship, contact: TacticalContact) -> EstimatedContact:
    """Project an ownship-relative ENU track into a read-only unit-like view."""
    state = np.asarray(contact.state, dtype=float)
    lat, lon, alt = enu_to_geodetic(
        state[:3], ownship.position.lat, ownship.position.lon, ownship.position.alt
    )
    east, north, up = (float(value) for value in state[3:6])
    horizontal_speed = math.hypot(east, north)
    speed = math.sqrt(east * east + north * north + up * up)
    yaw_deg = math.degrees(math.atan2(east, north)) % 360.0 if horizontal_speed else 0.0
    pitch_deg = math.degrees(math.atan2(up, horizontal_speed)) if speed else 0.0
    return EstimatedContact(
        id=contact.track_id,
        position=Position(lat=float(lat), lon=float(lon), alt=float(alt)),
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        speed=speed,
        velocity=Velocity(east, north, up),
        confidence=contact.confidence,
        classification=contact.classification,
        operational_contact=contact,
    )


__all__ = ["EstimatedContact", "estimated_contact_from_track"]
