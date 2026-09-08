"""Build immutable friendly self-reports for a receiving platform."""

from __future__ import annotations

from bvr_marl_core.domain.team_information import FriendlyDatalinkReport
from bvr_marl_core.radar.core.utils import geodetic_to_enu


def _velocity_enu(unit) -> tuple[float, float, float]:
    velocity = getattr(unit, "velocity", None)
    if velocity is not None:
        try:
            return (float(velocity.vx), float(velocity.vy), float(velocity.vz))
        except (AttributeError, TypeError, ValueError):
            pass
    values = getattr(unit, "velocity_enu", (0.0, 0.0, 0.0))
    try:
        return tuple(float(value) for value in values[:3])
    except (TypeError, ValueError):
        return (0.0, 0.0, 0.0)


def _missile_phase(unit) -> float:
    phase = str(getattr(unit, "phase", "")).lower()
    if "boost" in phase:
        return 0.33
    if "terminal" in phase:
        return 1.0
    if phase:
        return 0.67
    time_alive_s = float(getattr(unit, "time_alive_s", 5.0))
    return 0.33 if time_alive_s < 5.0 else 0.67


def _seeker_locked(unit) -> bool:
    if hasattr(unit, "seeker_locked"):
        return bool(unit.seeker_locked)
    seeker = getattr(unit, "seeker", None)
    return bool(getattr(seeker, "has_lock", getattr(seeker, "locked", False)))


def _can_transmit(unit) -> bool:
    if getattr(unit, "is_mortally_hit", False):
        return False
    radar = getattr(unit, "radar", None)
    link = getattr(radar, "data_link", None)
    if link is None:
        link = getattr(unit, "data_link", None)
    if link is None:
        # Aircraft team-state telemetry is part of the environment data link even
        # when a lightweight test double has no explicit radio component.
        return not bool(getattr(unit, "is_missile", False))
    return getattr(link, "get_mode", lambda: "none")() != "none"


class FriendlyPictureAdapter:
    """Authorized boundary that converts friendly self-state into value reports."""

    def __init__(self, simulator):
        self.simulator = simulator

    def reports_for(self, receiver) -> tuple[FriendlyDatalinkReport, ...]:
        receiver_id = getattr(receiver, "id", None)
        receiver_group = getattr(receiver, "group", None)
        receiver_position = receiver.position
        receiver_velocity = _velocity_enu(receiver)
        acquisition_time_s = float(getattr(self.simulator, "elapsed_time_s", 0.0))
        reports = []
        for source in self.simulator.active_units.values():
            source_id = getattr(source, "id", None)
            if (
                source_id == receiver_id
                or getattr(source, "group", None) != receiver_group
                or not _can_transmit(source)
            ):
                continue
            if hasattr(self.simulator, "is_datalink_up") and not self.simulator.is_datalink_up(
                source_id, receiver_id
            ):
                continue
            source_position = source.position
            relative_position = geodetic_to_enu(
                source_position.lat,
                source_position.lon,
                source_position.alt,
                receiver_position.lat,
                receiver_position.lon,
                receiver_position.alt,
            )
            source_velocity = _velocity_enu(source)
            relative_velocity = tuple(
                source_value - receiver_value
                for source_value, receiver_value in zip(
                    source_velocity, receiver_velocity, strict=True
                )
            )
            is_missile = bool(getattr(source, "is_missile", False))
            reports.append(
                FriendlyDatalinkReport(
                    source_id=source_id,
                    receiver_id=receiver_id,
                    acquisition_time_s=acquisition_time_s,
                    relative_state_enu=tuple(relative_position) + relative_velocity,
                    platform_kind="missile" if is_missile else "aircraft",
                    phase=_missile_phase(source) if is_missile else 0.0,
                    seeker_locked=_seeker_locked(source) if is_missile else False,
                    target_track_id=(
                        getattr(source, "designated_track_id", None) if is_missile else None
                    ),
                    lock_track_id=(
                        getattr(source, "selected_track_id", None) if not is_missile else None
                    ),
                )
            )
        return tuple(sorted(reports, key=lambda report: str(report.source_id)))
