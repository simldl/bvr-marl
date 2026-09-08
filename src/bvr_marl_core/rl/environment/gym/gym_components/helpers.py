"""
Helper functions for agent/enemy management and radar operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bvr_marl_core.domain.tactical_contact import TacticalContact
from bvr_marl_core.rl.environment.rewards.estimated_contact import estimated_contact_from_track

if TYPE_CHECKING:
    from bvr_marl_core.simulator import Simulator


class AgentHelpers:
    """Helper methods for agent/enemy queries and radar lock management."""

    def __init__(
        self,
        simulator: Simulator,
        agent_ids: list[str],
        opponent_ids: list[str],
        agent_to_unit_id: dict[str, int],
        force_locks_all_enemies: bool = False,
    ):
        self.simulator = simulator
        self.agent_ids = agent_ids
        self.opponent_ids = opponent_ids
        self.agent_to_unit_id = agent_to_unit_id
        self.force_locks_all_enemies = force_locks_all_enemies

    @staticmethod
    def get_locked_target_ids(unit) -> set:
        """Return the current lock set for a unit using the sensor/radar APIs."""
        if unit is None:
            return set()

        sensor = getattr(unit, "sensor", None)
        if sensor is not None and hasattr(sensor, "get_locked_targets"):
            try:
                return set(sensor.get_locked_targets() or [])
            except (AttributeError, TypeError, ValueError, KeyError, IndexError, ZeroDivisionError):
                pass

        radar = getattr(unit, "radar", None)
        if radar is not None and hasattr(radar, "get_locked_targets"):
            try:
                return set(radar.get_locked_targets() or [])
            except (AttributeError, TypeError, ValueError, KeyError, IndexError, ZeroDivisionError):
                pass

        locked_targets = getattr(unit, "locked_targets", None)
        return set(locked_targets or [])

    def get_enemies_for_agent(self, agent_id: str) -> list:
        """Get list of enemy units for an agent."""
        enemies = []
        enemy_agent_ids = self.opponent_ids if agent_id in self.agent_ids else self.agent_ids

        for enemy_aid in enemy_agent_ids:
            enemy_uid = self.agent_to_unit_id.get(enemy_aid)
            if enemy_uid is not None and enemy_uid in self.simulator.active_units:
                enemies.append(self.simulator.active_units[enemy_uid])
        return enemies

    def get_targets_for_agent(self, agent_id: str) -> list:
        """Get list of targets being tracked by an agent."""
        uid = self.agent_to_unit_id.get(agent_id)
        if uid is None or uid not in self.simulator.active_units:
            return []

        unit = self.simulator.active_units[uid]
        targets = []
        for target_id in self.get_locked_target_ids(unit):
            if target_id in self.simulator.active_units:
                targets.append(self.simulator.active_units[target_id])

        return targets

    def get_incoming_missiles_for_agent(self, agent_id: str) -> list:
        """Get list of missiles targeting an agent."""
        uid = self.agent_to_unit_id.get(agent_id)
        if uid is None:
            return []

        incoming = []
        for unit_id, unit in self.simulator.active_units.items():
            if getattr(unit, "is_missile", False):
                # Accept either .target_id or .target (object with .id)
                tid = getattr(unit, "target_id", None)
                tgt_obj = getattr(unit, "target", None)
                tgt_id = (
                    tid
                    if tid is not None
                    else (getattr(tgt_obj, "id", None) if tgt_obj is not None else None)
                )
                if tgt_id == uid:
                    incoming.append(unit)
        return incoming

    def get_estimated_contacts_for_agent(
        self,
        agent_id: str,
        *,
        fighter_limit: int | None = None,
        missile_limit: int | None = None,
    ) -> list:
        """Return unit-like contacts built only from the agent's sensor tracks."""
        uid = self.agent_to_unit_id.get(agent_id)
        ownship = self.simulator.active_units.get(uid) if uid is not None else None
        if ownship is None:
            return []
        contacts = []
        fighter_count = 0
        missile_count = 0
        sensor = getattr(ownship, "sensor", None)
        raw_tracks = getattr(sensor, "sensor_tracks", ()) or ()
        converter = getattr(sensor, "tactical_contact", TacticalContact.from_track_snapshot)
        for track in sorted(raw_tracks, key=lambda item: str(item.track_id)):
            unit_type = track.classification
            missile_hint = "missile" in unit_type
            if missile_hint and missile_limit is not None and missile_count >= missile_limit:
                continue
            if not missile_hint and fighter_limit is not None and fighter_count >= fighter_limit:
                continue
            try:
                contact = converter(track)
            except (TypeError, ValueError):
                continue
            if contact.suspect_deception:
                continue
            if contact.is_missile:
                if missile_limit is not None and missile_count >= missile_limit:
                    continue
                missile_count += 1
            else:
                if not contact.engageable:
                    continue
                if fighter_limit is not None and fighter_count >= fighter_limit:
                    continue
                fighter_count += 1
            contacts.append(estimated_contact_from_track(ownship, contact))
            if (
                fighter_limit is not None
                and missile_limit is not None
                and fighter_count >= fighter_limit
                and missile_count >= missile_limit
            ):
                break
        return contacts

    def get_estimated_enemies_for_agent(self, agent_id: str) -> list:
        """Return engageable non-missile contacts for observation-only rewards."""
        return [
            contact
            for contact in self.get_estimated_contacts_for_agent(agent_id)
            if contact.operational_contact.engageable and not contact.is_missile
        ]

    def get_estimated_targets_for_agent(self, agent_id: str) -> list:
        """Return currently locked operational contacts without truth resolution."""
        uid = self.agent_to_unit_id.get(agent_id)
        ownship = self.simulator.active_units.get(uid) if uid is not None else None
        locked_ids = self.get_locked_target_ids(ownship)
        return [
            contact
            for contact in self.get_estimated_enemies_for_agent(agent_id)
            if contact.id in locked_ids
        ]

    def get_estimated_incoming_missiles_for_agent(self, agent_id: str) -> list:
        """Return missile contacts seen by ownship rather than simulator missiles."""
        return [
            contact
            for contact in self.get_estimated_contacts_for_agent(agent_id)
            if contact.is_missile
        ]

    def get_estimated_reward_context(
        self,
        agent_id: str,
        *,
        fighter_limit: int | None = None,
        missile_limit: int | None = None,
    ) -> tuple[list, list, list]:
        """Partition one contact snapshot into enemies, locked targets, and missiles."""
        uid = self.agent_to_unit_id.get(agent_id)
        ownship = self.simulator.active_units.get(uid) if uid is not None else None
        locked_ids = self.get_locked_target_ids(ownship)
        contacts = self.get_estimated_contacts_for_agent(
            agent_id, fighter_limit=fighter_limit, missile_limit=missile_limit
        )
        enemies = [
            contact
            for contact in contacts
            if contact.operational_contact.engageable and not contact.is_missile
        ]
        targets = [contact for contact in enemies if contact.id in locked_ids]
        missiles = [contact for contact in contacts if contact.is_missile]
        return enemies, targets, missiles

    def fix_all_radar_locks(self, all_agent_ids: list[str]):
        """
        Training aid: Force radar locks on enemy aircraft.

        Two modes:
        1. force_locks_all_enemies=False (DEFAULT): Only lock detected targets
           - Agents must point radar at enemies to get locks
           - More realistic but harder to learn initially

        2. force_locks_all_enemies=True (TRAINING WHEELS):
           - Lock ALL enemies regardless of detection
           - Use this in early training (iter 0-500) when agents haven't learned
             to point at enemies yet, causing a chicken-and-egg problem
           - Disable once agents start firing missiles regularly
        """
        for aid in all_agent_ids:
            uid = self.agent_to_unit_id.get(aid)
            if uid is None or uid not in self.simulator.active_units:
                continue

            unit = self.simulator.active_units[uid]

            # Check if unit has radar and sensor systems
            if not hasattr(unit, "radar") or unit.radar is None:
                continue
            if not hasattr(unit, "sensor") or unit.sensor is None:
                continue

            # Get the lock controller
            lock_controller = getattr(unit.radar, "lock_ctrl", None)
            if lock_controller is None:
                continue

            # Get targets based on mode
            targets_to_lock = []
            for other_id, other_unit in self.simulator.active_units.items():
                if other_unit.group != unit.group and not getattr(other_unit, "is_missile", False):
                    if self.force_locks_all_enemies:
                        # TRAINING WHEELS MODE: Lock ALL enemies unconditionally
                        targets_to_lock.append(other_unit.id)
                    else:
                        # NORMAL MODE: Only lock detected targets
                        if hasattr(unit.sensor, "sensor_tracks"):
                            # sensor_tracks format: [(target_id, state, cov, ...)]
                            detected_ids = [track.track_id for track in unit.sensor.sensor_tracks]
                            if other_unit.id in detected_ids:
                                targets_to_lock.append(other_unit.id)

            # Force lock on selected targets by setting lock state directly
            if targets_to_lock and hasattr(lock_controller, "_lock_state"):
                for target_id in targets_to_lock:
                    # Set confirm count high enough to be considered locked
                    # Format: [n_confirm, n_miss]
                    confirm_needed = getattr(lock_controller, "_confirm_needed", 2)
                    lock_controller._lock_state[target_id] = [confirm_needed, 0]

                # CRITICAL FIX: Sync locked_targets set in radar after modifying lock_state
                # Without this, get_locked_targets() returns empty set and fire_missile() fails!
                if hasattr(unit.radar, "locked_targets"):
                    unit.radar.locked_targets = set(lock_controller.locked_target_ids())
