"""
Helper functions for agent/enemy management and radar operations.
"""

from __future__ import annotations
from typing import List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from simulator.core.simulator import Simulator


class AgentHelpers:
    """Helper methods for agent/enemy queries and radar lock management."""

    def __init__(self, simulator: Simulator, agent_ids: List[str], opponent_ids: List[str],
                 agent_to_unit_id: Dict[str, int], force_locks_all_enemies: bool = False):
        self.simulator = simulator
        self.agent_ids = agent_ids
        self.opponent_ids = opponent_ids
        self.agent_to_unit_id = agent_to_unit_id
        self.force_locks_all_enemies = force_locks_all_enemies

    def get_enemies_for_agent(self, agent_id: str) -> List:
        """Get list of enemy units for an agent."""
        enemies = []
        enemy_agent_ids = self.opponent_ids if agent_id in self.agent_ids else self.agent_ids

        for enemy_aid in enemy_agent_ids:
            enemy_uid = self.agent_to_unit_id.get(enemy_aid)
            if enemy_uid is not None and enemy_uid in self.simulator.active_units:
                enemies.append(self.simulator.active_units[enemy_uid])
        return enemies

    def get_targets_for_agent(self, agent_id: str) -> List:
        """Get list of targets being tracked by an agent."""
        uid = self.agent_to_unit_id.get(agent_id)
        if uid is None or uid not in self.simulator.active_units:
            return []

        unit = self.simulator.active_units[uid]
        targets = []

        if hasattr(unit, "locked_targets") and unit.locked_targets:
            for target_id in unit.locked_targets:
                if target_id in self.simulator.active_units:
                    targets.append(self.simulator.active_units[target_id])

        return targets

    def get_incoming_missiles_for_agent(self, agent_id: str) -> List:
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
                tgt_id = tid if tid is not None else (getattr(tgt_obj, "id", None) if tgt_obj is not None else None)
                if tgt_id == uid:
                    incoming.append(unit)
        return incoming

    def fix_all_radar_locks(self, all_agent_ids: List[str]):
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
            if not hasattr(unit, 'radar') or unit.radar is None:
                continue
            if not hasattr(unit, 'sensor') or unit.sensor is None:
                continue

            # Get the lock controller
            if not hasattr(unit.radar, 'lock_controller'):
                continue

            lock_controller = unit.radar.lock_controller

            # Get targets based on mode
            targets_to_lock = []
            for other_id, other_unit in self.simulator.active_units.items():
                if (other_unit.group != unit.group and
                    not getattr(other_unit, "is_missile", False)):

                    if self.force_locks_all_enemies:
                        # TRAINING WHEELS MODE: Lock ALL enemies unconditionally
                        targets_to_lock.append(other_unit.id)
                    else:
                        # NORMAL MODE: Only lock detected targets
                        if hasattr(unit.sensor, 'sensor_tracks'):
                            # sensor_tracks format: [(target_id, state, cov, ...)]
                            detected_ids = [track[0] for track in unit.sensor.sensor_tracks]
                            if other_unit.id in detected_ids:
                                targets_to_lock.append(other_unit.id)

            # Force lock on selected targets by setting lock state directly
            if targets_to_lock and hasattr(lock_controller, '_lock_state'):
                for target_id in targets_to_lock:
                    # Set confirm count high enough to be considered locked
                    # Format: [n_confirm, n_miss]
                    confirm_needed = getattr(lock_controller, '_confirm_needed', 2)
                    lock_controller._lock_state[target_id] = [confirm_needed, 0]

                # CRITICAL FIX: Sync locked_targets set in radar after modifying lock_state
                # Without this, get_locked_targets() returns empty set and fire_missile() fails!
                if hasattr(unit.radar, 'locked_targets'):
                    unit.radar.locked_targets = set(lock_controller.locked_target_ids())
