"""
Automated missile firing system based on SQI.
Manages missiles-per-target limits and auto-fire logic.
"""

import logging

logger = logging.getLogger(__name__)


class MissileAutomation:
    """Automated missile firing based on tactical criteria."""

    def __init__(
        self,
        enable: bool = False,
        sqi_threshold: float = 0.3,
        max_per_target: int = 2,
        long_cooldown_s: float = 10.0,
    ):
        """
        Initialize missile automation.

        Args:
            enable: Enable automated missile firing
            sqi_threshold: SQI threshold for auto-firing
            max_per_target: Max missiles to fire at one target before long cooldown
            long_cooldown_s: Cooldown duration after max missiles reached
        """
        self.enable = enable
        self.sqi_threshold = sqi_threshold
        self.max_per_target = max_per_target
        self.long_cooldown_s = long_cooldown_s
        self.missiles_per_target = {}
        self.suppressed_unit_ids: set[int] = set()

    def configure(
        self,
        enable: bool = False,
        sqi_threshold: float = 0.3,
        max_per_target: int = 2,
        long_cooldown_s: float = 10.0,
    ):
        """Update automation configuration."""
        self.enable = enable
        self.sqi_threshold = sqi_threshold
        self.max_per_target = max_per_target
        self.long_cooldown_s = long_cooldown_s

        logger.info(f"Automated missile firing: {'ENABLED' if enable else 'DISABLED'}")
        if enable:
            logger.info(f"  SQI threshold: {sqi_threshold}")
            logger.info(f"  Max missiles per target: {max_per_target}")
            logger.info(f"  Long cooldown: {long_cooldown_s}s")

    def set_suppressed_unit_ids(self, unit_ids):
        """Disable missile automation for specific unit ids while leaving manual fire intact."""
        self.suppressed_unit_ids = {int(uid) for uid in unit_ids if uid is not None}

    def should_auto_fire(self, unit, selected_target) -> tuple:
        """
        Check if missile should be automatically fired.

        Args:
            unit: Aircraft unit
            selected_target: Target unit

        Returns:
            (should_fire, veto_reason): tuple of (bool, str or None)
        """
        if not self.enable:
            return (False, "automation_disabled")

        if getattr(unit, "id", None) in self.suppressed_unit_ids:
            return (False, "automation_suppressed_for_unit")

        if selected_target is None:
            return (False, "no_target")

        # Check missiles-per-target limit
        target_id = getattr(selected_target, "id", None)
        if target_id is not None:
            missiles_at_target = self.missiles_per_target.get(target_id, 0)
            if missiles_at_target >= self.max_per_target:
                return (
                    False,
                    f"max_missiles_at_target({missiles_at_target}/{self.max_per_target})",
                )

        # Check if unit has metrics system for SQI calculation
        if not hasattr(unit, "metrics"):
            return (False, "no_metrics_system")

        try:
            # Get SQI from metrics helper
            sqi_result = unit.metrics.get_sqi(selected_target)

            if not sqi_result.get("valid", False):
                error = sqi_result.get("error", "unknown")
                return (False, f"sqi_invalid({error})")

            sqi = sqi_result.get("sqi", 0.0)

            # Check SQI threshold
            if sqi >= self.sqi_threshold:
                return (True, None)  # Clear to fire
            else:
                return (False, f"sqi_too_low({sqi:.3f}<{self.sqi_threshold})")

        except Exception as e:
            logger.debug(f"Auto-fire SQI check failed: {e}")
            return (False, f"sqi_exception({type(e).__name__})")

    def update_missiles_per_target(self, target_id: int, missile_fired: bool):
        """Track missiles fired at each target."""
        if missile_fired and target_id is not None:
            self.missiles_per_target[target_id] = self.missiles_per_target.get(target_id, 0) + 1

    def cleanup_destroyed_targets(self, active_unit_ids: set):
        """Remove destroyed targets from missile tracking."""
        targets_to_remove = [
            tid for tid in self.missiles_per_target.keys() if tid not in active_unit_ids
        ]
        for tid in targets_to_remove:
            del self.missiles_per_target[tid]

    @staticmethod
    def weapons_in_flight(unit, active_units):
        """A shooter's own weapons that are still airborne."""
        return [
            missile
            for missile in getattr(unit, "missiles", None) or ()
            if getattr(missile, "id", None) in (active_units or {})
        ]

    def sync_missiles_in_flight(self, missiles):
        """Rebuild the per-target tally from the weapons actually still in flight.

        Saturation means "this many weapons are *currently* committed to that target",
        so the tally has to be derived rather than accumulated. Accumulating it needs a
        matching expiry, and there is no correct one here: a sensor-limited shooter
        selects in contact space, so expiring against unit ids (the only expiry that
        existed, and oracle-gated at that) either never fires or fires on a coincidental
        id collision. Left accumulating, the cap latches permanently -- once an agent
        has fired its quota at a contact it can never engage that contact again, however
        long ago those weapons were spent.

        Keys stay in whichever namespace the shooter committed in: the launch contact
        for a weapon-track shot, the designated unit for an oracle/legacy one. No truth
        is consulted -- these are the shooter's own weapons.
        """
        counts: dict = {}
        for missile in missiles:
            key = getattr(missile, "launch_contact_id", None)
            if key is None:
                key = getattr(missile, "designated_target_id", None)
            if key is not None:
                counts[key] = counts.get(key, 0) + 1
        self.missiles_per_target = counts

    def get_missiles_at_target(self, target_id: int) -> int:
        """Get count of missiles fired at a target."""
        return self.missiles_per_target.get(target_id, 0)
