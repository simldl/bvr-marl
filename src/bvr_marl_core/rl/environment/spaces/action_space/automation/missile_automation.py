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

    def get_missiles_at_target(self, target_id: int) -> int:
        """Get count of missiles fired at a target."""
        return self.missiles_per_target.get(target_id, 0)
