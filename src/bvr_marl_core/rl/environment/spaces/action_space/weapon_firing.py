"""
Weapon firing logic for missiles, guns, and countermeasures.
Handles firing gates, cooldowns, and automation.
"""

import logging

logger = logging.getLogger(__name__)


class WeaponFiringHandler:
    """Handle weapon firing with safety gates and cooldowns."""

    def __init__(self, missile_auto, weapon_cooldowns, trigger_proc):
        """
        Initialize weapon firing handler.

        Args:
            missile_auto: MissileAutomation instance
            weapon_cooldowns: WeaponCooldowns instance
            trigger_proc: TriggerProcessor instance
        """
        self.missile_auto = missile_auto
        self.weapon_cooldowns = weapon_cooldowns
        self.trigger_proc = trigger_proc

    def handle_missile_firing(self, unit, action, selected_target, state, dt):
        """Handle missile firing with automation and cooldowns."""
        aircraft_id = getattr(unit, "id", "unknown")

        # Automated missile firing override
        if self.missile_auto.enable:
            should_auto_fire, auto_veto = self.missile_auto.should_auto_fire(unit, selected_target)
            if should_auto_fire:
                action[4] = 1.0
                logger.debug(
                    f"Aircraft {aircraft_id}: AUTO_FIRE_TRIGGERED - target={getattr(selected_target, 'id', 'none')}"
                )
            elif auto_veto and auto_veto not in ("automation_disabled", "no_target"):
                logger.debug(f"Aircraft {aircraft_id}: AUTO_FIRE_VETO - {auto_veto}")

        # Check firing conditions
        trigger_result = self.trigger_proc.apply_trigger(action[4], 4, state)
        has_missiles = bool(unit.missile_types)
        missile_cooldown_ok = self.weapon_cooldowns.is_missile_ready(state)

        # Pre-flight checks
        veto_reason = None
        target_id = getattr(selected_target, "id", None) if selected_target else None
        target_saturated = self.weapon_cooldowns.is_target_saturated(
            self.simulator, unit.group, target_id
        )
        if selected_target is None:
            veto_reason = "no_target_selected"
        elif not missile_cooldown_ok:
            veto_reason = f"missile_cooldown({state['missile_cooldown_left_s']:.2f}s)"
        elif target_saturated:
            veto_reason = (
                f"target_saturated(>={self.weapon_cooldowns.max_missiles_per_target}_active)"
            )

        if (
            trigger_result
            and selected_target is not None
            and missile_cooldown_ok
            and has_missiles
            and not target_saturated
        ):
            self._fire_missile(unit, selected_target, state, aircraft_id)
        elif trigger_result:
            state["vetoed_missile_attempts_this_step"] = 1
            state["last_lock_ok"] = False
            state["last_fov_ok"] = False
            state["last_target_id"] = (
                getattr(selected_target, "id", None) if selected_target else None
            )
            if veto_reason:
                logger.debug(f"Aircraft {aircraft_id}: FIRE_PRE_VETO - {veto_reason}")

    def _fire_missile(self, unit, selected_target, state, aircraft_id):
        """Execute missile firing."""
        try:
            missile, veto, diagnostics = unit.weapons.fire_missile(
                self.simulator, selected_target, unit.missile_types[0]
            )

            state["last_lock_ok"] = diagnostics.get("has_lock", False)
            state["last_fov_ok"] = diagnostics.get("in_fov", False)
            state["last_target_id"] = diagnostics.get("target_id", None)

            if missile is not None:
                target_id = diagnostics.get("target_id")
                self.missile_auto.update_missiles_per_target(target_id, True)
                missiles_at_target = self.missile_auto.get_missiles_at_target(target_id)

                if missiles_at_target >= self.missile_auto.max_per_target:
                    self.weapon_cooldowns.set_missile_cooldown(
                        state, self.missile_auto.long_cooldown_s
                    )
                    logger.info(
                        f"Aircraft {aircraft_id}: MISSILE_FIRED_SUCCESS - LONG_COOLDOWN={self.missile_auto.long_cooldown_s}s"
                    )
                else:
                    self.weapon_cooldowns.set_missile_cooldown(state)
                    logger.info(
                        f"Aircraft {aircraft_id}: MISSILE_FIRED_SUCCESS - missiles_at_target={missiles_at_target}"
                    )

                state["valid_missile_fires_this_step"] = 1
            else:
                state["vetoed_missile_attempts_this_step"] = 1
                logger.debug(f"Aircraft {aircraft_id}: MISSILE_FIRE_VETO - {veto}")

        except Exception as e:
            logger.warning(f"Aircraft {aircraft_id}: MISSILE_FIRE_EXCEPTION - {e}")
            state["vetoed_missile_attempts_this_step"] = 1
            state["last_lock_ok"] = False
            state["last_fov_ok"] = False
            state["last_target_id"] = (
                getattr(selected_target, "id", None) if selected_target else None
            )

    def handle_gun_firing(self, unit, action, selected_target, state):
        """Handle gun firing with FOV and cooldown checks."""
        aircraft_id = getattr(unit, "id", "unknown")
        gun_trigger_result = self.trigger_proc.apply_trigger(action[5], 5, state)
        gun_cooldown_ok = self.weapon_cooldowns.is_gun_ready(state)

        gun_fov_ok = self._check_gun_fov(unit, selected_target)

        if gun_trigger_result and gun_fov_ok and gun_cooldown_ok:
            self._fire_gun(unit, selected_target, state, aircraft_id)
        elif gun_trigger_result:
            state["vetoed_gun_attempts_this_step"] = 1

    def _check_gun_fov(self, unit, selected_target) -> bool:
        """Check if target is in gun FOV."""
        if selected_target is None:
            return True
        if hasattr(unit, "weapons") and unit.weapons and hasattr(unit.weapons, "is_target_in_fov"):
            try:
                return bool(unit.weapons.is_target_in_fov(selected_target))
            except Exception:
                return True
        return True

    def _fire_gun(self, unit, selected_target, state, aircraft_id):
        """Execute gun firing."""
        try:
            gun_result = unit.weapons.fire_gun(self.simulator, selected_target)
            if gun_result:
                self.weapon_cooldowns.set_gun_cooldown(state)
                state["valid_gun_fires_this_step"] = 1
                logger.info(f"Aircraft {aircraft_id}: GUN_FIRED_SUCCESS")
        except Exception as e:
            if "velocity" not in str(e) or "setter" not in str(e):
                logger.warning(f"Aircraft {aircraft_id}: GUN_FIRE_EXCEPTION - {e}")

    def handle_countermeasures(self, unit, action, state):
        """Handle countermeasure deployment."""
        if self.trigger_proc.apply_trigger(action[6], 6, state):
            unit.countermeasures.launch_flares()
        if self.trigger_proc.apply_trigger(action[7], 7, state):
            unit.countermeasures.launch_chaff()
        if self.trigger_proc.apply_trigger(action[8], 8, state):
            unit.countermeasures.activate_ecm()
        if self.trigger_proc.apply_trigger(action[9], 9, state):
            unit.countermeasures.deploy_decoys()
