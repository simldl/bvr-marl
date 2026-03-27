"""
Weapon cooldown management.
Tracks firing cooldowns for missiles and guns.
"""


class WeaponCooldowns:
    """Manage weapon firing cooldowns."""

    def __init__(
        self,
        missile_cooldown_s: float = 2.0,
        gun_cooldown_s: float = 0.15,
        max_missiles_per_target: int = 2,
    ):
        """
        Initialize weapon cooldowns.

        Args:
            missile_cooldown_s: Seconds between missile fires
            gun_cooldown_s: Seconds between gun bursts
            max_missiles_per_target: Maximum active friendly missiles allowed per
                                     target.  Firing at a target is blocked while
                                     active_friendly_missiles_at_target
                                     >= max_missiles_per_target.  Counts missiles
                                     from ALL friendly units.
        """
        self.missile_cooldown_s = missile_cooldown_s
        self.gun_cooldown_s = gun_cooldown_s
        self.max_missiles_per_target = max_missiles_per_target

    def init_agent_cooldowns(self) -> dict:
        """Initialize cooldown state for an agent."""
        return {
            "missile_cooldown_left_s": 0.0,
            "gun_cooldown_left_s": 0.0,
        }

    def update_cooldowns(self, state: dict, dt: float):
        """
        Update cooldown timers.

        Args:
            state: Agent state dict
            dt: Time delta (seconds)
        """
        state["missile_cooldown_left_s"] = max(0.0, state["missile_cooldown_left_s"] - dt)
        state["gun_cooldown_left_s"] = max(0.0, state["gun_cooldown_left_s"] - dt)

    def is_missile_ready(self, state: dict) -> bool:
        """Check if missile cooldown has expired."""
        return state.get("missile_cooldown_left_s", 0.0) <= 0.0

    def is_gun_ready(self, state: dict) -> bool:
        """Check if gun cooldown has expired."""
        return state.get("gun_cooldown_left_s", 0.0) <= 0.0

    def set_missile_cooldown(self, state: dict, cooldown_s: float = None):
        """Set missile cooldown timer."""
        state["missile_cooldown_left_s"] = (
            cooldown_s if cooldown_s is not None else self.missile_cooldown_s
        )

    def set_gun_cooldown(self, state: dict):
        """Set gun cooldown timer."""
        state["gun_cooldown_left_s"] = self.gun_cooldown_s

    def is_target_saturated(self, simulator, group: str, target_id) -> bool:
        """Return True if >= max_missiles_per_target friendly missiles are already
        active against target_id.  Uses the simulator's authoritative missile-target
        registry which is updated atomically at add_unit / remove_unit.
        """
        if target_id is None or simulator is None:
            return False
        count = simulator.count_missiles_at_target(group, target_id)
        return count >= self.max_missiles_per_target
