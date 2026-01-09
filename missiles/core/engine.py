class MissileEngine:
    def __init__(self, missile, initial_fuel_s: float):
        self.missile = missile
        self.fuel_s = float(initial_fuel_s)

    def update(self, tick_secs: float, sim=None):
        # Decrement fuel
        self.fuel_s -= float(tick_secs)
        if self.fuel_s <= 0.0:
            self.fuel_s = 0.0
            # No fuel => no thrust regardless of phase
            self.missile.physics.constant_engine_F = 0.0
            return

        # Fuel available => thrust according to current phase
        self._update_thrust()

    def _update_thrust(self):
        thrust_kN = float(self.missile.phase_manager.get_thrust_kN())
        # Safety: if we somehow entered terminal while fuel remains, thrust is already 0.0
        self.missile.physics.constant_engine_F = max(0.0, thrust_kN) * 1_000

    # Optional: call this AFTER kinematics to decide removal on energy
    def should_remove_missile_on_energy(self, missile) -> bool:
        current_speed = float(getattr(missile, 'speed', 0.0))
        current_alt = float(getattr(getattr(missile, 'position', None), 'alt', 0.0))
        # Energy-based removal criteria (tunable)
        min_effective_speed = 100.0  # m/s
        min_effective_alt = -100.0   # m
        if current_speed < min_effective_speed:
            return True
        if current_alt < min_effective_alt:
            return True
        return False