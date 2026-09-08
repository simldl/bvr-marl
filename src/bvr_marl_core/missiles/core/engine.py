class MissileEngine:
    def __init__(self, missile, initial_fuel_s: float):
        self.missile = missile
        self.fuel_s = float(initial_fuel_s)
        self.initial_fuel_s = float(initial_fuel_s)

    def update(self, tick_secs: float, sim=None):
        self.fuel_s = max(0.0, self.fuel_s - float(tick_secs))
        # Propellant burns off: mass drops from launch mass to burnout mass over the
        # motor burn, so thrust-to-mass (and hence acceleration) rises as it burns.
        self._update_mass()
        if self.fuel_s <= 0.0:
            self.missile.physics.constant_engine_F = 0.0
            return

        self._update_thrust()

    def _update_mass(self):
        m = self.missile
        launch = float(getattr(m, "launch_mass_kg", getattr(m.physics, "mass_kg", 0.0)))
        burnout = float(getattr(m, "burnout_mass_kg", launch))
        if launch <= 0.0 or self.initial_fuel_s <= 0.0:
            return
        frac = max(0.0, min(1.0, self.fuel_s / self.initial_fuel_s))
        mass = burnout + (launch - burnout) * frac
        m.physics.mass_kg = mass
        force_model = getattr(m.physics, "force_model", None)
        if force_model is not None:
            force_model.mass_kg = mass

    def _update_thrust(self):
        thrust_kN = float(self.missile.phase_manager.get_thrust_kN())
        # Safety: if we somehow entered terminal while fuel remains, thrust is already 0.0
        self.missile.physics.constant_engine_F = max(0.0, thrust_kN) * 1_000

    def should_remove_missile_on_energy(self, missile) -> bool:
        current_speed = float(getattr(missile, "speed", 0.0))
        current_alt = float(getattr(getattr(missile, "position", None), "alt", 0.0))
        min_effective_speed = 50.0
        min_effective_alt = -100.0

        if current_alt < min_effective_alt:
            return True
        if current_speed < min_effective_speed:
            return True
        return False
