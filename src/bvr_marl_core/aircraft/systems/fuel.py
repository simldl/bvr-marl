"""Aircraft fuel system: consumable fuel coupled to mass.

The aircraft's flying mass is ``empty_mass + fuel``; a full tank reproduces the
type's configured ``mass_kg`` (so a full-fuel start is identical to the previous
fixed-mass model). Fuel is drained each tick by the engine's fuel flow (specific
fuel consumption, higher on afterburner), and the current mass is pushed into the
dynamics — a lighter jet climbs, accelerates and sustains turn better. The engine's
maximum thrust is *not* scaled with mass (that reference stays at the full-mass
value), so burning fuel improves thrust-to-weight, as in reality.

At empty the engine flames out (thrust cut elsewhere); the aircraft then glides.
"""

# Loiter thrust as a fraction of the reference (weight) thrust, used to turn
# remaining fuel into an on-station endurance ("station time").
_LOITER_THRUST_FRACTION = 0.5


class FuelSystem:
    def __init__(self, aircraft, capacity_kg: float, empty_mass_kg: float):
        self.aircraft = aircraft
        self.capacity_kg = float(capacity_kg)
        self.empty_mass_kg = float(empty_mass_kg)
        self.fuel_kg = float(capacity_kg)
        # Disabled (fixed mass, no depletion) when no capacity is configured.
        self.enabled = self.capacity_kg > 0.0
        self.flamed_out = False

    @property
    def fuel_fraction(self) -> float:
        """Remaining fuel in [0, 1]; 1.0 when fuel is not modelled."""
        if self.capacity_kg <= 0.0:
            return 1.0
        return max(0.0, min(1.0, self.fuel_kg / self.capacity_kg))

    @property
    def station_time_s(self) -> float:
        """Remaining on-station endurance (s): how long the current fuel lasts at a
        nominal loiter burn (military-power SFC at a loiter thrust ~ half weight).
        A very large value when fuel is not modelled (unlimited endurance)."""
        if not self.enabled:
            return 1e9
        physics = getattr(self.aircraft, "physics", None)
        ab = getattr(physics, "afterburner", None)
        if physics is None or ab is None:
            return 1e9
        sfc = float(getattr(ab.params, "sfc_mil_kgps_per_N", 1.8e-5))
        g = float(getattr(physics, "g", 9.80665))
        mass = self.empty_mass_kg + self.fuel_kg
        loiter_flow = sfc * _LOITER_THRUST_FRACTION * mass * g
        return self.fuel_kg / max(loiter_flow, 1e-9)

    def consume(self, thrust_N: float, dt: float) -> None:
        """Burn fuel for this tick's thrust and update the flying mass."""
        if not self.enabled:
            return
        ab = getattr(self.aircraft.physics, "afterburner", None)
        flow = ab.fuel_flow_kgps(max(0.0, float(thrust_N))) if ab is not None else 0.0
        self.fuel_kg = max(0.0, self.fuel_kg - flow * float(dt))
        if self.fuel_kg <= 0.0:
            self.flamed_out = True
        self._apply_mass()

    def _apply_mass(self) -> None:
        mass = self.empty_mass_kg + self.fuel_kg
        physics = getattr(self.aircraft, "physics", None)
        if physics is None:
            return
        physics.mass_kg = mass
        force_model = getattr(physics, "force_model", None)
        if force_model is not None:
            force_model.mass_kg = mass
