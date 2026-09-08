"""
Drag calculation with load factor.
Computes total drag including induced drag from maneuvers.
"""


class DragCalculator:
    """Calculate drag forces including load factor effects."""

    def __init__(self, envelope_calculator):
        """
        Initialize drag calculator.

        Args:
            envelope_calculator: EnvelopeCalculator instance for induced drag factor
        """
        self.envelope_calculator = envelope_calculator

    def compute_drag_with_load_factor(
        self, unit, v_mps: float, alt_m: float, n_factor: float
    ) -> float:
        """
        Compute drag D(V,h,n) with induced drag from load factor.

        Args:
            unit: Aircraft unit
            v_mps: Velocity (m/s)
            alt_m: Altitude (meters)
            n_factor: Load factor (g's)

        Returns:
            Total drag force (Newtons)
        """
        if hasattr(unit.physics, "compute_total_drag"):
            return unit.physics.compute_total_drag(v_mps, alt_m, n_factor)
        else:
            return self._compute_drag_fallback(unit, v_mps, alt_m, n_factor)

    def _compute_drag_fallback(self, unit, v_mps: float, alt_m: float, n_factor: float) -> float:
        """Fallback: basic drag calculation when physics doesn't provide it."""
        rho = float(unit.physics.air.get_density(alt_m))
        q = 0.5 * rho * v_mps**2
        cd0 = float(unit.physics.get_base_drag_cd(v_mps, alt_m))
        k = self.envelope_calculator.compute_induced_drag_factor(unit)
        W = float(unit.physics.mass_kg) * float(getattr(unit.physics, "g", 9.81))
        S = float(unit.physics.A_m2)

        if q * S > 1e-5:
            CL = n_factor * W / (q * S)
            CD = cd0 + k * CL**2
            return q * S * CD
        else:
            return 0.0

    def get_min_thrust(self, unit, v_mps: float, alt_m: float) -> float:
        """
        Get minimum/idle thrust at current state.

        Args:
            unit: Aircraft unit
            v_mps: Velocity (m/s)
            alt_m: Altitude (meters)

        Returns:
            Minimum thrust (Newtons)
        """
        if hasattr(unit.physics, "get_engine_force"):
            # Use zero thrust for minimum to ensure negative P_s capability
            # In reality this would be small positive idle thrust, but for P_s range
            # we want to show full deceleration capability
            return 0.0
        else:
            return 0.0  # fallback
