"""
Envelope calculations for flight physics.
Handles stall margins and service ceiling constraints.
"""

import math


class EnvelopeCalculator:
    """Calculate flight envelope scalars for safe operation."""

    def __init__(
        self, delta_v_stall: float = 20.0, h_ceil_default: float = 18000.0, gamma: float = 2.0
    ):
        """
        Initialize envelope calculator.

        Args:
            delta_v_stall: Stall margin speed buffer (m/s)
            h_ceil_default: Default service ceiling if not specified (meters)
            gamma: Exponent for ceiling scaling
        """
        self.delta_v_stall = delta_v_stall
        self.h_ceil_default = h_ceil_default
        self.gamma = gamma

    def compute_envelope_scalars(self, unit, v_bar: float, alt: float) -> tuple:
        """
        Compute envelope scalars s(v_bar,h) and c(h).

        Args:
            unit: Aircraft unit
            v_bar: Filtered velocity (m/s)
            alt: Altitude (meters)

        Returns:
            (s_factor, c_factor): Stall and ceiling scaling factors [0,1]
        """
        # Stall margin scaling s(v_bar,h)
        s_factor = self._compute_stall_factor(unit, v_bar, alt)

        # Ceiling scaling c(h)
        c_factor = self._compute_ceiling_factor(unit, alt)

        return s_factor, c_factor

    def _compute_stall_factor(self, unit, v_bar: float, alt: float) -> float:
        """Compute stall margin scaling factor."""
        if hasattr(unit.physics, "_dynamic_stall"):
            v_stall = unit.physics._dynamic_stall(alt)
            if math.isfinite(v_stall):
                s_factor = max(0.0, min(1.0, (v_bar - v_stall) / self.delta_v_stall))
            else:
                s_factor = 1.0
        else:
            s_factor = 1.0
        return s_factor

    def _compute_ceiling_factor(self, unit, alt: float) -> float:
        """Compute service ceiling scaling factor."""
        h_ceil_attr = getattr(unit.physics, "service_ceiling", None)
        try:
            h_ceil = float(h_ceil_attr) if h_ceil_attr is not None else float(self.h_ceil_default)
        except (TypeError, ValueError):
            h_ceil = float(self.h_ceil_default)

        c_factor = max(0.0, min(1.0, 1.0 - (alt / max(h_ceil, 1e-3)) ** self.gamma))
        return c_factor

    def compute_induced_drag_factor(self, unit) -> float:
        """
        Compute k = 1/(π e AR) for induced drag (robust to Mocks).

        Args:
            unit: Aircraft unit

        Returns:
            Induced drag factor
        """
        e_attr = getattr(unit.physics, "oswald_e", None)
        AR_attr = getattr(unit.physics, "aspect_ratio", None)
        try:
            e = float(e_attr)
            AR = float(AR_attr)
            if e > 0.0 and AR > 0.0 and math.isfinite(e) and math.isfinite(AR):
                return 1.0 / (math.pi * e * AR)
        except (TypeError, ValueError):
            pass
        # Fallback typical value when parameters are absent or mocked
        return 0.05
