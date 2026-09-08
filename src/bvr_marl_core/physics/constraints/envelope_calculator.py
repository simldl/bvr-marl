"""
Envelope calculations for flight physics.
Handles stall margins and service ceiling constraints.
"""

import math


class EnvelopeCalculator:
    """Calculate flight envelope scalars for safe operation."""

    #: Floor on the ceiling scalar. ``c_factor`` multiplies ``n_max``, ``phi_max_deg``
    #: and the energy envelope, so letting it reach 0 does not model "hard to
    #: manoeuvre near the ceiling" -- it removes the controls. That is an ABSORBING
    #: state: escaping the ceiling needs exactly the authority it has just taken
    #: away, so the aircraft cannot bank, pull, or even pitch down to descend.
    #:
    #: Observed with scripted controllers: a fighter climbed to exactly 18 000 m and
    #: froze -- altitude 18000.0, speed 405, yaw 74, roll 0.0, pitch 0.0, unchanged
    #: for 200+ steps -- while its controller correctly commanded a full-deflection
    #: bank to a recovery heading. It flew straight out of the map; a removal-reason
    #: tally over 3 episodes was 7 of 7
    #: ``boundary_violation``.
    #:
    #: A service ceiling is where climb rate falls to ~0, not where an aircraft
    #: stops flying: it can still bank and can always descend. 0.20 leaves roughly
    #: 1.8 g and ~16 deg of bank at the ceiling -- enough to turn round slowly and
    #: get down, not enough to fight there.
    CEILING_FACTOR_FLOOR: float = 0.20

    def __init__(
        self,
        delta_v_stall: float = 20.0,
        h_ceil_default: float = 18000.0,
        gamma: float = 2.0,
        ceiling_factor_floor: float | None = None,
    ):
        """
        Initialize envelope calculator.

        Args:
            delta_v_stall: Stall margin speed buffer (m/s)
            h_ceil_default: Default service ceiling if not specified (meters)
            gamma: Exponent for ceiling scaling
            ceiling_factor_floor: Minimum ceiling scalar; see CEILING_FACTOR_FLOOR
        """
        self.delta_v_stall = delta_v_stall
        self.h_ceil_default = h_ceil_default
        self.gamma = gamma
        self.ceiling_factor_floor = (
            self.CEILING_FACTOR_FLOOR
            if ceiling_factor_floor is None
            else float(ceiling_factor_floor)
        )

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
        return max(self.ceiling_factor_floor, c_factor)

    def compute_induced_drag_factor(self, unit) -> float:
        """
        Induced-drag factor k, as flown: ``Cd_induced = k * CL**2``.

        **Read ``K_ind`` first.** Production ``AircraftPhysics`` stores this factor
        directly as ``K_ind`` and exposes neither ``oswald_e`` nor ``aspect_ratio``,
        so every real aircraft fell through to the 0.05 "absent or mocked" fallback
        below -- a generic high-aspect-ratio value. The Eurofighter's own ``K_ind``
        is 0.1617 (delta wing, AR ~2.4), so the energy model computed induced drag
        **3.2x too low** while the airframe flew the real number.

        Because the error enters as ``k * CL**2`` it is negligible at 1 g and
        dominant in a hard turn, which is exactly where it was found: at a sustained
        3-5 g the Ps model reported tens of m/s of energy available while the aircraft
        actually realised a large negative rate. It bled several hundred m/s of speed
        and kilometres of altitude over a few hundred steps while being told it was
        gaining energy the whole way, which latches any low-speed abort a controller
        has and stops it engaging at all.

        Args:
            unit: Aircraft unit

        Returns:
            Induced drag factor
        """
        k_attr = getattr(unit.physics, "K_ind", None)
        try:
            k = float(k_attr)
            if k > 0.0 and math.isfinite(k):
                return k
        except (TypeError, ValueError):
            pass
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
