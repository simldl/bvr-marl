from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional
import math


@dataclass
class AfterburnerParams:
    """
    Configuration for the afterburner model.

    Notes:
      - thrust_mult: AB full-cap is thrust_mult × MIL full-cap (after density correction).
      - density_exp: exponent for AB density scaling (MIL uses 0.8 here).
      - min_mach:    AB may only light above this Mach (prevents AB at very low speed).
      - tau_on/off:  first-order spool time constants for AB command transitions.
      - delta_cd0:   added parasite drag when AB is lit (nozzle open, plume losses).
      - sfc_mil/ab:  specific fuel consumption [kg / (N·s)] for MIL and AB.
    """
    enabled: bool = True
    thrust_mult: float = 1.6
    density_exp: float = 0.8
    min_mach: float = 0.70
    tau_on_s: float = 0.8
    tau_off_s: float = 0.4
    delta_cd0: float = 0.002
    sfc_mil_kgps_per_N: float = 1.8e-5
    sfc_ab_kgps_per_N: float = 3.8e-5


class Afterburner:
    """
    Self-contained afterburner unit that wraps MIL thrust and exposes:
      - update_spool(dt, v, alt): smooth state change 0..1
      - set_command(on):          request AB on/off (controller-level)
      - engine_force(v, alt, throttle): blended thrust cap × throttle
      - effective_cd0(base_cd0):  CD0 with AB-induced delta
      - fuel_flow_kgps(T):        fuel flow estimate for logging/reward

    The model keeps MIL thrust as the single source of truth for speed/altitude scaling.
    """

    def __init__(
        self,
        *,
        air,
        mass_kg: float,
        g: float,
        engine_prof: Optional[Dict] = None,
        speed_full_thrust: float = 340.0,
        get_speed_of_sound_fn: Callable[[float], float],
        params: Optional[AfterburnerParams] = None,
    ) -> None:
        """
        Args:
            air: atmosphere model with .get_density(alt_m) and .rho0
            mass_kg: aircraft mass
            g: gravity
            engine_prof: dict with optional "full_thrust_speed" override
            speed_full_thrust: fallback full-thrust speed if not in engine_prof
            get_speed_of_sound_fn(alt_m): returns local speed of sound [m/s]
            params: AfterburnerParams (optional; default reasonable values)
        """
        self.air = air
        self.mass_kg = float(mass_kg)
        self.g = float(g)
        self.engine_prof = dict(engine_prof or {})
        self.speed_full_thrust = float(speed_full_thrust)
        self._a = get_speed_of_sound_fn
        self.params = params or AfterburnerParams()

        # State
        self._on_cmd: bool = False  # controller command (boolean)
        self._blend: float = 0.0    # actual spool state (0..1), filtered

    # ----- Public control API -------------------------------------------------

    def set_params(self, params: AfterburnerParams) -> None:
        self.params = params

    def set_command(self, on: bool) -> None:
        """Controller requests AB on/off (spool dynamics handle the rest)."""
        self._on_cmd = bool(on)

    def update_spool(self, dt: float, v_mps: float, alt_m: float) -> None:
        """
        First-order spool:
            blend(t+dt) = blend + alpha * (target - blend),
            alpha = 1 - exp(-dt / tau)
        """
        if not self.params.enabled:
            self._blend = 0.0
            return

        a = max(self._a(alt_m), 1e-6)
        M = v_mps / a
        want = bool(self._on_cmd and (M >= self.params.min_mach))
        tau = self.params.tau_on_s if want else self.params.tau_off_s
        alpha = 1.0 - math.exp(-dt / max(tau, 1e-6))
        target = 1.0 if want else 0.0
        self._blend += alpha * (target - self._blend)
        # numeric hygiene
        self._blend = 0.0 if self._blend < 1e-6 else (1.0 if self._blend > 1.0 - 1e-6 else self._blend)

    # ----- Core physics API ---------------------------------------------------

    def engine_force(self, v_mps: float, alt_m: float, throttle: float) -> float:
        """
        Final thrust = throttle ∈ [0,1] × blended cap between MIL and AB full caps.
        """
        cap = self.effective_thrust_cap(v_mps, alt_m)
        th = max(0.0, float(throttle))
        return th * cap

    def effective_thrust_cap(self, v_mps: float, alt_m: float) -> float:
        """
        Convex combination of full-throttle MIL and AB thrust caps according to spool.
        """
        T_mil = self._mil_thrust_cap(v_mps, alt_m)
        if not self.params.enabled:
            return T_mil
        T_ab = self._ab_thrust_cap(v_mps, alt_m)
        return (1.0 - self._blend) * T_mil + self._blend * T_ab

    def effective_cd0(self, base_cd0: float) -> float:
        """Extra parasite drag when AB is lit (spool-weighted)."""
        return float(base_cd0) + self._blend * float(self.params.delta_cd0)

    def fuel_flow_kgps(self, thrust_N: float) -> float:
        """Specific fuel consumption interpolation for logging/reward shaping."""
        sfc_mil = float(self.params.sfc_mil_kgps_per_N)
        sfc_ab  = float(self.params.sfc_ab_kgps_per_N)
        sfc = (1.0 - self._blend) * sfc_mil + self._blend * sfc_ab
        return sfc * max(0.0, float(thrust_N))

    # ----- Internals: caps based on MIL thrust curve -------------------------

    def _mil_thrust_cap(self, v_mps: float, alt_m: float) -> float:
        """
        Full MIL thrust cap at (v, h) with your existing speed & density scaling:
          T_mil = (m g) * f_v(v) * (rho/ rho0)^0.8
        where f_v increases mildly above a "full_thrust_speed".
        """
        rho_rel = self.air.get_density(alt_m) / self.air.rho0
        T_max0 = self.mass_kg * self.g
        v_full = float(self.engine_prof.get("full_thrust_speed", self.speed_full_thrust))
        if v_mps <= v_full:
            fv = 1.0
        else:
            a = max(self._a(alt_m), 1e-6)
            fv = 1.0 + 0.7 * ((v_mps / a) ** 2 - (v_full / a) ** 2)
            fv = min(max(1.0, fv), 2.4)
        return T_max0 * fv * (rho_rel ** 0.8)

    def _ab_thrust_cap(self, v_mps: float, alt_m: float) -> float:
        """
        AB cap as MIL cap × thrust_mult × (rho^density_exp / rho^0.8) to keep AB/MIL
        scaling consistent at sea level while allowing different altitude sensitivity.
        """
        base_mil_full = self._mil_thrust_cap(v_mps, alt_m)
        rho_rel = self.air.get_density(alt_m) / self.air.rho0
        # relative correction vs MIL exponent
        rho_corr = (rho_rel ** self.params.density_exp) / max(rho_rel ** 0.8, 1e-6)
        return self.params.thrust_mult * base_mil_full * rho_corr

    # ----- Convenience getters ------------------------------------------------

    @property
    def is_enabled(self) -> bool:
        return bool(self.params.enabled)

    @property
    def is_commanded(self) -> bool:
        return bool(self._on_cmd)

    @property
    def spool(self) -> float:
        """0..1 smooth AB state."""
        return float(self._blend)
