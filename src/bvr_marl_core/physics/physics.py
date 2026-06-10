from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

SEA_LEVEL_DENSITY = 1.225
STANDARD_GRAVITY = 9.80665
AIRSPEED_OF_SOUND = 340.29
SEA_LEVEL_TEMPERATURE = 288.15  # K
TEMPERATURE_LAPSE_RATE = 0.0065  # K/m


def get_speed_of_sound(alt_m: float, temp_k: float = None) -> float:
    """Get local speed of sound based on altitude and temperature."""
    if temp_k is None:
        if alt_m < 11000.0:
            temp_k = SEA_LEVEL_TEMPERATURE - TEMPERATURE_LAPSE_RATE * alt_m
        else:
            temp_k = SEA_LEVEL_TEMPERATURE - TEMPERATURE_LAPSE_RATE * 11000.0
    # a = sqrt(γ * R * T), where γ = 1.4, R = 287 J/(kg·K)
    return math.sqrt(1.4 * 287 * temp_k)


class AirLayer:
    _ALT_MAX = 25_000.0
    _ALT_STEP = 50.0

    def __init__(self, sea_level_density: float = SEA_LEVEL_DENSITY):
        self.rho0 = sea_level_density
        self._alt_table = np.arange(
            0.0, self._ALT_MAX + self._ALT_STEP, self._ALT_STEP, dtype=np.float32
        )
        self._rho_table = self._compute_density_vec(self._alt_table)
        # Precompute inverse step for fast index arithmetic (uniform grid)
        self._inv_step = 1.0 / self._ALT_STEP
        self._n_entries = len(self._rho_table)

    @staticmethod
    def _density_scalar(alt_m: float, rho0: float = SEA_LEVEL_DENSITY) -> float:
        if alt_m < 11_000.0:
            return rho0 * math.exp(-alt_m / 8500.0)
        elif alt_m < 20_000.0:
            rho11 = rho0 * math.exp(-11_000.0 / 8500.0)
            return rho11 * math.exp(-(alt_m - 11_000.0) / 6500.0)
        else:
            rho11 = rho0 * math.exp(-11_000.0 / 8500.0)
            rho20 = rho11 * math.exp(-(20_000.0 - 11_000.0) / 6500.0)
            return rho20 * math.exp(-(alt_m - 20_000.0) / 6000.0)

    def _compute_density_vec(self, altitudes: np.ndarray) -> np.ndarray:
        vecf = np.vectorize(self._density_scalar)
        return vecf(altitudes, self.rho0)

    def get_density(self, alt_m: float) -> float:
        if 0.0 <= alt_m <= self._ALT_MAX:
            # Uniform grid: direct index arithmetic avoids np.interp overhead
            idx_f = alt_m * self._inv_step
            idx0 = int(idx_f)
            idx1 = idx0 + 1 if idx0 + 1 < self._n_entries else idx0
            t = idx_f - idx0
            rho = self._rho_table
            return float(rho[idx0] + t * (rho[idx1] - rho[idx0]))
        return self._density_scalar(alt_m, self.rho0)


class ForceModel:
    def __init__(
        self,
        mass_kg: float,
        reference_area_m2: float,
        gravity: float,
        air: AirLayer,
        cd0: float,
        K_ind: float,
        get_base_drag_cd,
        get_engine_force,
    ):
        self.mass_kg = mass_kg
        self.A_m2 = reference_area_m2
        self.g = gravity
        self.air = air
        self.cd0 = cd0
        self.K_ind = K_ind
        self.get_base_drag_cd = get_base_drag_cd
        self.get_engine_force = get_engine_force

    def compute_total_drag(self, v_mps: float, alt_m: float, load_factor: float) -> float:
        rho = self.air.get_density(alt_m)
        q = 0.5 * rho * v_mps**2
        Cd_base = self.get_base_drag_cd(v_mps, alt_m)
        if q * self.A_m2 < 1e-5:
            Cd_induced = 0.0
        else:
            CL = load_factor * self.mass_kg * self.g / (q * self.A_m2)
            Cd_induced = self.K_ind * CL**2
        return (Cd_base + Cd_induced) * q * self.A_m2

    def compute_specific_energy_rate(
        self, v_mps: float, alt_m: float, load_factor: float, throttle: float
    ) -> float:
        D = self.compute_total_drag(v_mps, alt_m, load_factor)
        T = self.get_engine_force(v_mps, alt_m, throttle)
        Ps = ((T - D) * v_mps) / (self.mass_kg * self.g)
        return Ps


@dataclass
class PhysicsParams:
    mass_kg: float
    reference_area_m2: float
    gravity_m_s2: float = STANDARD_GRAVITY
    air: Any = field(default_factory=AirLayer)
    cd0: float = 0.01
    aspect_ratio: float = 7.5
    oswald_e: float = 0.82
    max_speed_mps: float = 340.0
    n_max: float = 9.0
    engine_thrust_profile: dict = field(default_factory=dict)
    descent_multiplier: float = 1.0
    stall0_mps: float | None = None
    service_ceiling_m: float | None = None
    stall_sink_pitch_deg: float = -5.0
    speed_full_thrust: float = 250.0
    stall_margin_mps: float = 10.0
    energy_relief_gain: float = 2.0
    stall_recovery_min_time_s: float = 1.5
    ground_clearance_taper_m: float = 500.0
    ground_min_climb_pitch_deg: float = 5.0


class BasePhysics:
    def __init__(self, params: PhysicsParams):
        self.params = params
