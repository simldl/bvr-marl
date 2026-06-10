import math
from dataclasses import dataclass

from bvr_marl_core.missiles.missile_parameters import MissileParameters
from bvr_marl_core.simulator.utils.angles import signed_yaw_deg_diff
from bvr_marl_core.simulator.utils.geodesics import geodetic_bearing_deg

# ---------------------------------------------------------------------------
# Per-tick geometry caches — cleared together at start of each simulator tick
# Key: (own_id, target_id)
# ---------------------------------------------------------------------------
_dlz_cache: dict = {}
_slant_range_cache: dict = {}
_rel_bearing_cache: dict = {}


def clear_dlz_cache() -> None:
    """Clear all per-tick geometry caches. Must be called at the start of each simulator tick."""
    _dlz_cache.clear()
    _slant_range_cache.clear()
    _rel_bearing_cache.clear()


class NoEscapeZoneCalculator:
    def __init__(self, own_aircraft):
        self.own = own_aircraft

    def _get_best_missile_params(self, aircraft) -> MissileParameters:
        """Get the best missile parameters from pre-computed templates."""
        best_params = None
        max_range = -1

        if hasattr(aircraft, "missiles") and aircraft.missiles:
            for missile in aircraft.missiles:
                r = self._get_missile_effective_range(missile, aircraft)
                if r > max_range:
                    best_params = self._missile_to_params(missile)
                    max_range = r

        elif hasattr(aircraft, "missile_params") and aircraft.missile_params:
            for missile_name, params in aircraft.missile_params.items():
                r = self._get_missile_effective_range_from_params(params, aircraft)
                if r > max_range:
                    best_params, max_range = params, r

        elif hasattr(aircraft, "missile_types") and aircraft.missile_types:
            for missile_cls in aircraft.missile_types:
                try:
                    missile = missile_cls(None, None, aircraft, aircraft.map_limits, aircraft.group)
                    r = self._get_missile_effective_range(missile, aircraft)
                    if r > max_range:
                        best_params = self._missile_to_params(missile)
                        max_range = r
                except Exception:
                    continue

        return best_params

    def _get_best_missile(self, aircraft):
        """
        Legacy compatibility shim. Returns a lightweight mock built from
        pre-computed parameters instead of a real missile object.
        """
        params = self._get_best_missile_params(aircraft)
        if params is None:
            return None

        class MissileMock:
            def __init__(self, params):
                self.min_range_m = params.min_range_m
                self.seeker_sensitivity = params.seeker_sensitivity
                self.fox_type = params.fox_type
                self.max_range_m = params.max_range_m
                self.radar = type("obj", (object,), {"max_range_m": params.radar.max_range_m})()
                self.physics = type(
                    "obj", (object,), {"params": type("obj", (object,), {"n_max": params.max_g})()}
                )()

        return MissileMock(params)

    def _missile_to_params(self, missile) -> MissileParameters:
        """Convert a missile object to MissileParameters."""
        from bvr_marl_core.missiles.missile_parameters import MissileRadarParameters

        return MissileParameters(
            name=getattr(missile, "name", "Unknown"),
            max_range_m=missile.radar.max_range_m if hasattr(missile, "radar") else 75000.0,
            min_range_m=getattr(missile, "min_range_m", 1500.0),
            max_speed_mps=getattr(missile, "max_speed_mps", 1000.0),
            max_g=missile.physics.params.n_max if hasattr(missile, "physics") else 30.0,
            seeker_sensitivity=getattr(missile, "seeker_sensitivity", 1.0),
            fox_type=getattr(missile, "fox_type", 3),
            hit_probability=getattr(missile, "hit_probability", 0.8),
            life_time_s=getattr(missile, "life_time_s", 100.0),
            motor_burn_s=getattr(missile, "motor_burn_s", 30.0),
            mass_kg=missile.physics.params.mass_kg if hasattr(missile, "physics") else 150.0,
            radar=MissileRadarParameters(
                horizontal_fov_deg=getattr(missile.radar, "h_fov_deg", 60.0)
                if hasattr(missile, "radar")
                else 60.0,
                vertical_fov_deg=getattr(missile.radar, "v_fov_deg", 30.0)
                if hasattr(missile, "radar")
                else 30.0,
                max_range_m=getattr(missile.radar, "max_range_m", 75000.0)
                if hasattr(missile, "radar")
                else 75000.0,
                radar_frequency_hz=getattr(missile.radar, "freq_hz", 10e9)
                if hasattr(missile, "radar")
                else 10e9,
                tx_power_w=getattr(missile.radar, "tx_power_w", 5000.0)
                if hasattr(missile, "radar")
                else 5000.0,
                antenna_gain_db=getattr(missile.radar, "antenna_gain_db", 30.0)
                if hasattr(missile, "radar")
                else 30.0,
                snr_threshold_db=getattr(missile.radar, "snr_threshold_db", 8.0)
                if hasattr(missile, "radar")
                else 8.0,
            ),
        )

    def _get_missile_effective_range_from_params(
        self, params: MissileParameters, aircraft
    ) -> float:
        """Calculate effective range from missile parameters (fast, no object creation)."""
        base_range = params.radar.max_range_m

        fox_type = params.fox_type

        if fox_type == 1:
            parent_radar_factor = self._get_parent_radar_factor(aircraft)
            return base_range * parent_radar_factor
        elif fox_type == 2:
            return base_range * 0.3
        else:
            return base_range

    def _get_missile_effective_range(self, missile, aircraft):
        """Legacy method using missile objects."""
        base_range = getattr(missile.radar, "max_range_m", getattr(missile, "max_range_m", 75000.0))

        fox_type = getattr(missile, "fox_type", 3)

        if fox_type == 1:
            parent_radar_factor = self._get_parent_radar_factor(aircraft)
            return base_range * parent_radar_factor
        elif fox_type == 2:
            return base_range * 0.3
        else:
            return base_range

    def _get_parent_radar_factor(self, aircraft) -> float:
        if not hasattr(aircraft, "radar"):
            return 0.5

        radar = aircraft.radar
        tx_power = getattr(radar, "tx_power_w", 10000.0)
        antenna_gain = getattr(radar, "antenna_gain_db", 25.0)

        power_factor = (tx_power / 10000.0) * (10 ** (antenna_gain / 10) / 316.22)
        return max(0.3, min(power_factor, 1.5))

    def _rcs(self, obj):
        return getattr(obj, "rcs", 0.5 * getattr(obj, "reference_area_m2", 1.0))

    def _kinematic_range_from_params(
        self, params: MissileParameters, own_speed, tgt_speed, rel_angle, own_alt, tgt_alt
    ) -> float:
        """Calculate kinematic range from missile parameters (fast, no object creation)."""
        max_range = float(params.radar.max_range_m)
        min_range = float(params.min_range_m)

        # Altitude above target extends range; head-on closure increases it
        height_bonus = ((float(own_alt) - float(tgt_alt)) / 1000.0) * 0.03
        range_with_height = max_range * (1.0 + height_bonus)

        try:
            closing_speed = float(own_speed) * math.cos(math.radians(float(rel_angle))) - float(
                tgt_speed
            )
        except Exception:
            closing_speed = 0.0
        closing_factor = 0.20 * max(-1.0, min(closing_speed / 400.0, 1.0))

        base = range_with_height * (1.0 + closing_factor)
        if not math.isfinite(base):
            base = max_range

        return max(min_range, min(base, max_range * 1.3))

    def _kinematic_range(self, missile, own_speed, tgt_speed, rel_angle, own_alt, tgt_alt):
        """Legacy method using missile objects."""
        max_range = float(
            getattr(missile.radar, "max_range_m", getattr(missile, "max_range_m", 75000.0))
        )
        min_range = float(getattr(missile, "min_range_m", 1500.0))

        height_bonus = ((float(own_alt) - float(tgt_alt)) / 1000.0) * 0.03
        range_with_height = max_range * (1.0 + height_bonus)

        try:
            closing_speed = float(own_speed) * math.cos(math.radians(float(rel_angle))) - float(
                tgt_speed
            )
        except Exception:
            closing_speed = 0.0
        closing_factor = 0.20 * max(-1.0, min(closing_speed / 400.0, 1.0))

        base = range_with_height * (1.0 + closing_factor)
        if not math.isfinite(base):
            base = max_range

        return max(min_range, min(base, max_range * 1.3))

    def _rcs_factor(self, rcs, seeker_sensitivity=1.0):
        base_rcs = 1.0
        rcs_effect = math.log10(rcs / base_rcs + 1e-3)
        return 1.0 + seeker_sensitivity * 0.15 * rcs_effect

    @staticmethod
    def _relative_bearing(own, tgt):
        own_id = getattr(own, "id", None)
        tgt_id = getattr(tgt, "id", None)
        if own_id is not None and tgt_id is not None:
            cached = _rel_bearing_cache.get((own_id, tgt_id))
            if cached is not None:
                return cached
        tgt_bearing = geodetic_bearing_deg(
            own.position.lat, own.position.lon, tgt.position.lat, tgt.position.lon
        )
        result = abs(signed_yaw_deg_diff(own.yaw_deg, tgt_bearing))
        if own_id is not None and tgt_id is not None:
            _rel_bearing_cache[(own_id, tgt_id)] = result
        return result

    @staticmethod
    def _slant_range_m(own, tgt) -> float:
        """Flat-earth slant range, accurate to <0.1% over BVR distances."""
        own_id = getattr(own, "id", None)
        tgt_id = getattr(tgt, "id", None)
        if own_id is not None and tgt_id is not None:
            cached = _slant_range_cache.get((own_id, tgt_id))
            if cached is not None:
                return cached
        R_earth = 6_371_000.0
        deg2rad = math.pi / 180.0
        dlat = (tgt.position.lat - own.position.lat) * deg2rad
        dlon = (tgt.position.lon - own.position.lon) * deg2rad
        lat = 0.5 * (tgt.position.lat + own.position.lat) * deg2rad
        dx = R_earth * dlon * math.cos(lat)
        dy = R_earth * dlat
        dz = tgt.position.alt - own.position.alt
        result = float(math.sqrt(dx * dx + dy * dy + dz * dz))
        if own_id is not None and tgt_id is not None:
            _slant_range_cache[(own_id, tgt_id)] = result
        return result

    @dataclass(slots=True)
    class DLZ:
        """Dynamic Launch Zone data."""

        r_min_m: float
        r_tr_m: float
        r_pi_m: float
        r_aero_m: float
        r_nez_in_m: float
        r_nez_out_m: float

    def compute_dlz(self, target) -> "NoEscapeZoneCalculator.DLZ":
        """Compute DLZ (Dynamic Launch Zone) for a target.

        Results are cached per tick by (own_id, target_id). Call clear_dlz_cache()
        at the start of each simulation tick to invalidate.
        """
        own = self.own
        tgt = target

        own_id = getattr(own, "id", None)
        tgt_id = getattr(tgt, "id", None)
        if own_id is not None and tgt_id is not None:
            cache_key = (own_id, tgt_id)
            cached = _dlz_cache.get(cache_key)
            if cached is not None:
                return cached

        params = self._get_best_missile_params(own)
        if not params:
            r_min = 1500.0
            r_tr = 15000.0
            r_pi = 30000.0
            r_aero = 40000.0
            r_nez_out = 8000.0
            return self.DLZ(r_min, r_tr, r_pi, r_aero, r_min, r_nez_out)

        own_speed = getattr(own, "speed", 250.0)
        own_alt = getattr(own.position, "alt", 5000.0)
        tgt_speed = getattr(tgt, "speed", 250.0)
        tgt_alt = getattr(tgt.position, "alt", 5000.0)
        rel_angle = self._relative_bearing(own, tgt)

        base_range = float(
            self._kinematic_range_from_params(
                params, own_speed, tgt_speed, rel_angle, own_alt, tgt_alt
            )
        )
        base_range *= float(self._rcs_factor(self._rcs(tgt), params.seeker_sensitivity))

        r_min = float(params.min_range_m)
        r_tr = r_min + 0.60 * max(0.0, base_range - r_min)  # turning target
        r_pi = r_min + 0.88 * max(0.0, base_range - r_min)  # non-maneuvering
        r_aero = r_min + 1.04 * max(0.0, base_range - r_min)  # aerodynamic limit

        # NEZ: range band from which a tail-on, max-speed target cannot escape.
        # Outer boundary ≈ base_range scaled down for cold-aspect (tail-on) scenario.
        # Inner boundary = r_min (below this, missile cannot arm/guide).
        cold_aspect_factor = max(0.0, 1.0 - tgt_speed / max(params.max_speed_mps, 1.0)) * 0.40
        r_nez_out = r_min + cold_aspect_factor * max(0.0, base_range - r_min)
        r_nez_out = max(r_nez_out, r_min + 500.0)
        r_nez_in = r_min
        result = self.DLZ(r_min, r_tr, r_pi, r_aero, r_nez_in, r_nez_out)
        if own_id is not None and tgt_id is not None:
            _dlz_cache[cache_key] = result
        return result

    def zone_for_range(self, slant_range_m: float, dlz: "NoEscapeZoneCalculator.DLZ") -> str:
        if slant_range_m < dlz.r_min_m:
            return "R1"
        if slant_range_m < dlz.r_tr_m:
            return "R2"
        if slant_range_m < dlz.r_pi_m:
            return "R3"
        return "R4"

    def nez_visible(
        self, slant_range_m: float, dlz: "NoEscapeZoneCalculator.DLZ", show_in=("R2", "R3")
    ) -> bool:
        return self.zone_for_range(slant_range_m, dlz) in show_in

    # SQI: instantaneous intercept-quality 0..1
    def sqi(self, own, tgt, missile=None, dlz: "NoEscapeZoneCalculator.DLZ" = None) -> float:
        if dlz is None:
            dlz = self.compute_dlz(tgt)
        if missile is None:
            missile = self._get_best_missile(own)

        d = self._slant_range_m(own, tgt)
        denom = max(1.0, dlz.r_aero_m - dlz.r_min_m)
        phi_d = max(0.0, min(1.0 - (d - dlz.r_min_m) / denom, 1.0))

        own_speed = getattr(own, "speed", 250.0)
        tgt_speed = getattr(tgt, "speed", 250.0)
        rel_angle = self._relative_bearing(own, tgt)
        cos_aspect = math.cos(math.radians(rel_angle))  # computed once, reused for Vc and aspect
        Vc = own_speed * cos_aspect - tgt_speed
        Vc_n = max(-1.0, min(Vc / 400.0, 1.0))  # normalize by ~M1.2 closure rate

        rho_ratio = 1.0
        try:
            air = getattr(getattr(own, "physics", None), "air", None)
            if air is not None:
                rho_own = air.get_density(getattr(own.position, "alt", 5000.0))
                rho_ratio = max(0.3, min(rho_own / max(getattr(air, "rho0", 1.225), 1e-3), 1.2))
        except Exception:
            pass

        a0, a_d, a_Vc, a_th, a_rho = -1.4, 3.0, 1.2, 0.8, 0.25
        x = a0 + a_d * phi_d + a_Vc * Vc_n + a_th * cos_aspect + a_rho * (rho_ratio - 1.0)
        return 1.0 / (1.0 + math.exp(-x))

    def active_nez(self, target):
        dlz = self.compute_dlz(target)
        return max(dlz.r_nez_out_m, dlz.r_min_m)

    def passive_nez(self, target):
        """Mirror view: DLZ from target's perspective toward self.own.

        Checks the per-tick DLZ cache first (key: (target.id, own.id)).
        On cache miss, uses target.wez (the target's own stable calculator)
        to avoid constructing a throwaway NoEscapeZoneCalculator instance.
        """
        own_id = getattr(self.own, "id", None)
        tgt_id = getattr(target, "id", None)
        if own_id is not None and tgt_id is not None:
            cached = _dlz_cache.get((tgt_id, own_id))
            if cached is not None:
                return max(cached.r_nez_out_m, cached.r_min_m)
        # Cache miss — use target's own stable wez calculator if available
        tgt_wez = getattr(target, "wez", None)
        if tgt_wez is not None:
            dlz = tgt_wez.compute_dlz(self.own)
        else:
            dlz = NoEscapeZoneCalculator(target).compute_dlz(self.own)
        return max(dlz.r_nez_out_m, dlz.r_min_m)
