import math
from dataclasses import dataclass

import numpy as np

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


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


#: Fraction of a missile's peak speed that it holds as an average GROUND speed
#: across a full flyout, used for the DLZ's time-of-flight ceiling. Measured
#: directly off flyouts rather than assumed: an AMRAAM (v_max 1370) covers
#: 110-116 km of ground track in its 120 s life depending on how much it has to
#: lead, and 100.5 km in 98 s on a straight head-on run -- 918, 965 and 1026 m/s,
#: i.e. 0.67, 0.70 and 0.75 of peak. A Meteor (v_max 1360) covers 143.9 km in
#: 160 s = 899 m/s = 0.66. See ``_achievable_max_range``.
_TOF_GROUND_SPEED_FRACTION: float = 0.70

#: Floor on the modelled closing rate, as a fraction of the missile's average
#: ground speed. Without it a target whose line-of-sight speed approaches the
#: weapon's own collapses every zone onto ``r_min`` and the DLZ stops being a
#: usable gradient; with it a runaway target shrinks the envelope hard but does
#: not erase it.
_TOF_MIN_CLOSING_FRACTION: float = 0.15


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
                except (
                    AttributeError,
                    TypeError,
                    ValueError,
                    KeyError,
                    IndexError,
                    ZeroDivisionError,
                ):
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
            # Kinematic max range (DLZ anchor), distinct from the seeker range.
            max_range_m=getattr(
                missile,
                "max_range_m",
                missile.radar.max_range_m if hasattr(missile, "radar") else 75000.0,
            ),
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
        # Use the kinematic max range (how far the missile can fly), not the
        # seeker/radar activation range, when ranking missiles by reach.
        base_range = params.max_range_m

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
        base_range = getattr(missile, "max_range_m", getattr(missile.radar, "max_range_m", 75000.0))

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

    def _achievable_max_range(
        self,
        params: MissileParameters,
        own_speed: float,
        own_alt: float,
        escape_mps: float,
        rcs_factor: float,
        alt_advantage_m: float = 0.0,
    ) -> float:
        """Achievable maximum range (R_aero) at the current geometry.

        Anchored on the missile's cited *kinematic* max range (achieved head-on,
        high-altitude, high-speed) and scaled down for a lower/slower launch and
        for the target aspect via ``escape_mps``, the target's line-of-sight
        velocity component (positive = running away).
        """
        r_max = float(params.max_range_m)
        r_min = float(params.min_range_m)

        # Mild launch-energy modifier: thinner air aloft + faster shooter extend
        # reach. ~0.92 at 8 km / 300 m/s, ~1.05 at 12 km / 400 m/s.
        f_alt_speed = _clamp(
            0.80 + 0.015 * (float(own_alt) / 1000.0) + (float(own_speed) - 300.0) / 1500.0,
            0.70,
            1.15,
        )
        # Aspect: dominant effect, and it is a property of the TARGET's escape, not
        # of the shooter's own closure.
        #
        # This used to be `0.45 + Vc/1000` over the TOTAL closing speed, which
        # conflates the two. A stationary target with a non-closing shooter gave
        # Vc ~ 0 -> f_aspect 0.45, so the easiest target in the game -- one that
        # cannot run away at all -- was scored almost identically to a target
        # fleeing at 250 m/s (0.40). Measured consequence on a stationary-target warmup stage, whose
        # opponent is an anchored hold: R_pi came out 37-57 km against an
        # engagement that opens at 50-100 km, so the shot was out of envelope for
        # the whole episode and neither the BT nor the RL policy could ever fire.
        #
        # What actually sets the achievable launch range is how fast the target can
        # open the range while the missile flies. The shooter's own energy already
        # enters through `f_alt_speed` above, so counting its closure again here
        # double-counted it and penalised a shooter that had merely stopped
        # closing. `escape_mps` is the target's velocity component ALONG the
        # line of sight, positive when it is running.
        #
        # The response is ASYMMETRIC on purpose. A target running away subtracts
        # directly from the missile's reach, so it is penalised steeply. A target
        # closing adds little: the weapon is already far faster than either
        # aircraft, and `r_max` is quoted for a head-on shot to begin with, so
        # extra closure only tops it up. A symmetric slope steep enough to model
        # the tail chase drove every head-on geometry into the upper clamp, which
        # flattened R_aero across quite different closures.
        escape = float(escape_mps)
        f_aspect = _clamp(0.95 - (escape / 700.0 if escape > 0.0 else escape / 5000.0), 0.40, 1.05)
        # Height advantage: shooter altitude MINUS target altitude, positive when
        # shooting downhill. `f_alt_speed` above only knows the shooter's own
        # altitude, so a shot taken from far below its target was scored as if it
        # were level -- and the production path lost this term entirely, though the
        # legacy `_kinematic_range` still carries it as `height_bonus`.
        #
        # Without it, a long uphill shot is scored as if it were level: the DLZ calls
        # it in-envelope (R2/R3) and most of those missiles then never reach a terminal
        # event at all, because they spend their energy climbing.
        #
        # Asymmetric on purpose: climbing costs the weapon much more than diving
        # gains it, since the missile burns energy against gravity on the way up and
        # cannot recover it on the way down within the flyout.
        delta_km = float(alt_advantage_m) / 1000.0
        f_height = _clamp(1.0 + (0.03 if delta_km > 0.0 else 0.06) * delta_km, 0.45, 1.20)
        f_geo = f_alt_speed * f_aspect * f_height * _clamp(float(rcs_factor), 0.85, 1.15)

        r_aero = r_min + (r_max - r_min) * _clamp(f_geo, 0.12, 1.20)

        # Time-of-flight ceiling. Everything above scales a *range*; nothing in it
        # knows the weapon has a clock. The missile is retired at `life_time_s`
        # (`missile.py::_removal_reason`) whether or not it still has energy -- an
        # AMRAAM hits that cap still doing ~630 m/s -- so beyond a certain launch
        # range the shot cannot arrive no matter how the aerodynamic terms score it.
        #
        # The weapon has to cover the launch range PLUS whatever the target opens
        # while it flies, within `life_time_s`:
        #
        #     R_tof = (v_ground - escape) * life_time_s
        #
        # `v_ground` is the missile's measured average GROUND speed over a flyout
        # (see `_TOF_GROUND_SPEED_FRACTION`), so the only prediction being made here
        # is that the target holds its current line-of-sight velocity. That is the
        # honest reading of the information a launch decision actually has, and the
        # 80% step from `r_aero` down to `r_pi` is what carries the margin for a
        # target that subsequently accelerates or turns.
        #
        # Verified against flyouts at the stage-2 launch geometry (probe:
        # scripts/eval/missile_geometry_matrix.py in the behavior repo). This caps
        # R_AERO, the absolute kinematic edge, so `r_pi` keeps sitting at 80% of the
        # span and lands inside the measured arrival limit rather than on top of it
        # -- which is what `r_pi` is supposed to mean.
        # `v_ground` scales with the SAME launch-energy and climb terms as the
        # range above, not just with the missile's spec sheet: a weapon lofted from
        # high and fast really does average a higher ground speed than one dragged
        # uphill, and without that the ceiling -- once it binds -- would flatten
        # `f_alt_speed`/`f_height` out of the answer entirely and leave the DLZ with
        # no altitude gradient to learn from.
        # Taken as a SQUARE ROOT of the energy terms, not the product: both already
        # scale `r_aero` above, so applying them again at full strength charges the
        # same climb twice. Measured effect of the double charge on a 10 km uphill
        # shot: the ceiling came out at 4.9 km against a geometric r_aero of 20.1 km,
        # i.e. the ceiling, not the range model, was deciding the envelope. The root
        # keeps the ordering (it is monotonic, so uphill still scores below level
        # still below downhill) while leaving the range model in charge.
        v_ground = _TOF_GROUND_SPEED_FRACTION * float(params.max_speed_mps)
        v_ground *= math.sqrt(max(f_alt_speed * f_height, 1e-6))
        closing = max(v_ground - escape, _TOF_MIN_CLOSING_FRACTION * v_ground)
        r_tof = closing * float(params.life_time_s)
        r_aero = max(r_min, min(r_aero, r_tof))

        return r_aero if math.isfinite(r_aero) else r_max

    @staticmethod
    def _closing_speed_mps(own, tgt) -> float:
        """Closing speed (m/s, positive = closing) summing each platform's
        velocity component along the mutual line of sight."""
        own_to_tgt = geodetic_bearing_deg(
            own.position.lat, own.position.lon, tgt.position.lat, tgt.position.lon
        )
        tgt_to_own = geodetic_bearing_deg(
            tgt.position.lat, tgt.position.lon, own.position.lat, own.position.lon
        )
        own_speed = float(getattr(own, "speed", 0.0))
        tgt_speed = float(getattr(tgt, "speed", 0.0))
        own_closing = own_speed * math.cos(
            math.radians(signed_yaw_deg_diff(own.yaw_deg, own_to_tgt))
        )
        tgt_closing = tgt_speed * math.cos(
            math.radians(signed_yaw_deg_diff(tgt.yaw_deg, tgt_to_own))
        )
        return own_closing + tgt_closing

    @staticmethod
    def _target_escape_mps(own, tgt) -> float:
        """Target's line-of-sight velocity component, positive when it is running.

        The half of the closing speed the TARGET controls. `_closing_speed_mps`
        sums both platforms' contributions, which is the right quantity for
        time-to-intercept but the wrong one for achievable launch range: the
        shooter's own energy is already accounted for separately, so counting its
        closure again would make a shooter that merely stopped closing look like it
        was chasing a fleeing target.
        """
        tgt_to_own = geodetic_bearing_deg(
            tgt.position.lat, tgt.position.lon, own.position.lat, own.position.lon
        )
        tgt_speed = float(getattr(tgt, "speed", 0.0))
        # cos(0) = 1 when the target points AT the shooter, i.e. closing; negate so
        # positive means opening the range.
        return -tgt_speed * math.cos(math.radians(signed_yaw_deg_diff(tgt.yaw_deg, tgt_to_own)))

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

    @dataclass(slots=True)
    class EstimatedDLZ:
        """Launch-zone estimate with covariance-derived uncertainty bounds."""

        conservative: "NoEscapeZoneCalculator.DLZ"
        nominal: "NoEscapeZoneCalculator.DLZ"
        optimistic: "NoEscapeZoneCalculator.DLZ"
        range_sigma_m: float
        closing_speed_mps: float

    def _dlz_from_estimated_kinematics(
        self,
        *,
        target_speed_mps: float,
        closing_speed_mps: float,
        escape_mps: float | None = None,
        alt_advantage_m: float = 0.0,
        rcs_factor: float = 1.0,
    ) -> "NoEscapeZoneCalculator.DLZ":
        params = self._get_best_missile_params(self.own)
        if not params:
            return self.DLZ(1500.0, 15000.0, 30000.0, 40000.0, 1500.0, 8000.0)
        r_min = float(params.min_range_m)
        # Callers that cannot resolve the target's own line-of-sight component fall
        # back to the total closure, which is the pre-existing behaviour.
        if escape_mps is None:
            escape_mps = -float(closing_speed_mps)
        r_aero = self._achievable_max_range(
            params,
            float(getattr(self.own, "speed", 250.0)),
            float(getattr(self.own.position, "alt", 5000.0)),
            escape_mps,
            rcs_factor,
            alt_advantage_m,
        )
        span = max(0.0, r_aero - r_min)
        r_pi = r_min + 0.80 * span
        r_tr = r_min + 0.55 * span
        nez_frac = _clamp(0.35 - target_speed_mps / 2000.0, 0.12, 0.35)
        r_nez_out = max(r_min + nez_frac * span, r_min + 500.0)
        return self.DLZ(r_min, r_tr, r_pi, r_aero, r_min, r_nez_out)

    def compute_dlz_from_track(self, state, covariance) -> "NoEscapeZoneCalculator.EstimatedDLZ":
        """Compute launch zones strictly from an estimated ENU track.

        ``state`` is ``[east, north, up, v_east, v_north, v_up]`` in the
        ownship export frame.  The covariance is used to widen the closing-speed
        hypothesis and to report radial range uncertainty.  No target object,
        platform identity, or exact target energy is accepted by this API.
        """
        x = np.asarray(state, dtype=float).reshape(-1)
        p = np.asarray(covariance, dtype=float)
        if x.size < 6 or p.shape != (6, 6):
            raise ValueError("estimated DLZ requires a 6D state and 6x6 covariance")
        relative_position = x[:3]
        range_m = max(float(np.linalg.norm(relative_position)), 1.0)
        los = relative_position / range_m
        own_speed = float(getattr(self.own, "speed", 0.0))
        own_yaw = math.radians(float(getattr(self.own, "yaw_deg", 0.0)))
        own_pitch = math.radians(float(getattr(self.own, "pitch_deg", 0.0)))
        own_velocity = np.array(
            [
                own_speed * math.cos(own_pitch) * math.sin(own_yaw),
                own_speed * math.cos(own_pitch) * math.cos(own_yaw),
                own_speed * math.sin(own_pitch),
            ]
        )
        target_velocity = x[3:6]
        relative_velocity = target_velocity - own_velocity
        closing_speed = -float(np.dot(los, relative_velocity))
        target_speed = float(np.linalg.norm(target_velocity))
        # Positive when the target is opening the range. `los` points shooter->target.
        target_escape = float(np.dot(los, target_velocity))
        # Shooter minus target altitude: `relative_position[2]` is the target's UP
        # offset from ownship, so the shooter's advantage is its negation.
        alt_advantage = -float(relative_position[2])
        range_variance = max(float(los @ p[:3, :3] @ los), 0.0)
        closing_variance = max(float(los @ p[3:6, 3:6] @ los), 0.0)
        range_sigma = math.sqrt(range_variance)
        closing_sigma = math.sqrt(closing_variance)

        nominal = self._dlz_from_estimated_kinematics(
            target_speed_mps=target_speed,
            closing_speed_mps=closing_speed,
            escape_mps=target_escape,
            alt_advantage_m=alt_advantage,
        )
        conservative = self._dlz_from_estimated_kinematics(
            target_speed_mps=target_speed + closing_sigma,
            closing_speed_mps=closing_speed - 2.0 * closing_sigma,
            escape_mps=target_escape + 2.0 * closing_sigma,
            alt_advantage_m=alt_advantage,
        )
        optimistic = self._dlz_from_estimated_kinematics(
            target_speed_mps=max(0.0, target_speed - closing_sigma),
            closing_speed_mps=closing_speed + 2.0 * closing_sigma,
            escape_mps=target_escape - 2.0 * closing_sigma,
            alt_advantage_m=alt_advantage,
        )
        return self.EstimatedDLZ(
            conservative=conservative,
            nominal=nominal,
            optimistic=optimistic,
            range_sigma_m=range_sigma,
            closing_speed_mps=closing_speed,
        )

    @staticmethod
    def sqi_from_estimate(
        slant_range_m: float, closing_speed_mps: float, dlz: "NoEscapeZoneCalculator.DLZ"
    ) -> float:
        """Shot quality using only estimated range, closure, and launch-zone bands."""
        d = float(slant_range_m)
        if d < dlz.r_min_m:
            quality = 0.15 * d / max(dlz.r_min_m, 1.0)
        elif d <= dlz.r_nez_out_m:
            quality = 0.80 + 0.10 * (1.0 - d / max(dlz.r_nez_out_m, 1.0))
        elif d <= dlz.r_pi_m:
            fraction = (d - dlz.r_nez_out_m) / max(dlz.r_pi_m - dlz.r_nez_out_m, 1.0)
            quality = 0.80 - 0.35 * fraction
        elif d < dlz.r_aero_m:
            fraction = (d - dlz.r_pi_m) / max(dlz.r_aero_m - dlz.r_pi_m, 1.0)
            quality = 0.45 - 0.42 * fraction
        else:
            quality = 0.02
        quality *= _clamp(0.85 + float(closing_speed_mps) / 3000.0, 0.70, 1.12)
        return _clamp(quality, 0.0, 1.0)

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
        tgt_speed = float(getattr(tgt, "speed", 250.0))
        vc = self._closing_speed_mps(own, tgt)
        escape = self._target_escape_mps(own, tgt)
        alt_advantage = float(getattr(own.position, "alt", 0.0)) - float(
            getattr(getattr(tgt, "position", None), "alt", 0.0)
        )
        rcs_factor = (
            float(self._rcs_factor(self._rcs(tgt), params.seeker_sensitivity))
            if params is not None
            else 1.0
        )
        result = self._dlz_from_estimated_kinematics(
            target_speed_mps=tgt_speed,
            closing_speed_mps=vc,
            escape_mps=escape,
            alt_advantage_m=alt_advantage,
            rcs_factor=rcs_factor,
        )
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

    # SQI: instantaneous shot-quality 0..1, calibrated against the DLZ bands.
    def sqi(self, own, tgt, missile=None, dlz: "NoEscapeZoneCalculator.DLZ" = None) -> float:
        """Shot-quality index in [0, 1].

        High and flat inside the no-escape zone, ~0.45 at the non-maneuvering max
        (r_pi), decaying to ~0 at the aerodynamic limit, and ~0 beyond it — so a
        shot the missile cannot kinematically reach no longer reads as a viable
        shot (the previous logistic returned ~0.35 even far past max range). A
        below-minimum-range shot is poor (arming/timeline). A mild closure
        modifier favours head-on over low-closure crossing/running shots.
        """
        if dlz is None:
            dlz = self.compute_dlz(tgt)

        d = self._slant_range_m(own, tgt)
        r_min, r_nez, r_pi, r_aero = dlz.r_min_m, dlz.r_nez_out_m, dlz.r_pi_m, dlz.r_aero_m

        if d < r_min:
            q = 0.15 * (d / max(r_min, 1.0))  # inside min range: poor shot
        elif d <= r_nez:
            q = 0.80 + 0.10 * (1.0 - d / max(r_nez, 1.0))  # no-escape zone: 0.80..0.90
        elif d <= r_pi:
            f = (d - r_nez) / max(r_pi - r_nez, 1.0)
            q = 0.80 - 0.35 * f  # 0.80 -> 0.45
        elif d < r_aero:
            f = (d - r_pi) / max(r_aero - r_pi, 1.0)
            q = 0.45 - 0.42 * f  # 0.45 -> 0.03
        else:
            q = 0.02  # beyond aerodynamic max range: not a viable shot

        vc = self._closing_speed_mps(own, tgt)
        q *= _clamp(0.85 + vc / 3000.0, 0.70, 1.12)
        return _clamp(q, 0.0, 1.0)

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
