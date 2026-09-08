from dataclasses import asdict

import numpy as np

from bvr_marl_core.aircraft.control.countermeasure import AircraftCountermeasureSystem
from bvr_marl_core.aircraft.control.movement_control import AircraftControlSystem
from bvr_marl_core.aircraft.control.weapon_system import AircraftWeaponSystem
from bvr_marl_core.aircraft.core.nez import NoEscapeZoneCalculator
from bvr_marl_core.aircraft.systems.fuel import FuelSystem
from bvr_marl_core.aircraft.systems.sensor import AircraftSensorSystem
from bvr_marl_core.missiles.missile_parameters import MissileParameters
from bvr_marl_core.physics.aircraft import AircraftPhysics
from bvr_marl_core.radar.core.data_link import DataLink
from bvr_marl_core.radar.obs.observation import DEFAULT_NOTCH_VELOCITY_MPS
from bvr_marl_core.radar.units.aircraft import AircraftRadar
from bvr_marl_core.simulator.core.events import AircraftDestroyedEvent, Event
from bvr_marl_core.simulator.core.units import FlyingUnit
from bvr_marl_core.simulator.simulator import Simulator


class Aircraft(FlyingUnit):
    unit_kind: str = "aircraft"
    # Target-vulnerability class used by the kill model's VulnerabilityModel.
    # Subclasses override (e.g. AWACS -> "awacs"); a config "vulnerability_class"
    # key overrides per instance.
    vulnerability_class: str = "fighter"

    def __init__(self, name, position, yaw_deg, speed_mps, group, map_limits, config):
        super().__init__(name=name, position=position, yaw_deg=yaw_deg, speed=speed_mps)
        self.group = group
        self.map_limits = map_limits
        self.type = "Aircraft"
        self.wez = NoEscapeZoneCalculator(self)

        self.should_be_removed = False
        self.removal_reason = None
        # Stochastic kill delay: set when a lethal missile hit lands. The aircraft
        # spirals out of the fight and dies (event + removal) at _death_time_s.
        self.is_mortally_hit = False
        self._death_time_s = None
        self._death_killer = None
        self.boundary_violation_active = False
        self.boundary_violation_countdown = 0
        self.boundary_violation_penalty_per_step = -10.0

        if hasattr(config, "__dataclass_fields__"):
            config = asdict(config)
        self.config = config

        # Resolve the vulnerability class: explicit config override, else the
        # subclass default. Instance attribute so it travels with the unit.
        self.vulnerability_class = config.get("vulnerability_class", type(self).vulnerability_class)

        cfg_mass = config.get("mass_kg", 20000.0)
        cfg_Sref = config.get("reference_area_m2", 27.87)
        cfg_AR = config.get("aspect_ratio", 4.0)
        cfg_oswald = config.get("oswald_e", 0.78)
        cfg_vmax = float(config.get("max_speed_mps", 680.0))
        cfg_nmax = float(config.get("n_max", 9.0))

        self.physics = AircraftPhysics(
            AircraftPhysics.Params(
                mass_kg=cfg_mass,
                reference_area_m2=cfg_Sref,
                aspect_ratio=cfg_AR,
                oswald_e=cfg_oswald,
                max_speed_mps=cfg_vmax,
                n_max=cfg_nmax,
                stall0_mps=config.get("stall0_mps", 60.0),
            )
        )

        self.min_speed_mps = config.get("min_speed_mps", 0.0)
        self.max_speed_mps = cfg_vmax
        self.n_max = cfg_nmax
        self.min_alt_m = config.get("min_alt_m", getattr(map_limits, "min_alt", 0.0))
        self.max_alt_m = config.get("max_alt_m", getattr(map_limits, "max_alt", 20000.0))

        radar_max_range = config.get("radar_max_range_m", 60000.0)
        # Support both key names for backward compatibility
        snr_threshold = config.get("radar_snr_threshold_db", config.get("snr_threshold_db", 10.0))
        self.radar = AircraftRadar(
            horizontal_fov_deg=config.get("radar_horizontal_fov_deg", 60.0),
            vertical_fov_deg=config.get("radar_vertical_fov_deg", 30.0),
            max_range_m=radar_max_range,
            radar_frequency_hz=config.get("radar_frequency_hz", 10e9),
            tx_power_w=config.get("radar_tx_power_w", 5e3),
            antenna_gain_db=config.get("radar_antenna_gain_db", 30.0),
            snr_threshold_db=snr_threshold,
            owner=self,
            data_link=DataLink(mode="full"),
            notch_velocity_mps=config.get("radar_notch_velocity_mps", DEFAULT_NOTCH_VELOCITY_MPS),
            meas_angular_noise_deg=config.get("radar_meas_angular_noise_deg", 0.0),
            meas_range_noise_m=config.get("radar_meas_range_noise_m", 0.0),
            dl_delay_base_s=config.get("radar_dl_delay_base_s", 0.0),
            dl_delay_per_km_s=config.get("radar_dl_delay_per_km_s", 0.0),
            processing_gain_db=config.get("radar_processing_gain_db", 30.0),
            tracker_motion_model=config.get("radar_tracker_motion_model", "cv"),
        )

        # EMCON: when False the radar is silent — it emits no active detections
        # (the agent must rely on the datalink picture) and cannot be seen by an
        # enemy RWR/passive receiver. Driven by the radar on/off action each tick.
        self.radar_emitting = True
        # Per-episode radar-emission statistics (duty cycle / toggles) for the
        # active-sensing observation and the sensing Pareto analysis. Fresh per
        # episode because units are re-spawned on reset.
        from bvr_marl_core.aircraft.systems.emission_tracker import EmissionTracker

        self.emission_tracker = EmissionTracker()

        # Noise-jammer capability (self-protection). Nominal burn-through range in km
        # against a reference radar (see radar/ew/noise_jammer.py); 0 = no jammer.
        # When >0 the aircraft denies range to any enemy radar beyond that radar's
        # (geometry-scaled) burn-through range. Default off so existing configs are
        # unchanged; enable per aircraft type or scenario to turn on EW.
        self.noise_jammer_burn_through_km = float(config.get("noise_jammer_burn_through_km", 0.0))

        # Infrared Search & Track: a passive, angle-only IR sensor. Present only
        # when irst_base_range_m > 0 (e.g. the F-22 omits an IRST). Uses the radar's
        # RNG for measurement noise so a seed reproduces both.
        irst_range = float(config.get("irst_base_range_m", 0.0))
        self.irst = None
        if irst_range > 0.0:
            from bvr_marl_core.aircraft.systems.irst import IRSTSensor

            self.irst = IRSTSensor(
                self,
                fov_deg=float(config.get("irst_fov_deg", 140.0)),
                base_range_m=irst_range,
                angular_noise_deg=float(config.get("irst_angular_noise_deg", 0.4)),
                np_rng=getattr(self.radar, "np_rng", None),
            )

        self.rcs = config.get("rcs", 10.0)
        self.rcs_pattern = self.config.get(
            "rcs_pattern",
            {
                "front_floor": 0.25 if "F-22" in self.name else 0.35,
                "tail_floor": 0.50,
                "n_az": 1.2,
                "k_top": 0.20,
                "k_bottom": 0.35,
                "n_el": 1.5,
                "sigma_min": 0.03,
                "sigma_max": 3.0,
            },
        )

        self.passive_radar_angular_error_deg = config.get("passive_radar_angular_error_deg", 5.0)
        self.passive_radar_range_error_m = config.get("passive_radar_range_error_m", 2000.0)
        self.passive_radar_max_age_s = config.get("passive_radar_max_age_s", 3.0)

        self.missile_warning_delay_s = config.get("missile_warning_delay_s", 1.0)
        self.missile_warning_delay_std = config.get("missile_warning_delay_std", 0.5)

        # Countermeasure inventory (read by AircraftCountermeasureSystem at
        # construction, so it must be set first). Without this the counts defaulted
        # to 0 and no countermeasure could ever be launched.
        self.flares = config.get("flares", 0)
        self.chaff = config.get("chaff", 0)
        self.ecm = config.get("ecm", 0)
        self.decoys = config.get("decoys", 0)

        self.sensor = AircraftSensorSystem(self)
        self.control = AircraftControlSystem(self)
        self.weapons = AircraftWeaponSystem(self)
        self.countermeasures = AircraftCountermeasureSystem(self)

        # Fuel: mass_kg is the full-fuel flying mass, so empty = mass - fuel and a
        # full tank reproduces the previous fixed mass at spawn. Capacity 0 disables
        # fuel (fixed mass, no depletion). Empty mass is floored so a spent jet is
        # not unrealistically light.
        fuel_capacity_kg = float(config.get("fuel_capacity_kg", 0.0))
        empty_mass_kg = max(0.5 * cfg_mass, cfg_mass - fuel_capacity_kg)
        self.fuel = FuelSystem(self, fuel_capacity_kg, empty_mass_kg)

        self.weapon_lock_threshold_km = config.get(
            "weapon_lock_threshold_km",
            radar_max_range * 0.01 / 1000.0,
        )
        self.lock_threshold_km = self.weapon_lock_threshold_km
        self.missile_types = list(config.get("missile_types", []))
        self.max_missiles = config.get("max_missiles", 8)
        self.missiles = []
        self.target = None

        # Pre-compute missile parameters once at init to avoid creating full missile
        # objects on every NEZ/DLZ calculation.
        self.missile_params = {}
        for missile_class in self.missile_types:
            try:
                params = MissileParameters.from_missile_class(missile_class)
                self.missile_params[missile_class.__name__] = params
            except Exception as e:
                print(f"Warning: Could not extract parameters for {missile_class.__name__}: {e}")

        from bvr_marl_core.aircraft.systems.metrics_helper import MetricsHelper

        self.metrics = MetricsHelper(self)

    @property
    def locked_targets(self) -> set:
        """Expose radar locks on the aircraft object as the canonical lock set."""
        if hasattr(self, "sensor") and self.sensor is not None:
            try:
                return set(self.sensor.get_locked_targets() or [])
            except (AttributeError, TypeError, ValueError, KeyError, IndexError, ZeroDivisionError):
                pass
        if hasattr(self, "radar") and self.radar is not None:
            try:
                return set(self.radar.get_locked_targets() or [])
            except (AttributeError, TypeError, ValueError, KeyError, IndexError, ZeroDivisionError):
                pass
        return set()

    @locked_targets.setter
    def locked_targets(self, value) -> None:
        locked = set(value or [])
        if hasattr(self, "radar") and self.radar is not None:
            self.radar.locked_targets = locked

    @property
    def remaining_missiles(self) -> int:
        """Mirror the live missile inventory maintained by the weapon system."""
        if hasattr(self, "weapons") and self.weapons is not None:
            return int(getattr(self.weapons, "remaining_missiles", 0))
        return int(getattr(self, "_remaining_missiles", self.max_missiles))

    @remaining_missiles.setter
    def remaining_missiles(self, value: int) -> None:
        remaining = int(value)
        self._remaining_missiles = remaining
        if hasattr(self, "weapons") and self.weapons is not None:
            self.weapons.remaining_missiles = remaining

    def update(self, tick_secs: float, sim: Simulator) -> list[Event]:
        if not self.is_mortally_hit:
            self.sensor.update_sensor_data(sim, tick_secs)
        return self.update_after_staged_sensors(tick_secs, sim)

    def stage_sensor_reports(self, tick_secs: float, sim: Simulator) -> None:
        """Generate raw reports only; peer reports are not consumed in this phase."""
        if not self.is_mortally_hit and not self.should_be_removed:
            self.sensor.stage_sensor_reports(sim, tick_secs)

    def update_staged_sensor_products(self, tick_secs: float, sim: Simulator) -> None:
        """Fuse the globally frozen report set into this aircraft's local picture."""
        if not self.is_mortally_hit and not self.should_be_removed:
            self.sensor.update_from_staged_reports(sim, tick_secs)

    def update_after_staged_sensors(self, tick_secs: float, sim: Simulator) -> list[Event]:
        """Advance aircraft dynamics after every platform has published sensor products."""
        # Stochastic kill delay: once the scheduled death time arrives, fire the
        # (deferred) kill event so the shot is credited, then remove the wreck.
        if self.is_mortally_hit and not self.should_be_removed:
            now = float(getattr(sim, "elapsed_time_s", 0.0))
            if self._death_time_s is not None and now >= float(self._death_time_s):
                sim.log_event(AircraftDestroyedEvent(sim, self._death_killer, self))
                self.should_be_removed = True
                self.removal_reason = "missile_kill"
                return []
            # DYING aircraft retain only uncontrolled flight dynamics. They do
            # not receive a normal sensor update or re-enter tactical control.
            self.control.update_movement(tick_secs)
            self._publish_tick_path(sim)
            return []

        self.control.update_movement(tick_secs)
        self._publish_tick_path(sim)

        # Fuel exhaustion: a flamed-out jet glides (thrust cut in movement) and is
        # lost once it has descended to the altitude floor — a fuel-starvation crash.
        if getattr(self, "fuel", None) is not None and self.fuel.flamed_out:
            if self.position.alt <= self.min_alt_m + 50.0:
                self.should_be_removed = True
                if self.removal_reason is None:
                    self.removal_reason = "out_of_fuel"

        if self.boundary_violation_active:
            self.boundary_violation_countdown -= 1
            if self.boundary_violation_countdown <= 0:
                self.should_be_removed = True

        return []

    def _publish_tick_path(self, sim) -> None:
        """Expose the path just flown, stamped with the tick it belongs to.

        Missiles integrate after aircraft within a tick's dynamics stage (the
        roster is sorted with missiles last), so a missile reading this gets the
        current tick's real arc. The timestamp is what makes that safe: an
        aircraft skipped during a tick keeps a stale path, and a consumer must be
        able to tell. TickStateBuffer copies only position and the kinematic
        scalars, so these two attributes survive its restore/publish cycle.
        """
        self.tick_path = self.control.tick_path
        self.tick_path_time_s = float(getattr(sim, "elapsed_time_s", 0.0))

    def get_state_representation(self):
        state = {
            "name": self.name,
            "id": self.id,
            "position": {
                "lat": self.position.lat,
                "lon": self.position.lon,
                "alt": self.position.alt,
            },
            "yaw_deg": self.yaw_deg,
            "pitch_deg": getattr(self.control, "pitch_deg", 0.0),
            "roll_deg": getattr(self.control, "roll_deg", 0.0),
            "speed": self.speed,
            "missiles_loaded": len(self.missiles),
            "max_missiles": self.max_missiles,
            "flares": getattr(self, "flares", 0),
            "chaff": getattr(self, "chaff", 0),
            "ecm": getattr(self, "ecm", 0),
            "decoys": getattr(self, "decoys", 0),
        }

        if getattr(self, "target", None) is not None:
            dlz = self.wez.compute_dlz(self.target)
            rng = self.wez._slant_range_m(self, self.target)
            zone = self.wez.zone_for_range(rng, dlz)
            nez_on = self.wez.nez_visible(rng, dlz, show_in=("R2", "R3"))
            tip = self.wez.sqi(self, self.target, None, dlz)
            state["dlz"] = {
                "r_min_m": dlz.r_min_m,
                "r_tr_m": dlz.r_tr_m,
                "r_pi_m": dlz.r_pi_m,
                "r_aero_m": dlz.r_aero_m,
                "r_nez_in_m": dlz.r_nez_in_m,
                "r_nez_out_m": dlz.r_nez_out_m,
                "slant_range_m": rng,
            }
            state["dlz_zone"] = zone
            state["nez_visible"] = nez_on
            state["sqi"] = tip
        else:
            state.update({"dlz": None, "dlz_zone": None, "nez_visible": False, "sqi": 0.0})
        return state

    def apply_rl_action(self, action: np.ndarray, simulator: Simulator):
        # EMCON radar toggle (index 9, when present); silent only in the top
        # quartile so the radar stays ON by default.
        if isinstance(action, np.ndarray) and len(action) > 9:
            self.radar_emitting = bool(float(action[9]) < 0.75)

        # Allow shorter action vectors by padding with zeros
        a = np.zeros(9, dtype=float)
        if isinstance(action, np.ndarray):
            a[: min(len(action), 9)] = action[: min(len(action), 9)]

        self.control.set_throttle(min(max(float(a[0]), 0.0), 1.0))

        yaw_deg_change = (a[1] * 2.0 - 1.0) * 180.0
        pitch_deg_change = (a[2] * 2.0 - 1.0) * 90.0
        self.control.set_yaw_deg((self.yaw_deg + yaw_deg_change) % 360.0)
        self.control.set_pitch_deg(self.pitch_deg + pitch_deg_change)

        self.target = self.weapons.select_and_engage_target(
            self._get_target_candidates(simulator),
            a[4],  # target_selector
            a[3],  # fire_action (missile)
            a[5],  # gun_fire_action
            simulator,
        )

        if a[6] > 0.5:
            self.countermeasures.launch_flares(simulator)
        if a[7] > 0.5:
            self.countermeasures.launch_chaff(simulator)
        if a[8] > 0.5:
            self.countermeasures.deploy_decoys(simulator)

        self.countermeasures.cleanup_expired_countermeasures()

    def _get_target_candidates(self, simulator: Simulator):
        return [
            u
            for u in simulator.active_units.values()
            if u.group != self.group and not getattr(u, "is_missile", False)
        ]

    def substep_update(self, dt: float, sim) -> list:
        self.control.update_movement(dt)
        return []
