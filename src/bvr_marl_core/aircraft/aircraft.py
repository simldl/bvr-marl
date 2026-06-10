from dataclasses import asdict

import numpy as np

from bvr_marl_core.aircraft.control.countermeasure import AircraftCountermeasureSystem
from bvr_marl_core.aircraft.control.movement_control import AircraftControlSystem
from bvr_marl_core.aircraft.control.weapon_system import AircraftWeaponSystem
from bvr_marl_core.aircraft.core.nez import NoEscapeZoneCalculator
from bvr_marl_core.aircraft.systems.sensor import AircraftSensorSystem
from bvr_marl_core.missiles.missile_parameters import MissileParameters
from bvr_marl_core.physics.aircraft import AircraftPhysics
from bvr_marl_core.radar.core.data_link import DataLink
from bvr_marl_core.radar.obs.observation import DEFAULT_NOTCH_VELOCITY_MPS
from bvr_marl_core.radar.units.aircraft import AircraftRadar
from bvr_marl_core.simulator.core.events import Event
from bvr_marl_core.simulator.core.units import FlyingUnit
from bvr_marl_core.simulator.simulator import Simulator


class Aircraft(FlyingUnit):
    unit_kind: str = "aircraft"

    def __init__(self, name, position, yaw_deg, speed_mps, group, map_limits, config):
        super().__init__(name=name, position=position, yaw_deg=yaw_deg, speed=speed_mps)
        self.group = group
        self.map_limits = map_limits
        self.type = "Aircraft"
        self.wez = NoEscapeZoneCalculator(self)

        self.should_be_removed = False
        self.removal_reason = None
        self.boundary_violation_active = False
        self.boundary_violation_countdown = 0
        self.boundary_violation_penalty_per_step = -10.0

        if hasattr(config, "__dataclass_fields__"):
            config = asdict(config)
        self.config = config

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

        self.sensor = AircraftSensorSystem(self)
        self.control = AircraftControlSystem(self)
        self.weapons = AircraftWeaponSystem(self)
        self.countermeasures = AircraftCountermeasureSystem(self)

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
            except Exception:
                pass
        if hasattr(self, "radar") and self.radar is not None:
            try:
                return set(self.radar.get_locked_targets() or [])
            except Exception:
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
        self.control.update_movement(tick_secs)
        self.sensor.update_sensor_data(sim, tick_secs)

        if self.boundary_violation_active:
            self.boundary_violation_countdown -= 1
            if self.boundary_violation_countdown <= 0:
                self.should_be_removed = True

        return []

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
        # Allow shorter action vectors by padding with zeros
        a = np.zeros(10, dtype=float)
        if isinstance(action, np.ndarray):
            a[: min(len(action), 10)] = action[: min(len(action), 10)]

        self.control.set_throttle(np.clip(a[0], 0.0, 1.0))

        yaw_deg_change = (a[1] * 2.0 - 1.0) * 180.0
        pitch_deg_change = (a[2] * 2.0 - 1.0) * 90.0
        self.control.set_yaw_deg((self.yaw_deg + yaw_deg_change) % 360.0)
        self.control.set_pitch_deg(self.pitch_deg + pitch_deg_change)

        self.target = self.weapons.select_and_engage_target(
            self._get_target_candidates(simulator),
            a[3],  # target_selector
            a[4],  # fire_action (missile)
            a[5],  # gun_fire_action
            simulator,
        )

        if a[6] > 0.5:
            self.countermeasures.launch_flares(simulator)
        if a[7] > 0.5:
            self.countermeasures.launch_chaff(simulator)
        if a[8] > 0.5:
            self.countermeasures.activate_ecm()
        if a[9] > 0.5:
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
