from dataclasses import asdict
from typing import List
import numpy as np
from simulator.simulator import Simulator
from simulator.core.events import Event
from simulator.core.units import FlyingUnit
from physics.aircraft import AircraftPhysics
from radar.units.aircraft import AircraftRadar
from radar.core.data_link import DataLink    
from aircrafts.control.movement_control import AircraftControlSystem
from aircrafts.systems.sensor import AircraftSensorSystem
from aircrafts.control.weapon_system import AircraftWeaponSystem
from aircrafts.control.countermeasure import AircraftCountermeasureSystem
from aircrafts.core.nez import NoEscapeZoneCalculator
from missiles.missile_parameters import MissileParameters

class Aircraft(FlyingUnit):
    def __init__(self, name, position, yaw_deg, speed_mps, group, map_limits, config):
        super().__init__(name=name, position=position, yaw_deg=yaw_deg, speed=speed_mps)
        self.group = group
        self.map_limits = map_limits
        self.type = "Aircraft"
        self.wez = NoEscapeZoneCalculator(self)

        # Removal flags for boundary violations
        self.should_be_removed = False
        self.removal_reason = None

        # Boundary violation state tracking
        self.boundary_violation_active = False
        self.boundary_violation_countdown = 0
        self.boundary_violation_penalty_per_step = -10.0  # Applied each step during violation

        if hasattr(config, "__dataclass_fields__"):
            config = asdict(config)
        self.config = config

        # --- Flight Physics & Envelopes ---
        cfg_mass   = config.get("mass_kg", 20000.0)
        cfg_Sref   = config.get("reference_area_m2", 27.87)
        cfg_AR     = config.get("aspect_ratio", 4.0)
        cfg_oswald = config.get("oswald_e", 0.78)
        cfg_vmax   = float(config.get("max_speed_mps", 680.0))   
        cfg_nmax   = float(config.get("n_max", 9.0))

        self.physics = AircraftPhysics(AircraftPhysics.Params(
            mass_kg=cfg_mass,
            reference_area_m2=cfg_Sref,
            aspect_ratio=cfg_AR,
            oswald_e=cfg_oswald,
            max_speed_mps=cfg_vmax,
            n_max=cfg_nmax,
            stall0_mps=config.get("stall0_mps", 60.0),
        ))

        # Mirror key envelope limits on the entity from the same values
        self.min_speed_mps = config.get("min_speed_mps", 0.0)
        self.max_speed_mps = cfg_vmax
        self.n_max         = cfg_nmax
        self.min_alt_m     = config.get("min_alt_m", getattr(map_limits, "min_alt", 0.0))
        self.max_alt_m     = config.get("max_alt_m", getattr(map_limits, "max_alt", 20000.0))

        # --- Radar & DataLink ---
        radar_max_range = config.get("radar_max_range_m", 60000.0)
        self.radar = AircraftRadar(
            horizontal_fov_deg=config.get("radar_horizontal_fov_deg", 60.0),
            vertical_fov_deg=config.get("radar_vertical_fov_deg", 30.0),
            max_range_m=radar_max_range,
            radar_frequency_hz=config.get("radar_frequency_hz", 10e9),
            tx_power_w=config.get("radar_tx_power_w", 5e3),
            antenna_gain_db=config.get("radar_antenna_gain_db", 30.0),
            snr_threshold_db=config.get("snr_threshold_db", 10.0),
            owner=self,
            data_link=DataLink(mode="full"),
        )

        self.rcs = config.get("rcs", 10.0)
        self.rcs_pattern = self.config.get("rcs_pattern", {
            # Front/beam/tail & top/bottom shaping
            "front_floor": 0.25 if "F-22" in self.name else 0.35,
            "tail_floor":  0.50,
            "n_az":        1.2,
            "k_top":       0.20,
            "k_bottom":    0.35,
            "n_el":        1.5,
            "sigma_min":   0.03,
            "sigma_max":   3.0,
        })

        self.passive_radar_angular_error_deg = config.get("passive_radar_angular_error_deg", 5.0)
        self.passive_radar_range_error_m     = config.get("passive_radar_range_error_m", 2000.0)
        self.passive_radar_max_age_s         = config.get("passive_radar_max_age_s", 3.0)

        self.missile_warning_delay_s   = config.get("missile_warning_delay_s", 1.0)
        self.missile_warning_delay_std = config.get("missile_warning_delay_std", 0.5)

        # --- Subsystems ---
        self.sensor = AircraftSensorSystem(self)
        self.control = AircraftControlSystem(self)
        self.weapons = AircraftWeaponSystem(self)
        self.countermeasures = AircraftCountermeasureSystem(self)

        # --- Flight Limits & Weapons ---
        self.min_speed_mps = config.get("min_speed_mps", 0.0)
        self.max_speed_mps = config.get("max_speed_mps", 340.0)
        self.n_max = config.get("n_max", 9.0)
        self.min_alt_m = config.get("min_alt_m", getattr(map_limits, "min_alt", 0.0))
        self.max_alt_m = config.get("max_alt_m", getattr(map_limits, "max_alt", 20000.0))
        self.weapon_lock_threshold_km = config.get("weapon_lock_threshold_km",
            config.get("radar_max_range_m", 20000) * 0.01 / 1000)
        self.lock_threshold_km = self.weapon_lock_threshold_km
        self.missile_types = list(config.get("missile_types", []))
        self.max_missiles = config.get("max_missiles", 8)
        self.missiles = []
        self.target = None

        # --- Missile Parameter Templates (Performance Optimization) ---
        # Pre-compute missile parameters once at initialization instead of creating
        # full missile objects during every NEZ/DLZ calculation.
        # This eliminates ~78% of simulation overhead (see PERFORMANCE_ANALYSIS.md)
        self.missile_params = {}
        for missile_class in self.missile_types:
            try:
                params = MissileParameters.from_missile_class(missile_class)
                self.missile_params[missile_class.__name__] = params
            except Exception as e:
                print(f"Warning: Could not extract parameters for {missile_class.__name__}: {e}")

        # Initialize metrics helper for tactical calculations
        from aircrafts.systems.metrics_helper import MetricsHelper
        self.metrics = MetricsHelper(self)
                # Continue without this missile type

    def update(self, tick_secs: float, sim: Simulator) -> List[Event]:
        self.control.update_movement(tick_secs)
        self.sensor.update_sensor_data(sim, tick_secs)

        # Handle boundary violation countdown
        if self.boundary_violation_active:
            self.boundary_violation_countdown -= 1
            if self.boundary_violation_countdown <= 0:
                # Time's up - mark for immediate removal
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
            nez_on = self.wez.nez_visible(rng, dlz, show_in=("R2","R3"))
            tip = self.wez.sqi(self, self.target, None, dlz)
            state["dlz"] = {
                "r_min_m": dlz.r_min_m, "r_tr_m": dlz.r_tr_m,
                "r_pi_m": dlz.r_pi_m, "r_aero_m": dlz.r_aero_m,
                "r_nez_in_m": dlz.r_nez_in_m, "r_nez_out_m": dlz.r_nez_out_m,
                "slant_range_m": rng,
            }
            state["dlz_zone"] = zone
            state["nez_visible"] = nez_on
            state["sqi"] = tip
        else:
            state.update({"dlz": None, "dlz_zone": None, "nez_visible": False, "sqi": 0.0})
        return state

    def apply_rl_action(self, action: np.ndarray, simulator: Simulator):
        # Defensive: allow shorter vectors (fill with zeros)
        a = np.zeros(10, dtype=float)
        if isinstance(action, np.ndarray):
            a[:min(len(action), 10)] = action[:min(len(action), 10)]

        # Throttle (0..1)
        self.control.set_throttle(np.clip(a[0], 0.0, 1.0))

        # Yaw/pitch deltas
        yaw_deg_change   = (a[1] * 2.0 - 1.0) * 180.0
        pitch_deg_change = (a[2] * 2.0 - 1.0) * 90.0
        self.control.set_yaw_deg((self.yaw_deg + yaw_deg_change) % 360.0)
        self.control.set_pitch_deg(self.pitch_deg + pitch_deg_change)

        # Targeting & fire (missile + gun)
        self.target = self.weapons.select_and_engage_target(
            self._get_target_candidates(simulator),
            a[3],  # target_selector
            a[4],  # fire_action (missile)
            a[5],  # gun_fire_action
            simulator
        )

        # Countermeasures (indices 6..9) - now create physical objects in simulation
        if a[6] > 0.5: self.countermeasures.launch_flares(simulator)
        if a[7] > 0.5: self.countermeasures.launch_chaff(simulator)
        if a[8] > 0.5: self.countermeasures.activate_ecm()
        if a[9] > 0.5: self.countermeasures.deploy_decoys(simulator)

        # Clean up expired countermeasures
        self.countermeasures.cleanup_expired_countermeasures()


    def _get_target_candidates(self, simulator: Simulator):
        # Selection of all valid target units (no missiles, not own group)
        return [
            u for u in simulator.active_units.values()
            if u.group != self.group and not getattr(u, "is_missile", False)
        ]

    def substep_update(self, dt: float, sim) -> list:
        self.control.update_movement(dt)
        return []