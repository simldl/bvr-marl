from simulator.simulator import Simulator
from simulator.core.units import FlyingUnit
from simulator.core.events import Event
from missiles.core.engine import MissileEngine
from missiles.core.movement import MissileMovement
from missiles.guidance.target_provider import GuidanceTargetProvider
from missiles.guidance.guidance import MissileGuidance
from missiles.core.phases import MissilePhaseManager
from simulator.core.hit_calculation import MissileHitCalculator
from radar.units.missile import MissileRadar
from radar.core.data_link import DataLink
from physics.missiles import MissilePhysics
import numpy as np
from dataclasses import asdict

class Missile(FlyingUnit):
    def __init__(
        self,
        name: str,
        firing_time_s,
        target,
        source,
        map_limits,
        group: str,
        config: dict,
        data_link_mode: str = "full"
    ):
        cfg = asdict(config) if not isinstance(config, dict) else config

        super().__init__(
            name=name,
            position=source.position.copy(),
            yaw_deg=source.yaw_deg,
            speed=source.speed,
            pitch_deg=source.pitch_deg,
            roll_deg=0.0
        )
        
        self.type = "Missile"
        self.source = source
        self.target = target
        self.group = group if group else source.group
        self.map_limits = map_limits
        self.firing_time_s = firing_time_s
        self.max_speed_mps = cfg["max_speed_mps"]
        self.hit_probability = cfg["hit_probability"]
        self.life_time_s = cfg["life_time_s"]
        self.motor_burn_s = cfg.get("motor_burn_s", 30.0)
        self.elapsed_time_s = 0.0
        self.is_missile = True

        self.desired_yaw_deg = self.yaw_deg
        self.desired_pitch_deg = self.pitch_deg
        self.latest_target_velocity = np.zeros(3)
        self.phase_manager = MissilePhaseManager(cfg.get("flight_phases"),
                                                life_time_s=cfg["life_time_s"],
                                                motor_burn_s=cfg.get("motor_burn_s", 30.0))
        self.engine = MissileEngine(self, cfg.get("motor_burn_s", 30.0))
        self.physics = self._init_physics(cfg)
        self.movement = MissileMovement(self, self.physics)
        self.radar = self._init_radar(cfg["radar"], data_link_mode)
        self.target_provider = GuidanceTargetProvider(self)
        self.radar.target_provider = self.target_provider
        self.guidance = MissileGuidance(self, self.target_provider)
        self.hitCalculator = MissileHitCalculator(self)

    def update(self, tick_secs: float, sim: Simulator) -> list[Event]:
        self.elapsed_time_s += tick_secs
        self.phase_manager.update(self.elapsed_time_s)        
        self.engine.update(tick_secs, sim)

        # Check engine-based energy removal (after fuel depletion)
        if self.engine.should_remove_missile_on_energy(self):
            sim.remove_unit(self.id)
            return []

        # Gather all potential targets for the radar to track
        # This allows retargeting if the missile gets a better lock on a different target
        radar_targets = []
        for unit in sim.active_units.values():
            # Find units from opposing group (enemies) that are aircraft (not missiles or countermeasures)
            if (hasattr(unit, 'group') and unit.group != self.group and
                not getattr(unit, 'is_missile', False) and
                not getattr(unit, 'is_countermeasure', False)):
                radar_targets.append(unit)

        self.radar.update(
            tick_secs,
            sim,
            targets=radar_targets,
            owner_position=self.position,
        )

        _ti = self.radar.get_tracker_info()
        if _ti is not None and _ti.get("velocity") is not None:
            try:
                self.latest_target_velocity = np.array(_ti["velocity"], dtype=float)
            except Exception:
                pass

        self.target_provider.update(sim, tick_secs)
        yaw, pch = self.guidance.compute_guidance(
            self,
            self.target_provider,
            self.radar.tracker_manager,
            tick_secs
        )
        self.desired_yaw_deg = yaw
        self.desired_pitch_deg = pch

        self.movement.update(tick_secs)

        # Check for missile expiration conditions AFTER movement update
        # This ensures we catch boundary violations that occur during this tick
        if self._should_remove_missile(sim):
            sim.remove_unit(self.id)
            return []

        return []

    def _should_remove_missile(self, sim) -> bool:
        """
        Determine if missile should be removed from simulation.
        Returns True if missile should be removed.
        """
        # 1. Boundary violation check
        if hasattr(self, 'map_limits') and self.map_limits is not None:
            lat_violated = (self.position.lat <= self.map_limits.bottom_lat or
                           self.position.lat >= self.map_limits.top_lat)
            lon_violated = (self.position.lon <= self.map_limits.left_lon or
                           self.position.lon >= self.map_limits.right_lon)
            if lat_violated or lon_violated:
                return True
        
        # 2. Lifetime expiration
        life_time_s = getattr(self, 'life_time_s', 100.0)
        
        if self.elapsed_time_s >= life_time_s:
            return True
        
        # 3. Out of fuel and low energy (long glide phase)
        if self.engine.fuel_s == 0.0:
            # Check if missile has been gliding for too long after fuel depletion
            motor_burn_s = getattr(self, 'motor_burn_s', 12.0)
            glide_time = self.elapsed_time_s - motor_burn_s
            max_glide_time = 120.0

            if glide_time > max_glide_time:
                return True
        
        # 4. Range-based removal (flew too far from any valid targets)
        if hasattr(self, 'target') and self.target is not None:
            try:
                # Check distance to original target
                from simulator.core.helpers import units_distance_km
                dist_km = units_distance_km(self, self.target)

                # If missile is very far from target and out of fuel, remove it
                if self.engine.fuel_s == 0.0 and dist_km > 100.0:  # 100km threshold
                    return True
                    
            except Exception as e:
                print(f"[DEBUG] Exception in range-based removal for missile {self.id}: {e}")
                pass
        
        # 5. Lost target and can't reacquire
        locked_target = None
        if hasattr(self.radar, 'get_locked_target'):
            try:
                locked_target = self.radar.get_locked_target()
            except Exception:
                pass
        
        # If missile has no target lock and has been flying for a while, remove it  
        if locked_target is None and self.elapsed_time_s > 120.0:
            return True
        
        return False

    def _init_physics(self, cfg: dict) -> MissilePhysics:
        return MissilePhysics(
            MissilePhysics.Params(
                mass_kg=cfg["mass_kg"],
                reference_area_m2=cfg["reference_area_m2"],
                aspect_ratio=cfg["aspect_ratio"],
                oswald_e=cfg["oswald_e"],
                cd0=cfg["drag_coefficient"],
                constant_engine_F=cfg["constant_engine_N"],
                n_max=cfg["n_max"],
                max_speed_mps=cfg["max_speed_mps"]
            )
        )

    def _init_radar(self, rc: dict, data_link_mode: str) -> MissileRadar:
        return MissileRadar(
            horizontal_fov_deg=rc["horizontal_fov_deg"],
            vertical_fov_deg=rc["vertical_fov_deg"],
            max_range_m=rc["max_range_m"],
            radar_frequency_hz=rc["radar_frequency_hz"],
            tx_power_w=rc["tx_power_w"],
            antenna_gain_db=rc["antenna_gain_db"],
            snr_threshold_db=rc["snr_threshold_db"],
            false_alarm_rate=rc.get("false_alarm_rate", 0.005),
            range_resolution_m=rc.get("range_resolution_m", 100.0),
            angular_resolution_deg=rc.get("angular_resolution_deg", 2.0),
            lut_bins=rc.get("lut_bins", (256, 256)),
            owner=self,
            data_link=DataLink(data_link_mode),
            initial_datalink_mode="other",
            original_data_link_mode=data_link_mode
        )

    def substep_update(self, dt: float, sim) -> list:
        try:
            self.target_provider.update(sim, dt)
        except Exception:
            pass

        yaw, pch = self.guidance.compute_guidance(
            self, self.target_provider, self.radar.tracker_manager, dt
        )
        self.desired_yaw_deg = yaw
        self.desired_pitch_deg = pch

        self.movement.update(dt)
        return []