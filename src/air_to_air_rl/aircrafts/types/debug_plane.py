from dataclasses import dataclass
from typing import Any

from air_to_air_rl.aircrafts.aircraft import Aircraft
from air_to_air_rl.missiles.fox3.default_missile import LongRangeMissile
from air_to_air_rl.simulator.core.helpers import Position


class DebugPlane(Aircraft):
    @dataclass
    class Config:
        # Physics
        mass_kg: float = 20000.0
        reference_area_m2: float = 30.0
        aspect_ratio: float = 7.5
        oswald_e: float = 0.82
        max_speed_mps: float = 680.0
        n_max: float = 4.0
        stall0_mps: float = 65.0
        # Flight envelope
        min_speed_mps: float = 100.0
        max_climb_angle_deg: float = 60.0
        min_alt_m: float = 0.0
        max_alt_m: float = 20000.0
        # Radar
        radar_horizontal_fov_deg: float = 45.0
        radar_vertical_fov_deg: float = 30.0
        radar_max_range_m: float = 300_000.0
        radar_frequency_hz: float = 10e9
        radar_tx_power_w: float = 15e3
        radar_antenna_gain_db: float = 35.0
        radar_snr_threshold_db: float = 10.0
        rcs: float = 1000
        radar_beam_rate_hz: float = 5.0
        radar_beam_rate_p_hz: float = 3.0
        # Passive Radar
        passive_radar_angular_error_deg: float = 5.0
        passive_radar_range_error_m: float = 2000.0
        passive_radar_max_age_s: float = 3.0
        # Missile Warner
        missile_warning_delay_s: float = 1.0
        missile_warning_delay_std: float = 0.5
        # Weapons
        missile_types: tuple = (LongRangeMissile,)
        max_missiles: int = 3
        flares: int = 3
        chaff: int = 5
        ecm: int = 2
        decoys: int = 5

    def __init__(
        self,
        position: Position,
        yaw_deg: float,
        speed_mps: float,
        group: str,
        map_limits: Any,
        min_alt_m: float,
        max_alt_m: float,
    ):
        cfg = self.Config()
        cfg.min_alt_m = min_alt_m
        cfg.max_alt_m = max_alt_m
        super().__init__(
            name="DebugPlane",
            position=position,
            yaw_deg=yaw_deg,
            speed_mps=speed_mps,
            group=group,
            map_limits=map_limits,
            config=cfg,
        )
