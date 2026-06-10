from dataclasses import dataclass
from typing import Any

from bvr_marl_core.aircraft.aircraft import Aircraft
from bvr_marl_core.missiles.fox3.meteor import Meteor
from bvr_marl_core.simulator.core.helpers import Position


class Eurofighter(Aircraft):
    @dataclass
    class Config:
        # --- References ---
        # OEM site (performance overview): https://www.eurofighter.com/the-aircraft/performance  :contentReference[oaicite:18]{index=18}
        # Geometry (span 10.95 m, area 50 m², AR ~ 2.40): https://de.wikipedia.org/wiki/Eurofighter_Typhoon  :contentReference[oaicite:19]{index=19}

        # Physics (geometry corrected)
        mass_kg: float = 11000.0
        reference_area_m2: float = 50.0
        aspect_ratio: float = 2.40
        oswald_e: float = 0.82
        max_speed_mps: float = (
            680.0  # ~Mach 2.3 at altitude  :contentReference[oaicite:20]{index=20}
        )
        n_max: float = 9.0
        stall0_mps: float = 70.0
        # Flight envelope
        min_speed_mps: float = 70.0
        max_climb_angle_deg: float = 70.0
        min_alt_m: float = 0.0
        max_alt_m: float = 20000.0
        # Radar
        radar_horizontal_fov_deg: float = 120.0
        radar_vertical_fov_deg: float = 60.0
        radar_max_range_m: float = 185_000.0
        radar_frequency_hz: float = 10e9
        radar_tx_power_w: float = 18e3
        radar_antenna_gain_db: float = 36.0
        radar_snr_threshold_db: float = 9.0
        radar_beam_rate_hz: float = 12.0
        radar_beam_rate_p_hz: float = 10.0
        rcs: float = 3.0
        # Passive Radar
        passive_radar_angular_error_deg: float = 5.0
        passive_radar_range_error_m: float = 2000.0
        passive_radar_max_age_s: float = 3.0
        # Missile Warner
        missile_warning_delay_s: float = 1.0
        missile_warning_delay_std: float = 0.5
        # Weapons
        missile_types: tuple = (Meteor,)
        max_missiles: int = 10
        flares: int = 2
        chaff: int = 2
        ecm: int = 2
        decoys: int = 2

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
            name="Eurofighter",
            position=position,
            yaw_deg=yaw_deg,
            speed_mps=speed_mps,
            group=group,
            map_limits=map_limits,
            config=cfg,
        )
        # RCS pattern: LO, not as deep as F-22/35 in front aspect; stronger ventral sensitivity.
        # Aligned with public modeling papers and general LO expectations (orders of magnitude only).
        self.rcs_pattern = {
            "front_floor": 0.50,  # nose-on is reduced but not VLO-deep
            "tail_floor": 0.70,  # tail-on slightly higher fraction
            "n_az": 1.2,  # noticeable azimuth dependence
            "k_top": 0.20,
            "k_bottom": 0.40,  # more ventral gain than dorsal
            "n_el": 1.5,
            "sigma_min": 0.10,
            "sigma_max": 3.5,
        }
