from dataclasses import dataclass
from typing import Any, Tuple
from simulator.core.helpers import Position
from missiles.fox3.amraam import AIM120_AMRAAM
from aircrafts.aircraft import Aircraft

# ---------------------------------------------
# F35
# ---------------------------------------------
class F35(Aircraft):
    @dataclass
    class Config:
        # --- References ---
        # USAF Fact Sheet (general performance): https://www.af.mil/About-Us/Fact-Sheets/Display/Article/478441/f-35a-lightning-ii/  :contentReference[oaicite:12]{index=12}
        # Wing area 42.7 m² / span 10.7 m: https://en.wikipedia.org/wiki/Lockheed_Martin_F-35_Lightning_II  :contentReference[oaicite:13]{index=13}

        # Physics (geometry corrected)
        mass_kg: float = 13300.0
        reference_area_m2: float = 42.7        # 460 ft^2  :contentReference[oaicite:14]{index=14}
        aspect_ratio: float = 2.68             # 10.7^2 / 42.7  :contentReference[oaicite:15]{index=15}
        oswald_e: float = 0.82
        max_speed_mps: float = 540.0           # ~Mach 1.6  :contentReference[oaicite:16]{index=16}
        n_max: float = 9.0
        stall0_mps: float = 68.0
        # Flight envelope
        min_speed_mps: float = 70.0
        max_climb_angle_deg: float = 60.0
        min_alt_m: float = 0.0
        max_alt_m: float = 20000.0
        # Radar
        radar_horizontal_fov_deg: float = 120.0
        radar_vertical_fov_deg: float = 60.0
        radar_max_range_m: float = 150_000.0
        radar_frequency_hz: float = 9.8e9
        radar_tx_power_w: float = 20e3
        radar_antenna_gain_db: float = 38.0
        radar_snr_threshold_db: float = 8.0
        radar_beam_rate_hz: float = 8.0
        radar_beam_rate_p_hz: float = 6.0
        rcs: float = 0.005
        # Passive Radar
        passive_radar_angular_error_deg: float = 5.0
        passive_radar_range_error_m:   float = 2000.0
        passive_radar_max_age_s:       float = 3.0
        # Missile Warner
        missile_warning_delay_s:   float = 1.0
        missile_warning_delay_std: float = 0.5
        # Weapons
        missile_types: Tuple = (AIM120_AMRAAM,)
        max_missiles: int = 6
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
        max_alt_m: float
    ):
        cfg = self.Config()
        cfg.min_alt_m = min_alt_m
        cfg.max_alt_m = max_alt_m
        super().__init__(
            name="F35",
            position=position,
            yaw_deg=yaw_deg,
            speed_mps=speed_mps,
            group=group,
            map_limits=map_limits,
            config=cfg
        )
        # RCS pattern: VLO head-on, a bit less extreme than F-22; side/tail closer to baseline.
        # Backed by open-source engineering sims (POFACETS F-35 az/el sweeps) and general VLO ranges.
        self.rcs_pattern = {
            "front_floor": 0.30,   # nose-on fraction a bit higher than F-22
            "tail_floor":  0.55,
            "n_az":        1.3,    # still strong azimuth shaping, slightly softer than F-22
            "k_top":       0.18,
            "k_bottom":    0.32,
            "n_el":        1.6,
            "sigma_min":   0.05,
            "sigma_max":   4.0,
        }