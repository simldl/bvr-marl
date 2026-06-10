from dataclasses import dataclass
from typing import Any

from bvr_marl_core.aircraft.aircraft import Aircraft
from bvr_marl_core.missiles.fox3.amraam import AIM120_AMRAAM
from bvr_marl_core.simulator.core.helpers import Position


class F22(Aircraft):
    @dataclass
    class Config:
        # --- References ---
        # USAF Fact Sheet (general performance): https://www.af.mil/About-Us/Fact-Sheets/Display/Article/104506/f-22-raptor/  :contentReference[oaicite:5]{index=5}
        # Wing area/wingspan (78.04 m², 13.56 m): https://en.wikipedia.org/wiki/Lockheed_Martin_F-22_Raptor  :contentReference[oaicite:6]{index=6}

        # Physics (geometry corrected)
        mass_kg: float = 19700.0  # empty mass (USAF)  :contentReference[oaicite:7]{index=7}
        reference_area_m2: float = 78.0  # ~840 ft^2  :contentReference[oaicite:8]{index=8}
        aspect_ratio: float = 2.36  # b^2/S = 13.56^2 / 78.04  :contentReference[oaicite:9]{index=9}
        oswald_e: float = 0.82
        max_speed_mps: float = (
            680.0  # ~Mach 2.0 at altitude  :contentReference[oaicite:10]{index=10}
        )
        n_max: float = 9.0
        stall0_mps: float = 75.0
        # Flight envelope
        min_speed_mps: float = 80.0
        max_climb_angle_deg: float = 70.0
        min_alt_m: float = 0.0
        max_alt_m: float = 20000.0
        # Radar (X-band AESA; placeholders tuned for sim)
        radar_horizontal_fov_deg: float = 120.0
        radar_vertical_fov_deg: float = 60.0
        radar_max_range_m: float = 230_000.0
        radar_frequency_hz: float = 9.5e9
        radar_tx_power_w: float = 25e3
        radar_antenna_gain_db: float = 40.0
        radar_snr_threshold_db: float = 8.0
        radar_beam_rate_hz: float = 10.0
        radar_beam_rate_p_hz: float = 8.0
        rcs: float = 0.0001  # public estimates vary; highly aspect/condition dependent
        # Passive Radar
        passive_radar_angular_error_deg: float = 5.0
        passive_radar_range_error_m: float = 2000.0
        passive_radar_max_age_s: float = 3.0
        # Missile Warner
        missile_warning_delay_s: float = 1.0
        missile_warning_delay_std: float = 0.5
        # Weapons
        missile_types: tuple = (AIM120_AMRAAM,)
        max_missiles: int = 8
        flares: int = 2
        chaff: int = 2
        ecm: int = 2
        decoys: int = 2
        # Gun configuration
        gun_config: dict = None

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

        # Add M61A2 Vulcan cannon configuration
        cfg.gun_config = {
            "max_ammo": 480,  # 480 rounds standard loadout
            "muzzle_velocity_mps": 1050.0,  # 20mm rounds at ~1050 m/s
            "max_range_m": 1500.0,  # Effective range ~1.5km
            "burst_size": 50,  # 50 rounds per trigger pull (realistic burst)
            "burst_duration_s": 0.5,  # 0.5 second burst (6000 rpm = 100 rps, so 50 rounds in 0.5s)
        }

        super().__init__(
            name="F22",
            position=position,
            yaw_deg=yaw_deg,
            speed_mps=speed_mps,
            group=group,
            map_limits=map_limits,
            config=cfg,
        )
        # RCS angular pattern tuned to public-order estimates (VLO front << beam) and to academic discussions.
        # Sources (general orders): MIT LL Radar primer; Skolnik; open-source summaries of -40 dBsm claims.
        self.rcs_pattern = {
            # Very low nose-on fraction; tail still low but a bit higher than nose.
            "front_floor": 0.20,  # φ=0° → ~0.2 * σ0  (nose-on)
            "tail_floor": 0.45,  # φ=180° → ~0.45 * σ0 (tail-on)
            "n_az": 1.5,  # sharper beam peak (steeper azimuth dependence)
            # Elevation: dorsal (top) modest gain; ventral (bottom) larger (inlets/bays).
            "k_top": 0.15,
            "k_bottom": 0.35,
            "n_el": 1.7,
            # Safety clamps (fractions of σ0); keep generous ceiling since σ0 is already tiny.
            "sigma_min": 0.05,
            "sigma_max": 5.0,
        }
