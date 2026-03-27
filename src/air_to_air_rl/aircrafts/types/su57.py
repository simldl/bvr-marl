from dataclasses import dataclass
from typing import Any

from air_to_air_rl.aircrafts.aircraft import Aircraft
from air_to_air_rl.missiles.fox3.k77m import K77M


class Su57(Aircraft):
    @dataclass
    class Config:
        # === Sources (public) ===
        # General dimensions, weights, performance (wingspan 14.1 m, wing area 78.8 m²,
        # empty weight ~18,500 kg, max speed Mach 2, service ceiling 20,000 m):
        #   https://en.wikipedia.org/wiki/Sukhoi_Su-57
        # Radar suite (N036 “Byelka/Belka”: X-band AESA nose + X-band cheek arrays + L-band LEVCON arrays):
        #   https://en.wikipedia.org/wiki/Byelka_%28radar%29
        # Notes:
        # * Many parameters (RCS, radar power/gain) are not officially published; values below are
        #   reasonable, documented placeholders suitable for simulation. Adjust per your tuning.

        # --- Flight Physics / Geometry (SI units) ---
        # Empty mass (kg). Wiki lists empty ~18,500 kg.
        mass_kg: float = 18_500.0  # [Ref: Wikipedia Su-57 — empty weight]
        # Wing reference area S (m²); Wikipedia: 78.8 m²
        reference_area_m2: float = 78.8  # [Ref: Wikipedia Su-57 — wing area]
        # Aspect ratio AR = b² / S with b = 14.1 m ⇒ AR ≈ 14.1² / 78.8 ≈ 2.52
        aspect_ratio: float = 2.52  # [Computed from Wikipedia wingspan & area]
        # Oswald efficiency factor (typical clean high-performance)
        oswald_e: float = 0.82

        # Max speed at altitude ≈ Mach 2.0 ⇒ ~2,135 km/h ⇒ ~592 m/s
        max_speed_mps: float = 595.0  # [Ref: Wikipedia Su-57 — maximum speed]
        n_max: float = 9.0  # [Ref: Wikipedia Su-57 — g limits +9]
        stall0_mps: float = 70.0  # heuristic for sim’s stall model

        # Envelope (altitudes)
        min_speed_mps: float = 80.0
        max_climb_angle_deg: float = 70.0
        min_alt_m: float = 0.0
        max_alt_m: float = 20_000.0  # [Ref: Wikipedia Su-57 — service ceiling 20,000 m]

        # --- Radar / Sensors (Byelka N036 AESA + L-band) ---
        # Public sources indicate X-band AESA frontal array, X-band cheek arrays, L-band LEVCON arrays (IFF/EW).
        # Frequencies in X-band ~ 8–12 GHz; we choose 9.5 GHz nominal.
        radar_horizontal_fov_deg: float = (
            140.0  # Byelka has wide coverage with cheek arrays (sim nominal)
        )
        radar_vertical_fov_deg: float = 60.0
        radar_max_range_m: float = (
            300_000.0  # public estimates vary ~200–400 km vs fighter-size target
        )
        radar_frequency_hz: float = 9.5e9  # X-band nominal center frequency
        radar_tx_power_w: float = 20e3  # plausible order-of-magnitude placeholder
        radar_antenna_gain_db: float = 38.0  # plausible AESA nose-array gain placeholder
        radar_snr_threshold_db: float = 8.0  # sim detection threshold
        radar_beam_rate_hz: float = 10.0
        radar_beam_rate_p_hz: float = 8.0

        # --- RCS (very scenario/angle dependent; public numbers vary widely) ---
        # Use a conservative front-hemisphere placeholder for relative sim balancing.
        rcs: float = 0.1  # m^2 (note: highly uncertain; for sim use only)

        # --- Passive / MWS / Stores (inherit reasonable defaults or tune as needed) ---
        passive_radar_angular_error_deg: float = 5.0
        passive_radar_range_error_m: float = 2000.0
        passive_radar_max_age_s: float = 3.0

        # Missile warning system (notional)
        mws_detection_delay_s: float = 0.6
        mws_detection_delay_std: float = 0.2

        # Weapons
        missile_types: tuple = (K77M,)
        max_missiles: int = 8
        flares: int = 2
        chaff: int = 2
        ecm: int = 2
        decoys: int = 2

        # Stores, gun, pylons etc. (keep aligned with other types in your sim)
        internal_gun_effective_range_m: float = 800.0
        internal_gun_damage: float = 45.0

    def __init__(self, name: str = "Su-57", config: "Su57.Config | None" = None, **kwargs):
        cfg = config or Su57.Config()
        super().__init__(name=name, config=cfg, **kwargs)

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
