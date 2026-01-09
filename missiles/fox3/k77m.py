# Sources:
# - Wikipedia R-77 page (R-77M range figure 193 km noted): https://en.wikipedia.org/wiki/R-77
# - Defence-ua explainer (claim ~190 km and dual-pulse motor): https://en.defence-ua.com/weapon_and_tech/izdelie_180_k_77m_r_77m_what_we_know_about_russias_new_long_range_missile-15268.html

from missiles.missile import Missile

class K77M(Missile):
    def __init__(self, firing_time_s, target, source, map_limits, group=None):
        # Assume similar diameter to R-77 family (0.20 m), mass ~200 kg class (not officially published).
        config = {
            "mass_kg": 200.0,
            "reference_area_m2": 0.031,      # Ø 200 mm cross-section
            "aspect_ratio": 3.3,
            "oswald_e": 0.78,
            "drag_coefficient": 0.075,

            # Dual-pulse surrogate: slightly longer burn & higher thrust than R-77-1
            "constant_engine_N": 11000.0,
            "motor_burn_s": 12.0,

            "n_max": 40.0,
            "max_speed_mps": 1500.0,         # high-Mach envelope (surrogate)
            "min_range_m": 2000.0,

            "seeker_sensitivity": 1.3,       # AESA seeker (publicly claimed)
            "life_time_s": 150.0,
            "hit_probability": 0.85,

            "radar": {
                "horizontal_fov_deg": 60.0,
                "vertical_fov_deg": 30.0,
                "max_range_m": 45_000.0,     # ARH acquisition (nominal)
                "radar_frequency_hz": 10e9,
                "tx_power_w": 9e3,
                "antenna_gain_db": 34.0,
                "snr_threshold_db": 8.0,
            }
        }
        super().__init__(
            name="K77M",
            firing_time_s=firing_time_s,
            target=target,
            source=source,
            map_limits=map_limits,
            group=group or source.group,
            config=config,
            data_link_mode="msl_support"      # use your restricted missile DL mode
        )
        self.fox_type = 3