# Sources:
# - MBDA official datasheet (PDF): https://www.mbda-systems.com/sites/mbda/files/2024-06/2023%20METEOR%20datasheet.pdf
# - MBDA US site: https://mbdainc.com/products/meteor/
# - Wikipedia summary (cross-check dims): https://en.wikipedia.org/wiki/MBDA_Meteor

from missiles.missile import Missile

class Meteor(Missile):
    def __init__(self, firing_time_s, target, source, map_limits, group=None):
        # Public dims: mass ~190 kg, Ø 178 mm (Sref ≈ 0.0249 m²), length ~3.65 m, wingspan ~0.54 m.
        # Ramjet throttleable sustain; here modeled as long burn constant thrust surrogate.
        # Range officially 100km+
        config = {
            "mass_kg": 190.0,
            "reference_area_m2": 0.025,       # π*(0.089)^2 ≈ 0.0249 m²
            "aspect_ratio": 3.0,
            "oswald_e": 0.80,
            "drag_coefficient": 0.07,

            # Surrogate ramjet: lower thrust but long burn to keep Mach high across envelope.
            "constant_engine_N": 6000.0,
            "motor_burn_s": 60.0,             # long sustain (sim surrogate for ducted rocket)

            "n_max": 40.0,
            "max_speed_mps": 1360.0,          # Mach >4 class (sea-level equivalent)
            "min_range_m": 2000.0,

            "seeker_sensitivity": 1.3,
            "life_time_s": 160.0,             # long time-of-flight compatible with LR shots
            "hit_probability": 0.88,

            "radar": {
                "horizontal_fov_deg": 60.0,
                "vertical_fov_deg": 30.0,
                "max_range_m": 200_000.0,     # true kinematic range (100-200+ km real-world; using 200 km per MBDA specs)
                "radar_frequency_hz": 10e9,
                "tx_power_w": 9e3,            # surrogate
                "antenna_gain_db": 34.0,      # surrogate
                "snr_threshold_db": 8.0,
            }
        }
        super().__init__(
            name="METEOR",
            firing_time_s=firing_time_s,
            target=target,
            source=source,
            map_limits=map_limits,
            group=group or source.group,
            config=config,
            data_link_mode="full"             # Meteor: network-enabled two-way DL (public)
        )
        self.fox_type = 3