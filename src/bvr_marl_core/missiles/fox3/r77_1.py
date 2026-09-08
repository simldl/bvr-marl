# Sources:
# - R-77 page (R-77-1 specs: mass ~190 kg, length 3.71 m, Ø 200 mm, range 110 km): https://en.wikipedia.org/wiki/R-77
# - Missilery info (RVV-SD overview): https://en.missilery.info/missile/rvv-cd
# - Deagel summary (RVV-SD 110 km): https://www.deagel.com/Weapons/R-77/a001032

from bvr_marl_core.missiles.missile import Missile


class R77_1(Missile):
    def __init__(self, firing_time_s, target, source, map_limits, group=None):
        # Ø 200 mm => Sref ≈ π*(0.10)^2 ≈ 0.0314 m²; mass ~190 kg.
        config = {
            "mass_kg": 190.0,
            "reference_area_m2": 0.031,
            "aspect_ratio": 3.2,
            "oswald_e": 0.78,
            "drag_coefficient": 0.08,
            "constant_engine_N": 9500.0,
            "motor_burn_s": 8.0,
            "n_max": 40.0,
            "max_speed_mps": 1360.0,  # Mach ~4 class
            "min_range_m": 1500.0,
            # Cited kinematic max range (RVV-SD, ~110 km). Short boost + long glide
            # over-ranged in sim; a higher drag_scale bleeds the glide tail.
            "max_range_m": 110_000.0,
            "drag_scale": 2.5,
            "seeker_sensitivity": 1.2,
            "life_time_s": 120.0,
            "hit_probability": 0.82,
            "use_apn": True,
            # Weapon-effectiveness submodels: capable ARH, slightly lower
            # seeker/datalink reliability than the Western/Meteor set.
            "warhead_effectiveness": 1.0,
            # A clean, non-maneuvering intercept should not lose nearly one shot
            # in three to compounded reliability terms. Keep the R-77 below the
            # newer Western weapons, but calibrate its zero-miss Pk to ~0.76.
            "fuze_reliability": 0.96,
            "guidance_reliability": 0.96,
            "seeker_reliability": 0.94,
            "datalink_reliability": 0.92,
            "radar": {
                "horizontal_fov_deg": 60.0,
                "vertical_fov_deg": 30.0,
                "max_range_m": 110_000.0,
                "radar_frequency_hz": 10e9,
                "tx_power_w": 8e3,
                "antenna_gain_db": 32.0,
                "snr_threshold_db": 8.0,
            },
        }
        super().__init__(
            name="R77_1",
            firing_time_s=firing_time_s,
            target=target,
            source=source,
            map_limits=map_limits,
            group=group or source.group,
            config=config,
            data_link_mode="msl_support",
        )
        self.fox_type = 3
