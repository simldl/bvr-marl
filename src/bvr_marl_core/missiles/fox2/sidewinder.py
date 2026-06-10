from bvr_marl_core.missiles.fox2.base_fox2 import Fox2Missile


class AIM9_Sidewinder(Fox2Missile):
    def __init__(self, firing_time_s, target, source, map_limits, group=None):
        config = {
            "mass_kg": 85.0,
            "reference_area_m2": 0.15,
            "aspect_ratio": 2.8,
            "oswald_e": 0.8,
            "drag_coefficient": 0.08,
            "constant_engine_N": 8000.0,
            "n_max": 35.0,
            "max_speed_mps": 900.0,
            "motor_burn_s": 6.0,
            "min_range_m": 500.0,
            "seeker_sensitivity": 1.0,
            "life_time_s": 45.0,
            "hit_probability": 0.90,
            "radar": {
                "fov_deg": 15.0,
                "max_range_m": 15000.0,
                "sensitivity": 1.0,
            },
        }
        super().__init__(
            name="AIM9_Sidewinder",
            firing_time_s=firing_time_s,
            target=target,
            source=source,
            map_limits=map_limits,
            group=group or source.group,
            config=config,
            data_link_mode="none",
        )
