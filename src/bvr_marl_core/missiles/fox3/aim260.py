# Next-generation networked BVR missile (AIM-260-class surrogate).
# Sources (public, order-of-magnitude only):
# - AIM-260 JATM overview: https://en.wikipedia.org/wiki/AIM-260_JATM
# The defining trait modelled here is a persistent two-way datalink: the missile
# keeps receiving the launching side's fused (and, against a jammer, triangulated)
# target track for the whole flight, so it is not forced into home-on-jam terminal
# guidance the way a classic Fox-3 is. Kinematics are AMRAAM/Meteor-class.

from bvr_marl_core.missiles.missile import Missile


class AIM260_JATM(Missile):
    def __init__(self, firing_time_s, target, source, map_limits, group=None):
        config = {
            "mass_kg": 200.0,
            "reference_area_m2": 0.025,
            "aspect_ratio": 3.0,
            "oswald_e": 0.80,
            "drag_coefficient": 0.07,
            "constant_engine_N": 7000.0,
            "motor_burn_s": 40.0,
            "n_max": 40.0,
            "max_speed_mps": 1400.0,
            "min_range_m": 2000.0,
            "max_range_m": 200_000.0,
            "drag_scale": 1.6,
            "seeker_sensitivity": 1.3,
            "life_time_s": 160.0,
            "hit_probability": 0.9,
            "lethal_radius_m": 250.0,
            "warhead_effectiveness": 1.0,
            "fuze_reliability": 0.96,
            "guidance_reliability": 0.97,
            "seeker_reliability": 0.96,
            "datalink_reliability": 0.98,
            # Persistent network datalink: inherits cooperative triangulation on a
            # jammer through terminal, instead of going seeker-autonomous (HOJ).
            "full_datalink": True,
            "use_apn": True,  # augmented PN vs maneuvering targets
            "radar": {
                "horizontal_fov_deg": 60.0,
                "vertical_fov_deg": 30.0,
                "max_range_m": 200_000.0,
                "radar_frequency_hz": 10e9,
                "tx_power_w": 9e3,
                "antenna_gain_db": 34.0,
                "snr_threshold_db": 8.0,
            },
        }
        super().__init__(
            name="AIM260_JATM",
            firing_time_s=firing_time_s,
            target=target,
            source=source,
            map_limits=map_limits,
            group=group or source.group,
            config=config,
            data_link_mode="full",
        )
        self.fox_type = 3
