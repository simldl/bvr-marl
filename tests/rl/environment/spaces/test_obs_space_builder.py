from types import SimpleNamespace

import numpy as np

from bvr_marl_core.rl.environment.spaces.obs_space_builder import ObservationBuilder
from bvr_marl_core.rl.environment.spaces.observation.friendly_info_builder import (
    FriendlyInfoBuilder,
)


class DummyRadar:
    sensor_tracks = []


class DummySensor:
    def __init__(self):
        self.radar_system = DummyRadar()
        self.sensor_tracks = []  # required by builder._get_enemy_info
        self.nez_calc = type(
            "NEZ", (), {"active_nez": lambda self, t: 1.0, "passive_nez": lambda self, t: 1.0}
        )()
        self.get_locked_targets = lambda self: {"B1": 1}
        self.passive_radar = type(
            "PR", (), {"clear_old": lambda self, t: None, "get_observation": lambda self, t: []}
        )()

    def get_nez_features(self, simulator, target_id):
        """Return NEZ features for observation building."""
        return {"active_nez": 1.0, "passive_nez": 1.0, "nez_visible": True}


class DummyUnit:
    def __init__(self, id_="A1", group="BLUE"):
        self.id = id_
        self.group = group
        self.position = type("P", (), {"lat": 0, "lon": 0, "alt": 0})()
        self.yaw_deg = 0.0
        self.speed = 100.0
        self.pitch_deg = 0.0
        self.max_missiles = 2
        self.missiles = []
        self.sensor = DummySensor()

    def get_state(self):
        return {
            "sqi": 0.5,
            "dlz_zone": "R2",
            "nez_visible": True,
            "dlz": {
                "r_min_m": 2000.0,
                "r_tr_m": 25000.0,
                "r_pi_m": 40000.0,
                "r_aero_m": 60000.0,
                "slant_range_m": 30000.0,
            },
        }

    def get_state_representation(self):
        """Return a dictionary representing the unit's state for observations."""
        # Return the same as get_state() since that's what the observation builder expects
        return self.get_state()


class DummySim:
    def __init__(self):
        self.active_units = {"A1": DummyUnit(), "B1": DummyUnit("B1", "RED")}
        self.elapsed_time_s = 0.0


class DummyConfig:
    own_dim = 21
    fm_slots = 2
    ff_slots = 1
    em_slots = 2
    ef_slots = 1
    pr_slots = 1
    warn_sectors = 4


def test_observation_builder_build():
    sim = DummySim()
    cfg = DummyConfig()
    builder = ObservationBuilder(sim, ["A1", "B1"], cfg)
    obs = builder.build("A1")
    assert isinstance(obs, dict)
    assert "own_state" in obs

    # Updated: friendly missiles are now 8 dims per slot (was 6)
    assert obs["friendly_missiles"].shape[0] == cfg.fm_slots * 8
    assert obs["mask_friendly_missiles"].shape == (cfg.fm_slots,)


def test_friendly_target_indices_use_enemy_fighter_slots_for_sim_unit_ids():
    class Config:
        fm_slots = 1
        ff_slots = 1
        ef_slots = 2
        all_agent_ids = ("A0", "B0", "B1")

    def pos(lat=0.0, lon=0.0, alt=8_000.0):
        return SimpleNamespace(lat=lat, lon=lon, alt=alt)

    red_1 = SimpleNamespace(
        id=101,
        group="red",
        type="Aircraft",
        position=pos(lon=0.10),
        speed=300.0,
        pitch_deg=0.0,
        yaw_deg=270.0,
    )
    red_2 = SimpleNamespace(
        id=202,
        group="red",
        type="Aircraft",
        position=pos(lon=0.20),
        speed=300.0,
        pitch_deg=0.0,
        yaw_deg=270.0,
    )
    own = SimpleNamespace(
        id="A0",
        group="blue",
        type="Aircraft",
        position=pos(),
        speed=300.0,
        pitch_deg=0.0,
        yaw_deg=90.0,
    )
    wingman = SimpleNamespace(
        id="A1",
        group="blue",
        type="Aircraft",
        position=pos(lat=0.01),
        speed=300.0,
        pitch_deg=0.0,
        yaw_deg=90.0,
        sensor=SimpleNamespace(get_locked_targets=lambda: {101}),
    )
    missile = SimpleNamespace(
        id=301,
        group="blue",
        is_missile=True,
        type="Missile",
        position=pos(lon=0.02),
        speed=700.0,
        pitch_deg=0.0,
        yaw_deg=90.0,
        target=red_1,
        designated_target_id=101,
        target_provider=SimpleNamespace(current_target_id=202),
    )

    # Track order mirrors EnemyInfoBuilder: sorted by track id, then filtered.
    own.sensor = SimpleNamespace(
        sensor_tracks=[
            (
                10,
                np.zeros(6),
                None,
                red_1,
                "AIRCRAFT",
                own.position,
                1.0,
                1,
                1.0,
                1,
                False,
                False,
                None,
                None,
                True,
            ),
            (
                11,
                np.zeros(6),
                None,
                red_2,
                "AIRCRAFT",
                own.position,
                1.0,
                1,
                1.0,
                1,
                False,
                False,
                None,
                None,
                True,
            ),
        ]
    )
    sim = SimpleNamespace(
        active_units={
            own.id: own,
            wingman.id: wingman,
            red_1.id: red_1,
            red_2.id: red_2,
            missile.id: missile,
        }
    )

    _, _, _, _, fm_targets, _, ff_locks, _ = FriendlyInfoBuilder(sim, Config()).build(own.id)

    assert fm_targets.tolist() == [1.0]
    assert ff_locks.tolist() == [0.0]
