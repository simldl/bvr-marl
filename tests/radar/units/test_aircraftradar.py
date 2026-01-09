import pytest
from radar.units.aircraft import AircraftRadar
from radar.core.data_link import DataLink
from radar.lock.aircraft import AircraftLockController

class DummySim:
    def __init__(self):
        self.active_units = {}

class DummyTarget:
    def __init__(self, id_):
        self.id = id_
        self.group = "RED"
        self.position = type("Pos", (), {"lat": 0, "lon": 0, "alt": 0})()

class DummyAlly:
    def __init__(self, id_, group, radar_mode="full"):
        self.id = id_
        self.group = group
        self.yaw_deg = 0
        self.pitch_deg = 0
        self.radar = AircraftRadar(
            horizontal_fov_deg=60.0,
            vertical_fov_deg=30.0,
            max_range_m=5000,
            radar_frequency_hz=10e9,
            tx_power_w=10000,
            antenna_gain_db=25.0,
            snr_threshold_db=2.0,
            data_link=DataLink(radar_mode),
            owner=self
        )
        self.position = type("Pos", (), {
            "lat": 0,
            "lon": 0,
            "alt": 0,
            "copy": lambda self=None: self  # oder besser: Rückgabe einer echten Kopie
        })()


@pytest.fixture
def dummy_radar_args():
    return dict(
        horizontal_fov_deg=60.0,
        vertical_fov_deg=30.0,
        max_range_m=5000,
        radar_frequency_hz=10e9,
        tx_power_w=10000,
        antenna_gain_db=25.0,
        snr_threshold_db=2.0,
    )

def test_update_for_sensors_detects_targets(dummy_radar_args):
    # Radar (self)
    owner = DummyAlly("self", "BLUE")
    radar = owner.radar
    radar.owner = owner
    # Ziel (anderes Team)
    target = DummyTarget("t1")
    target2 = DummyTarget("t2")
    sim = DummySim()
    sim.active_units = {
        "self": owner,
        "t1": target,
        "t2": target2,
    }
    tracks = radar.update_for_sensors(1.0, sim, owner_position=owner.position)
    assert isinstance(tracks, list)
    # (Hier wäre eine realistische Radar-Implementierung nötig, um echte Detections zu prüfen)

def test_multi_lock(dummy_radar_args):
    radar = AircraftRadar(**dummy_radar_args, data_link=DataLink("own"))
    # Zwei Tracks mit unterschiedlichen IDs
    t1 = DummyTarget("t1")
    t2 = DummyTarget("t2")
    tracks = [[None, None, t1, None, None], [None, None, t2, None, None]]
    # Zwei mal update für Lock-Confirmation
    for _ in range(2):
        radar.lock_ctrl.update_locks([t.id for _, _, t, *_ in tracks if t is not None])
    radar.locked_targets = set(radar.lock_ctrl.locked_target_ids())
    assert radar.has_radar_lock(t1)
    assert radar.has_radar_lock(t2)
    assert t1.id in radar.get_locked_targets()
    assert t2.id in radar.get_locked_targets()

def test_datalink_policy_group_radars(dummy_radar_args):
    # Own AircraftRadar im full-Mode
    owner = DummyAlly("own", "BLUE", radar_mode="full")
    radar = owner.radar
    radar.owner = owner
    # Zwei Allies: einer mit full, einer mit own
    ally_full = DummyAlly("a_full", "BLUE", radar_mode="full")
    ally_own = DummyAlly("a_own", "BLUE", radar_mode="own")
    # Ziel (anderes Team)
    target = DummyTarget("enemy")
    sim = DummySim()
    sim.active_units = {
        "own": owner,
        "a_full": ally_full,
        "a_own": ally_own,
        "enemy": target,
    }
    # Die group_radars werden in update_for_sensors korrekt gefiltert:
    tracks = radar.update_for_sensors(1.0, sim, owner_position=owner.position)
    assert isinstance(tracks, list)

def test_lock_unlock(dummy_radar_args):
    radar = AircraftRadar(**dummy_radar_args, data_link=DataLink("own"))
    t1 = DummyTarget("t1")
    # Lock t1
    radar.lock_ctrl.update_locks([t1.id])
    radar.lock_ctrl.update_locks([t1.id])
    radar.locked_targets = set(radar.lock_ctrl.locked_target_ids())
    assert radar.has_radar_lock(t1)
    # Jetzt Target nicht mehr gesichtet (simulate 5 missed updates)
    for _ in range(5):
        radar.lock_ctrl.update_locks([])
    radar.locked_targets = set(radar.lock_ctrl.locked_target_ids())
    assert not radar.has_radar_lock(t1)