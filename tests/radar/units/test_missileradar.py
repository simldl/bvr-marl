import pytest
from radar.units.missile import MissileRadar
from radar.core.data_link import DataLink
from radar.lock.missile import MissileLockController

class DummyTargetProvider:
    def get_guidance_target(self):
        # Liefert eine Zielposition in Reichweite
        pos = type("Pos", (), {"lat": 0, "lon": 0.01, "alt": 1000})()
        return pos

class DummyOwner:
    def __init__(self):
        self.position = type("Pos", (), {"lat": 0, "lon": 0, "alt": 1000})()
        self.phase_manager = type("Phase", (), {"current_phase": "boost"})()
        self.yaw_deg = 0
        self.pitch_deg = 0

@pytest.fixture
def dummy_radar_args():
    return dict(
        horizontal_fov_deg=60.0,
        vertical_fov_deg=30.0,
        max_range_m=20000,
        radar_frequency_hz=10e9,
        tx_power_w=15000,
        antenna_gain_db=30.0,
        snr_threshold_db=3.0,
    )

def test_missileradar_starts_in_other(dummy_radar_args):
    radar = MissileRadar(
        **dummy_radar_args,
        target_provider=DummyTargetProvider(),
        owner=DummyOwner(),
        initial_datalink_mode="other"
    )
    assert radar.data_link.get_mode() == "other"

def test_missileradar_switches_to_own(dummy_radar_args):
    class DummyProvider:
        def get_guidance_target(self):
            return type("Pos", (), {"lat": 0, "lon": 0, "alt": 1000})()

    radar = MissileRadar(
        **dummy_radar_args,
        target_provider=DummyProvider(),
        owner=DummyOwner(),
        initial_datalink_mode="other",
        original_data_link_mode="own"
    )
    radar.update(
        tick_secs=1.0,
        sim=None,
        targets=[],
        owner_position=radar.owner.position,
        group_radars=[],
    )
    radar.update(
        tick_secs=1.0,
        sim=None,
        targets=[],
        owner_position=radar.owner.position,
        group_radars=[],
    )
    assert radar.data_link.get_mode() == "own"


def test_missileradar_lock_logic(dummy_radar_args):
    radar = MissileRadar(**dummy_radar_args, owner=DummyOwner(), target_provider=DummyTargetProvider())
    class DummyTarget:
        id = "tlock"

    # Update locks twice to confirm lock
    radar.lock_ctrl.update_locks(["tlock"])
    radar.lock_ctrl.update_locks(["tlock"])

    # MissileRadar sets locked_target during its lock update cycle
    # Manually trigger the same logic as in update_for_sensors
    radar.locked_target = radar.lock_ctrl.get_locked()

    assert radar.has_radar_lock(DummyTarget())