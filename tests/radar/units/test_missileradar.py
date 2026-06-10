import pytest

from bvr_marl_core.radar.core.data_link import DataLink
from bvr_marl_core.radar.lock.missile import MissileLockController
from bvr_marl_core.radar.units.missile import MissileRadar


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
        initial_datalink_mode="other",
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
        original_data_link_mode="own",
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


def test_has_stable_seeker_track_threshold(dummy_radar_args):
    radar = MissileRadar(
        **dummy_radar_args, owner=DummyOwner(), target_provider=DummyTargetProvider()
    )
    radar._seeker_meas_streak = 0
    assert not radar.has_stable_seeker_track()
    radar._seeker_meas_streak = radar._seeker_stable_streak
    assert radar.has_stable_seeker_track()


def test_datalink_reacquire_consumed_once_per_tick(dummy_radar_args):
    """Option A: a coasting seeker consumes the staged datalink measurement on the
    first substep (one tracker advance WITH a measurement), then predicts-only on
    subsequent coasting substeps."""
    from unittest.mock import Mock

    radar = MissileRadar(
        **dummy_radar_args, owner=DummyOwner(), target_provider=DummyTargetProvider()
    )
    radar.tracker_manager = Mock()
    radar.tracker_manager.update_tracks = Mock(return_value=[])

    fake_cluster = {"T": type("T", (), {"id": 7})(), "az": 0.0, "el": 0.0, "d": 1000.0}
    radar._pending_datalink_meas = fake_cluster
    radar._datalink_meas_used = False
    radar._seeker_meas_streak = 5  # was tracking; this substep is a seeker miss
    pos = radar.owner.position

    # First coasting substep: the datalink measurement is fed to the tracker.
    radar._predict_designated_track_substep(0.1, pos)
    assert radar._datalink_meas_used is True
    assert radar._seeker_meas_streak == 0  # seeker itself did not measure
    args, kwargs = radar.tracker_manager.update_tracks.call_args
    assert args[0] == [fake_cluster]
    assert "missed_increment" not in kwargs

    # Second coasting substep: nothing left to feed -> predict-only, and because a
    # measurement (the datalink) was already taken THIS tick, the coast substep
    # does not age the track (missed_increment 0.0), so it stays fresh mid-tick.
    radar.tracker_manager.update_tracks.reset_mock()
    radar._predict_designated_track_substep(0.1, pos)
    args, kwargs = radar.tracker_manager.update_tracks.call_args
    assert args[0] == []
    assert kwargs.get("missed_increment") == 0.0

    # A new tick with no measurement DOES age the track.
    radar.begin_designated_tick()
    radar._pending_datalink_meas = None
    radar.tracker_manager.update_tracks.reset_mock()
    radar._predict_designated_track_substep(0.1, pos)
    _args, kwargs = radar.tracker_manager.update_tracks.call_args
    assert kwargs.get("missed_increment") == 0.1


def test_missileradar_lock_logic(dummy_radar_args):
    radar = MissileRadar(
        **dummy_radar_args, owner=DummyOwner(), target_provider=DummyTargetProvider()
    )

    class DummyTarget:
        id = "tlock"

    # Update locks twice to confirm lock
    radar.lock_ctrl.update_locks(["tlock"])
    radar.lock_ctrl.update_locks(["tlock"])

    # MissileRadar sets locked_target during its lock update cycle
    # Manually trigger the same logic as in update_for_sensors
    radar.locked_target = radar.lock_ctrl.get_locked()

    assert radar.has_radar_lock(DummyTarget())
