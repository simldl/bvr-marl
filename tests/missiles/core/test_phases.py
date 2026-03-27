import pytest

from air_to_air_rl.missiles.core.phases import MissilePhaseManager


def test_phase_manager_default_and_update():
    # Default config with motor_burn_s=100
    # Boost phase lasts 20s, then middle phase lasts until motor_burn_s (100s), then terminal
    manager = MissilePhaseManager(motor_burn_s=100)
    # 0s: boost
    manager.update(0)
    assert manager.current_phase == "boost"
    # 21s: middle (after boost ends at 20s)
    manager.update(21)
    assert manager.current_phase == "middle"
    # 80s: still middle (motor burns until 100s)
    manager.update(80)
    assert manager.current_phase == "middle"
    # 100s+: terminal (after motor burnout)
    manager.update(100)
    assert manager.current_phase == "terminal"
    # Check thrust (should be 0 in terminal)
    thrust = manager.get_thrust_kN()
    assert thrust == 0.0  # No thrust in terminal phase


def test_phase_manager_custom_config():
    phases = {
        "boost": {"duration_s": 5.0, "thrust_kN": 60.0},
        "middle": {"duration_s": 10.0, "thrust_kN": 50.0},
        "terminal": {"duration_s": 2.0, "thrust_kN": 20.0},
    }
    manager = MissilePhaseManager(phases)
    manager.update(0)
    assert manager.current_phase == "boost"
    manager.update(5.1)
    assert manager.current_phase == "middle"
    manager.update(16)
    assert manager.current_phase == "terminal"
