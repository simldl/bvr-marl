from bvr_marl_core.missiles.core.phases import MissilePhaseManager


def test_phase_manager_default_and_update():
    manager = MissilePhaseManager(motor_burn_s=100)
    manager.update(0)
    assert manager.current_phase == "boost"
    manager.update(21)
    assert manager.current_phase == "middle"
    manager.update(80)
    assert manager.current_phase == "middle"
    manager.update(100)
    assert manager.current_phase == "terminal"
    assert manager.get_thrust_kN() == 0.0


def test_default_phase_durations_are_independent_between_instances():
    short_burn = MissilePhaseManager(motor_burn_s=30)
    long_burn = MissilePhaseManager(motor_burn_s=100)

    assert short_burn.flight_phases["middle"]["duration_s"] == 10.0
    assert long_burn.flight_phases["middle"]["duration_s"] == 80.0
    assert MissilePhaseManager.DEFAULT_PHASES["middle"]["duration_s"] == 30.0


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
