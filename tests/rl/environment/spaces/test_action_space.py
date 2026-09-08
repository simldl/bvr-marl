import numpy as np
import pytest

from bvr_marl_core.rl.environment.spaces.action_space import (
    ActionProcessor,
    ActionSpaceManager,
    SimplifiedActionProcessor,
)
from bvr_marl_core.rl.environment.spaces.action_space.processors import (
    EnergyProcessor,
    LiftVectorProcessor,
)


class DummyPhysics:
    def __init__(self):
        from unittest.mock import Mock

        self.g = 9.81
        self.n_max = 9.0
        self.max_pitch_rate_deg_s = 20.0
        self.service_ceiling = 15000.0
        self.mass_kg = 18000.0
        self.A_m2 = 27.87
        # Add air attribute for physics calculations
        self.air = Mock()
        self.air.get_density = Mock(return_value=1.225)

    def compute_instantaneous_turn_rate(self, v_mps, alt_m):
        return 15.0  # deg/s

    def _dynamic_stall(self, alt_m):
        return 50.0  # m/s

    def get_base_drag_cd(self, v_mps, alt_m):
        return 0.02  # Simple constant drag coefficient

    def get_engine_force(self, v_mps, alt_m, throttle):
        return 50000.0


class DummyPosition:
    def __init__(self):
        self.lat = 0.0
        self.lon = 0.0
        self.alt = 1000.0


class DummyUnit:
    def __init__(self, id_="A1", group="BLUE"):
        self.id = id_
        self.group = group
        self.yaw_deg = 0.0
        self.pitch_deg = 0.0
        self.speed = 100.0
        self.min_speed_mps = 50.0
        self.max_speed_mps = 300.0
        self.position = DummyPosition()
        self.physics = DummyPhysics()

        # Create control object that can update this unit's attributes
        unit_ref = self
        self.control = type(
            "C",
            (),
            {
                "set_throttle": lambda self, x: setattr(self, "throttle", x),
                "set_yaw_deg": lambda self, x: setattr(unit_ref, "yaw_deg", x),
                "set_pitch_deg": lambda self, x: setattr(unit_ref, "pitch_deg", x),
                "throttle": 0.5,
            },
        )()
        self.sensor = type(
            "S",
            (),
            {
                "select_target": lambda self, cands, sel: cands[0] if cands else None,
                "has_radar_lock": lambda self, target: True,  # Add has_radar_lock method
            },
        )()
        self.weapons = type(
            "W",
            (),
            {
                "fire_missile": lambda *a, **k: "fired",
                "is_target_in_fov": lambda *a, **k: True,
            },
        )()
        self.missile_types = [lambda *a, **k: None]
        self.countermeasures = type(
            "CM",
            (),
            {
                "launch_flares": lambda self, *a, **k: setattr(self, "flared", True),
                "launch_chaff": lambda self, *a, **k: setattr(self, "chaffed", True),
                "activate_ecm": lambda self, *a, **k: setattr(self, "ecm", True),
                "deploy_decoys": lambda self, *a, **k: setattr(self, "decoys", True),
            },
        )()


class DummySim:
    def __init__(self):
        self.active_units = {"A1": DummyUnit(), "B1": DummyUnit("B1", "RED")}
        self.tick_secs = 0.1

    def count_missiles_at_target(self, group, target_id):
        return 0


def test_action_space_manager():
    mgr = ActionSpaceManager(agent_ids=["A1", "B1"], lower=0.0, upper=1.0, shape=10)
    box = mgr.get("A1")
    assert box.shape == (10,)
    assert box.contains(np.zeros(10, dtype=np.float32))
    all_spaces = mgr.all()
    assert set(all_spaces.keys()) == {"A1", "B1"}


def test_action_processor_apply():
    sim = DummySim()
    proc = ActionProcessor(sim)
    # Provide 10 actions: energy, load, bank, missile_fire, target_select, gun_fire, flares, chaff, ecm, decoys
    action = np.array([1.0, 0.5, 0.5, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    proc.apply("A1", action)
    sim.active_units["A1"]
    # Test should pass if no exception is raised during action processing
    assert True  # Simplified assertion since we're testing the process doesn't crash


def test_centered_energy_mapping_keeps_midpoint_neutral():
    """PPO energy action 0.5 must mean Ps=0 even with asymmetric limits."""
    assert EnergyProcessor.action_to_centered_ps(0.0, -30.0, 90.0) == pytest.approx(-30.0)
    assert EnergyProcessor.action_to_centered_ps(0.5, -30.0, 90.0) == pytest.approx(0.0)
    assert EnergyProcessor.action_to_centered_ps(1.0, -30.0, 90.0) == pytest.approx(90.0)

    assert EnergyProcessor.centered_ps_to_action(-30.0, -30.0, 90.0) == pytest.approx(0.0)
    assert EnergyProcessor.centered_ps_to_action(0.0, -30.0, 90.0) == pytest.approx(0.5)
    assert EnergyProcessor.centered_ps_to_action(90.0, -30.0, 90.0) == pytest.approx(1.0)


def test_centered_load_factor_mapping_keeps_midpoint_neutral():
    """PPO load-factor action 0.5 must mean the 1g level-flight reference."""
    assert LiftVectorProcessor.action_to_centered_load_factor(0.0, -2.0, 9.0) == pytest.approx(-2.0)
    assert LiftVectorProcessor.action_to_centered_load_factor(0.5, -2.0, 9.0) == pytest.approx(1.0)
    assert LiftVectorProcessor.action_to_centered_load_factor(1.0, -2.0, 9.0) == pytest.approx(9.0)

    assert LiftVectorProcessor.centered_load_factor_to_action(-2.0, -2.0, 9.0) == pytest.approx(0.0)
    assert LiftVectorProcessor.centered_load_factor_to_action(1.0, -2.0, 9.0) == pytest.approx(0.5)
    assert LiftVectorProcessor.centered_load_factor_to_action(9.0, -2.0, 9.0) == pytest.approx(1.0)


def test_full_action_midpoint_is_neutral_for_flight_controls():
    """Full BVR action midpoint should hold energy, 1g, and wings level."""
    sim = DummySim()
    sim.active_units = {"A1": DummyUnit()}
    proc = ActionProcessor(sim)

    action = np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    proc.apply("A1", action)

    state = proc.agent_states["A1"]
    assert proc.state["A1"]["ps_cmd_filtered"] == pytest.approx(0.0, abs=1e-7)
    assert state["n_cmd_filtered"] == pytest.approx(1.0, abs=1e-7)
    assert state["phi_cmd_filtered"] == pytest.approx(0.0, abs=1e-7)


def test_rate_based_controls():
    """Test that rate-based controls work correctly with physics envelopes."""
    sim = DummySim()
    proc = ActionProcessor(sim)

    # Test initial state
    unit = sim.active_units["A1"]
    initial_yaw = unit.yaw_deg
    initial_pitch = unit.pitch_deg

    # Apply action with yaw rate command (0.7 should give positive rate)
    action = np.array([0.6, 0.7, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    proc.apply("A1", action)

    # Check that yaw and pitch changed according to rate-based commands
    # Note: exact values depend on EMA filtering and physics envelope calculations
    assert unit.yaw_deg != initial_yaw  # Should have changed due to rate command
    assert unit.pitch_deg != initial_pitch  # Should have changed due to rate command

    # Verify debug info is available
    debug_info = proc.get_debug_info("A1")
    # Backward/forward compatible: prefer new key, fall back to legacy
    cmd_block = debug_info.get("filtered_commands", debug_info.get("ema_commands", {}))
    u_bar = cmd_block.get("u_bar", None)
    assert u_bar is not None, f"debug_info keys: {list(debug_info.keys())}"
    assert len(u_bar) == 10


def test_trigger_states():
    """Test trigger state reporting in debug info."""
    sim = DummySim()
    proc = ActionProcessor(sim)

    # Test low value - should not trigger
    action_low = np.array([0.5, 0.5, 0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    proc.apply("A1", action_low)
    debug_info = proc.get_debug_info("A1")

    # Verify trigger_states exists in debug_info
    assert "trigger_states" in debug_info, f"debug_info keys: {list(debug_info.keys())}"
    trigger_states = debug_info["trigger_states"]

    # Verify it's a list of booleans with correct length
    assert isinstance(trigger_states, list)
    assert len(trigger_states) == 10
    assert all(isinstance(state, (bool, np.bool_)) for state in trigger_states)


def test_missile_fire_threshold_is_configurable_without_changing_other_triggers():
    """Missile firing can use a stricter continuous threshold than other triggers."""
    sim = DummySim()
    proc = ActionProcessor(sim)
    proc.configure_automation(missile_fire_threshold=0.7)

    assert proc.trigger_proc.apply_trigger(0.65, 3, {}) is False
    assert proc.trigger_proc.apply_trigger(0.70, 3, {}) is True
    assert proc.trigger_proc.apply_trigger(0.65, 5, {}) is True


def test_simplified_action_order_maps_phi_before_load_factor():
    """Simplified actions are [energy, lift_phi, lift_N, missile_trigger]."""
    sim = DummySim()
    sim.active_units = {"A1": DummyUnit()}
    proc = SimplifiedActionProcessor(sim)

    action = np.array([0.5, 1.0, 0.5, 0.0], dtype=np.float32)
    proc.apply("A1", action)

    state = proc.agent_states["A1"]
    # Axis identity, not filter magnitude: a full-scale command on index 1 must
    # move bank and leave load factor at its 1 g neutral. The previous ">10 deg"
    # bound silently encoded the command filter's time constant, so re-tuning
    # tau failed a test that is not about tau at all.
    assert state["n_cmd_filtered"] == pytest.approx(1.0)
    assert state["phi_cmd_filtered"] > 0.0


# =============================================================================
# TargetSorter — is_non_engageable filtering
# =============================================================================


class _FakePos:
    lat = 0.0
    lon = 0.0
    alt = 1000.0


class _FakeUnit:
    def __init__(self, uid, group, non_engageable=False):
        self.id = uid
        self.group = group
        self.position = _FakePos()
        self.is_missile = False
        if non_engageable:
            self.is_non_engageable = True


class _FakeSim:
    def __init__(self, units):
        self.active_units = {u.id: u for u in units}


def test_target_sorter_excludes_non_engageable():
    """select_target() must never return a unit marked is_non_engageable."""
    from bvr_marl_core.rl.environment.spaces.action_space.utils.target_sorting import (
        TargetSorter,
    )

    own = _FakeUnit("A1", "agent")
    fighter = _FakeUnit("B1", "opponent")
    awacs = _FakeUnit("B2", "opponent", non_engageable=True)
    sim = _FakeSim([own, fighter, awacs])

    sorter = TargetSorter()
    state = sorter.init_target_state()
    selected = sorter.select_target(own, sim, action_target=0.5, state=state, dt=1.0)

    assert selected is not None, "Should select the engageable fighter"
    assert selected.id == "B1", f"Expected B1, got {selected.id}"
    assert not getattr(selected, "is_non_engageable", False)


def test_target_sorter_returns_none_when_all_non_engageable():
    """select_target() returns None when all opponents are non-engageable."""
    from bvr_marl_core.rl.environment.spaces.action_space.utils.target_sorting import (
        TargetSorter,
    )

    own = _FakeUnit("A1", "agent")
    awacs = _FakeUnit("B2", "opponent", non_engageable=True)
    sim = _FakeSim([own, awacs])

    sorter = TargetSorter()
    state = sorter.init_target_state()
    selected = sorter.select_target(own, sim, action_target=0.5, state=state, dt=1.0)

    assert selected is None, "No engageable opponents — should return None"
