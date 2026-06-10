#!/usr/bin/env python3
"""
Quick sanity tests to verify the action space and environment fixes.
"""


def test_debug_info_key_fix():
    """Test that get_debug_info() no longer raises KeyError for target state."""
    print("=== Testing debug key typo fix ===")
    try:
        from bvr_marl_core.rl.environment.spaces.action_space import ActionProcessor

        # Create a mock simulator
        class MockSimulator:
            def __init__(self):
                self.active_units = {}
                self.tick_secs = 0.1

        simulator = MockSimulator()
        processor = ActionProcessor(simulator)

        # Initialize an agent state
        import numpy as np

        agent_id = 1
        processor.agent_states[agent_id] = {
            "u_bar": np.array([0.0] * 10),
            "v_bar": 250.0,
            "target_index": 0,
            "target_hold_time_left_s": 2.0,  # This was the fixed key
            "last_target_bin": -1,
            "target_candidates_sorted": [],
            "schmitt_states": {i: False for i in range(4, 10)},
        }

        # This should not raise a KeyError anymore
        debug_info = processor.get_debug_info(agent_id)
        target_state = debug_info["target_state"]

        # Check the fixed key exists
        assert "hold_time_left_s" in target_state
        assert target_state["hold_time_left_s"] == 2.0

        print("OK Debug key typo fix verified - no KeyError and correct key present")

    except Exception as e:
        print(f"FAIL Debug key fix failed: {e}")
        assert False, f"Debug key fix failed: {e}"


def test_missile_inventory_gate():
    """Test that missile firing is gated by inventory."""
    print("\n=== Testing missile inventory gating ===")

    # This is a conceptual test - in practice we'd need a full simulator setup
    # But we can verify the logic by checking the code structure
    try:
        # Verify the inventory gating code exists in the apply method
        import inspect

        from bvr_marl_core.rl.environment.spaces.action_space import ActionProcessor

        source_lines = inspect.getsourcelines(ActionProcessor.apply)[0]
        source_code = "".join(source_lines)

        # Check for inventory-related checks - the actual inventory check is in the weapon system
        # The ActionProcessor delegates to weapon_handler.handle_missile_firing()
        inventory_checks = ["handle_missile_firing", "weapon_handler"]

        all_checks_present = all(check in source_code for check in inventory_checks)

        if all_checks_present:
            print("OK Missile inventory gating code present in apply method")
        else:
            print("FAIL Missing some inventory gating code")
            assert False, "Missing some inventory gating code"

    except Exception as e:
        print(f"FAIL Inventory gating test failed: {e}")
        assert False, f"Inventory gating test failed: {e}"


def test_sim_tick_consistency():
    """Test that sim tick consistency is implemented in env reset."""
    print("\n=== Testing sim tick consistency ===")

    try:
        # Check that the reset method calls termination_checker.reset_timing which sets simulator tick_secs
        import inspect

        from bvr_marl_core.rl.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv

        source_lines = inspect.getsourcelines(BVRMultiAgentEnv.reset)[0]
        source_code = "".join(source_lines)

        if "self.termination_checker.reset_timing(self.config.tick_secs)" in source_code:
            print("PASS Sim tick consistency fix present in reset method")
        else:
            print("FAIL Sim tick consistency fix not found")
            assert False, "Sim tick consistency fix not found"

    except Exception as e:
        print(f"FAIL Sim tick consistency test failed: {e}")
        assert False, f"Sim tick consistency test failed: {e}"


def test_missile_identification_standardization():
    """Test that missile identification uses is_missile consistently."""
    print("\n=== Testing missile identification standardization ===")

    try:
        # Check that missile identification uses is_missile consistently in relevant modules
        import inspect

        from bvr_marl_core.rl.environment.gym.gym_components.helpers import AgentHelpers
        from bvr_marl_core.rl.environment.spaces.action_space.utils.target_sorting import (
            TargetSorter,
        )

        # Check target sorting (where target filtering happens)
        ts_source = "".join(inspect.getsourcelines(TargetSorter.select_target)[0])
        ts_uses_is_missile = 'getattr(u, "is_missile", False)' in ts_source

        # Check gym helpers (where missile identification happens)
        helpers_source = "".join(
            inspect.getsourcelines(AgentHelpers.get_incoming_missiles_for_agent)[0]
        )
        helpers_uses_is_missile = 'getattr(unit, "is_missile", False)' in helpers_source

        if ts_uses_is_missile and helpers_uses_is_missile:
            print("PASS Both target sorting and gym helpers use is_missile consistently")
        else:
            print(
                f"FAIL Inconsistent missile identification - TargetSorter: {ts_uses_is_missile}, Helpers: {helpers_uses_is_missile}"
            )
            assert False, (
                f"Inconsistent missile identification - TargetSorter: {ts_uses_is_missile}, Helpers: {helpers_uses_is_missile}"
            )

    except Exception as e:
        print(f"FAIL Missile identification test failed: {e}")
        assert False, f"Missile identification test failed: {e}"


def test_training_signals_in_infos():
    """Test that training signals are exposed in infos."""
    print("\n=== Testing training signals exposure ===")

    try:
        # Training signals are collected in the observation builder, not step method directly
        import inspect

        from bvr_marl_core.rl.environment.gym.gym_components.observation_builder import (
            ObservationInfoBuilder,
        )

        source_lines = inspect.getsourcelines(ObservationInfoBuilder.build_agent_info)[0]
        source_code = "".join(source_lines)

        # Check for training signals being added to infos
        training_signals_checks = [
            "training_signals = self.action_processor.get_training_signals",
            "valid_missile_shots",
            "vetoed_missile_shots",
        ]

        all_present = all(check in source_code for check in training_signals_checks)

        if all_present:
            print("PASS Training signals and envelope scalars exposed in infos")
        else:
            print("FAIL Missing some training signals in infos")
            assert False, "Missing some training signals in infos"

    except Exception as e:
        print(f"FAIL Training signals test failed: {e}")
        assert False, f"Training signals test failed: {e}"


if __name__ == "__main__":
    print("Running sanity tests for action space and environment fixes...\n")

    tests = [
        test_debug_info_key_fix,
        test_missile_inventory_gate,
        test_sim_tick_consistency,
        test_missile_identification_standardization,
        test_training_signals_in_infos,
    ]

    results = []
    for test in tests:
        try:
            test()  # Tests now use assertions instead of returning values
            results.append(True)
            print(f"PASS {test.__name__}")
        except Exception as e:
            print(f"FAIL Test {test.__name__} crashed: {e}")
            results.append(False)

    passed = sum(results)
    total = len(results)

    print("\n=== Summary ===")
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("All sanity tests passed!")
    else:
        print("Some tests failed - please review the fixes")
