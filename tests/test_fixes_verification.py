#!/usr/bin/env python3
"""
Quick sanity tests to verify the action space and environment fixes.
"""

def test_debug_info_key_fix():
    """Test that get_debug_info() no longer raises KeyError for target state."""
    print("=== Testing debug key typo fix ===")
    try:
        from reinforcement_learning.environment.spaces.action_space import ActionProcessor

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
            'u_bar': np.array([0.0] * 10),
            'v_bar': 250.0,
            'target_index': 0,
            'target_hold_time_left_s': 2.0,  # This was the fixed key
            'last_target_bin': -1,
            'target_candidates_sorted': [],
            'schmitt_states': {i: False for i in range(4, 10)},
        }

        # This should not raise a KeyError anymore
        debug_info = processor.get_debug_info(agent_id)
        target_state = debug_info['target_state']

        # Check the fixed key exists
        assert 'hold_time_left_s' in target_state
        assert target_state['hold_time_left_s'] == 2.0

        print("OK Debug key typo fix verified - no KeyError and correct key present")
        return True

    except Exception as e:
        print(f"FAIL Debug key fix failed: {e}")
        return False


def test_missile_inventory_gate():
    """Test that missile firing is gated by inventory."""
    print("\n=== Testing missile inventory gating ===")

    # This is a conceptual test - in practice we'd need a full simulator setup
    # But we can verify the logic by checking the code structure
    try:
        from reinforcement_learning.environment.spaces.action_space import ActionProcessor

        # Verify the inventory gating code exists in the apply method
        import inspect
        source_lines = inspect.getsourcelines(ActionProcessor.apply)[0]
        source_code = ''.join(source_lines)

        # Check for inventory-related checks
        inventory_checks = [
            'has_inventory = getattr(unit, \'remaining_missiles\', None)',
            'inventory_ok = has_inventory is None or has_inventory > 0',
            'inventory_ok and',
            'no_inventory'
        ]

        all_checks_present = all(check in source_code for check in inventory_checks)

        if all_checks_present:
            print("OK Missile inventory gating code present in apply method")
            return True
        else:
            print("FAIL Missing some inventory gating code")
            return False

    except Exception as e:
        print(f"FAIL Inventory gating test failed: {e}")
        return False


def test_sim_tick_consistency():
    """Test that sim tick consistency is implemented in env reset."""
    print("\n=== Testing sim tick consistency ===")

    try:
        from reinforcement_learning.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv

        # Check that the reset method contains the tick_secs assignment
        import inspect
        source_lines = inspect.getsourcelines(BVRMultiAgentEnv.reset)[0]
        source_code = ''.join(source_lines)

        if 'self.simulator.tick_secs = self.tick_secs' in source_code:
            print("✅ Sim tick consistency fix present in reset method")
            return True
        else:
            print("❌ Sim tick consistency fix not found")
            return False

    except Exception as e:
        print(f"❌ Sim tick consistency test failed: {e}")
        return False


def test_missile_identification_standardization():
    """Test that missile identification uses is_missile consistently."""
    print("\n=== Testing missile identification standardization ===")

    try:
        # Check both action processor and environment use is_missile
        from reinforcement_learning.environment.spaces.action_space import ActionProcessor
        from reinforcement_learning.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv

        import inspect

        # Check action processor
        ap_source = ''.join(inspect.getsourcelines(ActionProcessor.apply)[0])
        ap_uses_is_missile = 'getattr(u, "is_missile", False)' in ap_source

        # Check environment
        env_source = ''.join(inspect.getsourcelines(BVRMultiAgentEnv._get_incoming_missiles_for_agent)[0])
        env_uses_is_missile = 'getattr(unit, "is_missile", False)' in env_source

        if ap_uses_is_missile and env_uses_is_missile:
            print("✅ Both action processor and environment use is_missile consistently")
            return True
        else:
            print(f"❌ Inconsistent missile identification - AP: {ap_uses_is_missile}, Env: {env_uses_is_missile}")
            return False

    except Exception as e:
        print(f"❌ Missile identification test failed: {e}")
        return False


def test_training_signals_in_infos():
    """Test that training signals are exposed in infos."""
    print("\n=== Testing training signals exposure ===")

    try:
        from reinforcement_learning.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv

        import inspect
        source_lines = inspect.getsourcelines(BVRMultiAgentEnv.step)[0]
        source_code = ''.join(source_lines)

        # Check for training signals being added to infos
        training_signals_checks = [
            'training_signals = self.action_processor.get_training_signals(unit_id)',
            'valid_missile_shots',
            'vetoed_missile_shots',
            'cooldown_left_missile_s',
            's_factor',
            'c_factor'
        ]

        all_present = all(check in source_code for check in training_signals_checks)

        if all_present:
            print("✅ Training signals and envelope scalars exposed in infos")
            return True
        else:
            print("❌ Missing some training signals in infos")
            return False

    except Exception as e:
        print(f"❌ Training signals test failed: {e}")
        return False


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
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)

    passed = sum(results)
    total = len(results)

    print(f"\n=== Summary ===")
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All sanity tests passed!")
    else:
        print("⚠️  Some tests failed - please review the fixes")