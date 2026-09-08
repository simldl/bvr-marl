import pytest


class TestBVRMultiAgentEnv:
    """Test the BVR (Beyond Visual Range) multi-agent environment."""

    def test_environment_import(self):
        """Test that BVRMultiAgentEnv can be imported successfully."""
        try:
            from bvr_marl_core.rl.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv

            assert BVRMultiAgentEnv is not None
        except ImportError as e:
            pytest.fail(f"Expected imports failed: {e}")

    def test_tacview_logging_integration(self):
        """Test TacView logging integration if available."""
        try:
            from bvr_marl_core.rl.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv

            # Basic smoke test - just verify the class exists and can be referenced
            assert BVRMultiAgentEnv is not None
        except ImportError as e:
            pytest.fail(f"Expected imports failed: {e}")

    def test_error_handling_missing_simulator(self):
        """Test error handling for invalid configurations."""
        try:
            from bvr_marl_core.rl.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv

            # Test with missing simulator - should raise an error
            invalid_config = {}

            with pytest.raises((ValueError, KeyError, AttributeError, TypeError)):
                BVRMultiAgentEnv(invalid_config)
        except ImportError as e:
            pytest.fail(f"Expected imports failed: {e}")

    def test_reset_keeps_agent_unit_mapping_in_sync_for_controlled_geometry(self):
        """Controlled-geometry reset must not replace the shared mapping object."""
        from copy import deepcopy

        from bvr_marl_core.rl.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv
        from bvr_marl_core.simulator import Simulator
        from bvr_marl_core.utils.config_loader import load_train_config

        env_config = deepcopy(load_train_config("configs/training/basic.yaml")["env"])
        env_config["scenario_config"]["allowed_regimes"] = ["hot_80km"]
        env_config["scenario_config"]["randomize_regime"] = False
        env_config["simulator"] = Simulator()

        env = BVRMultiAgentEnv(env_config)

        try:
            env.reset()

            assert env.agent_to_unit_id is env.episode_manager.agent_to_unit_id
            assert env.helpers.agent_to_unit_id is env.agent_to_unit_id
            assert set(env.agent_to_unit_id) == set(env.all_agent_ids)
            assert all(uid in env.simulator.active_units for uid in env.agent_to_unit_id.values())
        finally:
            if hasattr(env, "close"):
                env.close()
