"""Pytest tests for CustomMultiAgentRLModule."""
import pytest
import torch
import torch.nn as nn
from gymnasium.spaces import Box, Dict as GymDict, Discrete
import numpy as np
from ray.rllib.core.columns import Columns


class TestCustomMultiAgentRLModule:
    """Test the custom multi-agent RL module."""

    @pytest.fixture
    def sample_obs_space(self):
        """Create a sample observation space for testing."""
        return GymDict({
            "own_state": Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32),
            "enemy_state": Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32),
        })

    @pytest.fixture
    def sample_action_space(self):
        """Create a sample action space for testing."""
        return Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)

    @pytest.fixture
    def discrete_action_space(self):
        """Create a discrete action space for testing."""
        return Discrete(5)

    @pytest.fixture
    def mlp_model_config(self):
        """Create a sample MLP model configuration."""
        return {
            "hidden_dim": 64,
            "num_hidden_layers": 2,
            "activation": "relu",
            "use_neural_wrapper": True,
            "wrapped_action_dim": 4,
            "full_action_dim": 10,
            "init_log_std": -0.5,
        }

    def test_module_initialization_mlp(self, sample_obs_space, sample_action_space, mlp_model_config):
        """Test module initialization with MLP encoder."""
        from reinforcement_learning.networks.rl_module.custom_multi_agent_model import CustomMultiAgentRLModule

        module = CustomMultiAgentRLModule(
            observation_space=sample_obs_space,
            action_space=sample_action_space,
            model_config=mlp_model_config
        )

        assert module is not None
        assert hasattr(module, 'encoder')
        assert hasattr(module, 'policy_head')
        assert hasattr(module, 'value_head')
        assert hasattr(module, 'action_wrapper')

    def test_discrete_action_space_handling(self, sample_obs_space, discrete_action_space, mlp_model_config):
        """Test module with discrete action space."""
        from reinforcement_learning.networks.rl_module.custom_multi_agent_model import CustomMultiAgentRLModule

        # Adjust config for discrete actions
        config = mlp_model_config.copy()
        config['action_dim'] = discrete_action_space.n
        config['use_neural_wrapper'] = False  # Discrete actions don't use wrapper

        module = CustomMultiAgentRLModule(
            observation_space=sample_obs_space,
            action_space=discrete_action_space,
            model_config=config
        )

        assert module is not None
        assert not module.is_continuous

    def test_forward_inference(self, sample_obs_space, sample_action_space, mlp_model_config):
        """Test forward pass in inference mode."""
        from reinforcement_learning.networks.rl_module.custom_multi_agent_model import CustomMultiAgentRLModule

        module = CustomMultiAgentRLModule(
            observation_space=sample_obs_space,
            action_space=sample_action_space,
            model_config=mlp_model_config
        )

        # Create sample batch
        batch = {
            Columns.OBS: {
                "own_state": torch.randn(4, 10),
                "enemy_state": torch.randn(4, 8),
            }
        }

        # Forward pass
        module.eval()
        with torch.no_grad():
            output = module.forward_inference(batch)

        # Check outputs
        assert Columns.ACTION_DIST_INPUTS in output
        assert Columns.ACTIONS in output
        assert Columns.VF_PREDS in output

        # Check shapes
        action_dist_inputs = output[Columns.ACTION_DIST_INPUTS]
        actions = output[Columns.ACTIONS]
        vf_preds = output[Columns.VF_PREDS]

        assert action_dist_inputs.shape[0] == 4  # batch size
        assert actions.shape[0] == 4  # batch size
        assert vf_preds.shape[0] == 4  # batch size

    def test_forward_exploration(self, sample_obs_space, sample_action_space, mlp_model_config):
        """Test forward pass in exploration mode."""
        from reinforcement_learning.networks.rl_module.custom_multi_agent_model import CustomMultiAgentRLModule

        module = CustomMultiAgentRLModule(
            observation_space=sample_obs_space,
            action_space=sample_action_space,
            model_config=mlp_model_config
        )

        # Create sample batch
        batch = {
            Columns.OBS: {
                "own_state": torch.randn(4, 10),
                "enemy_state": torch.randn(4, 8),
            }
        }

        # Forward pass
        module.eval()
        with torch.no_grad():
            output = module.forward_exploration(batch)

        # Check outputs (similar to inference)
        assert Columns.ACTION_DIST_INPUTS in output
        assert Columns.ACTIONS in output
        assert Columns.VF_PREDS in output

    def test_forward_train(self, sample_obs_space, sample_action_space, mlp_model_config):
        """Test forward pass in training mode."""
        from reinforcement_learning.networks.rl_module.custom_multi_agent_model import CustomMultiAgentRLModule

        module = CustomMultiAgentRLModule(
            observation_space=sample_obs_space,
            action_space=sample_action_space,
            model_config=mlp_model_config
        )

        # Create sample batch
        batch = {
            Columns.OBS: {
                "own_state": torch.randn(4, 10),
                "enemy_state": torch.randn(4, 8),
            }
        }

        # Forward pass
        module.train()
        output = module.forward_train(batch)

        # Training forward should include value function outputs
        assert Columns.ACTION_DIST_INPUTS in output
        assert Columns.VF_PREDS in output

    def test_gradient_flow(self, sample_obs_space, sample_action_space, mlp_model_config):
        """Test gradient flow through the network."""
        from reinforcement_learning.networks.rl_module.custom_multi_agent_model import CustomMultiAgentRLModule

        module = CustomMultiAgentRLModule(
            observation_space=sample_obs_space,
            action_space=sample_action_space,
            model_config=mlp_model_config
        )

        # Enable gradient computation
        module.train()

        # Create sample batch
        batch = {
            Columns.OBS: {
                "own_state": torch.randn(4, 10, requires_grad=True),
                "enemy_state": torch.randn(4, 8, requires_grad=True),
            }
        }

        # Forward pass
        output = module.forward_train(batch)

        # Compute dummy loss including both policy and value outputs for full gradient flow
        vf_loss = output[Columns.VF_PREDS].sum()
        policy_loss = output[Columns.ACTION_DIST_INPUTS].sum()
        total_loss = vf_loss + policy_loss
        total_loss.backward()

        # Check that gradients exist
        has_grad = False
        for param in module.parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                break
        assert has_grad, "No gradients found in model parameters"

    def test_model_serialization(self, sample_obs_space, sample_action_space, mlp_model_config):
        """Test model serialization and deserialization."""
        from reinforcement_learning.networks.rl_module.custom_multi_agent_model import CustomMultiAgentRLModule

        module = CustomMultiAgentRLModule(
            observation_space=sample_obs_space,
            action_space=sample_action_space,
            model_config=mlp_model_config
        )

        # Test state_dict serialization
        state_dict = module.state_dict()
        assert len(state_dict) > 0

        # Test loading state dict
        new_module = CustomMultiAgentRLModule(
            observation_space=sample_obs_space,
            action_space=sample_action_space,
            model_config=mlp_model_config
        )

        new_module.load_state_dict(state_dict)

        # Check that parameters match
        for (name1, param1), (name2, param2) in zip(
            module.named_parameters(),
            new_module.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1.data, param2.data, atol=1e-6)

    def test_batch_size_handling(self, sample_obs_space, sample_action_space, mlp_model_config):
        """Test handling of different batch sizes."""
        from reinforcement_learning.networks.rl_module.custom_multi_agent_model import CustomMultiAgentRLModule

        module = CustomMultiAgentRLModule(
            observation_space=sample_obs_space,
            action_space=sample_action_space,
            model_config=mlp_model_config
        )

        module.eval()
        # Test different batch sizes
        for batch_size in [1, 8, 16, 32]:
            batch = {
                Columns.OBS: {
                    "own_state": torch.randn(batch_size, 10),
                    "enemy_state": torch.randn(batch_size, 8),
                }
            }

            with torch.no_grad():
                output = module.forward_inference(batch)

            assert output[Columns.ACTION_DIST_INPUTS].shape[0] == batch_size
            assert output[Columns.ACTIONS].shape[0] == batch_size

    def test_compute_values_api(self, sample_obs_space, sample_action_space, mlp_model_config):
        """Test ValueFunctionAPI compute_values method."""
        from reinforcement_learning.networks.rl_module.custom_multi_agent_model import CustomMultiAgentRLModule

        module = CustomMultiAgentRLModule(
            observation_space=sample_obs_space,
            action_space=sample_action_space,
            model_config=mlp_model_config
        )

        batch = {
            Columns.OBS: {
                "own_state": torch.randn(4, 10),
                "enemy_state": torch.randn(4, 8),
            }
        }

        module.eval()
        with torch.no_grad():
            values = module.compute_values(batch)

        assert values.shape[0] == 4
        assert values.dim() == 1  # Should be flat (batch,)

    def test_action_wrapper_integration(self, sample_obs_space, sample_action_space, mlp_model_config):
        """Test action wrapper integration."""
        from reinforcement_learning.networks.rl_module.custom_multi_agent_model import CustomMultiAgentRLModule

        module = CustomMultiAgentRLModule(
            observation_space=sample_obs_space,
            action_space=sample_action_space,
            model_config=mlp_model_config
        )

        # Check that action wrapper is properly configured
        assert module.action_wrapper is not None
        assert module.action_wrapper.use_wrapper
        assert module.action_wrapper.wrapped_action_dim == 4
        assert module.action_wrapper.full_action_dim == 10

        # Test that actions are properly expanded
        batch = {
            Columns.OBS: {
                "own_state": torch.randn(2, 10),
                "enemy_state": torch.randn(2, 8),
            }
        }

        module.eval()
        with torch.no_grad():
            output = module.forward_inference(batch)

        # Actions should be expanded to full dimension
        actions = output[Columns.ACTIONS]
        assert actions.shape[-1] == 10  # Network outputs full_action_dim after expansion
