import pytest
import numpy as np
from reinforcement_learning.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv
from simulator.simulator import Simulator


def test_environment_creation_with_speed_control():
    """Test that the environment can be created with speed control enabled."""
    config = {
        "simulator": Simulator(tick_secs=0.1),
        "num_agents_per_team": 1,
        "map_size": 100.0,
        "max_steps": 10,
    }

    # Should create without errors
    env = BVRMultiAgentEnv(config)

    # Should have action processor (speed control is enabled by default)
    assert hasattr(env, 'action_processor')
    # Check if either speed control or energy space is enabled (both use advanced control)
    has_advanced_control = (env.action_processor.use_speed_control or
                           env.action_processor.use_energy_space)
    assert has_advanced_control == True

    # Should be able to reset
    obs, infos = env.reset()

    # Should have agents
    assert len(obs) > 0


def test_environment_step_with_speed_control():
    """Test that the environment can step with speed control actions."""
    config = {
        "simulator": Simulator(tick_secs=0.1),
        "num_agents_per_team": 1,
        "map_size": 100.0,
        "max_steps": 10,
    }

    env = BVRMultiAgentEnv(config)
    obs, infos = env.reset()

    # Get agent ID
    agent_id = list(obs.keys())[0]

    # Create action with speed control
    # action[0] = 0.7 should request speed around Mach 0.7
    action = np.array([0.7, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    actions = {agent_id: action}

    # Should step without errors
    obs, rewards, terminateds, truncateds, infos = env.step(actions)

    # Should have computed a throttle value
    debug_info = env.action_processor.get_debug_info(env.agent_to_unit_id[agent_id])

    # Check if using speed control or energy space mode
    if env.action_processor.use_speed_control and not env.action_processor.use_energy_space:
        assert 'speed_control' in debug_info
        speed_info = debug_info['speed_control']
        # Should have reasonable values
        assert 'v_desired' in speed_info
        assert 'throttle_filtered' in speed_info  # Changed from 'throttle_computed'
        assert 0.0 <= speed_info['throttle_filtered'] <= 1.0
    elif env.action_processor.use_energy_space:
        # Energy space mode has different debug structure
        assert 'energy_space' in debug_info
    else:
        # Legacy mode
        assert debug_info['mode'] == 'legacy'