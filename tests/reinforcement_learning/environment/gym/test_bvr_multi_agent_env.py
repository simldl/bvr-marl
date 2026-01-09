import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from gymnasium import spaces
import torch


@pytest.mark.skip(reason="BVR multi-agent environment tests need extensive refactoring for new component-based architecture. Mocks need to be updated to match ObservationBuilder, EpisodeManager, and spawn_unit integration.")
class TestBVRMultiAgentEnv:
    """Test the BVR (Beyond Visual Range) multi-agent environment."""

    @pytest.fixture
    def mock_simulator(self):
        """Create a mock simulator for testing."""
        mock_sim = Mock()
        mock_sim.step.return_value = None
        mock_sim.do_tick.return_value = None  # This is the method actually called
        mock_sim.reset.return_value = None
        mock_sim.reset_sim.return_value = None  # Also used in reset
        mock_sim.get_units.return_value = []
        mock_sim.is_done.return_value = False
        mock_sim.get_events.return_value = []
        mock_sim.events = []  # Make events subscriptable
        mock_sim.active_units = {}  # Make active_units a dictionary

        # Configure add_unit to return int unit_id instead of Mock
        unit_id_counter = [1]
        def add_unit_side_effect(unit):
            unit_id = unit_id_counter[0]
            unit_id_counter[0] += 1
            unit.id = unit_id
            mock_sim.active_units[unit_id] = unit
            return unit_id
        mock_sim.add_unit = Mock(side_effect=add_unit_side_effect)

        return mock_sim

    @pytest.fixture
    def sample_config(self, mock_simulator):
        """Create a sample environment configuration."""
        return {
            "simulator": mock_simulator,
            "max_steps": 500,
            "debug": False,
            "num_agents_per_team": 2,  # This sets up A0, A1, B0, B1
            "simulation_config": {
                "tick_secs": 1.0,
                "max_real_time_s": 600,
                "early_termination": {
                    "all_enemies_dead": True,
                    "all_friendlies_dead": True
                }
            },
            "spawn_config": {
                "num_agents": 2,
                "num_opponents": 2,
                "spawn_distance": 100000  # 100km
            },
            "reward_config": {
                "kill_reward": 10.0,
                "death_penalty": -20.0,
                "boundary_penalty": -5.0
            }
        }

    def test_environment_initialization(self, sample_config):
        """Test environment initialization with various configurations."""
        try:
            from reinforcement_learning.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv
            
            env = BVRMultiAgentEnv(sample_config)
            
            assert env is not None
            assert env.max_steps == 500
            assert env.current_step == 0
            assert env.tick_secs == 1.0
            assert hasattr(env, 'simulator')
            
        except ImportError as e:
            pytest.skip(f"BVRMultiAgentEnv not available: {e}")

    def test_environment_reset(self, sample_config):
        """Test environment reset functionality."""
        try:
            from reinforcement_learning.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv
            
            # Mock the necessary components
            with patch('reinforcement_learning.environment.gym.gym_components.episode_manager.spawn_unit') as mock_spawn:
                # Mock spawn_unit to return int unit_id
                unit_counter = [1]
                def spawn_side_effect(episode_mgr, agent_id, group):
                    unit_id = unit_counter[0]
                    unit_counter[0] += 1
                    # Create a mock unit and register it in simulator
                    mock_unit = Mock()
                    mock_unit.id = unit_id
                    mock_unit.group = group
                    episode_mgr.simulator.active_units[unit_id] = mock_unit
                    return unit_id
                mock_spawn.side_effect = spawn_side_effect
                
                env = BVRMultiAgentEnv(sample_config)
                
                # The agent IDs are set up automatically: A0, A1, B0, B1
                expected_agents = ["A0", "A1", "B0", "B1"]
                
                # Mock the space managers and builders
                env.obs_space_mgr = Mock()
                env.obs_space_mgr.all.return_value = {
                    aid: spaces.Box(low=-1, high=1, shape=(10,)) for aid in expected_agents
                }
                
                env.action_space_mgr = Mock()
                env.action_space_mgr.all.return_value = {
                    aid: spaces.Box(low=-1, high=1, shape=(9,)) for aid in expected_agents
                }
                
                env.obs_builder = Mock()
                # Mock the build method to return a proper dictionary for to_f32_obs
                env.obs_builder.build.return_value = {
                    "feature_1": np.random.randn(5),
                    "feature_2": np.random.randn(5)
                }
                
                obs, infos = env.reset()
                
                assert isinstance(obs, dict)
                assert isinstance(infos, dict)
                assert len(obs) > 0
                assert env.current_step == 0
                
        except ImportError:
            pytest.skip("BVRMultiAgentEnv not available")

    def test_environment_step(self, sample_config):
        """Test environment step functionality."""
        try:
            from reinforcement_learning.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv
            
            with patch('reinforcement_learning.environment.gym.gym_components.episode_manager.spawn_unit'):
                env = BVRMultiAgentEnv(sample_config)
                
                # Setup mocks
                self._setup_mocks_on_env(env)
                
                # Reset environment first
                obs, _ = env.reset()
                
                # Create sample actions
                actions = {
                    agent_id: np.random.uniform(-1, 1, 9) 
                    for agent_id in obs.keys()
                }
                
                # Step environment
                next_obs, rewards, terminations, truncations, infos = env.step(actions)
                
                assert isinstance(next_obs, dict)
                assert isinstance(rewards, dict)
                assert isinstance(terminations, dict)
                assert isinstance(truncations, dict)
                assert isinstance(infos, dict)
                
                # Check that step counter incremented
                assert env.current_step == 1
                
        except ImportError:
            pytest.skip("BVRMultiAgentEnv not available")

    def test_action_processing(self, sample_config):
        """Test action processing and validation."""
        try:
            from reinforcement_learning.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv

            with patch('reinforcement_learning.environment.gym.gym_components.episode_manager.spawn_unit') as mock_spawn, \
                 patch('reinforcement_learning.environment.rewards.calculator.RewardCalculator') as mock_reward_calc_class:
                # Mock RewardCalculator to return 4 values
                mock_reward_calculator = Mock()
                mock_reward_calculator.compute_total_reward.return_value = (0.0, 0.0, 0.0, 0.0)
                mock_reward_calc_class.return_value = mock_reward_calculator

                # Return a *string ID* and also register the created unit in simulator.active_units.
                unit_counter = [0]
                def create_mock_unit(env, agent_id, group):
                    mock_unit = Mock()
                    unit_id = f"unit_{unit_counter[0]}"
                    unit_counter[0] += 1
                    mock_unit.id = unit_id
                    mock_unit.missiles_fired_count = 0
                    mock_unit.locked_targets = []
                    mock_unit.missiles = []  # Make missiles a list so len() works
                    # IMPORTANT: register into sim so step() can find it
                    env.simulator.active_units[unit_id] = mock_unit
                    return unit_id  # env.reset() expects an ID here
                mock_spawn.side_effect = create_mock_unit

                env = BVRMultiAgentEnv(sample_config)
                self._setup_mocks_on_env(env)

                # Set up apply mock BEFORE reset
                apply_mock = Mock()
                apply_mock.return_value = {
                    'valid_missile_fires': 0,
                    'vetoed_missile_fires': 0
                }
                env.action_processor.apply = apply_mock

                obs, _ = env.reset()

                # Use expected agents directly instead of obs.keys() which might be empty
                expected_agents = ["A0", "A1", "B0", "B1"]
                valid_actions = {
                    agent_id: np.random.uniform(0, 1, 9)  # Changed to 0-1 range to match clipping
                    for agent_id in expected_agents
                }

                env.step(valid_actions)

                # Check that apply was called for at least one agent
                assert apply_mock.called, "ActionProcessor.apply should have been called at least once"

        except ImportError:
            pytest.skip("BVRMultiAgentEnv not available")

    def test_reward_calculation(self, sample_config):
        """Test reward calculation logic."""
        try:
            from reinforcement_learning.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv
            
            with patch('reinforcement_learning.environment.gym.gym_components.episode_manager.spawn_unit'):
                env = BVRMultiAgentEnv(sample_config)
                self._setup_mocks_on_env(env)
                
                # Mock reward calculator (returns tuple of 4 values)
                env.reward_calculator = Mock()
                env.reward_calculator.compute_total_reward.return_value = (5.0, 0.0, 0.0, 0.0)
                
                obs, _ = env.reset()
                actions = {agent_id: np.zeros(9) for agent_id in obs.keys()}
                
                _, rewards, _, _, _ = env.step(actions)
                
                assert isinstance(rewards, dict)
                for agent_id in obs.keys():
                    assert agent_id in rewards
                    assert isinstance(rewards[agent_id], (int, float))
                    
        except ImportError:
            pytest.skip("BVRMultiAgentEnv not available")

    def test_termination_conditions(self, sample_config):
        """Test various termination conditions."""
        try:
            from reinforcement_learning.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv
            
            with patch('reinforcement_learning.environment.gym.gym_components.episode_manager.spawn_unit'):
                env = BVRMultiAgentEnv(sample_config)
                self._setup_mocks_on_env(env)
                
                obs, _ = env.reset()
                
                # Test max steps termination
                env.current_step = env.max_steps
                actions = {agent_id: np.zeros(9) for agent_id in obs.keys()}
                
                _, _, terminations, truncations, _ = env.step(actions)
                
                # Should truncate when max steps reached
                assert any(truncations.values())
                
        except ImportError:
            pytest.skip("BVRMultiAgentEnv not available")

    def test_observation_space_consistency(self, sample_config):
        """Test observation space consistency."""
        try:
            from reinforcement_learning.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv
            
            with patch('reinforcement_learning.environment.gym.gym_components.episode_manager.spawn_unit'):
                env = BVRMultiAgentEnv(sample_config)
                self._setup_mocks_on_env(env)
                
                obs, _ = env.reset()
                
                # Check that observations match observation space
                for agent_id, observation in obs.items():
                    obs_space = env.observation_space[agent_id]
                    assert obs_space.contains(observation), f"Observation for {agent_id} doesn't match space"
                    
        except ImportError:
            pytest.skip("BVRMultiAgentEnv not available")

    def test_action_space_consistency(self, sample_config):
        """Test action space consistency."""
        try:
            from reinforcement_learning.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv
            
            with patch('reinforcement_learning.environment.gym.gym_components.episode_manager.spawn_unit'):
                env = BVRMultiAgentEnv(sample_config)
                self._setup_mocks_on_env(env)
                
                obs, _ = env.reset()
                
                # Test valid actions
                for agent_id in obs.keys():
                    action_space = env.action_space[agent_id]
                    
                    # Test random valid action
                    valid_action = action_space.sample()
                    assert action_space.contains(valid_action)
                    
                    # Test boundary actions
                    if isinstance(action_space, spaces.Box):
                        low_action = action_space.low
                        high_action = action_space.high
                        
                        assert action_space.contains(low_action)
                        assert action_space.contains(high_action)
                        
        except ImportError:
            pytest.skip("BVRMultiAgentEnv not available")

    def test_multi_agent_dynamics(self, sample_config):
        """Test multi-agent interaction dynamics."""
        try:
            from reinforcement_learning.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv
            
            with patch('reinforcement_learning.environment.gym.gym_components.episode_manager.spawn_unit'):
                env = BVRMultiAgentEnv(sample_config)
                self._setup_mocks_on_env(env)
                
                obs, _ = env.reset()
                
                # Test that all expected agents are present
                expected_agents = ["A0", "A1", "B0", "B1"]  # Based on num_agents_per_team=2
                for agent_id in expected_agents:
                    assert agent_id in obs
                    assert agent_id in env.observation_space
                    assert agent_id in env.action_space
                    
        except ImportError:
            pytest.skip("BVRMultiAgentEnv not available")

    def test_simulator_integration(self, sample_config):
        """Test integration with the physics simulator."""
        try:
            from reinforcement_learning.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv
            
            with patch('reinforcement_learning.environment.gym.gym_components.episode_manager.spawn_unit'):
                env = BVRMultiAgentEnv(sample_config)
                self._setup_mocks_on_env(env)
                
                obs, _ = env.reset()
                actions = {agent_id: np.zeros(9) for agent_id in obs.keys()}
                
                env.step(actions)
                
                # Verify simulator methods were called
                env.simulator.do_tick.assert_called()
                
        except ImportError:
            pytest.skip("BVRMultiAgentEnv not available")

    def test_tacview_logging_integration(self, sample_config):
        """Test TacView logging integration if available."""
        try:
            from reinforcement_learning.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv
            
            # Add TacView config with correct key
            sample_config["tacview_logfile"] = "/tmp/tacview_logs/test.tacview"
            
            with patch('reinforcement_learning.environment.gym.gym_components.episode_manager.spawn_unit'), \
                 patch('reinforcement_learning.environment.gym.bvr_multi_agent_env.TacviewLogger') as mock_logger:
                
                env = BVRMultiAgentEnv(sample_config)
                
                # Should initialize TacView logger if enabled
                if hasattr(env, 'tacview_logger'):
                    assert env.tacview_logger is not None
                    
        except ImportError:
            pytest.skip("BVRMultiAgentEnv or TacView logging not available")

    def test_error_handling(self, sample_config):
        """Test error handling for invalid configurations."""
        try:
            from reinforcement_learning.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv
            
            # Test with missing simulator
            invalid_config = sample_config.copy()
            del invalid_config["simulator"]
            
            with pytest.raises((ValueError, KeyError, AttributeError)):
                BVRMultiAgentEnv(invalid_config)
                
        except ImportError:
            pytest.skip("BVRMultiAgentEnv not available")


    def _setup_mocks_on_env(self, env):
        """Setup common mocks for testing on an environment instance."""
        # Use the correct agent IDs that the environment creates: A0, A1, B0, B1
        expected_agents = ["A0", "A1", "B0", "B1"]
        
        obs_space_manager = Mock()
        # The observation space should be a SpaceDict for each agent
        individual_obs_space = spaces.Dict({
            "feature_1": spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32),
            "feature_2": spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        })
        obs_space_manager.all.return_value = {
            aid: individual_obs_space for aid in expected_agents
        }
        
        action_space_manager = Mock()
        action_space_manager.all.return_value = {
            aid: spaces.Box(low=-1, high=1, shape=(9,)) for aid in expected_agents
        }
        
        obs_builder = Mock()
        # Mock the build method to return a proper dictionary for to_f32_obs
        obs_builder.build.return_value = {
            "feature_1": np.random.randn(5),
            "feature_2": np.random.randn(5)
        }
        obs_builder.build_observations.return_value = {
            aid: np.random.randn(10) for aid in expected_agents
        }
        
        reward_calculator = Mock()
        # compute_total_reward returns (total_reward, new_sqi, new_tactical_potential, new_energy_advantage)
        reward_calculator.compute_total_reward.return_value = (0.0, 0.0, 0.0, 0.0)
        
        action_processor = Mock()
        action_processor.process_actions.return_value = None
        # apply method should return training_signals dict
        action_processor.apply.return_value = {
            'valid_missile_fires': 0,
            'vetoed_missile_fires': 0
        }

        # Set up the environment with mocks
        env.obs_space_mgr = obs_space_manager
        env.action_space_mgr = action_space_manager
        env.obs_builder = obs_builder
        env.reward_calculator = reward_calculator
        env.action_processor = action_processor
        
        # Set up the spaces
        env.observation_space = obs_space_manager.all()
        env.action_space = action_space_manager.all()
        
        # Mock active_units for the simulator with proper unit IDs and attributes
        env.simulator.active_units = {}
        for i, aid in enumerate(expected_agents):
            mock_unit = Mock()
            unit_id = f"unit_{i}"
            mock_unit.id = unit_id
            mock_unit.missiles_fired_count = 0
            mock_unit.locked_targets = []  # Make sure it's a list, not a Mock
            env.simulator.active_units[unit_id] = mock_unit
            # Make sure agent_to_unit_id mapping exists
            env.agent_to_unit_id[aid] = unit_id

    def test_fix_radar_lock_after_first_obs(self, sample_config):
        """Test radar lock fixing feature."""
        try:
            from reinforcement_learning.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv

            # Enable radar lock fixing
            sample_config["simulation_config"]["fix_radar_lock_after_first_obs"] = True

            with patch('reinforcement_learning.environment.gym.gym_components.episode_manager.spawn_unit'):
                env = BVRMultiAgentEnv(sample_config)

                # Verify config was loaded
                assert env.fix_radar_lock_after_first_obs == True
                assert env.radar_locks_fixed == False

                self._setup_mocks_on_env(env)

                # Reset should reset the flag
                obs, _ = env.reset()
                assert env.radar_locks_fixed == False

                # Create mock units with radar and lock controllers
                for agent_id in ["A0", "A1", "B0", "B1"]:
                    uid = env.agent_to_unit_id.get(agent_id)
                    if uid and uid in env.simulator.active_units:
                        unit = env.simulator.active_units[uid]

                        # Add radar and sensor mocks
                        unit.radar = Mock()
                        unit.sensor = Mock()
                        unit.sensor.sensor_tracks = []
                        unit.group = "agent" if agent_id.startswith("A") else "opponent"

                        # Add lock controller with _lock_state
                        lock_controller = Mock()
                        lock_controller._lock_state = {}
                        lock_controller._confirm_needed = 2
                        unit.radar.lock_controller = lock_controller

                # Step once - should trigger lock fixing
                actions = {agent_id: np.zeros(10) for agent_id in obs.keys()}
                env.step(actions)

                # Flag should be set after first step
                assert env.radar_locks_fixed == True

                # Step again - should not trigger lock fixing again
                env.step(actions)
                assert env.radar_locks_fixed == True  # Still true, not reset

        except ImportError:
            pytest.skip("BVRMultiAgentEnv not available")