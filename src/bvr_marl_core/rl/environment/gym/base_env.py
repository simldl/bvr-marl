import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class BaseMultiAgentEnv(MultiAgentEnv):
    """
    Base class for multi-agent environments.
    - Initializes space manager, observation builder, action processor.
    - Implements generic reset/step logic that can be customized by subclasses.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.simulator = config["simulator"]
        self.max_steps = config.get("max_steps", 128)
        self.current_step = 0

        self.agent_ids = []
        self.all_agent_ids = []
        self.agent_to_unit_id = {}
        self.obs_space_mgr = None
        self.obs_builder = None
        self.action_processor = None
        self.reward_calculator = None

    def set_spaces(self, obs_space_mgr, action_space_mgr):
        self.obs_space_mgr = obs_space_mgr
        self.observation_space = obs_space_mgr.all()
        self.action_space = action_space_mgr.all()

    def set_obs_builder(self, builder):
        self.obs_builder = builder

    def set_action_processor(self, processor):
        self.action_processor = processor

    def set_reward_calculator(self, calculator):
        self.reward_calculator = calculator

    def reset(self, *, seed=None, options=None):
        raise NotImplementedError("reset() must be implemented in subclass.")

    def step(self, actions: dict[str, np.ndarray] | None = None):
        raise NotImplementedError("step() must be implemented in subclass.")

    def _spawn_agents(self):
        """Can be provided by subclass or env_helpers.py."""
        raise NotImplementedError("_spawn_agents() must be implemented in subclass or env_helpers.")

    def seed(self, seed: int | None = None):
        if seed is None:
            return
        if hasattr(self.simulator, "seed"):
            try:
                self.simulator.seed(seed)
            except (AttributeError, TypeError, ValueError, KeyError, IndexError, ZeroDivisionError):
                pass

    def get_action_space(self, agent_id):
        return self.action_space[agent_id]

    def get_observation_space(self, agent_id):
        return self.observation_space[agent_id]
