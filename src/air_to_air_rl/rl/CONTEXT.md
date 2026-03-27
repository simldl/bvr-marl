# Reinforcement Learning Module - Context

## Overview

The reinforcement learning module implements multi-agent air combat training using PPO (Proximal Policy Optimization) via Ray RLlib. The system trains fighter aircraft to perform BVR (Beyond Visual Range) combat through self-play, learning tactical maneuvers, energy management, weapon employment, and team coordination.

**Primary Components:**
- Multi-agent training infrastructure (PPO + Ray RLlib)
- BVR combat environment with realistic physics integration
- Energy + Lift-Vector action space (10-dimensional continuous control)
- Rich observation space with tactical overlays (NEZ, DLZ, SQI)
- Public baseline reward system for immediate training
- Standard PPO neural networks for policy and value functions

---

## Directory Structure

```
reinforcement_learning/
├── train.py                          # Main training script (PPO + Ray RLlib)
├── configs/                          # Hydra configuration files
│   └── train_config.yaml            # Main training configuration
│
├── training/                        # Modular training infrastructure
│   ├── callbacks/
│   │   ├── metrics.py               # Episode metrics logging to TensorBoard
│   │   ├── progress.py              # Progress bar display
│   │   └── checkpoint.py            # Smart checkpoint management
│   ├── environment_setup.py         # Environment configuration
│   ├── config_builder.py            # PPO configuration builder
│   ├── checkpoint_utils.py          # Checkpoint loading utilities
│   └── restore_algorithm.py         # Weight restoration from checkpoints
│
├── environment/                     # RL environment implementations
│   ├── gym/
│   │   └── bvr_multi_agent_env.py  # Main multi-agent BVR environment
│   ├── spaces/
│   │   ├── action_space/           # Energy + Lift-Vector action processing
│   │   └── observation/            # Tactical observation construction
│   └── rewards/
│       └── calculator.py           # Public baseline reward calculator
│
```

---

## Training Infrastructure

### Algorithm: PPO (Proximal Policy Optimization)

The project uses **Ray RLlib 2.50+** with the modern API stack. Training configuration in `train.py`:

```python
from ray.rllib.algorithms.ppo import PPOConfig

ppo_config = (
    PPOConfig()
    .framework("torch")
    .api_stack(
        enable_rl_module_and_learner=True,       # Modern API stack
        enable_env_runner_and_connector_v2=True,
    )
    .environment(
        env="BVRMultiAgentEnv",
        env_config=cfg.env,
        clip_actions=True,
    )
    .multi_agent(
        policies={"shared_policy"},              # Single shared policy
        policy_mapping_fn=policy_mapping_fn,
        policies_to_train=["shared_policy"],
    )
    .env_runners(
        num_env_runners=30,                      # Parallel rollouts
        rollout_fragment_length=128,
        batch_mode="truncate_episodes",
    )
    .learners(num_learners=0)                    # Local learner
    .training(
        gamma=0.995,
        lambda_=0.95,
        use_gae=True,                            # Enables GeneralAdvantageEstimation
        use_critic=True,
        train_batch_size_per_learner=4096,       # New API: per-learner batch size
        minibatch_size=128,
        num_epochs=1,
        vf_loss_coeff=0.5,
        entropy_coeff=0.01,
        clip_param=0.2,
    )
)
```

**Key Training Features:**
- **Modern API Stack**: Uses RLlib's standard networks
- **Self-play**: Shared policy trains against itself
- **Multi-agent coordination**: 1v1 to NvN scenarios supported
- **GPU acceleration**: Neural network training on GPU
- **Entropy regularization**: 0.01 coefficient to encourage exploration

---

## Environment: BVRMultiAgentEnv

**File:** `environment/gym/bvr_multi_agent_env.py`

The main RL environment orchestrates multi-agent air combat simulations. It inherits from `gym.Env` and implements the standard Gym interface with multi-agent extensions.

### Episode Flow

1. **reset()**: Initialize simulator, spawn aircraft at random positions/headings
2. **step(actions)**: Apply actions to all agents, tick simulator, compute rewards
3. **Termination conditions**:
   - All agents on one team destroyed
   - Max steps reached (1000 timesteps)
   - All agents out of bounds for extended period
4. **Return**: `obs_dict, reward_dict, done_dict, info_dict`

---

## Action Space

The action space uses the **Energy + Lift-Vector** paradigm with **10 continuous dimensions** in range [0, 1]:

| Index | Parameter | Description |
|-------|-----------|-------------|
| 0 | **Ps** | **Specific Energy Rate** (climb/dive/accelerate) |
| 1 | **n** | **Normal Load Factor** (turn intensity/g-load) |
| 2 | **φ** | **Bank Angle** (turn direction/roll) |
| 3 | **target_id** | Target selection |
| 4 | **missile_fire** | Missile launch trigger |
| 5 | **gun_fire** | Gun firing trigger |
| 6-9 | **CM[0:4]** | Countermeasures |

---

## Observation Space

The observation space provides rich tactical information to enable sophisticated behavior learning.

### Multi-slot design with variable-length contact lists:

```python
observation = {
    "own_state": np.ndarray(22),              # Own aircraft state
    "friendly_fighters": np.ndarray(N, 18),   # Friendly aircraft
    "enemy_fighters": np.ndarray(M, 20),      # Enemy aircraft  
    "friendly_missiles": np.ndarray(K, 12),   # Friendly missiles
    "enemy_missiles": np.ndarray(L, 12),      # Enemy missiles
}
```

**Key Features:**
- **DLZ (Dynamic Launch Zone)**: Missile employment envelope
- **SQI**: Shot quality metric
- **Energy state**: Specific energy and rates
- **Tactical geometry**: Range, aspect, closure rate

---

## Reward System

**File:** `environment/rewards/calculator.py`

Minimal public baseline reward calculator with simple terminal rewards:

```python
total_reward = (
    kill_reward +                    # +1.0 for enemy kill
    destruction_penalty +            # -1.0 for being shot down
    boundary_violation_penalty +     # -1.0 for map boundary violation
    last_team_reward +               # +0.5 for being last team standing
)
```

This baseline provides immediate training capability while allowing users to implement their own custom reward functions.

---

## Neural Network Architecture

This public release uses Ray RLlib's standard PPO networks. The framework supports custom network implementations that users can add.

**Standard Network Features:**
- Feedforward neural network
- Configurable hidden layers (default: 256x3)
- Separate policy and value heads
- Continuous action output via Gaussian distribution

---

## Configuration Management

### train_config.yaml

**Main configuration file:**

```yaml
env:
  num_agents_per_side: 2
  map_size: 200
  max_steps: 1024

training:
  learning_rate: 0.0003
  batch_size: 256
  n_envs: 1

model:
  model_config:
    action_dim: 10
    hidden_dim: 256
    num_hidden_layers: 3
    activation: relu

logging:
  log_dir: logs
  save_dir: models
  use_tensorboard: true
```

---

## Training Workflow

### 1. Start Training

```bash
python -m air_to_air_rl.training.train
```

### 2. Monitor Progress

```bash
tensorboard --logdir=logs
```

### 3. Visualize Results

```bash
python -m air_to_air_rl.visualization.live_view --checkpoint=models/checkpoint.pkl
```

---

## Integration Points

### With Simulator
- Environment orchestrates physics simulation
- Applies actions to aircraft through action processors
- Extracts observations through observation managers

### With Aircraft
- Energy + lift-vector control commands
- State extraction for observations
- Weapon system integration

### With Radar
- Detection processing for observations
- Tactical overlay computation (DLZ, SQI)
- Track quality assessment

---

## Future Directions

This public release provides a solid foundation for air combat RL research. Users can extend the system by:

- Implementing custom reward functions
- Adding curriculum learning
- Developing custom network architectures
- Extending to new aircraft types
- Adding multi-modal observations

---

## References

**Ray RLlib Documentation:**
- PPO: https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#ppo
- Multi-agent: https://docs.ray.io/en/latest/rllib/rllib-concepts.html#multi-agent

**Air Combat Literature:**
- "Fighter Combat: Tactics and Maneuvering" by Robert Shaw