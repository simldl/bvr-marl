# BVR-MARL

A high-fidelity air-to-air combat simulation environment with a comprehensive reinforcement learning API for training autonomous fighter aircraft agents. 
This system combines realistic flight dynamics, radar simulation, missile guidance, and multi-agent RL training capabilities.
This is the repository corresponding to the paper "Multi-Agent Reinforcement Learning Environment for Beyond Visual Range Air Combat (BVR-MARL)" by Schosser et al.
This is currently the beta version of this repository and will be updated in the future.

## Overview

This repository implements a sophisticated Beyond Visual Range (BVR) air combat simulator designed for reinforcement learning research. The system features:

- **Realistic Flight Dynamics**: Energy-based aircraft control with authentic flight envelopes, drag, lift, and thrust modeling
- **Advanced Sensor Simulation**: Radar Cross Section (RCS)-based detection, Kalman filter tracking, and electronic warfare
- **Comprehensive Weapon Systems**: Fox-1 (SARH), Fox-2 (IR), and Fox-3 (ARH) missiles with multiple guidance algorithms
- **Multi-Agent RL Training**: Self-play training with Ray RLlib and PPO algorithm
- **Professional Analysis Tools**: Real-time 2D visualization and Tacview integration for 3D replay

## Example Scenarios

<!-- Video: Basic 2v2 BVR Engagement -->
[*Example video: 2v2 engagement with trained agents in Tacview*](https://github.com/simldl/air_to_air_rl/issues/4#issue-3795996273)

<!-- Video: Complex 4v4 Multi-Agent Scenario -->
[*Example video: 2v2 engagement with trained agents with custom visualization*](https://github.com/simldl/air_to_air_rl/issues/5#issue-3796003331)

---

## Installation

### Prerequisites

- Python 3.12+
- CUDA 12.8+ (for GPU acceleration)
- Conda or Miniconda (recommended)
- Git

### Method 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/simldl/bvr-marl
cd bvr-marl

# Create conda environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate rlenv

# Install PyTorch with CUDA support
# Windows:
install_pytorch_cuda.bat

# Linux/Mac:
chmod +x install_pytorch_cuda.sh
./install_pytorch_cuda.sh

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import ray; print(f'Ray: {ray.__version__}')"
```

### Method 2: Using pip

```bash
# Create virtual environment
python -m venv rlenv

# Activate environment
# Windows:
rlenv\Scripts\activate
# Linux/Mac:
source rlenv/bin/activate

# Install requirements
pip install -r requirements.txt

# Install PyTorch with CUDA separately
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
---

## Usage

### Training

Training is configured via [reinforcement_learning/configs/train_config.yaml](reinforcement_learning/configs/train_config.yaml).

#### Basic Training

```bash
# Activate environment
conda activate rlenv

# Start training
python reinforcement_learning/train.py
```

#### Monitor Training Progress

```bash
# Launch TensorBoard
tensorboard --logdir ./logs
```

Open your browser to `http://localhost:6006` to view training metrics.

#### Key Training Configuration

Edit [train_config.yaml](reinforcement_learning/configs/train_config.yaml) to customize training:

```yaml
env:
  num_agents_per_side: 2        # 2v2 combat scenario
  map_size: 200                 # 200km x 200km battlefield
  max_steps: 1024               # Maximum episode length

  aircraft_config:
    agent_type: "Eurofighter"   # F22, F35, Su57, Eurofighter
    opponent_type: "Eurofighter"

  missile_config:
    agent_missiles: ["AIM120_AMRAAM"]    # Meteor, AIM120_AMRAAM, etc.
    opponent_missiles: ["AIM120_AMRAAM"]

training:
  steps: 512                    # Training iterations
  learning_rate: 0.0003
  batch_size: 256
  num_epochs: 3
  gamma: 0.995                  # Discount factor

model:
  use_neural_wrapper: true      # Enable automation + neural control
  neural_wrapper_config:
    wrapped_action_dim: 4       # Neural network controls 4 actions
    full_action_dim: 10         # Total action space dimension
    automation_level: "balanced" # defensive/balanced/aggressive

  model_config:
    hidden_dim: 256             # Hidden layer size
    num_hidden_layers: 3        # Number of hidden layers
    activation: relu            # relu, tanh, elu
```

#### Resume Training from Checkpoint

```yaml
# Add to train_config.yaml:
resume_checkpoint: "models/bvr_combat/checkpoint_000100"
```

#### Load Weights Without Resuming

```yaml
# Add to train_config.yaml:
load_weights_from_checkpoint: "models/bvr_combat/checkpoint_000100"
```

---

### 2D Live View Visualization

Real-time 2D visualization of combat scenarios with trajectory plotting and reward tracking.

#### Usage

```bash
# View random policy (no trained model)
python visualization/2d_live_view.py

# View trained model
python visualization/2d_live_view.py --checkpoint models/train_config/checkpoint

# Customize visualization
python visualization/2d_live_view.py \
    --checkpoint <path> \
    --frames 200 \
    --save-video \
    --save-rewards

# Real-time playback (1 second per frame)
python visualization/2d_live_view.py \
    --checkpoint <path> \
    --interval 1000
```

#### Configuration

Edit [visualization/viz_config.yaml](visualization/viz_config.yaml):

```yaml
checkpoint_path: null  # or path to trained model
train_config_path: reinforcement_learning/configs/train_config.yaml

visualization:
  frames: 100              # Simulation length
  interval: 100            # Milliseconds between frames
  real_time_speed: false   # Sync to simulation tick rate
  save_video: false        # Save as video file
  save_rewards: false      # Save reward logs (CSV/JSON)
```

#### Features

- Live aircraft positions and trajectories
- Missile tracks and impacts
- Radar coverage visualization
- Ground track plotting
- Real-time reward tracking

---

### Tacview Integration

Generate professional ACMI format files for 3D replay in [Tacview](https://www.tacview.net/).

#### Usage

```bash
python tacview/generate_scenario.py \
    --checkpoint models/train_config/checkpoint \
    --frames 500
```

Output files are saved in [tacview/logs/](tacview/logs/) as `.txt.acmi` files.

#### Open in Tacview

1. Install Tacview (free or paid version)
2. Open the generated `.txt.acmi` file
3. Use Tacview's tools for:
   - 3D visualization of the engagement
   - Timeline navigation and slow-motion replay
   - Detailed object properties and statistics
   - Professional mission debriefing

---

## Customization

### Writing Custom Rewards

Reward functions are defined in [reinforcement_learning/environment/rewards/](reinforcement_learning/environment/rewards/).

#### Current Reward Structure

The system uses terminal/sparse rewards to avoid reward shaping issues:

```yaml
# In train_config.yaml
rewards:
  destruction_penalty: -200.0    # Agent is shot down
  kill_reward: 200.0             # Agent shoots down enemy
  last_team_reward: 40.0         # Team survival bonus
  boundary_violation_penalty: -200.0  # Leave combat area
```

#### Customizing Rewards

**Method 1: Modify configuration values**

Edit reward values in [train_config.yaml](reinforcement_learning/configs/train_config.yaml):

```yaml
rewards:
  destruction_penalty: -300.0    # Increase penalty
  kill_reward: 250.0             # Increase reward
  missile_hit_reward: 50.0       # Add new component
```

**Method 2: Extend reward calculator**

1. Open [terminal_rewards.py](reinforcement_learning/environment/rewards/terminal_rewards.py) or integrate a new file into the code.
2. Add new reward components:

```python
def calculate_missile_fired_penalty(self, agent_id: str) -> float:
    """Penalize excessive missile usage."""
    missiles_fired = self.get_missiles_fired(agent_id)
    return -10.0 * missiles_fired
```

3. Update [calculator.py](reinforcement_learning/environment/rewards/calculator.py) to integrate:

```python
def calculate_rewards(self) -> Dict[str, float]:
    rewards = {}
    for agent_id in self.agents:
        reward = 0.0
        reward += self.terminal_rewards.calculate_destruction_penalty(agent_id)
        reward += self.terminal_rewards.calculate_kill_reward(agent_id)
        reward += self.terminal_rewards.calculate_missile_fired_penalty(agent_id)  # New
        rewards[agent_id] = reward
    return rewards
```

---

### Writing Custom Network Architectures

Neural network architectures are defined in [reinforcement_learning/networks/](reinforcement_learning/networks/).

#### Current Architecture

The default architecture is a configurable MLP (Multi-Layer Perceptron):

```yaml
# In train_config.yaml
model:
  model_config:
    action_dim: 4              # Number of actions network controls
    hidden_dim: 256            # Hidden layer size
    num_hidden_layers: 3       # Number of hidden layers
    activation: relu           # relu, tanh, elu
    init_log_std: -0.5        # Initial log standard deviation
```

#### Creating a Custom Architecture

**Step 1: Create new encoder**

Create a file in [reinforcement_learning/networks/encoders/](reinforcement_learning/networks/encoders/):

```python
# custom_encoder.py
import torch
import torch.nn as nn
from typing import Dict, Any

class CustomEncoder(nn.Module):
    """Custom encoder architecture."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        input_dim = config["input_dim"]
        hidden_dim = config["hidden_dim"]

        # Define your architecture
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.mean(dim=-1)  # Global pooling
        x = self.fc(x)
        return x
```

**Step 2: Register encoder**

Update [reinforcement_learning/networks/rl_module/custom_multi_agent_model.py](reinforcement_learning/networks/rl_module/custom_multi_agent_model.py):

```python
from reinforcement_learning.networks.encoders.custom_encoder import CustomEncoder

# In setup() method:
if encoder_type == "custom":
    self.encoder = CustomEncoder(config)
elif encoder_type == "mlp":
    self.encoder = MLPEncoder(config)
# ... etc
```

**Step 3: Configure in training**

Update [train_config.yaml](reinforcement_learning/configs/train_config.yaml):

```yaml
model:
  model_config:
    encoder_type: "custom"    # Use your custom encoder
    hidden_dim: 256
    # Add any custom config parameters
    ...
```

#### Action Space and Neural Wrapper

The **neural wrapper** allows gradual curriculum learning by controlling which actions the neural network handles vs. scripted automation.

**Full action space (10 dimensions):**

- `[0]` **Ps**: Specific energy rate (climb/dive/accelerate)
- `[1]` **n**: Normal load factor (turn intensity)
- `[2]` **φ** (phi): Bank angle (turn direction)
- `[3]` **Target selection**: Which enemy to engage
- `[4]` **Missile firing**: Fire decision
- `[5]` **Gun firing**: Gun trigger
- `[6-9]` **Countermeasures**: Flares, chaff, ECM, decoys

**Neural wrapper configuration:**

```yaml
model:
  use_neural_wrapper: true
  neural_wrapper_config:
    wrapped_action_dim: 4           # NN controls: Ps, n, φ, missile_fire
    full_action_dim: 10             # Total action space
    automation_level: "balanced"    # defensive/balanced/aggressive
    enable_countermeasures: true    # Automation handles countermeasures
    enable_target_selection: true   # Automation handles targeting
    enable_gun_control: true        # Automation handles gun
```

**Automation strategies:**

- **defensive**: Conservative countermeasure use, prioritizes survival
- **balanced**: Moderate automation, balanced offense/defense
- **aggressive**: Proactive targeting, aggressive tactics

Implementation: [automation/strategies/](automation/strategies/)

---

## Project Structure

```
air_to_air_rl/
├── aircrafts/              # Aircraft models (F-22, F-35, Eurofighter, Su-57)
│   ├── types/              # Specific aircraft implementations
│   ├── control/            # Movement, weapons, countermeasures
│   └── systems/            # Sensors, radar, missile warner
├── missiles/               # Missile systems (Fox-1, Fox-2, Fox-3)
│   ├── fox3/               # Active radar homing (AMRAAM, Meteor)
│   └── guidance/           # Guidance algorithms (PN, APN, Lead, Loft)
├── radar/                  # Radar simulation with tracking
│   ├── tracking/           # Kalman filters (CV, CT, IMM)
│   └── ew/                 # Electronic warfare
├── physics/                # Flight dynamics and atmosphere
├── simulator/              # Core simulation engine
├── automation/             # AI automation strategies
│   ├── strategies/         # Defensive/Balanced/Aggressive
│   └── core/neural_wrapper/ # Neural + scripted control
├── reinforcement_learning/ # RL training system
│   ├── train.py            # Main training script
│   ├── configs/            # Training configurations
│   ├── environment/        # Gym environments
│   │   ├── gym/            # Multi-agent BVR environment
│   │   ├── rewards/        # Reward calculation
│   │   └── spaces/         # Observation/action spaces
│   ├── networks/           # Neural architectures
│   │   ├── encoders/       # MLP, LSTM, Transformer
│   │   └── rl_module/      # RLlib integration
│   └── models/             # Saved checkpoints
├── visualization/          # Visualization tools
│   ├── 2d_live_view.py     # Real-time 2D viewer
│   └── reward_logging/     # Reward tracking
├── tacview/                # Tacview ACMI integration
│   └── generate_scenario.py # Scenario generator
└── tests/                  # Comprehensive test suite
```

---

## Aircraft Types

- **F-22 Raptor**: US air superiority fighter
- **F-35 Lightning II**: US multirole fighter
- **Eurofighter Typhoon**: European multirole fighter
- **Su-57 Felon**: Russian 5th generation fighter

Each aircraft has unique flight characteristics, radar signatures, and performance envelopes.

---

## Missile Types

### Fox-3 (Active Radar Homing)
- **AIM-120 AMRAAM**: US beyond-visual-range missile
- **Meteor**: European long-range ramjet missile
- **R-77-1**: Russian medium-range missile
- **R-37M**: Russian ultra-long-range missile
---


## Contributing

Contributions are welcome! Please contact the author if you have any questions or face technical difficulties.

---

## License

MIT License

Copyright (c) 2026 Simon Schosser

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Citation

If you use this code, or any substantial portion of it, in academic research,
please cite the following paper:

```
@inbook{doi:10.2514/6.2026-1592,
author = {Simon Schosser and Carl O. Retzlaff and Axel Schulte},
title = {Multi-Agent Reinforcement Learning Environment for Beyond Visual Range Air Combat (BVR-MARL)},
booktitle = {AIAA SCITECH 2026 Forum},
chapter = {},
pages = {},
doi = {10.2514/6.2026-1592},
URL = {https://arc.aiaa.org/doi/abs/10.2514/6.2026-1592},
eprint = {https://arc.aiaa.org/doi/pdf/10.2514/6.2026-1592},
    abstract = { We present BVR-MARL, a Multi-Agent Reinforcement Learning (MARL) simulation environment for beyond-visual-range (BVR) air-to-air combat that targets the realism–throughput frontier required for scalable training. Existing platforms either omit BVR-critical effects (sensors/EW, multi-platform fusion, seeker-aware missiles) or are too slow for long Reinforcement Learning (RL) runs, motivating a modular, RL-native alternative. Our engine advances aircraft and missiles with lightweight 4-DoF (Degrees of Freedom) kinematics on a geodetic frame and an energy-based aerodynamic model; models terminal interactions via a small engagement zone with probabilistic proximity hits; implements a sensor stack with SNR-based radar detection, aspect-dependent Radar Cross Section (RCS), Doppler, and Electronic Warefare (EW, noise jamming and deception) with basic Electronic Counter-Countermeasures (ECCM); fuses detections over a data link with clustering and Constant Velocity (CV) Kalman tracking; and exposes stable multi-agent observations and safety-gated actions that integrate with common RL tooling for parallel rollouts. We demonstrate training readiness with a simple baseline and report per-tick complexity of the simulation while isolating the costs of physics, sensors, and terminal endgame. We contribute a fully specified, modular, open-source, RL-ready BVR simulator and results of a baseline that together form a platform for reproducible research on learned BVR tactics and fair benchmarking in this domain. }
}
```

---

## Contact

If you have any questions regarding the code, the paper or want to collaborate on this topic, feel free to contact the author Simon Schosser via email: simon.schosser@unibw.de.
