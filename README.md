# BVR-MARL: Air Combat RL Environment

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A high-fidelity air-to-air combat simulation environment with GUI and training support for reinforcement learning research. This public release provides a complete BVR (Beyond Visual Range) combat environment with immediate training capabilities.

## 🚀 Quick Start

```bash
# Install the package  
pip install -e .

# Launch the GUI
air2air-gui

# Or start training directly
air2air-train --config configs/train_config_simplified.yaml
```

## 🎯 What This Release Provides

### ✅ Complete BVR Combat Simulator
- **Realistic aircraft models**: F-22, F-35, Eurofighter Typhoon, Su-57  
- **Advanced missile systems**: AIM-120 AMRAAM, Meteor, R-77, Python, Sidewinder
- **Radar and EW simulation**: RCS-based detection, ECM/ECCM, data fusion
- **Physics engine**: Energy-based flight dynamics with authentic performance models

### ✅ Training-Ready RL Environment  
- **Gymnasium environments**: `BVRMultiAgentEnv` for full complexity, `SimplifiedMultiAgentEnv` for fast training
- **Ray RLlib integration**: Uses default PPO with framework-default models for public release
- **Multi-agent framework**: Self-play and asymmetric training scenarios
- **Immediate training**: Start training agents right away with included configurations

### ✅ Professional GUI & Analysis Tools
- **Streamlit GUI**: Complete control panel for training, visualization, and analysis
- **Live visualization**: Real-time 2D combat visualization with tactical overlays  
- **Tacview export**: Professional 3D replay analysis compatible with DCS World
- **Automation testing**: Compare scripted controllers vs RL agents

### ✅ Research & Extension Framework
- **Open environment API**: Full access to simulator state for research and analysis
- **Automation system**: Behavior trees and tactical controllers for baseline comparisons
- **Training-ready baseline setup**: Public-safe configuration for immediate training

## 🏗️ Architecture Overview

```
air_to_air_rl/
├── aircrafts/          # Aircraft models and systems
├── missiles/           # Missile guidance and physics  
├── radar/              # Radar simulation and tracking
├── physics/            # Flight dynamics and physics
├── simulator/          # Core simulation engine
├── rl/environment/     # Gymnasium environments
├── gui/                # Streamlit control panel
├── visualization/      # Live view and analysis tools
├── tacview/            # Tacview export functionality
└── training/           # Training entry points
```

## 📦 Installation

### Prerequisites
- Python 3.12+
- Node.js (for visualization symbols)
- CUDA 12.8+ (optional, for GPU acceleration)
- Git

### Install from Source
```bash
git clone https://github.com/simldl/air_to_air_rl
cd air_to_air_rl
pip install -e .

# Generate visualization symbols (required for proper GUI visualization)
# Option 1: Automatic setup (recommended)
air2air-setup-symbols

# Option 2: Manual setup
# cd src/air_to_air_rl/visualization/symbols
# npm install milsymbol canvas
# node generate_symbols.js
# cd ../../../../
```

### With GPU Support
```bash
pip install -e .[gpu]
```

### Development Dependencies
```bash
pip install -e .[dev]   # Adds pytest for testing
```

**Note**: Code quality tools (ruff) are included by default for all installations.

## 🎮 Usage

### GUI Workflow (Recommended for New Users)
```bash
# Launch the control panel
air2air-gui
```

The GUI provides:
- **Training Dashboard**: Configure and start training runs
- **Visualization Panel**: Real-time combat visualization  
- **Tacview Generator**: Export scenarios for 3D analysis
- **Analysis Interface**: Training metrics and performance analysis

### Command Line Training

#### Simple Environment (Faster Training)
```bash
air2air-train-simple --config src/air_to_air_rl/rl/configs/train_config_simplified.yaml
```

#### Full BVR Environment  
```bash
air2air-train --config src/air_to_air_rl/rl/configs/train_config_standard.yaml
```

#### Adaptive Training (Recommended)
The system can automatically detect your hardware capabilities and adjust training parameters accordingly:

```bash
# Adaptive training with automatic system detection
air2air-train --adaptive --config src/air_to_air_rl/rl/configs/train_config_standard.yaml

# Simplified adaptive training
air2air-train-simple --adaptive --config src/air_to_air_rl/rl/configs/train_config_simplified.yaml

# Adaptive training without fallback on failure
air2air-train --adaptive --no-fallback --config your_config.yaml
```

**Adaptive Training Features:**
- **Automatic system assessment**: Detects CPU, memory, and GPU capabilities
- **Tier-based configuration**: Automatically selects optimal training parameters
- **Intelligent fallback**: If training fails, automatically tries lower-resource configurations
- **Configuration tiers**:
  - `minimal`: 1 worker, small batch sizes (8GB+ RAM, 4+ CPU threads)
  - `low`: 2 workers, modest batches (8GB+ RAM, 4+ CPU threads)
  - `medium`: 4 workers, balanced config (16GB+ RAM, 8+ CPU threads)  
  - `high`: 8 workers, full performance (32GB+ RAM, 16+ CPU threads, 8GB+ GPU)

### Live Visualization
```bash
# Watch live training or replay scenarios
air2air-view --config configs/viz_config_default.yaml

# Behavior tree visualization
air2air-view-behavior-tree --scenario 2v2
```

### Tacview Export
```bash
# Generate Tacview files for 3D replay
air2air-tacview --checkpoint path/to/checkpoint --episodes 10
```

## 🔧 Customization & Extension


### Environment Configuration
Modify scenarios in `src/air_to_air_rl/rl/configs/`:

```yaml
env:
  num_agents_per_team: 2
  map_size_km: 300
  max_steps: 1200
  agent_aircraft_type: "F22"
  opponent_aircraft_type: "Su57"
```

## 📊 Training Results & Analysis

### Visualization Features
- Real-time tactical situation display
- Missile flight paths and guidance
- Radar coverage and detections  
- Energy management visualization
- Performance metrics dashboard

### Export & Analysis
- Tacview (.acmi) files for DCS-compatible 3D replay
- Training metrics logging with TensorBoard
- Episode performance analysis
- Automated comparison vs scripted baselines

## 🎯 Public Release Scope

### What's Included
- ✅ Complete BVR combat simulator
- ✅ Training-ready RL environments  
- ✅ GUI and visualization tools
- ✅ Tacview export and analysis
- ✅ Scripted automation for baselines
- ✅ Public reward baseline for immediate training

### What's Not Included
- ❌ Advanced reward systems
- ❌ Internal performance benchmarks

This design allows immediate research use while providing clear extension points for custom implementations.

## 📚 Documentation

- **Environment API**: See `src/air_to_air_rl/rl/CONTEXT.md` for detailed environment documentation
- **Aircraft Models**: Individual aircraft specifications in `aircrafts/types/`  
- **Missile Systems**: Guidance algorithms and parameters in `missiles/`
- **Training Guide**: Configuration examples in `src/air_to_air_rl/rl/configs/`

## 🤝 Contributing

This is a research codebase designed for experimentation and extension. Users are encouraged to:

1. **Add aircraft/missile models** for expanded scenario coverage
2. **Extend environment features** and scenario configurations
3. **Improve GUI components** and user experience
4. **Enhance visualization tools** and analysis capabilities
5. **Share public-safe analysis tools** and automation improvements

Please note: This public release excludes proprietary components. Contributions should use only publicly available domain knowledge.

## 📄 Citation

If you use this environment in your research, please cite:

```bibtex
@article{schosser2024bvrmarl,
  title={Multi-Agent Reinforcement Learning Environment for Beyond Visual Range Air Combat (BVR-MARL)},
  author={Schosser, Simon and [Additional Authors]},
  journal={[Journal Name]},
  year={2024}
}
```

## 🔧 System Requirements

### Minimum Requirements
- Python 3.12+
- 8GB RAM
- 2GB disk space
- CPU: 4+ cores recommended

### Recommended for Training
- Python 3.12+
- 16GB+ RAM  
- NVIDIA GPU with 8GB+ VRAM
- CUDA 12.8+
- SSD storage

## 🐛 Issues & Support

For bug reports, feature requests, or questions about the public release:

1. Check existing issues for similar problems
2. Provide minimal reproduction steps
3. Include environment details (OS, Python version, GPU)
4. Note: Issues related to proprietary components cannot be supported

## 📈 Performance Notes

### Training Performance
- **SimplifiedMultiAgentEnv**: ~1000 steps/sec (CPU), ~5000 steps/sec (GPU)
- **BVRMultiAgentEnv**: ~200 steps/sec (CPU), ~800 steps/sec (GPU)  
- **Typical training**: 10M-100M steps depending on scenario complexity

### Visualization Performance  
- Real-time visualization supports up to 8v8 scenarios
- Tacview export handles unlimited scenario sizes
- GUI responsive for training runs up to 48 hours

## 🎛️ Console Commands

All commands available after `pip install -e .`:

```bash
# Training
air2air-train              # Full environment training
air2air-train-simple       # Simplified environment training

# Adaptive Training (Recommended)
air2air-train --adaptive           # Auto-detect system and adapt config
air2air-train-simple --adaptive    # Simplified environment with adaptation  
air2air-train --adaptive --no-fallback  # No automatic fallback on failure

# Visualization  
air2air-view               # Live scenario visualization
air2air-view-behavior-tree # Behavior tree visualization
air2air-view-commands      # Command visualization

# Analysis
air2air-export-plots       # Generate analysis plots

# Tacview
air2air-tacview           # Generate Tacview files
air2air-tacview-bt        # Behavior tree Tacview export

# GUI
air2air-gui              # Launch control panel

# Setup utilities  
air2air-setup-symbols    # Generate visualization symbols from SVG assets

# Code quality
air2air-lint             # Run ruff linting and formatting
```

### Code Quality

The package includes ruff for code formatting and linting:

```bash
# Check code quality
air2air-lint check

# Format code
air2air-lint format  

# Auto-fix issues
air2air-lint fix

# Show formatting diff without applying
air2air-lint format --diff
```

## 🛠️ Troubleshooting

### Visualization Shows Basic Shapes Instead of Aircraft Symbols

If the GUI visualization displays simple shapes instead of detailed aircraft symbols:

```bash
# Navigate to symbols directory and generate PNG assets
cd src/air_to_air_rl/visualization/symbols
npm install milsymbol canvas
node generate_symbols.js
```

This converts the SVG symbol files to PNG format required by the visualization system.

### Common Installation Issues

**Node.js not found**: Install Node.js from [nodejs.org](https://nodejs.org/)

**Canvas compilation errors**: On Windows, you may need Visual Studio Build Tools:
```bash
npm install --global windows-build-tools
```

**CUDA issues**: Install CUDA Toolkit 12.8+ or use CPU-only installation without `[gpu]` extra.

### Training Issues

**Out of memory errors**: Use adaptive training to automatically scale down configuration:
```bash
air2air-train --adaptive --config your_config.yaml
```

**Training fails to start**: The adaptive system will automatically try fallback configurations:
```bash
# System will automatically detect resources and adjust
air2air-train-simple --adaptive
```

**Performance tuning**: Check system assessment before training:
```bash
python -c "from air_to_air_rl.rl.training.adaptive_config import SystemResourceChecker; 
checker = SystemResourceChecker(); 
info = checker.get_system_info(); 
tier, rec = checker.assess_training_capability(); 
print(f'Detected tier: {tier}'); 
print(f'Recommendation: {rec[\"description\"]}')"
```

## 🔒 License

MIT License - See [LICENSE](LICENSE) for details.

This public release is provided for research and educational purposes. Commercial use should verify compatibility with domain-specific regulations.

---

**🎯 Ready to start? Try: `pip install -e . && air2air-gui`**