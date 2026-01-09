# Environment Setup Guide

This guide explains how to set up the Python environment for the BVR Air-to-Air RL project.

## Prerequisites

- Python 3.12+
- CUDA 12.8+ (for GPU support)
- Conda or Miniconda

## Option 1: Using Conda (Recommended)

### Create Environment from environment.yml

```bash
# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate rlenv

# Install PyTorch with CUDA support separately
# Windows:
install_pytorch_cuda.bat

# Linux/Mac:
chmod +x install_pytorch_cuda.sh
./install_pytorch_cuda.sh
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import ray; print(f'Ray: {ray.__version__}')"
python -c "from automation.core.neural_wrapper import NeuralWrapper; print('Neural wrapper imports OK')"
```

## Option 2: Using pip with requirements.txt

### Create Virtual Environment

```bash
# Create virtual environment
python -m venv rlenv

# Activate
# Windows:
rlenv\Scripts\activate
# Linux/Mac:
source rlenv/bin/activate

# Install requirements
pip install -r requirements.txt

# Install PyTorch with CUDA support separately
# Windows:
install_pytorch_cuda.bat
# Linux/Mac:
./install_pytorch_cuda.sh
```

## Important Notes

### PyTorch with CUDA

PyTorch with CUDA support must be installed separately because:
- The CUDA-enabled versions are hosted on PyTorch's own index, not PyPI
- Different CUDA versions require different builds
- The project uses CUDA 12.8, which requires the cu128 build

**Always install PyTorch after creating the environment:**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Known Package Issues Fixed

1. **dm-tree**: Previously might have been listed as just "tree" - now correctly specified as `dm-tree==0.1.9`
2. **PyTorch CUDA**: Now has separate installation script to ensure correct CUDA version
3. **Ray dependencies**: All Ray-related packages are properly included
4. **Geospatial libraries**: Cartopy and dependencies properly configured

## Package Categories

### Core ML/RL
- `ray[default]==2.50.1` - Distributed RL framework
- `gymnasium==1.2.1` - OpenAI Gym successor
- `torch` + `torchvision` - Deep learning framework (CUDA version)

### Configuration
- `hydra-core==1.3.2` - Configuration management
- `omegaconf==2.3.0` - Configuration system
- `pyyaml==6.0.3` - YAML parsing

### Data & Visualization
- `numpy`, `pandas`, `scipy` - Numerical computing
- `matplotlib==3.10.7` - Plotting
- `cartopy==0.25.0` - Geographic plotting
- `tensorboard==2.20.0` - Training visualization

### Automation System
- All packages needed for the neural wrapper and automation levels
- Web/API frameworks for monitoring (FastAPI, uvicorn)
- Telemetry and metrics (OpenTelemetry, Prometheus)

## Troubleshooting

### CUDA Not Available

If `torch.cuda.is_available()` returns False:

1. Verify NVIDIA drivers are installed:
   ```bash
   nvidia-smi
   ```

2. Reinstall PyTorch with correct CUDA version:
   ```bash
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   ```

### Import Errors

If you get import errors:

1. Verify environment is activated:
   ```bash
   conda activate rlenv  # or source rlenv/bin/activate
   ```

2. Check package installation:
   ```bash
   pip list | grep <package-name>
   ```

3. Reinstall specific package:
   ```bash
   pip install --force-reinstall <package-name>==<version>
   ```


## Development Setup

For development, also install:

```bash
pip install pytest==8.4.2
pip install py-spy==0.4.1  # Performance profiling
```