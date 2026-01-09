# Technology Stack

## Core Technologies
- **Python**: 3.12.9
- **Package Manager**: pip (in conda environment)
- **Environment Manager**: Conda (environment name: `rlenv`)

## Key Dependencies

### Machine Learning & RL
- **PyTorch**: 2.7.0+cu128 (CUDA 12.8 support)
- **Ray**: 2.44.1 (distributed RL training framework)
- **Gymnasium**: 1.1.1 (RL environment interface)
- **TensorBoard**: 2.19.0 (training visualization)

### Scientific Computing
- **NumPy**: 2.1.2
- **SciPy**: 1.15.2
- **Pandas**: 2.2.3
- **SymPy**: 1.13.3

### Visualization & Plotting
- **Matplotlib**: 3.10.1
- **Cartopy**: 0.24.1 (geospatial plotting)
- **Pillow**: 11.0.0 (image processing)

### Geographic & Physics
- **GeographicLib**: 2.0 (geodesic calculations)
- **PyProj**: 3.7.1 (coordinate transformations)
- **Shapely**: 2.1.0 (geometric operations)

### Configuration & Utilities
- **Hydra-core**: 1.3.2 (configuration management)
- **OmegaConf**: 2.3.0 (YAML configuration)
- **PyYAML**: 6.0.2
- **python-dotenv**: 1.1.0

### Testing
- **pytest**: 8.3.5

### Web & API (for dashboards/monitoring)
- **FastAPI**: 0.115.12
- **Uvicorn**: 0.34.2
- **aiohttp**: 3.11.18

### Other Notable Libraries
- **cloudpickle**: 3.1.1 (serialization)
- **msgpack**: 1.1.0 (efficient serialization)
- **protobuf**: 6.30.2 (protocol buffers)
- **grpcio**: 1.71.0 (RPC framework)
- **lz4**: 4.4.4 (compression)

## Hardware Requirements
- **GPU**: CUDA-capable GPU (CUDA 12.8)
- **OS**: Windows (project developed on Windows)
- **RAM**: Sufficient for multi-agent RL training (likely 16GB+)

## Framework Choices
- **RL Framework**: Ray RLlib with PyTorch backend
- **Testing Framework**: pytest
- **Configuration**: Hydra for hierarchical configuration
- **Logging**: TensorBoard for training metrics
