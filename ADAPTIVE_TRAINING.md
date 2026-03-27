# Adaptive Training Configuration System

The adaptive training system automatically detects your system's hardware capabilities and adjusts training parameters accordingly, ensuring optimal performance and preventing common issues like out-of-memory errors.

## 🎯 Quick Start

```bash
# Enable adaptive training for any configuration
air2air-train --adaptive --config your_config.yaml

# For simplified environment
air2air-train-simple --adaptive --config your_config.yaml

# Check what tier your system would be assigned
python -c "
from air_to_air_rl.rl.training.adaptive_config import SystemResourceChecker
checker = SystemResourceChecker()
tier, rec = checker.assess_training_capability()
print(f'Recommended tier: {tier}')
print(f'Description: {rec[\"description\"]}')
"
```

## 🔍 System Assessment

The adaptive system evaluates your hardware and assigns one of four configuration tiers:

### Tier Requirements
- **minimal**: Basic systems (8GB+ RAM, 4+ CPU threads)
- **low**: Light training systems (8GB+ RAM, 4+ CPU threads) 
- **medium**: Standard systems (16GB+ RAM, 8+ CPU threads, optional GPU)
- **high**: High-performance systems (32GB+ RAM, 16+ CPU threads, 8GB+ GPU)

### System Detection
The system automatically detects:
- CPU core count and thread count
- Total and available RAM
- GPU availability and VRAM
- Platform-specific optimizations

## ⚙️ Configuration Tiers

### Minimal Tier
- **Environment**: SimplifiedMultiAgentEnv
- **Workers**: 1 environment runner
- **Batch Size**: 2,000 samples
- **Minibatch Size**: 128
- **SGD Iterations**: 10
- **Use Case**: Learning the system, basic laptops

### Low Tier  
- **Environment**: SimplifiedMultiAgentEnv
- **Workers**: 2 environment runners
- **Batch Size**: 4,000 samples
- **Minibatch Size**: 256
- **SGD Iterations**: 15
- **Use Case**: Development machines, light training

### Medium Tier
- **Environment**: BVRMultiAgentEnv (full)
- **Workers**: 4 environment runners
- **Batch Size**: 8,000 samples
- **Minibatch Size**: 512
- **SGD Iterations**: 20
- **Use Case**: Standard workstations, research

### High Tier
- **Environment**: BVRMultiAgentEnv (full)
- **Workers**: 8 environment runners  
- **Batch Size**: 16,000 samples
- **Minibatch Size**: 1,024
- **SGD Iterations**: 30
- **Use Case**: High-end workstations, production training

## 🔄 Automatic Fallback

If training fails (e.g., out of memory), the system automatically tries progressively lower tiers:

```bash
# Training with automatic fallback (default)
air2air-train --adaptive --config your_config.yaml

# Disable fallback to fail immediately 
air2air-train --adaptive --no-fallback --config your_config.yaml
```

### Fallback Sequence
1. **Initial tier**: Based on system assessment
2. **First fallback**: Next lower tier (high → medium → low → minimal)
3. **Second fallback**: Two tiers lower
4. **Final fallback**: Minimal tier with absolute minimum resources

## 📁 Configuration Files

The adaptive system saves configuration files for reference:

```
models/your_model_name/
├── adaptive_config.yaml          # Initial adaptive configuration
├── fallback_config_attempt_1.yaml # First fallback attempt
├── fallback_config_attempt_2.yaml # Second fallback attempt  
└── train_config.yaml             # Original user configuration
```

## 🎛️ Advanced Usage

### Manual Tier Selection
```python
from air_to_air_rl.rl.training.adaptive_config import AdaptiveTrainingConfig

trainer = AdaptiveTrainingConfig("your_config.yaml")
config = trainer.get_adaptive_config(target_tier="medium")
```

### System Information
```python
from air_to_air_rl.rl.training.adaptive_config import SystemResourceChecker

checker = SystemResourceChecker()
info = checker.get_system_info()
print(f"CPU: {info['cpu']['threads']} threads")
print(f"Memory: {info['memory']['total_gb']} GB")
print(f"GPU: {info['gpu']['available']}")
```

### Interactive Assessment
```python
from air_to_air_rl.rl.training.adaptive_config import create_adaptive_trainer

# Interactive system assessment
trainer = create_adaptive_trainer("config.yaml", auto_detect=True)
trainer.print_system_summary()
```

## 🔧 Customization

### Custom Tier Definitions
You can extend the adaptive system with custom tiers by subclassing `AdaptiveTrainingConfig`:

```python
class CustomAdaptiveTrainingConfig(AdaptiveTrainingConfig):
    def get_custom_tier_config(self):
        return {
            "description": "Custom configuration",
            "num_env_runners": 6,
            "train_batch_size": 12000,
            # ... other parameters
        }
```

### Environment-Specific Adaptations
The system automatically adjusts:
- **Environment class**: SimplifiedMultiAgentEnv for lower tiers
- **Map size**: Reduced for lower-tier environments  
- **Agent count**: Limited for resource-constrained systems
- **Episode length**: Shortened for faster iterations

## 📊 Performance Guidelines

### Expected Training Performance
- **Minimal**: ~500-1000 steps/sec (CPU-only)
- **Low**: ~1000-2000 steps/sec (CPU-only)  
- **Medium**: ~2000-4000 steps/sec (CPU+GPU)
- **High**: ~4000-8000 steps/sec (High-end GPU)

### Memory Usage
- **Minimal**: ~2-4 GB RAM
- **Low**: ~4-8 GB RAM
- **Medium**: ~8-16 GB RAM  
- **High**: ~16-32 GB RAM + 4-8 GB VRAM

## 🐛 Troubleshooting

### Common Issues

**"No more fallback configurations available"**: Your system may be below minimum requirements. Try:
- Close other applications to free memory
- Use `air2air-train-simple` for lighter training
- Check system requirements

**Training still fails on minimal tier**: 
- Verify 8GB+ RAM availability
- Ensure no memory leaks from previous runs
- Check disk space for temporary files

**Adaptive detection incorrect**:
```python
# Override system detection
trainer = AdaptiveTrainingConfig("config.yaml")
config = trainer.get_adaptive_config(target_tier="low")  # Force specific tier
```

### Debugging System Detection
```bash
# Detailed system information
python -c "
from air_to_air_rl.rl.training.adaptive_config import SystemResourceChecker
import json
checker = SystemResourceChecker()
info = checker.get_system_info()
print(json.dumps(info, indent=2))
"
```

## 🎯 Best Practices

1. **Always start with adaptive training** for new systems
2. **Monitor memory usage** during initial runs
3. **Save successful configurations** for future use
4. **Use simplified environment** for development/testing
5. **Graduate to full environment** once training is stable

## 🔗 Integration

The adaptive system integrates seamlessly with:
- **GUI training**: Automatic detection in Streamlit interface
- **Checkpoint resumption**: Preserves adaptive settings
- **Distributed training**: Scales across multiple GPUs
- **Framework integration**: Works with Ray RLlib default networks

---

For more information, see the main [README](README_PUBLIC.md) or check the source code in `src/air_to_air_rl/rl/training/adaptive_config.py`.