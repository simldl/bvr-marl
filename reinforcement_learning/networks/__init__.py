"""Neural network architectures for reinforcement learning.

Main components:
- rl_module: Custom RLModule for multi-agent PPO training
- components: Reusable network components (attention, transformers, encoders, etc.)
- encoders: Complete encoder architectures
"""

# Import main RLModule for backward compatibility
from .rl_module import CustomMultiAgentRLModule

__all__ = ['CustomMultiAgentRLModule']
