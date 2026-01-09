"""
Backward compatibility layer for neural_wrapper module.
Imports from the new modular structure.
"""
# Import from new modular structure
from .neural_wrapper import (
    NeuralWrapper,
    FullyScriptedController,
)

# Export all classes for backward compatibility
__all__ = [
    'NeuralWrapper',
    'FullyScriptedController',
]
