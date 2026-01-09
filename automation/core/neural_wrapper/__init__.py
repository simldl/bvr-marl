"""Neural network wrapper for semi-automated control."""
from .wrapper import NeuralWrapper
from .scripted_controller import FullyScriptedController

__all__ = ['NeuralWrapper', 'FullyScriptedController']
