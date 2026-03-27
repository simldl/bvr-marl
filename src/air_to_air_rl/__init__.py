"""
BVR-MARL — Beyond Visual Range Multi-Agent Reinforcement Learning.

A high-fidelity air-to-air combat simulation environment with multi-agent
reinforcement learning capabilities.

Public API surfaces
-------------------
Entrypoints (installed by ``pip install -e .``):
    air2air-train, air2air-train-simple   — start training
    air2air-view, air2air-view-*          — 2D live visualization
    air2air-tacview, air2air-tacview-bt   — generate Tacview ACMI files
    air2air-gui                           — Streamlit control panel
    air2air-export-plots                  — export TensorBoard plots

Stable imports for external use:
    air_to_air_rl.rl.environment.gym.bvr_multi_agent_env.BVRMultiAgentEnv
    air_to_air_rl.rl.environment.gym.simplified_multi_agent_env.SimplifiedMultiAgentEnv
    air_to_air_rl.core.paths              — canonical path helpers
    air_to_air_rl.services.*             — shared orchestration logic (CLI and GUI)

Internal implementation (not stable API):
    air_to_air_rl.aircrafts.*
    air_to_air_rl.missiles.*
    air_to_air_rl.radar.*
    air_to_air_rl.physics.*
    air_to_air_rl.simulator.*
    air_to_air_rl.gui.*                   — Streamlit app internals
"""

__version__ = "0.1.0"
