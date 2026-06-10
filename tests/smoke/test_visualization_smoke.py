"""
Smoke tests for the visualization pipeline.

Verifies that:
- ScenarioPlotter can be instantiated without a display
- to_rgba([]) returns a valid RGBA numpy array (headless, no disk I/O)
- The visualization config loader works
"""

from copy import deepcopy

import numpy as np
import pytest

pytestmark = pytest.mark.smoke


def test_scenario_plotter_imports():
    """ScenarioPlotter and its dependencies can be imported."""
    from bvr_marl_core.simulator.utils.map_limits import MapLimits
    from bvr_marl_core.visualization.scenplotter.scenario_plotter import (
        PlotConfig,
        ScenarioPlotter,
    )


def test_scenario_plotter_instantiation():
    """ScenarioPlotter can be instantiated with a basic MapLimits."""
    from bvr_marl_core.simulator.utils.map_limits import MapLimits
    from bvr_marl_core.visualization.scenplotter.scenario_plotter import (
        PlotConfig,
        ScenarioPlotter,
    )

    map_extents = MapLimits(
        left_lon=-1.0,
        bottom_lat=-1.0,
        right_lon=1.0,
        top_lat=1.0,
        min_alt=0,
        max_alt=10000,
    )
    config = PlotConfig()
    config.symbol_mode = "procedural"  # no PNG files needed
    plotter = ScenarioPlotter(map_extents, dpi=72, config=config)
    assert plotter is not None


def test_scenario_plotter_to_rgba_headless():
    """to_rgba([]) renders an empty frame headlessly and returns an RGBA array."""
    from bvr_marl_core.simulator.utils.map_limits import MapLimits
    from bvr_marl_core.visualization.scenplotter.scenario_plotter import (
        PlotConfig,
        ScenarioPlotter,
    )

    map_extents = MapLimits(
        left_lon=-1.0,
        bottom_lat=-1.0,
        right_lon=1.0,
        top_lat=1.0,
        min_alt=0,
        max_alt=10000,
    )
    config = PlotConfig()
    config.symbol_mode = "procedural"
    plotter = ScenarioPlotter(map_extents, dpi=72, config=config)

    frame = plotter.to_rgba([])

    assert isinstance(frame, np.ndarray), "to_rgba must return a numpy array"
    assert frame.ndim == 3, "Frame must be 3-dimensional (H, W, C)"
    assert frame.shape[2] == 4, "Frame must have 4 channels (RGBA)"
    assert frame.dtype == np.uint8, "Frame must be uint8"
    assert frame.shape[0] > 0 and frame.shape[1] > 0, "Frame must have positive dimensions"


def test_scenario_plotter_can_hide_text_drawables_headless():
    """Text-only drawables become no-ops when PlotConfig.show_text is false."""
    from bvr_marl_core.simulator.utils.map_limits import MapLimits
    from bvr_marl_core.visualization.scenplotter.scenario_plotter import (
        PlotConfig,
        ScenarioPlotter,
        StatusMessage,
        TopLeftMessage,
    )

    map_extents = MapLimits(
        left_lon=-1.0,
        bottom_lat=-1.0,
        right_lon=1.0,
        top_lat=1.0,
        min_alt=0,
        max_alt=10000,
    )
    config = PlotConfig()
    config.symbol_mode = "procedural"
    config.show_text = False
    plotter = ScenarioPlotter(map_extents, dpi=72, config=config)

    empty_frame = plotter.to_rgba([])
    text_frame = plotter.to_rgba(
        [StatusMessage("Aircraft destroyed."), TopLeftMessage("Duration: 1.0 s")]
    )

    assert np.array_equal(empty_frame, text_frame)


def test_viz_config_loads():
    """Visualization config loader returns a dict without raising."""
    from bvr_marl_core.utils.config_loader import load_viz_config

    cfg = load_viz_config()
    assert isinstance(cfg, dict), "load_viz_config() must return a dict"


def test_default_viz_config_uses_public_basic_training_config():
    """Default visualization should use a core-owned public training config."""
    from bvr_marl_core.utils.config_loader import load_train_config, load_viz_config

    viz_cfg = load_viz_config()
    assert viz_cfg["checkpoint_path"] is None
    assert viz_cfg["visualization"]["show_text"] is True
    assert viz_cfg["train_config_path"] == "configs/training/basic.yaml"

    train_cfg = load_train_config(viz_cfg["train_config_path"])
    assert isinstance(train_cfg["env"], dict)
    assert train_cfg["env"]["num_agents_per_side"] > 0


def test_default_viz_config_steps_with_random_actions():
    """The default visualization env should reset and step random actions."""
    from bvr_marl_core.rl.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv
    from bvr_marl_core.simulator import Simulator
    from bvr_marl_core.utils.config_loader import load_train_config, load_viz_config
    from bvr_marl_core.visualization.model_wrapper.model_wrapper import DefaultModel

    viz_cfg = load_viz_config()
    train_cfg = load_train_config(viz_cfg["train_config_path"])
    env_config = deepcopy(train_cfg["env"])
    env_config["max_steps"] = 2
    env_config["simulator"] = Simulator()
    env = BVRMultiAgentEnv(env_config)

    try:
        observations, _ = env.reset()
        model = DefaultModel(env)
        actions = {
            agent_id: model.compute_single_action(observations[agent_id], agent_id)
            for agent_id in env.agents
            if agent_id in observations
        }
        _, rewards, terminateds, truncateds, _ = env.step(actions)

        assert isinstance(rewards, dict)
        assert "__all__" in terminateds
        assert "__all__" in truncateds
    finally:
        if hasattr(env, "close"):
            env.close()
