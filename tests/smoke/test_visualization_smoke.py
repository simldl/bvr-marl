"""
Smoke tests for the visualization pipeline.

Verifies that:
- ScenarioPlotter can be instantiated without a display
- to_rgba([]) returns a valid RGBA numpy array (headless, no disk I/O)
- The visualization config loader works
"""

import numpy as np
import pytest

pytestmark = pytest.mark.smoke


def test_scenario_plotter_imports():
    """ScenarioPlotter and its dependencies can be imported."""
    from air_to_air_rl.simulator.utils.map_limits import MapLimits
    from air_to_air_rl.visualization.scenplotter.scenario_plotter import (
        PlotConfig,
        ScenarioPlotter,
    )


def test_scenario_plotter_instantiation():
    """ScenarioPlotter can be instantiated with a basic MapLimits."""
    from air_to_air_rl.simulator.utils.map_limits import MapLimits
    from air_to_air_rl.visualization.scenplotter.scenario_plotter import (
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
    from air_to_air_rl.simulator.utils.map_limits import MapLimits
    from air_to_air_rl.visualization.scenplotter.scenario_plotter import (
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


def test_viz_config_loads():
    """Visualization config loader returns a dict without raising."""
    from air_to_air_rl.utils.config_loader import load_viz_config

    cfg = load_viz_config()
    assert isinstance(cfg, dict), "load_viz_config() must return a dict"
