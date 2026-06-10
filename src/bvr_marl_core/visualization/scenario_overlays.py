"""Scenario overlay helpers shared by visualization UIs and renderers."""

from __future__ import annotations

from dataclasses import dataclass

_DEG_PER_KM = 1.0 / 111.0
_MAP_EXTENTS_MODES = {"auto", "combat", "full"}


@dataclass(frozen=True)
class VisualizationScenario:
    """Metadata for a lightweight visualization scenario overlay."""

    key: str
    label: str
    description: str
    line_of_engagement_north_m: float | None = None
    line_of_engagement_east_m: float | None = None
    use_combat_map_limits: bool = False
    line_color: tuple[float, float, float, float] = (0.96, 0.78, 0.14, 0.95)
    dash: tuple[float, float] | None = (18.0, 10.0)

    @property
    def has_line_of_engagement(self) -> bool:
        """Return True when the overlay should draw a line of engagement."""
        return (
            self.line_of_engagement_north_m is not None
            or self.line_of_engagement_east_m is not None
        )

    def line_of_engagement_text(self) -> str | None:
        """Return a short label for the configured line of engagement."""
        if self.line_of_engagement_east_m is not None:
            east_km = self.line_of_engagement_east_m / 1000.0
            return f"LOE: E {east_km:+.0f} km"
        if self.line_of_engagement_north_m is not None:
            north_km = self.line_of_engagement_north_m / 1000.0
            return f"LOE: N {north_km:+.0f} km"
        return None


_SCENARIOS: tuple[VisualizationScenario, ...] = (
    VisualizationScenario(
        key="default",
        label="Default (No Scenario Overlay)",
        description="Use the visualization without extra scenario annotations.",
    ),
    VisualizationScenario(
        key="oca_v_dca_scenario",
        label="OCA v DCA Scenario (Line of Engagement)",
        description=(
            "Show the 400 x 200 km combat zone, draw the north-south line of "
            "engagement through the center of the map, and print its status "
            "in the map overlay."
        ),
        line_of_engagement_east_m=0.0,
        use_combat_map_limits=True,
    ),
)

_SCENARIOS_BY_KEY = {scenario.key: scenario for scenario in _SCENARIOS}
_SCENARIOS_BY_LABEL = {scenario.label: scenario for scenario in _SCENARIOS}


def list_visualization_scenarios() -> list[VisualizationScenario]:
    """Return all known visualization scenarios in display order."""
    return list(_SCENARIOS)


def normalize_map_extents_mode(value: str | None) -> str:
    """Resolve a display-bounds mode to ``auto``, ``combat``, or ``full``."""
    if not value:
        return "auto"
    mode = str(value).strip().lower()
    if mode in _MAP_EXTENTS_MODES:
        return mode
    return "auto"


def normalize_visualization_scenario(value: str | None) -> str:
    """Resolve a scenario key or label to a canonical key."""
    if not value:
        return "default"
    if value in _SCENARIOS_BY_KEY:
        return value
    scenario = _SCENARIOS_BY_LABEL.get(value)
    if scenario is not None:
        return scenario.key
    return "default"


def get_visualization_scenario(value: str | None) -> VisualizationScenario:
    """Return a scenario definition for a key or display label."""
    return _SCENARIOS_BY_KEY[normalize_visualization_scenario(value)]


def _resolve_effective_line_of_engagement(
    scenario: VisualizationScenario,
    *,
    force_line_overlay: bool = False,
    line_of_engagement_east_m: float | None = None,
) -> tuple[str | None, float | None]:
    """Return the effective LOE axis/value after manual overrides are applied."""
    if line_of_engagement_east_m is not None and (
        force_line_overlay or scenario.has_line_of_engagement
    ):
        return ("east", float(line_of_engagement_east_m))
    if scenario.line_of_engagement_east_m is not None:
        return ("east", float(scenario.line_of_engagement_east_m))
    if scenario.line_of_engagement_north_m is not None:
        return ("north", float(scenario.line_of_engagement_north_m))
    if force_line_overlay:
        return (
            "east",
            0.0 if line_of_engagement_east_m is None else float(line_of_engagement_east_m),
        )
    return (None, None)


def build_scenario_overlay_drawables(
    map_limits,
    scenario_value: str | None,
    *,
    force_line_overlay: bool = False,
    line_of_engagement_east_m: float | None = None,
):
    """Build renderer drawables for the selected scenario overlay."""
    scenario = get_visualization_scenario(scenario_value)
    axis, value_m = _resolve_effective_line_of_engagement(
        scenario,
        force_line_overlay=force_line_overlay,
        line_of_engagement_east_m=line_of_engagement_east_m,
    )
    if axis is None or value_m is None:
        return []

    from bvr_marl_core.visualization.scenplotter import PolyLine

    if axis == "east":
        east_km = value_m / 1000.0
        line_lon = east_km * _DEG_PER_KM
        return [
            PolyLine(
                points=[
                    (map_limits.bottom_lat, line_lon),
                    (map_limits.top_lat, line_lon),
                ],
                line_width=3.0,
                dash=scenario.dash,
                edge_color=scenario.line_color,
                zorder=35,
            )
        ]

    north_km = value_m / 1000.0
    line_lat = north_km * _DEG_PER_KM
    return [
        PolyLine(
            points=[
                (line_lat, map_limits.left_lon),
                (line_lat, map_limits.right_lon),
            ],
            line_width=3.0,
            dash=scenario.dash,
            edge_color=scenario.line_color,
            zorder=35,
        )
    ]


def build_scenario_status_lines(
    scenario_value: str | None,
    *,
    force_line_overlay: bool = False,
    line_of_engagement_east_m: float | None = None,
) -> list[str]:
    """Build short status lines describing the active scenario overlay."""
    scenario = get_visualization_scenario(scenario_value)
    axis, value_m = _resolve_effective_line_of_engagement(
        scenario,
        force_line_overlay=force_line_overlay,
        line_of_engagement_east_m=line_of_engagement_east_m,
    )

    status_lines: list[str] = []
    if scenario.key != "default":
        status_lines.append(f"Scenario: {scenario.label}")
    if axis == "east" and value_m is not None:
        status_lines.append(f"LOE: E {value_m / 1000.0:+.0f} km")
    elif axis == "north" and value_m is not None:
        status_lines.append(f"LOE: N {value_m / 1000.0:+.0f} km")
    return status_lines


def select_display_limits(
    env,
    scenario_value: str | None,
    *,
    map_extents_mode: str | None = None,
):
    """Select the display bounds for the active visualization scenario."""
    mode = normalize_map_extents_mode(map_extents_mode)
    if mode == "combat":
        return getattr(env, "map_limits", getattr(env, "full_map_limits", None))
    if mode == "full":
        return getattr(env, "full_map_limits", getattr(env, "map_limits", None))

    scenario = get_visualization_scenario(scenario_value)
    if scenario.use_combat_map_limits:
        return getattr(env, "map_limits", getattr(env, "full_map_limits", None))
    return getattr(env, "full_map_limits", getattr(env, "map_limits", None))
