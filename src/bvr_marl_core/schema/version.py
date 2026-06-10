SCHEMA_VERSION: str = "2.0"
SUPPORTED_VERSIONS: list[str] = ["1.0", "2.0"]

# Human-readable description of what each version introduced.
VERSION_CHANGELOG: dict[str, str] = {
    "1.0": "Initial versioned schema: scenarios, agents, teams, weapons, sensors, simulation config.",
    "2.0": (
        "Renamed ScenarioConfig.duration_s -> max_duration_s (semantic clarity). "
        "Renamed SimulationConfig.gun_hit_radius_m -> gun_lethal_radius_m (terminology). "
        "Schema version is now only carried by top-level document models."
    ),
}
