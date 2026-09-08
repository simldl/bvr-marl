"""
Configuration parser and validator for BVR Multi-Agent Environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from bvr_marl_core.rl.environment.spaces.observation.constants import d_OWN as _D_OWN
from bvr_marl_core.rl.environment.spaces.observation.constants import (
    own_state_dim as _own_state_dim,
)
from bvr_marl_core.simulator import MapLimits


@dataclass
class AWACSConfigData:
    """Configuration for AWACS aircraft in scenarios."""

    # Whether to spawn AWACS for each team
    agent_awacs: bool = False
    opponent_awacs: bool = False

    # Number of AWACS platforms spawned per team when enabled. The workshop moved
    # away from a single very-potent AWACS towards two less-potent 360-degree
    # systems, so the default is 2. They are placed with north/south spatial
    # separation (see awacs_pair_spacing_km) so the flank is covered from two
    # aspects and losing one does not blind the team.
    count_per_team: int = 2
    # North/south separation between paired AWACS orbit centres (km). Only used
    # when count_per_team > 1.
    awacs_pair_spacing_km: float = 120.0

    # Optional radar overrides applied on top of the (now less-potent) AWACS.Config
    # radar defaults. None -> use the type default. Exposed so scenarios can retune
    # the surveillance picture without editing the aircraft type.
    radar_max_range_m: float | None = None
    radar_tx_power_w: float | None = None
    radar_antenna_gain_db: float | None = None
    radar_snr_threshold_db: float | None = None

    # AWACS positioning (kept for backward compatibility)
    orbit_distance_km: float = 80.0
    orbit_altitude_m: float = 10000.0
    orbit_speed_mps: float = 200.0

    # Orbit pattern configuration
    orbit_pattern: str = "figure8"  # "racetrack", "figure8", "circle", "random", "random_all"
    orbit_leg_length_km: float = 40.0  # racetrack straight-leg length
    orbit_radius_km: float = 25.0  # figure-8 circle radius (also sets side-zone width)
    orbit_clockwise: bool = True

    # Fighter trailing is available for custom experiments, but the default is a
    # fixed side-zone orbit so AWACS spawn in the same places every episode and
    # keep their patrol pattern independent of fighter spawn geometry.
    trail_fighters: bool = False
    trail_standoff_km: float = 40.0

    # Lock FOV constraint (detection is still 360°, but locking requires facing target)
    lock_fov_deg: float = 60.0

    # When True, AWACS cannot be targeted by fighters and is shown with a different obs flag
    awacs_non_engageable: bool = True

    # When True, AWACS is dropped from every HOSTILE sensing path at enumeration --
    # before RCS, SNR, line-of-sight, association or track maintenance run for it. This
    # is a different rule from ``awacs_non_engageable``: that one refuses the LAUNCH,
    # and a refused launch still costs everything upstream of it -- the unit is swept,
    # it becomes a detection, a track, a contact, and it OCCUPIES A TARGET SLOT.
    #
    # Kept as an option rather than hardcoded because a scenario may legitimately want
    # a visible-but-protected support asset (or a visible one, to study the cost).
    # Default True: measured over 10 matched pairs, blinding the AWACS raised real shot
    # opportunities 1.75x in 10 of 10 episodes, and it removes two units per team from
    # the per-candidate sensor chain every tick.
    awacs_sensor_invisible: bool = True


@dataclass
class RandomMapConfigData:
    """Tuning for the ``random_map`` spawn regime (full-map random spawns)."""

    # Fraction of each map half kept clear of spawns so aircraft do not start on
    # top of the boundary. 0.2 keeps spawns within the inner 60% of the box.
    margin_frac: float = 0.2
    # Minimum distance between the two team centroids at spawn (meters).
    min_separation_m: float = 40_000.0
    # Optional MAXIMUM distance between the two team centroids at spawn (meters).
    # None keeps the historical behaviour (floor only). Set it on stages that must
    # train weapon employment rather than transit: with a floor alone, an early stage's
    # 400 km map produced a median spawn separation of 164 km against a 185 km radar
    # range, so most episodes were spent closing rather than engaging.
    max_separation_m: float | None = None


@dataclass
class ScenarioConfigData:
    """Configuration for scenario selection and variation."""

    # list of geometry regimes to randomly sample from each episode
    # If empty, uses the single regime from geometry_config
    allowed_regimes: list[str] = field(default_factory=list)

    # Whether to randomize regime selection each episode
    randomize_regime: bool = True

    # AWACS configuration
    awacs_config: AWACSConfigData = field(default_factory=AWACSConfigData)

    # Opponent scripting for curriculum warmup stages.
    #   "policy"          -> opponent acts on its own policy/controller (default)
    #   "stationary_hold" -> opponent flies straight-and-level (neutral controls),
    #                        a predictable non-maneuvering target. Spawn marks it
    #                        boundary-kept so it cannot die at the map edge.
    #   "anchored_hold"   -> opponent flies a constant gentle level turn, with a
    #                        spawn-time boundary-keep flag so it remains a
    #                        predictable circling target in early warmup stages.
    opponent_behavior: str = "policy"

    # When True, opponent missile/gun/countermeasure actions are forced to zero so
    # the opponent never fires (used for the early "learn to shoot" warmup stages).
    # Implied by ``opponent_behavior`` in ("stationary_hold", "anchored_hold").
    opponent_hold_fire: bool = False

    # When True, opponent fighters are flagged ``is_non_engageable`` at spawn, so the
    # agent cannot lock, target, or kill them (mirrors ``awacs_non_engageable``).
    # Used by the boundary-survival warmup stage so a stray shot can neither be
    # rewarded nor end the episode early. Like AWACS, non-engageable units are also
    # excluded from the enemy observation.
    opponent_non_engageable: bool = False

    # Tuning for the random_map regime.
    random_map_config: RandomMapConfigData = field(default_factory=RandomMapConfigData)


@dataclass
class GeometryConfigData:
    """Configuration for controlled scenario geometry."""

    use_controlled_geometry: bool = False
    regime: str = "hot_120km"
    formation_spread_m: float = 2000.0
    position_noise_m: float = 0.0
    altitude_noise_m: float = 0.0
    heading_noise_deg: float = 0.0


@dataclass
class MetricsConfigData:
    """Configuration for episode metrics collection."""

    enable_metrics: bool = False
    output_dir: str = "study_results"
    export_interval: int = 100


@dataclass
class BVREnvConfig:
    """Configuration for BVR Multi-Agent Environment."""

    # Agent configuration
    num_agents_per_team: int
    agent_ids: list[str] = field(default_factory=list)
    opponent_ids: list[str] = field(default_factory=list)
    all_agent_ids: list[str] = field(default_factory=list)
    possible_agents: list[str] = field(default_factory=list)
    aircraft_type_map: dict[str, str] = field(default_factory=dict)

    # Episode/time configuration
    max_steps: int = 600
    tick_secs: float = 1.0
    max_real_time_s: float = None

    # Termination configuration
    early_term_all_enemies_dead: bool = True
    early_term_mission_complete: bool = True
    early_term_no_missiles: bool = False

    # Training aids
    fix_radar_lock_after_first_obs: bool = False
    lock_fix_duration_s: float = 0.0
    force_locks_all_enemies: bool = False

    # Map configuration
    map_size_km: float = 1000.0
    map_width_km: float = 1000.0
    map_height_km: float = 1000.0
    max_alt: float = 25000.0
    map_limits: MapLimits = None  # Combat zone — agents destroyed if they leave this
    full_map_limits: MapLimits = None  # Full display area including AWACS side zones

    # Observation slots
    num_fm: int = 4
    num_ff: int = 2
    num_em: int = 4
    num_ef: int = 2
    num_pr: int = 2
    num_warn: int = 4
    # Must equal observation.constants.d_OWN (the own-state builder asserts that
    # length). Kept as a field so the space and the builder never drift apart.
    own_state_dim: int = _D_OWN

    # Tacview logging
    tacview_logfile: str = None

    # Neural wrapper config
    enable_missile_automation: bool = False
    missile_auto_sqi_threshold: float = 0.3
    missile_auto_max_per_target: int = 2
    missile_auto_long_cooldown_s: float = 10.0
    missile_fire_threshold: float = 0.5

    # Weapon system toggles
    enable_gun: bool = False

    # Datalink configuration
    datalink_mode: str = "full"
    # Per-link, per-tick probability the datalink to a friendly is down (0/1 model).
    # 0.0 = always up (default, current configs unchanged); raise to model dropouts.
    datalink_drop_prob: float = 0.0

    # Per-tick position/velocity trace recording, consumed only by the
    # visualization/tacview path. Default True (unchanged behavior); set False in
    # headless training configs to avoid the unbounded per-episode trace growth.
    record_traces: bool = True

    # Geometry configuration for controlled scenarios
    geometry_config: GeometryConfigData = field(default_factory=GeometryConfigData)

    # Scenario configuration (AWACS, regime randomization)
    scenario_config: ScenarioConfigData = field(default_factory=ScenarioConfigData)

    # Metrics configuration for study
    metrics_config: MetricsConfigData = field(default_factory=MetricsConfigData)

    # Debugging
    debug: bool = False
    information_mode: str = "sensor_limited"
    oracle_use_reason: str | None = None
    reward_information_mode: str = "observation_only"
    # Extra enemy-fighter token columns supplied by an extension package's
    # EnemyInfoBuilder subclass. 0 keeps the base 19-wide token.
    ef_extra_dim: int = 0
    # Opaque options forwarded to an extension package's builders. Core never reads
    # the contents; it only carries them to EnvConfig.
    extension_options: dict = field(default_factory=dict)
    # Active sensing (EMCON): expose the radar on/off toggle as action index 9,
    # config-driven instead of the source-level EMCON_ACTION_ENABLED constant.
    emcon_action_enabled: bool = False
    # Scripted sensing baseline. "learned" uses the policy action; the others force
    # the radar schedule for the active-sensing baselines (see EmconController).
    emcon_policy: str = "learned"
    emcon_period_steps: int = 10
    emcon_duty: float = 0.5

    @classmethod
    def from_dict(cls, config: dict) -> BVREnvConfig:
        """Create configuration from dictionary."""
        information_mode = str(config.get("information_mode", "sensor_limited")).strip().lower()
        if information_mode not in {"sensor_limited", "oracle"}:
            raise ValueError("information_mode must be 'sensor_limited' or 'oracle'.")
        ef_extra_dim = int(config.get("ef_extra_dim", 0) or 0)
        if ef_extra_dim < 0:
            raise ValueError("ef_extra_dim must be >= 0.")
        extension_options = dict(config.get("extension_options") or {})
        oracle_use_reason = config.get("oracle_use_reason")
        if information_mode == "oracle" and not str(oracle_use_reason or "").strip():
            raise ValueError("oracle_use_reason is required when information_mode='oracle'.")
        reward_information_mode = (
            str(config.get("reward_information_mode", "observation_only")).strip().lower()
        )
        if reward_information_mode not in {
            "observation_only",
            "team_shared",
            "privileged_training",
            "evaluator_terminal_only",
        }:
            raise ValueError("Invalid reward_information_mode.")
        # Accept both keys for num_agents. Opponent count may differ (asymmetric
        # scenarios); it defaults to the agent count for backward compatibility.
        n = int(config.get("num_agents_per_team", config.get("num_agents_per_side", 2)))
        n_opp = int(config.get("num_opponents", n))

        agent_ids = [f"A{i}" for i in range(n)]
        opponent_ids = [f"B{i}" for i in range(n_opp)]
        all_agent_ids = agent_ids + opponent_ids

        # Aircraft class mapping. ``aircraft_types.agent`` / ``.opponent`` may be a
        # single type string (the whole team flies it) or a list of types assigned
        # per slot and cycled (heterogeneous formation, e.g. one F-22 leading F-35s).
        grp_map = config.get("aircraft_types", {})
        # Resolve the string form (``aircraft_config.agent_type`` / ``.opponent_type``)
        # to classes when the pre-resolved ``aircraft_types`` is absent. The training
        # path resolves it in ``env_creator``, but direct constructions (live view,
        # web live view, tacview) pass only the string config -- without this they
        # silently fell back to DebugPlane, whose RCS made every fighter classify as a
        # support aircraft and vanish from the scripted BT's target picture.
        if not grp_map and config.get("aircraft_config"):
            from bvr_marl_core.rl.utils.type_maps import resolve_aircraft_config

            grp_map = resolve_aircraft_config(config).get("aircraft_types", {})

        def _type_for(spec, idx):
            if isinstance(spec, (list, tuple)):
                return spec[idx % len(spec)] if spec else None
            return spec

        aircraft_type_map = {
            aid: _type_for(grp_map.get("agent"), i) for i, aid in enumerate(agent_ids)
        }
        aircraft_type_map.update(
            {bid: _type_for(grp_map.get("opponent"), i) for i, bid in enumerate(opponent_ids)}
        )

        # Combat zone map limits (agents are destroyed if they cross these)
        map_size_km = float(config.get("map_size", 1000.0))
        map_width_km = float(config.get("map_width_km", config.get("map_width", map_size_km)))
        map_height_km = float(config.get("map_height_km", config.get("map_height", map_size_km)))
        half_width = (map_width_km / 2.0) / 111.0
        half_height = (map_height_km / 2.0) / 111.0
        max_alt_val = float(config.get("max_alt", 25000.0))
        map_limits = MapLimits(
            left_lon=-half_width,
            bottom_lat=-half_height,
            right_lon=half_width,
            top_lat=half_height,
            min_alt=0.0,
            max_alt=max_alt_val,
        )

        # Full display map limits: expand only when AWACS geometry actually
        # extends beyond the combat zone. If AWACS remain inside the combat map,
        # full/combat bounds stay identical so a 400 x 200 km scenario really
        # renders as 400 x 200 km.
        awacs_cfg_pre = config.get("scenario_config", {}).get("awacs_config", {})
        awacs_enabled = bool(
            awacs_cfg_pre.get("agent_awacs", False) or awacs_cfg_pre.get("opponent_awacs", False)
        )
        if awacs_enabled:
            awacs_radius_km = float(awacs_cfg_pre.get("orbit_radius_km", 25.0))
            awacs_distance_km = float(awacs_cfg_pre.get("orbit_distance_km", 80.0))
            awacs_required_half_width_km = max(
                map_width_km / 2.0,
                awacs_distance_km + awacs_radius_km,
            )
            awacs_required_half_height_km = max(
                map_height_km / 2.0,
                2.0 * awacs_radius_km,
            )
        else:
            awacs_required_half_width_km = map_width_km / 2.0
            awacs_required_half_height_km = map_height_km / 2.0
        half_ew = awacs_required_half_width_km / 111.0
        half_ns = awacs_required_half_height_km / 111.0
        full_map_limits = MapLimits(
            left_lon=-half_ew,
            bottom_lat=-half_ns,
            right_lon=half_ew,
            top_lat=half_ns,
            min_alt=0.0,
            max_alt=max_alt_val,
        )

        sim_config = config.get("simulation_config", {})
        early_term = sim_config.get("early_termination", {})

        neural_wrapper_config = config.get("neural_wrapper_config", {})

        weapon_config = config.get("weapon_config", {})

        geometry_cfg = config.get("geometry_config", {})
        geometry_config = GeometryConfigData(
            use_controlled_geometry=geometry_cfg.get("use_controlled_geometry", False),
            regime=geometry_cfg.get("regime", "hot_120km"),
            formation_spread_m=float(geometry_cfg.get("formation_spread_m", 2000.0)),
            position_noise_m=float(geometry_cfg.get("position_noise_m", 0.0)),
            altitude_noise_m=float(geometry_cfg.get("altitude_noise_m", 0.0)),
            heading_noise_deg=float(geometry_cfg.get("heading_noise_deg", 0.0)),
        )

        metrics_cfg = config.get("metrics_config", {})
        metrics_config = MetricsConfigData(
            enable_metrics=metrics_cfg.get("enable_metrics", False),
            output_dir=metrics_cfg.get("output_dir", "study_results"),
            export_interval=int(metrics_cfg.get("export_interval", 100)),
        )

        # Scenario config (AWACS and regime randomization)
        scenario_cfg = config.get("scenario_config", {})
        awacs_cfg = scenario_cfg.get("awacs_config", {})
        awacs_non_engageable = awacs_cfg.get("awacs_non_engageable", True)

        def _opt_float(key):
            v = awacs_cfg.get(key, None)
            return None if v is None else float(v)

        awacs_config = AWACSConfigData(
            agent_awacs=awacs_cfg.get("agent_awacs", False),
            opponent_awacs=awacs_cfg.get("opponent_awacs", False),
            count_per_team=int(awacs_cfg.get("count_per_team", 2)),
            awacs_pair_spacing_km=float(awacs_cfg.get("awacs_pair_spacing_km", 120.0)),
            radar_max_range_m=_opt_float("radar_max_range_m"),
            radar_tx_power_w=_opt_float("radar_tx_power_w"),
            radar_antenna_gain_db=_opt_float("radar_antenna_gain_db"),
            radar_snr_threshold_db=_opt_float("radar_snr_threshold_db"),
            orbit_distance_km=float(awacs_cfg.get("orbit_distance_km", 80.0)),
            orbit_altitude_m=float(awacs_cfg.get("orbit_altitude_m", 10000.0)),
            orbit_speed_mps=float(awacs_cfg.get("orbit_speed_mps", 200.0)),
            orbit_pattern=awacs_cfg.get("orbit_pattern", "figure8"),
            orbit_leg_length_km=float(awacs_cfg.get("orbit_leg_length_km", 40.0)),
            orbit_radius_km=float(awacs_cfg.get("orbit_radius_km", 25.0)),
            orbit_clockwise=awacs_cfg.get("orbit_clockwise", True),
            trail_fighters=bool(awacs_cfg.get("trail_fighters", False)),
            trail_standoff_km=float(awacs_cfg.get("trail_standoff_km", 40.0)),
            lock_fov_deg=float(awacs_cfg.get("lock_fov_deg", 60.0)),
            awacs_non_engageable=bool(awacs_non_engageable),
            awacs_sensor_invisible=bool(awacs_cfg.get("awacs_sensor_invisible", True)),
        )
        random_map_cfg = scenario_cfg.get("random_map_config", {}) or {}
        opponent_behavior = str(scenario_cfg.get("opponent_behavior", "policy") or "policy")
        scenario_config = ScenarioConfigData(
            allowed_regimes=scenario_cfg.get("allowed_regimes", []),
            randomize_regime=scenario_cfg.get("randomize_regime", True),
            awacs_config=awacs_config,
            opponent_behavior=opponent_behavior,
            # Scripted (stationary/anchored) opponents never fire, regardless of
            # the explicit flag.
            opponent_hold_fire=bool(scenario_cfg.get("opponent_hold_fire", False))
            or opponent_behavior in ("stationary_hold", "anchored_hold"),
            opponent_non_engageable=bool(scenario_cfg.get("opponent_non_engageable", False)),
            random_map_config=RandomMapConfigData(
                margin_frac=float(random_map_cfg.get("margin_frac", 0.2)),
                min_separation_m=float(random_map_cfg.get("min_separation_m", 40_000.0)),
                max_separation_m=(
                    float(raw_max)
                    if (raw_max := random_map_cfg.get("max_separation_m")) is not None
                    else None
                ),
            ),
        )

        return cls(
            num_agents_per_team=n,
            agent_ids=agent_ids,
            opponent_ids=opponent_ids,
            all_agent_ids=all_agent_ids,
            possible_agents=list(all_agent_ids),
            aircraft_type_map=aircraft_type_map,
            max_steps=int(config.get("max_steps", 600)),
            tick_secs=float(sim_config.get("tick_secs", 1.0)),
            max_real_time_s=sim_config.get("max_real_time_s", None),
            early_term_all_enemies_dead=early_term.get("all_enemies_dead", True),
            early_term_mission_complete=early_term.get("mission_complete", True),
            early_term_no_missiles=early_term.get("no_missiles_remaining", False),
            fix_radar_lock_after_first_obs=sim_config.get("fix_radar_lock_after_first_obs", False),
            lock_fix_duration_s=float(sim_config.get("lock_fix_duration_s", 0.0)),
            force_locks_all_enemies=sim_config.get("force_locks_all_enemies", False),
            map_size_km=map_width_km,
            map_width_km=map_width_km,
            map_height_km=map_height_km,
            max_alt=max_alt_val,
            map_limits=map_limits,
            full_map_limits=full_map_limits,
            num_fm=int(config.get("num_fm", 4)),
            num_ff=int(config.get("num_ff", 2)),
            num_em=int(config.get("num_em", 4)),
            # Auto-bump num_ef by 1 when opponent AWACS is present and non-engageable
            # so there is always a dedicated slot in the enemy observation for the AWACS.
            num_ef=int(config.get("num_enemy", 2))
            + (1 if awacs_cfg.get("opponent_awacs", False) and bool(awacs_non_engageable) else 0),
            num_pr=int(config.get("pr_slots", int(config.get("num_enemy", 2)))),
            num_warn=int(config.get("warn_sectors", 4)),
            own_state_dim=_own_state_dim(bool(config.get("emcon_action_enabled", False))),
            tacview_logfile=config.get("tacview_logfile", None),
            enable_missile_automation=neural_wrapper_config.get("enable_missile_automation", False),
            missile_auto_sqi_threshold=neural_wrapper_config.get("missile_auto_sqi_threshold", 0.3),
            missile_auto_max_per_target=neural_wrapper_config.get("missile_auto_max_per_target", 2),
            missile_auto_long_cooldown_s=neural_wrapper_config.get(
                "missile_auto_long_cooldown_s", 10.0
            ),
            missile_fire_threshold=float(neural_wrapper_config.get("missile_fire_threshold", 0.5)),
            enable_gun=bool(weapon_config.get("enable_gun", False)),
            datalink_mode=config.get("datalink_mode", "full"),
            datalink_drop_prob=float(config.get("datalink_drop_prob", 0.0)),
            record_traces=bool(config.get("record_traces", True)),
            geometry_config=geometry_config,
            scenario_config=scenario_config,
            metrics_config=metrics_config,
            debug=bool(config.get("debug", False)),
            information_mode=information_mode,
            oracle_use_reason=(str(oracle_use_reason).strip() if oracle_use_reason else None),
            reward_information_mode=reward_information_mode,
            ef_extra_dim=ef_extra_dim,
            extension_options=extension_options,
            emcon_action_enabled=bool(config.get("emcon_action_enabled", False)),
            emcon_policy=str(config.get("emcon_policy", "learned")).strip().lower(),
            emcon_period_steps=int(config.get("emcon_period_steps", 10)),
            emcon_duty=float(config.get("emcon_duty", 0.5)),
        )
