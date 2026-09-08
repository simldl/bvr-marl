"""Deterministic replay recorder for bvr_marl_core.Simulator.

Records the minimal information required to reproduce a simulation run exactly:
  - schema and simulator versions
  - the initial random seed
  - per-tick random state snapshots (so replay can restore RNG state)
  - per-tick external commands (controller outputs fed into the sim)

The recorder intentionally does not store full unit state — that can be
reconstructed by re-running from the seed with the same commands.
"""

from __future__ import annotations

import importlib.metadata
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Replay log schema version — bump when the log format itself changes.
_REPLAY_SCHEMA_VERSION = "1.2"


def _core_version() -> str:
    """Return the installed bvr-marl package version, or 'unknown'."""
    try:
        return importlib.metadata.version("bvr-marl")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


@dataclass
class TickRecord:
    tick: int
    utc_time_iso: str
    rng_state: Any  # random.getstate() snapshot before the tick
    commands: dict[int, Any]  # unit_id -> command dict (serialisable)
    events_summary: list[str]  # event class names emitted this tick


@dataclass
class ReplayLog:
    schema_version: str = _REPLAY_SCHEMA_VERSION
    scenario_id: str = ""
    scenario_version: str = ""  # scenario schema version at record time
    simulator_version: str = ""  # bvr-marl package version at record time
    initial_seed: int | None = None
    start_utc_iso: str = ""
    tick_secs: float = 1.0
    env_settings: dict[str, Any] = field(default_factory=dict)  # extra determinants
    experiment_metadata: dict[str, Any] = field(default_factory=dict)
    setup_events: list[str] = field(
        default_factory=list
    )  # event class names from pre-tick setup phase
    ticks: list[TickRecord] = field(default_factory=list)


class ReplayRecorder:
    """Attach to a Simulator to record a replayable trace."""

    def __init__(self, scenario_id: str = "", scenario_version: str = ""):
        self._scenario_id = scenario_id
        self._scenario_version = scenario_version
        self._log = ReplayLog(scenario_id=scenario_id)
        self._tick_counter = 0
        self._pending_rng_state: Any = None
        self._pending_commands: dict[int, Any] = {}
        self._pre_tick_event_count: int = 0

    def on_reset(
        self,
        simulator: Any,
        env_settings: dict[str, Any] | None = None,
    ) -> None:
        """Call after simulator.reset_sim() to capture initial state.

        Args:
            simulator: The Simulator instance.
            env_settings: Optional dict of extra determinants (e.g. map size,
                physics flags) that affect reproducibility.
        """
        self._tick_counter = 0
        self._log = ReplayLog(
            scenario_id=self._scenario_id,
            scenario_version=self._scenario_version,
            simulator_version=_core_version(),
            initial_seed=getattr(simulator, "random_seed", None),
            start_utc_iso=simulator.utc_time.isoformat(),
            tick_secs=simulator.tick_secs,
            env_settings=dict(env_settings or {}),
            experiment_metadata=dict(getattr(simulator, "replay_metadata", {})),
        )

    def record_setup_phase(self, simulator: Any) -> None:
        """Capture events emitted during the setup phase (before any tick runs).

        Call this after all pre-tick setup (e.g. add_unit calls) is complete and
        before the first before_tick/after_tick pair.  Snapshots simulator.events
        into log.setup_events so that registration and other setup events are part
        of the replay contract without being misattributed to a tick.
        """
        self._log.setup_events = [type(e).__name__ for e in getattr(simulator, "events", [])]

    def before_tick(self, simulator: Any, commands: dict[int, Any] | None = None) -> None:
        """Capture RNG state and pending commands before a tick executes."""
        self._pending_rng_state = simulator.rnd_gen.getstate()
        self._pending_commands = commands or {}
        self._pre_tick_event_count = len(getattr(simulator, "events", []))

    def after_tick(self, simulator: Any, events: list[Any]) -> None:
        """Record the tick outcome.

        Derives events_summary from simulator.events[pre_tick_count:] to capture
        all events logged during the tick, including those emitted via
        sim.log_event() rather than returned directly by do_tick().
        """
        sim_events: list[Any] = getattr(simulator, "events", [])
        tick_events = sim_events[self._pre_tick_event_count :]
        record = TickRecord(
            tick=self._tick_counter,
            utc_time_iso=simulator.utc_time.isoformat(),
            rng_state=self._pending_rng_state,
            commands={str(k): v for k, v in self._pending_commands.items()},
            events_summary=[type(e).__name__ for e in tick_events],
        )
        self._log.ticks.append(record)
        self._tick_counter += 1

    def save(self, path: str | Path) -> None:
        """Persist the replay log as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = {
            "schema_version": self._log.schema_version,
            "scenario_id": self._log.scenario_id,
            "scenario_version": self._log.scenario_version,
            "simulator_version": self._log.simulator_version,
            "initial_seed": self._log.initial_seed,
            "start_utc_iso": self._log.start_utc_iso,
            "tick_secs": self._log.tick_secs,
            "env_settings": self._log.env_settings,
            "experiment_metadata": self._log.experiment_metadata,
            "setup_events": self._log.setup_events,
            "ticks": [
                {
                    "tick": t.tick,
                    "utc_time_iso": t.utc_time_iso,
                    "rng_state": _serialise_rng_state(t.rng_state),
                    "commands": t.commands,
                    "events_summary": t.events_summary,
                }
                for t in self._log.ticks
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> ReplayLog:
        """Load a previously saved replay log."""
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        log = ReplayLog(
            schema_version=raw.get("schema_version", "1.0"),
            scenario_id=raw.get("scenario_id", ""),
            scenario_version=raw.get("scenario_version", ""),
            simulator_version=raw.get("simulator_version", ""),
            initial_seed=raw.get("initial_seed"),
            start_utc_iso=raw.get("start_utc_iso", ""),
            tick_secs=raw.get("tick_secs", 1.0),
            env_settings=raw.get("env_settings", {}),
            experiment_metadata=raw.get("experiment_metadata", {}),
        )
        log.setup_events = raw.get("setup_events", [])
        for t in raw.get("ticks", []):
            log.ticks.append(
                TickRecord(
                    tick=t["tick"],
                    utc_time_iso=t["utc_time_iso"],
                    rng_state=_deserialise_rng_state(t["rng_state"]),
                    commands={int(k): v for k, v in t.get("commands", {}).items()},
                    events_summary=t.get("events_summary", []),
                )
            )
        return log


def _serialise_rng_state(state):
    """Convert random.getstate() output to a JSON-serialisable structure."""
    if state is None:
        return None
    version, internalstate, gauss_next = state
    return {"version": version, "state": list(internalstate), "gauss_next": gauss_next}


def _deserialise_rng_state(raw):
    """Reconstruct a random.getstate()-compatible tuple."""
    if raw is None:
        return None
    return (raw["version"], tuple(raw["state"]), raw["gauss_next"])
