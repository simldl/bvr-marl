from bvr_marl_core.simulator.core.experiment_metadata import (
    _static_metadata,
    build_experiment_metadata,
    canonical_hash,
    prototype_warnings,
)
from bvr_marl_core.simulator.core.recorder import ReplayRecorder
from bvr_marl_core.simulator.simulator import Simulator


def test_configuration_hash_is_order_independent_and_sensitive():
    assert canonical_hash({"b": 2, "a": 1}) == canonical_hash({"a": 1, "b": 2})
    assert canonical_hash({"a": 1}) != canonical_hash({"a": 2})


def test_metadata_contains_required_reproducibility_fields():
    metadata = build_experiment_metadata({"tick_secs": 0.5})
    required = {
        "core_commit",
        "extension_commit",
        "configuration_hash",
        "python_version",
        "dependency_fingerprint",
        "observation_schema_version",
        "action_schema_version",
        "model_status_matrix_version",
        "network_picture_version",
        "platform_parameter_set_version",
        "weapon_parameter_set_version",
        "validation_suite_version",
    }
    assert required <= metadata.keys()
    assert len(metadata["configuration_hash"]) == 64


def test_process_stable_metadata_is_cached_but_configuration_hash_is_not():
    _static_metadata.cache_clear()
    first = build_experiment_metadata({"case": 1})
    cache_after_first = _static_metadata.cache_info()
    second = build_experiment_metadata({"case": 2})
    cache_after_second = _static_metadata.cache_info()

    assert cache_after_second.hits == cache_after_first.hits + 1
    assert first["dependency_fingerprint"] == second["dependency_fingerprint"]
    assert first["configuration_hash"] != second["configuration_hash"]


def test_replay_persists_experiment_metadata(tmp_path):
    simulator = Simulator(random_seed=12)
    recorder = ReplayRecorder()
    recorder.on_reset(simulator)
    path = tmp_path / "replay.json"
    recorder.save(path)
    loaded = recorder.load(path)
    assert loaded.experiment_metadata["root_seed"] == 12
    assert "core_commit" in loaded.experiment_metadata


def test_privileged_modes_produce_governance_warnings():
    warnings = prototype_warnings(
        {"information_mode": "oracle", "reward_information_mode": "privileged_training"}
    )
    assert len(warnings) == 2
