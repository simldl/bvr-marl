from types import SimpleNamespace

import pytest

from bvr_marl_core.domain.information_mode import (
    InformationMode,
    TruthAccessError,
    reject_truth_handle,
    resolve_information_mode,
)
from bvr_marl_core.rl.environment.gym.gym_components.config import BVREnvConfig
from bvr_marl_core.rl.environment.spaces.action_space.base_processor import ActionProcessorBase


def test_information_mode_parsing_is_explicit():
    assert (
        resolve_information_mode("sensor_limited", default=InformationMode.ORACLE)
        is InformationMode.SENSOR_LIMITED
    )
    with pytest.raises(ValueError, match="Unknown information mode"):
        resolve_information_mode("sometimes_truth", default=InformationMode.ORACLE)


def test_sensor_limited_boundary_rejects_entity_shaped_truth():
    entity = SimpleNamespace(id=7, position=object(), velocity=object())
    with pytest.raises(TruthAccessError, match="policy_observation"):
        reject_truth_handle(entity, boundary="policy_observation")


def test_sensor_limited_boundary_accepts_immutable_values():
    reject_truth_handle((1.0, 2.0, 3.0), boundary="policy_observation")


def test_action_processing_omission_fails_closed_to_sensor_limited():
    processor = ActionProcessorBase(SimpleNamespace())
    assert processor.information_mode is InformationMode.SENSOR_LIMITED


def test_environment_config_defaults_to_sensor_limited():
    config = BVREnvConfig.from_dict({"num_agents_per_team": 1})
    assert config.information_mode == "sensor_limited"
    assert config.oracle_use_reason is None


def test_oracle_environment_requires_a_nonempty_reason():
    with pytest.raises(ValueError, match="oracle_use_reason"):
        BVREnvConfig.from_dict({"num_agents_per_team": 1, "information_mode": "oracle"})

    config = BVREnvConfig.from_dict(
        {
            "num_agents_per_team": 1,
            "information_mode": "oracle",
            "oracle_use_reason": "diagnostic upper bound",
        }
    )
    assert config.information_mode == "oracle"
