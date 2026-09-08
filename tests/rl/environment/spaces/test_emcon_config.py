"""Config-driven EMCON action enable."""

from bvr_marl_core.rl.environment.gym.gym_components.config import BVREnvConfig
from bvr_marl_core.rl.environment.spaces.action_space import (
    BASE_ACTION_DIM,
    ActionSpaceManager,
    emcon_action_dim,
)


def test_emcon_action_dim_helper():
    assert emcon_action_dim(False) == BASE_ACTION_DIM
    assert emcon_action_dim(True) == BASE_ACTION_DIM + 1
    assert emcon_action_dim(None) == BASE_ACTION_DIM  # module default (disabled)


def test_config_defaults_emcon_off():
    assert BVREnvConfig.from_dict({}).emcon_action_enabled is False


def test_config_enables_emcon():
    assert BVREnvConfig.from_dict({"emcon_action_enabled": True}).emcon_action_enabled is True


def test_action_space_shape_tracks_emcon():
    off = ActionSpaceManager(["a"], shape=emcon_action_dim(False))
    on = ActionSpaceManager(["a"], shape=emcon_action_dim(True))
    assert off.get("a").shape[0] == 9
    assert on.get("a").shape[0] == 10
