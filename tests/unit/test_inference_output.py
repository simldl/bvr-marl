from types import SimpleNamespace

import pytest
import torch

from bvr_marl_core.visualization.model_wrapper.inference_output import deterministic_actions

_COLUMNS = SimpleNamespace(ACTIONS="actions_column", ACTION_DIST_INPUTS="dist_column")


@pytest.mark.parametrize("key", [_COLUMNS.ACTIONS, "actions"])
def test_extracts_batched_action_vector(key):
    output = {key: torch.tensor([[0.2, 0.8]])}

    assert deterministic_actions(output, _COLUMNS).tolist() == pytest.approx([0.2, 0.8])


@pytest.mark.parametrize("key", [_COLUMNS.ACTION_DIST_INPUTS, "action_dist_inputs"])
def test_extracts_distribution_means_from_time_batch(key):
    output = {key: torch.tensor([[[0.2, 0.8, -1.0, -2.0]]])}

    assert deterministic_actions(output, _COLUMNS).tolist() == pytest.approx([0.2, 0.8])


def test_rejects_unknown_output_schema():
    with pytest.raises(ValueError, match="Unexpected output format"):
        deterministic_actions({"other": torch.ones(1)}, _COLUMNS)
