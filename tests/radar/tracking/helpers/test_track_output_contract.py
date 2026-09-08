import numpy as np

from bvr_marl_core.domain.tactical_contact import TacticalContact
from bvr_marl_core.radar.tracking.helpers.track_manager import build_track_snapshots
from bvr_marl_core.simulator.core.helpers import Position


class _Filter:
    missed_updates = 0

    def __init__(self, state: np.ndarray, covariance: np.ndarray):
        self._state = state
        self._covariance = covariance

    def get_state(self) -> np.ndarray:
        return self._state.copy()

    def get_covariance(self) -> np.ndarray:
        return self._covariance.copy()


def test_production_track_output_preserves_full_covariance_for_contact_boundary():
    state = np.array([1_000.0, 2_000.0, 3_000.0, 100.0, 200.0, 300.0])
    factor = np.array(
        [
            [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 11.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 3.0, 12.0, 0.0, 0.0, 0.0],
            [4.0, 0.0, 1.0, 13.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 2.0, 14.0, 0.0],
            [0.0, 0.0, 6.0, 0.0, 3.0, 15.0],
        ]
    )
    covariance = factor @ factor.T
    track_filter = _Filter(state, covariance)
    metadata = {
        "n_obs_hist": [2],
        "update_count": 2,
        "updates_with_meas": 2,
        "unit_type": "Aircraft",
    }

    tracks = build_track_snapshots(
        {4: track_filter},
        {4: Position(48.0, 11.0, 5_000.0)},
        {4: metadata},
        clusters=[],
    )

    exported_covariance = np.asarray(tracks[0].covariance)
    assert exported_covariance.shape == (6, 6)
    np.testing.assert_allclose(exported_covariance, covariance)
    assert np.allclose(exported_covariance, exported_covariance.T)
    assert np.linalg.eigvalsh(exported_covariance).min() >= 0.0

    contact = TacticalContact.from_sensor_track(tracks[0])
    np.testing.assert_allclose(contact.covariance, covariance)
