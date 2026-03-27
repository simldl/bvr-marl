import numpy as np

from air_to_air_rl.radar.obs.data_fusion import TrackFusion


def test_fuse_estimates_simple():
    estimates = [np.array([1, 2, 3]), np.array([2, 4, 6])]
    covs = [np.eye(3), np.eye(3) * 2]
    fused_state, fused_cov = TrackFusion.fuse_estimates(estimates, covs)
    assert np.allclose(fused_state, [1.3333, 2.6667, 4.0], atol=0.01)
    assert fused_cov.shape == (3, 3)


def test_process_with_fusion_combines_all():
    own_pos = type("Dummy", (), {})()
    own_pos.lat, own_pos.lon, own_pos.alt = 0.0, 0.0, 0.0
    dummy_track = (np.array([10.0, 20.0, 30.0]), np.eye(3), "tgt", None, None)

    class DummyRadar:
        def process(self, pos, targets, dt, a, b):
            return [(np.array([40.0, 50.0, 60.0]), np.eye(3) * 3, "tgt", None, None)]

    fused = TrackFusion.process_with_fusion(
        own_pos, [dummy_track], [own_pos], [DummyRadar()], ["tgt"], 1.0
    )
    assert len(fused) == 1
    fused_state, fused_cov = fused[0]
    assert fused_cov.shape == (3, 3)
