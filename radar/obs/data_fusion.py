import numpy as np
from radar.core.utils import enu_to_geodetic, geodetic_to_enu

class TrackFusion:
    @staticmethod
    def fuse_estimates(estimates, covariances):
        inv_covs = [np.linalg.inv(cov) for cov in covariances]
        precision_sum = sum(inv_covs)
        weighted_sum = sum(inv.dot(est) for inv, est in zip(inv_covs, estimates))
        fused_state = np.linalg.solve(precision_sum, weighted_sum)
        fused_cov = np.linalg.inv(precision_sum)
        return fused_state, fused_cov

    @staticmethod
    def process_with_fusion(own_position, own_tracks, external_positions, external_radars, targets, dt):
        fused_results = []
        for target in targets:
            estimates, covariances = [], []
            for state, cov, tgt, utype, ref_pos in own_tracks:
                estimates.append(state[:3])
                covariances.append(cov)
            for ext_pos, ext_radar in zip(external_positions, external_radars):
                ext_tracks = ext_radar.process(ext_pos, [target], dt, 0, 0)
                for state, cov, tgt, utype, ref_pos in ext_tracks:
                    lat, lon, alt = enu_to_geodetic(state[:3], ext_pos.lat, ext_pos.lon, ext_pos.alt)
                    enu_transformed = geodetic_to_enu(lat, lon, alt, own_position.lat, own_position.lon, own_position.alt)
                    estimates.append(enu_transformed)
                    covariances.append(cov)
            if estimates:
                fused_state, fused_cov = TrackFusion.fuse_estimates(estimates, covariances)
                fused_results.append((fused_state, fused_cov))
        return fused_results
