import numpy as np
from typing import List, Optional
from collections import defaultdict

class Clusterer:
    """Clusters raw radar detections into bins based on range and angles."""
    def __init__(self, angular_res_deg: float, range_res_m: float, device: Optional[object] = None):
        self.angular_res_deg = angular_res_deg
        self.range_res_m = range_res_m
        # device parameter kept for API compatibility but not used with NumPy

    def cluster(self, dets: List[dict]) -> List[dict]:
        """Group detections whose (azimuth, elevation, range) fall into the same discrete bins."""
        if not dets:
            return []

        # Extract arrays using NumPy (much faster than PyTorch for small arrays)
        n = len(dets)
        az = np.empty(n, dtype=np.float32)
        el = np.empty(n, dtype=np.float32)
        di = np.empty(n, dtype=np.float32)
        dop = np.empty(n, dtype=np.float32)
        lat = np.empty(n, dtype=np.float32)
        lon = np.empty(n, dtype=np.float32)
        alt = np.empty(n, dtype=np.float32)

        for i, d in enumerate(dets):
            az[i] = d['az']
            el[i] = d['el']
            di[i] = d['d']
            dop[i] = d.get('dop', 0.0)
            lat[i] = d.get('lat', 0.0)
            lon[i] = d.get('lon', 0.0)
            alt[i] = d.get('alt', 0.0)

        # Discrete bin indices
        bix = np.floor(az / self.angular_res_deg).astype(np.int32)
        biy = np.floor(el / self.angular_res_deg).astype(np.int32)
        biz = np.floor(di / self.range_res_m).astype(np.int32)

        # Group by bin using dictionary (faster than unique for small arrays)
        bin_dict = defaultdict(list)
        for i in range(n):
            key = (bix[i], biy[i], biz[i])
            bin_dict[key].append(i)

        clusters = []
        for sel_list in bin_dict.values():
            # Aggregate ECM fields from detections
            is_deception = any(dets[j].get('is_deception', False) for j in sel_list)
            engagement_ids = [dets[j].get('engagement_id') for j in sel_list if dets[j].get('engagement_id') is not None]
            jammer_ids = [dets[j].get('jammer_id') for j in sel_list if dets[j].get('jammer_id') is not None]

            # Majority vote for engagement_id (most common value)
            engagement_id = max(set(engagement_ids), key=engagement_ids.count) if engagement_ids else None
            jammer_id = max(set(jammer_ids), key=jammer_ids.count) if jammer_ids else None

            # Use NumPy indexing to compute means (fast vectorized operations)
            sel_arr = np.array(sel_list, dtype=np.int32)
            cluster = {
                'az': float(az[sel_arr].mean()),
                'el': float(el[sel_arr].mean()),
                'd': float(di[sel_arr].mean()),
                'dop': float(dop[sel_arr].mean()),
                'lat': float(lat[sel_arr].mean()),
                'lon': float(lon[sel_arr].mean()),
                'alt': float(alt[sel_arr].mean()),
                'T': [dets[j].get('T', None) for j in sel_list],
                'is_deception': is_deception,
                'engagement_id': engagement_id,
                'jammer_id': jammer_id
            }

            # Never set 'T' on deception clusters (enforce no trusted data on ghosts)
            if is_deception:
                cluster['T'] = None

            clusters.append(cluster)
        return clusters