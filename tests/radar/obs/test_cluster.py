import torch
from radar.obs.cluster import Clusterer

def test_empty_cluster_returns_empty():
    clusterer = Clusterer(angular_res_deg=2.0, range_res_m=50.0, device=torch.device("cpu"))
    assert clusterer.cluster([]) == []

def test_single_detection_returns_one_cluster():
    clusterer = Clusterer(angular_res_deg=2.0, range_res_m=50.0, device=torch.device("cpu"))
    dets = [{'az': 0, 'el': 0, 'd': 100, 'dop': 0.1, 'T': 'a'}]
    clusters = clusterer.cluster(dets)
    assert len(clusters) == 1
    assert clusters[0]['T'] == ['a']

def test_multiple_bins():
    clusterer = Clusterer(angular_res_deg=2.0, range_res_m=100.0, device=torch.device("cpu"))
    dets = [
        {'az': 1.0, 'el': 1.0, 'd': 100, 'dop': 0.0, 'T': 'A'},
        {'az': 1.2, 'el': 1.0, 'd': 105, 'dop': 0.0, 'T': 'B'},
        {'az': 10.0, 'el': 1.0, 'd': 300, 'dop': 0.0, 'T': 'C'}
    ]
    clusters = clusterer.cluster(dets)
    assert len(clusters) == 2
    all_Ts = [t for cl in clusters for t in cl['T']]
    assert 'A' in all_Ts and 'B' in all_Ts and 'C' in all_Ts

def test_detections_are_grouped_by_discrete_bins():
    clusterer = Clusterer(angular_res_deg=5.0, range_res_m=200.0, device=torch.device("cpu"))
    dets = [
        {'az': 0, 'el': 0, 'd': 100, 'dop': 0.0, 'T': 'X'},
        {'az': 4, 'el': 0, 'd': 150, 'dop': 0.0, 'T': 'Y'},
        {'az': 6, 'el': 0, 'd': 700, 'dop': 0.0, 'T': 'Z'}
    ]
    clusters = clusterer.cluster(dets)
    assert len(clusters) == 2