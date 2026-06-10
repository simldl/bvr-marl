import math

import numpy as np

from bvr_marl_core.rl.environment.spaces.obs_helper import (
    build_warn_sector_features,
    extract_feature_matrix,
    mask_for_units,
    onehot_encode,
    pad_cached,
    pad_generic,
    pad_indices,
    rel_state,
)


class DummyUnit:
    def __init__(self, lat=0, lon=0, alt=0, speed=0, yaw_deg=0, pitch_deg=0):
        self.position = type("P", (), {"lat": lat, "lon": lon, "alt": alt})()
        self.speed = speed
        self.yaw_deg = yaw_deg
        self.pitch_deg = pitch_deg


def test_rel_state_and_position():
    a = DummyUnit(0, 0, 0, 100, 0, 0)
    b = DummyUnit(0.001, 0.002, 10, 150, 90, 10)
    s = rel_state(a, b)
    assert isinstance(s, np.ndarray) and s.shape == (6,)


def test_pad_generic_and_indices():
    arr = [[1, 2, 3], [4, 5, 6]]
    padded, mask = pad_generic(arr, 4, 3)
    assert padded.shape == (4, 3)
    assert (mask[:2] == 1).all() and (mask[2:] == 0).all()
    idxs, idx_mask = pad_indices([3, 2], 4)
    assert (idx_mask[:2] == 1).all() and (idx_mask[2:] == 0).all()
    assert idxs.shape == (4,)


def test_pad_cached():
    arrs = [np.ones(3), np.zeros(3)]
    padded, mask = pad_cached(arrs, 4, 3)
    assert padded.shape == (4, 3)
    assert mask.shape == (4,)


def test_extract_feature_matrix():
    ul = [DummyUnit(i, i, i) for i in range(3)]
    feats, mask = extract_feature_matrix(ul, lambda u: [u.position.lat], 5, 1)
    assert feats.shape == (5, 1)
    assert (mask[:3] == 1).all()


def test_build_warn_sector_features():
    arr = build_warn_sector_features(None, {0: 2, 2: 1}, 4)
    assert (arr[0] == 2) and (arr[2] == 1) and arr.shape == (4,)


def test_onehot_encode():
    arr = onehot_encode(2, 5)
    assert arr[2] == 1.0 and arr.sum() == 1.0


def test_mask_for_units():
    units = [object(), None, object()]
    mask = mask_for_units(units, 4)
    assert (mask[:1] == 1).all() and mask.shape == (4,)
