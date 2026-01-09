import pytest
from radar.core.parameter_policy import RadarParamPolicy

@pytest.fixture
def param_policy():
    return RadarParamPolicy(
        search_h_fov_deg=120.0, search_v_fov_deg=60.0,
        search_gain_db=40.0, search_snr_db=8.0
    )

def test_search_mode_returns_search_values(param_policy):
    params = param_policy.get_params_for_mode("search")
    assert params["h_fov_deg"] == 120.0
    assert params["antenna_gain_db"] == 40.0

def test_lock_mode_narrower_fov(param_policy):
    params = param_policy.get_params_for_mode("lock")
    assert params["h_fov_deg"] < 120.0
    assert params["antenna_gain_db"] > 40.0

def test_adaptive_fov_interpolates(param_policy):
    fov_far = param_policy.get_adaptive_fov(40000)
    fov_near = param_policy.get_adaptive_fov(10000)
    assert fov_far > fov_near
