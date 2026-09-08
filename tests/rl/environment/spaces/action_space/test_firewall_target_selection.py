"""Firewall audit: sensor-limited target selection cannot reach world truth.

The sensor-limited path selects from operational contacts via ``select_contact``,
which takes no ``simulator`` argument and therefore structurally cannot read truth;
the truth-based ``select_target`` is reserved for the oracle path. This test locks
that invariant (item 1) so a future refactor cannot merge the two and leak truth.
"""

import inspect

from bvr_marl_core.domain.information_mode import InformationMode
from bvr_marl_core.rl.environment.spaces.action_space.utils.target_sorting import TargetSorter


def test_select_contact_has_no_simulator_access():
    params = set(inspect.signature(TargetSorter.select_contact).parameters)
    assert "simulator" not in params, "sensor-limited target selection must not receive truth"
    assert "sensor" not in params  # it reads unit.sensor.sensor_tracks, nothing global


def test_select_target_is_the_separate_truth_path():
    params = set(inspect.signature(TargetSorter.select_target).parameters)
    assert "simulator" in params  # the oracle path is the one that may read truth


def test_action_processor_dispatches_sensor_limited_to_contact_path():
    from bvr_marl_core.rl.environment.spaces.action_space import action_processor

    src = inspect.getsource(action_processor)
    # The sensor-limited branch must call select_contact, not select_target.
    idx = src.index("InformationMode.SENSOR_LIMITED")
    window = src[idx : idx + 400]
    assert "select_contact" in window
    assert window.index("select_contact") < window.find("select_target") % (10**9)
    assert InformationMode.SENSOR_LIMITED is not None
