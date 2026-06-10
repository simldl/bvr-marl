"""Target-allocation: the selector steers a fighter off a saturated bandit."""

from types import SimpleNamespace

from bvr_marl_core.rl.environment.spaces.action_space.utils.target_sorting import TargetSorter


def _u(uid, group, lat=0.0, lon=0.0, alt=10000.0, is_missile=False, target=None):
    return SimpleNamespace(
        id=uid,
        group=group,
        position=SimpleNamespace(lat=lat, lon=lon, alt=alt),
        is_missile=is_missile,
        is_non_engageable=False,
        is_countermeasure=False,
        target=target,
    )


def _sim(units):
    return SimpleNamespace(active_units={u.id: u for u in units})


def test_saturated_bandit_is_not_selectable_when_alternative_exists():
    """bx has 2 inbound missiles (saturated); even an action that points at it
    (action_target ~ 1.0 -> last/most-engaged) must yield the unsaturated by."""
    ts = TargetSorter(max_missiles_per_target=2)
    shooter = _u(1, "A")
    bx = _u(2, "B", lat=0.10)  # closer bandit, saturated
    by = _u(3, "B", lat=0.20)  # farther bandit, unsaturated
    m1 = _u(10, "A", is_missile=True, target=bx)
    m2 = _u(11, "A", is_missile=True, target=bx)
    sim = _sim([shooter, bx, by, m1, m2])

    sel = ts.select_target(shooter, sim, action_target=0.99, state=ts.init_target_state(), dt=1.0)
    assert sel is by


def test_without_filter_saturated_bandit_is_still_selectable():
    """With the cap disabled (0), the legacy behaviour stands: action_target ~ 1.0
    can still pick the saturated bandit bx."""
    ts = TargetSorter(max_missiles_per_target=0)
    shooter = _u(1, "A")
    bx = _u(2, "B", lat=0.10)
    by = _u(3, "B", lat=0.20)
    m1 = _u(10, "A", is_missile=True, target=bx)
    m2 = _u(11, "A", is_missile=True, target=bx)
    sim = _sim([shooter, bx, by, m1, m2])

    sel = ts.select_target(shooter, sim, action_target=0.99, state=ts.init_target_state(), dt=1.0)
    assert sel is bx


def test_falls_back_to_saturated_when_no_alternative():
    """1v1: the only bandit is saturated -> it is still selectable (no veto here;
    the firing handler enforces the cap)."""
    ts = TargetSorter(max_missiles_per_target=2)
    shooter = _u(1, "A")
    bx = _u(2, "B", lat=0.10)
    m1 = _u(10, "A", is_missile=True, target=bx)
    m2 = _u(11, "A", is_missile=True, target=bx)
    sim = _sim([shooter, bx, m1, m2])

    sel = ts.select_target(shooter, sim, action_target=0.0, state=ts.init_target_state(), dt=1.0)
    assert sel is bx
