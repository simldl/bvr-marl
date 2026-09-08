import numpy as np

from bvr_marl_core.visualization.scenplotter.symbol_registry import SymbolRegistry


def _visible_size(surface) -> tuple[int, int]:
    surface.flush()
    pixels = np.frombuffer(surface.get_data(), dtype=np.uint8).reshape(
        surface.get_height(), surface.get_stride() // 4, 4
    )
    ys, xs = np.nonzero(pixels[:, : surface.get_width(), 3])
    return int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def test_all_fighter_symbols_have_same_visible_longest_dimension():
    registry = SymbolRegistry(mode="flying_objects")

    surfaces = [
        registry.get_flying(fighter, "blue", 64)
        for fighter in ("f22", "f35", "eurofighter", "su57", "f15ex")
    ]

    # The original 64 px Eurofighter asset occupied 38 pixels along its longest
    # axis. All fighter artwork should retain that compact visible scale.
    assert all(max(_visible_size(surface)) >= 37 for surface in surfaces)
    assert {max(surface.get_width(), surface.get_height()) for surface in surfaces} == {38}
