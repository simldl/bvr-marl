# Graphics

Scripts that render figures from the actual `bvr_marl_core` physics, radar, and
engagement models — ported from `air-to-air-rl/.../graphics` and restyled for
publication.

## Readability at 3.25 in

Every figure is authored **at its final placement width** (a single column,
3.25 in) instead of being drawn large and shrunk afterwards. Because matplotlib
font sizes are absolute (points), the size set in the style *is* the size on the
page: drop a PNG into a 3.25 in column at 100 % and all text is **≥ 8 pt**.

The shared style lives in [`paper_style.py`](paper_style.py):

- `COLUMN_WIDTH_IN = 3.25`, smallest font 8 pt (ticks, legend, inline labels).
- `paper_figure(...)` returns a figure sized for the column; multi-panel figures
  are **stacked vertically** so each panel keeps the full column width.
- `save_paper_figure(fig, name)` writes `output/<name>.png` at 300 dpi.

A few comparison figures (RCS-vs-azimuth) are rendered at double-column width
(`2 * COLUMN_WIDTH_IN`); place them at the full text width without scaling and
the text is still ≥ 8 pt.

## Usage

Run a single figure:

```bash
python scripts/graphics/physics/drag_aircraft.py
```

Regenerate everything into `scripts/graphics/output/`:

```bash
python scripts/graphics/generate_all.py
```

Or import a plot function and tweak it:

```python
from scripts.graphics.physics.drag_aircraft import plot_aircraft_drag

fig = plot_aircraft_drag()
```

Output PNGs land in `scripts/graphics/output/` (git-ignored).

## Layout

| Folder      | Figures |
|-------------|---------|
| `physics/`  | air density, aircraft/missile drag, drag coefficients, thrust, turn rate |
| `radar/`    | RCS vs azimuth/elevation, detection-probability heatmaps |
| `aircraft/` | SQI vs aspect / range-closure, SQI heatmap, NEZ/DLZ zones |

`paper_style.py` (shared style) and `aircraft/sqi_model.py` (shared SQI model)
are helpers, not figure scripts.
