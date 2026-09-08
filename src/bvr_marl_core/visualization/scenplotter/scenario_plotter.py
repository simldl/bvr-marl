import io
import math
from collections import namedtuple

import cairo
import cartopy
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from bvr_marl_core.simulator import MapLimits


def get_plotter_angle(yaw_deg: float) -> float:
    return -math.radians(yaw_deg)


ColorRGBA = namedtuple("ColorRGBA", ["red", "green", "blue", "alpha"])
_WHITE = ColorRGBA(1, 1, 1, 1)
_TRANSPARENT_WHITE = ColorRGBA(1, 1, 1, 0)
_GRAY = ColorRGBA(0.5, 0.5, 0.5, 1)


def _affiliation_to_side(affiliation: str) -> str:
    """Map doctrinal affiliation to blue/red side for flying_objects symbol lookup."""
    return {"friendly": "blue", "hostile": "red"}.get(affiliation, "blue")


class PlotConfig:
    def __init__(self):
        self.show_grid = True
        self.units_scale = 35
        self.background_color = "#1e1e1e"
        self.borders_color = "#ffffff"
        self.coastlines_color = "#607080"  # colour passed to ax.coastlines()
        self.grid_color = "#404050"  # colour passed to ax.gridlines()
        self.sprites_info_font = "DejaVu Sans Condensed"
        self.sprites_info_font_style = cairo.FONT_SLANT_NORMAL
        self.sprites_info_font_weight = cairo.FONT_WEIGHT_NORMAL
        self.sprites_info_font_size = 10
        self.sprites_info_spacing = 26
        self.status_message_font = "DejaVu Sans Condensed"
        self.status_message_font_style = cairo.FONT_SLANT_NORMAL
        self.status_message_font_weight = cairo.FONT_WEIGHT_BOLD
        self.status_message_font_size = 14
        # Master text toggle for map labels, status overlays, and scale-bar labels.
        self.show_text = True
        # Symbol rendering mode: "nato" (MIL-STD-2525C), "flying_objects" (type-specific
        # aircraft silhouettes), or "procedural" (Cairo shapes, no external files)
        self.symbol_mode = "nato"
        # Symbol size in pixels for aircraft/AWACS (request size — actual may differ)
        self.symbol_size = 32
        self.missile_symbol_size = 20
        # Scale factor applied when rendering sprites (>1.0 makes symbols larger)
        self.symbol_scale = 1.0
        # Missiles rendered at a fraction of symbol_scale (set by initialize_plotter)
        self.missile_symbol_scale = 0.5


class Drawable:
    def __init__(self, zorder):
        """Base class for any drawable object."""
        self.zorder = zorder


class StatusMessage(Drawable):
    def __init__(self, text, text_color=_WHITE, zorder: int = 0):
        """Displays a message in the bottom left position."""
        super().__init__(zorder)
        self.text = text
        self.text_color = text_color


class TopLeftMessage(Drawable):
    def __init__(self, text, text_color=_WHITE, zorder: int = 0):
        """Displays a message in the top left position."""
        super().__init__(zorder)
        self.text = text
        self.text_color = text_color


class PolyLine(Drawable):
    def __init__(
        self,
        points: list[tuple[float, float]],
        line_width: float = 1.0,
        dash: tuple[float, float] | None = None,
        edge_color=_WHITE,
        zorder: int = 0,
    ):
        """Draws a series of connected lines."""
        super().__init__(zorder)
        self.points = points
        self.line_width = line_width
        self.dash = dash
        self.edge_color = edge_color


class Rect(Drawable):
    def __init__(
        self,
        left_lon: float,
        bottom_lat: float,
        right_lon: float,
        top_lat: float,
        line_width: float = 1.0,
        edge_color=_WHITE,
        fill_color=_TRANSPARENT_WHITE,
        zorder: int = 0,
    ):
        """Draws a rectangle."""
        super().__init__(zorder)
        self.left_lon = left_lon
        self.bottom_lat = bottom_lat
        self.right_lon = right_lon
        self.top_lat = top_lat
        self.line_width = line_width
        self.edge_color = edge_color
        self.fill_color = fill_color


class Arc(Drawable):
    def __init__(
        self,
        center_lat: float,
        center_lon: float,
        radius: float,
        angle1: float,
        angle2: float,
        line_width: float = 1.0,
        dash: tuple[float, float] | None = None,
        edge_color=None,
        fill_color=None,
        zorder: int = 0,
    ):
        """Draws an arc."""
        super().__init__(zorder)
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.radius = radius
        self.angle1 = angle1
        self.angle2 = angle2
        self.line_width = line_width
        self.dash = dash
        self.edge_color = edge_color
        self.fill_color = fill_color


class Sprite(Drawable):
    def __init__(
        self,
        lat: float,
        lon: float,
        yaw_deg: float,
        edge_color=_WHITE,
        fill_color=_GRAY,
        info_text: str | None = None,
        zorder: int = 0,
        affiliation: str = "unknown",
    ):
        """Base class for drawable shapes with a position."""
        super().__init__(zorder)
        self.lat = lat
        self.lon = lon
        self.yaw_deg = yaw_deg
        self.edge_color = edge_color
        self.fill_color = fill_color
        self.info_text = info_text
        self.affiliation = affiliation


class MapLabel(Drawable):
    """A map-anchored label whose pixel offset is assigned by the label manager."""

    def __init__(
        self,
        lat: float,
        lon: float,
        text: str,
        *,
        unit_id: object = None,
        priority: int = 0,
        affiliation: str = "unknown",
        cluster_name: str = "aircraft",
        zorder: int = 20,
    ):
        super().__init__(zorder)
        self.lat = lat
        self.lon = lon
        self.text = text
        self.unit_id = unit_id
        self.priority = priority
        self.affiliation = affiliation
        self.cluster_name = cluster_name
        self.cluster_count = 1
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.draw_leader = False
        self.visible = True


class Airplane(Sprite):
    def __init__(
        self,
        lat: float,
        lon: float,
        yaw_deg: float,
        edge_color="#ffffff",
        fill_color="#888888",
        info_text: str | None = None,
        zorder: int = 0,
        affiliation: str = "unknown",
        aircraft_type: str = "f22",
    ):
        """Represents an airplane sprite.

        Args:
            aircraft_type: Type key matching a flying_objects SVG stem (e.g. "f22", "eurofighter").
        """
        super().__init__(lat, lon, yaw_deg, edge_color, fill_color, info_text, zorder, affiliation)
        self.aircraft_type = aircraft_type


class AWACS(Sprite):
    def __init__(
        self,
        lat: float,
        lon: float,
        yaw_deg: float,
        edge_color="#ffffff",
        fill_color="#888888",
        info_text: str | None = None,
        zorder: int = 0,
        affiliation: str = "unknown",
        aircraft_type: str = "awacs",
    ):
        """Represents an AWACS (Airborne Warning and Control System) sprite with distinctive rotodome."""
        super().__init__(lat, lon, yaw_deg, edge_color, fill_color, info_text, zorder, affiliation)
        self.aircraft_type = aircraft_type


class SamBattery(Sprite):
    def __init__(
        self,
        lat: float,
        lon: float,
        yaw_deg: float,
        missile_range_km: float,
        radar_range_km: float,
        radar_amplitude_deg: float,
        edge_color="#ffffff",
        fill_color="#888888",
        info_text: str | None = None,
        zorder: int = 0,
        affiliation: str = "unknown",
    ):
        """Represents a SAM battery sprite."""
        super().__init__(lat, lon, yaw_deg, edge_color, fill_color, info_text, zorder, affiliation)
        self.missile_range_km = missile_range_km
        self.radar_range_km = radar_range_km
        self.radar_amplitude_deg = radar_amplitude_deg


class Missile(Sprite):
    def __init__(
        self,
        lat: float,
        lon: float,
        yaw_deg: float,
        edge_color="#ffffff",
        fill_color="#888888",
        info_text: str | None = None,
        zorder: int = 0,
        affiliation: str = "unknown",
    ):
        """Represents a missile sprite."""
        super().__init__(lat, lon, yaw_deg, edge_color, fill_color, info_text, zorder, affiliation)


class Waypoint(Sprite):
    def __init__(
        self,
        lat: float,
        lon: float,
        edge_color="#ffffff",
        fill_color="#888888",
        info_text: str | None = None,
        zorder: int = 0,
    ):
        """Represents a waypoint sprite."""
        super().__init__(lat, lon, 0, edge_color, fill_color, info_text, zorder)


class RadarCone(Drawable):
    def __init__(
        self,
        center_lat: float,
        center_lon: float,
        radius: float,
        angle1: float,
        angle2: float,
        edge_color=None,
        fill_color=None,
        line_width: float = 1.0,
        zorder: int = 0,
    ):
        """Represents a radar detection cone."""
        super().__init__(zorder)
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.radius = radius
        self.angle1 = angle1
        self.angle2 = angle2
        self.edge_color = edge_color
        self.fill_color = fill_color
        self.line_width = line_width


class ScaleBar(Drawable):
    def __init__(
        self,
        length_km: float,
        x: float,
        y: float,
        line_width: float = 2,
        edge_color=(1, 1, 1, 1),
        zorder: int = 50,
    ):
        """Represents a scale bar to indicate distance."""
        super().__init__(zorder)
        self.length_km = length_km
        self.x = x  # x-coordinate in pixels (center of the scale bar)
        self.y = y  # y-coordinate in pixels (baseline of the scale bar)
        self.line_width = line_width
        self.edge_color = edge_color


class BackgroundMesh:
    def __init__(self, lons, lats, vals, cmap: str, vmin: float = None, vmax: float = None):
        """Defines a background mesh for the map."""
        self.lons = lons
        self.lats = lats
        self.vals = vals
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax


def compute_map_figure_size_inches(
    map_extents: MapLimits,
    base_height_in: float = 6.0,
) -> tuple[float, float]:
    """Return a figure size that preserves the map's width/height ratio."""
    # Use the scenario box extents directly so rectangular training maps rendered with
    # the simulator's flat 111 km/deg convention keep their intended aspect ratio.
    width_deg = max(1e-6, map_extents.longitude_extent())
    height_deg = max(1e-6, map_extents.latitude_extent())
    aspect_ratio = max(0.25, min(8.0, width_deg / height_deg))
    return (base_height_in * aspect_ratio, base_height_in)


class ScenarioPlotter:
    def __init__(
        self,
        map_extents: MapLimits,
        dpi=200,
        background_mesh: BackgroundMesh | None = None,
        config: PlotConfig | None = None,
        symbol_registry=None,
    ):
        self.map_extents = map_extents
        self.dpi = dpi
        self.bg_mesh = background_mesh
        self.cfg = config or PlotConfig()
        self.symbol_registry = symbol_registry
        self.projection = cartopy.crs.Mercator(
            central_longitude=(map_extents.left_lon + map_extents.right_lon) / 2
        )
        self._background = self._build_background_image()
        self.img_width = self._background.get_width()
        self.img_height = self._background.get_height()
        self.pixels_km = self.img_width / map_extents.max_longitude_extent_km()
        # Reused across frames — avoids a per-frame malloc of the full image buffer.
        self._frame_surface = cairo.ImageSurface(
            cairo.FORMAT_ARGB32, self.img_width, self.img_height
        )

    def _build_background_image(self):
        """Builds the background map image using Cartopy and Cairo."""
        fig_width_in, fig_height_in = compute_map_figure_size_inches(self.map_extents)
        fig = Figure(figsize=(fig_width_in, fig_height_in), dpi=self.dpi)
        fig.patch.set_facecolor(self.cfg.background_color)
        FigureCanvasAgg(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection=self.projection)
        ax.set_extent(
            (
                self.map_extents.left_lon,
                self.map_extents.right_lon,
                self.map_extents.bottom_lat,
                self.map_extents.top_lat,
            )
        )
        if self.bg_mesh is not None:
            ax.pcolormesh(
                self.bg_mesh.lons,
                self.bg_mesh.lats,
                self.bg_mesh.vals,
                vmin=self.bg_mesh.vmin,
                vmax=self.bg_mesh.vmax,
                cmap=self.bg_mesh.cmap,
                shading="nearest",
                transform=cartopy.crs.PlateCarree(),
            )
        ax.patch.set_facecolor(self.cfg.background_color)
        ax.add_feature(
            cartopy.feature.BORDERS,
            edgecolor=self.cfg.borders_color,
            linewidth=0.2,
            linestyle="-",
            alpha=1,
        )
        ax.coastlines(resolution="110m", color=self.cfg.coastlines_color, linewidth=0.5)
        if self.cfg.show_grid:
            ax.gridlines(linewidth=0.2, color=self.cfg.grid_color)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=self.dpi, pad_inches=0)
        buf.seek(0)
        src = cairo.ImageSurface.create_from_png(buf)

        # Convert to ARGB32 once so every subsequent frame paint is a same-format
        # blit (pure memcpy) with no per-frame format conversion overhead.
        argb = cairo.ImageSurface(cairo.FORMAT_ARGB32, src.get_width(), src.get_height())
        ctx = cairo.Context(argb)
        ctx.set_source_surface(src, 0, 0)
        ctx.paint()
        return argb

    def to_png(self, filename: str, objects: list[Drawable]):
        """Renders all drawable objects to a PNG file."""
        surface = cairo.ImageSurface(cairo.FORMAT_RGB24, self.img_width, self.img_height)
        self._render_to_surface(surface, objects)
        surface.write_to_png(filename)

    def to_rgba(self, objects: list[Drawable]) -> np.ndarray:
        """Renders all drawable objects and returns an H×W×4 RGBA uint8 array (no disk I/O).

        Reuses a pre-allocated surface to avoid per-frame memory allocation.
        """
        self._render_to_surface(self._frame_surface, objects)
        self._frame_surface.flush()  # ensure all Cairo rendering is complete
        buf = self._frame_surface.get_data()
        # Cairo ARGB32 stores bytes as [B, G, R, A] on little-endian; reorder to RGBA.
        arr = np.ndarray(shape=(self.img_height, self.img_width, 4), dtype=np.uint8, buffer=buf)
        return arr[:, :, [2, 1, 0, 3]].copy()

    def to_pdf(self, filename: str, objects: list[Drawable]) -> None:
        """Export the scene as a PDF (raster background embedded + vector overlays).

        The PDF page is sized in points at 1 pt per pixel so that coordinate
        arithmetic inside _render_to_surface works unchanged.  The resulting
        document is at 72 DPI in physical terms; all entity and text geometry
        is vector and stays crisp at any zoom level.
        """
        surface = cairo.PDFSurface(filename, self.img_width, self.img_height)
        self._render_to_surface(surface, objects, height=self.img_height)
        surface.finish()

    def to_svg(self, filename: str, objects: list[Drawable]) -> None:
        """Export the scene as an SVG (raster background embedded + vector overlays)."""
        surface = cairo.SVGSurface(filename, self.img_width, self.img_height)
        self._render_to_surface(surface, objects, height=self.img_height)
        surface.finish()

    def _render_to_surface(
        self, surface: cairo.Surface, objects: list[Drawable], height: int | None = None
    ):
        """Common rendering logic shared by to_png, to_rgba, to_pdf, and to_svg.

        height: pixel height for the Y-flip transform.  Defaults to
        surface.get_height() which works for ImageSurface but not for
        PDFSurface / SVGSurface — callers must pass self.img_height in those cases.
        """
        ctx = cairo.Context(surface)

        # Enable best antialiasing
        ctx.set_antialias(cairo.ANTIALIAS_BEST)

        # Draw background
        ctx.set_source_surface(self._background)
        ctx.paint()

        # Set coordinate system with origin at bottom-left
        h = height if height is not None else surface.get_height()
        ctx.translate(0, h)
        ctx.scale(1, -1)

        # Draw each object in z-order (lowest first = furthest back)
        for o in sorted(objects, key=lambda o: o.zorder):
            if isinstance(o, AWACS):
                self._draw_awacs(ctx, o)
            elif isinstance(o, Airplane):
                self._draw_airplane(ctx, o)
            elif isinstance(o, SamBattery):
                self._draw_sam_battery(ctx, o)
            elif isinstance(o, Waypoint):
                self._draw_waypoint(ctx, o)
            elif isinstance(o, Missile):
                self._draw_missile(ctx, o)
            elif isinstance(o, StatusMessage):
                self._draw_status_message(ctx, o)
            elif isinstance(o, TopLeftMessage):
                self._draw_top_left_message(ctx, o)
            elif isinstance(o, MapLabel):
                self._draw_map_label(ctx, o)
            elif isinstance(o, PolyLine):
                self._draw_poly_line(ctx, o)
            elif isinstance(o, Arc):
                self._draw_arc(ctx, o)
            elif isinstance(o, Rect):
                self._draw_rect(ctx, o)
            elif isinstance(o, RadarCone):
                self._draw_radar_cone(ctx, o)
            elif isinstance(o, ScaleBar):
                self._draw_scale_bar(ctx, o)
            else:
                raise RuntimeError(f"Can't draw object of type {type(o)}")

    @staticmethod
    def _get_image_angle(yaw_deg: float):
        return -yaw_deg / 180 * math.pi

    def _get_image_xya(plotter, lat: float, lon: float, yaw_deg: float):
        lat_rel = (lat - plotter.map_extents.bottom_lat) / (
            plotter.map_extents.top_lat - plotter.map_extents.bottom_lat
        )
        lon_rel = (lon - plotter.map_extents.left_lon) / (
            plotter.map_extents.right_lon - plotter.map_extents.left_lon
        )
        a = get_plotter_angle(yaw_deg)
        return lon_rel * plotter.img_width, lat_rel * plotter.img_height, a

    def _get_image_distance(self, dst_km: float):
        total_km_width = self.map_extents.max_longitude_extent_km()
        pixels_per_km = self.img_width / total_km_width
        scaled_distance = dst_km * pixels_per_km
        return scaled_distance

    # Symbol Rendering
    def _draw_symbol(self, ctx, x, y, angle, symbol_surface, info_text, edge_color, scale=None):
        """Generic symbol rendering: translate, rotate, blit cached surface.

        scale overrides cfg.symbol_scale when provided (used for missiles).
        """
        ctx.save()
        x = round(x * 2) / 2
        y = round(y * 2) / 2
        ctx.translate(x, y)
        if self.cfg.show_text and info_text:
            ctx.set_source_rgba(1.0, 1.0, 1.0, 1.0)
            self._draw_text(ctx, 0, self.cfg.sprites_info_spacing, info_text)
        ctx.rotate(angle)
        s = scale if scale is not None else self.cfg.symbol_scale
        w = symbol_surface.get_width()
        h = symbol_surface.get_height()
        ctx.translate(-w * s / 2, -h * s / 2)
        ctx.scale(s, -s)
        ctx.translate(0, -h)
        ctx.set_source_surface(symbol_surface, 0, 0)
        ctx.paint()
        ctx.restore()

    # Drawing Methods
    def _draw_status_message(self, ctx, o: StatusMessage):
        if not self.cfg.show_text:
            return

        msg = f"> {o.text}"
        ctx.save()
        ctx.select_font_face(
            self.cfg.status_message_font,
            self.cfg.status_message_font_style,
            self.cfg.status_message_font_weight,
        )
        ctx.set_font_matrix(
            cairo.Matrix(
                xx=self.cfg.status_message_font_size, yy=-self.cfg.status_message_font_size
            )
        )

        # Background panel for readability — baseline at y=8 in Y-up Cairo space,
        # glyphs extend upward by ~font_size pixels.
        _, _, text_w, _ = ctx.text_extents(msg)[:4]
        fs = self.cfg.status_message_font_size
        pad = 5
        ctx.set_source_rgba(0.0, 0.0, 0.0, 0.60)
        ctx.rectangle(10 - pad, 8 - 3, text_w + 2 * pad, fs + 6)
        ctx.fill()

        ctx.set_source_rgba(*o.text_color)
        ctx.move_to(10, 8)
        ctx.show_text(msg)
        ctx.new_path()
        ctx.restore()

    def _draw_top_left_message(self, ctx, o: TopLeftMessage):
        if not self.cfg.show_text:
            return

        msg = f"{o.text}"
        ctx.save()
        ctx.select_font_face(
            self.cfg.status_message_font,
            self.cfg.status_message_font_style,
            self.cfg.status_message_font_weight,
        )
        ctx.set_font_matrix(
            cairo.Matrix(
                xx=self.cfg.status_message_font_size, yy=-self.cfg.status_message_font_size
            )
        )

        lines = msg.split("\n")
        fs = self.cfg.status_message_font_size
        line_spacing = fs + 4

        # Measure all lines
        max_width = 0
        line_dimensions = []
        for line in lines:
            x_bearing, y_bearing, width, height = ctx.text_extents(line)[:4]
            line_dimensions.append((x_bearing, y_bearing, width, height))
            max_width = max(max_width, width)

        # Background panel covering all lines.
        # In Y-up Cairo space: last line (i=len-1) is at the highest y (near top of screen).
        # y_last ≈ img_height - fs - 5, y_first = y_last - (n-1)*line_spacing.
        pad = 5
        y_last = self.img_height - fs - 5
        y_first = y_last - (len(lines) - 1) * line_spacing
        ctx.set_source_rgba(0.0, 0.0, 0.0, 0.60)
        ctx.rectangle(
            self.img_width - max_width - 10 - pad,
            y_first - pad,
            max_width + 2 * pad,
            (len(lines) - 1) * line_spacing + fs + 6 + 2 * pad,
        )
        ctx.fill()

        # Draw each line right-aligned
        ctx.set_source_rgba(*o.text_color)
        for i, (line, (x_bearing, y_bearing, width, height)) in enumerate(
            zip(lines, line_dimensions)
        ):
            y_pos = self.img_height - height - 5 - (len(lines) - 1 - i) * line_spacing
            x_pos = self.img_width - width - 10
            ctx.move_to(x_pos, y_pos)
            ctx.show_text(line)

        ctx.new_path()
        ctx.restore()

    def _draw_text(self, ctx, x, y, text):
        if not self.cfg.show_text:
            return

        ctx.select_font_face(
            self.cfg.sprites_info_font,
            self.cfg.sprites_info_font_style,
            self.cfg.sprites_info_font_weight,
        )
        ctx.set_font_matrix(
            cairo.Matrix(xx=self.cfg.sprites_info_font_size, yy=-self.cfg.sprites_info_font_size)
        )
        lines = text.split("\n")
        line_spacing = self.cfg.sprites_info_font_size + 2
        total_height = len(lines) * line_spacing
        start_y = y - total_height / 2

        # Accumulate text paths for all lines
        for i, line in enumerate(lines):
            x_bearing, y_bearing, width, _ = ctx.text_extents(line)[:4]
            line_x = x - width / 2 - x_bearing
            line_y = start_y + i * line_spacing - y_bearing
            ctx.move_to(line_x, line_y)
            ctx.text_path(line)

        # Dark halo outline for readability on any background
        ctx.set_source_rgba(0.0, 0.0, 0.0, 0.75)
        ctx.set_line_width(2.5)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        ctx.stroke_preserve()
        # White glyph fill
        ctx.set_source_rgba(1.0, 1.0, 1.0, 1.0)
        ctx.fill()

    def _draw_map_label(self, ctx, o: MapLabel):
        if not self.cfg.show_text or not o.visible or not o.text:
            return
        x, y, _ = self._get_image_xya(o.lat, o.lon, 0)
        label_x = x + o.offset_x
        label_y = y + o.offset_y
        ctx.save()
        if o.draw_leader:
            ctx.set_source_rgba(0.8, 0.85, 0.9, 0.65)
            ctx.set_line_width(1.0)
            ctx.move_to(x, y)
            ctx.line_to(label_x, label_y)
            ctx.stroke()
        self._draw_text(ctx, label_x, label_y, o.text)
        ctx.restore()

    def _draw_airplane(self, ctx, o: Airplane):
        x, y, angle = self._get_image_xya(o.lat, o.lon, o.yaw_deg)

        if self.symbol_registry is not None and self.cfg.symbol_mode != "procedural":
            if self.cfg.symbol_mode == "flying_objects":
                side = _affiliation_to_side(o.affiliation)
                if self.symbol_registry.has_flying(o.aircraft_type, side):
                    surface = self.symbol_registry.get_flying(
                        o.aircraft_type, side, self.cfg.symbol_size
                    )
                    self._draw_symbol(ctx, x, y, angle, surface, o.info_text, o.edge_color)
                    return
            elif self.cfg.symbol_mode == "nato":
                if self.symbol_registry.has_nato("fighter", o.affiliation):
                    surface = self.symbol_registry.get_nato(
                        "fighter", o.affiliation, self.cfg.symbol_size
                    )
                    self._draw_symbol(ctx, x, y, angle, surface, o.info_text, o.edge_color)
                    return

        # Fallback: procedural drawing
        ctx.save()
        ctx.translate(x, y)
        if self.cfg.show_text and o.info_text:
            ctx.set_source_rgba(1.0, 1.0, 1.0, 1.0)  # Always white
            self._draw_text(ctx, 0, self.cfg.sprites_info_spacing, o.info_text)
        ctx.rotate(angle)
        ctx.set_source_rgba(*o.fill_color)
        ctx.set_line_width(2)
        # Draw airplane shape
        ctx.move_to(0.00 * self.cfg.units_scale, -0.38 * self.cfg.units_scale)
        ctx.line_to(0.06 * self.cfg.units_scale, -0.38 * self.cfg.units_scale)
        ctx.line_to(0.08 * self.cfg.units_scale, -0.31 * self.cfg.units_scale)
        ctx.line_to(0.28 * self.cfg.units_scale, -0.29 * self.cfg.units_scale)
        ctx.line_to(0.28 * self.cfg.units_scale, -0.19 * self.cfg.units_scale)
        ctx.line_to(0.09 * self.cfg.units_scale, 0.03 * self.cfg.units_scale)
        ctx.line_to(0.09 * self.cfg.units_scale, 0.05 * self.cfg.units_scale)
        ctx.line_to(0.13 * self.cfg.units_scale, 0.04 * self.cfg.units_scale)
        ctx.line_to(0.13 * self.cfg.units_scale, 0.08 * self.cfg.units_scale)
        ctx.line_to(0.05 * self.cfg.units_scale, 0.15 * self.cfg.units_scale)
        ctx.line_to(0.0 * self.cfg.units_scale, 0.44 * self.cfg.units_scale)
        ctx.line_to(-0.05 * self.cfg.units_scale, 0.15 * self.cfg.units_scale)
        ctx.line_to(-0.13 * self.cfg.units_scale, 0.08 * self.cfg.units_scale)
        ctx.line_to(-0.13 * self.cfg.units_scale, 0.04 * self.cfg.units_scale)
        ctx.line_to(-0.09 * self.cfg.units_scale, 0.05 * self.cfg.units_scale)
        ctx.line_to(-0.09 * self.cfg.units_scale, 0.03 * self.cfg.units_scale)
        ctx.line_to(-0.28 * self.cfg.units_scale, -0.19 * self.cfg.units_scale)
        ctx.line_to(-0.28 * self.cfg.units_scale, -0.29 * self.cfg.units_scale)
        ctx.line_to(-0.08 * self.cfg.units_scale, -0.31 * self.cfg.units_scale)
        ctx.line_to(-0.06 * self.cfg.units_scale, -0.38 * self.cfg.units_scale)
        ctx.close_path()
        ctx.fill_preserve()
        ctx.set_source_rgba(*o.edge_color)
        ctx.stroke()
        ctx.move_to(0.1 * self.cfg.units_scale, 0.0 * self.cfg.units_scale)
        ctx.line_to(-0.1 * self.cfg.units_scale, 0.0 * self.cfg.units_scale)
        ctx.move_to(0.0 * self.cfg.units_scale, 0.1 * self.cfg.units_scale)
        ctx.line_to(0.0 * self.cfg.units_scale, -0.1 * self.cfg.units_scale)
        ctx.stroke()
        ctx.restore()

    def _draw_awacs(self, ctx, o: AWACS):
        """Draw AWACS with distinctive rotodome (rotating radar disc)."""
        x, y, angle = self._get_image_xya(o.lat, o.lon, o.yaw_deg)

        if self.symbol_registry is not None and self.cfg.symbol_mode != "procedural":
            if self.cfg.symbol_mode == "flying_objects":
                side = _affiliation_to_side(o.affiliation)
                if self.symbol_registry.has_flying(o.aircraft_type, side):
                    surface = self.symbol_registry.get_flying(
                        o.aircraft_type, side, self.cfg.symbol_size
                    )
                    self._draw_symbol(ctx, x, y, angle, surface, o.info_text, o.edge_color)
                    return
            elif self.cfg.symbol_mode == "nato":
                if self.symbol_registry.has_nato("awacs", o.affiliation):
                    surface = self.symbol_registry.get_nato(
                        "awacs", o.affiliation, self.cfg.symbol_size
                    )
                    self._draw_symbol(ctx, x, y, angle, surface, o.info_text, o.edge_color)
                    return

        # Fallback: procedural drawing
        ctx.save()
        ctx.translate(x, y)
        if self.cfg.show_text and o.info_text:
            ctx.set_source_rgba(1.0, 1.0, 1.0, 1.0)  # Always white
            self._draw_text(ctx, 0, self.cfg.sprites_info_spacing, o.info_text)
        ctx.rotate(angle)
        ctx.set_source_rgba(*o.fill_color)
        ctx.set_line_width(2)

        # Draw larger aircraft body (AWACS are typically large aircraft like Boeing 707)
        ctx.move_to(0.00 * self.cfg.units_scale, -0.45 * self.cfg.units_scale)
        ctx.line_to(0.10 * self.cfg.units_scale, -0.45 * self.cfg.units_scale)
        ctx.line_to(0.12 * self.cfg.units_scale, -0.35 * self.cfg.units_scale)
        ctx.line_to(0.35 * self.cfg.units_scale, -0.33 * self.cfg.units_scale)
        ctx.line_to(0.35 * self.cfg.units_scale, -0.22 * self.cfg.units_scale)
        ctx.line_to(0.12 * self.cfg.units_scale, 0.10 * self.cfg.units_scale)
        ctx.line_to(0.12 * self.cfg.units_scale, 0.15 * self.cfg.units_scale)
        ctx.line_to(0.15 * self.cfg.units_scale, 0.14 * self.cfg.units_scale)
        ctx.line_to(0.15 * self.cfg.units_scale, 0.20 * self.cfg.units_scale)
        ctx.line_to(0.08 * self.cfg.units_scale, 0.25 * self.cfg.units_scale)
        ctx.line_to(0.0 * self.cfg.units_scale, 0.50 * self.cfg.units_scale)
        ctx.line_to(-0.08 * self.cfg.units_scale, 0.25 * self.cfg.units_scale)
        ctx.line_to(-0.15 * self.cfg.units_scale, 0.20 * self.cfg.units_scale)
        ctx.line_to(-0.15 * self.cfg.units_scale, 0.14 * self.cfg.units_scale)
        ctx.line_to(-0.12 * self.cfg.units_scale, 0.15 * self.cfg.units_scale)
        ctx.line_to(-0.12 * self.cfg.units_scale, 0.10 * self.cfg.units_scale)
        ctx.line_to(-0.35 * self.cfg.units_scale, -0.22 * self.cfg.units_scale)
        ctx.line_to(-0.35 * self.cfg.units_scale, -0.33 * self.cfg.units_scale)
        ctx.line_to(-0.12 * self.cfg.units_scale, -0.35 * self.cfg.units_scale)
        ctx.line_to(-0.10 * self.cfg.units_scale, -0.45 * self.cfg.units_scale)
        ctx.close_path()
        ctx.fill_preserve()
        ctx.set_source_rgba(*o.edge_color)
        ctx.stroke()

        # Draw distinctive rotodome (rotating radar disc on top of fuselage)
        ctx.save()
        ctx.translate(0.0, -0.05 * self.cfg.units_scale)  # Position above fuselage
        ctx.scale(1.2, 0.4)  # Make it an oval
        ctx.arc(0.0, 0.0, 0.18 * self.cfg.units_scale, 0, 2 * math.pi)
        ctx.restore()
        ctx.set_source_rgba(*o.fill_color)
        ctx.fill_preserve()
        ctx.set_source_rgba(*o.edge_color)
        ctx.set_line_width(2)
        ctx.stroke()

        # Add center cross for reference
        ctx.set_line_width(1)
        ctx.move_to(0.12 * self.cfg.units_scale, 0.0 * self.cfg.units_scale)
        ctx.line_to(-0.12 * self.cfg.units_scale, 0.0 * self.cfg.units_scale)
        ctx.move_to(0.0 * self.cfg.units_scale, 0.12 * self.cfg.units_scale)
        ctx.line_to(0.0 * self.cfg.units_scale, -0.12 * self.cfg.units_scale)
        ctx.stroke()
        ctx.restore()

    def _draw_sam_battery(self, ctx, o: SamBattery):
        x, y, angle = self._get_image_xya(o.lat, o.lon, o.yaw_deg)

        # SAMs always use NATO symbols regardless of the active mode
        use_symbol = self.symbol_registry is not None and self.symbol_registry.has_nato(
            "sam", o.affiliation
        )

        ctx.save()
        ctx.translate(x, y)
        if self.cfg.show_text and o.info_text:
            ctx.set_source_rgba(*o.edge_color)
            self._draw_text(ctx, 0, -self.cfg.sprites_info_spacing, o.info_text)
        ctx.rotate(angle)

        if use_symbol:
            symbol_surface = self.symbol_registry.get_nato(
                "sam", o.affiliation, self.cfg.symbol_size
            )
            w = symbol_surface.get_width()
            h = symbol_surface.get_height()
            ctx.save()
            ctx.translate(-w / 2, -h / 2)
            ctx.scale(1, -1)
            ctx.translate(0, -h)
            ctx.set_source_surface(symbol_surface, 0, 0)
            ctx.paint()
            ctx.restore()
        else:
            # Fallback: procedural square
            ctx.set_source_rgba(*o.fill_color)
            ctx.set_line_width(2)
            ctx.move_to(0.15 * self.cfg.units_scale, -0.15 * self.cfg.units_scale)
            ctx.line_to(0.15 * self.cfg.units_scale, 0.15 * self.cfg.units_scale)
            ctx.line_to(-0.15 * self.cfg.units_scale, 0.15 * self.cfg.units_scale)
            ctx.line_to(-0.15 * self.cfg.units_scale, -0.15 * self.cfg.units_scale)
            ctx.close_path()
            ctx.fill_preserve()
            ctx.set_source_rgba(*o.edge_color)
            ctx.stroke()
            ctx.arc(0.0, 0.0, 0.4 * self.cfg.units_scale, -1 + math.pi / 2.0, 1 + math.pi / 2.0)
            ctx.new_sub_path()
            ctx.arc(0.0, 0.0, 0.6 * self.cfg.units_scale, -1 + math.pi / 2.0, 1 + math.pi / 2.0)
            ctx.stroke()
            ctx.move_to(0.1 * self.cfg.units_scale, 0.0 * self.cfg.units_scale)
            ctx.line_to(-0.1 * self.cfg.units_scale, 0.0 * self.cfg.units_scale)
            ctx.move_to(0.0 * self.cfg.units_scale, 0.1 * self.cfg.units_scale)
            ctx.line_to(0.0 * self.cfg.units_scale, -0.1 * self.cfg.units_scale)
            ctx.stroke()

        # Always draw tactical range overlays
        ctx.set_source_rgba(*o.edge_color)
        ctx.set_line_width(1)
        ctx.set_dash([5, 5])
        ctx.move_to(0, 0)
        ctx.arc(
            0.0,
            0.0,
            self.pixels_km * o.radar_range_km,
            math.pi / 2.0 - o.radar_amplitude_deg / 180 * math.pi,
            math.pi / 2.0,
        )
        ctx.close_path()
        ctx.stroke()
        ctx.arc(0.0, 0.0, self.pixels_km * o.missile_range_km, 0, 2.0 * math.pi)
        ctx.stroke()
        ctx.restore()

    def _draw_waypoint(self, ctx, o: Waypoint):
        x, y, angle = self._get_image_xya(o.lat, o.lon, o.yaw_deg)
        ctx.save()
        ctx.translate(x, y)
        if self.cfg.show_text and o.info_text:
            ctx.set_source_rgba(*o.edge_color)
            self._draw_text(ctx, 0, -self.cfg.sprites_info_spacing, o.info_text)
        ctx.rotate(angle)
        ctx.set_line_width(1)
        ctx.new_path()
        ctx.arc(0.0, 0.0, 0.1 * self.cfg.units_scale, 0, 2 * math.pi)
        ctx.close_path()
        ctx.set_source_rgba(*o.fill_color)
        ctx.fill_preserve()
        ctx.set_source_rgba(*o.edge_color)
        ctx.stroke()
        ctx.move_to(0.0, 0.1 * self.cfg.units_scale)
        ctx.line_to(0.0, 0.25 * self.cfg.units_scale)
        ctx.stroke()
        ctx.move_to(0.0, 0.25 * self.cfg.units_scale)
        ctx.line_to(0.0, 0.45 * self.cfg.units_scale)
        ctx.line_to(0.25 * self.cfg.units_scale, 0.35 * self.cfg.units_scale)
        ctx.close_path()
        ctx.set_source_rgba(*o.fill_color)
        ctx.fill_preserve()
        ctx.set_source_rgba(*o.edge_color)
        ctx.stroke()
        ctx.restore()

    def _draw_missile(self, ctx, o: Missile):
        x, y, angle = self._get_image_xya(o.lat, o.lon, o.yaw_deg)

        # Missiles always use flying_objects symbols regardless of the active mode.
        # The missile SVG is landscape (nose points left), so subtract π/2 to align
        # the nose with the actual heading.
        if self.symbol_registry is not None:
            side = _affiliation_to_side(o.affiliation)
            if self.symbol_registry.has_flying("missile", side):
                surface = self.symbol_registry.get_flying(
                    "missile", side, self.cfg.missile_symbol_size
                )
                self._draw_symbol(
                    ctx,
                    x,
                    y,
                    angle - math.pi / 2,
                    surface,
                    o.info_text,
                    o.edge_color,
                    scale=self.cfg.missile_symbol_scale,
                )
                return

        # Fallback: procedural drawing
        ctx.save()
        ctx.translate(x, y)
        if self.cfg.show_text and o.info_text:
            ctx.set_source_rgba(1.0, 1.0, 1.0, 1.0)
            self._draw_text(ctx, 0, -self.cfg.sprites_info_spacing, o.info_text)
        ctx.rotate(angle)
        ctx.set_line_width(1)
        ctx.move_to(0.05 * self.cfg.units_scale, -0.3 * self.cfg.units_scale)
        ctx.line_to(0.07 * self.cfg.units_scale, 0.1 * self.cfg.units_scale)
        ctx.line_to(0.0 * self.cfg.units_scale, 0.4 * self.cfg.units_scale)
        ctx.line_to(-0.07 * self.cfg.units_scale, 0.1 * self.cfg.units_scale)
        ctx.line_to(-0.05 * self.cfg.units_scale, -0.3 * self.cfg.units_scale)
        ctx.close_path()
        ctx.set_source_rgba(*o.fill_color)
        ctx.fill_preserve()
        ctx.set_source_rgba(*o.edge_color)
        ctx.stroke()
        ctx.restore()

    def _draw_poly_line(self, ctx, o: PolyLine):
        if len(o.points) <= 1:
            return
        ctx.save()
        ctx.set_line_width(o.line_width)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        if o.dash:
            ctx.set_dash(o.dash)
            ctx.set_line_cap(cairo.LINE_CAP_BUTT)
        else:
            ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_source_rgba(*o.edge_color)
        ctx.new_path()
        x, y, _ = self._get_image_xya(o.points[0][0], o.points[0][1], 0)
        ctx.move_to(x, y)
        for i in range(1, len(o.points)):
            x, y, _ = self._get_image_xya(o.points[i][0], o.points[i][1], 0)
            ctx.line_to(x, y)
        ctx.stroke()
        ctx.restore()

    def _draw_rect(self, ctx, o: Rect):
        ctx.save()
        left, top, _ = self._get_image_xya(o.top_lat, o.left_lon, 0)
        right, bottom, _ = self._get_image_xya(o.bottom_lat, o.right_lon, 0)
        ctx.rectangle(left, top, right - left, bottom - top)
        ctx.set_line_width(o.line_width)
        ctx.set_source_rgba(*o.fill_color)
        ctx.fill_preserve()
        ctx.set_source_rgba(*o.edge_color)
        ctx.stroke()
        ctx.restore()

    def _draw_arc(self, ctx, o: Arc):
        ctx.save()
        x, y, _ = self._get_image_xya(o.center_lat, o.center_lon, 0)
        radius_px = self._get_image_distance(o.radius)
        a1 = math.radians(o.angle1 - 90)
        a2 = math.radians(o.angle2 - 90)
        ctx.new_path()
        ctx.arc(x, y, radius_px, min(a1, a2), max(a1, a2))
        ctx.set_line_width(o.line_width)
        if o.fill_color:
            ctx.set_source_rgba(*o.fill_color)
            ctx.fill_preserve()
        if o.edge_color:
            ctx.set_source_rgba(*o.edge_color)
            ctx.stroke()
        ctx.restore()

    def _draw_radar_cone(self, ctx, o: RadarCone):
        x, y, _ = self._get_image_xya(o.center_lat, o.center_lon, 0)
        radius_px = self._get_image_distance(o.radius)
        yaw_deg_offset = 90
        a1 = math.radians(-o.angle1 + yaw_deg_offset)
        a2 = math.radians(-o.angle2 + yaw_deg_offset)
        full_circle = abs(float(o.angle2) - float(o.angle1)) >= 360.0 - 1e-6

        ctx.save()
        ctx.new_path()
        if full_circle:
            # Drawing a sector for 360 degrees adds a center-to-edge radial line
            # at 0/360. A circle has the same filled area without that seam.
            ctx.arc(x, y, radius_px, 0.0, 2.0 * math.pi)
        else:
            ctx.move_to(x, y)
            ctx.arc_negative(x, y, radius_px, a1, a2)
            ctx.line_to(x, y)
        ctx.close_path()
        if o.fill_color:
            ctx.set_source_rgba(*o.fill_color)
            ctx.fill_preserve()
        if o.edge_color:
            less_visible_edge = (
                o.edge_color[0],
                o.edge_color[1],
                o.edge_color[2],
                o.edge_color[3] * 0.3,
            )
            ctx.set_source_rgba(*less_visible_edge)
            ctx.set_line_width(o.line_width)
            ctx.stroke()
        ctx.restore()

    def _draw_scale_bar(self, ctx, o: ScaleBar):
        length_px = self._get_image_distance(o.length_km)
        font_size = self.cfg.sprites_info_font_size
        tick_h = max(4, font_size // 2)

        ctx.save()
        ctx.set_line_width(o.line_width)
        ctx.set_source_rgba(*o.edge_color)

        start_x = o.x - length_px / 2
        end_x = o.x + length_px / 2

        # Horizontal bar
        ctx.move_to(start_x, o.y)
        ctx.line_to(end_x, o.y)
        ctx.stroke()

        # End ticks
        for tx in (start_x, end_x):
            ctx.move_to(tx, o.y - tick_h)
            ctx.line_to(tx, o.y + tick_h)
            ctx.stroke()

        # Label — the outer context has Y going UP (global scale(1,-1) in to_png).
        # We do scale(1,-1) here to return to normal screen-space (Y going down),
        # so yy in the font matrix must be POSITIVE to avoid a second flip.
        if self.cfg.show_text:
            label = f"{o.length_km} km"
            ctx.save()
            ctx.scale(1, -1)
            ctx.select_font_face(
                self.cfg.sprites_info_font,
                self.cfg.sprites_info_font_style,
                cairo.FONT_WEIGHT_BOLD,
            )
            ctx.set_font_matrix(cairo.Matrix(xx=font_size, yy=font_size))
            x_bearing, _, text_w, _ = ctx.text_extents(label)[:4]
            # In flipped space: negate y to move "above" the bar in screen coords
            ctx.move_to(o.x - text_w / 2 - x_bearing, -(o.y + tick_h + font_size + 2))
            ctx.show_text(label)
            ctx.restore()

        ctx.restore()
