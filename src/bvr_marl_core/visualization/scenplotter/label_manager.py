"""Collision-aware label placement for the raster-backed 2D tactical map."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class _Plotter(Protocol):
    img_width: int
    img_height: int
    cfg: object

    def _get_image_xya(self, lat: float, lon: float, yaw_deg: float): ...


@dataclass
class _Box:
    left: float
    bottom: float
    right: float
    top: float

    def intersects(self, other: _Box, padding: float = 4.0) -> bool:
        return not (
            self.right + padding < other.left
            or other.right + padding < self.left
            or self.top + padding < other.bottom
            or other.top + padding < self.bottom
        )


class FighterLabelManager:
    """Place high-priority labels first and move lower-priority labels around them."""

    def __init__(self, *, cluster_distance_px: float = 34.0):
        self.cluster_distance_px = float(cluster_distance_px)

    def layout(self, labels: list, plotter: _Plotter) -> list:
        """Mutate label offsets/visibility and return the visible labels."""
        if not labels:
            return labels
        for label in labels:
            label.visible = bool(label.text)
            label.cluster_count = 1

        self._collapse_dense_low_priority_groups(labels, plotter)
        occupied: list[_Box] = []
        font_size = float(getattr(plotter.cfg, "sprites_info_font_size", 10))
        spacing = float(getattr(plotter.cfg, "sprites_info_spacing", 26))
        offsets = (
            (spacing, spacing),
            (-spacing, spacing),
            (spacing, -spacing),
            (-spacing, -spacing),
            (0.0, spacing * 1.5),
            (0.0, -spacing * 1.5),
            (spacing * 1.7, 0.0),
            (-spacing * 1.7, 0.0),
        )
        ordered = sorted(
            (label for label in labels if label.visible),
            key=lambda label: (-int(label.priority), str(label.unit_id)),
        )
        for label in ordered:
            x, y, _ = plotter._get_image_xya(label.lat, label.lon, 0)
            width, height = self._estimate_size(label.text, font_size)
            chosen = offsets[0]
            chosen_box = None
            for dx, dy in offsets:
                box = self._box(x + dx, y + dy, width, height, plotter)
                if not any(box.intersects(other) for other in occupied):
                    chosen = (dx, dy)
                    chosen_box = box
                    break
            if chosen_box is None:
                lane_x = spacing if x < plotter.img_width / 2 else -spacing
                lane_step = (len(occupied) % 7) - 3
                chosen = (lane_x * 2.2, lane_step * (font_size + 6))
                chosen_box = self._box(x + chosen[0], y + chosen[1], width, height, plotter)
            label.offset_x, label.offset_y = chosen
            label.draw_leader = abs(chosen[0]) > 1 or abs(chosen[1]) > spacing * 1.1
            occupied.append(chosen_box)
        return [label for label in labels if label.visible]

    def _collapse_dense_low_priority_groups(self, labels: list, plotter: _Plotter) -> None:
        low = [label for label in labels if label.visible and int(label.priority) <= 0]
        consumed: set[int] = set()
        for index, label in enumerate(low):
            if index in consumed:
                continue
            x, y, _ = plotter._get_image_xya(label.lat, label.lon, 0)
            group = [(index, label)]
            for other_index in range(index + 1, len(low)):
                if other_index in consumed:
                    continue
                other = low[other_index]
                if other.affiliation != label.affiliation:
                    continue
                ox, oy, _ = plotter._get_image_xya(other.lat, other.lon, 0)
                if (ox - x) ** 2 + (oy - y) ** 2 <= self.cluster_distance_px**2:
                    group.append((other_index, other))
            if len(group) < 3:
                continue
            for other_index, other in group[1:]:
                consumed.add(other_index)
                other.visible = False
            label.cluster_count = len(group)
            label.text = f"{len(group)}× {label.cluster_name}"

    @staticmethod
    def _estimate_size(text: str, font_size: float) -> tuple[float, float]:
        lines = str(text).splitlines() or [""]
        return (
            max(len(line) for line in lines) * font_size * 0.58 + 10,
            len(lines) * (font_size + 2) + 6,
        )

    @staticmethod
    def _box(x: float, y: float, width: float, height: float, plotter: _Plotter) -> _Box:
        half_w = width / 2
        half_h = height / 2
        cx = min(max(x, half_w + 3), plotter.img_width - half_w - 3)
        cy = min(max(y, half_h + 3), plotter.img_height - half_h - 3)
        return _Box(cx - half_w, cy - half_h, cx + half_w, cy + half_h)


__all__ = ["FighterLabelManager"]
