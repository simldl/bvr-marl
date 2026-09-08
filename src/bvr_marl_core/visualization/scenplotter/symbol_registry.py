"""Load pre-generated tactical symbols as cached Cairo surfaces."""

from pathlib import Path

import cairo
from PIL import Image


class SymbolRegistry:
    """Loads PNG symbols, converts to Cairo ImageSurface, caches by lookup key."""

    _DEFAULT_DIR = Path(__file__).resolve().parent.parent / "symbols"
    _AVAILABLE_SIZES = (20, 28, 32, 40, 64, 96, 128)
    _FIGHTER_TYPES = frozenset({"f22", "f35", "eurofighter", "su57", "f15ex"})

    def __init__(self, symbols_dir: str | Path | None = None, mode: str = "nato"):
        """
        Args:
            symbols_dir: Root symbols directory (contains png/nato/, png/flying_objects/).
            mode: "nato" or "flying_objects" — determines which source is used for aircraft.
        """
        self.symbols_dir = Path(symbols_dir) if symbols_dir else self._DEFAULT_DIR
        self.mode = mode
        self._cache: dict[tuple, cairo.ImageSurface] = {}
        self._data_refs: dict[tuple, bytearray] = {}

    @property
    def _nato_dir(self) -> Path:
        return self.symbols_dir / "png" / "nato"

    @property
    def _flying_dir(self) -> Path:
        return self.symbols_dir / "png" / "flying_objects"

    def has_nato(self, symbol_id: str, affiliation: str) -> bool:
        """Return True if a PNG exists for this NATO symbol/affiliation."""
        for s in self._AVAILABLE_SIZES:
            if (self._nato_dir / f"{symbol_id}_{affiliation}_{s}.png").exists():
                return True
        return False

    def get_nato(self, symbol_id: str, affiliation: str, size: int = 32) -> cairo.ImageSurface:
        """Return a Cairo surface for a NATO symbol (MIL-STD-2525C frame)."""
        key = ("nato", symbol_id, affiliation, size)
        if key not in self._cache:
            png_size = self._best_size(size)
            png_path = self._nato_dir / f"{symbol_id}_{affiliation}_{png_size}.png"
            if not png_path.exists():
                raise FileNotFoundError(
                    f"NATO symbol PNG not found: {png_path}\n"
                    f"Run 'node visualization/symbols/generate_symbols.js' to generate."
                )
            surface, data = self._load_png(png_path, size)
            self._cache[key] = surface
            self._data_refs[key] = data
        return self._cache[key]

    def has_flying(self, aircraft_type: str, side: str) -> bool:
        """Return True if a PNG exists for this aircraft type and side."""
        for s in self._AVAILABLE_SIZES:
            if (self._flying_dir / f"{aircraft_type}_{side}_{s}.png").exists():
                return True
        return False

    def get_flying(self, aircraft_type: str, side: str, size: int = 32) -> cairo.ImageSurface:
        """Return a Cairo surface for a flying-object silhouette."""
        key = ("flying", aircraft_type, side, size)
        if key not in self._cache:
            png_size = self._best_size(size)
            png_path = self._flying_dir / f"{aircraft_type}_{side}_{png_size}.png"
            if not png_path.exists():
                raise FileNotFoundError(
                    f"Flying-object symbol PNG not found: {png_path}\n"
                    f"Run 'node visualization/symbols/generate_symbols.js' to generate."
                )
            normalized_size = size
            is_fighter = aircraft_type.lower() in self._FIGHTER_TYPES
            if is_fighter:
                normalized_size = self._eurofighter_visible_size(side, png_size, size)
            surface, data = self._load_png(png_path, normalized_size, normalize_visible=is_fighter)
            self._cache[key] = surface
            self._data_refs[key] = data
        return self._cache[key]

    def available_symbols(self) -> list[str]:
        """List available PNG symbol stems across both source directories."""
        results = []
        for d in [self._nato_dir, self._flying_dir]:
            if d.exists():
                results.extend(f.stem for f in d.glob("*.png"))
        return results

    def _best_size(self, requested: int) -> int:
        """Pick the smallest pre-generated PNG size that is >= requested."""
        for s in self._AVAILABLE_SIZES:
            if s >= requested:
                return s
        return self._AVAILABLE_SIZES[-1]

    def _eurofighter_visible_size(self, side: str, png_size: int, requested: int) -> int:
        """Scale fighters to the original Eurofighter's occupied-pixel extent."""
        reference = self._flying_dir / f"eurofighter_{side}_{png_size}.png"
        if not reference.exists():
            return requested
        with Image.open(reference) as source:
            bounds = source.convert("RGBA").getchannel("A").getbbox()
        if bounds is None:
            return requested
        visible_extent = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
        return max(1, round(visible_extent * requested / png_size))

    def _load_png(
        self, png_path: Path, size: int, *, normalize_visible: bool = False
    ) -> tuple[cairo.ImageSurface, bytearray]:
        """Load a PNG, optionally rescale to size, return (Cairo surface, backing buffer).

        Converts RGBA → Cairo ARGB32 (pre-multiplied alpha, BGRA byte order) using
        NumPy vectorised operations instead of a Python pixel loop.
        """
        import numpy as np

        img = Image.open(png_path).convert("RGBA")

        # Source fighter artwork has very different transparent margins (for
        # example, the old 64 px Eurofighter occupies only 27x38 px while the
        # Su-57 occupies 46x61 px).  Crop those margins before scaling so every
        # fighter's *visible* longest dimension equals the requested symbol size.
        if normalize_visible:
            visible_bounds = img.getchannel("A").getbbox()
            if visible_bounds is not None:
                img = img.crop(visible_bounds)

        actual = max(img.width, img.height)
        if actual != size:
            ratio = size / actual
            new_w = max(1, int(img.width * ratio))
            new_h = max(1, int(img.height * ratio))
            img = img.resize((new_w, new_h), Image.LANCZOS)

        w, h = img.size
        rgba = np.array(img, dtype=np.float32)
        alpha_norm = rgba[:, :, 3] / 255.0

        pm = (rgba[:, :, :3] * alpha_norm[:, :, np.newaxis]).astype(np.uint8)

        bgra = np.empty((h, w, 4), dtype=np.uint8)
        bgra[:, :, 0] = pm[:, :, 2]
        bgra[:, :, 1] = pm[:, :, 1]
        bgra[:, :, 2] = pm[:, :, 0]
        bgra[:, :, 3] = rgba[:, :, 3].astype(np.uint8)

        data = bytearray(bgra.tobytes())
        surface = cairo.ImageSurface.create_for_data(data, cairo.FORMAT_ARGB32, w, h, w * 4)
        return surface, data
