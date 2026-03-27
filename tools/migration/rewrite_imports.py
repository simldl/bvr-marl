"""
Import rewriting script for Phase 1 namespace consolidation.

Rewrites all legacy top-level package imports to use the
unified air_to_air_rl namespace.
"""

import re
import sys
from pathlib import Path

# Mapping: (old_prefix, new_prefix)
# Order matters — more specific patterns first
REPLACEMENTS = [
    # reinforcement_learning → rl (must come before shorter matches)
    ("from reinforcement_learning.", "from air_to_air_rl.rl."),
    ("import reinforcement_learning.", "import air_to_air_rl.rl."),
    ("import reinforcement_learning\n", "import air_to_air_rl.rl\n"),
    ("import reinforcement_learning ", "import air_to_air_rl.rl "),
    # Core simulation packages
    ("from aircrafts.", "from air_to_air_rl.aircrafts."),
    ("import aircrafts.", "import air_to_air_rl.aircrafts."),
    ("from missiles.", "from air_to_air_rl.missiles."),
    ("import missiles.", "import air_to_air_rl.missiles."),
    ("from physics.", "from air_to_air_rl.physics."),
    ("import physics.", "import air_to_air_rl.physics."),
    ("from radar.", "from air_to_air_rl.radar."),
    ("import radar.", "import air_to_air_rl.radar."),
    ("from simulator.", "from air_to_air_rl.simulator."),
    ("import simulator.", "import air_to_air_rl.simulator."),
    # Shared utilities
    ("from utils.", "from air_to_air_rl.utils."),
    ("import utils.", "import air_to_air_rl.utils."),
    # Tacview
    ("from tacview.", "from air_to_air_rl.tacview."),
    ("import tacview.", "import air_to_air_rl.tacview."),
    # Automation
    ("from automation.", "from air_to_air_rl.automation."),
    ("import automation.", "import air_to_air_rl.automation."),
    # GUI
    ("from gui.", "from air_to_air_rl.gui."),
    ("import gui.", "import air_to_air_rl.gui."),
    # Visualization
    ("from visualization.", "from air_to_air_rl.visualization."),
    ("import visualization.", "import air_to_air_rl.visualization."),
    # patch() string literals in tests — same mapping applied to quoted module paths
    ("'simulator.", "'air_to_air_rl.simulator."),
    ('"simulator.', '"air_to_air_rl.simulator.'),
    ("'aircrafts.", "'air_to_air_rl.aircrafts."),
    ('"aircrafts.', '"air_to_air_rl.aircrafts.'),
    ("'missiles.", "'air_to_air_rl.missiles."),
    ('"missiles.', '"air_to_air_rl.missiles.'),
    ("'physics.", "'air_to_air_rl.physics."),
    ('"physics.', '"air_to_air_rl.physics.'),
    ("'radar.", "'air_to_air_rl.radar."),
    ('"radar.', '"air_to_air_rl.radar.'),
    ("'reinforcement_learning.", "'air_to_air_rl.rl."),
    ('"reinforcement_learning.', '"air_to_air_rl.rl.'),
    ("'automation.", "'air_to_air_rl.automation."),
    ('"automation.', '"air_to_air_rl.automation.'),
    ("'tacview.", "'air_to_air_rl.tacview."),
    ('"tacview.', '"air_to_air_rl.tacview.'),
    ("'visualization.", "'air_to_air_rl.visualization."),
    ('"visualization.', '"air_to_air_rl.visualization.'),
    ("'utils.", "'air_to_air_rl.utils."),
    ('"utils.', '"air_to_air_rl.utils.'),
    ("'gui.", "'air_to_air_rl.gui."),
    ('"gui.', '"air_to_air_rl.gui.'),
]

# File extensions to process
EXTENSIONS = {".py"}

# Directories to skip
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
    "src/air_to_air_rl.egg-info",
}


def should_skip(path: Path) -> bool:
    for part in path.parts:
        if part in SKIP_DIRS:
            return True
    return False


def rewrite_file(path: Path, dry_run: bool = False) -> bool:
    """Return True if file was modified."""
    try:
        original = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, PermissionError):
        return False

    content = original
    for old, new in REPLACEMENTS:
        content = content.replace(old, new)

    if content == original:
        return False

    if not dry_run:
        path.write_text(content, encoding="utf-8")
    return True


def main():
    dry_run = "--dry-run" in sys.argv
    root = Path(__file__).parent.parent.parent  # project root

    search_dirs = [
        root / "src",
        root / "tests",
        root / "scripts",
    ]

    changed = []
    for search_dir in search_dirs:
        for path in search_dir.rglob("*"):
            if path.suffix not in EXTENSIONS:
                continue
            if should_skip(path):
                continue
            if rewrite_file(path, dry_run=dry_run):
                changed.append(path.relative_to(root))

    print(f"{'[DRY RUN] Would modify' if dry_run else 'Modified'} {len(changed)} files:")
    for f in sorted(changed):
        print(f"  {f}")


if __name__ == "__main__":
    main()
