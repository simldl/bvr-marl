"""Generate every figure under scripts/graphics into scripts/graphics/output.

Each plot script is run in its own subprocess so matplotlib state never leaks
between figures and one failure does not abort the rest.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

GRAPHICS_DIR = Path(__file__).resolve().parent
CATEGORIES = ("physics", "radar", "aircraft")
SKIP = {"paper_style.py", "sqi_model.py", "generate_all.py", "__init__.py"}


def main() -> None:
    scripts = [
        p
        for cat in CATEGORIES
        for p in sorted((GRAPHICS_DIR / cat).glob("*.py"))
        if p.name not in SKIP
    ]

    failures = []
    for script in scripts:
        rel = script.relative_to(GRAPHICS_DIR)
        print(f"\n=== {rel} ===")
        if subprocess.run([sys.executable, str(script)], check=False).returncode != 0:
            failures.append(str(rel))

    print("\n" + "=" * 50)
    if failures:
        print(f"{len(failures)} script(s) FAILED: {', '.join(failures)}")
        sys.exit(1)
    print(f"All {len(scripts)} figures written to {GRAPHICS_DIR / 'output'}")


if __name__ == "__main__":
    main()
