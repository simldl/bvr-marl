"""Command-line interface for simulator validation studies."""

from __future__ import annotations

import argparse
from pathlib import Path

from bvr_marl_core.validation.studies import STUDIES, run_studies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/validation/results"))
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--study", action="append", choices=sorted(STUDIES))
    args = parser.parse_args()
    manifest = run_studies(args.output, args.study, args.seed)
    for name, result in manifest["results"].items():
        print(f"{name}: {result}")


if __name__ == "__main__":
    main()
