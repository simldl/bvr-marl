#!/usr/bin/env python3
"""
TensorBoard plot export launcher — thin wrapper around air2air-export-plots.

All argument handling is in the package entrypoint.
See: air2air-export-plots --help
"""

from air_to_air_rl.analysis.export_plots import main

if __name__ == "__main__":
    exit(main())
