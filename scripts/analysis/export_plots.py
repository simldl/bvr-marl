#!/usr/bin/env python3
"""
TensorBoard plot export launcher — thin wrapper around bvr-export-plots.

All argument handling is in the package entrypoint.
See: bvr-export-plots --help
"""

from bvr_marl_core.analysis.export_plots import main

if __name__ == "__main__":
    exit(main())
