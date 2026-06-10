#!/usr/bin/env python3
"""
Simplified training launcher — thin wrapper around bvr-train-simple.

All argument handling is in the package entrypoint.
See: bvr-train-simple --help
"""

from bvr_marl_core.training.train_simple import main

if __name__ == "__main__":
    exit(main())
