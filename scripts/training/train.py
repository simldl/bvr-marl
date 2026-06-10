#!/usr/bin/env python3
"""
Training launcher — thin wrapper around bvr-train.

All argument handling is in the package entrypoint.
See: bvr-train --help
"""

from bvr_marl_core.training.train import main

if __name__ == "__main__":
    exit(main())
