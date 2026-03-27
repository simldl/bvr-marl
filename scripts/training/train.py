#!/usr/bin/env python3
"""
Training launcher — thin wrapper around air2air-train.

All argument handling is in the package entrypoint.
See: air2air-train --help
"""

from air_to_air_rl.training.train import main

if __name__ == "__main__":
    exit(main())
