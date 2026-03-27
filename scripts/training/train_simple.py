#!/usr/bin/env python3
"""
Simplified training launcher — thin wrapper around air2air-train-simple.

All argument handling is in the package entrypoint.
See: air2air-train-simple --help
"""

from air_to_air_rl.training.train_simple import main

if __name__ == "__main__":
    exit(main())
