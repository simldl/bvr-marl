#!/usr/bin/env python3
"""
Launch the BVR-MARL-Core GUI Control Panel.

Convenience script — delegates to the installed entry point.
Equivalent to running: bvr-gui
"""

from bvr_marl_core.gui_launcher import main

if __name__ == "__main__":
    exit(main())
