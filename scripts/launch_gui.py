#!/usr/bin/env python3
"""
Launch the BVR-MARL GUI Control Panel.

Convenience script — delegates to the installed entry point.
Equivalent to running: air2air-gui
"""

from air_to_air_rl.gui_launcher import main

if __name__ == "__main__":
    exit(main())
