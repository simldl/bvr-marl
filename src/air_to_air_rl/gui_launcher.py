#!/usr/bin/env python3
"""
BVR-MARL GUI launcher — starts the Streamlit-based control panel.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from air_to_air_rl.core.paths import gui_app as _gui_app_path


def check_streamlit():
    """Check if streamlit is installed."""
    try:
        import streamlit

        return True
    except ImportError:
        return False


def install_streamlit():
    """Install streamlit if missing."""
    print("Streamlit not found. Installing...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit"], check=True)
        print("Streamlit installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("Failed to install streamlit. Please install manually:")
        print("  pip install streamlit")
        return False


def launch_gui(port=8501, host="localhost", auto_open=True):
    """Launch the Streamlit GUI."""

    gui_app_path = _gui_app_path()

    if not gui_app_path.exists():
        print(f"Error: GUI app not found at {gui_app_path}")
        return 1

    # Build streamlit command
    cmd = [
        "streamlit",
        "run",
        str(gui_app_path),
        "--server.port",
        str(port),
        "--server.address",
        host,
        "--theme.base",
        "dark",
        "--browser.gatherUsageStats",
        "false",
        "--global.developmentMode",
        "false",
    ]

    if not auto_open:
        cmd.extend(["--server.headless", "true"])

    print("=" * 60)
    print("Launching BVR-MARL Control Panel")
    print("=" * 60)
    print(f"GUI application: {gui_app_path}")
    print(f"Primary URL: http://{host}:{port}")
    if host == "localhost":
        print(f"Alternative URL: http://127.0.0.1:{port}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    print("Troubleshooting tips:")
    print("   - If localhost doesn't work, try 127.0.0.1")
    print("   - Check Windows Firewall settings")
    print("   - Try a different port with --port 8502")
    print("=" * 60)
    print("Press Ctrl+C to stop the GUI")
    print()

    try:
        subprocess.run(cmd, check=True)

    except KeyboardInterrupt:
        print("\nGUI stopped.")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Error launching GUI: {e}")
        return 1
    except FileNotFoundError:
        print("Error: Streamlit not found. Please install with:")
        print("  pip install streamlit")
        return 1

    return 0


def main():
    """Main GUI launcher."""
    parser = argparse.ArgumentParser(
        description="Launch BVR-MARL GUI Control Panel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch GUI on default port
  python scripts/launch_gui.py

  # Launch on custom port
  python scripts/launch_gui.py --port 8502

  # Launch headless (no auto browser open)
  python scripts/launch_gui.py --no-auto-open

  # Launch on all interfaces
  python scripts/launch_gui.py --host 0.0.0.0
        """,
    )

    parser.add_argument("--port", type=int, default=8501, help="Port for the GUI (default: 8501)")
    parser.add_argument(
        "--host", type=str, default="localhost", help="Host for the GUI (default: localhost)"
    )
    parser.add_argument(
        "--no-auto-open", action="store_true", help="Don't automatically open browser"
    )
    parser.add_argument(
        "--install-deps", action="store_true", help="Install missing dependencies automatically"
    )

    args = parser.parse_args()

    # Check dependencies
    if not check_streamlit():
        if args.install_deps:
            if not install_streamlit():
                return 1
        else:
            print("Error: Streamlit is required but not installed.")
            print("Install with: pip install streamlit")
            print("Or run with --install-deps to install automatically")
            return 1

    # Launch GUI
    return launch_gui(port=args.port, host=args.host, auto_open=not args.no_auto_open)


if __name__ == "__main__":
    exit(main())
