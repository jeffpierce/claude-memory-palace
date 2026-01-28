#!/usr/bin/env python3
"""
Claude Memory Palace — Linux GUI Launcher

Thin wrapper that launches the cross-platform GUI installer.
Works on Desktop Linux and Steam Deck (SteamOS).
Can be packaged as an AppImage.
"""

import sys
from pathlib import Path

# Ensure shared modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gui.app import InstallerApp


def main():
    # Detect Steam Deck for title
    title = "Claude Memory Palace Setup"
    try:
        with open("/etc/os-release", "r") as f:
            if "steamos" in f.read().lower():
                title = "Claude Memory Palace Setup — Steam Deck 🎮"
    except (OSError, IOError):
        pass

    app = InstallerApp(title=title)
    app.run()


if __name__ == "__main__":
    main()
