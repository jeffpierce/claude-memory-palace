#!/usr/bin/env python3
"""
Claude Memory Palace — macOS GUI Launcher

Thin wrapper that launches the cross-platform GUI installer.
Can be bundled with py2app into a .app bundle.
"""

import sys
from pathlib import Path

# Ensure shared modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gui.app import InstallerApp


def main():
    app = InstallerApp(title="Claude Memory Palace Setup")
    app.run()


if __name__ == "__main__":
    main()
