#!/usr/bin/env python3
"""
Claude Memory Palace — Windows GUI Launcher

Thin wrapper that launches the cross-platform GUI installer.
Can be compiled with PyInstaller into a standalone .exe.
"""

import sys
from pathlib import Path

# Ensure shared modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gui.app import InstallerApp


def main():
    app = InstallerApp(title="Claude Memory Palace Setup — Windows")
    app.run()


if __name__ == "__main__":
    main()
