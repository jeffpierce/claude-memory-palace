#!/usr/bin/env python3
"""
Build script for creating MemoryPalaceSetup.exe using PyInstaller.

Usage:
    python build_exe.py

Requirements:
    pip install pyinstaller

This will create:
    dist/MemoryPalaceSetup.exe - Single-file Windows installer
"""

import subprocess
import sys
import os
from pathlib import Path


def check_pyinstaller():
    """Verify PyInstaller is installed."""
    try:
        import PyInstaller
        print(f"PyInstaller version: {PyInstaller.__version__}")
        return True
    except ImportError:
        print("ERROR: PyInstaller is not installed.")
        print("Install it with: pip install pyinstaller")
        return False


def build_exe():
    """Build the executable using PyInstaller."""

    # Get paths
    script_dir = Path(__file__).parent.absolute()
    setup_script = script_dir / "setup_gui.py"
    dist_dir = script_dir / "dist"
    build_dir = script_dir / "build"
    spec_file = script_dir / "MemoryPalaceSetup.spec"

    # Verify source exists
    if not setup_script.exists():
        print(f"ERROR: Cannot find {setup_script}")
        return False

    print(f"Building from: {setup_script}")
    print(f"Output directory: {dist_dir}")
    print()

    # PyInstaller command
    cmd = [
        sys.executable,
        "-m", "PyInstaller",
        "--onefile",           # Single executable file
        "--windowed",          # No console window (GUI app)
        "--name", "MemoryPalaceSetup",
        "--distpath", str(dist_dir),
        "--workpath", str(build_dir),
        "--specpath", str(script_dir),
        # Clean build
        "--clean",
        # Add icon if it exists
        # "--icon", str(script_dir / "icon.ico"),
        str(setup_script)
    ]

    # Add icon if available
    icon_path = script_dir / "icon.ico"
    if icon_path.exists():
        cmd.insert(-1, "--icon")
        cmd.insert(-1, str(icon_path))

    print("Running PyInstaller...")
    print(f"Command: {' '.join(cmd)}")
    print()

    # Run PyInstaller
    result = subprocess.run(cmd)

    if result.returncode == 0:
        exe_path = dist_dir / "MemoryPalaceSetup.exe"
        if exe_path.exists():
            size_mb = exe_path.stat().st_size / (1024 * 1024)
            print()
            print("=" * 50)
            print("BUILD SUCCESSFUL!")
            print("=" * 50)
            print()
            print(f"Executable: {exe_path}")
            print(f"Size: {size_mb:.1f} MB")
            print()
            print("You can distribute this file to users.")
            print("They just need to double-click it to run the installer.")
            return True
        else:
            print("ERROR: Build completed but executable not found.")
            return False
    else:
        print()
        print("=" * 50)
        print("BUILD FAILED")
        print("=" * 50)
        print()
        print("Check the output above for errors.")
        return False


def main():
    """Main entry point."""
    print()
    print("=" * 50)
    print("Claude Memory Palace - Windows Installer Builder")
    print("=" * 50)
    print()

    if not check_pyinstaller():
        sys.exit(1)

    print()
    success = build_exe()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
