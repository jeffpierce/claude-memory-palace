#!/usr/bin/env python3
"""
Build script for creating a self-contained MemoryPalaceSetup.exe using PyInstaller.

This bundles the ENTIRE claude-memory-palace package into the executable.
When the user runs it:
1. The bundled package is extracted to ~/memory-palace/
2. The GUI installer runs and does pip install -e on the extracted package
3. Ollama models are downloaded
4. Claude Desktop is configured

Usage:
    python build_bundled_exe.py

Requirements:
    pip install pyinstaller

Output:
    dist/MemoryPalaceSetup.exe - Single-file Windows installer (self-contained)
"""

import subprocess
import sys
import os
from pathlib import Path
import shutil


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


def get_project_root():
    """Get the claude-memory-palace project root directory."""
    # This script is at: claude-memory-palace/installer/windows/build_bundled_exe.py
    script_dir = Path(__file__).parent.absolute()
    return script_dir.parent.parent


def collect_package_files():
    """
    Collect all package files that need to be bundled.

    Returns a list of (source, dest) tuples for PyInstaller --add-data
    """
    project_root = get_project_root()
    data_files = []

    # Directories to include
    include_dirs = [
        "memory_palace",
        "mcp_server",
        "setup",
        "tools",
        "docs",
        "installer",  # Include installer dir for any assets
    ]

    # Single files to include
    include_files = [
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "install.bat",
        "install.sh",
        "install.ps1",
    ]

    # Collect directories
    for dir_name in include_dirs:
        src_dir = project_root / dir_name
        if src_dir.exists() and src_dir.is_dir():
            # PyInstaller format: source;dest_in_bundle
            # We want these at claude-memory-palace/<dir_name> in the bundle
            data_files.append((str(src_dir), f"claude-memory-palace/{dir_name}"))
            print(f"  + {dir_name}/")

    # Collect individual files
    for file_name in include_files:
        src_file = project_root / file_name
        if src_file.exists() and src_file.is_file():
            # These go at claude-memory-palace/ root
            data_files.append((str(src_file), "claude-memory-palace"))
            print(f"  + {file_name}")

    return data_files


def build_exe():
    """Build the self-contained executable using PyInstaller."""

    # Get paths
    script_dir = Path(__file__).parent.absolute()
    project_root = get_project_root()

    # Entry point is our bundled wrapper
    entry_script = script_dir / "bundled_setup_gui.py"

    # Output directories
    dist_dir = script_dir / "dist"
    build_dir = script_dir / "build"
    spec_file = script_dir / "MemoryPalaceSetup.spec"

    # Verify source exists
    if not entry_script.exists():
        print(f"ERROR: Cannot find entry script at {entry_script}")
        return False

    print(f"Entry script: {entry_script}")
    print(f"Project root: {project_root}")
    print(f"Output directory: {dist_dir}")
    print()

    # Collect data files to bundle
    print("Collecting package files to bundle:")
    data_files = collect_package_files()
    print()

    if not data_files:
        print("ERROR: No package files found to bundle!")
        return False

    # Build PyInstaller command
    cmd = [
        sys.executable,
        "-m", "PyInstaller",
        "--onefile",           # Single executable file
        "--windowed",          # No console window (GUI app)
        "--name", "MemoryPalaceSetup",
        "--distpath", str(dist_dir),
        "--workpath", str(build_dir),
        "--specpath", str(script_dir),
        "--clean",             # Clean build
    ]

    # Add all data files
    for src, dest in data_files:
        # Windows uses ; as separator, other platforms use :
        separator = ";" if sys.platform == "win32" else ":"
        cmd.extend(["--add-data", f"{src}{separator}{dest}"])

    # Add icon if available
    icon_path = script_dir / "icon.ico"
    if icon_path.exists():
        cmd.extend(["--icon", str(icon_path)])
        print(f"Using icon: {icon_path}")

    # Add the entry script
    cmd.append(str(entry_script))

    print("Running PyInstaller...")
    print()
    print("Command (abbreviated):")
    print(f"  pyinstaller --onefile --windowed --name MemoryPalaceSetup \\")
    print(f"    --add-data <package_files> \\")
    print(f"    {entry_script.name}")
    print()

    # Run PyInstaller
    result = subprocess.run(cmd)

    if result.returncode == 0:
        exe_path = dist_dir / "MemoryPalaceSetup.exe"
        if exe_path.exists():
            size_mb = exe_path.stat().st_size / (1024 * 1024)
            print()
            print("=" * 60)
            print("BUILD SUCCESSFUL!")
            print("=" * 60)
            print()
            print(f"Executable: {exe_path}")
            print(f"Size: {size_mb:.1f} MB")
            print()
            print("This is a self-contained installer.")
            print("Users can download and run it - no other files needed.")
            print()
            print("The installer will:")
            print("  1. Extract the package to ~/memory-palace/")
            print("  2. Run pip install -e on the extracted package")
            print("  3. Download Ollama models (if selected)")
            print("  4. Configure Claude Desktop (if selected)")
            print()
            return True
        else:
            print("ERROR: Build completed but executable not found.")
            return False
    else:
        print()
        print("=" * 60)
        print("BUILD FAILED")
        print("=" * 60)
        print()
        print("Check the output above for errors.")
        return False


def main():
    """Main entry point."""
    print()
    print("=" * 60)
    print("Claude Memory Palace - Bundled Installer Builder")
    print("=" * 60)
    print()
    print("This will create a self-contained Windows installer that")
    print("bundles the entire claude-memory-palace package.")
    print()

    if not check_pyinstaller():
        print()
        print("Installing PyInstaller...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"])
        if result.returncode != 0:
            print("Failed to install PyInstaller")
            sys.exit(1)
        print()

    print()
    success = build_exe()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
