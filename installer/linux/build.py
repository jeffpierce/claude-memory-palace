#!/usr/bin/env python3
"""
Build Memory Palace Setup AppImage for Linux (including Steam Deck).

Creates a portable AppImage that runs on any Linux distro without
installation. Users download, chmod +x, double-click.

Usage:
    python build.py

Requirements:
    - appimagetool (downloaded automatically if not present)
    - Python 3.10+ on the build system

Output:
    dist/MemoryPalaceSetup-x86_64.AppImage

How it works:
    AppImage bundles a minimal Python + the installer code into a
    self-contained executable. When run, it launches the tkinter GUI
    which handles the actual Memory Palace installation.
"""

import subprocess
import sys
import os
import shutil
import urllib.request
import stat
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def ensure_appimagetool(build_dir: Path) -> Path:
    """Download appimagetool if not present."""
    tool = build_dir / "appimagetool"
    if tool.exists():
        return tool

    print("Downloading appimagetool...")
    url = "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage"
    urllib.request.urlretrieve(url, str(tool))
    tool.chmod(tool.stat().st_mode | stat.S_IEXEC)
    return tool


def create_appdir(build_dir: Path, project_root: Path) -> Path:
    """Create the AppDir structure."""
    appdir = build_dir / "MemoryPalaceSetup.AppDir"
    if appdir.exists():
        shutil.rmtree(appdir)
    appdir.mkdir(parents=True)

    # AppRun entry point
    apprun = appdir / "AppRun"
    apprun.write_text('''#!/bin/bash
# AppImage entry point for Memory Palace Setup
SELF_DIR="$(dirname "$(readlink -f "$0")")"

# Try system Python first, fall back to bundled
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        version="$($cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)"
        major="${version%%.*}"
        minor="${version#*.}"
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ] 2>/dev/null; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    # Try using zenity or xmessage for GUI error
    if command -v zenity &>/dev/null; then
        zenity --error --text="Python 3.10+ is required but not found.\\n\\nInstall it with:\\n  sudo apt install python3\\n  or: sudo pacman -S python" --title="Memory Palace Setup"
    elif command -v xmessage &>/dev/null; then
        xmessage "Python 3.10+ is required. Install: sudo apt install python3"
    else
        echo "ERROR: Python 3.10+ is required."
        echo "Install it with your package manager:"
        echo "  Ubuntu/Debian: sudo apt install python3 python3-tk"
        echo "  Arch/SteamOS:  sudo pacman -S python tk"
        echo "  Fedora:        sudo dnf install python3 python3-tkinter"
    fi
    exit 1
fi

# Check for tkinter
$PYTHON -c "import tkinter" 2>/dev/null
if [ $? -ne 0 ]; then
    if command -v zenity &>/dev/null; then
        zenity --error --text="Python tkinter is required but not installed.\\n\\nInstall it with:\\n  Ubuntu: sudo apt install python3-tk\\n  Arch: sudo pacman -S tk\\n  Fedora: sudo dnf install python3-tkinter" --title="Memory Palace Setup"
    else
        echo "ERROR: Python tkinter is required."
        echo "Install it with:"
        echo "  Ubuntu/Debian: sudo apt install python3-tk"
        echo "  Arch/SteamOS:  sudo pacman -S tk"
        echo "  Fedora:        sudo dnf install python3-tkinter"
    fi
    exit 1
fi

# Set up PYTHONPATH and launch
export PYTHONPATH="$SELF_DIR/app:$SELF_DIR/app/installer:$PYTHONPATH"
exec $PYTHON "$SELF_DIR/app/installer/linux/run_gui.py" "$@"
''')
    apprun.chmod(apprun.stat().st_mode | stat.S_IEXEC)

    # Desktop file
    desktop = appdir / "memory-palace-setup.desktop"
    desktop.write_text('''[Desktop Entry]
Type=Application
Name=Memory Palace Setup
Comment=Install Claude Memory Palace — persistent AI memory
Exec=AppRun
Icon=memory-palace
Categories=Utility;Development;
Terminal=false
''')

    # Icon (simple SVG — brain emoji as placeholder)
    icon_dir = appdir / "usr" / "share" / "icons" / "hicolor" / "256x256" / "apps"
    icon_dir.mkdir(parents=True)
    icon_svg = appdir / "memory-palace.svg"
    icon_svg.write_text('''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="256" height="256">
  <rect width="100" height="100" rx="20" fill="#6B46C1"/>
  <text x="50" y="68" font-size="60" text-anchor="middle" fill="white">🧠</text>
</svg>
''')
    # Also copy as png name (AppImage wants it)
    shutil.copy2(str(icon_svg), str(icon_dir / "memory-palace.svg"))
    # Symlink at root for AppImage
    (appdir / "memory-palace.png").symlink_to("memory-palace.svg")

    # Copy application files
    app_dir = appdir / "app"
    app_dir.mkdir()

    # Copy project files
    for name in ["memory_palace", "mcp_server", "installer", "pyproject.toml", "README.md"]:
        src = project_root / name
        dst = app_dir / name
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
                "__pycache__", "*.egg-info", "venv", "build", "dist",
                "*.pyc", ".git", "tmpclaude-*"
            ))
        elif src.is_file():
            shutil.copy2(src, dst)

    print(f"AppDir created at {appdir}")
    return appdir


def build():
    script_dir = Path(__file__).resolve().parent
    project_root = get_project_root()
    build_dir = script_dir / "build"
    dist_dir = script_dir / "dist"

    build_dir.mkdir(exist_ok=True)
    dist_dir.mkdir(exist_ok=True)

    # Get appimagetool
    appimagetool = ensure_appimagetool(build_dir)

    # Create AppDir
    appdir = create_appdir(build_dir, project_root)

    # Build AppImage
    import platform
    arch = platform.machine()
    output = dist_dir / f"MemoryPalaceSetup-{arch}.AppImage"

    print(f"\nBuilding AppImage...")
    env = os.environ.copy()
    env["ARCH"] = arch

    result = subprocess.run(
        [str(appimagetool), "--no-appstream", str(appdir), str(output)],
        env=env,
    )

    if result.returncode == 0 and output.exists():
        # Make executable
        output.chmod(output.stat().st_mode | stat.S_IEXEC)
        size_mb = output.stat().st_size / (1024 * 1024)
        print(f"\n{'='*50}")
        print(f"BUILD SUCCESSFUL")
        print(f"{'='*50}")
        print(f"\n  {output}")
        print(f"  Size: {size_mb:.1f} MB")
        print(f"\n  Users: chmod +x {output.name} && ./{output.name}\n")
        return True

    print("\nBUILD FAILED — check output above")
    return False


def main():
    if sys.platform != "linux":
        print("This build script must be run on Linux.")
        print("(Use WSL if on Windows)")
        sys.exit(1)

    print("Claude Memory Palace — Linux AppImage Builder\n")
    sys.exit(0 if build() else 1)


if __name__ == "__main__":
    main()
