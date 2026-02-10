#!/usr/bin/env python3
"""
Build Memory Palace Setup.app for macOS using py2app.

Creates a standalone .app bundle that users can drag to Applications
or just double-click.

Usage:
    python build.py

Requirements:
    pip install py2app

Output:
    dist/Memory Palace Setup.app
"""

import subprocess
import sys
import os
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def check_py2app() -> bool:
    try:
        import py2app
        print(f"py2app {py2app.__version__}")
        return True
    except ImportError:
        return False


def write_setup_py(script_dir: Path, project_root: Path) -> Path:
    """Write a temporary setup.py for py2app."""
    setup_path = script_dir / "setup_build.py"

    # Collect data files
    data_dirs = []
    for name in ["memory_palace", "mcp_server", "installer/shared", "installer/gui"]:
        src = project_root / name
        if src.exists():
            data_dirs.append(str(src))

    data_files_str = ",\n            ".join(f'"{d}"' for d in data_dirs)

    setup_path.write_text(f'''
from setuptools import setup

APP = ["{script_dir / 'run_gui.py'}"]
DATA_FILES = []
OPTIONS = {{
    "argv_emulation": False,
    "packages": ["tkinter"],
    "includes": [
        "shared.detect",
        "shared.clients",
        "shared.models",
        "shared.install_core",
        "gui.app",
    ],
    "resources": [
        {data_files_str}
    ],
    "plist": {{
        "CFBundleName": "Memory Palace Setup",
        "CFBundleDisplayName": "Memory Palace Setup",
        "CFBundleIdentifier": "com.memorypalace.installer",
        "CFBundleVersion": "2.0.0",
        "CFBundleShortVersionString": "2.0",
        "LSMinimumSystemVersion": "10.15",
        "NSHighResolutionCapable": True,
    }},
}}

setup(
    app=APP,
    name="Memory Palace Setup",
    data_files=DATA_FILES,
    options={{"py2app": OPTIONS}},
    setup_requires=["py2app"],
)
''')
    return setup_path


def build():
    script_dir = Path(__file__).resolve().parent
    project_root = get_project_root()
    dist_dir = script_dir / "dist"

    setup_py = write_setup_py(script_dir, project_root)

    print("Building .app bundle...")
    result = subprocess.run(
        [sys.executable, str(setup_py), "py2app"],
        cwd=str(script_dir),
    )

    # Cleanup
    setup_py.unlink(missing_ok=True)

    if result.returncode == 0:
        app_path = dist_dir / "Memory Palace Setup.app"
        if app_path.exists():
            print(f"\n{'='*50}")
            print(f"BUILD SUCCESSFUL")
            print(f"{'='*50}")
            print(f"\n  {app_path}\n")

            # Optionally create DMG
            dmg_path = dist_dir / "MemoryPalaceSetup.dmg"
            print("Creating DMG...")
            dmg_result = subprocess.run([
                "hdiutil", "create",
                "-volname", "Memory Palace Setup",
                "-srcfolder", str(app_path),
                "-ov", "-format", "UDZO",
                str(dmg_path),
            ])
            if dmg_result.returncode == 0:
                size_mb = dmg_path.stat().st_size / (1024 * 1024)
                print(f"  {dmg_path}")
                print(f"  Size: {size_mb:.1f} MB\n")
            return True

    print("\nBUILD FAILED — check output above")
    return False


def main():
    if sys.platform != "darwin":
        print("This build script must be run on macOS.")
        sys.exit(1)

    print("Memory Palace — macOS Installer Builder\n")

    if not check_py2app():
        print("Installing py2app...")
        subprocess.run([sys.executable, "-m", "pip", "install", "py2app"], check=True)

    sys.exit(0 if build() else 1)


if __name__ == "__main__":
    main()
