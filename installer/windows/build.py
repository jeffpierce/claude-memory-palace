#!/usr/bin/env python3
"""
Build MemoryPalaceSetup.exe using PyInstaller.

Creates a self-contained Windows installer that bundles the entire
claude-memory-palace package. Users double-click, it extracts, installs,
and configures their AI clients.

Usage:
    python build.py

Requirements:
    pip install pyinstaller

Output:
    dist/MemoryPalaceSetup.exe
"""

import subprocess
import sys
from pathlib import Path


def get_project_root() -> Path:
    """Project root is two levels up from installer/windows/."""
    return Path(__file__).resolve().parent.parent.parent


def check_pyinstaller() -> bool:
    try:
        import PyInstaller
        print(f"PyInstaller {PyInstaller.__version__}")
        return True
    except ImportError:
        return False


def collect_data() -> list:
    """Collect package files to bundle."""
    root = get_project_root()
    data = []
    sep = ";" if sys.platform == "win32" else ":"

    # Directories to bundle
    for name in ["memory_palace", "mcp_server", "installer/shared", "installer/gui"]:
        src = root / name
        if src.exists():
            data.append(f"{src}{sep}claude-memory-palace/{name}")
            print(f"  + {name}/")

    # Files to bundle
    for name in ["pyproject.toml", "README.md", "LICENSE"]:
        src = root / name
        if src.exists():
            data.append(f"{src}{sep}claude-memory-palace")
            print(f"  + {name}")

    return data


def build():
    root = get_project_root()
    script_dir = Path(__file__).resolve().parent
    dist_dir = script_dir / "dist"
    build_dir = script_dir / "build"

    # Entry point: bundled launcher that extracts + runs GUI
    entry = script_dir / "bundled_entry.py"

    # Write the entry point
    entry.write_text('''#!/usr/bin/env python3
"""
PyInstaller entry point — extracts bundled package and launches GUI.
"""
import sys
import os
import shutil
from pathlib import Path


def get_bundled_path() -> Path:
    """PyInstaller extracts to sys._MEIPASS."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "claude-memory-palace"
    return Path(__file__).resolve().parent.parent.parent


def get_install_dir() -> Path:
    return Path.home() / "memory-palace"


def extract():
    """Extract bundled package to install location."""
    src = get_bundled_path()
    dst = get_install_dir()

    print(f"Extracting to {dst}...")
    dst.mkdir(parents=True, exist_ok=True)

    # Smart copy: only copy new/changed files
    for src_file in src.rglob("*"):
        if src_file.is_dir():
            continue
        rel = src_file.relative_to(src)
        # Skip build artifacts
        if any(part in str(rel) for part in ["__pycache__", ".egg-info", "venv"]):
            continue
        dst_file = dst / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        if not dst_file.exists():
            shutil.copy2(src_file, dst_file)
        else:
            # Compare size as quick check
            if src_file.stat().st_size != dst_file.stat().st_size:
                shutil.copy2(src_file, dst_file)


def main():
    extract()

    # Add install dir to path and launch GUI
    install_dir = get_install_dir()
    sys.path.insert(0, str(install_dir / "installer"))
    sys.path.insert(0, str(install_dir))

    from gui.app import InstallerApp
    app = InstallerApp(title="Claude Memory Palace Setup — Windows")
    app.run()


if __name__ == "__main__":
    main()
''')

    print("\nCollecting package files:")
    data_args = collect_data()

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--windowed",
        "--name", "MemoryPalaceSetup",
        "--distpath", str(dist_dir),
        "--workpath", str(build_dir),
        "--specpath", str(script_dir),
        "--clean",
    ]

    for d in data_args:
        cmd.extend(["--add-data", d])

    icon = script_dir / "icon.ico"
    if icon.exists():
        cmd.extend(["--icon", str(icon)])

    cmd.append(str(entry))

    print(f"\nBuilding...")
    result = subprocess.run(cmd)

    # Cleanup temp entry
    entry.unlink(missing_ok=True)

    if result.returncode == 0:
        exe = dist_dir / "MemoryPalaceSetup.exe"
        if exe.exists():
            size_mb = exe.stat().st_size / (1024 * 1024)
            print(f"\n{'='*50}")
            print(f"BUILD SUCCESSFUL")
            print(f"{'='*50}")
            print(f"\n  {exe}")
            print(f"  Size: {size_mb:.1f} MB\n")
            return True

    print("\nBUILD FAILED — check output above")
    return False


def main():
    print("Claude Memory Palace — Windows Installer Builder\n")

    if not check_pyinstaller():
        print("Installing PyInstaller...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)

    sys.exit(0 if build() else 1)


if __name__ == "__main__":
    main()
