"""
Core installation logic for Memory Palace.

Handles venv creation, package installation, and Ollama setup.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable

import time
import zipfile
import tempfile
from urllib.request import urlopen, Request
from urllib.error import URLError

from .detect import PlatformInfo, PostgresInfo, _find_pg_bin, _run_cmd


@dataclass
class InstallResult:
    """Result of an installation step."""
    success: bool
    message: str
    detail: Optional[str] = None


def get_default_install_dir(plat: PlatformInfo) -> Path:
    """
    Get the default installation directory.
    
    Platform conventions:
    - Windows: ~/memory-palace
    - macOS: ~/memory-palace
    - Linux: ~/memory-palace
    - WSL: ~/memory-palace (Linux side)
    """
    return Path.home() / "memory-palace"


def find_python() -> Optional[str]:
    """
    Find a suitable Python 3.10+ interpreter.
    
    Checks: python3, python, py (Windows launcher).
    Returns the command string or None.
    """
    candidates = ["python3", "python"]
    if sys.platform == "win32":
        candidates.append("py")

    for cmd in candidates:
        try:
            kwargs = {
                "capture_output": True,
                "text": True,
                "timeout": 5,
            }
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

            result = subprocess.run([cmd, "--version"], **kwargs)
            if result.returncode == 0:
                version_str = result.stdout.strip() or result.stderr.strip()
                # Parse "Python 3.12.1"
                parts = version_str.split()
                if len(parts) >= 2:
                    ver = parts[1].split(".")
                    if int(ver[0]) >= 3 and int(ver[1]) >= 10:
                        return cmd
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            continue

    return None


def create_venv(
    install_dir: Path,
    python_cmd: Optional[str] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> InstallResult:
    """
    Create a Python virtual environment in the install directory.
    
    Args:
        install_dir: Target directory for the installation
        python_cmd: Python command to use (auto-detected if None)
        progress: Optional callback for status messages
    """
    if python_cmd is None:
        python_cmd = find_python()
        if python_cmd is None:
            return InstallResult(
                success=False,
                message="Python 3.10+ not found",
                detail="Install Python from https://www.python.org/downloads/"
            )

    venv_dir = install_dir / "venv"

    if progress:
        progress(f"Creating virtual environment at {venv_dir}...")

    try:
        kwargs = {
            "capture_output": True,
            "text": True,
            "timeout": 120,
        }
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        result = subprocess.run([python_cmd, "-m", "venv", str(venv_dir)], **kwargs)

        if result.returncode != 0:
            return InstallResult(
                success=False,
                message="Failed to create virtual environment",
                detail=result.stderr[:300]
            )

        if progress:
            progress("✓ Virtual environment created")

        return InstallResult(success=True, message="Virtual environment created")

    except subprocess.TimeoutExpired:
        return InstallResult(
            success=False,
            message="Timed out creating virtual environment"
        )
    except Exception as e:
        return InstallResult(
            success=False,
            message=f"Error creating venv: {str(e)[:100]}"
        )


def get_venv_pip(install_dir: Path) -> Path:
    """Get the path to pip in the virtual environment."""
    venv_dir = install_dir / "venv"
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "pip.exe"
    return venv_dir / "bin" / "pip"


def get_venv_python(install_dir: Path) -> Path:
    """Get the path to python in the virtual environment."""
    venv_dir = install_dir / "venv"
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def install_package(
    install_dir: Path,
    package_dir: Optional[Path] = None,
    extras: Optional[list[str]] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> InstallResult:
    """
    Install the Memory Palace package into the venv.

    Args:
        install_dir: Directory containing the venv
        package_dir: Source package directory (defaults to install_dir for editable install)
        extras: Optional list of extras to install (e.g. ["postgres"])
        progress: Optional callback for status messages
    """
    if package_dir is None:
        package_dir = install_dir

    pip_path = get_venv_pip(install_dir)

    if not pip_path.exists():
        return InstallResult(
            success=False,
            message="pip not found in virtual environment",
            detail=f"Expected at: {pip_path}"
        )

    if progress:
        progress("Installing Memory Palace package...")

    try:
        kwargs = {
            "capture_output": True,
            "text": True,
            "timeout": 300,  # 5 minutes
        }
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        # Build install target: "path" or "path[postgres]"
        install_target = str(package_dir)
        if extras:
            install_target += "[" + ",".join(extras) + "]"

        result = subprocess.run(
            [str(pip_path), "install", "-e", install_target],
            **kwargs
        )

        if result.returncode == 0:
            if progress:
                progress("✓ Memory Palace package installed")
            return InstallResult(success=True, message="Package installed")
        else:
            return InstallResult(
                success=False,
                message="Package installation failed",
                detail=result.stderr[:400]
            )

    except subprocess.TimeoutExpired:
        return InstallResult(
            success=False,
            message="Package installation timed out"
        )
    except Exception as e:
        return InstallResult(
            success=False,
            message=f"Error installing package: {str(e)[:100]}"
        )


def install_ollama(
    plat: PlatformInfo,
    progress: Optional[Callable[[str], None]] = None,
) -> InstallResult:
    """
    Install Ollama for the current platform.
    
    Uses the official install method for each platform:
    - Linux/WSL: curl -fsSL https://ollama.com/install.sh | sh
    - macOS: brew install ollama (or download link)
    - Windows: winget install Ollama.Ollama
    """
    if progress:
        progress("Installing Ollama...")

    try:
        if plat.os == "linux" or plat.is_wsl:
            # Official Linux installer
            result = subprocess.run(
                ["bash", "-c", "curl -fsSL https://ollama.com/install.sh | sh"],
                capture_output=True,
                text=True,
                timeout=300,
            )
        elif plat.os == "macos":
            # Try brew first
            if shutil.which("brew"):
                result = subprocess.run(
                    ["brew", "install", "ollama"],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
            else:
                return InstallResult(
                    success=False,
                    message="Please install Ollama from https://ollama.com/download",
                    detail="Homebrew not found for automatic installation"
                )
        elif plat.os == "windows":
            result = subprocess.run(
                ["winget", "install", "Ollama.Ollama",
                 "--accept-package-agreements", "--accept-source-agreements"],
                capture_output=True,
                text=True,
                timeout=600,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        else:
            return InstallResult(
                success=False,
                message=f"Unsupported platform: {plat.os}"
            )

        if result.returncode == 0:
            if progress:
                progress("✓ Ollama installed")
            return InstallResult(success=True, message="Ollama installed")
        else:
            return InstallResult(
                success=False,
                message="Ollama installation failed",
                detail=result.stderr[:200]
            )

    except subprocess.TimeoutExpired:
        return InstallResult(
            success=False,
            message="Ollama installation timed out"
        )
    except FileNotFoundError as e:
        return InstallResult(
            success=False,
            message="Install command not found",
            detail=f"Please install Ollama manually from https://ollama.com/download ({e})"
        )
    except Exception as e:
        return InstallResult(
            success=False,
            message=f"Error installing Ollama: {str(e)[:100]}"
        )


def install_postgres(
    plat: PlatformInfo,
    progress: Optional[Callable[[str], None]] = None,
) -> InstallResult:
    """
    Install PostgreSQL 16 for the current platform.

    Uses platform package managers:
    - Windows: winget install PostgreSQL.PostgreSQL.16
    - Linux (Debian/Ubuntu): sudo apt-get install postgresql-16
    - Linux (Fedora): sudo dnf install postgresql16-server
    - macOS: brew install postgresql@16
    """
    if progress:
        progress("Installing PostgreSQL 16...")

    try:
        if plat.os == "windows":
            result = subprocess.run(
                ["winget", "install", "PostgreSQL.PostgreSQL.16",
                 "--accept-package-agreements", "--accept-source-agreements"],
                capture_output=True,
                text=True,
                timeout=600,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        elif plat.os == "linux":
            if plat.distro in ("ubuntu", "debian"):
                result = subprocess.run(
                    ["sudo", "apt-get", "install", "-y", "postgresql-16"],
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
            elif plat.distro == "fedora":
                result = subprocess.run(
                    ["sudo", "dnf", "install", "-y", "postgresql16-server"],
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
            else:
                return InstallResult(
                    success=False,
                    message=f"Unsupported Linux distro: {plat.distro}",
                    detail="Please install PostgreSQL 16 manually."
                )
        elif plat.os == "macos":
            if shutil.which("brew"):
                result = subprocess.run(
                    ["brew", "install", "postgresql@16"],
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
            else:
                return InstallResult(
                    success=False,
                    message="Homebrew not found",
                    detail="Install from https://www.postgresql.org/download/macosx/"
                )
        else:
            return InstallResult(
                success=False,
                message=f"Unsupported platform: {plat.os}"
            )

        if result.returncode == 0:
            # Start the service and verify it's running
            if plat.os == "windows":
                if progress:
                    progress("Starting PostgreSQL service...")
                # Windows service name is typically postgresql-x64-16
                for svc_name in [f"postgresql-x64-{16}", "postgresql", "PostgreSQL"]:
                    svc_result = subprocess.run(
                        ["net", "start", svc_name],
                        capture_output=True, text=True, timeout=30,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                    if svc_result.returncode == 0 or "already been started" in (svc_result.stderr + svc_result.stdout):
                        break

            elif plat.os == "linux":
                if plat.distro == "fedora":
                    # Fedora needs explicit initdb + start
                    subprocess.run(
                        ["sudo", "postgresql-setup", "--initdb"],
                        capture_output=True, text=True, timeout=30,
                    )
                subprocess.run(
                    ["sudo", "systemctl", "start", "postgresql"],
                    capture_output=True, text=True, timeout=30,
                )
                subprocess.run(
                    ["sudo", "systemctl", "enable", "postgresql"],
                    capture_output=True, text=True, timeout=10,
                )

            elif plat.os == "macos":
                subprocess.run(
                    ["brew", "services", "start", "postgresql@16"],
                    capture_output=True, text=True, timeout=30,
                )

            # Give the service a moment, then verify with pg_isready
            time.sleep(3)
            pg_isready = _find_pg_bin("pg_isready")
            if pg_isready:
                for attempt in range(5):
                    ok, _, _ = _run_cmd([pg_isready], timeout=5)
                    if ok:
                        if progress:
                            progress("PostgreSQL service is running")
                        break
                    time.sleep(2)
                else:
                    if progress:
                        progress("PostgreSQL installed but service may not be running yet")

            if progress:
                progress("PostgreSQL 16 installed")
            return InstallResult(success=True, message="PostgreSQL 16 installed")
        else:
            return InstallResult(
                success=False,
                message="PostgreSQL installation failed",
                detail=result.stderr[:300] if result.stderr else result.stdout[:300]
            )

    except subprocess.TimeoutExpired:
        return InstallResult(
            success=False,
            message="PostgreSQL installation timed out (>10 minutes)"
        )
    except FileNotFoundError as e:
        return InstallResult(
            success=False,
            message="Install command not found",
            detail=f"Install PostgreSQL manually: https://www.postgresql.org/download/ ({e})"
        )
    except Exception as e:
        return InstallResult(
            success=False,
            message=f"Error installing PostgreSQL: {str(e)[:100]}"
        )


def _get_pg_major_version(pg_info: PostgresInfo) -> Optional[int]:
    """Extract major version number from PostgresInfo (e.g. '16' from '16.2')."""
    if pg_info.version:
        try:
            return int(pg_info.version.split(".")[0])
        except (ValueError, IndexError):
            pass
    return None


def _download_pgvector_windows(
    pg_info: PostgresInfo,
    progress: Optional[Callable[[str], None]] = None,
) -> InstallResult:
    """
    Download and install prebuilt pgvector for Windows.

    Uses community builds from andreiramani/pgvector_pgsql_windows.
    Extracts vector.dll, vector.control, and SQL files to the PG install dir.
    """
    major = _get_pg_major_version(pg_info)
    if not major:
        return InstallResult(
            success=False,
            message="Cannot determine PostgreSQL major version"
        )

    # Find the install directory
    pg_config = _find_pg_bin("pg_config")
    if not pg_config:
        return InstallResult(
            success=False,
            message="pg_config not found — cannot locate PostgreSQL install directory"
        )

    # Get lib and share dirs from pg_config
    ok, libdir_str, _ = _run_cmd([pg_config, "--pkglibdir"])
    if not ok or not libdir_str:
        return InstallResult(
            success=False,
            message="Could not determine PostgreSQL lib directory"
        )
    ok, sharedir_str, _ = _run_cmd([pg_config, "--sharedir"])
    if not ok or not sharedir_str:
        return InstallResult(
            success=False,
            message="Could not determine PostgreSQL share directory"
        )

    libdir = Path(libdir_str.strip())
    ext_dir = Path(sharedir_str.strip()) / "extension"

    # Download pgvector prebuilt Windows binaries from andreiramani/pgvector_pgsql_windows.
    # This repo creates separate releases per PG major version. We query the GitHub
    # API to find the right release tag (contains "_{major}") and asset filename
    # (matches "vector.*-pg{major}.zip").
    if progress:
        progress("Finding pgvector Windows binaries...")

    zip_data = None
    used_url = None
    api_url = "https://api.github.com/repos/andreiramani/pgvector_pgsql_windows/releases"

    try:
        import json as _json
        req = Request(api_url, headers={
            "User-Agent": "MemoryPalaceInstaller/2.0",
            "Accept": "application/vnd.github+json",
        })
        with urlopen(req, timeout=30) as resp:
            releases = _json.loads(resp.read())

        # Find a release whose tag contains our PG major version
        for release in releases:
            tag = release.get("tag_name", "")
            if f"_{major}" not in tag:
                continue
            # Find the zip asset for our PG major
            for asset in release.get("assets", []):
                name = asset.get("name", "")
                if name.endswith(".zip") and f"pg{major}" in name:
                    used_url = asset["browser_download_url"]
                    break
            if used_url:
                break
    except Exception as e:
        if progress:
            progress(f"  GitHub API query failed: {e}")

    if not used_url:
        # Fallback: try known versioned URL pattern
        used_url = (
            f"https://github.com/andreiramani/pgvector_pgsql_windows"
            f"/releases/download/0.8.1_{major}/vector.v0.8.1-pg{major}.zip"
        )

    if progress:
        progress("Downloading pgvector Windows binaries...")

    try:
        req = Request(used_url, headers={"User-Agent": "MemoryPalaceInstaller/2.0"})
        with urlopen(req, timeout=120) as resp:
            zip_data = resp.read()
    except (URLError, OSError) as e:
        zip_data = None

    if not zip_data:
        return InstallResult(
            success=False,
            message="Could not download pgvector Windows binaries",
            detail=f"Tried URLs for PG {major}. Install manually: "
                   "https://github.com/pgvector/pgvector#windows"
        )

    if progress:
        progress(f"Downloaded pgvector ({len(zip_data) // 1024}KB)")

    # Extract to a temp dir, then copy files to the PG install.
    # Copying to Program Files requires admin — try normal first, then
    # escalate via UAC prompt if needed.
    try:
        with tempfile.TemporaryDirectory(delete=False) as tmpdir:
            zip_path = Path(tmpdir) / "pgvector.zip"
            zip_path.write_bytes(zip_data)

            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmpdir)

            tmp = Path(tmpdir)

            # Find required files
            dll_files = list(tmp.rglob("vector.dll"))
            if not dll_files:
                return InstallResult(
                    success=False,
                    message="vector.dll not found in downloaded archive",
                    detail=f"Archive from: {used_url}"
                )

            control_files = list(tmp.rglob("vector.control"))
            sql_files = list(tmp.rglob("vector--*.sql"))

            if progress:
                progress("Copying pgvector files to PostgreSQL installation...")

            # Try direct copy first (works if already admin)
            try:
                shutil.copy2(dll_files[0], libdir / "vector.dll")
                if control_files:
                    ext_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(control_files[0], ext_dir / "vector.control")
                for sql_file in sql_files:
                    shutil.copy2(sql_file, ext_dir / sql_file.name)

            except PermissionError:
                # Need elevation — write a batch script and run it via UAC
                if progress:
                    progress("Admin access needed — requesting permission...")

                bat_path = tmp / "install_pgvector.bat"
                lines = ["@echo off"]
                lines.append(f'copy /Y "{dll_files[0]}" "{libdir / "vector.dll"}"')
                lines.append(f'mkdir "{ext_dir}" 2>nul')
                if control_files:
                    lines.append(f'copy /Y "{control_files[0]}" "{ext_dir / "vector.control"}"')
                for sql_file in sql_files:
                    lines.append(f'copy /Y "{sql_file}" "{ext_dir / sql_file.name}"')

                bat_path.write_text("\n".join(lines), encoding="utf-8")

                # Run elevated via PowerShell Start-Process -Verb RunAs -Wait
                # This triggers a UAC prompt for the user
                ps_cmd = (
                    f'Start-Process cmd.exe '
                    f'-ArgumentList \'/c "{bat_path}"\' '
                    f'-Verb RunAs -Wait'
                )
                kwargs = {
                    "capture_output": True,
                    "text": True,
                    "timeout": 60,
                }
                if sys.platform == "win32":
                    kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

                result = subprocess.run(
                    ["powershell", "-Command", ps_cmd],
                    **kwargs,
                )

                # Verify the files actually landed
                if not (libdir / "vector.dll").exists():
                    return InstallResult(
                        success=False,
                        message="pgvector install failed — admin permission denied or copy failed",
                        detail=f"vector.dll was not copied to {libdir}. "
                               "Try right-clicking the installer and selecting 'Run as Administrator'."
                    )

        # Clean up temp dir (delete=False above so it survives the elevation)
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass

        if progress:
            progress("pgvector files installed")
        return InstallResult(success=True, message="pgvector installed (Windows)")

    except Exception as e:
        return InstallResult(
            success=False,
            message=f"Error extracting pgvector: {str(e)[:150]}"
        )


def install_pgvector(
    plat: PlatformInfo,
    pg_info: PostgresInfo,
    progress: Optional[Callable[[str], None]] = None,
) -> InstallResult:
    """
    Install the pgvector extension for PostgreSQL.

    - Windows: Download prebuilt binaries from community GitHub releases
    - Linux (Debian/Ubuntu): sudo apt-get install postgresql-16-pgvector
    - macOS: brew install pgvector
    """
    if progress:
        progress("Installing pgvector extension...")

    try:
        if plat.os == "windows":
            return _download_pgvector_windows(pg_info, progress)

        elif plat.os == "linux":
            major = _get_pg_major_version(pg_info) or 16
            if plat.distro in ("ubuntu", "debian"):
                result = subprocess.run(
                    ["sudo", "apt-get", "install", "-y", f"postgresql-{major}-pgvector"],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
            elif plat.distro == "fedora":
                result = subprocess.run(
                    ["sudo", "dnf", "install", "-y", "pgvector_16"],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
            else:
                return InstallResult(
                    success=False,
                    message=f"Unsupported distro for pgvector: {plat.distro}",
                    detail="Install pgvector manually: https://github.com/pgvector/pgvector"
                )

            if result.returncode == 0:
                if progress:
                    progress("pgvector installed")
                return InstallResult(success=True, message="pgvector installed")
            else:
                return InstallResult(
                    success=False,
                    message="pgvector installation failed",
                    detail=result.stderr[:300]
                )

        elif plat.os == "macos":
            if shutil.which("brew"):
                result = subprocess.run(
                    ["brew", "install", "pgvector"],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if result.returncode == 0:
                    if progress:
                        progress("pgvector installed")
                    return InstallResult(success=True, message="pgvector installed")
                else:
                    return InstallResult(
                        success=False,
                        message="pgvector installation failed",
                        detail=result.stderr[:300]
                    )
            else:
                return InstallResult(
                    success=False,
                    message="Homebrew not found",
                    detail="Install pgvector manually: https://github.com/pgvector/pgvector"
                )
        else:
            return InstallResult(
                success=False,
                message=f"Unsupported platform: {plat.os}"
            )

    except subprocess.TimeoutExpired:
        return InstallResult(
            success=False,
            message="pgvector installation timed out"
        )
    except Exception as e:
        return InstallResult(
            success=False,
            message=f"Error installing pgvector: {str(e)[:100]}"
        )


def _run_psql(psql: str, args: list, password: str = "", timeout: int = 15):
    """
    Run a psql command with -w (never prompt for password) and optional PGPASSWORD.

    Returns (success, stdout, stderr).
    """
    env = os.environ.copy()
    if password:
        env["PGPASSWORD"] = password

    cmd = [psql, "-w"] + args  # -w = never prompt for password

    try:
        kwargs = {
            "capture_output": True,
            "text": True,
            "timeout": timeout,
            "env": env,
        }
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        result = subprocess.run(cmd, **kwargs)
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except FileNotFoundError:
        return False, "", "psql not found"
    except subprocess.TimeoutExpired:
        return False, "", "command timed out"
    except Exception as e:
        return False, "", str(e)


def _extract_pg_password(pg_url: str) -> str:
    """Extract password from a PostgreSQL connection URL, if present."""
    # postgresql://user:password@host:port/dbname
    try:
        from urllib.parse import urlparse
        parsed = urlparse(pg_url)
        return parsed.password or ""
    except Exception:
        return ""


def setup_postgres_database(
    pg_url: str,
    progress: Optional[Callable[[str], None]] = None,
) -> InstallResult:
    """
    Set up the memory_palace database in PostgreSQL.

    Creates the database if it doesn't exist, enables the pgvector extension,
    and tests the connection. Uses psql with -w flag (never prompt for password)
    to avoid hanging on auth prompts. Tries password from the connection URL
    first, then common defaults.
    """
    if progress:
        progress("Setting up PostgreSQL database...")

    psql = _find_pg_bin("psql")
    if not psql:
        return InstallResult(
            success=False,
            message="psql not found — cannot set up database",
            detail="Make sure PostgreSQL bin directory is on your PATH"
        )

    # Figure out password: try URL password, then common defaults
    url_password = _extract_pg_password(pg_url)
    passwords_to_try = []
    if url_password:
        passwords_to_try.append(url_password)
    passwords_to_try.extend(["postgres", ""])  # Common defaults

    # Find a working password by testing connection
    working_password = None
    for pw in passwords_to_try:
        ok, _, stderr = _run_psql(
            psql,
            ["-U", "postgres", "-c", "SELECT 1"],
            password=pw,
            timeout=10,
        )
        if ok:
            working_password = pw
            if progress:
                progress("Connected to PostgreSQL")
            break

    if working_password is None:
        return InstallResult(
            success=False,
            message="Cannot connect to PostgreSQL — authentication failed",
            detail="Set the postgres user password in your connection URL: "
                   "postgresql://postgres:YOUR_PASSWORD@localhost:5432/memory_palace"
        )

    errors = []

    # Create database (ignore error if it already exists)
    if progress:
        progress("Creating memory_palace database...")
    ok, stdout, stderr = _run_psql(
        psql,
        ["-U", "postgres", "-c", "CREATE DATABASE memory_palace"],
        password=working_password,
    )
    if not ok:
        if "already exists" in stderr:
            if progress:
                progress("Database memory_palace already exists")
        else:
            errors.append(f"Create database: {stderr[:200]}")

    # Enable pgvector extension
    if progress:
        progress("Enabling pgvector extension...")
    ok, stdout, stderr = _run_psql(
        psql,
        ["-U", "postgres", "-d", "memory_palace",
         "-c", "CREATE EXTENSION IF NOT EXISTS vector"],
        password=working_password,
    )
    if not ok:
        errors.append(f"Enable pgvector: {stderr[:200]}")

    # Test connection — try to query the extension
    if progress:
        progress("Testing database connection...")
    ok, stdout, stderr = _run_psql(
        psql,
        ["-U", "postgres", "-d", "memory_palace",
         "-c", "SELECT extname FROM pg_extension WHERE extname = 'vector'"],
        password=working_password,
        timeout=10,
    )
    if ok and "vector" in stdout:
        if progress:
            progress("PostgreSQL database ready with pgvector")
        return InstallResult(
            success=True,
            message="Database ready with pgvector enabled"
        )
    elif ok:
        # Connected but pgvector not showing
        msg = "Database created but pgvector extension may not be enabled"
        if errors:
            msg += f" ({'; '.join(errors)})"
        return InstallResult(
            success=True,
            message=msg,
            detail="You can enable it later: CREATE EXTENSION vector"
        )
    else:
        return InstallResult(
            success=False,
            message="Database setup incomplete",
            detail="; ".join(errors) if errors else stderr[:200]
        )


def clone_or_update_repo(
    install_dir: Path,
    repo_url: str = "https://github.com/jeffpierce/memory-palace.git",
    branch: str = "main",
    progress: Optional[Callable[[str], None]] = None,
) -> InstallResult:
    """
    Clone or update the Memory Palace repository.
    
    For fresh installs: git clone
    For existing installs: git pull
    """
    if progress:
        progress("Downloading Memory Palace...")

    try:
        if (install_dir / ".git").exists():
            # Update existing
            if progress:
                progress("Updating existing installation...")
            result = subprocess.run(
                ["git", "pull", "origin", branch],
                cwd=str(install_dir),
                capture_output=True,
                text=True,
                timeout=120,
            )
        else:
            # Fresh clone
            install_dir.parent.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                ["git", "clone", "--branch", branch, repo_url, str(install_dir)],
                capture_output=True,
                text=True,
                timeout=120,
            )

        if result.returncode == 0:
            if progress:
                progress("✓ Memory Palace downloaded")
            return InstallResult(success=True, message="Repository ready")
        else:
            return InstallResult(
                success=False,
                message="Failed to download Memory Palace",
                detail=result.stderr[:300]
            )

    except FileNotFoundError:
        return InstallResult(
            success=False,
            message="git not found — please install git first",
            detail="https://git-scm.com/downloads"
        )
    except subprocess.TimeoutExpired:
        return InstallResult(success=False, message="Download timed out")
    except Exception as e:
        return InstallResult(
            success=False,
            message=f"Error downloading: {str(e)[:100]}"
        )


def verify_installation(
    install_dir: Path,
    progress: Optional[Callable[[str], None]] = None,
) -> InstallResult:
    """
    Verify the installation works by importing the package and checking MCP server.
    """
    if progress:
        progress("Verifying installation...")

    python_path = get_venv_python(install_dir)
    if not python_path.exists():
        return InstallResult(
            success=False,
            message="Python not found in venv",
            detail=f"Expected at: {python_path}"
        )

    try:
        # Test that the package can be imported
        result = subprocess.run(
            [str(python_path), "-c", "import memory_palace; import mcp_server; print('OK')"],
            cwd=str(install_dir),
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0 and "OK" in result.stdout:
            if progress:
                progress("✓ Installation verified")
            return InstallResult(success=True, message="Installation verified")
        else:
            return InstallResult(
                success=False,
                message="Package import failed",
                detail=result.stderr[:300]
            )

    except Exception as e:
        return InstallResult(
            success=False,
            message=f"Verification failed: {str(e)[:100]}"
        )
