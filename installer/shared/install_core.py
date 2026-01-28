"""
Core installation logic for Claude Memory Palace.

Handles venv creation, package installation, and Ollama setup.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable

from .detect import PlatformInfo


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
    progress: Optional[Callable[[str], None]] = None,
) -> InstallResult:
    """
    Install the Memory Palace package into the venv.
    
    Args:
        install_dir: Directory containing the venv
        package_dir: Source package directory (defaults to install_dir for editable install)
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

        result = subprocess.run(
            [str(pip_path), "install", "-e", str(package_dir)],
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


def clone_or_update_repo(
    install_dir: Path,
    repo_url: str = "https://github.com/clawdbot/claude-memory-palace.git",
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
