"""
Platform and dependency detection for Claude Memory Palace installer.

Detects: OS, WSL, Python, Ollama, GPU capabilities, installed AI clients.
"""

import os
import sys
import platform
import subprocess
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class GPUInfo:
    """Detected GPU information."""
    available: bool = False
    vendor: str = ""  # "nvidia", "amd", "apple", "none"
    name: str = ""
    vram_gb: int = 0
    detail: str = ""


@dataclass
class PlatformInfo:
    """Detected platform information."""
    os: str = ""  # "linux", "macos", "windows"
    is_wsl: bool = False
    wsl_version: int = 0  # 1 or 2
    windows_user_dir: Optional[Path] = None  # /mnt/c/Users/<name> if WSL
    arch: str = ""  # "x86_64", "arm64"
    distro: str = ""  # Linux distro name (e.g., "ubuntu", "steamos")
    is_steam_deck: bool = False


@dataclass
class PythonInfo:
    """Detected Python information."""
    available: bool = False
    version: str = ""
    meets_minimum: bool = False
    path: str = ""


@dataclass
class OllamaInfo:
    """Detected Ollama information."""
    installed: bool = False
    running: bool = False
    version: str = ""
    models: List[str] = field(default_factory=list)
    has_embedding_model: bool = False
    has_llm_model: bool = False


@dataclass
class SystemInfo:
    """Complete system detection results."""
    platform: PlatformInfo = field(default_factory=PlatformInfo)
    python: PythonInfo = field(default_factory=PythonInfo)
    ollama: OllamaInfo = field(default_factory=OllamaInfo)
    gpu: GPUInfo = field(default_factory=GPUInfo)


def _run_cmd(cmd: list, timeout: int = 10) -> Tuple[bool, str, str]:
    """Run a command and return (success, stdout, stderr)."""
    try:
        kwargs = {
            "capture_output": True,
            "text": True,
            "timeout": timeout,
        }
        # Hide console window on Windows
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        result = subprocess.run(cmd, **kwargs)
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except FileNotFoundError:
        return False, "", "command not found"
    except subprocess.TimeoutExpired:
        return False, "", "command timed out"
    except Exception as e:
        return False, "", str(e)


def detect_platform() -> PlatformInfo:
    """Detect the current platform, including WSL and Steam Deck."""
    info = PlatformInfo()
    info.arch = platform.machine().lower()

    system = platform.system().lower()

    if system == "darwin":
        info.os = "macos"
        return info

    if system == "windows":
        info.os = "windows"
        return info

    if system == "linux":
        info.os = "linux"

        # Check for WSL
        try:
            with open("/proc/version", "r") as f:
                proc_version = f.read().lower()
            if "microsoft" in proc_version or "wsl" in proc_version:
                info.is_wsl = True
                # Detect WSL version
                if "wsl2" in proc_version:
                    info.wsl_version = 2
                else:
                    info.wsl_version = 1

                # Find Windows user directory
                mnt_c_users = Path("/mnt/c/Users")
                if mnt_c_users.exists():
                    for user_dir in mnt_c_users.iterdir():
                        if user_dir.is_dir() and user_dir.name not in (
                            "Public", "Default", "Default User", "All Users"
                        ):
                            # Check for AppData as a proxy for "real user dir"
                            if (user_dir / "AppData").exists():
                                info.windows_user_dir = user_dir
                                break
        except (OSError, IOError):
            pass

        # Detect Linux distro
        try:
            with open("/etc/os-release", "r") as f:
                os_release = f.read().lower()
            if "steamos" in os_release:
                info.distro = "steamos"
                info.is_steam_deck = True
            elif "ubuntu" in os_release:
                info.distro = "ubuntu"
            elif "debian" in os_release:
                info.distro = "debian"
            elif "fedora" in os_release:
                info.distro = "fedora"
            elif "arch" in os_release:
                info.distro = "arch"
            else:
                # Extract ID from os-release
                for line in os_release.split("\n"):
                    if line.startswith("id="):
                        info.distro = line.split("=", 1)[1].strip().strip('"')
                        break
        except (OSError, IOError):
            pass

        return info

    # Fallback
    info.os = system
    return info


def detect_python() -> PythonInfo:
    """Detect Python version and suitability."""
    info = PythonInfo()
    info.available = True
    info.version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    info.path = sys.executable
    info.meets_minimum = sys.version_info >= (3, 10)
    return info


def detect_ollama() -> OllamaInfo:
    """Detect Ollama installation, status, and available models."""
    info = OllamaInfo()

    # Check if installed
    success, stdout, stderr = _run_cmd(["ollama", "--version"])
    if not success:
        return info

    info.installed = True
    version_text = stdout or stderr
    match = re.search(r"(\d+\.\d+\.?\d*)", version_text)
    if match:
        info.version = match.group(1)

    # Check if running by listing models
    success, stdout, _ = _run_cmd(["ollama", "list"])
    if success:
        info.running = True
        # Parse model list
        for line in stdout.split("\n")[1:]:  # Skip header
            if line.strip():
                model_name = line.split()[0] if line.split() else ""
                if model_name:
                    info.models.append(model_name)

        # Check for our preferred models
        models_lower = [m.lower() for m in info.models]
        info.has_embedding_model = any(
            m.startswith("nomic-embed-text") for m in models_lower
        )
        info.has_llm_model = any(
            m.startswith(prefix) for m in models_lower
            for prefix in ["qwen3:", "qwen2.5:", "llama3", "mistral"]
        )

    return info


def detect_gpu() -> GPUInfo:
    """Detect GPU capabilities (NVIDIA, AMD, Apple Silicon)."""
    info = GPUInfo()

    # Check NVIDIA first
    success, stdout, _ = _run_cmd(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
    )
    if success and stdout:
        parts = stdout.split(",")
        if len(parts) >= 2:
            info.available = True
            info.vendor = "nvidia"
            info.name = parts[0].strip()
            try:
                info.vram_gb = int(parts[1].strip()) // 1024
            except ValueError:
                info.vram_gb = 0
            info.detail = f"{info.name} ({info.vram_gb}GB VRAM)"
            return info

    # Check Apple Silicon
    if platform.system() == "Darwin":
        success, stdout, _ = _run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
        if success and "apple" in stdout.lower():
            info.available = True
            info.vendor = "apple"
            info.name = stdout.strip()

            # Get unified memory (approximate — Apple doesn't expose GPU VRAM separately)
            success, stdout, _ = _run_cmd(["sysctl", "-n", "hw.memsize"])
            if success:
                try:
                    total_bytes = int(stdout.strip())
                    # Apple Silicon shares memory — GPU can use most of it
                    # Conservative estimate: 75% available for GPU
                    info.vram_gb = int(total_bytes / (1024**3) * 0.75)
                except ValueError:
                    info.vram_gb = 8  # Safe default for any Apple Silicon
            info.detail = f"{info.name} (~{info.vram_gb}GB unified memory)"
            return info

    # Check AMD ROCm
    success, stdout, _ = _run_cmd(["rocm-smi", "--showmeminfo", "vram"])
    if success and stdout:
        info.available = True
        info.vendor = "amd"
        # Parse ROCm output
        for line in stdout.split("\n"):
            if "total" in line.lower():
                match = re.search(r"(\d+)", line)
                if match:
                    # ROCm reports in bytes or MB depending on version
                    val = int(match.group(1))
                    if val > 1024 * 1024:  # Bytes
                        info.vram_gb = val // (1024**3)
                    elif val > 1024:  # MB
                        info.vram_gb = val // 1024
                    else:
                        info.vram_gb = val
        info.name = "AMD GPU"
        info.detail = f"AMD GPU ({info.vram_gb}GB VRAM)"

        # Try to get name
        success2, stdout2, _ = _run_cmd(["rocm-smi", "--showproductname"])
        if success2:
            for line in stdout2.split("\n"):
                if line.strip() and not line.startswith("="):
                    info.name = line.strip()
                    info.detail = f"{info.name} ({info.vram_gb}GB VRAM)"
                    break
        return info

    # No GPU detected
    info.detail = "No dedicated GPU detected (CPU-only mode)"
    return info


def detect_all() -> SystemInfo:
    """Run all detection checks and return complete system info."""
    return SystemInfo(
        platform=detect_platform(),
        python=detect_python(),
        ollama=detect_ollama(),
        gpu=detect_gpu(),
    )
