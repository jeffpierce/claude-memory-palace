"""
AI client discovery and MCP configuration for Claude Memory Palace installer.

Discovers installed MCP-compatible AI clients and configures them.
"""

import json
import os
import shutil
import platform
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from .detect import PlatformInfo


@dataclass
class ClientInfo:
    """Information about a detected AI client."""
    name: str  # Display name
    id: str  # Internal identifier
    installed: bool = False
    config_path: Optional[Path] = None
    already_configured: bool = False  # Memory Palace already in config
    description: str = ""


@dataclass
class ConfigResult:
    """Result of configuring a single client."""
    client_id: str
    success: bool
    message: str
    backup_path: Optional[Path] = None


# --- Client Config Path Resolution ---

def _get_home() -> Path:
    """Get user home directory."""
    return Path.home()


def _get_appdata() -> Optional[Path]:
    """Get Windows %APPDATA% path."""
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata)
    return Path.home() / "AppData" / "Roaming"


def _get_claude_desktop_config(plat: PlatformInfo) -> Optional[Path]:
    """Get Claude Desktop config path for current platform."""
    if plat.os == "macos":
        return _get_home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    elif plat.os == "windows":
        return _get_appdata() / "Claude" / "claude_desktop_config.json"
    elif plat.os == "linux":
        return _get_home() / ".config" / "Claude" / "claude_desktop_config.json"
    return None


def _get_claude_desktop_config_wsl(windows_user_dir: Path) -> Optional[Path]:
    """Get Claude Desktop config path on Windows side from WSL."""
    return windows_user_dir / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json"


def _get_claude_code_config() -> Path:
    """Get Claude Code config path (same on all platforms)."""
    return _get_home() / ".claude.json"


def _get_cursor_config() -> Path:
    """Get Cursor MCP config path (same on all platforms)."""
    return _get_home() / ".cursor" / "mcp.json"


def _get_windsurf_config() -> Path:
    """Get Windsurf MCP config path (same on all platforms)."""
    return _get_home() / ".codeium" / "windsurf" / "mcp_config.json"


def _get_continue_config() -> Path:
    """Get VS Code Continue config path (same on all platforms)."""
    return _get_home() / ".continue" / "config.json"


# --- Client Detection ---

def _is_app_installed(plat: PlatformInfo, app_name: str) -> bool:
    """Check if an application is likely installed by looking for common indicators."""
    if plat.os == "macos":
        return Path(f"/Applications/{app_name}.app").exists()
    elif plat.os == "windows":
        # Check common install locations
        program_files = Path(os.environ.get("PROGRAMFILES", "C:/Program Files"))
        local_appdata = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
        return (
            (program_files / app_name).exists() or
            (local_appdata / app_name).exists() or
            (local_appdata / "Programs" / app_name).exists()
        )
    elif plat.os == "linux":
        # Check if binary exists in PATH
        import shutil as sh
        return sh.which(app_name.lower()) is not None
    return False


def _check_client_installed(config_path: Optional[Path], plat: PlatformInfo, app_hints: List[str] = None) -> bool:
    """
    Check if a client is installed.
    
    We consider it installed if:
    1. Its config file already exists, OR
    2. Its config directory exists, OR
    3. The application is found via app_hints
    """
    if config_path and config_path.exists():
        return True
    if config_path and config_path.parent.exists():
        return True
    if app_hints:
        return any(_is_app_installed(plat, hint) for hint in app_hints)
    return False


def _check_already_configured(config_path: Optional[Path]) -> bool:
    """Check if memory-palace is already configured in a client's config."""
    if not config_path or not config_path.exists():
        return False
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        servers = config.get("mcpServers", {})
        return "memory-palace" in servers
    except (json.JSONDecodeError, OSError):
        return False


def discover_clients(plat: PlatformInfo) -> List[ClientInfo]:
    """
    Discover all MCP-compatible AI clients on the system.
    
    Returns a list of ClientInfo with installation status.
    """
    clients = []

    # Claude Desktop
    config_path = _get_claude_desktop_config(plat)
    client = ClientInfo(
        name="Claude Desktop",
        id="claude-desktop",
        config_path=config_path,
        description="Anthropic's desktop app for Claude"
    )
    client.installed = _check_client_installed(config_path, plat, ["Claude"])
    client.already_configured = _check_already_configured(config_path)
    clients.append(client)

    # Claude Code
    config_path = _get_claude_code_config()
    client = ClientInfo(
        name="Claude Code",
        id="claude-code",
        config_path=config_path,
        description="Anthropic's CLI coding assistant"
    )
    # Claude Code is a CLI — check if the command exists
    client.installed = shutil.which("claude") is not None
    client.already_configured = _check_already_configured(config_path)
    clients.append(client)

    # Cursor
    config_path = _get_cursor_config()
    client = ClientInfo(
        name="Cursor",
        id="cursor",
        config_path=config_path,
        description="AI-first code editor"
    )
    client.installed = _check_client_installed(config_path, plat, ["Cursor"])
    client.already_configured = _check_already_configured(config_path)
    clients.append(client)

    # Windsurf
    config_path = _get_windsurf_config()
    client = ClientInfo(
        name="Windsurf",
        id="windsurf",
        config_path=config_path,
        description="Codeium's AI IDE"
    )
    client.installed = _check_client_installed(config_path, plat, ["Windsurf"])
    client.already_configured = _check_already_configured(config_path)
    clients.append(client)

    # VS Code + Continue (different config format)
    config_path = _get_continue_config()
    client = ClientInfo(
        name="VS Code + Continue",
        id="continue",
        config_path=config_path,
        description="Open-source AI coding assistant for VS Code"
    )
    client.installed = _check_client_installed(config_path, plat, [])
    # Continue uses a different config structure
    if config_path and config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            mcp_servers = config.get("experimental", {}).get("modelContextProtocolServers", [])
            client.already_configured = any(
                s.get("name") == "memory-palace" for s in mcp_servers
            )
        except (json.JSONDecodeError, OSError):
            pass
    clients.append(client)

    # WSL: Also check Windows-side clients
    if plat.is_wsl and plat.windows_user_dir:
        wsl_config = _get_claude_desktop_config_wsl(plat.windows_user_dir)
        client = ClientInfo(
            name="Claude Desktop (Windows)",
            id="claude-desktop-windows",
            config_path=wsl_config,
            description="Claude Desktop on the Windows side (from WSL)"
        )
        client.installed = _check_client_installed(wsl_config, plat)
        client.already_configured = _check_already_configured(wsl_config)
        clients.append(client)

        # Cursor Windows-side
        win_cursor = plat.windows_user_dir / ".cursor" / "mcp.json"
        client = ClientInfo(
            name="Cursor (Windows)",
            id="cursor-windows",
            config_path=win_cursor,
            description="Cursor on the Windows side (from WSL)"
        )
        client.installed = _check_client_installed(win_cursor, plat)
        client.already_configured = _check_already_configured(win_cursor)
        clients.append(client)

        # Windsurf Windows-side
        win_windsurf = plat.windows_user_dir / ".codeium" / "windsurf" / "mcp_config.json"
        client = ClientInfo(
            name="Windsurf (Windows)",
            id="windsurf-windows",
            config_path=win_windsurf,
            description="Windsurf on the Windows side (from WSL)"
        )
        client.installed = _check_client_installed(win_windsurf, plat)
        client.already_configured = _check_already_configured(win_windsurf)
        clients.append(client)

    return clients


# --- MCP Server Entry Generation ---

def build_mcp_entry(
    install_dir: Path,
    plat: PlatformInfo,
    client_id: str = "",
) -> Dict[str, Any]:
    """
    Build the MCP server entry for memory-palace.
    
    Args:
        install_dir: Where memory-palace is installed
        plat: Platform info for path resolution
        client_id: Client being configured (affects path format for WSL cross-platform)
    
    Returns:
        MCP server config dict
    """
    # Determine the Python executable path in the venv
    if plat.os == "windows" or client_id.endswith("-windows"):
        # Windows paths
        if plat.is_wsl and client_id.endswith("-windows"):
            # WSL → Windows: need to translate the path
            # install_dir is a Linux path, we need the Windows equivalent
            # This is tricky — the install might be in WSL filesystem or Windows filesystem
            # If installed to /mnt/c/..., we can translate directly
            install_str = str(install_dir)
            if install_str.startswith("/mnt/"):
                # /mnt/c/Users/X/memory-palace → C:\Users\X\memory-palace
                drive = install_str[5]  # 'c'
                rest = install_str[7:]  # 'Users/X/memory-palace'
                win_install = f"{drive.upper()}:\\{rest.replace('/', '\\')}"
                python_path = f"{win_install}\\venv\\Scripts\\python.exe"
                cwd = win_install
            else:
                # Installed in WSL filesystem — use wsl.exe wrapper
                python_path = "wsl.exe"
                return {
                    "command": python_path,
                    "args": [
                        "-d", _get_wsl_distro(),
                        "--", str(install_dir / "venv" / "bin" / "python"),
                        "-m", "mcp_server.server"
                    ],
                    "cwd": str(install_dir)
                }
        else:
            python_path = str(install_dir / "venv" / "Scripts" / "python.exe")
            cwd = str(install_dir)
    else:
        python_path = str(install_dir / "venv" / "bin" / "python")
        cwd = str(install_dir)

    return {
        "command": python_path,
        "args": ["-m", "mcp_server.server"],
        "cwd": cwd
    }


def _get_wsl_distro() -> str:
    """Get the current WSL distribution name."""
    try:
        with open("/etc/os-release", "r") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    return line.split("=", 1)[1].strip().strip('"')
    except (OSError, IOError):
        pass
    return os.environ.get("WSL_DISTRO_NAME", "Ubuntu")


# --- Configuration ---

def _backup_config(config_path: Path) -> Optional[Path]:
    """Create a backup of an existing config file. Returns backup path."""
    if not config_path.exists():
        return None

    backup_path = config_path.with_suffix(".json.backup")
    counter = 1
    while backup_path.exists():
        backup_path = config_path.with_suffix(f".json.backup.{counter}")
        counter += 1

    shutil.copy2(config_path, backup_path)
    return backup_path


def configure_client(
    client: ClientInfo,
    install_dir: Path,
    plat: PlatformInfo,
    force: bool = False,
) -> ConfigResult:
    """
    Configure a single AI client to use Memory Palace.
    
    Args:
        client: The client to configure
        install_dir: Where memory-palace is installed
        plat: Platform info
        force: If True, overwrite existing memory-palace config
    
    Returns:
        ConfigResult with success status and details
    """
    if not client.config_path:
        return ConfigResult(
            client_id=client.id,
            success=False,
            message="No config path known for this client"
        )

    if client.already_configured and not force:
        return ConfigResult(
            client_id=client.id,
            success=True,
            message="Memory Palace already configured (skipped)"
        )

    # Build the MCP entry
    mcp_entry = build_mcp_entry(install_dir, plat, client.id)

    # Handle Continue separately (different config structure)
    if client.id == "continue":
        return _configure_continue(client, mcp_entry)

    # Standard MCP config format (Claude Desktop, Cursor, Windsurf, Claude Code)
    config_path = client.config_path

    # Load existing config
    existing_config = {}
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                existing_config = json.load(f)
        except json.JSONDecodeError:
            # Invalid JSON — we'll create a fresh one
            pass

    # Backup
    backup_path = _backup_config(config_path)

    # Merge
    if "mcpServers" not in existing_config:
        existing_config["mcpServers"] = {}
    existing_config["mcpServers"]["memory-palace"] = mcp_entry

    # Write
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(existing_config, f, indent=2)

    return ConfigResult(
        client_id=client.id,
        success=True,
        message=f"Configured {client.name}",
        backup_path=backup_path
    )


def _configure_continue(client: ClientInfo, mcp_entry: Dict[str, Any]) -> ConfigResult:
    """Configure VS Code Continue (different config format)."""
    config_path = client.config_path
    if not config_path:
        return ConfigResult(client_id=client.id, success=False, message="No config path")

    existing_config = {}
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                existing_config = json.load(f)
        except json.JSONDecodeError:
            pass

    backup_path = _backup_config(config_path)

    # Continue uses experimental.modelContextProtocolServers array
    if "experimental" not in existing_config:
        existing_config["experimental"] = {}
    if "modelContextProtocolServers" not in existing_config["experimental"]:
        existing_config["experimental"]["modelContextProtocolServers"] = []

    servers = existing_config["experimental"]["modelContextProtocolServers"]

    # Remove existing memory-palace entry if present
    servers = [s for s in servers if s.get("name") != "memory-palace"]

    # Add our entry in Continue's format
    servers.append({
        "name": "memory-palace",
        "command": mcp_entry["command"],
        "args": mcp_entry.get("args", []),
        "cwd": mcp_entry.get("cwd")
    })

    existing_config["experimental"]["modelContextProtocolServers"] = servers

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(existing_config, f, indent=2)

    return ConfigResult(
        client_id=client.id,
        success=True,
        message=f"Configured {client.name}",
        backup_path=backup_path
    )


def configure_clients(
    clients: List[ClientInfo],
    selected_ids: List[str],
    install_dir: Path,
    plat: PlatformInfo,
) -> List[ConfigResult]:
    """
    Configure multiple AI clients.
    
    Args:
        clients: All discovered clients
        selected_ids: IDs of clients the user wants to configure
        install_dir: Where memory-palace is installed
        plat: Platform info
    
    Returns:
        List of ConfigResult, one per selected client
    """
    results = []
    for client in clients:
        if client.id in selected_ids:
            result = configure_client(client, install_dir, plat)
            results.append(result)
    return results
