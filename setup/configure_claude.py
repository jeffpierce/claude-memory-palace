#!/usr/bin/env python3
"""
Auto-configure Claude Desktop to use the Memory Palace MCP server.

Usage:
    python -m memory_palace.setup.configure_claude

This script:
1. Detects your OS (Windows/macOS/Linux)
2. Finds the Claude Desktop config file
3. Backs up existing config
4. Merges memory-palace MCP server entry (preserves other servers)
5. Writes updated config with user confirmation
"""

import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class ConfigureResult:
    """Result of a configure operation."""

    def __init__(self, success: bool, message: str, backup_path: Optional[Path] = None):
        self.success = success
        self.message = message
        self.backup_path = backup_path

    def __repr__(self) -> str:
        return f"ConfigureResult(success={self.success}, message={self.message!r})"


def get_claude_config_path() -> Tuple[Path, str]:
    """
    Get the Claude Desktop config file path for the current OS.

    Returns:
        Tuple of (config_path, os_name)
    """
    system = platform.system().lower()

    if system == "windows":
        # Windows: %APPDATA%\Claude\claude_desktop_config.json
        appdata = os.environ.get("APPDATA")
        if appdata:
            config_path = Path(appdata) / "Claude" / "claude_desktop_config.json"
        else:
            # Fallback if APPDATA not set
            config_path = Path.home() / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json"
        return config_path, "Windows"

    elif system == "darwin":
        # macOS: ~/Library/Application Support/Claude/claude_desktop_config.json
        config_path = Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
        return config_path, "macOS"

    else:
        # Linux: ~/.config/Claude/claude_desktop_config.json (best guess)
        config_path = Path.home() / ".config" / "Claude" / "claude_desktop_config.json"
        return config_path, "Linux"


def get_memory_palace_cwd() -> Path:
    """
    Get the working directory for the memory-palace MCP server.

    Returns the claude-memory-palace project root directory.
    """
    # This script is at: claude-memory-palace/memory_palace/setup/configure_claude.py
    # We want: claude-memory-palace/
    script_path = Path(__file__).resolve()
    return script_path.parent.parent.parent


def create_mcp_server_entry(cwd_override: Optional[str] = None) -> Dict[str, Any]:
    """
    Create the MCP server configuration entry for memory-palace.

    Args:
        cwd_override: Optional path to use instead of auto-detected path

    Returns:
        Dict containing the MCP server configuration
    """
    if cwd_override:
        cwd = cwd_override
    else:
        cwd = str(get_memory_palace_cwd())

    # Check if a venv exists in the package directory and use its Python
    cwd_path = Path(cwd)
    system = platform.system().lower()
    
    if system == "windows":
        venv_python = cwd_path / "venv" / "Scripts" / "python.exe"
    else:
        venv_python = cwd_path / "venv" / "bin" / "python"
    
    if venv_python.exists():
        command = str(venv_python)
    else:
        command = "python"

    return {
        "command": command,
        "args": ["-m", "mcp_server.server"],
        "cwd": cwd
    }


def load_existing_config(config_path: Path) -> Dict[str, Any]:
    """
    Load existing Claude Desktop config, or return empty dict if not found.

    Args:
        config_path: Path to the config file

    Returns:
        Existing config dict or empty dict with mcpServers key
    """
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse existing config: {e}")
            print("Creating new config...")

    return {"mcpServers": {}}


def backup_config(config_path: Path) -> Optional[Path]:
    """
    Create a backup of the existing config file.

    Args:
        config_path: Path to the config file

    Returns:
        Path to backup file, or None if original doesn't exist
    """
    if not config_path.exists():
        return None

    backup_path = config_path.with_suffix(".json.backup")

    # If backup already exists, add number suffix
    counter = 1
    while backup_path.exists():
        backup_path = config_path.with_suffix(f".json.backup.{counter}")
        counter += 1

    shutil.copy2(config_path, backup_path)
    return backup_path


def merge_config(existing: Dict[str, Any], memory_palace_entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge memory-palace entry into existing config without overwriting other servers.

    Args:
        existing: Existing config dict
        memory_palace_entry: The memory-palace server configuration

    Returns:
        Merged config dict
    """
    # Ensure mcpServers key exists
    if "mcpServers" not in existing:
        existing["mcpServers"] = {}

    # Add/update memory-palace entry
    existing["mcpServers"]["memory-palace"] = memory_palace_entry

    return existing


def prompt_user(message: str) -> bool:
    """
    Prompt user for yes/no confirmation.

    Args:
        message: The prompt message

    Returns:
        True if user confirms, False otherwise
    """
    while True:
        response = input(f"{message} [y/n]: ").strip().lower()
        if response in ("y", "yes"):
            return True
        if response in ("n", "no"):
            return False
        print("Please enter 'y' or 'n'")


def configure_claude_desktop(interactive: bool = True, cwd_override: Optional[str] = None) -> ConfigureResult:
    """
    Configure Claude Desktop to use the Memory Palace MCP server.

    Args:
        interactive: If True, prompts user for confirmation. If False, just does it.
        cwd_override: Optional path to use for the MCP server cwd instead of auto-detected.

    Returns:
        ConfigureResult with success status and message
    """
    # Step 1: Detect OS and config path
    config_path, os_name = get_claude_config_path()

    # Step 2: Check if Claude Desktop is installed
    config_dir = config_path.parent
    if not config_dir.exists():
        return ConfigureResult(
            success=False,
            message=f"Claude Desktop config directory not found at {config_dir}. Is Claude Desktop installed?"
        )

    # Step 3: Get memory-palace MCP server entry
    memory_palace_entry = create_mcp_server_entry(cwd_override)
    cwd = memory_palace_entry["cwd"]

    # Verify the MCP server exists
    mcp_server_path = Path(cwd) / "mcp_server" / "server.py"
    if not mcp_server_path.exists():
        return ConfigureResult(
            success=False,
            message=f"MCP server not found at {mcp_server_path}"
        )

    # Step 4: Load existing config
    existing_config = load_existing_config(config_path)

    # Check if memory-palace is already configured (in interactive mode, ask to update)
    if interactive and "mcpServers" in existing_config and "memory-palace" in existing_config["mcpServers"]:
        existing_entry = existing_config["mcpServers"]["memory-palace"]
        print("Memory-palace is already configured in Claude Desktop:")
        print(f"  {json.dumps(existing_entry, indent=2)}")
        print()
        if not prompt_user("Do you want to update the configuration?"):
            return ConfigureResult(success=True, message="No changes made (user declined)")

    # Step 5: Backup existing config
    backup_path = None
    if config_path.exists():
        backup_path = backup_config(config_path)

    # Step 6: Merge configs
    merged_config = merge_config(existing_config, memory_palace_entry)

    # In interactive mode, show what will be written and ask for confirmation
    if interactive:
        print("The following configuration will be written:")
        print("-" * 40)
        print(json.dumps(merged_config, indent=2))
        print("-" * 40)
        print()
        if not prompt_user("Write this configuration?"):
            return ConfigureResult(success=True, message="No changes made (user declined)")

    # Step 7: Write config
    try:
        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(merged_config, f, indent=2)

        return ConfigureResult(
            success=True,
            message="Claude Desktop configured successfully",
            backup_path=backup_path
        )

    except IOError as e:
        return ConfigureResult(
            success=False,
            message=f"Error writing config file: {e}"
        )


def main() -> int:
    """
    Main entry point for Claude Desktop configuration.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    print("=" * 60)
    print("Memory Palace - Claude Desktop Configuration")
    print("=" * 60)
    print()

    # Step 1: Detect OS and config path
    config_path, os_name = get_claude_config_path()
    print(f"Detected OS: {os_name}")
    print(f"Config path: {config_path}")
    print()

    # Step 2: Check if Claude Desktop is installed
    config_dir = config_path.parent
    if not config_dir.exists():
        print(f"Warning: Claude Desktop config directory not found at:")
        print(f"  {config_dir}")
        print()
        print("This usually means Claude Desktop is not installed.")
        print("Please install Claude Desktop first, then run this script again.")
        print()
        print("Download Claude Desktop from: https://claude.ai/download")
        return 1

    # Step 3: Get memory-palace MCP server entry
    memory_palace_entry = create_mcp_server_entry()
    cwd = memory_palace_entry["cwd"]
    print(f"Memory Palace project directory: {cwd}")
    print()

    # Verify the MCP server exists
    mcp_server_path = Path(cwd) / "mcp_server" / "server.py"
    if not mcp_server_path.exists():
        print(f"Error: MCP server not found at expected location:")
        print(f"  {mcp_server_path}")
        print()
        print("Please ensure you're running this script from the correct location.")
        return 1

    # Step 4: Load existing config
    existing_config = load_existing_config(config_path)

    # Check if memory-palace is already configured
    if "mcpServers" in existing_config and "memory-palace" in existing_config["mcpServers"]:
        existing_entry = existing_config["mcpServers"]["memory-palace"]
        print("Memory-palace is already configured in Claude Desktop:")
        print(f"  {json.dumps(existing_entry, indent=2)}")
        print()
        if not prompt_user("Do you want to update the configuration?"):
            print("No changes made.")
            return 0

    # Step 5: Backup existing config
    if config_path.exists():
        backup_path = backup_config(config_path)
        if backup_path:
            print(f"Backed up existing config to: {backup_path}")
            print()

    # Step 6: Merge configs
    merged_config = merge_config(existing_config, memory_palace_entry)

    # Show what will be written
    print("The following configuration will be written:")
    print("-" * 40)
    print(json.dumps(merged_config, indent=2))
    print("-" * 40)
    print()

    # Step 7: Ask for permission
    if not prompt_user("Write this configuration?"):
        print("No changes made.")
        return 0

    # Step 8: Write config
    try:
        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(merged_config, f, indent=2)

        print()
        print("=" * 60)
        print("Configuration written successfully!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Restart Claude Desktop for changes to take effect")
        print("2. In Claude Desktop, verify the memory-palace server is active")
        print("3. Test by asking your AI assistant to remember something")
        print()
        print("Example test:")
        print('  User: "Remember that my favorite color is blue"')
        print('  (new conversation)')
        print('  User: "What is my favorite color?"')
        print()
        return 0

    except IOError as e:
        print(f"Error writing config file: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
