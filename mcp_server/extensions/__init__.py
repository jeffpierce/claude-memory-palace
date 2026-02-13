"""
Extension loading system for Memory Palace MCP server.

Extensions are Python modules that register additional tools on the MCP server.
Each extension must provide a register(mcp) function.
"""
import importlib
import sys
from typing import Optional

from memory_palace.config_v2 import load_config


def load_extensions(mcp) -> None:
    """
    Load and register all configured extensions.

    Extensions are specified in config under "extensions" key as a list of
    module paths (e.g., "mcp_server.extensions.switch_db").

    Each extension module must provide a register(mcp) function that takes
    the FastMCP server instance and registers its tools.

    Extension loading errors are logged to stderr but do not crash the server.

    Args:
        mcp: FastMCP server instance to register tools on
    """
    config = load_config()
    extensions = config.get("extensions", [])

    if not extensions:
        return  # No extensions configured

    print(f"Loading {len(extensions)} extension(s)...", file=sys.stderr)

    for extension_path in extensions:
        try:
            # Import the extension module
            module = importlib.import_module(extension_path)

            # Verify it has a register function
            if not hasattr(module, "register"):
                print(
                    f"  [ERROR] {extension_path}: missing register() function",
                    file=sys.stderr
                )
                continue

            # Call the register function
            module.register(mcp)
            print(f"  [OK] {extension_path}", file=sys.stderr)

        except ImportError as e:
            print(
                f"  [ERROR] {extension_path}: import failed ({e})",
                file=sys.stderr
            )
        except Exception as e:
            print(
                f"  [ERROR] {extension_path}: registration failed ({e})",
                file=sys.stderr
            )


__all__ = ["load_extensions"]
