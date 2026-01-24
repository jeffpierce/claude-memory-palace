"""
MCP Server for Claude Memory Palace.

Provides tools for memory storage, retrieval, and inter-instance handoff.

Run with: python -m mcp_server.server
Or: python mcp_server/server.py
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mcp.server import FastMCP
import json

from memory_palace.database import init_db
from mcp_server.tools import register_all_tools

# Initialize the MCP server using FastMCP (has .tool() decorator)
server = FastMCP("memory-palace")

# Alias for backwards compatibility
mcp = server

# Register all tools
register_all_tools(server)


async def main_async():
    """Run the MCP server (async)."""
    # Initialize database
    init_db()

    # Run server with stdio transport (FastMCP has run_stdio_async)
    await server.run_stdio_async()


def main():
    """Entry point for script installation."""
    import asyncio
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
