"""
Memory stats tool for Claude Memory Palace MCP server.
"""
from typing import Any, Optional

from memory_palace.services import get_memory_stats
from mcp_server.toon_wrapper import toon_response


def register_memory_stats(mcp):
    """Register the memory_stats tool with the MCP server."""

    @mcp.tool()
    @toon_response
    async def memory_stats(toon: Optional[bool] = None) -> dict[str, Any]:
        """
        Get overview statistics of the memory system.

        Returns stats on:
        - Total memories (active and archived)
        - Counts by type
        - Counts by instance
        - Counts by project
        - Average importance
        - Most accessed memories
        - Recently added memories

        Returns:
            Dictionary with memory statistics
        """
        return get_memory_stats()
