"""
Memory stats tool for Claude Memory Palace MCP server.
"""
from typing import Any

from memory_palace.services import get_memory_stats


def register_memory_stats(mcp):
    """Register the memory_stats tool with the MCP server."""

    @mcp.tool()
    async def memory_stats() -> dict[str, Any]:
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
