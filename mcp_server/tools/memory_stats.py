"""Memory stats tool for Memory Palace MCP server."""
from typing import Any

from memory_palace.services import get_memory_stats
from mcp_server.toon_wrapper import toon_response


def register_memory_stats(mcp):
    """Register the memory_stats tool with the MCP server."""

    @mcp.tool()
    @toon_response
    async def memory_stats() -> dict[str, Any]:
        """
        Overview statistics: total memories, counts by type/instance/project,
        foundational count, most accessed, and recently added.

        Returns:
            Dictionary with memory statistics
        """
        return get_memory_stats()
