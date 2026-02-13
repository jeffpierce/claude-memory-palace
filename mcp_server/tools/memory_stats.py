"""Memory stats tool for Memory Palace MCP server."""
from typing import Any, Optional

from memory_palace.services import get_memory_stats
from mcp_server.toon_wrapper import toon_response


def register_memory_stats(mcp):
    """Register the memory_stats tool with the MCP server."""

    @mcp.tool()
    @toon_response
    async def memory_stats(
        database: Optional[str] = None,
    ) -> dict[str, Any]:
        # Stats: total, counts by type/instance/project, foundational, most accessed, recent.
        """
        Overview statistics.
        """
        return get_memory_stats(database=database)
