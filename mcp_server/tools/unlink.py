"""
Unlink tool for Memory Palace MCP server.
"""
from typing import Any, Optional

from memory_palace.services import unlink_memories
from mcp_server.toon_wrapper import toon_response


def register_unlink(mcp):
    """Register the memory_unlink tool with the MCP server."""

    @mcp.tool()
    @toon_response
    async def memory_unlink(
        source_id: int,
        target_id: int,
        relation_type: Optional[str] = None,
        database: Optional[str] = None
    ) -> dict[str, Any]:
        # Remove edge(s). relation_type=None removes ALL edges source→target.
        """
        Remove edge(s) between memories.

        relation_type: Specific type or None=ALL.
        """
        return unlink_memories(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            database=database
        )
