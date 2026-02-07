"""
Unlink tool for Claude Memory Palace MCP server.
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
        toon: Optional[bool] = None
    ) -> dict[str, Any]:
        """
        Remove relationship edge(s) between two memories.

        If relation_type is specified, only removes that specific edge.
        If relation_type is None, removes ALL edges from source to target.

        Note: This only removes edges in the source->target direction.
        To remove a bidirectional relationship, you may need to call twice
        (once for each direction) if edges were created separately.

        Args:
            source_id: ID of the source memory
            target_id: ID of the target memory
            relation_type: Specific relation to remove (optional - all if None)

        Returns:
            Dict with count of removed edges
        """
        return unlink_memories(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type
        )
