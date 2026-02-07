"""
Forget tool for Claude Memory Palace MCP server.
"""
from typing import Any, Optional

from memory_palace.services import forget
from mcp_server.toon_wrapper import toon_response


def register_forget(mcp):
    """Register the forget tool with the MCP server."""

    @mcp.tool()
    @toon_response
    async def memory_forget(
        memory_id: int,
        reason: Optional[str] = None,
        toon: Optional[bool] = None
    ) -> dict[str, Any]:
        """
        Archive a memory (soft delete).

        Claude chooses what to forget. Memories are not permanently deleted,
        just marked as archived and excluded from normal searches.

        Args:
            memory_id: ID of the memory to archive
            reason: Optional reason for archiving

        Returns:
            Compact confirmation: "Archived memory X (subject)"
        """
        return forget(
            memory_id=memory_id,
            reason=reason
        )
