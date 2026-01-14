"""
Forget tool for Claude Memory Palace MCP server.
"""
from typing import Any, Optional

from memory_palace.services import forget


def register_forget(mcp):
    """Register the forget tool with the MCP server."""

    @mcp.tool()
    async def memory_forget(
        memory_id: int,
        reason: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Archive a memory (soft delete).

        Claude chooses what to forget. Memories are not permanently deleted,
        just marked as archived and excluded from normal searches.

        Args:
            memory_id: ID of the memory to archive
            reason: Optional reason for archiving

        Returns:
            Dict with success status and message
        """
        return forget(
            memory_id=memory_id,
            reason=reason
        )
