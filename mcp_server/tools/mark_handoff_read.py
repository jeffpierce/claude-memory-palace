"""
Mark handoff read tool for Claude Memory Palace MCP server.
"""
from typing import Any

from memory_palace.services import mark_handoff_read


def register_mark_handoff_read(mcp):
    """Register the mark_handoff_read tool with the MCP server."""

    @mcp.tool()
    async def handoff_mark_read(
        message_id: int,
        read_by: str
    ) -> dict[str, Any]:
        """
        Mark a handoff message as read after processing it.

        Args:
            message_id: ID of the message to mark as read
            read_by: Which instance read it (e.g., "desktop", "code", "web")

        Returns:
            Dict with success status
        """
        return mark_handoff_read(
            message_id=message_id,
            read_by=read_by
        )
