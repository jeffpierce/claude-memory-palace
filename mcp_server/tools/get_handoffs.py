"""
Get handoffs tool for Claude Memory Palace MCP server.
"""
from typing import Any, Optional

from memory_palace.services import get_handoffs
from mcp_server.toon_wrapper import toon_response


def register_get_handoffs(mcp):
    """Register the get_handoffs tool with the MCP server."""

    @mcp.tool()
    @toon_response
    async def handoff_get(
        for_instance: str,
        unread_only: bool = True,
        message_type: Optional[str] = None,
        limit: int = 50,
        toon: Optional[bool] = None
    ) -> dict[str, Any]:
        """
        Check for messages from other Claude instances.

        Run this to see if Desktop Claude or Web Claude left you any notes.

        Args:
            for_instance: Which instance is checking their inbox (e.g., "desktop", "code", "web")
            unread_only: Only return unread messages (default true)
            message_type: Filter by type (optional): handoff, status, question, fyi, context
            limit: Maximum messages to return (default 50)

        Returns:
            Dict with count and list of messages
        """
        return get_handoffs(
            for_instance=for_instance,
            unread_only=unread_only,
            message_type=message_type,
            limit=limit
        )
