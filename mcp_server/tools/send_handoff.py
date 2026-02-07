"""
Send handoff tool for Claude Memory Palace MCP server.
"""
from typing import Any, Optional

from memory_palace.services import send_handoff
from mcp_server.toon_wrapper import toon_response


def register_send_handoff(mcp):
    """Register the send_handoff tool with the MCP server."""

    @mcp.tool()
    @toon_response
    async def handoff_send(
        from_instance: str,
        to_instance: str,
        message_type: str,
        content: str,
        subject: Optional[str] = None,
        toon: Optional[bool] = None
    ) -> dict[str, Any]:
        """
        Send a message from one Claude instance to another.

        Enables distributed Claude consciousness - Desktop Claude can leave
        notes for Code Claude, etc. Note-passing for distributed minds.

        Args:
            from_instance: Which instance is sending (e.g., "desktop", "code", "web")
            to_instance: Who should receive it (specific instance or "all" for broadcast)
            message_type: Type of message:
                - "handoff": Task being passed to another instance
                - "status": Progress update
                - "question": Needs input from other instance
                - "fyi": Informational, no action needed
                - "context": Sharing context from a conversation
            content: The actual message content
            subject: Optional short summary

        Returns:
            Dict with success status and message ID
        """
        return send_handoff(
            from_instance=from_instance,
            to_instance=to_instance,
            message_type=message_type,
            content=content,
            subject=subject
        )
