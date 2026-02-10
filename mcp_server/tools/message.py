"""
Message tool for Claude Memory Palace MCP server.
"""
from typing import Any, Optional

from memory_palace.services.message_service import (
    send_message,
    get_messages,
    mark_message_read,
    mark_message_unread,
    subscribe,
    unsubscribe,
)
from mcp_server.toon_wrapper import toon_response


def register_message(mcp):
    """Register the message tool with the MCP server."""

    @mcp.tool()
    @toon_response
    async def message(
        action: str,
        # Common params
        instance_id: Optional[str] = None,
        # Send params
        from_instance: Optional[str] = None,
        to_instance: Optional[str] = None,
        content: Optional[str] = None,
        message_type: str = "message",
        subject: Optional[str] = None,
        channel: Optional[str] = None,
        priority: int = 0,
        # Get params
        unread_only: bool = True,
        limit: int = 50,
        # Mark params
        message_id: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Inter-instance messaging with pubsub support. Replaces handoff_send/get/mark_read.

        Actions:
          send — Send message. Requires from_instance, to_instance, content.
          get — Get messages. Requires instance_id. Optional: channel, message_type, unread_only, limit.
          mark_read — Mark read. Requires message_id, instance_id.
          mark_unread — Mark unread. Requires message_id.
          subscribe — Subscribe to channel. Requires instance_id, channel.
          unsubscribe — Unsubscribe. Requires instance_id, channel.

        Message types: handoff, status, question, fyi, context, event, message.
        Postgres uses NOTIFY for real-time delivery; SQLite uses polling.

        Args:
            action: One of: send, get, mark_read, mark_unread, subscribe, unsubscribe
            instance_id: Instance performing the action (for get/mark_read/subscribe)
            from_instance: Sender (for send)
            to_instance: Recipient or "all" for broadcast (for send)
            content: Message content (for send)
            message_type: Type of message (default "message")
            subject: Optional short summary (for send)
            channel: Channel name (for send/get/subscribe/unsubscribe)
            priority: 0-10, higher = more urgent (for send, default 0)
            unread_only: Only unread messages (for get, default True)
            limit: Max messages to return (for get, default 50)
            message_id: Message ID (for mark_read/mark_unread)

        Returns:
            Action-specific result dict
        """
        if action == "send":
            if not from_instance or not to_instance or not content:
                return {"error": "send requires from_instance, to_instance, and content"}
            return send_message(
                from_instance=from_instance,
                to_instance=to_instance,
                content=content,
                message_type=message_type,
                subject=subject,
                channel=channel,
                priority=priority,
            )
        elif action == "get":
            if not instance_id:
                return {"error": "get requires instance_id"}
            return get_messages(
                instance_id=instance_id,
                unread_only=unread_only,
                channel=channel,
                message_type=message_type if message_type != "message" else None,
                limit=limit,
            )
        elif action == "mark_read":
            if message_id is None or not instance_id:
                return {"error": "mark_read requires message_id and instance_id"}
            return mark_message_read(message_id=message_id, instance_id=instance_id)
        elif action == "mark_unread":
            if message_id is None:
                return {"error": "mark_unread requires message_id"}
            return mark_message_unread(message_id=message_id)
        elif action == "subscribe":
            if not instance_id or not channel:
                return {"error": "subscribe requires instance_id and channel"}
            return subscribe(instance_id=instance_id, channel=channel)
        elif action == "unsubscribe":
            if not instance_id or not channel:
                return {"error": "unsubscribe requires instance_id and channel"}
            return unsubscribe(instance_id=instance_id, channel=channel)
        else:
            return {"error": f"Unknown action: {action}. Valid: send, get, mark_read, mark_unread, subscribe, unsubscribe"}
