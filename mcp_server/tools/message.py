"""
Message tool for Memory Palace MCP server.
"""
from typing import Any, Optional

from memory_palace.services.message_service import (
    send_message,
    get_messages,
    mark_message_read,
    mark_message_unread,
    subscribe,
    unsubscribe,
    execute_openclaw_wake,
    execute_notify_hook,
)
from memory_palace.config_v2 import (
    get_notify_command,
    get_instance_route,
    get_instance_routes,
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
        # Inter-instance messaging with pubsub. Replaces handoff_send/get/mark_read.
        # Postgres: real-time NOTIFY. SQLite: polling.
        """
        Inter-instance messaging with pubsub.

        Actions: send, get, mark_read, mark_unread, subscribe, unsubscribe
        Types: handoff, status, question, fyi, context, event, message
        to_instance: "all" for broadcast
        priority: 0-10 (default 0)
        """
        if action == "send":
            if not from_instance or not to_instance or not content:
                return {"error": "send requires from_instance, to_instance, and content"}
            result = send_message(
                from_instance=from_instance,
                to_instance=to_instance,
                content=content,
                message_type=message_type,
                subject=subject,
                channel=channel,
                priority=priority,
            )

            # Post-send notifications (fire-and-forget, never fail the send)
            if result.get("success"):
                _notify_params = {
                    "from_instance": from_instance,
                    "to_instance": to_instance,
                    "message_type": message_type,
                    "subject": subject,
                    "message_id": result.get("id", ""),
                    "priority": priority,
                }

                # 1. Try instance_routes (HTTP wake) — preferred
                route = get_instance_route(to_instance)
                if route:
                    execute_openclaw_wake(route=route, **_notify_params)
                elif to_instance == "all":
                    # Broadcast: wake all routed instances except sender
                    for inst_id, inst_route in get_instance_routes().items():
                        if inst_id != from_instance:
                            execute_openclaw_wake(
                                route=inst_route,
                                **{**_notify_params, "to_instance": inst_id},
                            )

                # 2. Fallback to notify_command (shell exec) — backwards compat
                notify_cmd = get_notify_command()
                if notify_cmd is not None:
                    execute_notify_hook(
                        command_template=notify_cmd,
                        send_result=result,
                        from_instance=from_instance,
                        to_instance=to_instance,
                        message_type=message_type,
                        subject=subject,
                        channel=channel,
                        priority=priority,
                    )

            return result
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
