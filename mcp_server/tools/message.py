"""
Message tool for Memory Palace MCP server.
"""
import shlex
import subprocess
import sys
from typing import Any, Optional

from memory_palace.services.message_service import (
    send_message,
    get_messages,
    mark_message_read,
    mark_message_unread,
    subscribe,
    unsubscribe,
)
from memory_palace.config_v2 import get_notify_command
from mcp_server.toon_wrapper import toon_response


def _execute_notify_hook(
    command_template: str,
    send_result: dict,
    from_instance: str,
    to_instance: str,
    message_type: str,
    subject: Optional[str],
    channel: Optional[str],
    priority: int,
) -> None:
    """
    Fire-and-forget post-send notification. Never raises.

    Executes the configured notification command with template variable
    substitution. All values are shell-escaped before substitution.

    Args:
        command_template: Shell command template with {variables}
        send_result: Result dict from send_message (contains message_id)
        from_instance: Sender instance
        to_instance: Recipient instance
        message_type: Type of message
        subject: Message subject (may be None)
        channel: Channel name (may be None)
        priority: Message priority
    """
    try:
        # Extract message_id from result
        message_id = send_result.get("id", "")

        # Build template variables dict
        template_vars = {
            "from_instance": str(from_instance),
            "to_instance": str(to_instance),
            "message_type": str(message_type),
            "subject": str(subject) if subject is not None else "",
            "channel": str(channel) if channel is not None else "",
            "priority": str(priority),
            "message_id": str(message_id),
        }

        # Shell-escape all values
        escaped_vars = {k: shlex.quote(v) for k, v in template_vars.items()}

        # Substitute into command template
        command = command_template.format(**escaped_vars)

        # Execute with timeout (fire-and-forget)
        subprocess.run(
            command,
            shell=True,
            timeout=5,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        # Log but don't fail the send
        print(
            f"Warning: Notification command timed out after 5s",
            file=sys.stderr,
        )
    except Exception as e:
        # Log but don't fail the send
        print(
            f"Warning: Notification hook failed: {e}",
            file=sys.stderr,
        )


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
            result = send_message(
                from_instance=from_instance,
                to_instance=to_instance,
                content=content,
                message_type=message_type,
                subject=subject,
                channel=channel,
                priority=priority,
            )

            # Execute post-send notification hook if configured
            notify_cmd = get_notify_command()
            if notify_cmd is not None and result.get("success"):
                _execute_notify_hook(
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
