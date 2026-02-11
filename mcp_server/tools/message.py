"""
Message tool for Memory Palace MCP server.
"""
import json as _json
import shlex
import subprocess
import sys
import urllib.request
from typing import Any, Dict, Optional

from memory_palace.services.message_service import (
    send_message,
    get_messages,
    mark_message_read,
    mark_message_unread,
    subscribe,
    unsubscribe,
)
from memory_palace.config_v2 import (
    get_notify_command,
    get_instance_route,
    get_instance_routes,
)
from mcp_server.toon_wrapper import toon_response


def _execute_openclaw_wake(
    route: Dict[str, str],
    from_instance: str,
    to_instance: str,
    message_type: str,
    subject: Optional[str],
    message_id: Any,
    priority: int,
) -> None:
    """
    Fire-and-forget HTTP wake to an OpenClaw gateway. Never raises.

    Sends a POST to the gateway's /hooks/agent endpoint to deliver a
    system event to the target agent session. Uses sessionKey for
    targeted delivery (not broadcast). Priority >= 5 uses wakeMode "now"
    (immediate), lower priority uses "next-heartbeat".

    Args:
        route: Dict with "gateway" (URL), "token" (auth secret), and
               optional "session" (agent session key) keys
        from_instance: Sender instance ID
        to_instance: Recipient instance ID
        message_type: Type of message sent
        subject: Message subject (may be None)
        message_id: ID of the sent message
        priority: Message priority (0-10)
    """
    try:
        gateway_url = route["gateway"].rstrip("/")
        token = route.get("token", "")
        session_key = route.get("session")

        wake_text = (
            f"Palace message from {from_instance}: "
            f"{subject or message_type} (msg #{message_id})"
        )

        payload_dict = {
            "message": wake_text,
            "wakeMode": "now" if priority >= 5 else "next-heartbeat",
        }
        if session_key:
            payload_dict["sessionKey"] = session_key

        payload = _json.dumps(payload_dict).encode("utf-8")

        req = urllib.request.Request(
            f"{gateway_url}/hooks/agent",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
            method="POST",
        )

        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        print(
            f"Warning: OpenClaw wake failed for {to_instance}: {e}",
            file=sys.stderr,
        )


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
                    _execute_openclaw_wake(route=route, **_notify_params)
                elif to_instance == "all":
                    # Broadcast: wake all routed instances except sender
                    for inst_id, inst_route in get_instance_routes().items():
                        if inst_id != from_instance:
                            _execute_openclaw_wake(
                                route=inst_route,
                                **{**_notify_params, "to_instance": inst_id},
                            )

                # 2. Fallback to notify_command (shell exec) — backwards compat
                notify_cmd = get_notify_command()
                if notify_cmd is not None:
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
