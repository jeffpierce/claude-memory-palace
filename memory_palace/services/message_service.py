"""
Message pubsub service for inter-instance communication.

Replaces handoff_service.py with a real pubsub message system supporting:
- Direct messages (instance-to-instance)
- Channel-based publish/subscribe
- Message priorities, TTL, and read tracking
- Postgres NOTIFY/LISTEN for real-time delivery
- SQLite polling fallback for portability

Valid instances are configured in ~/.memory-palace/config.json under the "instances" key.
"""

import json
import shlex
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from sqlalchemy import and_, or_, text

from memory_palace.models_v3 import Message
from memory_palace.database import get_session, get_engine
from memory_palace.config import get_instances, is_postgres

# Valid message types
VALID_MESSAGE_TYPES = ["handoff", "status", "question", "fyi", "context", "event", "message"]

# In-memory subscription registry (session-scoped, not persisted)
# Maps instance_id -> set of subscribed channels
_subscriptions: Dict[str, Set[str]] = {}

# Keep track of raw connections for LISTEN (Postgres only)
_listen_connections: Dict[str, Any] = {}  # instance_id -> raw psycopg2 connection


def _get_valid_instances() -> List[str]:
    """
    Get valid instance IDs from config.

    Returns the list configured in ~/.memory-palace/config.json.
    The "all" broadcast target is handled separately in send/get functions.
    """
    return get_instances()


def _validate_priority(priority: int) -> bool:
    """Check if priority is in valid range (0-10)."""
    return 0 <= priority <= 10


def _get_channel_name(channel: Optional[str], to_instance: Optional[str]) -> str:
    """
    Generate the notification channel name for Postgres.

    For channels: memory_palace_msg_{channel}
    For direct messages: memory_palace_msg_{to_instance}
    """
    if channel:
        return f"memory_palace_msg_{channel}"
    elif to_instance:
        return f"memory_palace_msg_{to_instance}"
    return "memory_palace_msg_all"


def _pg_notify(channel_name: str, payload: Dict[str, Any]) -> None:
    """
    Send a Postgres NOTIFY on the given channel.

    Only called when is_postgres() is True.

    Args:
        channel_name: The notification channel (e.g., "memory_palace_msg_channel1")
        payload: Dict to serialize as JSON and send as payload
    """
    try:
        engine = get_engine()
        with engine.connect() as conn:
            # Use raw SQL for NOTIFY
            payload_json = json.dumps(payload)
            # Escape single quotes in payload
            payload_json = payload_json.replace("'", "''")
            conn.execute(text(f"NOTIFY {channel_name}, '{payload_json}'"))
            conn.commit()
    except Exception as e:
        # Graceful fallback: if NOTIFY fails, messages are still in DB
        # Subscribers will get them via polling
        print(f"Warning: Postgres NOTIFY failed (graceful fallback to polling): {e}")


def send_message(
    from_instance: str,
    to_instance: str,
    content: str,
    message_type: str = "message",
    subject: Optional[str] = None,
    channel: Optional[str] = None,
    priority: int = 0,
    expires_at: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Send a message to an instance or channel.

    Enables distributed consciousness - Desktop instance can leave
    notes for Code instance, or broadcast to all instances on a channel.

    Args:
        from_instance: Which instance is sending (must be a configured instance)
        to_instance: Who should receive it (a configured instance or "all" for broadcast)
        content: The actual message content
        message_type: Type of message:
            - "handoff": Task handoff between instances
            - "status": Status update
            - "question": Needs input from other instance
            - "fyi": Informational, no action needed
            - "context": Sharing context from a conversation
            - "event": Event notification
            - "message": General message (default)
        subject: Optional short summary
        channel: Optional channel name for pubsub
        priority: Message priority (0-10, default 0). Higher = more urgent
        expires_at: Optional expiration timestamp (message will be filtered out after this)

    Returns:
        {"success": True, "id": X} on success
        {"error": "..."} on failure
    """
    session = get_session()
    try:
        valid_instances = _get_valid_instances()
        valid_to_instances = valid_instances + ["all"]

        # Validate from_instance - can't send FROM "all"
        if from_instance not in valid_instances:
            return {
                "error": f"Invalid from_instance '{from_instance}'. Must be one of the configured instances: {valid_instances}. Configure in ~/.memory-palace/config.json"
            }

        # Validate to_instance - can send TO "all"
        if to_instance not in valid_to_instances:
            return {
                "error": f"Invalid to_instance '{to_instance}'. Must be one of the configured instances: {valid_to_instances}. Configure in ~/.memory-palace/config.json"
            }

        # Validate message type
        if message_type not in VALID_MESSAGE_TYPES:
            return {
                "error": f"Invalid message_type '{message_type}'. Must be one of: {VALID_MESSAGE_TYPES}"
            }

        # Validate priority
        if not _validate_priority(priority):
            return {
                "error": f"Invalid priority {priority}. Must be between 0 and 10."
            }

        # Create message
        message = Message(
            from_instance=from_instance,
            to_instance=to_instance,
            message_type=message_type,
            subject=subject,
            content=content,
            channel=channel,
            priority=priority,
            expires_at=expires_at,
            delivery_status="pending"
        )
        session.add(message)
        session.commit()
        session.refresh(message)

        # If Postgres, send NOTIFY
        if is_postgres():
            try:
                # Determine notification channel
                notify_channel = _get_channel_name(channel, to_instance)

                # Prepare payload
                payload = {
                    "message_id": message.id,
                    "from_instance": from_instance,
                    "to_instance": to_instance,
                    "message_type": message_type,
                    "subject": subject,
                    "channel": channel,
                    "priority": priority,
                }

                _pg_notify(notify_channel, payload)

                # Mark as delivered
                message.delivery_status = "delivered"
                message.delivered_at = datetime.now(timezone.utc).replace(tzinfo=None)
                session.commit()
            except Exception as e:
                # Graceful fallback: message is in DB, will be polled
                print(f"Warning: NOTIFY failed, message will be polled: {e}")

        return {"success": True, "id": message.id}
    finally:
        session.close()


def get_messages(
    instance_id: str,
    unread_only: bool = True,
    channel: Optional[str] = None,
    message_type: Optional[str] = None,
    limit: int = 50,
    include_expired: bool = False,
) -> Dict[str, Any]:
    """
    Get messages for an instance.

    Filters by channel and/or message_type if specified.
    Excludes expired messages by default.
    Returns messages sorted by priority DESC, created_at DESC.

    Args:
        instance_id: Which instance is checking (must be a configured instance)
        unread_only: Only return unread messages (default True)
        channel: Optional channel filter
        message_type: Optional message type filter
        limit: Maximum messages to return (default 50)
        include_expired: Include expired messages (default False)

    Returns:
        {"count": N, "messages": [...]} on success
        {"error": "..."} on failure
    """
    session = get_session()
    try:
        valid_instances = _get_valid_instances()

        if instance_id not in valid_instances:
            return {
                "error": f"Invalid instance_id '{instance_id}'. Must be one of the configured instances: {valid_instances}. Configure in ~/.memory-palace/config.json"
            }

        # Build query: messages addressed to this instance OR to "all"
        query = session.query(Message).filter(
            or_(
                Message.to_instance == instance_id,
                Message.to_instance == "all"
            )
        )

        # Filter by unread
        if unread_only:
            query = query.filter(Message.read_at.is_(None))

        # Filter by channel
        if channel:
            query = query.filter(Message.channel == channel)

        # Filter by message type
        if message_type:
            if message_type not in VALID_MESSAGE_TYPES:
                return {
                    "error": f"Invalid message_type '{message_type}'. Must be one of: {VALID_MESSAGE_TYPES}"
                }
            query = query.filter(Message.message_type == message_type)

        # Filter expired messages
        if not include_expired:
            query = query.filter(
                or_(
                    Message.expires_at.is_(None),
                    Message.expires_at > datetime.now(timezone.utc).replace(tzinfo=None)
                )
            )

        # Sort by priority DESC, then created_at DESC
        query = query.order_by(Message.priority.desc(), Message.created_at.desc())

        # Limit
        query = query.limit(limit)

        messages = query.all()

        return {
            "count": len(messages),
            "messages": [m.to_dict() for m in messages]
        }
    finally:
        session.close()


def mark_message_read(message_id: int, instance_id: str) -> Dict[str, Any]:
    """
    Mark a message as read by the given instance.

    Args:
        message_id: ID of the message to mark
        instance_id: Which instance read it (must be a configured instance)

    Returns:
        {"message": "Marked read"} on success
        {"error": "..."} on failure
    """
    session = get_session()
    try:
        valid_instances = _get_valid_instances()

        if instance_id not in valid_instances:
            return {
                "error": f"Invalid instance_id '{instance_id}'. Must be one of the configured instances: {valid_instances}. Configure in ~/.memory-palace/config.json"
            }

        message = session.query(Message).filter(Message.id == message_id).first()

        if not message:
            return {"error": f"Message {message_id} not found"}

        message.read_at = datetime.now(timezone.utc).replace(tzinfo=None)
        message.read_by = instance_id
        session.commit()

        return {"message": "Marked read"}
    finally:
        session.close()


def mark_message_unread(message_id: int) -> Dict[str, Any]:
    """
    Mark a message as unread (clear read_at and read_by).

    Args:
        message_id: ID of the message to mark unread

    Returns:
        {"message": "Marked unread"} on success
        {"error": "..."} on failure
    """
    session = get_session()
    try:
        message = session.query(Message).filter(Message.id == message_id).first()

        if not message:
            return {"error": f"Message {message_id} not found"}

        message.read_at = None
        message.read_by = None
        session.commit()

        return {"message": "Marked unread"}
    finally:
        session.close()


def subscribe(instance_id: str, channel: str) -> Dict[str, Any]:
    """
    Subscribe an instance to a channel.

    For Postgres: Sets up LISTEN on the channel.
    For SQLite: Records subscription for polling.

    Subscriptions are stored in-memory (dict/set) since they're session-scoped.

    Args:
        instance_id: Which instance is subscribing (must be a configured instance)
        channel: Channel name to subscribe to

    Returns:
        {"message": "Subscribed to {channel}"} on success
        {"error": "..."} on failure
    """
    valid_instances = _get_valid_instances()

    if instance_id not in valid_instances:
        return {
            "error": f"Invalid instance_id '{instance_id}'. Must be one of the configured instances: {valid_instances}. Configure in ~/.memory-palace/config.json"
        }

    # Add to in-memory registry
    if instance_id not in _subscriptions:
        _subscriptions[instance_id] = set()
    _subscriptions[instance_id].add(channel)

    # If Postgres, set up LISTEN
    if is_postgres():
        try:
            # Get or create a persistent connection for this instance
            if instance_id not in _listen_connections:
                engine = get_engine()
                # Get a raw connection from the pool
                # Note: This connection should be kept alive for the duration of the subscription
                _listen_connections[instance_id] = engine.raw_connection()

            conn = _listen_connections[instance_id]
            cursor = conn.cursor()

            # Set up LISTEN
            channel_name = _get_channel_name(channel, None)
            cursor.execute(f"LISTEN {channel_name}")
            conn.commit()
            cursor.close()

            return {"message": f"Subscribed to {channel} (Postgres LISTEN active)"}
        except Exception as e:
            # Graceful fallback to polling
            print(f"Warning: Postgres LISTEN failed, falling back to polling: {e}")
            return {"message": f"Subscribed to {channel} (polling mode)"}

    return {"message": f"Subscribed to {channel} (polling mode)"}


def unsubscribe(instance_id: str, channel: str) -> Dict[str, Any]:
    """
    Unsubscribe an instance from a channel.

    Args:
        instance_id: Which instance is unsubscribing
        channel: Channel name to unsubscribe from

    Returns:
        {"message": "Unsubscribed from {channel}"} on success
        {"error": "..."} on failure
    """
    valid_instances = _get_valid_instances()

    if instance_id not in valid_instances:
        return {
            "error": f"Invalid instance_id '{instance_id}'. Must be one of the configured instances: {valid_instances}. Configure in ~/.memory-palace/config.json"
        }

    # Remove from in-memory registry
    if instance_id in _subscriptions:
        _subscriptions[instance_id].discard(channel)

    # If Postgres, remove LISTEN
    if is_postgres() and instance_id in _listen_connections:
        try:
            conn = _listen_connections[instance_id]
            cursor = conn.cursor()

            # Remove LISTEN
            channel_name = _get_channel_name(channel, None)
            cursor.execute(f"UNLISTEN {channel_name}")
            conn.commit()
            cursor.close()
        except Exception as e:
            print(f"Warning: Postgres UNLISTEN failed: {e}")

    return {"message": f"Unsubscribed from {channel}"}


def get_subscriptions(instance_id: str) -> Dict[str, Any]:
    """
    Get all channel subscriptions for an instance.

    Args:
        instance_id: Which instance to check subscriptions for

    Returns:
        {"subscriptions": [...]} on success
        {"error": "..."} on failure
    """
    valid_instances = _get_valid_instances()

    if instance_id not in valid_instances:
        return {
            "error": f"Invalid instance_id '{instance_id}'. Must be one of the configured instances: {valid_instances}. Configure in ~/.memory-palace/config.json"
        }

    subscriptions = list(_subscriptions.get(instance_id, set()))

    return {"subscriptions": subscriptions}


def poll_messages(
    instance_id: str,
    since: Optional[datetime] = None,
    channel: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Poll for new messages created after a given timestamp.

    This is the SQLite polling path, but also works on Postgres as a fallback.
    Caller is responsible for the polling loop.

    Args:
        instance_id: Which instance is polling
        since: Only return messages created after this timestamp (default: all unread)
        channel: Optional channel filter

    Returns:
        {"count": N, "messages": [...]} on success
        {"error": "..."} on failure
    """
    session = get_session()
    try:
        valid_instances = _get_valid_instances()

        if instance_id not in valid_instances:
            return {
                "error": f"Invalid instance_id '{instance_id}'. Must be one of the configured instances: {valid_instances}. Configure in ~/.memory-palace/config.json"
            }

        # Build query: messages addressed to this instance OR to "all"
        query = session.query(Message).filter(
            or_(
                Message.to_instance == instance_id,
                Message.to_instance == "all"
            )
        )

        # Filter by timestamp
        if since:
            query = query.filter(Message.created_at > since)
        else:
            # Default to unread only
            query = query.filter(Message.read_at.is_(None))

        # Filter by channel if specified
        if channel:
            query = query.filter(Message.channel == channel)

        # Exclude expired messages
        query = query.filter(
            or_(
                Message.expires_at.is_(None),
                Message.expires_at > datetime.now(timezone.utc).replace(tzinfo=None)
            )
        )

        # Sort by priority DESC, then created_at DESC
        query = query.order_by(Message.priority.desc(), Message.created_at.desc())

        messages = query.all()

        return {
            "count": len(messages),
            "messages": [m.to_dict() for m in messages]
        }
    finally:
        session.close()


def execute_openclaw_wake(
    route: Dict[str, str],
    from_instance: str,
    to_instance: str,
    message_type: str,
    subject: Optional[str],
    message_id: Any,
    priority: int,
) -> None:
    """
    Fire-and-forget HTTP notification to an OpenClaw gateway. Never raises.

    POSTs to /hooks/palace — a custom webhook endpoint with a JS
    transform (palace.js) that routes notifications to the correct
    Discord DM channel based on the target instance.

    The transform maps to_instance → Discord channel ID and returns
    an agent hook action with deliver=true targeting the right DM.

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

        # POST to /hooks/palace — custom webhook with JS transform
        # that routes to the correct Discord DM based on to_instance
        payload_dict = {
            "from_instance": from_instance,
            "to_instance": to_instance,
            "message_type": message_type,
            "subject": subject,
            "message_id": message_id,
            "priority": priority,
        }

        payload = json.dumps(payload_dict).encode("utf-8")

        req = urllib.request.Request(
            f"{gateway_url}/hooks/palace",
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


def execute_notify_hook(
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
