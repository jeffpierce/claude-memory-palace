"""
DEPRECATED: Use message_service.py instead.

This module is kept for backward compatibility during migration.
All functions delegate to message_service equivalents.

Handoff service for inter-instance communication.

Provides note-passing between AI instances. Valid instances are configured
in ~/.memory-palace/config.json under the "instances" key.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

# Valid message types for handoffs
VALID_MESSAGE_TYPES = ["handoff", "status", "question", "fyi", "context"]


def _get_message_service():
    """Lazy import to avoid circular dependency."""
    from memory_palace.services import message_service
    return message_service


def _get_valid_instances() -> List[str]:
    """
    Get valid instance IDs from config.

    Returns the list configured in ~/.memory-palace/config.json.
    The "all" broadcast target is handled separately in send/get functions.

    DEPRECATED: This is now handled by message_service.
    """
    message_service = _get_message_service()
    return message_service._get_valid_instances()


def send_handoff(
    from_instance: str,
    to_instance: str,
    message_type: str,
    content: str,
    subject: Optional[str] = None
) -> Dict[str, Any]:
    """
    Send a message from one instance to another.

    DEPRECATED: Use message_service.send_message() instead.
    This function delegates to the new message service.

    Enables distributed consciousness - Desktop instance can leave
    notes for Code instance, etc. Note-passing for distributed minds.

    Args:
        from_instance: Which instance is sending (must be a configured instance)
        to_instance: Who should receive it (a configured instance or "all" for broadcast)
        message_type: Type of message:
            - "handoff": Task handoff between instances
            - "status": Status update
            - "question": Needs input from other instance
            - "fyi": Informational, no action needed
            - "context": Sharing context from a conversation
        content: The actual message content
        subject: Optional short summary

    Returns:
        {"success": True, "id": X} on success
        {"error": "..."} on failure
    """
    # Delegate to the new message service
    message_service = _get_message_service()
    return message_service.send_message(
        from_instance=from_instance,
        to_instance=to_instance,
        content=content,
        message_type=message_type,
        subject=subject,
        channel=None,
        priority=0,
        expires_at=None,
    )


def get_handoffs(
    for_instance: str,
    unread_only: bool = True,
    message_type: Optional[str] = None,
    limit: int = 50
) -> Dict[str, Any]:
    """
    Get messages for an instance.

    DEPRECATED: Use message_service.get_messages() instead.
    This function delegates to the new message service.

    Args:
        for_instance: Which instance is checking (must be a configured instance)
        unread_only: Only return unread messages (default True)
        message_type: Filter by type (optional)
        limit: Maximum messages to return

    Returns:
        {"count": N, "messages": [...]} on success
        {"error": "..."} on failure
    """
    # Delegate to the new message service
    message_service = _get_message_service()
    return message_service.get_messages(
        instance_id=for_instance,
        unread_only=unread_only,
        channel=None,
        message_type=message_type,
        limit=limit,
        include_expired=False,
    )


def mark_handoff_read(
    message_id: int,
    read_by: str
) -> Dict[str, Any]:
    """
    Mark a message as read.

    DEPRECATED: Use message_service.mark_message_read() instead.
    This function delegates to the new message service.

    Args:
        message_id: ID of the message to mark
        read_by: Which instance read it (must be a configured instance)

    Returns:
        Compact confirmation string
    """
    # Delegate to the new message service
    message_service = _get_message_service()
    return message_service.mark_message_read(
        message_id=message_id,
        instance_id=read_by,
    )
