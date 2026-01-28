"""
Handoff service for inter-instance communication.

Provides note-passing between Claude instances (desktop, code, web, etc.).
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import or_

from memory_palace.models import HandoffMessage
from memory_palace.database import get_session
from memory_palace.config import get_instances


# Valid message types for handoffs
VALID_MESSAGE_TYPES = ["handoff", "status", "question", "fyi", "context"]


def _get_valid_instances() -> List[str]:
    """
    Get valid instance IDs from config.

    Returns configured instances plus "all" for broadcasts.
    """
    return get_instances()


def send_handoff(
    from_instance: str,
    to_instance: str,
    message_type: str,
    content: str,
    subject: Optional[str] = None
) -> Dict[str, Any]:
    """
    Send a message from one instance to another.

    Enables distributed Claude consciousness - Desktop Claude can leave
    notes for Code Claude, etc. Note-passing for distributed minds.

    Args:
        from_instance: Which instance is sending (must be in configured instances)
        to_instance: Who should receive it (configured instance or "all" for broadcast)
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
    session = get_session()
    try:
        valid_instances = _get_valid_instances()
        valid_to_instances = valid_instances + ["all"]

        # Validate from_instance - can't send FROM "all"
        if from_instance not in valid_instances:
            return {"error": f"Invalid from_instance '{from_instance}'. Must be one of: {valid_instances}"}

        # Validate to_instance - can send TO "all"
        if to_instance not in valid_to_instances:
            return {"error": f"Invalid to_instance '{to_instance}'. Must be one of: {valid_to_instances}"}

        # Validate message type
        if message_type not in VALID_MESSAGE_TYPES:
            return {"error": f"Invalid message_type '{message_type}'. Must be one of: {VALID_MESSAGE_TYPES}"}

        message = HandoffMessage(
            from_instance=from_instance,
            to_instance=to_instance,
            message_type=message_type,
            subject=subject,
            content=content
        )
        session.add(message)
        session.commit()
        session.refresh(message)

        return {"success": True, "id": message.id}
    finally:
        session.close()


def get_handoffs(
    for_instance: str,
    unread_only: bool = True,
    message_type: Optional[str] = None,
    limit: int = 50
) -> Dict[str, Any]:
    """
    Get messages for an instance.

    Args:
        for_instance: Which instance is checking (must be in configured instances)
        unread_only: Only return unread messages (default True)
        message_type: Filter by type (optional)
        limit: Maximum messages to return

    Returns:
        {"count": N, "messages": [...]} on success
        {"error": "..."} on failure
    """
    session = get_session()
    try:
        valid_instances = _get_valid_instances()

        if for_instance not in valid_instances:
            return {"error": f"Invalid for_instance '{for_instance}'. Must be one of: {valid_instances}"}

        # Get messages addressed to this instance OR to "all"
        query = session.query(HandoffMessage).filter(
            or_(
                HandoffMessage.to_instance == for_instance,
                HandoffMessage.to_instance == "all"
            )
        )

        if unread_only:
            query = query.filter(HandoffMessage.read_at.is_(None))

        if message_type:
            if message_type not in VALID_MESSAGE_TYPES:
                return {"error": f"Invalid message_type '{message_type}'. Must be one of: {VALID_MESSAGE_TYPES}"}
            query = query.filter(HandoffMessage.message_type == message_type)

        query = query.order_by(HandoffMessage.created_at.desc()).limit(limit)
        messages = query.all()

        return {
            "count": len(messages),
            "messages": [m.to_dict() for m in messages]
        }
    finally:
        session.close()


def mark_handoff_read(
    message_id: int,
    read_by: str
) -> Dict[str, Any]:
    """
    Mark a message as read.

    Args:
        message_id: ID of the message to mark
        read_by: Which instance read it (must be in configured instances)

    Returns:
        Compact confirmation string
    """
    session = get_session()
    try:
        valid_instances = _get_valid_instances()

        if read_by not in valid_instances:
            return {"error": f"Invalid read_by '{read_by}'. Must be one of: {valid_instances}"}

        message = session.query(HandoffMessage).filter(
            HandoffMessage.id == message_id
        ).first()

        if not message:
            return {"error": f"Message {message_id} not found"}

        message.read_at = datetime.utcnow()
        message.read_by = read_by
        session.commit()

        # Compact response
        return {"message": "Marked read"}
    finally:
        session.close()
