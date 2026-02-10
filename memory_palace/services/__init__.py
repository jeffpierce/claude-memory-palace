"""
Services for Memory Palace.

Each service encapsulates a logical unit of functionality.
"""

from memory_palace.services.handoff_service import (
    send_handoff,
    get_handoffs,
    mark_handoff_read,
    VALID_MESSAGE_TYPES,
)
from memory_palace.services.message_service import (
    send_message,
    get_messages,
    mark_message_read,
    mark_message_unread,
    subscribe,
    unsubscribe,
    get_subscriptions,
    poll_messages,
)
from memory_palace.services.memory_service import (
    remember,
    recall,
    forget,
    archive_memory,
    get_memory_stats,
    backfill_embeddings,
    get_memory_by_id,
    get_memories_by_ids,
    get_recent_memories,
    update_memory,
    jsonl_to_toon_chunks,
    VALID_SOURCE_TYPES,
)
from memory_palace.services.graph_service import (
    link_memories,
    unlink_memories,
    get_related_memories,
    supersede_memory,
    traverse_graph,
    get_relationship_types,
)
from memory_palace.services.reflection_service import reflect
from memory_palace.services.code_service import (
    code_remember,
    code_recall,
)
from memory_palace.services.maintenance_service import (
    audit_palace,
    batch_archive_memories,
    reembed_memories,
    cleanup_cross_project_auto_links,
)

__all__ = [
    # Handoff messaging (deprecated - use message_service equivalents)
    "send_handoff",
    "get_handoffs",
    "mark_handoff_read",
    "VALID_MESSAGE_TYPES",
    # Message pubsub (new in v2.0)
    "send_message",
    "get_messages",
    "mark_message_read",
    "mark_message_unread",
    "subscribe",
    "unsubscribe",
    "get_subscriptions",
    "poll_messages",
    # Memory operations
    "remember",
    "recall",
    "forget",
    "archive_memory",
    "get_memory_stats",
    "backfill_embeddings",
    "get_memory_by_id",
    "get_memories_by_ids",
    "get_recent_memories",
    "update_memory",
    "jsonl_to_toon_chunks",
    "VALID_SOURCE_TYPES",
    # Knowledge graph
    "link_memories",
    "unlink_memories",
    "get_related_memories",
    "supersede_memory",
    "traverse_graph",
    "get_relationship_types",
    # Reflection
    "reflect",
    # Code retrieval
    "code_remember",
    "code_recall",
    # Maintenance
    "audit_palace",
    "batch_archive_memories",
    "reembed_memories",
    "cleanup_cross_project_auto_links",
]
