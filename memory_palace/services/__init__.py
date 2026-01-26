"""
Services for Claude Memory Palace.

Each service encapsulates a logical unit of functionality.
"""

from memory_palace.services.handoff_service import (
    send_handoff,
    get_handoffs,
    mark_handoff_read,
    VALID_MESSAGE_TYPES,
)
from memory_palace.services.memory_service import (
    remember,
    recall,
    forget,
    get_memory_stats,
    backfill_embeddings,
    get_memory_by_id,
    get_memories_by_ids,
    update_memory,
    jsonl_to_toon_chunks,
    VALID_SOURCE_TYPES,
)
from memory_palace.services.reflection_service import reflect

__all__ = [
    # Handoff messaging
    "send_handoff",
    "get_handoffs",
    "mark_handoff_read",
    "VALID_MESSAGE_TYPES",
    # Memory operations
    "remember",
    "recall",
    "forget",
    "get_memory_stats",
    "backfill_embeddings",
    "get_memory_by_id",
    "get_memories_by_ids",
    "update_memory",
    "jsonl_to_toon_chunks",
    "VALID_SOURCE_TYPES",
    # Reflection
    "reflect",
]
