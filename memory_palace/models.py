"""
SQLAlchemy models for Claude Memory Palace.

v3: Re-exports from models_v3 with foundational memories and pubsub messaging.
For v2 models, see models_v2.py. For legacy SQLite models, see models_v1.py.
"""

# Re-export everything from v3
from memory_palace.models_v3 import (
    Base,
    Memory,
    MemoryEdge,
    Message,
    RELATIONSHIP_TYPES,
    validate_relation_type,
    validate_relationship_type,  # Legacy alias
    HAS_PGVECTOR,
)

# Backward compatibility alias
HandoffMessage = Message

__all__ = [
    "Base",
    "Memory",
    "MemoryEdge",
    "Message",
    "HandoffMessage",  # Legacy alias
    "RELATIONSHIP_TYPES",
    "validate_relation_type",
    "validate_relationship_type",
    "HAS_PGVECTOR",
]
