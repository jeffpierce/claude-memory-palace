"""
SQLAlchemy models for Claude Memory Palace.

v2: Re-exports from models_v2 for PostgreSQL + pgvector support.
For legacy SQLite models, see models_v1.py.
"""

# Re-export everything from v2
from memory_palace.models_v2 import (
    Base,
    Memory,
    MemoryEdge,
    HandoffMessage,
    RELATIONSHIP_TYPES,
    validate_relation_type,
    validate_relationship_type,  # Legacy alias
    HAS_PGVECTOR,
)

__all__ = [
    "Base",
    "Memory",
    "MemoryEdge",
    "HandoffMessage",
    "RELATIONSHIP_TYPES",
    "validate_relation_type",
    "validate_relationship_type",
    "HAS_PGVECTOR",
]
