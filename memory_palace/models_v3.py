"""
SQLAlchemy models for Memory Palace v3.

Key changes from v2:
- memories: Remove importance column, add foundational column
- handoff_messages → messages: Rename table and add pubsub support columns
- memory_edges: No changes

Portable: works on SQLite (default) and PostgreSQL + pgvector (upgrade path)
  - SQLite: JSON text for arrays, Text for embeddings, standard JSON for metadata
  - PostgreSQL: native ARRAY, JSONB, pgvector Vector types
"""

from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Boolean, Float, JSON,
    ForeignKey, CheckConstraint, UniqueConstraint, Index,
    event, text
)
from sqlalchemy.orm import relationship, declarative_base

from memory_palace.config_v2 import get_embedding_dimension, is_postgres

# Conditional PostgreSQL-specific imports
_USE_PG_TYPES = is_postgres()

if _USE_PG_TYPES:
    try:
        from sqlalchemy.dialects.postgresql import ARRAY as PG_ARRAY, JSONB
    except ImportError:
        _USE_PG_TYPES = False

# Conditional pgvector import
try:
    from pgvector.sqlalchemy import Vector
    HAS_PGVECTOR = True
except ImportError:
    HAS_PGVECTOR = False
    Vector = None

Base = declarative_base()

# --- Portable column type helpers ---

def _array_column(nullable=True):
    """ARRAY(Text) on PostgreSQL, JSON on SQLite."""
    if _USE_PG_TYPES:
        return Column(PG_ARRAY(Text), nullable=nullable)
    return Column(JSON, nullable=nullable)


def _jsonb_column(default=dict):
    """JSONB on PostgreSQL, JSON on SQLite."""
    if _USE_PG_TYPES:
        return Column(JSONB, default=default)
    return Column(JSON, default=default)


def _embedding_column():
    """Vector(dim) on PostgreSQL + pgvector, Text on SQLite."""
    if _USE_PG_TYPES and HAS_PGVECTOR:
        dim = get_embedding_dimension()
        return Column(Vector(dim), nullable=True)
    return Column(Text, nullable=True)


def _normalize_projects(project) -> List[str]:
    """Normalize project param to list. Accepts str or List[str]."""
    if project is None:
        return ["life"]
    if isinstance(project, list):
        return project
    return [project]


class Memory(Base):
    """
    Persistent memory system for AI instances.

    v3 changes from v2:
    - Removed: importance (Integer 1-10)
    - Added: foundational (Boolean, default=False) - memories immune to pruning
    """
    __tablename__ = "memories"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None), nullable=False)
    updated_at = Column(DateTime, onupdate=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

    # Instance and project scoping
    instance_id = Column(String(50), nullable=False, index=True)
    projects = _array_column(nullable=False)  # ARRAY(Text) on PG, JSON on SQLite

    # Content
    memory_type = Column(String(50), nullable=False, index=True)
    subject = Column(String(255), nullable=True, index=True)
    content = Column(Text, nullable=False)

    # Searchability — ARRAY(Text) on PostgreSQL, JSON on SQLite
    keywords = _array_column()  # For semantic search
    tags = _array_column()  # For organization
    foundational = Column(Boolean, default=False, index=True)  # Immune to pruning/archival

    # Source tracking
    source_type = Column(String(50), nullable=True)
    source_context = Column(Text, nullable=True)
    source_session_id = Column(String(100), nullable=True)

    # Embedding — Vector(dim) on PostgreSQL + pgvector, Text (JSON) on SQLite
    # nomic-embed-text (768d) is preferred: fits pgvector HNSW limits, runs on CPU
    embedding = _embedding_column()

    # Lifecycle
    last_accessed_at = Column(DateTime, nullable=True)
    access_count = Column(Integer, default=0)
    expires_at = Column(DateTime, nullable=True)
    is_archived = Column(Boolean, default=False)

    # Relationships
    outgoing_edges = relationship(
        "MemoryEdge",
        foreign_keys="MemoryEdge.source_id",
        back_populates="source",
        cascade="all, delete-orphan"
    )
    incoming_edges = relationship(
        "MemoryEdge",
        foreign_keys="MemoryEdge.target_id",
        back_populates="target",
        cascade="all, delete-orphan"
    )

    __table_args__ = (
        # Note: composite index on (instance_id, projects) only works when
        # projects is ARRAY(Text) on PostgreSQL. When projects is JSON
        # (SQLite or PG without _USE_PG_TYPES), btree indexing fails.
        # instance_id is already individually indexed (index=True on column).
        *(
            (Index("idx_memories_instance_projects", "instance_id", "projects"),)
            if _USE_PG_TYPES else ()
        ),
        Index("idx_memories_foundational", "foundational"),
    )

    def __repr__(self):
        subject_str = f", subject='{self.subject}'" if self.subject else ""
        foundational_str = " [foundational]" if self.foundational else ""
        return f"<Memory(id={self.id}, type='{self.memory_type}', projects={self.projects}{subject_str}{foundational_str})>"

    def to_dict(self, detail_level: str = "verbose", include_edges: bool = False):
        """
        Serialize to dictionary.

        Args:
            detail_level: 'summary' for compact output, 'verbose' for full details
            include_edges: Include relationship edges in output

        Returns:
            Dictionary representation of the memory
        """
        base = {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "instance_id": self.instance_id,
            "project": self.projects,
            "memory_type": self.memory_type,
            "subject": self.subject,
            "keywords": self.keywords,
            "tags": self.tags,
            "foundational": self.foundational,
            "access_count": self.access_count,
            "is_archived": self.is_archived
        }

        if detail_level == "summary":
            base["content_preview"] = (
                self.content[:200] + "..."
                if len(self.content) > 200
                else self.content
            )
        else:
            base["content"] = self.content
            base["source_type"] = self.source_type
            base["source_context"] = self.source_context
            base["source_session_id"] = self.source_session_id
            base["updated_at"] = self.updated_at.isoformat() if self.updated_at else None
            base["last_accessed_at"] = self.last_accessed_at.isoformat() if self.last_accessed_at else None
            base["expires_at"] = self.expires_at.isoformat() if self.expires_at else None

        if include_edges:
            base["outgoing_edges"] = [e.to_dict() for e in self.outgoing_edges]
            base["incoming_edges"] = [e.to_dict() for e in self.incoming_edges]

        return base

    def embedding_text(self) -> str:
        """
        Generate the text used for embedding generation.

        Includes memory_type and project as prefix to influence semantic matching.
        """
        parts = [f"[{self.memory_type}]"]
        if self.projects:
            non_life = [p for p in self.projects if p != "life"]
            if non_life:
                parts.append(f"[project:{','.join(non_life)}]")
        if self.subject:
            parts.append(self.subject)
        parts.append(self.content)
        return " ".join(parts)


def _project_contains(value: str):
    """Filter: memory's projects array contains this value."""
    if _USE_PG_TYPES:
        return Memory.projects.contains([value])  # PG: @> ARRAY['value']
    else:
        from sqlalchemy import String as SAString
        return Memory.projects.cast(SAString).like(f'%"{value}"%')  # SQLite JSON


def _projects_overlap(values: List[str]):
    """Filter: memory's projects array overlaps with any of these values."""
    from sqlalchemy import or_ as sa_or, String as SAString
    if _USE_PG_TYPES:
        return Memory.projects.overlap(values)  # PG: && operator
    else:
        return sa_or(*[Memory.projects.cast(SAString).like(f'%"{v}"%') for v in values])


class MemoryEdge(Base):
    """
    Knowledge graph edges connecting memories.

    Supports various relationship types for building a semantic network
    of memories that can be traversed.

    Relationship types:
    - supersedes: Newer memory replaces older (directional)
    - relates_to: General association (often bidirectional)
    - derived_from: This memory came from processing that one (directional)
    - contradicts: Memories are in tension (bidirectional)
    - exemplifies: This is an example of that concept (directional)
    - refines: Adds detail/nuance to another memory (directional)
    """
    __tablename__ = "memory_edges"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None), nullable=False)

    # Edge endpoints
    source_id = Column(
        Integer,
        ForeignKey("memories.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    target_id = Column(
        Integer,
        ForeignKey("memories.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Edge properties
    relation_type = Column(String(50), nullable=False, index=True)
    strength = Column(Float, default=1.0)  # 0-1, for weighted traversal
    bidirectional = Column(Boolean, default=False)  # If true, edge works both ways

    # Extra data — JSONB on PostgreSQL, JSON on SQLite
    edge_metadata = _jsonb_column(default=dict)
    created_by = Column(String(50), nullable=True)  # Which instance created this

    # Relationships
    source = relationship("Memory", foreign_keys=[source_id], back_populates="outgoing_edges")
    target = relationship("Memory", foreign_keys=[target_id], back_populates="incoming_edges")

    __table_args__ = (
        UniqueConstraint("source_id", "target_id", "relation_type", name="uq_edge_triple"),
        CheckConstraint("source_id != target_id", name="check_no_self_loops"),
        CheckConstraint("strength >= 0 AND strength <= 1", name="check_strength_range"),
        Index("idx_edges_source_rel", "source_id", "relation_type"),
    )

    def __repr__(self):
        direction = "<->" if self.bidirectional else "->"
        return f"<MemoryEdge({self.source_id} {direction}[{self.relation_type}]{direction} {self.target_id})>"

    def to_dict(self):
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "strength": self.strength,
            "bidirectional": self.bidirectional,
            "metadata": self.edge_metadata,  # Expose as 'metadata' in dict for API compatibility
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "created_by": self.created_by,
        }


class Message(Base):
    """
    Inter-instance communication for AI instances.

    v3 changes from v2 (HandoffMessage):
    - Table renamed from handoff_messages to messages
    - Class renamed from HandoffMessage to Message
    - Added pubsub support: channel, delivery_status, delivered_at, expires_at, priority
    """
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None), nullable=False)

    # Direct messaging (original handoff functionality)
    from_instance = Column(String(50), nullable=False, index=True)
    to_instance = Column(String(50), nullable=False, index=True)

    # Message content
    message_type = Column(String(50), nullable=False, index=True)
    subject = Column(String(255), nullable=True)
    content = Column(Text, nullable=False)

    # Read tracking
    read_at = Column(DateTime, nullable=True)
    read_by = Column(String(50), nullable=True)

    # Pubsub support (v3)
    channel = Column(String(100), nullable=True, index=True)  # Subscription channel name
    delivery_status = Column(String(20), default="pending", index=True)  # "pending", "delivered", "failed"
    delivered_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)  # Optional message TTL
    priority = Column(Integer, default=0, index=True)  # Higher = more urgent

    __table_args__ = (
        # Partial index on PostgreSQL for fast unread lookups
        # On SQLite this creates a normal index (still useful, just not filtered)
        Index("idx_messages_unread", "to_instance",
              **({"postgresql_where": text("read_at IS NULL")} if _USE_PG_TYPES else {})),
        # Index for pubsub channel queries
        Index("idx_messages_channel_status", "channel", "delivery_status"),
        Index("idx_messages_priority_desc", "priority", postgresql_ops={"priority": "DESC"} if _USE_PG_TYPES else None),
    )

    def __repr__(self):
        status = "read" if self.read_at else "unread"
        channel_str = f", channel='{self.channel}'" if self.channel else ""
        return f"<Message(id={self.id}, {self.from_instance}->{self.to_instance}, {status}{channel_str})>"

    def to_dict(self):
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "from_instance": self.from_instance,
            "to_instance": self.to_instance,
            "message_type": self.message_type,
            "subject": self.subject,
            "content": self.content,
            "read_at": self.read_at.isoformat() if self.read_at else None,
            "read_by": self.read_by,
            "channel": self.channel,
            "delivery_status": self.delivery_status,
            "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "priority": self.priority,
        }

    def is_for_instance(self, instance_id: str) -> bool:
        """Check if this message is intended for the given instance."""
        return self.to_instance == instance_id or self.to_instance == "all"


# Relationship type constants for validation
RELATIONSHIP_TYPES = {
    "supersedes",    # Newer memory replaces older
    "relates_to",    # General association
    "derived_from",  # This memory came from that one
    "contradicts",   # Memories are in tension
    "exemplifies",   # This is an example of that concept
    "refines",       # Adds detail/nuance
}


def validate_relation_type(relation_type: str) -> bool:
    """
    Check if a relation type is valid.

    Note: Custom types are allowed, these are just the standard ones.
    """
    return relation_type in RELATIONSHIP_TYPES


# Legacy alias
validate_relationship_type = validate_relation_type
