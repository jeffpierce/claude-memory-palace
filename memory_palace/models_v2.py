"""
SQLAlchemy models for Claude Memory Palace v2.

Key changes from v1:
- PostgreSQL with pgvector for native vector operations
- Knowledge graph via memory_edges table
- Project scoping for memories
- Tags separate from keywords
- Proper ARRAY types instead of JSON
"""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Boolean, Float,
    ForeignKey, CheckConstraint, UniqueConstraint, Index,
    event, text
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import relationship, declarative_base

# Conditional pgvector import
try:
    from pgvector.sqlalchemy import Vector
    HAS_PGVECTOR = True
except ImportError:
    HAS_PGVECTOR = False
    Vector = None

from memory_palace.config_v2 import get_embedding_dimension, is_postgres

Base = declarative_base()


def get_embedding_column():
    """
    Get the appropriate embedding column type based on database backend.
    
    Returns Vector(dim) for Postgres with pgvector, or Text for SQLite fallback.
    """
    if is_postgres() and HAS_PGVECTOR:
        dim = get_embedding_dimension()
        return Column(Vector(dim), nullable=True)
    else:
        # SQLite fallback - store as JSON text
        return Column(Text, nullable=True)


class Memory(Base):
    """
    Persistent memory system for Claude instances.

    v2 changes:
    - project: Organize memories by project (default: "life")
    - tags: Freeform organizational tags (separate from keywords)
    - embedding: Native pgvector type for Postgres
    - keywords/tags: Native ARRAY type for Postgres
    """
    __tablename__ = "memories"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
    
    # Instance and project scoping
    instance_id = Column(String(50), nullable=False, index=True)
    project = Column(String(100), nullable=False, default="life", index=True)
    
    # Content
    memory_type = Column(String(50), nullable=False, index=True)
    subject = Column(String(255), nullable=True, index=True)
    content = Column(Text, nullable=False)
    
    # Searchability - use ARRAY for Postgres, JSON string for SQLite
    # Note: These are set dynamically based on database type
    keywords = Column(ARRAY(Text), nullable=True)  # For semantic search
    tags = Column(ARRAY(Text), nullable=True)  # For organization
    importance = Column(Integer, default=5, index=True)
    
    # Source tracking
    source_type = Column(String(50), nullable=True)
    source_context = Column(Text, nullable=True)
    source_session_id = Column(String(100), nullable=True)
    
    # Embedding - Vector(768) for nomic-embed-text (recommended)
    # Note: Dimension must match your embedding model
    # nomic-embed-text (768d) is preferred because it fits pgvector HNSW index limits (<2000)
    # and runs efficiently on CPU for AWS deployment
    embedding = Column(Vector(768), nullable=True) if HAS_PGVECTOR else Column(Text, nullable=True)
    
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
        Index("idx_memories_instance_project", "instance_id", "project"),
        Index("idx_memories_importance_desc", importance.desc()),
        CheckConstraint("importance >= 1 AND importance <= 10", name="check_importance_range"),
    )

    def __repr__(self):
        subject_str = f", subject='{self.subject}'" if self.subject else ""
        return f"<Memory(id={self.id}, type='{self.memory_type}', project='{self.project}'{subject_str})>"

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
            "project": self.project,
            "memory_type": self.memory_type,
            "subject": self.subject,
            "keywords": self.keywords,
            "tags": self.tags,
            "importance": self.importance,
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
        if self.project and self.project != "life":
            parts.append(f"[project:{self.project}]")
        if self.subject:
            parts.append(self.subject)
        parts.append(self.content)
        return " ".join(parts)


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
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
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
    
    # Extra data
    edge_metadata = Column(JSONB, default=dict)  # Flexible extra data
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


class HandoffMessage(Base):
    """
    Inter-instance communication for Claude instances.
    
    Unchanged from v1 except for type cleanup (proper Boolean, TIMESTAMPTZ).
    """
    __tablename__ = "handoff_messages"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    from_instance = Column(String(50), nullable=False, index=True)
    to_instance = Column(String(50), nullable=False, index=True)
    message_type = Column(String(50), nullable=False, index=True)
    subject = Column(String(255), nullable=True)
    content = Column(Text, nullable=False)
    read_at = Column(DateTime, nullable=True)
    read_by = Column(String(50), nullable=True)

    __table_args__ = (
        Index("idx_handoff_unread", "to_instance", postgresql_where=text("read_at IS NULL")),
    )

    def __repr__(self):
        status = "read" if self.read_at else "unread"
        return f"<HandoffMessage(id={self.id}, {self.from_instance}->{self.to_instance}, {status})>"

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
            "read_by": self.read_by
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
