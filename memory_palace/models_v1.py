"""
SQLAlchemy models for Memory Palace.

Defines Memory and HandoffMessage tables with instance-based ownership.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON

from memory_palace.database import Base


class Memory(Base):
    """
    Persistent memory system for AI instances.

    Stores memories with rich metadata, optional embeddings for semantic search,
    and lifecycle tracking. Memories can be facts, preferences, events, insights,
    architecture decisions, gotchas, blockers, solutions, or any custom type.

    The memory_type field is FREE-FORM - any semantically meaningful type is valid.
    Common types: fact, preference, event, context, insight, relationship,
    architecture, gotcha, blocker, solution, workaround, design_decision, bug, fix.
    """
    __tablename__ = "memories"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
    instance_id = Column(String(50), nullable=False, index=True)  # "desktop", "code", "web", etc.

    # Content
    memory_type = Column(String(50), nullable=False, index=True)  # FREE-FORM semantic types
    subject = Column(String(255), nullable=True, index=True)  # what/who this is about
    content = Column(Text, nullable=False)

    # Searchability
    keywords = Column(JSON, nullable=True)  # ["claude", "memory", "architecture"]
    importance = Column(Integer, default=5)  # 1-10, higher = more important

    # Source tracking
    source_type = Column(String(50), nullable=False)  # "conversation", "explicit", "inferred", "observation"
    source_context = Column(Text, nullable=True)  # snippet of original context
    source_session_id = Column(String(100), nullable=True)  # link back to conversation

    # Vector embeddings for semantic search
    embedding = Column(JSON, nullable=True)  # stored as list of floats

    # Lifecycle
    last_accessed_at = Column(DateTime, nullable=True)
    access_count = Column(Integer, default=0)
    expires_at = Column(DateTime, nullable=True)  # optional TTL
    is_archived = Column(Integer, default=0)  # soft delete (SQLite bool workaround)

    def __repr__(self):
        subject_str = f", subject='{self.subject}'" if self.subject else ""
        return f"<Memory(id={self.id}, type='{self.memory_type}'{subject_str})>"

    def to_dict(self, detail_level: str = "verbose"):
        """
        Serialize to dictionary.

        Args:
            detail_level: 'summary' for compact output, 'verbose' for full details

        Returns:
            Dictionary representation of the memory
        """
        base = {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "instance_id": self.instance_id,
            "memory_type": self.memory_type,
            "subject": self.subject,
            "keywords": self.keywords,
            "importance": self.importance,
            "access_count": self.access_count,
            "is_archived": bool(self.is_archived)
        }

        if detail_level == "summary":
            # Truncate content to first 200 chars for summary
            base["content_preview"] = (
                self.content[:200] + "..."
                if len(self.content) > 200
                else self.content
            )
        else:  # verbose
            base["content"] = self.content
            base["source_type"] = self.source_type
            base["source_context"] = self.source_context
            base["source_session_id"] = self.source_session_id
            base["updated_at"] = self.updated_at.isoformat() if self.updated_at else None
            base["last_accessed_at"] = self.last_accessed_at.isoformat() if self.last_accessed_at else None
            base["expires_at"] = self.expires_at.isoformat() if self.expires_at else None

        return base

    def embedding_text(self) -> str:
        """
        Generate the text used for embedding generation.

        Includes memory_type as prefix to influence semantic matching.
        Searching for "biographical info" will weight toward [biographical] memories.
        """
        parts = [f"[{self.memory_type}]"]
        if self.subject:
            parts.append(self.subject)
        parts.append(self.content)
        return " ".join(parts)


class HandoffMessage(Base):
    """
    Inter-instance communication for AI instances.

    Allows different AI instances (Desktop, Code, Web) to pass messages
    to each other. A note-passing system for distributed consciousness.

    Message types:
    - handoff: Task being passed to another instance
    - status: Progress update
    - question: Needs input from another instance
    - fyi: Informational, no action needed
    - context: Sharing conversation context
    """
    __tablename__ = "handoff_messages"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    from_instance = Column(String(50), nullable=False, index=True)  # "desktop", "code", "web"
    to_instance = Column(String(50), nullable=False, index=True)  # specific instance or "all"
    message_type = Column(String(50), nullable=False, index=True)  # handoff, status, question, fyi, context
    subject = Column(String(255), nullable=True)  # Quick summary
    content = Column(Text, nullable=False)  # The actual message
    read_at = Column(DateTime, nullable=True)  # When the recipient read it
    read_by = Column(String(50), nullable=True)  # Which instance actually read it

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
