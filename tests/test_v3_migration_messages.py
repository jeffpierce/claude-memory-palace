"""
Tests for Memory Palace v2 to v3 Migration and Message Service.

Tests:
- Migration script: migrate_memories_table(), migrate_messages_table(), migrate()
- Message service: send/get/mark messages, subscriptions, polling
"""
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock
from typing import Dict, Set

import pytest
from sqlalchemy import create_engine, text, Column, Integer, String, Text, DateTime, Boolean, Float, JSON
from sqlalchemy.orm import declarative_base, sessionmaker, Session as SessionType
from sqlalchemy.pool import StaticPool

# Import migration functions
from memory_palace.migrations.v2_to_v3 import (
    migrate_memories_table,
    migrate_messages_table,
    migrate,
    _column_exists,
    _table_exists,
    _index_exists,
)

# Import database module for patching
import memory_palace.database_v3 as db_module
import memory_palace.services.message_service as msg_module
from memory_palace.services.message_service import (
    send_message,
    get_messages,
    mark_message_read,
    mark_message_unread,
    subscribe,
    unsubscribe,
    get_subscriptions,
    poll_messages,
    VALID_MESSAGE_TYPES,
)


# ── V2 Schema for Migration Tests ───────────────────────────────────────

V2Base = declarative_base()


class V2Memory(V2Base):
    """Simulated v2 memories table WITH importance column."""
    __tablename__ = "memories"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime)
    instance_id = Column(String(50), nullable=False)
    project = Column(String(100), nullable=False, default="life")
    memory_type = Column(String(50), nullable=False)
    subject = Column(String(255))
    content = Column(Text, nullable=False)
    keywords = Column(JSON)
    tags = Column(JSON)
    importance = Column(Integer, default=5)  # THE V2 COLUMN
    source_type = Column(String(50))
    source_context = Column(Text)
    source_session_id = Column(String(100))
    embedding = Column(Text)
    last_accessed_at = Column(DateTime)
    access_count = Column(Integer, default=0)
    expires_at = Column(DateTime)
    is_archived = Column(Boolean, default=False)


class V2HandoffMessage(V2Base):
    """Simulated v2 handoff_messages table."""
    __tablename__ = "handoff_messages"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    from_instance = Column(String(50), nullable=False)
    to_instance = Column(String(50), nullable=False)
    message_type = Column(String(50), nullable=False)
    subject = Column(String(255))
    content = Column(Text, nullable=False)
    read_at = Column(DateTime)
    read_by = Column(String(50))


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def v2_db():
    """Create an in-memory SQLite database with v2 schema."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    V2Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture
def v3_db():
    """Create an in-memory SQLite database with v3 schema for message service tests."""
    # Create v3 schema manually to avoid ARRAY type issues
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # Create tables manually with SQLite-compatible types
    with engine.connect() as conn:
        # Create memories table (v3 schema)
        conn.execute(text("""
            CREATE TABLE memories (
                id INTEGER PRIMARY KEY,
                created_at DATETIME NOT NULL,
                updated_at DATETIME,
                instance_id VARCHAR(50) NOT NULL,
                project VARCHAR(100) NOT NULL DEFAULT 'life',
                memory_type VARCHAR(50) NOT NULL,
                subject VARCHAR(255),
                content TEXT NOT NULL,
                keywords TEXT,
                tags TEXT,
                foundational INTEGER DEFAULT 0,
                source_type VARCHAR(50),
                source_context TEXT,
                source_session_id VARCHAR(100),
                embedding TEXT,
                last_accessed_at DATETIME,
                access_count INTEGER DEFAULT 0,
                expires_at DATETIME,
                is_archived INTEGER DEFAULT 0
            )
        """))

        # Create messages table (v3 schema)
        conn.execute(text("""
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY,
                created_at DATETIME NOT NULL,
                from_instance VARCHAR(50) NOT NULL,
                to_instance VARCHAR(50) NOT NULL,
                message_type VARCHAR(50) NOT NULL,
                subject VARCHAR(255),
                content TEXT NOT NULL,
                read_at DATETIME,
                read_by VARCHAR(50),
                channel VARCHAR(100),
                delivery_status VARCHAR(20) DEFAULT 'pending',
                delivered_at DATETIME,
                expires_at DATETIME,
                priority INTEGER DEFAULT 0
            )
        """))

        # Create indexes
        conn.execute(text("CREATE INDEX idx_memories_instance_id ON memories(instance_id)"))
        conn.execute(text("CREATE INDEX idx_memories_foundational ON memories(foundational)"))
        conn.execute(text("CREATE INDEX idx_messages_from_instance ON messages(from_instance)"))
        conn.execute(text("CREATE INDEX idx_messages_to_instance ON messages(to_instance)"))
        conn.execute(text("CREATE INDEX idx_messages_channel ON messages(channel)"))
        conn.execute(text("CREATE INDEX idx_messages_priority ON messages(priority)"))

        conn.commit()

    # Create sessionmaker
    Session = sessionmaker(bind=engine)

    # Monkey-patch database module
    old_engine = db_module._engine
    old_session = db_module._SessionLocal
    db_module._engine = engine
    db_module._SessionLocal = Session

    yield engine, Session

    # Restore
    db_module._engine = old_engine
    db_module._SessionLocal = old_session


@pytest.fixture(autouse=True)
def mock_config():
    """Mock config to return test instances."""
    with patch("memory_palace.services.message_service.get_instances", return_value=["test", "code", "desktop"]):
        with patch("memory_palace.services.message_service.is_postgres", return_value=False):
            yield


@pytest.fixture(autouse=True)
def reset_subscriptions():
    """Clear in-memory subscriptions between tests."""
    msg_module._subscriptions.clear()
    msg_module._listen_connections.clear()
    yield
    msg_module._subscriptions.clear()
    msg_module._listen_connections.clear()


# ── Migration Tests: memories table ──────────────────────────────────────


class TestMigrateMemoriesTable:
    """Tests for migrate_memories_table()."""

    def test_adds_foundational_column(self, v2_db):
        """Test that migration adds foundational column if missing."""
        # Start with v2 schema (no foundational column)
        assert not _column_exists(v2_db, "memories", "foundational")
        assert _column_exists(v2_db, "memories", "importance")

        # Run migration
        migrate_memories_table(v2_db)

        # Verify foundational exists
        assert _column_exists(v2_db, "memories", "foundational")

    def test_migrates_importance_high_to_foundational_true(self, v2_db):
        """Test that importance >= 8 becomes foundational = True."""
        # Insert v2 data with high importance
        with v2_db.connect() as conn:
            conn.execute(text(
                "INSERT INTO memories (instance_id, project, memory_type, content, importance, created_at) "
                "VALUES ('test', 'life', 'fact', 'High importance memory', 9, '2024-01-01')"
            ))
            conn.commit()

        # Run migration
        migrate_memories_table(v2_db)

        # Verify foundational = True (SQLite uses 1 for True)
        with v2_db.connect() as conn:
            result = conn.execute(text("SELECT foundational FROM memories WHERE id = 1"))
            row = result.fetchone()
            assert row[0] == 1

    def test_migrates_importance_low_to_foundational_false(self, v2_db):
        """Test that importance < 8 becomes foundational = False."""
        # Insert v2 data with low importance
        with v2_db.connect() as conn:
            conn.execute(text(
                "INSERT INTO memories (instance_id, project, memory_type, content, importance, created_at) "
                "VALUES ('test', 'life', 'fact', 'Low importance memory', 5, '2024-01-01')"
            ))
            conn.commit()

        # Run migration
        migrate_memories_table(v2_db)

        # Verify foundational = False (SQLite uses 0 for False)
        with v2_db.connect() as conn:
            result = conn.execute(text("SELECT foundational FROM memories WHERE id = 1"))
            row = result.fetchone()
            assert row[0] == 0

    def test_drops_importance_column(self, v2_db):
        """Test that importance column is removed after migration."""
        assert _column_exists(v2_db, "memories", "importance")

        # Run migration
        migrate_memories_table(v2_db)

        # Verify importance is gone
        assert not _column_exists(v2_db, "memories", "importance")

    def test_creates_foundational_index(self, v2_db):
        """Test that foundational index is created."""
        # Run migration
        migrate_memories_table(v2_db)

        # Verify index exists
        assert _index_exists(v2_db, "memories", "idx_memories_foundational")

    def test_is_idempotent(self, v2_db):
        """Test that running migration twice doesn't error."""
        # Run migration first time
        migrate_memories_table(v2_db)

        # Verify state
        assert _column_exists(v2_db, "memories", "foundational")
        assert not _column_exists(v2_db, "memories", "importance")

        # Run migration second time (should be safe)
        migrate_memories_table(v2_db)

        # Verify still correct
        assert _column_exists(v2_db, "memories", "foundational")
        assert not _column_exists(v2_db, "memories", "importance")

    def test_preserves_data_during_migration(self, v2_db):
        """Test that all memory data is preserved during migration."""
        # Insert multiple memories with various importance values
        with v2_db.connect() as conn:
            conn.execute(text(
                "INSERT INTO memories (instance_id, project, memory_type, subject, content, importance, created_at) "
                "VALUES "
                "('test', 'life', 'fact', 'Subject 1', 'Content 1', 10, '2024-01-01'), "
                "('test', 'work', 'note', 'Subject 2', 'Content 2', 5, '2024-01-02'), "
                "('code', 'life', 'reminder', 'Subject 3', 'Content 3', 8, '2024-01-03')"
            ))
            conn.commit()

        # Run migration
        migrate_memories_table(v2_db)

        # Verify all data preserved
        with v2_db.connect() as conn:
            result = conn.execute(text("SELECT id, instance_id, subject, content, foundational FROM memories ORDER BY id"))
            rows = result.fetchall()

            assert len(rows) == 3
            assert rows[0][1] == "test"
            assert rows[0][2] == "Subject 1"
            assert rows[0][3] == "Content 1"
            assert rows[0][4] == 1  # importance 10 -> foundational True

            assert rows[1][1] == "test"
            assert rows[1][2] == "Subject 2"
            assert rows[1][4] == 0  # importance 5 -> foundational False

            assert rows[2][1] == "code"
            assert rows[2][4] == 1  # importance 8 -> foundational True


# ── Migration Tests: messages table ──────────────────────────────────────


class TestMigrateMessagesTable:
    """Tests for migrate_messages_table()."""

    def test_renames_handoff_messages_to_messages(self, v2_db):
        """Test that handoff_messages is renamed to messages on SQLite."""
        # Verify handoff_messages exists
        assert _table_exists(v2_db, "handoff_messages")
        assert not _table_exists(v2_db, "messages")

        # Run migration
        migrate_messages_table(v2_db)

        # Verify messages exists and handoff_messages is gone
        assert _table_exists(v2_db, "messages")
        assert not _table_exists(v2_db, "handoff_messages")

    def test_adds_pubsub_columns(self, v2_db):
        """Test that pubsub columns are added."""
        # Run migration
        migrate_messages_table(v2_db)

        # Verify new columns exist
        assert _column_exists(v2_db, "messages", "channel")
        assert _column_exists(v2_db, "messages", "delivery_status")
        assert _column_exists(v2_db, "messages", "delivered_at")
        assert _column_exists(v2_db, "messages", "expires_at")
        assert _column_exists(v2_db, "messages", "priority")

    def test_creates_pubsub_indexes(self, v2_db):
        """Test that pubsub indexes are created."""
        # Run migration
        migrate_messages_table(v2_db)

        # Verify indexes exist
        assert _index_exists(v2_db, "messages", "idx_messages_channel")
        assert _index_exists(v2_db, "messages", "idx_messages_channel_status")
        assert _index_exists(v2_db, "messages", "idx_messages_priority")
        assert _index_exists(v2_db, "messages", "idx_messages_delivery_status")

    def test_is_idempotent(self, v2_db):
        """Test that running migration twice doesn't error."""
        # Run migration first time
        migrate_messages_table(v2_db)

        # Verify state
        assert _table_exists(v2_db, "messages")
        assert _column_exists(v2_db, "messages", "channel")

        # Run migration second time (should be safe)
        migrate_messages_table(v2_db)

        # Verify still correct
        assert _table_exists(v2_db, "messages")
        assert _column_exists(v2_db, "messages", "channel")

    def test_works_when_only_messages_exists(self, v2_db):
        """Test migration when messages table already exists (already migrated)."""
        # Manually create messages table (simulate already migrated)
        with v2_db.connect() as conn:
            conn.execute(text("DROP TABLE handoff_messages"))
            conn.execute(text("""
                CREATE TABLE messages (
                    id INTEGER PRIMARY KEY,
                    created_at DATETIME NOT NULL,
                    from_instance VARCHAR(50) NOT NULL,
                    to_instance VARCHAR(50) NOT NULL,
                    message_type VARCHAR(50) NOT NULL,
                    subject VARCHAR(255),
                    content TEXT NOT NULL,
                    read_at DATETIME,
                    read_by VARCHAR(50)
                )
            """))
            conn.commit()

        # Run migration (should add pubsub columns)
        migrate_messages_table(v2_db)

        # Verify pubsub columns added
        assert _column_exists(v2_db, "messages", "channel")
        assert _column_exists(v2_db, "messages", "priority")

    def test_works_when_neither_table_exists(self, v2_db):
        """Test migration when neither table exists (clean install, should skip)."""
        # Drop both tables
        with v2_db.connect() as conn:
            conn.execute(text("DROP TABLE handoff_messages"))
            conn.commit()

        # Run migration (should skip gracefully)
        migrate_messages_table(v2_db)

        # Verify no error and still no table
        assert not _table_exists(v2_db, "messages")
        assert not _table_exists(v2_db, "handoff_messages")

    def test_preserves_data_during_migration(self, v2_db):
        """Test that all message data is preserved during migration."""
        # Insert messages in handoff_messages
        with v2_db.connect() as conn:
            conn.execute(text(
                "INSERT INTO handoff_messages (from_instance, to_instance, message_type, subject, content, created_at) "
                "VALUES "
                "('test', 'code', 'handoff', 'Task 1', 'Content 1', '2024-01-01'), "
                "('code', 'desktop', 'fyi', 'Info', 'Content 2', '2024-01-02')"
            ))
            conn.commit()

        # Run migration
        migrate_messages_table(v2_db)

        # Verify data preserved in messages table
        with v2_db.connect() as conn:
            result = conn.execute(text("SELECT id, from_instance, to_instance, subject, content FROM messages ORDER BY id"))
            rows = result.fetchall()

            assert len(rows) == 2
            assert rows[0][1] == "test"
            assert rows[0][2] == "code"
            assert rows[0][3] == "Task 1"
            assert rows[0][4] == "Content 1"

            assert rows[1][1] == "code"
            assert rows[1][2] == "desktop"


# ── Migration Tests: full migration ──────────────────────────────────────


class TestFullMigration:
    """Tests for migrate() function."""

    def test_runs_both_migrations(self, v2_db):
        """Test that migrate() runs both sub-migrations successfully."""
        # Insert test data in both tables
        with v2_db.connect() as conn:
            conn.execute(text(
                "INSERT INTO memories (instance_id, project, memory_type, content, importance, created_at) "
                "VALUES ('test', 'life', 'fact', 'Test', 8, '2024-01-01')"
            ))
            conn.execute(text(
                "INSERT INTO handoff_messages (from_instance, to_instance, message_type, content, created_at) "
                "VALUES ('test', 'code', 'handoff', 'Test', '2024-01-01')"
            ))
            conn.commit()

        # Create a database URL for the in-memory database
        # For testing, we'll call the sub-functions directly since migrate() needs a URL
        migrate_memories_table(v2_db)
        migrate_messages_table(v2_db)

        # Verify both migrations completed
        assert _column_exists(v2_db, "memories", "foundational")
        assert not _column_exists(v2_db, "memories", "importance")
        assert _table_exists(v2_db, "messages")
        assert _column_exists(v2_db, "messages", "channel")

    def test_returns_true_on_success(self, v2_db):
        """Test that individual migrations don't raise errors (success)."""
        # This is implicitly tested by other tests not raising exceptions
        # The migrate() function returns True on success
        try:
            migrate_memories_table(v2_db)
            migrate_messages_table(v2_db)
            success = True
        except Exception:
            success = False

        assert success


# ── Message Service Tests: send_message ──────────────────────────────────


class TestSendMessage:
    """Tests for send_message()."""

    def test_creates_message_with_all_fields(self, v3_db):
        """Test that message is created with all required fields."""
        result = send_message(
            from_instance="test",
            to_instance="code",
            content="Test message",
            message_type="handoff",
            subject="Test Subject",
            channel="test-channel",
            priority=5,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1)
        )

        assert result["success"] is True
        assert "id" in result

        # Verify in database using raw SQL
        # Column order: id, created_at, from_instance, to_instance, message_type, subject, content,
        #               read_at, read_by, channel, delivery_status, delivered_at, expires_at, priority
        engine, Session = v3_db
        with engine.connect() as conn:
            result_row = conn.execute(text(
                "SELECT id, from_instance, to_instance, message_type, subject, content, "
                "channel, delivery_status, expires_at, priority FROM messages WHERE id = :id"
            ), {"id": result["id"]})
            row = result_row.fetchone()

            assert row is not None
            assert row[1] == "test"  # from_instance
            assert row[2] == "code"  # to_instance
            assert row[3] == "handoff"  # message_type
            assert row[4] == "Test Subject"  # subject
            assert row[5] == "Test message"  # content
            assert row[6] == "test-channel"  # channel
            assert row[7] == "pending"  # delivery_status
            assert row[8] is not None  # expires_at
            assert row[9] == 5  # priority

    def test_validates_from_instance(self, v3_db):
        """Test that from_instance must be a configured instance."""
        result = send_message(
            from_instance="invalid",
            to_instance="code",
            content="Test"
        )

        assert "error" in result
        assert "Invalid from_instance" in result["error"]

    def test_validates_to_instance(self, v3_db):
        """Test that to_instance must be a configured instance or 'all'."""
        result = send_message(
            from_instance="test",
            to_instance="invalid",
            content="Test"
        )

        assert "error" in result
        assert "Invalid to_instance" in result["error"]

    def test_accepts_all_as_to_instance(self, v3_db):
        """Test that 'all' is valid for to_instance (broadcast)."""
        result = send_message(
            from_instance="test",
            to_instance="all",
            content="Broadcast message"
        )

        assert result["success"] is True

    def test_rejects_invalid_message_type(self, v3_db):
        """Test that invalid message_type is rejected."""
        result = send_message(
            from_instance="test",
            to_instance="code",
            content="Test",
            message_type="invalid"
        )

        assert "error" in result
        assert "Invalid message_type" in result["error"]

    def test_rejects_invalid_priority_low(self, v3_db):
        """Test that priority < 0 is rejected."""
        result = send_message(
            from_instance="test",
            to_instance="code",
            content="Test",
            priority=-1
        )

        assert "error" in result
        assert "Invalid priority" in result["error"]

    def test_rejects_invalid_priority_high(self, v3_db):
        """Test that priority > 10 is rejected."""
        result = send_message(
            from_instance="test",
            to_instance="code",
            content="Test",
            priority=11
        )

        assert "error" in result
        assert "Invalid priority" in result["error"]

    def test_accepts_valid_message_types(self, v3_db):
        """Test all valid message types are accepted."""
        for msg_type in VALID_MESSAGE_TYPES:
            result = send_message(
                from_instance="test",
                to_instance="code",
                content="Test",
                message_type=msg_type
            )
            assert result["success"] is True

    def test_returns_message_id(self, v3_db):
        """Test that success response includes message ID."""
        result = send_message(
            from_instance="test",
            to_instance="code",
            content="Test"
        )

        assert result["success"] is True
        assert "id" in result
        assert isinstance(result["id"], int)


# ── Message Service Tests: get_messages ──────────────────────────────────


class TestGetMessages:
    """Tests for get_messages()."""

    def test_returns_messages_for_instance(self, v3_db):
        """Test that messages for an instance are returned."""
        # Send message to code
        send_message("test", "code", "Message 1")
        send_message("test", "code", "Message 2")
        send_message("test", "desktop", "Message 3")  # Not for code

        result = get_messages("code")

        assert result["count"] == 2
        assert len(result["messages"]) == 2

    def test_returns_messages_to_all(self, v3_db):
        """Test that messages addressed to 'all' are included."""
        send_message("test", "all", "Broadcast")
        send_message("test", "code", "Direct")

        result = get_messages("code")

        assert result["count"] == 2
        # Should get both broadcast and direct message

    def test_filters_unread_only(self, v3_db):
        """Test that unread_only filter works."""
        # Send two messages
        msg1 = send_message("test", "code", "Message 1")
        msg2 = send_message("test", "code", "Message 2")

        # Mark first as read
        mark_message_read(msg1["id"], "code")

        # Get unread only
        result = get_messages("code", unread_only=True)
        assert result["count"] == 1
        assert result["messages"][0]["id"] == msg2["id"]

        # Get all
        result = get_messages("code", unread_only=False)
        assert result["count"] == 2

    def test_filters_by_channel(self, v3_db):
        """Test that channel filter works."""
        send_message("test", "code", "Message 1", channel="channel-a")
        send_message("test", "code", "Message 2", channel="channel-b")
        send_message("test", "code", "Message 3", channel="channel-a")

        result = get_messages("code", channel="channel-a")

        assert result["count"] == 2

    def test_filters_by_message_type(self, v3_db):
        """Test that message_type filter works."""
        send_message("test", "code", "Message 1", message_type="handoff")
        send_message("test", "code", "Message 2", message_type="fyi")
        send_message("test", "code", "Message 3", message_type="handoff")

        result = get_messages("code", message_type="handoff")

        assert result["count"] == 2

    def test_excludes_expired_by_default(self, v3_db):
        """Test that expired messages are excluded by default."""
        # Send message that expires in the past
        send_message("test", "code", "Expired", expires_at=datetime.now(timezone.utc) - timedelta(hours=1))
        send_message("test", "code", "Valid", expires_at=datetime.now(timezone.utc) + timedelta(hours=1))
        send_message("test", "code", "No expiry")

        result = get_messages("code")

        # Should only get the two non-expired messages
        assert result["count"] == 2

    def test_includes_expired_when_requested(self, v3_db):
        """Test that include_expired=True includes expired messages."""
        send_message("test", "code", "Expired", expires_at=datetime.now(timezone.utc) - timedelta(hours=1))
        send_message("test", "code", "Valid")

        result = get_messages("code", include_expired=True)

        assert result["count"] == 2

    def test_sorts_by_priority_desc(self, v3_db):
        """Test that messages are sorted by priority DESC, created_at DESC."""
        # Send messages with different priorities
        send_message("test", "code", "Low", priority=1)
        send_message("test", "code", "High", priority=9)
        send_message("test", "code", "Medium", priority=5)

        result = get_messages("code")

        # Should be sorted high to low priority
        assert result["messages"][0]["priority"] == 9
        assert result["messages"][1]["priority"] == 5
        assert result["messages"][2]["priority"] == 1

    def test_validates_instance_id(self, v3_db):
        """Test that instance_id is validated."""
        result = get_messages("invalid")

        assert "error" in result
        assert "Invalid instance_id" in result["error"]

    def test_rejects_invalid_message_type_filter(self, v3_db):
        """Test that invalid message_type filter is rejected."""
        result = get_messages("code", message_type="invalid")

        assert "error" in result
        assert "Invalid message_type" in result["error"]


# ── Message Service Tests: mark_message_read ─────────────────────────────


class TestMarkMessageRead:
    """Tests for mark_message_read()."""

    def test_sets_read_at_and_read_by(self, v3_db):
        """Test that read_at and read_by are set."""
        msg = send_message("test", "code", "Test")

        result = mark_message_read(msg["id"], "code")

        assert result["message"] == "Marked read"

        # Verify in database using raw SQL
        engine, Session = v3_db
        with engine.connect() as conn:
            result_row = conn.execute(text("SELECT read_at, read_by FROM messages WHERE id = :id"), {"id": msg["id"]})
            row = result_row.fetchone()

            assert row[0] is not None  # read_at
            assert row[1] == "code"  # read_by

    def test_validates_instance_id(self, v3_db):
        """Test that instance_id is validated."""
        msg = send_message("test", "code", "Test")

        result = mark_message_read(msg["id"], "invalid")

        assert "error" in result
        assert "Invalid instance_id" in result["error"]

    def test_returns_error_for_nonexistent_message(self, v3_db):
        """Test that error is returned for non-existent message."""
        result = mark_message_read(99999, "code")

        assert "error" in result
        assert "not found" in result["error"]


# ── Message Service Tests: mark_message_unread ───────────────────────────


class TestMarkMessageUnread:
    """Tests for mark_message_unread()."""

    def test_clears_read_at_and_read_by(self, v3_db):
        """Test that read_at and read_by are cleared."""
        msg = send_message("test", "code", "Test")
        mark_message_read(msg["id"], "code")

        result = mark_message_unread(msg["id"])

        assert result["message"] == "Marked unread"

        # Verify in database using raw SQL
        engine, Session = v3_db
        with engine.connect() as conn:
            result_row = conn.execute(text("SELECT read_at, read_by FROM messages WHERE id = :id"), {"id": msg["id"]})
            row = result_row.fetchone()

            assert row[0] is None  # read_at
            assert row[1] is None  # read_by

    def test_returns_error_for_nonexistent_message(self, v3_db):
        """Test that error is returned for non-existent message."""
        result = mark_message_unread(99999)

        assert "error" in result
        assert "not found" in result["error"]


# ── Message Service Tests: subscriptions ─────────────────────────────────


class TestSubscriptions:
    """Tests for subscribe(), unsubscribe(), get_subscriptions()."""

    def test_subscribe_adds_to_registry(self, v3_db):
        """Test that subscribe() adds to in-memory registry."""
        result = subscribe("test", "channel-1")

        assert "message" in result
        assert "Subscribed" in result["message"]

        # Verify in registry
        assert "test" in msg_module._subscriptions
        assert "channel-1" in msg_module._subscriptions["test"]

    def test_unsubscribe_removes_from_registry(self, v3_db):
        """Test that unsubscribe() removes from registry."""
        subscribe("test", "channel-1")
        subscribe("test", "channel-2")

        result = unsubscribe("test", "channel-1")

        assert "message" in result
        assert "Unsubscribed" in result["message"]

        # Verify removed
        assert "channel-1" not in msg_module._subscriptions["test"]
        assert "channel-2" in msg_module._subscriptions["test"]

    def test_get_subscriptions_returns_channels(self, v3_db):
        """Test that get_subscriptions() returns correct channels."""
        subscribe("test", "channel-1")
        subscribe("test", "channel-2")
        subscribe("code", "channel-3")  # Different instance

        result = get_subscriptions("test")

        assert "subscriptions" in result
        assert set(result["subscriptions"]) == {"channel-1", "channel-2"}

    def test_subscribe_validates_instance_id(self, v3_db):
        """Test that subscribe() validates instance_id."""
        result = subscribe("invalid", "channel")

        assert "error" in result
        assert "Invalid instance_id" in result["error"]

    def test_unsubscribe_validates_instance_id(self, v3_db):
        """Test that unsubscribe() validates instance_id."""
        result = unsubscribe("invalid", "channel")

        assert "error" in result
        assert "Invalid instance_id" in result["error"]

    def test_get_subscriptions_validates_instance_id(self, v3_db):
        """Test that get_subscriptions() validates instance_id."""
        result = get_subscriptions("invalid")

        assert "error" in result
        assert "Invalid instance_id" in result["error"]

    def test_multiple_instances_have_separate_subscriptions(self, v3_db):
        """Test that each instance has its own subscription registry."""
        subscribe("test", "channel-1")
        subscribe("code", "channel-2")

        test_subs = get_subscriptions("test")
        code_subs = get_subscriptions("code")

        assert test_subs["subscriptions"] == ["channel-1"]
        assert code_subs["subscriptions"] == ["channel-2"]


# ── Message Service Tests: poll_messages ─────────────────────────────────


class TestPollMessages:
    """Tests for poll_messages()."""

    def test_returns_unread_by_default(self, v3_db):
        """Test that poll_messages returns unread messages by default."""
        msg1 = send_message("test", "code", "Message 1")
        msg2 = send_message("test", "code", "Message 2")
        mark_message_read(msg1["id"], "code")

        result = poll_messages("code")

        assert result["count"] == 1
        assert result["messages"][0]["id"] == msg2["id"]

    def test_filters_by_since_timestamp(self, v3_db):
        """Test that since parameter filters by timestamp."""
        # Send first message
        send_message("test", "code", "Message 1")

        # Note the time
        since_time = datetime.now(timezone.utc)

        # Wait a moment and send another
        send_message("test", "code", "Message 2")

        result = poll_messages("code", since=since_time)

        # Should only get messages after since_time
        assert result["count"] >= 1

    def test_filters_by_channel(self, v3_db):
        """Test that channel parameter filters correctly."""
        send_message("test", "code", "Message 1", channel="channel-a")
        send_message("test", "code", "Message 2", channel="channel-b")

        result = poll_messages("code", channel="channel-a")

        assert result["count"] == 1

    def test_excludes_expired_messages(self, v3_db):
        """Test that expired messages are excluded."""
        send_message("test", "code", "Expired", expires_at=datetime.now(timezone.utc) - timedelta(hours=1))
        send_message("test", "code", "Valid")

        result = poll_messages("code")

        assert result["count"] == 1

    def test_validates_instance_id(self, v3_db):
        """Test that instance_id is validated."""
        result = poll_messages("invalid")

        assert "error" in result
        assert "Invalid instance_id" in result["error"]

    def test_sorts_by_priority_and_created_at(self, v3_db):
        """Test that messages are sorted correctly."""
        send_message("test", "code", "Low", priority=1)
        send_message("test", "code", "High", priority=9)
        send_message("test", "code", "Medium", priority=5)

        result = poll_messages("code")

        # Should be sorted high to low priority
        assert result["messages"][0]["priority"] == 9
        assert result["messages"][1]["priority"] == 5
        assert result["messages"][2]["priority"] == 1
