"""
Tests for SQLite to PostgreSQL migration.

Note: These tests require a PostgreSQL database to be available.
Set MEMORY_PALACE_TEST_POSTGRES_URL to enable PostgreSQL tests.

Example:
    export MEMORY_PALACE_TEST_POSTGRES_URL="postgresql://localhost:5432/memory_palace_test"
    pytest tests/test_sqlite_to_postgres_migration.py
"""
import json
import os
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from memory_palace.migrations.sqlite_to_postgres import (
    verify_source_database,
    verify_target_database,
    migrate_memories,
    migrate_memory_edges,
    migrate_messages,
    verify_migration,
    _get_row_count,
)
from memory_palace.models_v3 import Base


# Skip tests if PostgreSQL URL not provided
POSTGRES_URL = os.environ.get("MEMORY_PALACE_TEST_POSTGRES_URL")
SKIP_POSTGRES_TESTS = not POSTGRES_URL

pytestmark = pytest.mark.skipif(
    SKIP_POSTGRES_TESTS,
    reason="PostgreSQL tests require MEMORY_PALACE_TEST_POSTGRES_URL environment variable"
)


@pytest.fixture
def sqlite_engine():
    """Create in-memory SQLite database with v3 schema."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture
def postgres_engine():
    """Create PostgreSQL test database with v3 schema."""
    if not POSTGRES_URL:
        pytest.skip("PostgreSQL tests require MEMORY_PALACE_TEST_POSTGRES_URL")

    engine = create_engine(POSTGRES_URL)

    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    yield engine

    Base.metadata.drop_all(bind=engine)


def test_verify_source_database_success(sqlite_engine):
    """Test source database verification succeeds for valid SQLite database."""
    success, message = verify_source_database(sqlite_engine)
    assert success
    assert "verified" in message.lower()


def test_verify_source_database_missing_table(sqlite_engine):
    """Test source verification fails if required tables missing."""
    with sqlite_engine.connect() as conn:
        conn.execute(text("DROP TABLE memories"))
        conn.commit()

    success, message = verify_source_database(sqlite_engine)
    assert not success
    assert "missing" in message.lower()


def test_verify_target_database_success(postgres_engine):
    """Test target database verification succeeds for valid PostgreSQL database."""
    success, message = verify_target_database(postgres_engine)
    assert success
    assert "verified" in message.lower()
    assert "pgvector" in message.lower()


def test_migrate_memories_basic(sqlite_engine, postgres_engine):
    """Test basic memories table migration."""
    with sqlite_engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO memories (
                id, instance_id, projects, memory_type, subject, content,
                keywords, tags, foundational
            ) VALUES (
                1, 'test', '["project1"]', 'fact', 'test subject', 'test content',
                '["key1", "key2"]', '["tag1"]', 0
            )
        """))
        conn.commit()

    rows_read, rows_written = migrate_memories(sqlite_engine, postgres_engine)

    assert rows_read == 1
    assert rows_written == 1

    with postgres_engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM memories WHERE id = 1"))
        row = result.fetchone()
        assert row is not None
        assert row.instance_id == "test"
        assert row.subject == "test subject"
        assert row.content == "test content"


def test_migrate_memories_with_embedding(sqlite_engine, postgres_engine):
    """Test memories migration with embedding conversion."""
    embedding = [0.1, 0.2, 0.3] * 256

    with sqlite_engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO memories (
                id, instance_id, projects, memory_type, content,
                embedding
            ) VALUES (
                1, 'test', '["project1"]', 'fact', 'test content',
                :embedding
            )
        """), {"embedding": json.dumps(embedding)})
        conn.commit()

    rows_read, rows_written = migrate_memories(sqlite_engine, postgres_engine)

    assert rows_read == 1
    assert rows_written == 1

    with postgres_engine.connect() as conn:
        result = conn.execute(text("SELECT embedding FROM memories WHERE id = 1"))
        row = result.fetchone()
        assert row.embedding is not None


def test_migrate_memories_arrays(sqlite_engine, postgres_engine):
    """Test conversion of JSON arrays to PostgreSQL ARRAY type."""
    with sqlite_engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO memories (
                id, instance_id, projects, memory_type, content,
                keywords, tags
            ) VALUES (
                1, 'test', '["proj1", "proj2"]', 'fact', 'test',
                '["key1", "key2", "key3"]', '["tag1", "tag2"]'
            )
        """))
        conn.commit()

    rows_read, rows_written = migrate_memories(sqlite_engine, postgres_engine)

    assert rows_read == 1
    assert rows_written == 1

    with postgres_engine.connect() as conn:
        result = conn.execute(text("SELECT projects, keywords, tags FROM memories WHERE id = 1"))
        row = result.fetchone()

        assert isinstance(row.projects, list)
        assert row.projects == ["proj1", "proj2"]

        assert isinstance(row.keywords, list)
        assert row.keywords == ["key1", "key2", "key3"]

        assert isinstance(row.tags, list)
        assert row.tags == ["tag1", "tag2"]


def test_migrate_memory_edges_basic(sqlite_engine, postgres_engine):
    """Test basic memory_edges migration."""
    with sqlite_engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO memories (id, instance_id, projects, memory_type, content)
            VALUES (1, 'test', '["project1"]', 'fact', 'mem1'),
                   (2, 'test', '["project1"]', 'fact', 'mem2')
        """))

        conn.execute(text("""
            INSERT INTO memory_edges (
                id, source_id, target_id, relation_type, strength, bidirectional
            ) VALUES (
                1, 1, 2, 'relates_to', 0.8, 0
            )
        """))
        conn.commit()

    with postgres_engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO memories (id, instance_id, projects, memory_type, content)
            VALUES (1, 'test', ARRAY['project1'], 'fact', 'mem1'),
                   (2, 'test', ARRAY['project1'], 'fact', 'mem2')
        """))
        conn.commit()

    rows_read, rows_written = migrate_memory_edges(sqlite_engine, postgres_engine)

    assert rows_read == 1
    assert rows_written == 1

    with postgres_engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM memory_edges WHERE id = 1"))
        row = result.fetchone()
        assert row.source_id == 1
        assert row.target_id == 2
        assert row.relation_type == "relates_to"
        assert abs(row.strength - 0.8) < 0.01


def test_migrate_memory_edges_with_metadata(sqlite_engine, postgres_engine):
    """Test memory_edges migration with JSON metadata conversion to JSONB."""
    with sqlite_engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO memories (id, instance_id, projects, memory_type, content)
            VALUES (1, 'test', '["project1"]', 'fact', 'mem1'),
                   (2, 'test', '["project1"]', 'fact', 'mem2')
        """))

        metadata = {"note": "test metadata", "confidence": 0.9}
        conn.execute(text("""
            INSERT INTO memory_edges (
                id, source_id, target_id, relation_type, edge_metadata
            ) VALUES (
                1, 1, 2, 'relates_to', :metadata
            )
        """), {"metadata": json.dumps(metadata)})
        conn.commit()

    with postgres_engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO memories (id, instance_id, projects, memory_type, content)
            VALUES (1, 'test', ARRAY['project1'], 'fact', 'mem1'),
                   (2, 'test', ARRAY['project1'], 'fact', 'mem2')
        """))
        conn.commit()

    rows_read, rows_written = migrate_memory_edges(sqlite_engine, postgres_engine)

    assert rows_read == 1
    assert rows_written == 1

    with postgres_engine.connect() as conn:
        result = conn.execute(text("SELECT edge_metadata FROM memory_edges WHERE id = 1"))
        row = result.fetchone()
        assert row.edge_metadata["note"] == "test metadata"
        assert abs(row.edge_metadata["confidence"] - 0.9) < 0.01


def test_migrate_messages_basic(sqlite_engine, postgres_engine):
    """Test basic messages migration."""
    with sqlite_engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO messages (
                id, from_instance, to_instance, message_type, content,
                channel, priority
            ) VALUES (
                1, 'instance1', 'instance2', 'handoff', 'test message',
                'test_channel', 5
            )
        """))
        conn.commit()

    rows_read, rows_written = migrate_messages(sqlite_engine, postgres_engine)

    assert rows_read == 1
    assert rows_written == 1

    with postgres_engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM messages WHERE id = 1"))
        row = result.fetchone()
        assert row.from_instance == "instance1"
        assert row.to_instance == "instance2"
        assert row.content == "test message"
        assert row.channel == "test_channel"
        assert row.priority == 5


def test_migrate_messages_batch(sqlite_engine, postgres_engine):
    """Test batch processing of messages."""
    with sqlite_engine.connect() as conn:
        for i in range(1, 251):
            conn.execute(text("""
                INSERT INTO messages (
                    id, from_instance, to_instance, message_type, content
                ) VALUES (
                    :id, 'instance1', 'instance2', 'message', :content
                )
            """), {"id": i, "content": f"message {i}"})
        conn.commit()

    rows_read, rows_written = migrate_messages(sqlite_engine, postgres_engine, batch_size=100)

    assert rows_read == 250
    assert rows_written == 250
    assert _get_row_count(postgres_engine, "messages") == 250


def test_verify_migration_success(sqlite_engine, postgres_engine):
    """Test migration verification when all counts match."""
    with sqlite_engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO memories (id, instance_id, projects, memory_type, content)
            VALUES (1, 'test', '["project1"]', 'fact', 'test')
        """))
        conn.execute(text("""
            INSERT INTO messages (id, from_instance, to_instance, message_type, content)
            VALUES (1, 'i1', 'i2', 'message', 'test')
        """))
        conn.commit()

    migrate_memories(sqlite_engine, postgres_engine)
    migrate_messages(sqlite_engine, postgres_engine)

    result = verify_migration(sqlite_engine, postgres_engine)
    assert result is True


def test_dry_run_mode(sqlite_engine, postgres_engine):
    """Test dry run mode doesn't write data."""
    with sqlite_engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO memories (id, instance_id, projects, memory_type, content)
            VALUES (1, 'test', '["project1"]', 'fact', 'test')
        """))
        conn.commit()

    rows_read, rows_written = migrate_memories(sqlite_engine, postgres_engine, dry_run=True)

    assert rows_read == 1
    assert rows_written == 1

    assert _get_row_count(postgres_engine, "memories") == 0


def test_on_conflict_do_nothing(sqlite_engine, postgres_engine):
    """Test that re-running migration skips existing rows."""
    with sqlite_engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO memories (id, instance_id, projects, memory_type, content)
            VALUES (1, 'test', '["project1"]', 'fact', 'test')
        """))
        conn.commit()

    migrate_memories(sqlite_engine, postgres_engine)
    assert _get_row_count(postgres_engine, "memories") == 1

    migrate_memories(sqlite_engine, postgres_engine)
    assert _get_row_count(postgres_engine, "memories") == 1
