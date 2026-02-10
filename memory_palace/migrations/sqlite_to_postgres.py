"""
SQLite to PostgreSQL data migration for Memory Palace.

For users who started on SQLite (1.x) and want to upgrade to PostgreSQL
for native vector search (pgvector), concurrent access, etc.

This script:
1. Connects to both source SQLite and target PostgreSQL databases
2. Verifies prerequisites (target exists, pgvector installed, tables exist)
3. Migrates all tables: memories, memory_edges, messages
4. Converts data types appropriately:
   - JSON text embeddings → pgvector Vector
   - JSON text arrays → PostgreSQL ARRAY(Text)
   - Integer booleans → PostgreSQL BOOLEAN
   - JSON metadata → JSONB
5. Preserves IDs to maintain foreign key relationships
6. Verifies row counts after migration

Usage:
    memory-palace-sqlite-to-pg --source "sqlite:///path/to/memories.db" --target "postgresql://localhost:5432/memory_palace"
"""

import argparse
import json
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from memory_palace.config_v2 import get_database_url, get_database_type, ensure_data_dir


def _is_postgres(engine: Engine) -> bool:
    """Check if engine is PostgreSQL."""
    return engine.dialect.name == "postgresql"


def _is_sqlite(engine: Engine) -> bool:
    """Check if engine is SQLite."""
    return engine.dialect.name == "sqlite"


def _table_exists(engine: Engine, table_name: str) -> bool:
    """Check if a table exists."""
    inspector = inspect(engine)
    return table_name in inspector.get_table_names()


def _get_row_count(engine: Engine, table_name: str) -> int:
    """Get row count for a table."""
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
        return result.scalar()


def verify_source_database(source_engine: Engine) -> Tuple[bool, str]:
    """
    Verify source database is SQLite and has expected tables.

    Returns:
        (success, message)
    """
    if not _is_sqlite(source_engine):
        return False, "Source database must be SQLite"

    required_tables = ["memories", "memory_edges", "messages"]
    missing = [t for t in required_tables if not _table_exists(source_engine, t)]

    if missing:
        return False, f"Source database missing required tables: {missing}"

    return True, "Source database verified"


def verify_target_database(target_engine: Engine) -> Tuple[bool, str]:
    """
    Verify target database is PostgreSQL with pgvector and tables created.

    Returns:
        (success, message)
    """
    if not _is_postgres(target_engine):
        return False, "Target database must be PostgreSQL"

    with target_engine.connect() as conn:
        result = conn.execute(text(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ))
        pgvector_version = result.scalar()

        if not pgvector_version:
            return False, "pgvector extension not installed. Run: CREATE EXTENSION vector;"

        required_tables = ["memories", "memory_edges", "messages"]
        missing = [t for t in required_tables if not _table_exists(target_engine, t)]

        if missing:
            return False, f"Target database missing tables: {missing}. Run the MCP server once to create tables with init_db()."

    return True, f"Target database verified (pgvector {pgvector_version})"


def check_target_has_data(target_engine: Engine) -> Dict[str, int]:
    """
    Check if target tables already have data.

    Returns:
        Dict mapping table names to row counts
    """
    tables = ["memories", "memory_edges", "messages"]
    counts = {}

    for table in tables:
        counts[table] = _get_row_count(target_engine, table)

    return counts


def migrate_memories(
    source_engine: Engine,
    target_engine: Engine,
    batch_size: int = 100,
    dry_run: bool = False
) -> Tuple[int, int]:
    """
    Migrate memories table.

    Converts:
    - embedding: JSON text → pgvector Vector
    - keywords/tags/projects: JSON text → ARRAY(Text)
    - is_archived/foundational: 0/1 → BOOLEAN

    Returns:
        (rows_read, rows_written)
    """
    print("\n=== Migrating memories table ===")

    with source_engine.connect() as src_conn:
        result = src_conn.execute(text("SELECT COUNT(*) FROM memories"))
        total = result.scalar()
        print(f"Source has {total} rows")

        if total == 0:
            print("No rows to migrate")
            return 0, 0

        offset = 0
        rows_written = 0

        while offset < total:
            result = src_conn.execute(text(
                f"SELECT * FROM memories ORDER BY id LIMIT {batch_size} OFFSET {offset}"
            ))
            rows = result.fetchall()
            columns = result.keys()

            if not rows:
                break

            print(f"Processing rows {offset + 1} to {offset + len(rows)}")

            if dry_run:
                print(f"  [DRY RUN] Would insert {len(rows)} rows")
                rows_written += len(rows)
                offset += len(rows)
                continue

            with target_engine.connect() as tgt_conn:
                for row in rows:
                    row_dict = dict(zip(columns, row))

                    embedding_val = None
                    if row_dict.get("embedding"):
                        try:
                            embedding_list = json.loads(row_dict["embedding"])
                            if isinstance(embedding_list, list) and len(embedding_list) > 0:
                                embedding_val = f"[{','.join(map(str, embedding_list))}]"
                        except (json.JSONDecodeError, TypeError):
                            print(f"  Warning: Could not parse embedding for row {row_dict['id']}")

                    keywords_val = None
                    if row_dict.get("keywords"):
                        try:
                            keywords_list = json.loads(row_dict["keywords"])
                            if isinstance(keywords_list, list):
                                keywords_val = keywords_list
                        except (json.JSONDecodeError, TypeError):
                            pass

                    tags_val = None
                    if row_dict.get("tags"):
                        try:
                            tags_list = json.loads(row_dict["tags"])
                            if isinstance(tags_list, list):
                                tags_val = tags_list
                        except (json.JSONDecodeError, TypeError):
                            pass

                    projects_val = None
                    if row_dict.get("projects"):
                        try:
                            if isinstance(row_dict["projects"], str):
                                projects_list = json.loads(row_dict["projects"])
                            else:
                                projects_list = row_dict["projects"]
                            if isinstance(projects_list, list):
                                projects_val = projects_list
                        except (json.JSONDecodeError, TypeError):
                            projects_val = ["life"]
                    else:
                        projects_val = ["life"]

                    tgt_conn.execute(text("""
                        INSERT INTO memories (
                            id, created_at, updated_at, instance_id, projects,
                            memory_type, subject, content, keywords, tags, foundational,
                            source_type, source_context, source_session_id, embedding,
                            last_accessed_at, access_count, expires_at, is_archived
                        ) VALUES (
                            :id, :created_at, :updated_at, :instance_id, :projects,
                            :memory_type, :subject, :content, :keywords, :tags, :foundational,
                            :source_type, :source_context, :source_session_id, :embedding,
                            :last_accessed_at, :access_count, :expires_at, :is_archived
                        )
                        ON CONFLICT (id) DO NOTHING
                    """), {
                        "id": row_dict["id"],
                        "created_at": row_dict.get("created_at"),
                        "updated_at": row_dict.get("updated_at"),
                        "instance_id": row_dict["instance_id"],
                        "projects": projects_val,
                        "memory_type": row_dict["memory_type"],
                        "subject": row_dict.get("subject"),
                        "content": row_dict["content"],
                        "keywords": keywords_val,
                        "tags": tags_val,
                        "foundational": bool(row_dict.get("foundational", 0)),
                        "source_type": row_dict.get("source_type"),
                        "source_context": row_dict.get("source_context"),
                        "source_session_id": row_dict.get("source_session_id"),
                        "embedding": embedding_val,
                        "last_accessed_at": row_dict.get("last_accessed_at"),
                        "access_count": row_dict.get("access_count", 0),
                        "expires_at": row_dict.get("expires_at"),
                        "is_archived": bool(row_dict.get("is_archived", 0)),
                    })

                tgt_conn.commit()
                rows_written += len(rows)

            offset += len(rows)

    print(f"Migration complete: {rows_written} rows written")
    return total, rows_written


def migrate_memory_edges(
    source_engine: Engine,
    target_engine: Engine,
    batch_size: int = 100,
    dry_run: bool = False
) -> Tuple[int, int]:
    """
    Migrate memory_edges table.

    Converts:
    - edge_metadata: JSON → JSONB
    - bidirectional: 0/1 → BOOLEAN

    Returns:
        (rows_read, rows_written)
    """
    print("\n=== Migrating memory_edges table ===")

    with source_engine.connect() as src_conn:
        result = src_conn.execute(text("SELECT COUNT(*) FROM memory_edges"))
        total = result.scalar()
        print(f"Source has {total} rows")

        if total == 0:
            print("No rows to migrate")
            return 0, 0

        offset = 0
        rows_written = 0

        while offset < total:
            result = src_conn.execute(text(
                f"SELECT * FROM memory_edges ORDER BY id LIMIT {batch_size} OFFSET {offset}"
            ))
            rows = result.fetchall()
            columns = result.keys()

            if not rows:
                break

            print(f"Processing rows {offset + 1} to {offset + len(rows)}")

            if dry_run:
                print(f"  [DRY RUN] Would insert {len(rows)} rows")
                rows_written += len(rows)
                offset += len(rows)
                continue

            with target_engine.connect() as tgt_conn:
                for row in rows:
                    row_dict = dict(zip(columns, row))

                    metadata_val = None
                    if row_dict.get("edge_metadata"):
                        try:
                            if isinstance(row_dict["edge_metadata"], str):
                                metadata_val = json.loads(row_dict["edge_metadata"])
                            else:
                                metadata_val = row_dict["edge_metadata"]
                        except (json.JSONDecodeError, TypeError):
                            metadata_val = {}
                    else:
                        metadata_val = {}

                    tgt_conn.execute(text("""
                        INSERT INTO memory_edges (
                            id, created_at, source_id, target_id,
                            relation_type, strength, bidirectional,
                            edge_metadata, created_by
                        ) VALUES (
                            :id, :created_at, :source_id, :target_id,
                            :relation_type, :strength, :bidirectional,
                            :edge_metadata, :created_by
                        )
                        ON CONFLICT (id) DO NOTHING
                    """), {
                        "id": row_dict["id"],
                        "created_at": row_dict.get("created_at"),
                        "source_id": row_dict["source_id"],
                        "target_id": row_dict["target_id"],
                        "relation_type": row_dict["relation_type"],
                        "strength": row_dict.get("strength", 1.0),
                        "bidirectional": bool(row_dict.get("bidirectional", 0)),
                        "edge_metadata": json.dumps(metadata_val),
                        "created_by": row_dict.get("created_by"),
                    })

                tgt_conn.commit()
                rows_written += len(rows)

            offset += len(rows)

    print(f"Migration complete: {rows_written} rows written")
    return total, rows_written


def migrate_messages(
    source_engine: Engine,
    target_engine: Engine,
    batch_size: int = 100,
    dry_run: bool = False
) -> Tuple[int, int]:
    """
    Migrate messages table.

    Direct copy with type adjustments (no complex conversions needed).

    Returns:
        (rows_read, rows_written)
    """
    print("\n=== Migrating messages table ===")

    with source_engine.connect() as src_conn:
        result = src_conn.execute(text("SELECT COUNT(*) FROM messages"))
        total = result.scalar()
        print(f"Source has {total} rows")

        if total == 0:
            print("No rows to migrate")
            return 0, 0

        offset = 0
        rows_written = 0

        while offset < total:
            result = src_conn.execute(text(
                f"SELECT * FROM messages ORDER BY id LIMIT {batch_size} OFFSET {offset}"
            ))
            rows = result.fetchall()
            columns = result.keys()

            if not rows:
                break

            print(f"Processing rows {offset + 1} to {offset + len(rows)}")

            if dry_run:
                print(f"  [DRY RUN] Would insert {len(rows)} rows")
                rows_written += len(rows)
                offset += len(rows)
                continue

            with target_engine.connect() as tgt_conn:
                for row in rows:
                    row_dict = dict(zip(columns, row))

                    tgt_conn.execute(text("""
                        INSERT INTO messages (
                            id, created_at, from_instance, to_instance,
                            message_type, subject, content, read_at, read_by,
                            channel, delivery_status, delivered_at, expires_at, priority
                        ) VALUES (
                            :id, :created_at, :from_instance, :to_instance,
                            :message_type, :subject, :content, :read_at, :read_by,
                            :channel, :delivery_status, :delivered_at, :expires_at, :priority
                        )
                        ON CONFLICT (id) DO NOTHING
                    """), {
                        "id": row_dict["id"],
                        "created_at": row_dict.get("created_at"),
                        "from_instance": row_dict["from_instance"],
                        "to_instance": row_dict["to_instance"],
                        "message_type": row_dict["message_type"],
                        "subject": row_dict.get("subject"),
                        "content": row_dict["content"],
                        "read_at": row_dict.get("read_at"),
                        "read_by": row_dict.get("read_by"),
                        "channel": row_dict.get("channel"),
                        "delivery_status": row_dict.get("delivery_status", "pending"),
                        "delivered_at": row_dict.get("delivered_at"),
                        "expires_at": row_dict.get("expires_at"),
                        "priority": row_dict.get("priority", 0),
                    })

                tgt_conn.commit()
                rows_written += len(rows)

            offset += len(rows)

    print(f"Migration complete: {rows_written} rows written")
    return total, rows_written


def verify_migration(source_engine: Engine, target_engine: Engine) -> bool:
    """
    Verify migration by comparing row counts.

    Returns:
        True if all counts match, False otherwise
    """
    print("\n=== Verifying migration ===")

    tables = ["memories", "memory_edges", "messages"]
    all_match = True

    for table in tables:
        source_count = _get_row_count(source_engine, table)
        target_count = _get_row_count(target_engine, table)

        match = "✓" if source_count == target_count else "✗"
        print(f"{table:20} source={source_count:6} target={target_count:6} {match}")

        if source_count != target_count:
            all_match = False

    return all_match


def migrate(
    source_url: str,
    target_url: str,
    force: bool = False,
    dry_run: bool = False,
    batch_size: int = 100
) -> bool:
    """
    Run the SQLite to PostgreSQL migration.

    Args:
        source_url: SQLite database URL
        target_url: PostgreSQL database URL
        force: Proceed even if target has data
        dry_run: Report what would be migrated without writing
        batch_size: Rows per batch (default 100)

    Returns:
        True if migration succeeded, False otherwise
    """
    start_time = time.time()

    print("=" * 70)
    print("Memory Palace: SQLite → PostgreSQL Migration")
    print("=" * 70)
    print()

    if dry_run:
        print("DRY RUN MODE: No data will be written")
        print()

    try:
        print(f"Source: {source_url}")
        print(f"Target: {target_url}")
        print()

        print("Connecting to databases...")
        source_engine = create_engine(source_url)
        target_engine = create_engine(target_url)

        success, message = verify_source_database(source_engine)
        print(f"Source verification: {message}")
        if not success:
            return False

        success, message = verify_target_database(target_engine)
        print(f"Target verification: {message}")
        if not success:
            return False

        target_counts = check_target_has_data(target_engine)
        total_target_rows = sum(target_counts.values())

        if total_target_rows > 0 and not force and not dry_run:
            print()
            print("WARNING: Target database already has data:")
            for table, count in target_counts.items():
                if count > 0:
                    print(f"  {table}: {count} rows")
            print()
            print("To proceed with migration anyway, use --force")
            print("This will skip existing rows (ON CONFLICT DO NOTHING)")
            return False

        print()

        memories_read, memories_written = migrate_memories(
            source_engine, target_engine, batch_size, dry_run
        )

        edges_read, edges_written = migrate_memory_edges(
            source_engine, target_engine, batch_size, dry_run
        )

        messages_read, messages_written = migrate_messages(
            source_engine, target_engine, batch_size, dry_run
        )

        if not dry_run:
            verification_passed = verify_migration(source_engine, target_engine)

            print()
            print("=" * 70)
            if verification_passed:
                print("Migration completed successfully!")
            else:
                print("Migration completed with warnings - row counts do not match")
            print("=" * 70)
        else:
            print()
            print("=" * 70)
            print("Dry run complete")
            print("=" * 70)

        elapsed = time.time() - start_time
        print()
        print(f"Summary:")
        print(f"  memories:     {memories_written} rows migrated")
        print(f"  memory_edges: {edges_written} rows migrated")
        print(f"  messages:     {messages_written} rows migrated")
        print(f"  Total time:   {elapsed:.2f} seconds")
        print()

        if dry_run:
            print("Run without --dry-run to perform the actual migration")
        else:
            print("Next steps:")
            print("1. Update your config to use PostgreSQL:")
            print(f'   {{"database": {{"type": "postgres", "url": "{target_url}"}}}}')
            print("2. Restart the MCP server")
            print("3. Verify embeddings are working with pgvector")

        return True if dry_run else verification_passed

    except SQLAlchemyError as e:
        print()
        print("=" * 70)
        print("Migration failed!")
        print("=" * 70)
        print(f"Database error: {e}")
        return False
    except Exception as e:
        print()
        print("=" * 70)
        print("Migration failed!")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """CLI entry point for SQLite to PostgreSQL migration."""
    parser = argparse.ArgumentParser(
        description="Migrate Memory Palace data from SQLite to PostgreSQL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run with default databases from config
  memory-palace-sqlite-to-pg --dry-run

  # Migrate from custom SQLite to local Postgres
  memory-palace-sqlite-to-pg \\
    --source "sqlite:///path/to/memories.db" \\
    --target "postgresql://localhost:5432/memory_palace"

  # Force migration even if target has data
  memory-palace-sqlite-to-pg --force

Prerequisites:
  1. PostgreSQL database must exist
  2. pgvector extension must be installed: CREATE EXTENSION vector;
  3. Target tables must exist (run MCP server once to create them)
"""
    )

    parser.add_argument(
        "--source",
        help="Source SQLite database URL (default: from config)",
        default=None
    )

    parser.add_argument(
        "--target",
        help="Target PostgreSQL database URL (default: from config)",
        default=None
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Proceed even if target database already has data"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be migrated without writing anything"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of rows to process per batch (default: 100)"
    )

    args = parser.parse_args()

    source_url = args.source
    if source_url is None:
        db_type = get_database_type()
        if db_type == "sqlite":
            source_url = get_database_url()
        else:
            data_dir = ensure_data_dir()
            source_url = f"sqlite:///{data_dir}/memories.db"

    target_url = args.target
    if target_url is None:
        db_type = get_database_type()
        if db_type == "postgres":
            target_url = get_database_url()
        else:
            target_url = "postgresql://localhost:5432/memory_palace"

    success = migrate(
        source_url=source_url,
        target_url=target_url,
        force=args.force,
        dry_run=args.dry_run,
        batch_size=args.batch_size
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
