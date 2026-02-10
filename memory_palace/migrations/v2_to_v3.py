"""
Migration from Memory Palace v2 to v3 schema.

Changes:
1. memories table:
   - Add foundational column (Boolean, default=False)
   - Migrate importance >= 8 to foundational = True
   - Remove importance column and its constraints/indexes

2. handoff_messages table -> messages table:
   - Rename table
   - Add pubsub columns: channel, delivery_status, delivered_at, expires_at, priority
   - Add new indexes for pubsub queries

3. memory_edges table:
   - No changes

Works on both PostgreSQL and SQLite.
"""

import sys
from typing import Optional

from sqlalchemy import create_engine, text, inspect, Column, Integer, String, Boolean, DateTime
from sqlalchemy.engine import Engine

from memory_palace.config_v2 import get_database_url, get_database_type


def _is_postgres(engine: Engine) -> bool:
    """Check if engine is PostgreSQL."""
    return engine.dialect.name == "postgresql"


def _is_sqlite(engine: Engine) -> bool:
    """Check if engine is SQLite."""
    return engine.dialect.name == "sqlite"


def _column_exists(engine: Engine, table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table."""
    inspector = inspect(engine)
    columns = [col["name"] for col in inspector.get_columns(table_name)]
    return column_name in columns


def _table_exists(engine: Engine, table_name: str) -> bool:
    """Check if a table exists."""
    inspector = inspect(engine)
    return table_name in inspector.get_table_names()


def _index_exists(engine: Engine, table_name: str, index_name: str) -> bool:
    """Check if an index exists on a table."""
    inspector = inspect(engine)
    indexes = [idx["name"] for idx in inspector.get_indexes(table_name)]
    return index_name in indexes


def migrate_memories_table(engine: Engine) -> None:
    """
    Migrate memories table: remove importance, add foundational.

    Steps:
    1. Add foundational column if it doesn't exist
    2. Migrate data: foundational = True WHERE importance >= 8
    3. Drop importance column and its constraints/indexes
    """
    print("Migrating memories table...")

    with engine.connect() as conn:
        # Check if already migrated
        if _column_exists(engine, "memories", "foundational") and not _column_exists(engine, "memories", "importance"):
            print("  memories table already migrated to v3")
            return

        # Step 1: Add foundational column if it doesn't exist
        if not _column_exists(engine, "memories", "foundational"):
            print("  Adding foundational column...")
            if _is_postgres(engine):
                conn.execute(text(
                    "ALTER TABLE memories ADD COLUMN foundational BOOLEAN DEFAULT FALSE"
                ))
            else:  # SQLite
                conn.execute(text(
                    "ALTER TABLE memories ADD COLUMN foundational INTEGER DEFAULT 0"
                ))
            conn.commit()
            print("  Added foundational column")
        else:
            print("  foundational column already exists")

        # Step 2: Migrate data if importance column still exists
        if _column_exists(engine, "memories", "importance"):
            print("  Migrating importance values to foundational...")
            if _is_postgres(engine):
                conn.execute(text(
                    "UPDATE memories SET foundational = TRUE WHERE importance >= 8"
                ))
            else:  # SQLite
                conn.execute(text(
                    "UPDATE memories SET foundational = 1 WHERE importance >= 8"
                ))
            conn.commit()
            print("  Migrated importance values")

            # Step 3: Drop importance column
            print("  Dropping importance column and constraints...")
            if _is_postgres(engine):
                # Drop check constraint first
                try:
                    conn.execute(text(
                        "ALTER TABLE memories DROP CONSTRAINT IF EXISTS check_importance_range"
                    ))
                except Exception as e:
                    print(f"  Note: Could not drop constraint check_importance_range: {e}")

                # Drop indexes
                try:
                    conn.execute(text("DROP INDEX IF EXISTS idx_memories_importance_desc"))
                except Exception as e:
                    print(f"  Note: Could not drop index idx_memories_importance_desc: {e}")

                # Drop column
                conn.execute(text("ALTER TABLE memories DROP COLUMN importance"))
                conn.commit()
                print("  Dropped importance column and constraints")

            else:  # SQLite - requires table recreation
                print("  SQLite detected: recreating table without importance column...")

                # Create new table without importance
                conn.execute(text("""
                    CREATE TABLE memories_new (
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

                # Build SELECT dynamically based on what columns exist in old table.
                # v1 schemas may be missing columns that v2 added (e.g. project).
                old_columns = {col["name"] for col in inspect(engine).get_columns("memories")}

                # Target columns and their defaults for missing source columns
                target_cols = [
                    ("id", None),
                    ("created_at", None),
                    ("updated_at", None),
                    ("instance_id", None),
                    ("project", "'life'"),
                    ("memory_type", None),
                    ("subject", None),
                    ("content", None),
                    ("keywords", None),
                    ("tags", None),
                    ("foundational", "0"),
                    ("source_type", None),
                    ("source_context", None),
                    ("source_session_id", None),
                    ("embedding", None),
                    ("last_accessed_at", None),
                    ("access_count", "0"),
                    ("expires_at", None),
                    ("is_archived", "0"),
                ]

                dest_cols = []
                select_exprs = []
                for col_name, default_val in target_cols:
                    dest_cols.append(col_name)
                    if col_name in old_columns:
                        select_exprs.append(col_name)
                    elif default_val is not None:
                        select_exprs.append(f"{default_val} AS {col_name}")
                    else:
                        select_exprs.append(f"NULL AS {col_name}")

                insert_sql = (
                    f"INSERT INTO memories_new ({', '.join(dest_cols)})\n"
                    f"SELECT {', '.join(select_exprs)}\n"
                    f"FROM memories"
                )
                print(f"  Copying {len(old_columns)} existing columns, defaulting missing ones...")
                conn.execute(text(insert_sql))

                # Drop old table
                conn.execute(text("DROP TABLE memories"))

                # Rename new table
                conn.execute(text("ALTER TABLE memories_new RENAME TO memories"))

                conn.commit()
                print("  Recreated memories table without importance column")
        else:
            print("  importance column already removed")

        # Create foundational index if it doesn't exist
        if not _index_exists(engine, "memories", "idx_memories_foundational"):
            print("  Creating index on foundational column...")
            conn.execute(text("CREATE INDEX idx_memories_foundational ON memories(foundational)"))
            conn.commit()
            print("  Created foundational index")

        # Recreate other indexes for SQLite
        if _is_sqlite(engine):
            print("  Recreating SQLite indexes...")
            indexes_to_create = [
                ("idx_memories_id", "CREATE INDEX IF NOT EXISTS idx_memories_id ON memories(id)"),
                ("idx_memories_instance_id", "CREATE INDEX IF NOT EXISTS idx_memories_instance_id ON memories(instance_id)"),
                ("idx_memories_project", "CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project)"),
                ("idx_memories_memory_type", "CREATE INDEX IF NOT EXISTS idx_memories_memory_type ON memories(memory_type)"),
                ("idx_memories_subject", "CREATE INDEX IF NOT EXISTS idx_memories_subject ON memories(subject)"),
                ("idx_memories_instance_project", "CREATE INDEX IF NOT EXISTS idx_memories_instance_project ON memories(instance_id, project)"),
            ]
            for idx_name, idx_sql in indexes_to_create:
                try:
                    conn.execute(text(idx_sql))
                except Exception as e:
                    print(f"  Note: Could not create index {idx_name}: {e}")
            conn.commit()

    print("memories table migration complete")


def migrate_messages_table(engine: Engine) -> None:
    """
    Migrate handoff_messages table to messages table with pubsub columns.

    Steps:
    1. If messages table exists and handoff_messages doesn't, assume already migrated
    2. If handoff_messages exists, rename/copy to messages
    3. Add new columns for pubsub support
    4. Create new indexes
    """
    print("Migrating handoff_messages -> messages table...")

    with engine.connect() as conn:
        has_messages = _table_exists(engine, "messages")
        has_handoff = _table_exists(engine, "handoff_messages")

        # Check if already migrated
        if has_messages and not has_handoff:
            print("  messages table already exists, checking for pubsub columns...")
            # Still need to add pubsub columns if they don't exist
        elif not has_messages and not has_handoff:
            print("  No messages or handoff_messages table found, skipping")
            return
        elif has_handoff and has_messages:
            # Both tables exist — init_db() created messages from v3 models
            # while handoff_messages still has old data. Merge and drop.
            print("  Both handoff_messages and messages tables exist")
            print("  Merging old data into messages table...")

            # Get columns that exist in both tables for safe copy
            insp = inspect(engine)
            handoff_cols = {c["name"] for c in insp.get_columns("handoff_messages")}
            messages_cols = {c["name"] for c in insp.get_columns("messages")}
            shared_cols = handoff_cols & messages_cols - {"id"}  # Skip id to avoid conflicts

            if shared_cols:
                cols_str = ", ".join(sorted(shared_cols))
                # Copy rows from handoff_messages that aren't already in messages
                conn.execute(text(f"""
                    INSERT INTO messages ({cols_str})
                    SELECT {cols_str} FROM handoff_messages h
                    WHERE NOT EXISTS (
                        SELECT 1 FROM messages m WHERE m.id = h.id
                    )
                """))
                conn.commit()
                print(f"  Merged data ({len(shared_cols)} shared columns)")

            conn.execute(text("DROP TABLE handoff_messages"))
            conn.commit()
            print("  Dropped handoff_messages table")
        elif has_handoff:
            # Need to migrate from handoff_messages to messages
            if _is_postgres(engine):
                print("  Renaming handoff_messages to messages...")
                conn.execute(text("ALTER TABLE handoff_messages RENAME TO messages"))
                conn.commit()
                print("  Renamed table")
            else:  # SQLite
                print("  SQLite detected: copying handoff_messages to messages...")
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

                conn.execute(text("""
                    INSERT INTO messages
                    SELECT id, created_at, from_instance, to_instance, message_type,
                           subject, content, read_at, read_by
                    FROM handoff_messages
                """))

                conn.execute(text("DROP TABLE handoff_messages"))
                conn.commit()
                print("  Copied and dropped old table")

        # Add pubsub columns if they don't exist
        pubsub_columns = [
            ("channel", "VARCHAR(100)", None),
            ("delivery_status", "VARCHAR(20)", "'pending'"),
            ("delivered_at", "DATETIME", None),
            ("expires_at", "DATETIME", None),
            ("priority", "INTEGER", "0"),
        ]

        for col_name, col_type, default in pubsub_columns:
            if not _column_exists(engine, "messages", col_name):
                print(f"  Adding {col_name} column...")
                if _is_postgres(engine):
                    default_clause = f" DEFAULT {default}" if default else ""
                    conn.execute(text(
                        f"ALTER TABLE messages ADD COLUMN {col_name} {col_type}{default_clause}"
                    ))
                else:  # SQLite
                    # SQLite uses INTEGER for boolean-like values
                    if col_type == "BOOLEAN":
                        col_type = "INTEGER"
                    default_clause = f" DEFAULT {default}" if default else ""
                    conn.execute(text(
                        f"ALTER TABLE messages ADD COLUMN {col_name} {col_type}{default_clause}"
                    ))
                conn.commit()
                print(f"  Added {col_name} column")
            else:
                print(f"  {col_name} column already exists")

        # Create indexes for pubsub queries
        pubsub_indexes = [
            ("idx_messages_channel", "CREATE INDEX IF NOT EXISTS idx_messages_channel ON messages(channel)"),
            ("idx_messages_channel_status", "CREATE INDEX IF NOT EXISTS idx_messages_channel_status ON messages(channel, delivery_status)"),
            ("idx_messages_priority", "CREATE INDEX IF NOT EXISTS idx_messages_priority ON messages(priority)"),
            ("idx_messages_delivery_status", "CREATE INDEX IF NOT EXISTS idx_messages_delivery_status ON messages(delivery_status)"),
        ]

        for idx_name, idx_sql in pubsub_indexes:
            if not _index_exists(engine, "messages", idx_name):
                print(f"  Creating index {idx_name}...")
                try:
                    conn.execute(text(idx_sql))
                    conn.commit()
                    print(f"  Created {idx_name}")
                except Exception as e:
                    print(f"  Note: Could not create index {idx_name}: {e}")
            else:
                print(f"  Index {idx_name} already exists")

        # Recreate basic indexes if needed (for both SQLite and Postgres)
        basic_indexes = [
            ("idx_messages_from_instance", "CREATE INDEX IF NOT EXISTS idx_messages_from_instance ON messages(from_instance)"),
            ("idx_messages_to_instance", "CREATE INDEX IF NOT EXISTS idx_messages_to_instance ON messages(to_instance)"),
            ("idx_messages_message_type", "CREATE INDEX IF NOT EXISTS idx_messages_message_type ON messages(message_type)"),
        ]

        for idx_name, idx_sql in basic_indexes:
            if not _index_exists(engine, "messages", idx_name):
                try:
                    conn.execute(text(idx_sql))
                except Exception as e:
                    print(f"  Note: Could not create index {idx_name}: {e}")
        conn.commit()

    print("messages table migration complete")


def migrate(database_url: Optional[str] = None) -> bool:
    """
    Run the v2 to v3 migration.

    Args:
        database_url: Optional database URL. If None, uses config default.

    Returns:
        True if migration succeeded, False otherwise
    """
    print("=" * 60)
    print("Memory Palace v2 -> v3 Migration")
    print("=" * 60)

    try:
        # Get database connection
        if database_url is None:
            database_url = get_database_url()
            db_type = get_database_type()
            print(f"Using database from config: {db_type}")
        else:
            print(f"Using provided database URL")

        print(f"Connecting to database...")
        engine = create_engine(database_url)

        # Test connection
        with engine.connect() as conn:
            if _is_postgres(engine):
                result = conn.execute(text("SELECT version()"))
                version = result.scalar()
                print(f"Connected to PostgreSQL: {version[:50]}...")
            else:
                result = conn.execute(text("SELECT sqlite_version()"))
                version = result.scalar()
                print(f"Connected to SQLite: {version}")

        print()

        # Run migrations
        migrate_memories_table(engine)
        print()
        migrate_messages_table(engine)

        print()
        print("=" * 60)
        print("Migration completed successfully!")
        print("=" * 60)

        return True

    except Exception as e:
        print()
        print("=" * 60)
        print("Migration failed!")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """CLI entry point for migration."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Migrate Memory Palace database from v2 to v3 schema"
    )
    parser.add_argument(
        "--database-url",
        help="Database URL (default: from config)",
        default=None
    )

    args = parser.parse_args()

    success = migrate(database_url=args.database_url)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
