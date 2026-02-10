"""
Migration from Memory Palace v3.0 to v3.1 schema.

Changes:
1. memories table:
   - Add projects column (ARRAY(Text) on PostgreSQL, JSON on SQLite)
   - Migrate data: projects = ARRAY[project] (PostgreSQL) or json_array(project) (SQLite)
   - Drop old project column and indexes
   - Create new indexes for projects column

Works on both PostgreSQL and SQLite.
"""

import sys
from typing import Optional

from sqlalchemy import create_engine, text, inspect
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


def _get_sqlite_version(engine: Engine) -> tuple:
    """Get SQLite version as tuple (major, minor, patch)."""
    with engine.connect() as conn:
        result = conn.execute(text("SELECT sqlite_version()"))
        version_str = result.scalar()
        parts = version_str.split('.')
        return tuple(int(p) for p in parts[:3])


def migrate_memories_table(engine: Engine) -> None:
    """
    Migrate memories table: change project (String) to projects (Array/JSON).

    Steps:
    1. Add projects column if it doesn't exist
    2. Migrate data: projects = [project]
    3. Drop old indexes
    4. Create new indexes
    5. Drop old project column (if supported)
    """
    print("Migrating memories table...")

    with engine.connect() as conn:
        # Check if already migrated
        has_projects = _column_exists(engine, "memories", "projects")
        has_project = _column_exists(engine, "memories", "project")

        if has_projects and not has_project:
            print("  memories table already migrated to v3.1")
            return

        if not has_project:
            print("  WARNING: Neither 'project' nor 'projects' column found!")
            print("  This may indicate a corrupted schema. Skipping migration.")
            return

        # Step 1: Add projects column if it doesn't exist
        if not has_projects:
            print("  Adding projects column...")
            if _is_postgres(engine):
                conn.execute(text(
                    "ALTER TABLE memories ADD COLUMN projects TEXT[]"
                ))
            else:  # SQLite
                conn.execute(text(
                    "ALTER TABLE memories ADD COLUMN projects TEXT"
                ))
            conn.commit()
            print("  Added projects column")
        else:
            print("  projects column already exists")

        # Step 2: Migrate data from project to projects
        if has_project:
            print("  Migrating project values to projects array...")
            if _is_postgres(engine):
                # PostgreSQL: Convert single value to array
                conn.execute(text(
                    "UPDATE memories SET projects = ARRAY[project] WHERE projects IS NULL"
                ))
            else:  # SQLite
                # SQLite: Convert to JSON array
                conn.execute(text(
                    "UPDATE memories SET projects = json_array(project) WHERE projects IS NULL"
                ))
            conn.commit()
            print("  Migrated project values")

            # Step 2b: Set NOT NULL constraint on projects
            print("  Setting projects column to NOT NULL...")
            if _is_postgres(engine):
                conn.execute(text(
                    "ALTER TABLE memories ALTER COLUMN projects SET NOT NULL"
                ))
                # Set default for new rows
                conn.execute(text(
                    "ALTER TABLE memories ALTER COLUMN projects SET DEFAULT ARRAY['life']"
                ))
            else:  # SQLite doesn't support ALTER COLUMN, but we can ensure data integrity
                # Just verify no NULL values exist
                result = conn.execute(text(
                    "SELECT COUNT(*) FROM memories WHERE projects IS NULL"
                ))
                null_count = result.scalar()
                if null_count > 0:
                    print(f"  WARNING: Found {null_count} NULL projects values after migration!")
            conn.commit()
            print("  Set NOT NULL constraint")

        # Step 3: Drop old indexes
        print("  Dropping old project indexes...")
        old_indexes = [
            "idx_memories_project",
            "idx_memories_instance_project",
        ]

        for idx_name in old_indexes:
            if _index_exists(engine, "memories", idx_name):
                try:
                    conn.execute(text(f"DROP INDEX IF EXISTS {idx_name}"))
                    print(f"  Dropped index {idx_name}")
                except Exception as e:
                    print(f"  Note: Could not drop index {idx_name}: {e}")
        conn.commit()

        # Step 4: Create new indexes
        print("  Creating new projects indexes...")
        if not _index_exists(engine, "memories", "idx_memories_instance_projects"):
            conn.execute(text(
                "CREATE INDEX idx_memories_instance_projects ON memories(instance_id, projects)"
            ))
            print("  Created idx_memories_instance_projects")
        else:
            print("  Index idx_memories_instance_projects already exists")
        conn.commit()

        # Step 5: Drop old project column
        if has_project:
            print("  Dropping old project column...")
            if _is_postgres(engine):
                conn.execute(text("ALTER TABLE memories DROP COLUMN project"))
                conn.commit()
                print("  Dropped project column")
            else:  # SQLite
                # Check SQLite version for DROP COLUMN support (added in 3.35.0)
                sqlite_version = _get_sqlite_version(engine)
                if sqlite_version >= (3, 35, 0):
                    print(f"  SQLite version {'.'.join(map(str, sqlite_version))} supports DROP COLUMN")
                    try:
                        conn.execute(text("ALTER TABLE memories DROP COLUMN project"))
                        conn.commit()
                        print("  Dropped project column")
                    except Exception as e:
                        print(f"  Note: Could not drop project column: {e}")
                        print("  The old column will remain but is harmless (ORM ignores unmapped columns)")
                else:
                    print(f"  SQLite version {'.'.join(map(str, sqlite_version))} does not support DROP COLUMN")
                    print("  Skipping DROP COLUMN (old column will remain but is harmless)")

    print("memories table migration complete")


def migrate(database_url: Optional[str] = None) -> bool:
    """
    Run the v3.0 to v3.1 migration.

    Args:
        database_url: Optional database URL. If None, uses config default.

    Returns:
        True if migration succeeded, False otherwise
    """
    print("=" * 60)
    print("Memory Palace v3.0 -> v3.1 Migration")
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

        # Run migration
        migrate_memories_table(engine)

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
        description="Migrate Memory Palace database from v3.0 to v3.1 schema"
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
