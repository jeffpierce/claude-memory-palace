#!/usr/bin/env python3
"""
Migration script: SQLite v1 → PostgreSQL v2

Migrates data from the legacy SQLite database to PostgreSQL with pgvector.

Transformations:
- keywords: JSON string → TEXT[]
- tags: (new column) → defaults to empty array
- embedding: JSON list → vector(4096)
- is_archived: INTEGER → BOOLEAN
- project: (new column) → defaults to "life", inferred where possible

Usage:
    python migrate_to_postgres.py --sqlite-path ~/.memory-palace/memories.db --postgres-url postgresql://localhost/memory_palace
    
    # Dry run (no writes to Postgres):
    python migrate_to_postgres.py --sqlite-path ~/.memory-palace/memories.db --postgres-url postgresql://localhost/memory_palace --dry-run
    
    # With project inference from content:
    python migrate_to_postgres.py --sqlite-path ~/.memory-palace/memories.db --postgres-url postgresql://localhost/memory_palace --infer-projects
"""

import argparse
import json
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import psycopg2
    from psycopg2.extras import execute_batch
except ImportError:
    print("Error: psycopg2 not installed. Run: pip install psycopg2-binary")
    sys.exit(1)

try:
    from pgvector.psycopg2 import register_vector
except ImportError:
    print("Error: pgvector not installed. Run: pip install pgvector")
    sys.exit(1)


# Project inference patterns
PROJECT_PATTERNS = [
    (r'\bmemory.?palace\b', 'memory-palace'),
    (r'\bwordleap\b', 'wordleap'),
    (r'\bclawdbot\b', 'clawdbot'),
    (r'\bmy.?startup.?life\b', 'my-startup-life'),
    (r'\befaas\b', 'efaas'),
]


def infer_project(content: str, keywords: List[str]) -> str:
    """
    Attempt to infer project from content and keywords.
    
    Returns project name or "life" as default.
    """
    text = (content or "").lower()
    kw_text = " ".join(keywords or []).lower()
    combined = f"{text} {kw_text}"
    
    for pattern, project in PROJECT_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            return project
    
    return "life"


def parse_json_safe(value: str) -> Any:
    """Safely parse JSON, returning None on failure."""
    if not value:
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


def transform_memory(row: Dict[str, Any], infer_projects: bool = False) -> Dict[str, Any]:
    """
    Transform a v1 memory row to v2 format.
    
    Args:
        row: Dictionary from SQLite row
        infer_projects: If True, attempt to infer project from content
        
    Returns:
        Transformed dictionary ready for Postgres insert
    """
    # Parse JSON fields
    keywords = parse_json_safe(row.get("keywords")) or []
    embedding = parse_json_safe(row.get("embedding"))
    
    # Ensure keywords is a list of strings
    if not isinstance(keywords, list):
        keywords = []
    keywords = [str(k) for k in keywords if k]
    
    # Determine project
    if infer_projects:
        project = infer_project(row.get("content", ""), keywords)
    else:
        project = "life"
    
    return {
        "id": row["id"],
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "instance_id": row.get("instance_id", "unknown"),
        "project": project,
        "memory_type": row.get("memory_type", "fact"),
        "subject": row.get("subject"),
        "content": row.get("content", ""),
        "keywords": keywords,
        "tags": [],  # New column, default empty
        "importance": row.get("importance", 5),
        "source_type": row.get("source_type"),
        "source_context": row.get("source_context"),
        "source_session_id": row.get("source_session_id"),
        "embedding": embedding,
        "last_accessed_at": row.get("last_accessed_at"),
        "access_count": row.get("access_count", 0),
        "expires_at": row.get("expires_at"),
        "is_archived": bool(row.get("is_archived", 0)),
    }


def transform_handoff(row: Dict[str, Any]) -> Dict[str, Any]:
    """Transform a v1 handoff message row to v2 format."""
    return {
        "id": row["id"],
        "created_at": row.get("created_at"),
        "from_instance": row.get("from_instance", "unknown"),
        "to_instance": row.get("to_instance", "unknown"),
        "message_type": row.get("message_type", "fyi"),
        "subject": row.get("subject"),
        "content": row.get("content", ""),
        "read_at": row.get("read_at"),
        "read_by": row.get("read_by"),
    }


def connect_sqlite(path: str) -> sqlite3.Connection:
    """Connect to SQLite database."""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def connect_postgres(url: str) -> psycopg2.extensions.connection:
    """Connect to PostgreSQL database and register vector type."""
    conn = psycopg2.connect(url)
    register_vector(conn)
    return conn


def setup_postgres_schema(pg_conn: psycopg2.extensions.connection) -> None:
    """Create the v2 schema in PostgreSQL."""
    with pg_conn.cursor() as cur:
        # Enable pgvector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        
        # Create memories table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMP DEFAULT now(),
                updated_at TIMESTAMP,
                instance_id TEXT NOT NULL,
                project TEXT NOT NULL DEFAULT 'life',
                memory_type TEXT NOT NULL,
                subject TEXT,
                content TEXT NOT NULL,
                keywords TEXT[],
                tags TEXT[],
                importance INTEGER DEFAULT 5 CHECK (importance >= 1 AND importance <= 10),
                source_type TEXT,
                source_context TEXT,
                source_session_id TEXT,
                embedding vector(768),  -- nomic-embed-text dimensions
                last_accessed_at TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                expires_at TIMESTAMP,
                is_archived BOOLEAN DEFAULT false
            )
        """)
        
        # Create memory_edges table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS memory_edges (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMP DEFAULT now(),
                source_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                target_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                relation_type TEXT NOT NULL,
                strength FLOAT DEFAULT 1.0 CHECK (strength >= 0 AND strength <= 1),
                bidirectional BOOLEAN DEFAULT false,
                edge_metadata JSONB DEFAULT '{}',
                created_by TEXT,
                CONSTRAINT no_self_loops CHECK (source_id != target_id),
                UNIQUE(source_id, target_id, relation_type)
            )
        """)
        
        # Create handoff_messages table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS handoff_messages (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMP DEFAULT now(),
                from_instance TEXT NOT NULL,
                to_instance TEXT NOT NULL,
                message_type TEXT NOT NULL,
                subject TEXT,
                content TEXT NOT NULL,
                read_at TIMESTAMP,
                read_by TEXT
            )
        """)
        
        # Create indexes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_memories_instance ON memories(instance_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_memories_instance_project ON memories(instance_id, project)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_memories_keywords ON memories USING gin(keywords)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories USING gin(tags)")
        
        cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON memory_edges(source_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON memory_edges(target_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_relation_type ON memory_edges(relation_type)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_source_rel ON memory_edges(source_id, relation_type)")
        
        cur.execute("CREATE INDEX IF NOT EXISTS idx_handoff_to ON handoff_messages(to_instance)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_handoff_from ON handoff_messages(from_instance)")
        
        pg_conn.commit()
        print("✓ PostgreSQL schema created")


def create_hnsw_index(pg_conn: psycopg2.extensions.connection) -> None:
    """Create HNSW index for vector similarity search."""
    with pg_conn.cursor() as cur:
        try:
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_embedding_hnsw 
                ON memories 
                USING hnsw (embedding vector_cosine_ops)
            """)
            pg_conn.commit()
            print("✓ HNSW index created")
        except Exception as e:
            print(f"⚠ Could not create HNSW index: {e}")
            print("  (Index will be created automatically when embeddings are added)")
            pg_conn.rollback()


def read_sqlite_data(sqlite_conn: sqlite3.Connection) -> Tuple[List[Dict], List[Dict]]:
    """Read all data from SQLite database."""
    memories = []
    handoffs = []
    
    cursor = sqlite_conn.cursor()
    
    # Read memories
    cursor.execute("SELECT * FROM memories")
    columns = [desc[0] for desc in cursor.description]
    for row in cursor.fetchall():
        memories.append(dict(zip(columns, row)))
    
    # Read handoffs
    try:
        cursor.execute("SELECT * FROM handoff_messages")
        columns = [desc[0] for desc in cursor.description]
        for row in cursor.fetchall():
            handoffs.append(dict(zip(columns, row)))
    except sqlite3.OperationalError:
        # Table might not exist in older versions
        pass
    
    return memories, handoffs


def insert_memories(
    pg_conn: psycopg2.extensions.connection,
    memories: List[Dict[str, Any]],
    dry_run: bool = False
) -> int:
    """Insert transformed memories into PostgreSQL."""
    if not memories:
        return 0
    
    insert_sql = """
        INSERT INTO memories (
            id, created_at, updated_at, instance_id, project, memory_type,
            subject, content, keywords, tags, importance, source_type,
            source_context, source_session_id, embedding, last_accessed_at,
            access_count, expires_at, is_archived
        ) VALUES (
            %(id)s, %(created_at)s, %(updated_at)s, %(instance_id)s, %(project)s,
            %(memory_type)s, %(subject)s, %(content)s, %(keywords)s, %(tags)s,
            %(importance)s, %(source_type)s, %(source_context)s, %(source_session_id)s,
            %(embedding)s, %(last_accessed_at)s, %(access_count)s, %(expires_at)s,
            %(is_archived)s
        )
        ON CONFLICT (id) DO NOTHING
    """
    
    if dry_run:
        print(f"  [DRY RUN] Would insert {len(memories)} memories")
        return len(memories)
    
    with pg_conn.cursor() as cur:
        execute_batch(cur, insert_sql, memories, page_size=100)
        
        # Reset sequence to max id
        cur.execute("SELECT setval('memories_id_seq', (SELECT MAX(id) FROM memories))")
        
    pg_conn.commit()
    return len(memories)


def insert_handoffs(
    pg_conn: psycopg2.extensions.connection,
    handoffs: List[Dict[str, Any]],
    dry_run: bool = False
) -> int:
    """Insert transformed handoff messages into PostgreSQL."""
    if not handoffs:
        return 0
    
    insert_sql = """
        INSERT INTO handoff_messages (
            id, created_at, from_instance, to_instance, message_type,
            subject, content, read_at, read_by
        ) VALUES (
            %(id)s, %(created_at)s, %(from_instance)s, %(to_instance)s,
            %(message_type)s, %(subject)s, %(content)s, %(read_at)s, %(read_by)s
        )
        ON CONFLICT (id) DO NOTHING
    """
    
    if dry_run:
        print(f"  [DRY RUN] Would insert {len(handoffs)} handoff messages")
        return len(handoffs)
    
    with pg_conn.cursor() as cur:
        execute_batch(cur, insert_sql, handoffs, page_size=100)
        
        # Reset sequence to max id
        cur.execute("SELECT setval('handoff_messages_id_seq', (SELECT MAX(id) FROM handoff_messages))")
        
    pg_conn.commit()
    return len(handoffs)


def migrate(
    sqlite_path: str,
    postgres_url: str,
    dry_run: bool = False,
    infer_projects: bool = False
) -> Dict[str, Any]:
    """
    Run the full migration.
    
    Args:
        sqlite_path: Path to SQLite database
        postgres_url: PostgreSQL connection URL
        dry_run: If True, don't write to Postgres
        infer_projects: If True, attempt to infer project from content
        
    Returns:
        Dict with migration statistics
    """
    stats = {
        "memories_read": 0,
        "memories_written": 0,
        "handoffs_read": 0,
        "handoffs_written": 0,
        "projects_inferred": {},
        "errors": [],
    }
    
    # Connect to databases
    print(f"Connecting to SQLite: {sqlite_path}")
    sqlite_conn = connect_sqlite(sqlite_path)
    
    if not dry_run:
        print(f"Connecting to PostgreSQL: {postgres_url}")
        pg_conn = connect_postgres(postgres_url)
        
        print("Setting up PostgreSQL schema...")
        setup_postgres_schema(pg_conn)
    else:
        pg_conn = None
        print("[DRY RUN] Skipping PostgreSQL connection")
    
    # Read source data
    print("Reading data from SQLite...")
    memories_raw, handoffs_raw = read_sqlite_data(sqlite_conn)
    stats["memories_read"] = len(memories_raw)
    stats["handoffs_read"] = len(handoffs_raw)
    print(f"  Found {len(memories_raw)} memories, {len(handoffs_raw)} handoff messages")
    
    # Transform data
    print("Transforming memories...")
    memories = []
    for raw in memories_raw:
        try:
            transformed = transform_memory(raw, infer_projects)
            memories.append(transformed)
            
            # Track project distribution
            project = transformed["project"]
            stats["projects_inferred"][project] = stats["projects_inferred"].get(project, 0) + 1
        except Exception as e:
            stats["errors"].append(f"Memory {raw.get('id')}: {e}")
    
    print("Transforming handoff messages...")
    handoffs = []
    for raw in handoffs_raw:
        try:
            handoffs.append(transform_handoff(raw))
        except Exception as e:
            stats["errors"].append(f"Handoff {raw.get('id')}: {e}")
    
    # Write to Postgres
    if pg_conn:
        print("Writing memories to PostgreSQL...")
        stats["memories_written"] = insert_memories(pg_conn, memories, dry_run)
        
        print("Writing handoff messages to PostgreSQL...")
        stats["handoffs_written"] = insert_handoffs(pg_conn, handoffs, dry_run)
        
        print("Creating HNSW index...")
        create_hnsw_index(pg_conn)
        
        pg_conn.close()
    else:
        stats["memories_written"] = len(memories)
        stats["handoffs_written"] = len(handoffs)
    
    sqlite_conn.close()
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Migrate Memory Palace from SQLite v1 to PostgreSQL v2"
    )
    parser.add_argument(
        "--sqlite-path",
        default=str(Path.home() / ".memory-palace" / "memories.db"),
        help="Path to SQLite database (default: ~/.memory-palace/memories.db)"
    )
    parser.add_argument(
        "--postgres-url",
        default="postgresql://localhost/memory_palace",
        help="PostgreSQL connection URL (default: postgresql://localhost/memory_palace)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually write to PostgreSQL, just show what would happen"
    )
    parser.add_argument(
        "--infer-projects",
        action="store_true",
        help="Attempt to infer project from memory content/keywords"
    )
    
    args = parser.parse_args()
    
    # Validate SQLite path
    if not Path(args.sqlite_path).exists():
        print(f"Error: SQLite database not found: {args.sqlite_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("Memory Palace Migration: SQLite v1 → PostgreSQL v2")
    print("=" * 60)
    print()
    
    stats = migrate(
        args.sqlite_path,
        args.postgres_url,
        dry_run=args.dry_run,
        infer_projects=args.infer_projects
    )
    
    # Print summary
    print()
    print("=" * 60)
    print("Migration Summary")
    print("=" * 60)
    print(f"Memories: {stats['memories_read']} read → {stats['memories_written']} written")
    print(f"Handoffs: {stats['handoffs_read']} read → {stats['handoffs_written']} written")
    
    if stats["projects_inferred"]:
        print()
        print("Project distribution:")
        for project, count in sorted(stats["projects_inferred"].items(), key=lambda x: -x[1]):
            print(f"  {project}: {count}")
    
    if stats["errors"]:
        print()
        print(f"Errors ({len(stats['errors'])}):")
        for error in stats["errors"][:10]:
            print(f"  - {error}")
        if len(stats["errors"]) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more")
    
    print()
    if args.dry_run:
        print("✓ Dry run complete. No data was written to PostgreSQL.")
    else:
        print("✓ Migration complete!")


if __name__ == "__main__":
    main()
