"""
MCP Server for Memory Palace v2.0.

Provides tools for memory storage, retrieval, and inter-instance messaging.

Run with: python -m mcp_server.server
Or: python mcp_server/server.py
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mcp.server import FastMCP

from memory_palace.database import ensure_database_exists, init_db, get_engine
from mcp_server.tools import register_all_tools
from mcp_server.extensions import load_extensions

# Initialize the MCP server using FastMCP (has .tool() decorator)
server = FastMCP("memory-palace-v2")

# Alias for backwards compatibility
mcp = server

# Register all tools
register_all_tools(server)

# Load extensions (if configured)
load_extensions(server)


def _check_schema_version():
    """
    Fail-fast schema check at startup.

    Detects if the database is still on v1 schema (has 'importance' column
    or 'handoff_messages' table) and tells the user to run migration.
    """
    from sqlalchemy import inspect
    engine = get_engine()
    inspector = inspect(engine)

    tables = inspector.get_table_names()

    # Check 1: memories table should have 'foundational', not 'importance'
    if "memories" in tables:
        columns = [c["name"] for c in inspector.get_columns("memories")]
        if "importance" in columns and "foundational" not in columns:
            print("=" * 60, file=sys.stderr)
            print("SCHEMA MIGRATION REQUIRED", file=sys.stderr)
            print("=" * 60, file=sys.stderr)
            print("", file=sys.stderr)
            print("Your database is on v1 schema (has 'importance' column).", file=sys.stderr)
            print("Memory Palace v2.0 requires the v3 schema.", file=sys.stderr)
            print("", file=sys.stderr)
            print("Run:  memory-palace-migrate", file=sys.stderr)
            print("  or: python -m memory_palace.migrations.v2_to_v3", file=sys.stderr)
            print("", file=sys.stderr)
            print("This is safe and non-destructive. It will:", file=sys.stderr)
            print("  - Convert importance → foundational flag", file=sys.stderr)
            print("  - Rename handoff_messages → messages (with pubsub columns)", file=sys.stderr)
            print("=" * 60, file=sys.stderr)
            sys.exit(1)

    # Check 2: handoff_messages should have been renamed to messages
    if "handoff_messages" in tables and "messages" not in tables:
        print("=" * 60, file=sys.stderr)
        print("SCHEMA MIGRATION REQUIRED", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print("", file=sys.stderr)
        print("Your database has 'handoff_messages' table (v1 schema).", file=sys.stderr)
        print("Memory Palace v2.0 uses the 'messages' table.", file=sys.stderr)
        print("", file=sys.stderr)
        print("Run:  memory-palace-migrate", file=sys.stderr)
        print("  or: python -m memory_palace.migrations.v2_to_v3", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        sys.exit(1)


async def main_async():
    """Run the MCP server (async)."""
    # Check schema version on default database before accepting connections
    # Named databases are initialized lazily on first access via get_engine()
    _check_schema_version()

    # Run server with stdio transport (FastMCP has run_stdio_async)
    await server.run_stdio_async()


def main():
    """Entry point for script installation."""
    import asyncio
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
