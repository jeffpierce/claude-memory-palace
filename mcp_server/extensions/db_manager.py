"""
Database manager extension for Memory Palace MCP server.

Provides admin tools for listing, registering, and switching between
named databases. Replaces switch_db.py with config-aware management.
"""
from typing import Any, Optional
from urllib.parse import urlparse

from sqlalchemy import text

from memory_palace.config_v2 import (
    load_config,
    get_configured_databases,
    get_default_database_name,
    set_default_database_name,
    get_database_url,
)
from memory_palace.database import (
    get_engine,
    reset_engine,
    init_db,
    check_connection,
    session_scope,
)
from mcp_server.toon_wrapper import toon_response


def _mask_password(url: str) -> str:
    """Mask password in database URL for safe display."""
    parsed = urlparse(url)
    if parsed.password:
        return url.replace(parsed.password, "***")
    return url


def _get_table_counts(db_name: str) -> dict:
    """Get table counts for a database."""
    try:
        with session_scope(db_name) as session:
            memories = session.execute(text("SELECT COUNT(*) FROM memories")).scalar()
            edges = session.execute(text("SELECT COUNT(*) FROM memory_edges")).scalar()
            messages = session.execute(text("SELECT COUNT(*) FROM messages")).scalar()
            return {"memories": memories, "edges": edges, "messages": messages}
    except Exception as e:
        return {"error": f"Could not query tables: {e}"}


def register(mcp):
    """Register database management tools with the MCP server."""

    @mcp.tool()
    @toon_response
    async def memory_list_databases() -> dict[str, Any]:
        """
        List all configured databases with connection status.

        Returns configured database names, URLs (masked), and basic stats.
        """
        databases = get_configured_databases()
        default_name = get_default_database_name()
        result = {"databases": {}, "default": default_name}

        for name, db_config in databases.items():
            try:
                url = get_database_url(name)
                conn = check_connection(name)
                entry = {
                    "url": _mask_password(url),
                    "type": db_config.get("type", "postgres"),
                    "status": conn.get("status", "unknown"),
                    "is_default": name == default_name,
                }
                if conn.get("status") == "connected":
                    entry["table_counts"] = _get_table_counts(name)
                result["databases"][name] = entry
            except Exception as e:
                result["databases"][name] = {"error": str(e)}

        return result

    @mcp.tool()
    @toon_response
    async def memory_register_database(name: str, url: Optional[str] = None) -> dict[str, Any]:
        """
        Register a database at runtime (not persisted to config file).

        Creates engine, initializes tables, verifies connection.

        Args:
            name: Logical name for the database (e.g., "life", "work")
            url: PostgreSQL or SQLite connection URL. If not provided, auto-derives from default database.
        """
        if url is None:
            # Auto-derive URL from default database
            try:
                url = get_database_url(name)  # This will auto-derive
            except KeyError as e:
                return {"error": str(e)}

        config = load_config()
        if "databases" not in config:
            config["databases"] = {}
        config["databases"][name] = {
            "type": "postgres" if "postgres" in url else "sqlite",
            "url": url,
        }

        try:
            init_db(name)
        except Exception as e:
            return {"error": f"Failed to initialize database '{name}'", "details": str(e)}

        conn = check_connection(name)
        if conn.get("status") != "connected":
            return {"error": f"Failed to connect to '{name}'", "connection": conn}

        return {
            "registered": name,
            "url": _mask_password(url),
            "connection": conn,
            "table_counts": _get_table_counts(name),
            "note": "Runtime only - not persisted to config file",
        }

    @mcp.tool()
    @toon_response
    async def memory_set_default_database(name: str) -> dict[str, Any]:
        """
        Change the default database target.

        Subsequent tool calls without explicit database= parameter will use this database.
        Runtime only - not persisted.

        Args:
            name: Logical database name to set as default
        """
        databases = get_configured_databases()
        if name not in databases:
            return {"error": f"Database '{name}' not found", "available": list(databases.keys())}

        old_default = get_default_database_name()
        set_default_database_name(name)

        return {
            "default_changed": {"from": old_default, "to": name},
            "note": "Runtime only - not persisted to config file",
        }

    @mcp.tool()
    @toon_response
    async def memory_current_database() -> dict[str, Any]:
        """
        Show current default database and connection info.

        Returns database name, masked URL, connection status, and table counts.
        """
        name = get_default_database_name()
        try:
            url = get_database_url(name)
            conn = check_connection(name)
            result = {
                "database": name,
                "url": _mask_password(url),
                "connection": conn,
            }
            if conn.get("status") == "connected":
                result["table_counts"] = _get_table_counts(name)
            return result
        except Exception as e:
            return {"database": name, "error": str(e)}
