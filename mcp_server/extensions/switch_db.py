"""
Database switching extension for Memory Palace MCP server.

Provides runtime database switching without persisting to config.
Useful for multi-tenancy, testing, or switching between development databases.
"""
from typing import Any
from urllib.parse import urlparse, urlunparse

from sqlalchemy import text

from memory_palace.config_v2 import load_config, clear_config_cache
from memory_palace.database import reset_engine, init_db, check_connection, session_scope
from mcp_server.toon_wrapper import toon_response


def register(mcp):
    """Register database switching tools with the MCP server."""

    @mcp.tool()
    @toon_response
    async def memory_switch_db(database_name: str) -> dict[str, Any]:
        """
        Switch to a different PostgreSQL database at runtime.

        This is a runtime switch only - does NOT persist to config.
        Useful for multi-tenancy or switching between dev databases.

        Args:
            database_name: PostgreSQL database name (e.g., "memory_palace2")

        Returns:
            Dict with switch confirmation and connection info
        """
        # Get current config to extract connection details
        config = load_config()
        current_url = config.get("database", {}).get("url")

        if not current_url:
            return {
                "error": "No database URL configured",
                "help": "Set database.url in ~/.memory-palace/config.json"
            }

        # Parse the current URL
        parsed = urlparse(current_url)

        # Construct new URL with same credentials/host/port but new database
        new_parsed = parsed._replace(path=f"/{database_name}")
        new_url = urlunparse(new_parsed)

        # Clear cache and reload from disk
        clear_config_cache()
        config = load_config()

        # Modify the in-memory config cache directly
        # This is safe because we just cleared and reloaded
        import memory_palace.config_v2 as config_module
        config_module._config_cache["database"]["url"] = new_url

        # Reset engine to pick up new URL
        reset_engine()

        # Initialize tables on new database
        try:
            init_db()
        except Exception as e:
            return {
                "error": f"Failed to initialize database '{database_name}'",
                "details": str(e),
                "help": "Ensure database exists and pgvector is available"
            }

        # Verify connection
        conn_status = check_connection()
        if conn_status.get("status") != "connected":
            return {
                "error": f"Failed to connect to database '{database_name}'",
                "connection": conn_status
            }

        # Get table counts from new database
        table_counts = {}
        try:
            with session_scope() as session:
                # Count memories
                result = session.execute(text("SELECT COUNT(*) FROM memories"))
                table_counts["memories"] = result.scalar()

                # Count edges
                result = session.execute(text("SELECT COUNT(*) FROM memory_edges"))
                table_counts["edges"] = result.scalar()

                # Count messages
                result = session.execute(text("SELECT COUNT(*) FROM messages"))
                table_counts["messages"] = result.scalar()

        except Exception as e:
            table_counts = {"error": f"Could not query tables: {e}"}

        # Mask password in URL for safe display
        masked_url = new_url
        if parsed.password:
            masked_url = new_url.replace(parsed.password, "***")

        return {
            "switched_to": database_name,
            "url": masked_url,
            "connection": conn_status,
            "table_counts": table_counts,
            "note": "Runtime switch only - not persisted to config"
        }

    @mcp.tool()
    @toon_response
    async def memory_current_db() -> dict[str, Any]:
        """
        Get current database name and connection info.

        Returns:
            Dict with database name and connection status
        """
        config = load_config()
        current_url = config.get("database", {}).get("url")

        if not current_url:
            return {
                "database": None,
                "error": "No database URL configured"
            }

        # Parse URL to extract database name
        parsed = urlparse(current_url)
        db_name = parsed.path.lstrip("/") if parsed.path else "unknown"

        # Mask password for safe display
        masked_url = current_url
        if parsed.password:
            masked_url = current_url.replace(parsed.password, "***")

        # Get connection status
        conn_status = check_connection()

        return {
            "database": db_name,
            "url": masked_url,
            "connection": conn_status
        }
