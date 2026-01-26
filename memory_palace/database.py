"""
Database connection and session management for Claude Memory Palace.

v2: Re-exports from database_v2 for PostgreSQL + pgvector support.
For legacy SQLite code, see database_v1.py.
"""

# Re-export everything from v2
from memory_palace.database_v2 import (
    Base,
    get_engine,
    get_session_factory,
    get_session,
    session_scope,
    init_db,
    drop_db,
    reset_engine,
    check_connection,
)

__all__ = [
    "Base",
    "get_engine",
    "get_session_factory",
    "get_session",
    "session_scope",
    "init_db",
    "drop_db",
    "reset_engine",
    "check_connection",
]
