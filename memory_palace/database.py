"""
Database connection and session management for Memory Palace.

v3: Re-exports from database_v3 with pubsub LISTEN/NOTIFY support.
For v2 code, see database_v2.py. For legacy SQLite code, see database_v1.py.
"""

# Re-export everything from v3
from memory_palace.database_v3 import (
    Base,
    get_engine,
    get_session_factory,
    get_session,
    session_scope,
    init_db,
    drop_db,
    reset_engine,
    check_connection,
    pg_listen,
    pg_notify,
    is_postgres_db,
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
    "pg_listen",
    "pg_notify",
    "is_postgres_db",
]
