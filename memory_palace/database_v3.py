"""
Database connection and session management for Memory Palace v3.

Supports PostgreSQL with pgvector (primary) and SQLite (legacy/migration).

v3 changes from v2:
- Imports from models_v3 instead of models_v2
- Added Postgres LISTEN/NOTIFY helper functions for pubsub
- Named engine registry: multiple databases can be managed by logical name
"""

from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool, QueuePool

from memory_palace.config_v2 import (
    get_database_url,
    get_database_type,
    is_postgres,
    is_sqlite,
    ensure_data_dir,
    get_default_database_name,
)
from memory_palace.models_v3 import Base

# Named engine registry (replaces old singleton)
_engines: Dict[str, Any] = {}
_session_factories: Dict[str, Any] = {}
_default_db: str = "default"


def _resolve_name(db_name: Optional[str] = None) -> str:
    """Resolve a database name to a concrete registry key."""
    return db_name if db_name is not None else _default_db


def _create_engine_for_db(db_name: Optional[str] = None):
    """Create a new SQLAlchemy engine for the given database name."""
    name = _resolve_name(db_name)
    db_url = get_database_url(name)

    if db_url.startswith("postgresql") or db_url.startswith("postgres"):
        engine = create_engine(
            db_url,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            echo=False,
        )

        @event.listens_for(engine, "connect")
        def create_pgvector_extension(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.close()
    else:
        ensure_data_dir()
        engine = create_engine(
            db_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False,
        )

        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def get_engine(db_name: Optional[str] = None):
    """Get the SQLAlchemy engine for db_name, creating it lazily if needed."""
    name = _resolve_name(db_name)
    if name not in _engines:
        # Auto-create database if it doesn't exist yet
        ensure_database_exists(name)
        _engines[name] = _create_engine_for_db(name)
    return _engines[name]


def get_session_factory(db_name: Optional[str] = None):
    """Get (or lazily create) the session factory for db_name."""
    name = _resolve_name(db_name)
    if name not in _session_factories:
        _session_factories[name] = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=get_engine(name),
        )
    return _session_factories[name]


def get_session(db_name: Optional[str] = None) -> Session:
    """
    Get a new database session for db_name.

    Usage:
        session = get_session()
        try:
            # do work
            session.commit()
        finally:
            session.close()

    Or use the context manager:
        with session_scope() as session:
            # do work (auto-commits on success, rolls back on exception)
    """
    return get_session_factory(db_name)()


@contextmanager
def session_scope(db_name: Optional[str] = None) -> Generator[Session, None, None]:
    """
    Provide a transactional scope around a series of operations.

    Usage:
        with session_scope() as session:
            session.add(memory)
            # auto-commits on exit, rolls back on exception
    """
    session = get_session(db_name)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def ensure_database_exists(db_name: Optional[str] = None) -> bool:
    """
    Ensure the PostgreSQL database exists, creating it if needed.

    Connects to the 'postgres' maintenance database to check/create.
    For SQLite, this is a no-op (file is created automatically).

    Args:
        db_name: Logical database name, or None for default.

    Returns:
        True if database was created, False if it already existed.
    """
    name = _resolve_name(db_name)

    if not is_postgres(name):
        return False  # SQLite auto-creates

    from urllib.parse import urlparse, urlunparse

    db_url = get_database_url(name)
    parsed = urlparse(db_url)
    target_db = parsed.path.lstrip("/")

    if not target_db:
        return False

    # Connect to 'postgres' maintenance database
    maintenance_parsed = parsed._replace(path="/postgres")
    maintenance_url = urlunparse(maintenance_parsed)

    maintenance_engine = create_engine(maintenance_url, isolation_level="AUTOCOMMIT")

    try:
        with maintenance_engine.connect() as conn:
            result = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :dbname"),
                {"dbname": target_db}
            )
            exists = result.scalar() is not None

            if not exists:
                # CREATE DATABASE must run outside transaction (AUTOCOMMIT handles this)
                conn.execute(text(f'CREATE DATABASE "{target_db}"'))
                return True
            return False
    finally:
        maintenance_engine.dispose()


def init_db(db_name: Optional[str] = None):
    """
    Initialize the database by creating all tables.

    For PostgreSQL, also ensures the pgvector extension is installed.
    Safe to call multiple times - only creates tables that don't exist.
    """
    name = _resolve_name(db_name)
    engine = get_engine(name)

    if is_postgres(name):
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()

    Base.metadata.create_all(bind=engine)

    if is_postgres(name):
        with engine.connect() as conn:
            try:
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_memories_embedding_hnsw
                    ON memories
                    USING hnsw (embedding vector_cosine_ops)
                """))
                conn.commit()
            except Exception as e:
                print(f"Note: Could not create HNSW index (will be created when embeddings exist): {e}")


def drop_db(db_name: Optional[str] = None):
    """Drop all tables. Use with caution! Primarily for testing."""
    Base.metadata.drop_all(bind=get_engine(db_name))


def reset_engine(db_name: Optional[str] = None):
    """
    Reset engine(s) and session factories.

    If db_name is None, resets ALL engines.
    If db_name is given, resets only that engine.
    """
    if db_name is None:
        for eng in _engines.values():
            eng.dispose()
        _engines.clear()
        _session_factories.clear()
    else:
        name = _resolve_name(db_name)
        if name in _engines:
            _engines[name].dispose()
            del _engines[name]
        _session_factories.pop(name, None)


def check_connection(db_name: Optional[str] = None) -> dict:
    """Check database connection and return status info."""
    name = _resolve_name(db_name)
    try:
        engine = get_engine(name)
        db_type = get_database_type(name)

        with engine.connect() as conn:
            if db_type == "postgres":
                result = conn.execute(text("SELECT version()"))
                version = result.scalar()
                result = conn.execute(text(
                    "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
                ))
                pgvector_version = result.scalar()
                return {
                    "status": "connected",
                    "type": "postgres",
                    "version": version,
                    "pgvector_version": pgvector_version,
                }
            else:
                result = conn.execute(text("SELECT sqlite_version()"))
                version = result.scalar()
                return {
                    "status": "connected",
                    "type": "sqlite",
                    "version": version,
                }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


# --- PostgreSQL LISTEN/NOTIFY helpers for pubsub (v3) ---

async def pg_listen(channel: str, callback=None, db_name: Optional[str] = None):
    """
    Set up LISTEN on a PostgreSQL channel for pubsub.
    Only works with PostgreSQL.
    """
    name = _resolve_name(db_name)
    if not is_postgres(name):
        raise RuntimeError("LISTEN/NOTIFY only supported on PostgreSQL")
    raise NotImplementedError(
        "Async LISTEN requires asyncpg or psycopg3. "
        "Use raw connection with threading for sync version."
    )


def pg_notify(channel: str, payload: str = "", db_name: Optional[str] = None):
    """
    Send a NOTIFY on a PostgreSQL channel for pubsub.
    Only works with PostgreSQL. For SQLite, this is a no-op.
    """
    name = _resolve_name(db_name)
    if not is_postgres(name):
        return
    engine = get_engine(name)
    with engine.connect() as conn:
        escaped_payload = payload.replace("'", "''")
        conn.execute(text(f'NOTIFY "{channel}", \'{escaped_payload}\''))
        conn.commit()


def is_postgres_db(db_name: Optional[str] = None) -> bool:
    """Expose whether db_name is running on PostgreSQL."""
    name = _resolve_name(db_name)
    return is_postgres(name)
