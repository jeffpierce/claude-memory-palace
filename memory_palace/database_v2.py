"""
Database connection and session management for Memory Palace v2.

Supports PostgreSQL with pgvector (primary) and SQLite (legacy/migration).
"""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool, QueuePool

from memory_palace.config_v2 import (
    get_database_url, 
    get_database_type,
    is_postgres,
    is_sqlite,
    ensure_data_dir
)
from memory_palace.models_v2 import Base

# Engine singleton
_engine = None
_SessionLocal = None


def get_engine():
    """
    Get the SQLAlchemy engine, creating it if needed.

    Configures appropriately for PostgreSQL or SQLite.
    """
    global _engine
    
    if _engine is not None:
        return _engine

    db_url = get_database_url()
    db_type = get_database_type()

    if db_type == "postgres":
        # PostgreSQL configuration
        _engine = create_engine(
            db_url,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,  # Verify connections before use
            echo=False
        )
        
        # Ensure pgvector extension exists
        @event.listens_for(_engine, "connect")
        def create_pgvector_extension(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.close()
            
    else:
        # SQLite configuration (legacy)
        ensure_data_dir()
        
        _engine = create_engine(
            db_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False
        )
        
        # Enable foreign keys for SQLite
        @event.listens_for(_engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return _engine


def get_session_factory():
    """Get the session factory, creating it if needed."""
    global _SessionLocal
    
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=get_engine()
        )
    return _SessionLocal


def get_session() -> Session:
    """
    Get a new database session.

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
    return get_session_factory()()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """
    Provide a transactional scope around a series of operations.

    Usage:
        with session_scope() as session:
            session.add(memory)
            # auto-commits on exit, rolls back on exception
    """
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db():
    """
    Initialize the database by creating all tables.

    For PostgreSQL, also ensures the pgvector extension is installed.
    Safe to call multiple times - only creates tables that don't exist.
    """
    engine = get_engine()
    
    if is_postgres():
        # Create pgvector extension first
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    if is_postgres():
        # Create HNSW index for vector similarity search
        # This is idempotent - IF NOT EXISTS handles it
        with engine.connect() as conn:
            try:
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_memories_embedding_hnsw 
                    ON memories 
                    USING hnsw (embedding vector_cosine_ops)
                """))
                conn.commit()
            except Exception as e:
                # Index might fail if no embeddings yet - that's fine
                print(f"Note: Could not create HNSW index (will be created when embeddings exist): {e}")


def drop_db():
    """
    Drop all tables. Use with caution!

    Primarily for testing.
    """
    Base.metadata.drop_all(bind=get_engine())


def reset_engine():
    """
    Reset the engine and session factory.
    
    Useful for testing or when switching databases.
    """
    global _engine, _SessionLocal
    
    if _engine is not None:
        _engine.dispose()
        _engine = None
    _SessionLocal = None


def check_connection() -> dict:
    """
    Check database connection and return status info.
    
    Returns:
        Dict with connection status, database type, and version info
    """
    try:
        engine = get_engine()
        db_type = get_database_type()
        
        with engine.connect() as conn:
            if db_type == "postgres":
                result = conn.execute(text("SELECT version()"))
                version = result.scalar()
                
                # Check pgvector
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
