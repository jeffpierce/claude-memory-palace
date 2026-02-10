"""
Database connection and session management for Memory Palace.

Database layer for Memory Palace.
"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import StaticPool

from memory_palace.config import DATABASE_URL, ensure_data_dir

# Base class for models
Base = declarative_base()

# Engine singleton - created lazily
_engine = None


def get_engine():
    """
    Get the SQLAlchemy engine, creating it if needed.

    Ensures the data directory exists before creating the engine.
    """
    global _engine
    if _engine is None:
        # Ensure ~/.memory-palace/ exists before creating DB
        ensure_data_dir()

        _engine = create_engine(
            DATABASE_URL,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False  # Set to True for SQL debugging
        )

        # Enable foreign keys for SQLite
        @event.listens_for(_engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return _engine


# Session factory - created lazily
_SessionLocal = None


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


def get_session():
    """
    Get a new database session.

    Usage:
        session = get_session()
        try:
            # do work
            session.commit()
        finally:
            session.close()

    Or as context manager:
        with get_session() as session:
            # do work
            session.commit()
    """
    return get_session_factory()()


def init_db():
    """
    Initialize the database by creating all tables.

    Safe to call multiple times - only creates tables that don't exist.
    """
    # Import models to ensure they're registered with Base
    from memory_palace.models import Memory, HandoffMessage  # noqa

    Base.metadata.create_all(bind=get_engine())


def drop_db():
    """
    Drop all tables. Use with caution!

    Primarily for testing.
    """
    Base.metadata.drop_all(bind=get_engine())
