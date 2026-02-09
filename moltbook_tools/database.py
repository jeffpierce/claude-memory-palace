"""
Database connection for Moltbook gateway.

Separate SQLite database at ~/.moltbook/gateway.db.
Completely independent from the memory palace database.
"""
import os
from pathlib import Path
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from moltbook_tools.models import Base

_engine = None
_SessionLocal = None


def _get_db_path() -> Path:
    """Get path to gateway database, creating directory if needed."""
    data_dir = Path.home() / ".moltbook"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "gateway.db"


def get_engine(db_url: str = None):
    """Get SQLAlchemy engine. Accepts optional URL override for testing."""
    global _engine

    if _engine is not None:
        return _engine

    if db_url is None:
        db_url = f"sqlite:///{_get_db_path()}"

    _engine = create_engine(
        db_url,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )

    @event.listens_for(_engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return _engine


def get_session_factory():
    """Get session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=get_engine()
        )
    return _SessionLocal


def get_session() -> Session:
    """Get a new database session."""
    return get_session_factory()()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Transactional scope — auto-commits on success, rolls back on exception."""
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(engine=None):
    """Create all tables. Safe to call multiple times."""
    eng = engine or get_engine()
    Base.metadata.create_all(bind=eng)


def reset_engine():
    """Reset engine and session factory (for testing)."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
        _engine = None
    _SessionLocal = None
