"""SQLite engine, session factory, and table bootstrap for the feedback log.

A single SQLite file lives at data/feedback.db. It is created on first import
and persists across restarts (success criterion for Phase 5).

Thread safety: SQLite's WAL mode is enabled so reads and the one active
writer don't block each other. The FastAPI service is single-process, so
contention is minimal.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from agent.config import REPO_ROOT
from agent.feedback_log.models import Base

logger = logging.getLogger(__name__)

DB_PATH: Path = REPO_ROOT / "data" / "feedback.db"

_engine: Engine | None = None
_SessionFactory: sessionmaker | None = None


def _enable_wal(dbapi_conn, _connection_record) -> None:
    """Turn on WAL mode and foreign-key enforcement for every new connection."""
    dbapi_conn.execute("PRAGMA journal_mode=WAL")
    dbapi_conn.execute("PRAGMA foreign_keys=ON")


def get_engine() -> Engine:
    """Return the singleton SQLAlchemy engine, creating it on first call."""
    global _engine
    if _engine is not None:
        return _engine

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{DB_PATH}",
        connect_args={"check_same_thread": False},
        echo=False,
    )
    event.listen(engine, "connect", _enable_wal)
    Base.metadata.create_all(engine)
    logger.info("Feedback DB initialised at %s", DB_PATH)
    _engine = engine
    return _engine


def get_session_factory() -> sessionmaker:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionFactory


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Yield a transactional Session; commits on exit, rolls back on error."""
    factory = get_session_factory()
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def ensure_tables() -> None:
    """Create all tables if they don't exist yet. Idempotent."""
    Base.metadata.create_all(get_engine())
