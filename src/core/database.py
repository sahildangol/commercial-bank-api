from __future__ import annotations

import logging

from decouple import config
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)


def _host_candidates(primary_host: str) -> list[str]:
    fallback_raw = config(
        "POSTGRES_HOST_FALLBACKS",
        default="db,postgres_db,localhost,127.0.0.1",
    )
    candidates = [primary_host]
    candidates.extend([item.strip() for item in str(fallback_raw).split(",") if item.strip()])

    deduped: list[str] = []
    seen: set[str] = set()
    for host in candidates:
        key = host.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(host)
    return deduped


POSTGRES_USER = config("POSTGRES_USER")
POSTGRES_PASSWORD = config("POSTGRES_PASSWORD")
POSTGRES_DB = config("POSTGRES_DB", default="postgres")
POSTGRES_HOST = config("POSTGRES_HOST", default="localhost")
POSTGRES_PORT = config("POSTGRES_PORT", default=5432, cast=int)
DB_CONNECT_TIMEOUT_SECONDS = config("DB_CONNECT_TIMEOUT_SECONDS", default=5, cast=int)


def _connect_with_fallback_hosts():
    last_error: Exception | None = None
    for host in _host_candidates(POSTGRES_HOST):
        try:
            return psycopg2.connect(
                user=POSTGRES_USER,
                password=POSTGRES_PASSWORD,
                host=host,
                port=POSTGRES_PORT,
                dbname=POSTGRES_DB,
                connect_timeout=DB_CONNECT_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            last_error = exc
            logger.warning("Database session connect failed via host '%s': %s", host, exc)
    if last_error is not None:
        raise last_error
    raise RuntimeError("No database hosts configured for SQLAlchemy session engine.")


engine = create_engine(
    "postgresql+psycopg2://",
    creator=_connect_with_fallback_hosts,
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
