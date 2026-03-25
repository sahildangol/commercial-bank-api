from __future__ import annotations

import logging
import time

from decouple import config
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from src.core.database import Base
from src.db.models.company import Company
from src.db.models.inference import ModelVersion, Prediction
from src.db.models.market_data import FinancialHistory, NepseIndex, TechnicalHistory
from src.db.models.user import User
from src.db.models.user_preference import Alert, UserWatchlist

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


def _admin_url(user: str, password: str, host: str, port: int, db_name: str) -> str:
    return f"postgresql://{user}:{password}@{host}:{port}/{db_name}"


def create_tables() -> None:
    admin_user = config("DB_ADMIN_USER", default=config("POSTGRES_USER"))
    admin_password = config("DB_ADMIN_PASSWORD", default=config("POSTGRES_PASSWORD"))
    db_host = config("POSTGRES_HOST", default="localhost")
    db_port = config("POSTGRES_PORT", default=5432, cast=int)
    db_name = config("POSTGRES_DB", default="postgres")

    max_attempts = config("DB_INIT_MAX_ATTEMPTS", default=12, cast=int)
    retry_seconds = config("DB_INIT_RETRY_SECONDS", default=2.0, cast=float)
    connect_timeout_seconds = config("DB_CONNECT_TIMEOUT_SECONDS", default=5, cast=int)

    hosts = _host_candidates(db_host)
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        for host in hosts:
            admin_url = _admin_url(
                user=admin_user,
                password=admin_password,
                host=host,
                port=db_port,
                db_name=db_name,
            )
            engine = create_engine(
                admin_url,
                pool_pre_ping=True,
                connect_args={"connect_timeout": connect_timeout_seconds},
            )
            try:
                with engine.connect() as connection:
                    connection.execute(text("SELECT 1"))
                Base.metadata.create_all(bind=engine)
                logger.info(
                    "Database initialization successful via host '%s' (%s/%s).",
                    host,
                    attempt,
                    max_attempts,
                )
                return
            except OperationalError as exc:
                last_error = exc
                logger.warning(
                    "Database init failed via host '%s' (%s/%s): %s",
                    host,
                    attempt,
                    max_attempts,
                    exc,
                )
            finally:
                engine.dispose()

        if attempt < max_attempts:
            time.sleep(retry_seconds)

    raise RuntimeError(
        "Database initialization failed after retries for hosts: "
        + ", ".join(hosts)
    ) from last_error
