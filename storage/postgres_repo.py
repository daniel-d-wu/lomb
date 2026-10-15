"""
Cloud SQL Postgres implementation of StorageRepository (2026-10-14, Google
Cloud build step 2). Every query is the shared one in storage.py's
SQLStorageRepository; this file only adds connections and the schema.

Connections: the Cloud SQL Python Connector with pg8000, logging in with
IAM (cloudsql.iam_authentication=on) -- the caller's Google identity is the
login, so no database password exists in code, config or Secret Manager.
Connections are pooled (SQLAlchemy's QueuePool, no ORM): opening a Cloud
SQL connection costs a round trip to Google for a certificate.

Schema: created on first connect from storage.py's DDL, as the NOLOGIN role
lomb_app, so tables belong to that role and every identity granted lomb_app
(Dan, the API and worker service accounts) can use them. One-time setup,
run as the postgres user in Cloud SQL Studio:

    CREATE ROLE lomb_app NOLOGIN;
    GRANT ALL ON SCHEMA public TO lomb_app;          -- in each database
    GRANT lomb_app TO "<google account email>" WITH ADMIN OPTION;

Like SQLite, CREATE TABLE IF NOT EXISTS only adds what's missing: a later
schema change to an existing table needs its own ALTER here.
"""

import re
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.dialects.postgresql.pg8000 import PGDialect_pg8000  # noqa: E402
from sqlalchemy.pool import QueuePool  # noqa: E402

from storage.storage import DDL, SQLStorageRepository  # noqa: E402

INSTANCE = "lomb-prod:us-east4:lomb-db"
SCHEMA_LOCK = 7011001  # pg_advisory_xact_lock key: one schema creation at a time across instances

PG_DDL = DDL.replace("BOOLEAN NOT NULL DEFAULT 0", "BOOLEAN NOT NULL DEFAULT FALSE")


def _statements(ddl: str) -> list[str]:
    without_comments = re.sub(r"--[^\n]*", "", ddl)
    return [s.strip() for s in without_comments.split(";") if s.strip()]


def cloud_sql_connect(db_user: str, db_name: str = "lomb", *, instance: str = INSTANCE,
                      credentials=None) -> Callable:
    """A connect() for PostgresStorageRepository. db_user is a Google
    account email, or a service account's email minus '.gserviceaccount.com'.
    credentials default to the environment's (the service account on Cloud
    Run, application-default credentials locally)."""
    from google.cloud.sql.connector import Connector, IPTypes

    connector = Connector(refresh_strategy="lazy", credentials=credentials)

    def connect():
        return connector.connect(instance, "pg8000", user=db_user, db=db_name,
                                 enable_iam_auth=True, ip_type=IPTypes.PUBLIC)

    connect.connector = connector
    return connect


class PostgresStorageRepository(SQLStorageRepository):
    placeholder = "%s"

    def __init__(self, connect: Callable, *, schema_role: str | None = "lomb_app",
                 pool_size: int = 5, max_overflow: int = 5):
        super().__init__()
        self._pool = QueuePool(connect, pool_size=pool_size, max_overflow=max_overflow,
                               pre_ping=True, reset_on_return="rollback",
                               dialect=PGDialect_pg8000())  # the dialect only supplies the pre-ping
        self.schema_role = schema_role
        self._create_schema()

    def _new_connection(self):
        return self._pool.connect()

    def _create_schema(self) -> None:
        with self.transaction():
            cur = self._local.conn.cursor()
            cur.execute("SELECT pg_advisory_xact_lock(%s)", (SCHEMA_LOCK,))
            if self.schema_role:
                cur.execute(f'SET LOCAL ROLE "{self.schema_role}"')
            for statement in _statements(PG_DDL):
                cur.execute(statement)
            cur.close()

    def close(self) -> None:
        self._pool.dispose()
