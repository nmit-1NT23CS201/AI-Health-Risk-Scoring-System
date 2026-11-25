"""Persistence for V2 assessment history (table `v2_assessments`).

Two backends, chosen automatically per call from the environment:

* **PostgreSQL** (e.g. Neon) when ``DATABASE_URL`` starts with ``postgres://`` or ``postgresql://``.
* **SQLite** (local file ``healthrisk.db``) when ``DATABASE_URL`` is unset or empty, e.g. in tests.

History is scoped per browser: every read and delete is filtered by ``client_id``, so one client can
never see or delete another client's submissions. The connection string is never logged or returned.
The table is created from the API lifespan (backend/api/main.py), never at import time.
"""
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

# SQLite location, anchored to this module so it works from any cwd (git-ignored; holds health data).
DB_PATH = Path(__file__).resolve().parent / "healthrisk.db"

V2_SCHEMA_VERSION = 1
DEFAULT_CLIENT_ID = "anonymous"

# Exceptions the API treats as "history storage unavailable" (connection strings are never echoed).
DB_ERRORS = (sqlite3.Error, psycopg.Error)


def _database_url():
    return os.getenv("DATABASE_URL", "").strip()


def backend_name():
    """'postgresql' or 'sqlite' (never exposes the connection string)."""
    url = _database_url()
    return "postgresql" if url.startswith(("postgres://", "postgresql://")) else "sqlite"


def get_connection():
    if backend_name() == "postgresql":
        return psycopg.connect(_database_url(), row_factory=dict_row)
    return sqlite3.connect(DB_PATH)


@contextmanager
def _session():
    """Connection with commit-on-success / rollback-on-error, always closed."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _q(sql):
    """Statements are written with '?' placeholders; PostgreSQL uses '%s'."""
    return sql.replace("?", "%s") if backend_name() == "postgresql" else sql


def _execute(conn, sql, params=()):
    return conn.execute(_q(sql), tuple(params))


def _dicts(cursor):
    rows = cursor.fetchall()
    if rows and isinstance(rows[0], dict):  # psycopg dict_row
        return rows
    names = [d[0] for d in cursor.description]
    return [dict(zip(names, r)) for r in rows]


_V2_COLUMNS = (
    "submission_id", "created_at", "disease", "mode", "config_key", "inputs_json",
    "model_input_json", "probability", "threshold", "predicted_positive", "imputed_json",
    "model_version", "artifact_sha256", "schema_version", "patient_name", "client_id",
)


def _existing_columns(conn):
    if backend_name() == "postgresql":
        cur = _execute(
            conn,
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = current_schema() AND table_name = 'v2_assessments'",
        )
        return {r["column_name"] for r in cur.fetchall()}
    return {r[1] for r in conn.execute("PRAGMA table_info(v2_assessments)")}


def create_v2_tables():
    """Creates the table and indexes if missing and upgrades older tables in place (idempotent)."""
    pg = backend_name() == "postgresql"
    if not pg:
        DB_PATH.parent.mkdir(exist_ok=True)
    pk = "BIGSERIAL PRIMARY KEY" if pg else "INTEGER PRIMARY KEY AUTOINCREMENT"
    real = "DOUBLE PRECISION" if pg else "REAL"
    with _session() as conn:
        _execute(conn, f"""
        CREATE TABLE IF NOT EXISTS v2_assessments (
            id {pk},
            submission_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            disease TEXT NOT NULL CHECK (disease IN ('cvd','diabetes','hypertension')),
            mode TEXT NOT NULL CHECK (mode IN ('A','B')),
            config_key TEXT NOT NULL,
            inputs_json TEXT NOT NULL,
            model_input_json TEXT NOT NULL,
            probability {real} NOT NULL,
            threshold {real} NOT NULL,
            predicted_positive INTEGER NOT NULL CHECK (predicted_positive IN (0,1)),
            imputed_json TEXT,
            model_version TEXT NOT NULL,
            artifact_sha256 TEXT NOT NULL,
            schema_version INTEGER NOT NULL DEFAULT 1,
            patient_name TEXT,
            client_id TEXT NOT NULL DEFAULT 'anonymous'
        )
        """)
        # Upgrade tables created by earlier versions without dropping any rows.
        existing = _existing_columns(conn)
        if "patient_name" not in existing:
            _execute(conn, "ALTER TABLE v2_assessments ADD COLUMN patient_name TEXT")
        if "client_id" not in existing:
            _execute(conn, "ALTER TABLE v2_assessments ADD COLUMN client_id TEXT NOT NULL DEFAULT 'anonymous'")
        _execute(conn, "CREATE INDEX IF NOT EXISTS idx_v2_assessments_submission ON v2_assessments(submission_id)")
        _execute(conn, "CREATE INDEX IF NOT EXISTS idx_v2_assessments_created ON v2_assessments(created_at)")
        _execute(
            conn,
            "CREATE INDEX IF NOT EXISTS idx_v2_assessments_client_created ON v2_assessments(client_id, id DESC)",
        )


def save_v2_submission(client_id, rows):
    """
    Inserts every disease row of one submission for `client_id` in a single transaction
    (all rows or none). `rows` are dicts keyed by the v2_assessments columns (without `id`
    and `client_id`; the latter is set from the argument). Returns the number of rows stored.
    """
    cols = ", ".join(_V2_COLUMNS)
    marks = ", ".join("?" for _ in _V2_COLUMNS)
    data = [
        tuple(client_id if c == "client_id" else (row.get(c) if c == "patient_name" else row[c]) for c in _V2_COLUMNS)
        for row in rows
    ]
    with _session() as conn:
        for params in data:
            _execute(conn, f"INSERT INTO v2_assessments ({cols}) VALUES ({marks})", params)
    return len(rows)


def list_v2_history(client_id, limit=20, before_id=None):
    """
    Newest-first rows of the `limit` most recent submissions of `client_id`. A submission's position
    is the highest row id it owns; `before_id` pages to submissions whose highest row id is below it.
    Rows are returned oldest first within a submission.
    """
    with _session() as conn:
        having, params = "", [client_id]
        if before_id is not None:
            having = "HAVING MAX(id) < ?"
            params.append(before_id)
        cur = _execute(
            conn,
            "SELECT submission_id, MAX(id) AS last_id FROM v2_assessments WHERE client_id = ? "
            f"GROUP BY submission_id {having} ORDER BY last_id DESC LIMIT ?",
            (*params, limit),
        )
        groups = _dicts(cur)
        if not groups:
            return []
        ids = [g["submission_id"] for g in groups]
        cur = _execute(
            conn,
            "SELECT * FROM v2_assessments WHERE client_id = ? AND submission_id IN (%s) ORDER BY id ASC"
            % ", ".join("?" for _ in ids),
            (client_id, *ids),
        )
        by_sub = {}
        for row in _dicts(cur):
            by_sub.setdefault(row["submission_id"], []).append(row)
        return [{"submission_id": g["submission_id"], "last_id": g["last_id"], "rows": by_sub[g["submission_id"]]}
                for g in groups]


def get_v2_submission(client_id, submission_id):
    """Rows of one of `client_id`'s submissions (oldest first); empty if it does not exist for that client."""
    with _session() as conn:
        cur = _execute(
            conn,
            "SELECT * FROM v2_assessments WHERE client_id = ? AND submission_id = ? ORDER BY id ASC",
            (client_id, submission_id),
        )
        return _dicts(cur)


def delete_v2_submission(client_id, submission_id):
    """Deletes one of `client_id`'s submissions. Returns rows deleted (0 if unknown or owned by another client)."""
    with _session() as conn:
        cur = _execute(
            conn, "DELETE FROM v2_assessments WHERE client_id = ? AND submission_id = ?", (client_id, submission_id)
        )
        return cur.rowcount


def delete_all_v2_history(client_id):
    """Deletes every row owned by `client_id` (other clients are untouched). Returns rows deleted."""
    with _session() as conn:
        cur = _execute(conn, "DELETE FROM v2_assessments WHERE client_id = ?", (client_id,))
        return cur.rowcount