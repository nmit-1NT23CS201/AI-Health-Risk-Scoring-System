"""SQLite persistence for V2 assessment history (table `v2_assessments`).

Raw sqlite3, per-call connections. The table is created from the API lifespan
(backend/api/main.py), never at import time.
"""
import sqlite3
from pathlib import Path

# Database location is anchored to this module so it works from any cwd.
# The file is git-ignored (it holds user-submitted health data).
DB_PATH = Path(__file__).resolve().parent / "healthrisk.db"


def get_connection():
    """
    Returns a connection to the SQLite database.
    """
    return sqlite3.connect(DB_PATH)

V2_SCHEMA_VERSION = 1

_V2_COLUMNS = (
    "submission_id", "created_at", "disease", "mode", "config_key", "inputs_json",
    "model_input_json", "probability", "threshold", "predicted_positive", "imputed_json",
    "model_version", "artifact_sha256", "schema_version", "patient_name",
)


def create_v2_tables():
    """Creates the V2 table and indexes if they do not exist (idempotent)."""
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = get_connection()
    try:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS v2_assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            submission_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            disease TEXT NOT NULL CHECK (disease IN ('cvd','diabetes','hypertension')),
            mode TEXT NOT NULL CHECK (mode IN ('A','B')),
            config_key TEXT NOT NULL,
            inputs_json TEXT NOT NULL,
            model_input_json TEXT NOT NULL,
            probability REAL NOT NULL,
            threshold REAL NOT NULL,
            predicted_positive INTEGER NOT NULL CHECK (predicted_positive IN (0,1)),
            imputed_json TEXT,
            model_version TEXT NOT NULL,
            artifact_sha256 TEXT NOT NULL,
            schema_version INTEGER NOT NULL DEFAULT 1,
            patient_name TEXT
        )
        """)
        # Idempotent upgrade for databases created before patient_name existed (no data is dropped).
        existing = {row[1] for row in conn.execute("PRAGMA table_info(v2_assessments)")}
        if "patient_name" not in existing:
            conn.execute("ALTER TABLE v2_assessments ADD COLUMN patient_name TEXT")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_v2_assessments_submission ON v2_assessments(submission_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_v2_assessments_created ON v2_assessments(created_at)"
        )
        conn.commit()
    finally:
        conn.close()


def save_v2_submission(rows):
    """
    Inserts every disease row of one submission in a single transaction
    (all rows are stored or none). `rows` is a list of dicts keyed by the
    v2_assessments columns (without `id`). Returns the number of rows stored.
    """
    conn = get_connection()
    try:
        conn.executemany(
            "INSERT INTO v2_assessments (%s) VALUES (%s)"
            % (", ".join(_V2_COLUMNS), ", ".join("?" for _ in _V2_COLUMNS)),
            [tuple(row.get(c) if c == "patient_name" else row[c] for c in _V2_COLUMNS) for row in rows],
        )
        conn.commit()
        return len(rows)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _v2_rows(cursor):
    names = [d[0] for d in cursor.description]
    return [dict(zip(names, r)) for r in cursor.fetchall()]


def list_v2_history(limit=20, before_id=None):
    """
    Newest-first (by id DESC) rows of the `limit` most recent submissions.
    A submission's position is the highest row id it owns; `before_id` pages
    to submissions whose highest row id is below it. Rows are returned oldest
    first within a submission.
    """
    conn = get_connection()
    try:
        having, params = "", []
        if before_id is not None:
            having = "HAVING MAX(id) < ?"
            params.append(before_id)
        cur = conn.execute(
            "SELECT submission_id, MAX(id) AS last_id FROM v2_assessments "
            "GROUP BY submission_id %s ORDER BY last_id DESC LIMIT ?" % having,
            (*params, limit),
        )
        groups = cur.fetchall()
        if not groups:
            return []
        ids = [g[0] for g in groups]
        cur = conn.execute(
            "SELECT * FROM v2_assessments WHERE submission_id IN (%s) ORDER BY id ASC"
            % ", ".join("?" for _ in ids),
            ids,
        )
        by_sub = {}
        for row in _v2_rows(cur):
            by_sub.setdefault(row["submission_id"], []).append(row)
        return [{"submission_id": sid, "last_id": last, "rows": by_sub[sid]} for sid, last in groups]
    finally:
        conn.close()


def get_v2_submission(submission_id):
    """All rows of one submission (oldest first); empty list if it does not exist."""
    conn = get_connection()
    try:
        cur = conn.execute(
            "SELECT * FROM v2_assessments WHERE submission_id = ? ORDER BY id ASC", (submission_id,)
        )
        return _v2_rows(cur)
    finally:
        conn.close()


def delete_v2_submission(submission_id):
    """Deletes one submission. Returns the number of rows deleted (0 if unknown)."""
    conn = get_connection()
    try:
        cur = conn.execute("DELETE FROM v2_assessments WHERE submission_id = ?", (submission_id,))
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def delete_all_v2_history():
    """Deletes every V2 history row. Returns rows deleted."""
    conn = get_connection()
    try:
        cur = conn.execute("DELETE FROM v2_assessments")
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()