import sqlite3
from pathlib import Path

# Database location is anchored to this module so it works from any cwd.
DB_PATH = Path(__file__).resolve().parent / "healthrisk.db"
print("DATABASE PATH:", DB_PATH.resolve())

def get_connection():
    """
    Returns a connection to the SQLite database.
    """
    return sqlite3.connect(DB_PATH)


def create_tables():
    """
    Creates the database and required tables if they do not exist.
    """
    print("CREATING TABLES...")
    # Ensure database folder exists
    DB_PATH.parent.mkdir(exist_ok=True)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS assessments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,

        age INTEGER,
        gender TEXT,

        bmi REAL,
        systolic_bp INTEGER,
        diastolic_bp INTEGER,

        cholesterol_mg_dl REAL,

        smoking TEXT,
        alcohol_consumption TEXT,
        physical_activity TEXT,
        family_history TEXT,

        heart_rate_bpm REAL,
        sdnn_hrv REAL,
        rmssd_hrv REAL,
        spo2 REAL,

        risk_score REAL,
        risk_level TEXT,

        model_version TEXT,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


def save_assessment(payload, risk_score, risk_level):
    """
    Stores a completed health assessment in the database.
    Returns the generated assessment ID.
    """

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO assessments (
        age,
        gender,
        bmi,
        systolic_bp,
        diastolic_bp,
        cholesterol_mg_dl,
        smoking,
        alcohol_consumption,
        physical_activity,
        family_history,
        heart_rate_bpm,
        sdnn_hrv,
        rmssd_hrv,
        spo2,
        risk_score,
        risk_level,
        model_version
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
    (
        payload.age,
        payload.gender,
        payload.bmi,
        payload.systolic_bp,
        payload.diastolic_bp,
        payload.cholesterol_mg_dl,
        payload.smoking,
        payload.alcohol_consumption,
        payload.physical_activity,
        payload.family_history,
        payload.heart_rate_bpm,
        payload.sdnn_hrv,
        payload.rmssd_hrv,
        payload.spo2,
        risk_score,
        risk_level,
        "RF_v1"
    ))

    assessment_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return assessment_id
def get_assessment_history(limit=20):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT
        id,
        age,
        risk_score,
        risk_level,
        created_at
    FROM assessments
    ORDER BY created_at DESC
    LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()

    conn.close()

    return rows


def delete_assessment(assessment_id: int) -> bool:
    """
    Deletes a single assessment by ID.
    Returns True if a row was deleted, False if the ID did not exist.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM assessments WHERE id = ?", (assessment_id,))
    if cursor.fetchone() is None:
        conn.close()
        return False

    cursor.execute("DELETE FROM assessments WHERE id = ?", (assessment_id,))
    conn.commit()
    conn.close()
    return True


def delete_all_assessments() -> int:
    """
    Deletes every row in the assessments table.
    Returns the number of rows deleted.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM assessments")
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted

# ---------------------------------------------------------------------------
# V2 persistence (Phase 2C). Additive only: the V1 `assessments` table and the
# V1 functions above are unchanged. V2 tables are created from the V2 lifespan
# (backend/api/main.py), never at import time.
# ---------------------------------------------------------------------------
V2_SCHEMA_VERSION = 1

_V2_COLUMNS = (
    "submission_id", "created_at", "disease", "mode", "config_key", "inputs_json",
    "model_input_json", "probability", "threshold", "predicted_positive", "imputed_json",
    "model_version", "artifact_sha256", "schema_version",
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
            schema_version INTEGER NOT NULL DEFAULT 1
        )
        """)
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
            [tuple(row[c] for c in _V2_COLUMNS) for row in rows],
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
    """Deletes every V2 row (V1 `assessments` is not touched). Returns rows deleted."""
    conn = get_connection()
    try:
        cur = conn.execute("DELETE FROM v2_assessments")
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()