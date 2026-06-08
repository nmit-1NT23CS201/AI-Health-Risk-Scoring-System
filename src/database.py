import sqlite3
from pathlib import Path

# Database location
DB_PATH = Path("database/healthrisk.db")
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