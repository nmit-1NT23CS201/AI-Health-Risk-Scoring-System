from typing import List, Tuple

import pandas as pd


FEATURE_COLUMNS: List[str] = [
    "age",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "cholesterol_mg_dl",
    "smoking",
    "alcohol_consumption",
    "physical_activity",
    "family_history",
    "heart_rate_bpm",
    "sdnn_hrv",
    "rmssd_hrv",
    "spo2",
]

TARGET_COLUMN = "risk_score"


def select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    missing = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y
