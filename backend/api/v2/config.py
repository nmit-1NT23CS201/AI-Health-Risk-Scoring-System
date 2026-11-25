"""Paths and constants for the V2 inference engine.

Feature lists, column order and thresholds are deliberately NOT defined here: they are
loaded from ``stage_1f_results.json`` by ``registry.py`` (single source of truth).
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = REPO_ROOT / "backend"

DEFAULT_RESULTS_JSON = BACKEND_DIR / "ml" / "evaluation" / "stage_1f" / "stage_1f_results.json"
SPLITS_DIR = BACKEND_DIR / "ml" / "data" / "processed" / "nhanes_2021_2023" / "splits"
ASSETS_DIR = Path(__file__).resolve().parent / "assets"

# Persisted training-split SHAP background for Hypertension Mode A (LinearExplainer).
HTN_A_BACKGROUND_FILE = ASSETS_DIR / "hypertension_mode_a_shap_background.parquet"
HTN_A_BACKGROUND_SIZE = 200
HTN_A_BACKGROUND_SEED = 42

SURVEY_METADATA_COLUMNS = ("SEQN", "WTINT2YR", "WTMEC2YR", "WTSAF2YR", "SDMVSTRA", "SDMVPSU")

TOP_K_FEATURES = 8
ADDITIVITY_TOLERANCE = 1e-5
MODEL_VERSION_PREFIX = "stage1f"

DISCLAIMER = (
    "Research prototype for risk estimation and preventive screening support. "
    "Not a clinically validated diagnostic product; not medical advice."
)

DISEASE_INFO = {
    "cvd": {
        "label": "Cardiovascular disease",
        "target_definition": (
            "Self-reported prior diagnosis of congestive heart failure, coronary heart disease, "
            "angina, heart attack or stroke (NHANES MCQ160B-F)."
        ),
    },
    "diabetes": {
        "label": "Diabetes",
        "target_definition": (
            "HbA1c >= 6.5%, fasting glucose >= 126 mg/dL, diagnosed diabetes, or insulin / "
            "diabetes-pill use (NHANES)."
        ),
    },
    "hypertension": {
        "label": "Hypertension",
        "target_definition": (
            "Mean systolic BP >= 140 or diastolic BP >= 90 mmHg, or a told-high-blood-pressure "
            "diagnosis (JNC7-style, NHANES)."
        ),
    },
}

PROBABILITY_NOTE = (
    "Sigmoid-calibrated probability learned on NHANES 2021-2023 adults; it reflects that cohort's "
    "prevalence and is not a diagnosis."
)

METRICS_CAVEATS = (
    "Offline research evaluation on the held-out NHANES test split; not a statement about any individual.",
    "Unweighted sample estimates (NHANES survey weights were not used).",
    "The test split also informed model-family selection in Stages 1D-1E, so metrics may be slightly optimistic.",
    "US cohort, no external validation; the CVD target is self-reported prior diagnosis, not incident risk.",
)


def results_json_path() -> Path:
    """Results file location; ``V2_RESULTS_JSON`` overrides it (used by degraded-startup tests)."""
    override = os.environ.get("V2_RESULTS_JSON")
    return Path(override) if override else DEFAULT_RESULTS_JSON
