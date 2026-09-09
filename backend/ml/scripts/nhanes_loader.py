"""
nhanes_loader.py
================
Stage 1C — NHANES Data Pipeline: Module 1 — Raw File Loader & Validator

Responsibilities
----------------
- Verify every required XPT file exists and is readable.
- Confirm every required NHANES variable is present in its expected file.
- Detect duplicate SEQNs.
- Load ONLY the required columns (plus SEQN) for efficiency.
- Handle the SAS floating-point underflow epsilon (5.3976e-79 -> 0.0).
- Write a machine-readable validation report to:
    backend/ml/data/interim/audit/nhanes_raw_validation.json

SAFETY CONTRACT
---------------
- Reads raw XPT files read-only; never writes to the raw directory.
- Raises a descriptive RuntimeError if any required variable is missing
  (does NOT silently substitute another variable).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PROJECT PATHS  (all relative to workspace root for reproducibility)
# ---------------------------------------------------------------------------
RAW_DIR = Path("backend/ml/data/raw/nhanes_2021_2023")
AUDIT_DIR = Path("backend/ml/data/interim/audit")

# ---------------------------------------------------------------------------
# VARIABLE MANIFEST
# Each entry: (file_basename, [required_columns], population_note)
# ---------------------------------------------------------------------------
VARIABLE_MANIFEST: dict[str, dict[str, Any]] = {
    "DEMO_L": {
        "file": "DEMO_L.xpt",
        "required": [
            "SEQN", "RIDAGEYR", "RIAGENDR", "DMDEDUC2", "INDFMPIR",
            "WTINT2YR", "WTMEC2YR", "SDMVSTRA", "SDMVPSU",
        ],
        "population": "All interviewed (N~11933)",
        "purpose": "Demographics, SES, survey design weights",
    },
    "BMX_L": {
        "file": "BMX_L.xpt",
        "required": ["SEQN", "BMXBMI", "BMXWAIST"],
        "population": "MEC examined (N~8860)",
        "purpose": "Anthropometrics (BMI, waist)",
    },
    "BPXO_L": {
        "file": "BPXO_L.xpt",
        "required": [
            "SEQN",
            "BPXOSY1", "BPXOSY2", "BPXOSY3",
            "BPXODI1", "BPXODI2", "BPXODI3",
            "BPXOPLS1", "BPXOPLS2", "BPXOPLS3",
        ],
        "population": "MEC oscillometric exam (N~7801)",
        "purpose": "Blood pressure readings (3 oscillometric readings each)",
    },
    "BPQ_L": {
        "file": "BPQ_L.xpt",
        "required": ["SEQN", "BPQ020"],
        "population": "Interviewed adults (N~8501)",
        "purpose": "HTN model target component (BPQ020=diagnosed hypertension)",
    },
    "SMQ_L": {
        "file": "SMQ_L.xpt",
        "required": ["SEQN", "SMQ020", "SMQ040"],
        "population": "Interviewed (N~9015)",
        "purpose": "Smoking status (Mode A predictor)",
    },
    "ALQ_L": {
        "file": "ALQ_L.xpt",
        "required": ["SEQN", "ALQ111", "ALQ121"],
        "population": "Interviewed adults (N~6337)",
        "purpose": "Alcohol consumption (Mode A predictor)",
    },
    "PAQ_L": {
        "file": "PAQ_L.xpt",
        "required": ["SEQN", "PAD790Q", "PAD810Q", "PAD680"],
        "population": "Interviewed (N~8153)",
        "purpose": "Physical activity (Mode A predictor)",
    },
    "DIQ_L": {
        "file": "DIQ_L.xpt",
        "required": ["SEQN", "DIQ010", "DIQ050", "DIQ070"],
        "population": "All interviewed (N~11744)",
        "purpose": "Diabetes target components (DIQ010=diagnosis, DIQ050=insulin, DIQ070=pills)",
    },
    "MCQ_L": {
        "file": "MCQ_L.xpt",
        "required": ["SEQN", "MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"],
        "population": "All interviewed (N~11744)",
        "purpose": "CVD target components (hard CVD history MCQ160B-F)",
    },
    "GHB_L": {
        "file": "GHB_L.xpt",
        "required": ["SEQN", "LBXGH"],
        "population": "Routine lab panel (N~7199)",
        "purpose": "HbA1c — Mode B predictor; PROHIBITED in DM model",
    },
    "GLU_L": {
        "file": "GLU_L.xpt",
        "required": ["SEQN", "LBXGLU", "WTSAF2YR"],
        "population": "Fasting subsample (N~3996)",
        "purpose": "Fasting glucose — Mode B predictor; PROHIBITED in DM model. WTSAF2YR=fasting weight",
    },
    "TCHOL_L": {
        "file": "TCHOL_L.xpt",
        "required": ["SEQN", "LBXTC"],
        "population": "Routine lab panel (N~8068)",
        "purpose": "Total cholesterol — Mode B predictor",
    },
    "HDL_L": {
        "file": "HDL_L.xpt",
        "required": ["SEQN", "LBDHDD"],
        "population": "Routine lab panel (N~8068)",
        "purpose": "HDL cholesterol — Mode B predictor",
    },
    "TRIGLY_L": {
        "file": "TRIGLY_L.xpt",
        "required": ["SEQN", "LBXTLG", "LBDLDL"],
        "population": "Fasting subsample (N~3996)",
        "purpose": "Triglycerides and calculated LDL — Mode B predictor (fasting subsample)",
    },
    "BIOPRO_L": {
        "file": "BIOPRO_L.xpt",
        "required": ["SEQN", "LBXSCR", "LBXSBU", "LBXSUA", "LBXSATSI", "LBXSASSI", "LBXSAL"],
        "population": "Routine lab panel (N~7199)",
        "purpose": "Biochemistry profile: creatinine, BUN, uric acid, ALT, AST, albumin",
    },
    "CBC_L": {
        "file": "CBC_L.xpt",
        "required": ["SEQN", "LBXHGB", "LBXWBCSI", "LBXPLTSI", "LBXRDW"],
        "population": "CBC lab panel (N~8727)",
        "purpose": "Complete blood count: hemoglobin, WBC, platelets, RDW",
    },
}

# SAS floating-point underflow epsilon — must be clamped to 0.0
_SAS_EPSILON = 5.397605346934028e-79
_SAS_EPSILON_THRESHOLD = 1e-70   # treat anything below this magnitude as 0


def _clamp_sas_epsilon(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Replace SAS floating-point underflow epsilons with 0.0.

    Returns modified DataFrame and a dict mapping column -> count of replacements.
    """
    epsilon_counts: dict[str, int] = {}
    for col in df.select_dtypes(include="number").columns:
        if col == "SEQN":
            continue
        mask = (df[col].abs() < _SAS_EPSILON_THRESHOLD) & (df[col] != 0.0) & df[col].notna()
        n = int(mask.sum())
        if n > 0:
            df[col] = df[col].where(~mask, 0.0)
            epsilon_counts[col] = n
    return df, epsilon_counts


def load_raw_component(
    key: str,
    raw_dir: Path = RAW_DIR,
) -> pd.DataFrame:
    """Load a single NHANES XPT component, selecting only required columns.

    Parameters
    ----------
    key     : Manifest key (e.g. 'DEMO_L')
    raw_dir : Path to the raw NHANES directory

    Returns
    -------
    DataFrame with only required columns loaded
    """
    spec = VARIABLE_MANIFEST[key]
    fpath = raw_dir / spec["file"]

    if not fpath.exists():
        raise FileNotFoundError(
            f"[nhanes_loader] Required file MISSING: {fpath}\n"
            f"  Expected: {spec['file']} for {spec['purpose']}"
        )

    log.info("Loading %s from %s ...", key, fpath)
    df = pd.read_sas(str(fpath), format="xport")

    # Verify all required columns exist
    missing_cols = [c for c in spec["required"] if c not in df.columns]
    if missing_cols:
        raise RuntimeError(
            f"[nhanes_loader] VARIABLE MISSING in {spec['file']}:\n"
            f"  Missing columns: {missing_cols}\n"
            f"  Available columns: {list(df.columns)}\n"
            "  Do NOT substitute another variable — update the manifest or source file."
        )

    # Select only required columns
    df = df[spec["required"]].copy()

    # Detect duplicate SEQN
    n_dup = int(df["SEQN"].duplicated().sum())
    if n_dup > 0:
        log.warning("  [%s] %d duplicate SEQN values detected!", key, n_dup)

    # Clamp SAS epsilon zeros
    df, eps_counts = _clamp_sas_epsilon(df)
    if eps_counts:
        log.info("  [%s] SAS epsilon clamped to 0.0 in %d column(s): %s", key, len(eps_counts), list(eps_counts.keys()))

    log.info("  [%s] Loaded: %d rows x %d cols. Duplicates: %d. Epsilon fixes: %s",
             key, len(df), len(df.columns), n_dup, eps_counts)

    return df


def validate_all_files(
    raw_dir: Path = RAW_DIR,
    audit_dir: Path = AUDIT_DIR,
) -> dict[str, Any]:
    """Validate all 16 required NHANES XPT files and write a JSON validation report.

    Returns the validation report dict.
    """
    audit_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"status": "OK", "files": {}, "errors": []}

    for key, spec in VARIABLE_MANIFEST.items():
        fpath = raw_dir / spec["file"]
        entry: dict[str, Any] = {
            "file": spec["file"],
            "path": str(fpath),
            "exists": fpath.exists(),
            "purpose": spec["purpose"],
            "population": spec["population"],
            "required_columns": spec["required"],
            "missing_columns": [],
            "duplicate_seqn": 0,
            "rows": None,
            "cols_in_file": None,
            "epsilon_fixes": {},
            "status": "PENDING",
            "error": None,
        }

        if not fpath.exists():
            entry["status"] = "FAIL"
            entry["error"] = f"File not found: {fpath}"
            report["errors"].append(entry["error"])
            report["status"] = "FAIL"
            report["files"][key] = entry
            continue

        try:
            df = pd.read_sas(str(fpath), format="xport")
            entry["rows"] = int(len(df))
            entry["cols_in_file"] = int(len(df.columns))

            missing = [c for c in spec["required"] if c not in df.columns]
            entry["missing_columns"] = missing
            if missing:
                entry["status"] = "FAIL"
                entry["error"] = f"Missing variables: {missing}"
                report["errors"].append(entry["error"])
                report["status"] = "FAIL"
                report["files"][key] = entry
                continue

            # Select required cols for epsilon + dup check
            sub = df[spec["required"]].copy()
            n_dup = int(sub["SEQN"].duplicated().sum())
            entry["duplicate_seqn"] = n_dup

            _, eps = _clamp_sas_epsilon(sub)
            entry["epsilon_fixes"] = {k: v for k, v in eps.items()}

            entry["status"] = "OK" if n_dup == 0 else "WARN_DUP_SEQN"

        except Exception as exc:  # noqa: BLE001
            entry["status"] = "FAIL"
            entry["error"] = str(exc)
            report["errors"].append(str(exc))
            report["status"] = "FAIL"

        report["files"][key] = entry

    out_path = audit_dir / "nhanes_raw_validation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log.info("Validation report written to %s", out_path)

    if report["status"] == "OK":
        log.info("ALL %d NHANES XPT files validated successfully.", len(VARIABLE_MANIFEST))
    else:
        log.error("VALIDATION FAILURES: %s", report["errors"])

    return report
