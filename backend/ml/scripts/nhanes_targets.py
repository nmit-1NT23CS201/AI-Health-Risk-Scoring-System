"""
nhanes_targets.py
=================
Stage 1C — NHANES Data Pipeline: Module 3 — Target Construction

Responsibilities
----------------
Build the three approved zero-leakage target variables from the merged
adult NHANES dataset as defined in MODEL_V2_DESIGN.md.

TARGET 1: target_cvd
    Hard CVD history (MCQ160B OR MCQ160C OR MCQ160D OR MCQ160E OR MCQ160F)
    1 = Positive (history of at least one hard CVD event)
    0 = Negative
    NaN = Not determinable (all five source variables are missing/unknown)

    Source variables: MCQ160B, MCQ160C, MCQ160D, MCQ160E, MCQ160F
    These variables are EXCLUDED as predictors in ALL models after being
    used to construct the target.

TARGET 2: target_diabetes
    ADA multi-criteria diabetes (HbA1c >= 6.5 OR FPG >= 126 OR diagnosis
    OR insulin OR diabetes pills)
    1 = Positive (meets at least one ADA criterion)
    0 = Negative
    NaN = Not determinable (insufficient data to evaluate any criterion)

    Target-defining variables: LBXGH / hba1c, LBXGLU / fasting_glucose,
    DIQ010, DIQ050, DIQ070
    PROHIBITED as predictors in the Diabetes model only.

TARGET 3: target_hypertension
    JNC7-defined hypertension (mean SBP >= 140 OR mean DBP >= 90 OR diagnosed)
    This is NOT Stage-2 HTN specifically; it is the JNC7 general hypertension
    threshold (SBP >= 140 or DBP >= 90 or prior physician diagnosis).
    1 = Positive (meets JNC7 criteria)
    0 = Negative
    NaN = Not determinable

    Target-defining variables: mean_sbp, mean_dbp, BPQ020
    PROHIBITED as predictors in the Hypertension model only.

SAFETY CONTRACT
---------------
- Reads only the merged interim DataFrame.
- Does not write any raw file.
- Returns a DataFrame with three added columns: target_cvd, target_diabetes,
  target_hypertension.
- Writes a JSON target construction report to:
    backend/ml/data/interim/audit/nhanes_target_construction.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

AUDIT_DIR = Path("backend/ml/data/interim/audit")


def _build_target_cvd(df: pd.DataFrame) -> pd.Series:
    """
    TARGET 1: Hard Cardiovascular Disease (CVD) History
    =====================================================
    Definition (from MODEL_V2_DESIGN.md Section 2.1):
        Target_CVD = 1 iff MCQ160B=1 OR MCQ160C=1 OR MCQ160D=1
                          OR MCQ160E=1 OR MCQ160F=1

    MCQ160B = Congestive Heart Failure diagnosis
    MCQ160C = Coronary Heart Disease diagnosis
    MCQ160D = Angina Pectoris diagnosis
    MCQ160E = Heart Attack / MI diagnosis
    MCQ160F = Stroke diagnosis

    NHANES coding: 1=Yes, 2=No, NaN=missing/refused/unknown
    After special-missing recode in nhanes_harmonizer, codes 7/9 → NaN.

    Logic:
    - If ANY source variable = 1: Target = 1 (positive CVD history)
    - If ALL source variables = 2 (No) or NaN, AND at least one = 2: Target = 0
    - If ALL source variables are NaN: Target = NaN (not determinable)
    """
    cvd_cols = ["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"]
    missing  = [c for c in cvd_cols if c not in df.columns]
    if missing:
        raise KeyError(f"[target_cvd] Missing source variables: {missing}")

    sub = df[cvd_cols].copy()
    # 1 = Yes, 2 = No. Convert to: 1 = positive, 0 = negative (2→0), NaN = missing
    sub_bin = sub.replace({2: 0})   # 2 (No) → 0

    # Any positive = 1: use max across columns (ignores NaN)
    any_positive = (sub_bin == 1).any(axis=1)

    # All NaN = undeterminable
    all_nan = sub_bin.isna().all(axis=1)

    target = pd.Series(np.nan, index=df.index, name="target_cvd", dtype="float64")
    target[any_positive] = 1.0
    target[~any_positive & ~all_nan] = 0.0   # at least one "No" answer, none positive
    # all_nan rows remain NaN

    return target


def _build_target_diabetes(df: pd.DataFrame) -> pd.Series:
    """
    TARGET 2: Diabetes Mellitus (ADA Multi-Criteria Definition)
    ============================================================
    Definition (from MODEL_V2_DESIGN.md Section 2.2):
        Target_DM = 1 iff (hba1c >= 6.5) OR (fasting_glucose >= 126)
                          OR (DIQ010 = 1) OR (DIQ050 = 1) OR (DIQ070 = 1)

    hba1c          = LBXGH (HbA1c %) — NHANES routine lab
    fasting_glucose= LBXGLU (fasting plasma glucose mg/dL) — fasting subsample
    DIQ010         = Self-reported diabetes diagnosis (1=Yes, 2=No)
    DIQ050         = Currently taking insulin (1=Yes, 2=No)
    DIQ070         = Currently taking diabetes pills (1=Yes, 2=No)

    ALL FIVE VARIABLES ARE PROHIBITED AS PREDICTORS IN THE DIABETES MODEL.

    Logic:
    - If ANY criterion is met: Target = 1
    - If NO criterion met AND at least one criterion evaluable: Target = 0
    - If ALL criteria are missing: Target = NaN

    Note on NaN handling:
    - A participant without an HbA1c measurement could still be positive via
      DIQ010=1 (diagnosis). We do NOT assume non-fasting = non-diabetic.
    """
    # HbA1c criterion
    hba1c_pos = df["hba1c"] >= 6.5      # NaN → False (safe for |= logic below)

    # Fasting glucose criterion (subsample; many are NaN)
    fg_pos = df["fasting_glucose"] >= 126  # NaN → False

    # Questionnaire criteria
    diq010_pos = df["DIQ010"] == 1       # NaN → False
    diq050_pos = df["DIQ050"] == 1
    diq070_pos = df["DIQ070"] == 1

    any_positive = hba1c_pos | fg_pos | diq010_pos | diq050_pos | diq070_pos

    # Evaluability: at least one criterion has a non-NaN value
    evaluable_cols = [
        df["hba1c"].notna(),
        df["fasting_glucose"].notna(),
        df["DIQ010"].notna(),
        df["DIQ050"].notna(),
        df["DIQ070"].notna(),
    ]
    any_evaluable = evaluable_cols[0]
    for e in evaluable_cols[1:]:
        any_evaluable = any_evaluable | e

    target = pd.Series(np.nan, index=df.index, name="target_diabetes", dtype="float64")
    target[any_positive] = 1.0
    target[~any_positive & any_evaluable] = 0.0
    # rows where nothing is evaluable remain NaN

    return target


def _build_target_hypertension(df: pd.DataFrame) -> pd.Series:
    """
    TARGET 3: Hypertension — JNC7 Definition
    =========================================
    Definition (from MODEL_V2_DESIGN.md Section 2.3):
        Target_HTN = 1 iff (mean_sbp >= 140) OR (mean_dbp >= 90) OR (BPQ020 = 1)

    This is the JNC 7 hypertension definition:
        - SBP >= 140 mmHg or DBP >= 90 mmHg (measured mean)
        - OR prior physician diagnosis (BPQ020=1)

    This is NOT limited to Stage-2 hypertension; it includes all JNC7
    hypertension (Stage 1 and Stage 2 by that classification).
    It is also NOT the 2017 ACC/AHA definition (which uses SBP >= 130).

    mean_sbp, mean_dbp, BPQ020 ARE PROHIBITED AS PREDICTORS IN THE HTN MODEL.

    Logic:
    - Any criterion met: Target = 1
    - No criterion met AND at least one evaluable: Target = 0
    - All missing: Target = NaN
    """
    sbp_pos   = df["mean_sbp"] >= 140       # NaN → False
    dbp_pos   = df["mean_dbp"] >= 90        # NaN → False
    diag_pos  = df["BPQ020"] == 1           # NaN → False

    any_positive = sbp_pos | dbp_pos | diag_pos

    sbp_eval  = df["mean_sbp"].notna()
    dbp_eval  = df["mean_dbp"].notna()
    diag_eval = df["BPQ020"].notna()
    any_evaluable = sbp_eval | dbp_eval | diag_eval

    target = pd.Series(np.nan, index=df.index, name="target_hypertension", dtype="float64")
    target[any_positive] = 1.0
    target[~any_positive & any_evaluable] = 0.0

    return target


def construct_targets(
    merged: pd.DataFrame,
    audit_dir: Path = AUDIT_DIR,
) -> pd.DataFrame:
    """
    Construct all three approved targets and append them to the merged DataFrame.

    Parameters
    ----------
    merged    : The harmonised NHANES adult DataFrame (from nhanes_harmonizer)
    audit_dir : Directory for writing the target construction report

    Returns
    -------
    DataFrame with three new columns: target_cvd, target_diabetes, target_hypertension
    """
    audit_dir.mkdir(parents=True, exist_ok=True)

    df = merged.copy()
    n_total = len(df)

    # Build targets
    df["target_cvd"]          = _build_target_cvd(df)
    df["target_diabetes"]     = _build_target_diabetes(df)
    df["target_hypertension"] = _build_target_hypertension(df)

    # -----------------------------------------------------------------------
    # Compute target statistics for the report
    # -----------------------------------------------------------------------
    report: dict = {
        "n_adults_total": n_total,
        "targets": {},
    }

    for tname, source_vars, definition in [
        (
            "target_cvd",
            ["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"],
            "MCQ160B=1 OR MCQ160C=1 OR MCQ160D=1 OR MCQ160E=1 OR MCQ160F=1",
        ),
        (
            "target_diabetes",
            ["hba1c", "fasting_glucose", "DIQ010", "DIQ050", "DIQ070"],
            "hba1c>=6.5 OR fasting_glucose>=126 OR DIQ010=1 OR DIQ050=1 OR DIQ070=1 (ADA criteria)",
        ),
        (
            "target_hypertension",
            ["mean_sbp", "mean_dbp", "BPQ020"],
            "mean_sbp>=140 OR mean_dbp>=90 OR BPQ020=1 (JNC7 definition)",
        ),
    ]:
        col = df[tname]
        n_pos  = int((col == 1).sum())
        n_neg  = int((col == 0).sum())
        n_nan  = int(col.isna().sum())
        n_eval = n_pos + n_neg
        prev   = round(100.0 * n_pos / n_eval, 2) if n_eval > 0 else None

        report["targets"][tname] = {
            "definition": definition,
            "source_variables": source_vars,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "n_undeterminable": n_nan,
            "n_evaluable": n_eval,
            "prevalence_pct": prev,
        }

        log.info(
            "Target %s: Positive=%d (%.1f%%), Negative=%d, Missing=%d",
            tname, n_pos, prev or 0.0, n_neg, n_nan,
        )

    # Write report
    out_path = audit_dir / "nhanes_target_construction.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log.info("Target construction report written: %s", out_path)

    return df
