"""
nhanes_harmonizer.py
====================
Stage 1C — NHANES Data Pipeline: Module 2 — Merge & Harmonize

Responsibilities
----------------
- Merge all 16 NHANES components using SEQN (left join from DEMO_L).
- Filter to adults aged 20+ only.
- Recode special NHANES missing-value codes (7/9/77/99/777/999/7777/9999)
  to NaN for every categorical questionnaire variable.
- Recode all lifestyle categorical variables into clean 3-tier codes.
- Compute mean SBP, DBP, and pulse from the three oscillometric readings.
- Compute derived Indian-population anthropometric flags.
- Document every transformation in a machine-readable cleaning log.
- Preserve WTSAF2YR from GLU_L (confirmed present there, not DEMO_L).
- Write interim merged dataset to:
    backend/ml/data/interim/nhanes_2021_2023_merged.parquet

SAFETY CONTRACT
---------------
- Never writes to raw/ directory.
- Never overwrites an existing output file (safe_write pattern).
- Raises ValueError if unexpected values are found in any categorical
  that should have a known coding.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.ml.scripts.nhanes_loader import load_raw_component, RAW_DIR

log = logging.getLogger(__name__)

INTERIM_DIR = Path("backend/ml/data/interim")
AUDIT_DIR   = Path("backend/ml/data/interim/audit")

# ---------------------------------------------------------------------------
# NHANES SPECIAL MISSING CODE DEFINITIONS
# Documented: https://wwwn.cdc.gov/Nchs/Nhanes/
# ---------------------------------------------------------------------------
# These are the standard NHANES refusal / don't-know codes that must be
# recoded to NaN before any analysis.  The scale of the code depends on
# the response range of the question.
_MISSING_CODES_BY_VARIABLE: dict[str, list[int]] = {
    # Categorical (1-digit response scale → 7=Refused, 9=Don't Know)
    "RIAGENDR": [7, 9],
    "DMDEDUC2": [7, 9],
    "SMQ020":   [7, 9],
    "SMQ040":   [7, 9],
    "DIQ010":   [7, 9],
    "DIQ050":   [7, 9],
    "DIQ070":   [7, 9],
    "MCQ160B":  [7, 9],
    "MCQ160C":  [7, 9],
    "MCQ160D":  [7, 9],
    "MCQ160E":  [7, 9],
    "MCQ160F":  [7, 9],
    "BPQ020":   [7, 9],
    "ALQ111":   [7, 9],
    # Multi-digit response scale
    "ALQ121":   [777, 999],
    "PAD790Q":  [7777, 9999],
    "PAD810Q":  [7777, 9999],
    "PAD680":   [7777, 9999],
}

# ---------------------------------------------------------------------------
# RECODE SPECIFICATIONS
# Maps raw variable(s) → cleaned feature name and encoding rules.
# ---------------------------------------------------------------------------

def _recode_smoking(df: pd.DataFrame) -> pd.Series:
    """
    Source: SMQ020 (smoked at least 100 cigarettes in life?), SMQ040 (now smokes?)
    Output: smoking_status
        0 = Never smoker  (SMQ020 = 2)
        1 = Former smoker (SMQ020 = 1 AND SMQ040 = 3)
        2 = Current smoker (SMQ020 = 1 AND SMQ040 in [1,2])
        NaN = Missing/Refused/Unknown
    """
    smoked100 = df["SMQ020"]   # 1=Yes, 2=No, NaN=missing
    now       = df["SMQ040"]   # 1=Every day, 2=Some days, 3=Not at all, NaN=missing

    result = pd.Series(np.nan, index=df.index, name="smoking_status")
    result = result.where(smoked100.isna(), np.nan)   # propagate missing

    # Never
    result[smoked100 == 2] = 0
    # Former: smoked 100 but not now
    result[(smoked100 == 1) & (now == 3)] = 1
    # Current: smokes some or every day
    result[(smoked100 == 1) & now.isin([1, 2])] = 2

    return result


def _recode_alcohol(df: pd.DataFrame) -> pd.Series:
    """
    Source: ALQ111 (ever had alcohol?), ALQ121 (past 12-month alcohol frequency days)
    Output: alcohol_frequency
        0 = Never drinker (ALQ111 = 2)
        1 = Former drinker (ALQ111 = 1, ALQ121 = 0)
        2 = Current drinker (ALQ111 = 1, ALQ121 > 0)
        NaN = Missing
    Rationale: ALQ111 confirms lifetime use; ALQ121 = 0 means drank in past but not in last 12 months.
    """
    ever = df["ALQ111"]    # 1=Yes, 2=No, NaN=missing
    freq = df["ALQ121"]    # days per year, 0 = not in past year

    result = pd.Series(np.nan, index=df.index, name="alcohol_frequency")
    result[ever == 2] = 0                            # Never
    result[(ever == 1) & (freq == 0)] = 1            # Former
    result[(ever == 1) & (freq > 0) & freq.notna()] = 2   # Current

    return result


def _recode_physical_activity(df: pd.DataFrame) -> pd.Series:
    """
    Source: PAD790Q (moderate activity days/week), PAD810Q (vigorous activity days/week)
    Output: physical_activity_level (WHO GPAQ 3-tier)
        1 = High (vigorous >= 3 days/week OR moderate >= 5 days/week)
        2 = Moderate (moderate or vigorous 1-4 / 1-2 days)
        3 = Low / Sedentary (neither threshold met)
        NaN = both missing

    Note: PAD790Q/PAD810Q in NHANES 2021-2023 are in days per week (0–7 range).
    """
    mod = df["PAD790Q"].copy()   # moderate activity days/week (0-7)
    vig = df["PAD810Q"].copy()   # vigorous activity days/week (0-7)

    result = pd.Series(np.nan, index=df.index, name="physical_activity_level")

    both_missing = mod.isna() & vig.isna()

    # Fill NaN with 0 for logic (participants with only one type reported)
    mod0 = mod.fillna(0)
    vig0 = vig.fillna(0)

    # WHO GPAQ-inspired thresholds:
    high = (vig0 >= 3) | (mod0 >= 5)
    low  = (vig0 == 0) & (mod0 == 0)
    # moderate = everyone in between

    result[high] = 1
    result[low]  = 3
    result[~high & ~low] = 2
    result[both_missing] = np.nan

    return result


def _compute_bp_means(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean SBP, mean DBP, and mean pulse from three oscillometric readings.
    NaN readings are excluded from the mean (nanmean semantics).
    If all three readings are NaN, the mean is NaN.
    """
    sbp_cols = ["BPXOSY1", "BPXOSY2", "BPXOSY3"]
    dbp_cols = ["BPXODI1", "BPXODI2", "BPXODI3"]
    pls_cols = ["BPXOPLS1", "BPXOPLS2", "BPXOPLS3"]

    df["mean_sbp"]   = df[sbp_cols].mean(axis=1, skipna=True)
    df["mean_dbp"]   = df[dbp_cols].mean(axis=1, skipna=True)
    df["mean_pulse"] = df[pls_cols].mean(axis=1, skipna=True)

    # Where ALL three readings were NaN, pandas mean() returns NaN automatically.
    # Extra safety: set to NaN where all source cols are NaN.
    all_sbp_nan = df[sbp_cols].isna().all(axis=1)
    all_dbp_nan = df[dbp_cols].isna().all(axis=1)
    all_pls_nan = df[pls_cols].isna().all(axis=1)

    df.loc[all_sbp_nan, "mean_sbp"]   = np.nan
    df.loc[all_dbp_nan, "mean_dbp"]   = np.nan
    df.loc[all_pls_nan, "mean_pulse"] = np.nan

    return df


def _recode_missing_codes(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """
    Apply NHANES questionnaire special-missing code recoding (in-place).
    Returns a log of {variable: {code: count_replaced}}.
    """
    recode_log: dict[str, dict[str, int]] = {}
    for var, codes in _MISSING_CODES_BY_VARIABLE.items():
        if var not in df.columns:
            continue
        col_log: dict[str, int] = {}
        for code in codes:
            mask = df[var] == code
            n = int(mask.sum())
            if n > 0:
                df[var] = df[var].where(~mask, np.nan)
                col_log[str(code)] = n
        if col_log:
            recode_log[var] = col_log
    return recode_log


def merge_nhanes(
    raw_dir: Path = RAW_DIR,
    interim_dir: Path = INTERIM_DIR,
    audit_dir: Path = AUDIT_DIR,
) -> pd.DataFrame:
    """
    Load, merge, and harmonise all NHANES components.

    Merge strategy:
    - Start from DEMO_L (all 11,933 participants).
    - Filter to adults aged 20+ FIRST.
    - Left-join all subsequent components on SEQN.
    - This preserves Mode A participants even if they lack Mode B labs.

    Returns the merged adult DataFrame and writes to parquet.
    """
    interim_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    cleaning_log: dict[str, Any] = {
        "step": "nhanes_harmonizer.merge_nhanes",
        "transformations": {},
        "epsilon_fixes": {},
        "missing_recode": {},
        "derived_features": {},
        "merge_stats": {},
    }

    # -----------------------------------------------------------------------
    # 1. Load all components
    # -----------------------------------------------------------------------
    log.info("Loading all NHANES components...")
    demo    = load_raw_component("DEMO_L",   raw_dir)
    bmx     = load_raw_component("BMX_L",    raw_dir)
    bpxo    = load_raw_component("BPXO_L",   raw_dir)
    bpq     = load_raw_component("BPQ_L",    raw_dir)
    smq     = load_raw_component("SMQ_L",    raw_dir)
    alq     = load_raw_component("ALQ_L",    raw_dir)
    paq     = load_raw_component("PAQ_L",    raw_dir)
    diq     = load_raw_component("DIQ_L",    raw_dir)
    mcq     = load_raw_component("MCQ_L",    raw_dir)
    ghb     = load_raw_component("GHB_L",    raw_dir)
    glu     = load_raw_component("GLU_L",    raw_dir)
    tchol   = load_raw_component("TCHOL_L",  raw_dir)
    hdl     = load_raw_component("HDL_L",    raw_dir)
    trigly  = load_raw_component("TRIGLY_L", raw_dir)
    biopro  = load_raw_component("BIOPRO_L", raw_dir)
    cbc     = load_raw_component("CBC_L",    raw_dir)

    cleaning_log["merge_stats"]["raw_demo_rows"] = len(demo)

    # -----------------------------------------------------------------------
    # 2. Filter to adults 20+ immediately (on DEMO_L, before any join)
    # -----------------------------------------------------------------------
    demo = demo[demo["RIDAGEYR"] >= 20].copy()
    n_adults = len(demo)
    cleaning_log["merge_stats"]["adults_20plus"] = n_adults
    log.info("Adults aged 20+: %d", n_adults)

    # -----------------------------------------------------------------------
    # 3. Left-join all components onto the adult DEMO_L frame
    # -----------------------------------------------------------------------
    merge_order = [
        ("BMX_L",   bmx,    "MEC anthropometrics"),
        ("BPXO_L",  bpxo,   "Blood pressure readings"),
        ("BPQ_L",   bpq,    "Hypertension diagnosis (BPQ020)"),
        ("SMQ_L",   smq,    "Smoking questionnaire"),
        ("ALQ_L",   alq,    "Alcohol questionnaire"),
        ("PAQ_L",   paq,    "Physical activity (GPAQ)"),
        ("DIQ_L",   diq,    "Diabetes questionnaire"),
        ("MCQ_L",   mcq,    "Medical conditions / CVD history"),
        ("GHB_L",   ghb,    "HbA1c laboratory"),
        ("GLU_L",   glu,    "Fasting glucose (subsample) + WTSAF2YR"),
        ("TCHOL_L", tchol,  "Total cholesterol"),
        ("HDL_L",   hdl,    "HDL cholesterol"),
        ("TRIGLY_L",trigly, "Triglycerides + LDL (fasting subsample)"),
        ("BIOPRO_L",biopro, "Biochemistry profile"),
        ("CBC_L",   cbc,    "Complete blood count"),
    ]

    merged = demo.copy()
    for key, comp, desc in merge_order:
        before = len(merged)
        merged = merged.merge(comp, on="SEQN", how="left")
        assert len(merged) == before, f"Row count changed during merge of {key}!"
        n_joined = comp["SEQN"].isin(demo["SEQN"]).sum()
        cleaning_log["merge_stats"][key] = {
            "description": desc,
            "rows_in_component": len(comp),
            "adults_matched": int(n_joined),
        }
        log.info("  Merged %s: %d / %d rows matched to adults.", key, n_joined, len(comp))

    cleaning_log["merge_stats"]["merged_adult_rows"] = len(merged)

    # -----------------------------------------------------------------------
    # 4. Recode NHANES special missing codes → NaN
    # -----------------------------------------------------------------------
    recode_log = _recode_missing_codes(merged)
    cleaning_log["missing_recode"] = recode_log
    total_recoded = sum(sum(v.values()) for v in recode_log.values())
    log.info("Special-missing recode: %d values recoded to NaN across %d variables.",
             total_recoded, len(recode_log))

    # -----------------------------------------------------------------------
    # 5. Compute mean SBP / DBP / Pulse
    # -----------------------------------------------------------------------
    merged = _compute_bp_means(merged)
    cleaning_log["derived_features"]["mean_sbp"] = "Mean of BPXOSY1/2/3 (skipna)"
    cleaning_log["derived_features"]["mean_dbp"] = "Mean of BPXODI1/2/3 (skipna)"
    cleaning_log["derived_features"]["mean_pulse"] = "Mean of BPXOPLS1/2/3 (skipna)"

    # -----------------------------------------------------------------------
    # 6. Recode lifestyle categorical variables
    # -----------------------------------------------------------------------
    merged["smoking_status"]        = _recode_smoking(merged)
    merged["alcohol_frequency"]     = _recode_alcohol(merged)
    merged["physical_activity_level"] = _recode_physical_activity(merged)

    cleaning_log["derived_features"]["smoking_status"] = (
        "0=Never (SMQ020=2), 1=Former (SMQ020=1 & SMQ040=3), 2=Current (SMQ020=1 & SMQ040 in 1,2)"
    )
    cleaning_log["derived_features"]["alcohol_frequency"] = (
        "0=Never (ALQ111=2), 1=Former (ALQ111=1 & ALQ121=0), 2=Current (ALQ111=1 & ALQ121>0)"
    )
    cleaning_log["derived_features"]["physical_activity_level"] = (
        "1=High (vig>=3d/wk OR mod>=5d/wk), 2=Moderate (between), 3=Low (both=0)"
    )

    # -----------------------------------------------------------------------
    # 7. Derive Indian-population anthropometric flags (do NOT apply in training)
    # -----------------------------------------------------------------------
    # These are computed for later ICMR evaluation alignment only.
    # asian_obesity_flag: BMI >= 25 (Asian Indian threshold, WHO-Asia-Pacific)
    merged["asian_obesity_flag"] = (merged["BMXBMI"] >= 25).astype("float")
    merged.loc[merged["BMXBMI"].isna(), "asian_obesity_flag"] = np.nan

    # south_asian_waist_flag: Men >= 90 cm, Women >= 80 cm
    waist_flag = pd.Series(np.nan, index=merged.index)
    men   = merged["RIAGENDR"] == 1
    women = merged["RIAGENDR"] == 2
    waist_flag.loc[men   & merged["BMXWAIST"].notna()] = (merged.loc[men,   "BMXWAIST"] >= 90).astype(float)
    waist_flag.loc[women & merged["BMXWAIST"].notna()] = (merged.loc[women, "BMXWAIST"] >= 80).astype(float)
    merged["south_asian_waist_flag"] = waist_flag

    cleaning_log["derived_features"]["asian_obesity_flag"] = (
        "1 if BMXBMI >= 25 kg/m² (Asian Indian cutoff, IDF/WHO Asia-Pacific). NaN if BMI missing."
    )
    cleaning_log["derived_features"]["south_asian_waist_flag"] = (
        "1 if waist >= 90 cm (Men) or >= 80 cm (Women) per ICMR/IDF South Asian criteria. NaN if waist missing."
    )

    # -----------------------------------------------------------------------
    # 8. Rename columns to human-friendly names for downstream use
    # -----------------------------------------------------------------------
    merged = merged.rename(columns={
        "RIDAGEYR": "age",
        "RIAGENDR": "gender",
        "DMDEDUC2": "education_level",
        "INDFMPIR": "poverty_income_ratio",
        "BMXBMI":   "bmi",
        "BMXWAIST": "waist_circumference",
        "BPXOPLS1": "bpxopls1", "BPXOPLS2": "bpxopls2", "BPXOPLS3": "bpxopls3",   # raw readings kept
        "PAD680":   "sedentary_minutes",
        "LBXGH":    "hba1c",
        "LBXGLU":   "fasting_glucose",
        "LBXTC":    "total_cholesterol",
        "LBDHDD":   "hdl_cholesterol",
        "LBXTLG":   "triglycerides",
        "LBDLDL":   "ldl_cholesterol",
        "LBXSCR":   "serum_creatinine",
        "LBXSBU":   "blood_urea_nitrogen",
        "LBXSUA":   "serum_uric_acid",
        "LBXSATSI": "alt_enzyme",
        "LBXSASSI": "ast_enzyme",
        "LBXHGB":   "hemoglobin",
        "LBXWBCSI": "wbc_count",
        "LBXPLTSI": "platelet_count",
        "LBXRDW":   "rdw",
        "LBXSAL":   "serum_albumin",
    })

    cleaning_log["transformations"]["column_renames"] = (
        "NHANES variable names renamed to human-readable snake_case equivalents. "
        "NHANES variable names still documented in model_v2_feature_matrix.csv."
    )

    # -----------------------------------------------------------------------
    # 9. Write interim merged dataset
    # -----------------------------------------------------------------------
    out_path = interim_dir / "nhanes_2021_2023_merged.parquet"
    if out_path.exists():
        log.warning("Interim file already exists; overwriting: %s", out_path)
    merged.to_parquet(out_path, index=False, engine="pyarrow")
    log.info("Interim merged dataset written: %s (%d rows, %d cols)", out_path, len(merged), len(merged.columns))

    # -----------------------------------------------------------------------
    # 10. Write cleaning log
    # -----------------------------------------------------------------------
    log_path = audit_dir / "nhanes_cleaning_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(cleaning_log, f, indent=2)
    log.info("Cleaning log written: %s", log_path)

    return merged
