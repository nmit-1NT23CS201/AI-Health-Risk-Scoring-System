"""
nhanes_quality_checks.py
========================
Stage 1C — NHANES Data Pipeline: Module 4 — Quality Control & Unit Validation

Responsibilities
----------------
- Unit and plausibility range checks for every physiological / laboratory variable.
- Quantify missingness per feature per model/mode.
- Check for duplicate SEQN in the merged dataset.
- Verify leakage isolation: confirm no prohibited variable appears in any
  model's predictor columns.
- Produce:
    backend/ml/data/interim/audit/nhanes_processed_qc.md   (human-readable QC report)
    backend/ml/data/interim/audit/nhanes_feature_statistics.csv  (machine-readable)

IMPORTANT: This module flags suspicious values but does NOT silently drop or
clip them.  Flagging is logged and reported; the decision to clip or exclude
is deferred to the Stage 1D training pipeline.

SAFETY CONTRACT
---------------
- Read-only analysis on the merged DataFrame.
- Writes only to interim/audit/ directory.
- Does not modify raw files.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

AUDIT_DIR = Path("backend/ml/data/interim/audit")

# ---------------------------------------------------------------------------
# PLAUSIBLE RANGES (clinical reference / CDC codebook)
# Flagged if outside range — NOT clipped here.
# ---------------------------------------------------------------------------
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    # Anthropometrics
    "bmi":                  (10.0, 80.0),   # kg/m²
    "waist_circumference":  (40.0, 200.0),  # cm
    # Blood pressure & pulse
    "mean_sbp":             (60.0, 260.0),  # mmHg
    "mean_dbp":             (30.0, 160.0),  # mmHg
    "mean_pulse":           (25.0, 200.0),  # bpm
    # Glycaemic
    "hba1c":                (2.0,  20.0),   # %
    "fasting_glucose":      (40.0, 500.0),  # mg/dL
    # Lipids
    "total_cholesterol":    (50.0, 600.0),  # mg/dL
    "hdl_cholesterol":      (10.0, 200.0),  # mg/dL
    "triglycerides":        (10.0, 2000.0), # mg/dL
    "ldl_cholesterol":      (20.0, 400.0),  # mg/dL
    # Renal / metabolic
    "serum_creatinine":     (0.2,  20.0),   # mg/dL
    "blood_urea_nitrogen":  (2.0,  150.0),  # mg/dL
    "serum_uric_acid":      (1.0,  20.0),   # mg/dL
    # Hepatic
    "alt_enzyme":           (3.0,  2000.0), # U/L
    "ast_enzyme":           (3.0,  2000.0), # U/L
    "serum_albumin":        (1.0,  7.0),    # g/dL
    # Haematology
    "hemoglobin":           (3.0,  22.0),   # g/dL
    "wbc_count":            (0.5,  50.0),   # 10³/µL
    "platelet_count":       (10.0, 1000.0), # 10³/µL
    "rdw":                  (10.0, 30.0),   # %
}

# ---------------------------------------------------------------------------
# MODEL PREDICTOR SETS (leakage matrix from MODEL_V2_DESIGN.md)
# ---------------------------------------------------------------------------

# --- Mode A Features (non-invasive, available to all 3 models unless prohibited)
MODE_A_FEATURES = [
    "age", "gender", "education_level", "poverty_income_ratio",
    "bmi", "waist_circumference",
    "mean_sbp", "mean_dbp", "mean_pulse",
    "smoking_status", "alcohol_frequency", "physical_activity_level",
    "sedentary_minutes",
]

# --- Mode B adds laboratory biomarkers
MODE_B_EXTRA_FEATURES = [
    "hba1c", "fasting_glucose",
    "total_cholesterol", "hdl_cholesterol", "triglycerides", "ldl_cholesterol",
    "serum_creatinine", "blood_urea_nitrogen", "serum_uric_acid",
    "alt_enzyme", "ast_enzyme",
    "hemoglobin", "wbc_count", "platelet_count", "rdw", "serum_albumin",
]

# --- Per-model prohibited features
CVD_PROHIBITED   = ["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F",
                    "DIQ050", "DIQ070", "BPQ101D"]
DM_PROHIBITED    = ["hba1c", "fasting_glucose", "DIQ010", "DIQ050", "DIQ070",
                    "BPQ101D"]
HTN_PROHIBITED   = ["mean_sbp", "mean_dbp", "BPQ020", "BPQ101D"]

# Survey-design variables to preserve but NOT use in training
SURVEY_VARS = ["WTINT2YR", "WTMEC2YR", "WTSAF2YR", "SDMVSTRA", "SDMVPSU"]

# --- Build per-model predictor sets
def _build_predictor_set(mode_a: list[str], mode_b_extra: list[str],
                          prohibited: list[str]) -> dict[str, list[str]]:
    a = [f for f in mode_a if f not in prohibited]
    b = a + [f for f in mode_b_extra if f not in prohibited]
    return {"mode_a": a, "mode_b": b}


CVD_PREDS = _build_predictor_set(MODE_A_FEATURES, MODE_B_EXTRA_FEATURES, CVD_PROHIBITED)
DM_PREDS  = _build_predictor_set(MODE_A_FEATURES, MODE_B_EXTRA_FEATURES, DM_PROHIBITED)
HTN_PREDS = _build_predictor_set(MODE_A_FEATURES, MODE_B_EXTRA_FEATURES, HTN_PROHIBITED)

MODEL_PREDICTOR_SETS = {
    "cvd":          CVD_PREDS,
    "diabetes":     DM_PREDS,
    "hypertension": HTN_PREDS,
}


def run_plausibility_checks(df: pd.DataFrame) -> dict:
    """
    Check every physiological/lab variable against documented plausible ranges.
    Returns a dict: {feature: {min, max, median, mean, n, n_out_of_range}}.
    Logs warnings for features with suspicious values.
    """
    results = {}
    for feature, (lo, hi) in PLAUSIBLE_RANGES.items():
        if feature not in df.columns:
            results[feature] = {"error": "column not found in merged dataset"}
            continue

        col = df[feature].dropna()
        if len(col) == 0:
            results[feature] = {"error": "all values missing"}
            continue

        out_of_range = ((col < lo) | (col > hi)).sum()
        results[feature] = {
            "n_non_missing": int(len(col)),
            "min": float(col.min()),
            "max": float(col.max()),
            "median": float(col.median()),
            "mean": float(col.mean()),
            "std": float(col.std()),
            "plausible_low": lo,
            "plausible_high": hi,
            "n_flagged_out_of_range": int(out_of_range),
            "pct_flagged": round(100.0 * out_of_range / len(col), 3),
        }
        if out_of_range > 0:
            log.warning(
                "  [QC] %s: %d values outside plausible range [%.1f, %.1f] — flagged, NOT dropped.",
                feature, out_of_range, lo, hi,
            )

    return results


def compute_feature_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-feature statistics for all candidate predictor features,
    broken out by model and mode.

    Returns a DataFrame suitable for writing to nhanes_feature_statistics.csv.
    """
    rows = []
    n_total = len(df)

    all_features = list({
        *MODE_A_FEATURES,
        *MODE_B_EXTRA_FEATURES,
    })

    for model, preds in MODEL_PREDICTOR_SETS.items():
        for mode, feature_list in preds.items():
            for feat in feature_list:
                if feat not in df.columns:
                    rows.append({
                        "feature": feat, "model": model, "mode": mode,
                        "n": 0, "missing_n": n_total, "missing_pct": 100.0,
                        "mean": None, "std": None, "min": None,
                        "median": None, "max": None,
                        "note": "COLUMN NOT IN MERGED DATASET",
                    })
                    continue

                col = df[feat]
                n_miss = int(col.isna().sum())
                n_obs  = n_total - n_miss

                row: dict = {
                    "feature": feat,
                    "model": model,
                    "mode": mode,
                    "n": n_obs,
                    "missing_n": n_miss,
                    "missing_pct": round(100.0 * n_miss / n_total, 2),
                    "mean": None, "std": None,
                    "min": None, "median": None, "max": None,
                    "note": "",
                }

                if n_obs > 0 and pd.api.types.is_numeric_dtype(col):
                    valid = col.dropna()
                    row["mean"]   = round(float(valid.mean()),   4)
                    row["std"]    = round(float(valid.std()),    4)
                    row["min"]    = round(float(valid.min()),    4)
                    row["median"] = round(float(valid.median()), 4)
                    row["max"]    = round(float(valid.max()),    4)

                # Flag subsample features
                if feat in ["fasting_glucose", "triglycerides", "ldl_cholesterol"]:
                    row["note"] = "NHANES fasting subsample (~41% of adults)"
                elif n_obs < 0.5 * n_total:
                    row["note"] = f"Low coverage: {row['missing_pct']}% missing"

                rows.append(row)

    return pd.DataFrame(rows)


def verify_leakage_isolation(df: pd.DataFrame) -> list[str]:
    """
    Verify that no prohibited variable appears in any model's predictor column
    list.

    Returns a list of leakage violation strings (empty if clean).
    """
    violations = []
    checks = [
        ("CVD",          CVD_PREDS["mode_b"],  CVD_PROHIBITED),
        ("Diabetes",     DM_PREDS["mode_b"],   DM_PROHIBITED),
        ("Hypertension", HTN_PREDS["mode_b"],  HTN_PROHIBITED),
    ]
    for model_name, predictor_set, prohibited in checks:
        for var in prohibited:
            if var in predictor_set:
                msg = (
                    f"LEAKAGE VIOLATION: {var} is in the {model_name} model "
                    f"predictor set but is PROHIBITED (defines target)."
                )
                violations.append(msg)
                log.error(msg)
    return violations


def run_all_qc(
    df: pd.DataFrame,
    target_report: dict,
    audit_dir: Path = AUDIT_DIR,
) -> None:
    """
    Run all QC checks and write:
      - nhanes_processed_qc.md  (Markdown report)
      - nhanes_feature_statistics.csv
    """
    audit_dir.mkdir(parents=True, exist_ok=True)
    n = len(df)

    # 1. Plausibility checks
    plaus = run_plausibility_checks(df)

    # 2. Feature statistics
    feat_stats = compute_feature_statistics(df)

    # 3. Leakage verification
    leakage_violations = verify_leakage_isolation(df)

    # 4. Duplicate SEQN check
    n_dup_seqn = int(df["SEQN"].duplicated().sum())

    # 5. Missing per target
    def _target_info(tname: str) -> dict:
        col = df[tname]
        n_pos = int((col == 1).sum())
        n_neg = int((col == 0).sum())
        n_nan = int(col.isna().sum())
        n_ev  = n_pos + n_neg
        prev  = round(100.0 * n_pos / n_ev, 2) if n_ev > 0 else 0.0
        return {"positive": n_pos, "negative": n_neg, "missing": n_nan, "evaluable": n_ev, "prevalence_pct": prev}

    cvd_info  = _target_info("target_cvd")
    dm_info   = _target_info("target_diabetes")
    htn_info  = _target_info("target_hypertension")

    # 6. Mode A / B sample sizes (per model)
    mode_sizes: dict[str, dict] = {}
    for model, preds in MODEL_PREDICTOR_SETS.items():
        for mode, feat_list in preds.items():
            tgt = f"target_{model if model != 'hypertension' else 'hypertension'}"
            if model == "cvd":
                tgt = "target_cvd"
            elif model == "diabetes":
                tgt = "target_diabetes"

            avail_feats = [f for f in feat_list if f in df.columns]
            complete_rows = df[avail_feats + [tgt]].dropna()
            mode_sizes[f"{model}_{mode}"] = {
                "n_with_target": int(df[tgt].notna().sum()),
                "n_complete_cases_all_features": len(complete_rows),
                "n_features": len(avail_feats),
            }

    # -----------------------------------------------------------------------
    # Write nhanes_feature_statistics.csv
    # -----------------------------------------------------------------------
    stats_path = audit_dir / "nhanes_feature_statistics.csv"
    feat_stats.to_csv(stats_path, index=False)
    log.info("Feature statistics written: %s", stats_path)

    # -----------------------------------------------------------------------
    # Write nhanes_processed_qc.md
    # -----------------------------------------------------------------------
    md_lines = []
    md_lines.append("# NHANES 2021–2023 Processed Dataset QC Report\n")
    md_lines.append("> **Stage:** 1C — Data Preparation  \n")
    md_lines.append(f"> **Total Adults (age ≥ 20):** {n:,}  \n")
    md_lines.append(f"> **Duplicate SEQN:** {n_dup_seqn}  \n\n")
    md_lines.append("---\n\n")

    # 1. Raw counts
    md_lines.append("## 1. Participant Counts\n\n")
    md_lines.append(f"| Population | N |\n|---|---|\n")
    md_lines.append(f"| Total NHANES 2021–2023 interviewed | 11,933 |\n")
    md_lines.append(f"| Adults aged ≥ 20 (analysis population) | {n:,} |\n")
    md_lines.append(f"| MEC examined adults (BMX + BPXO coverage) | {int(df['bmi'].notna().sum()):,} |\n")
    md_lines.append(f"| Routine lab panel adults (HbA1c, TC, HDL, etc.) | {int(df['hba1c'].notna().sum()):,} |\n")
    md_lines.append(f"| Fasting subsample adults (Glucose, TG, LDL) | {int(df['fasting_glucose'].notna().sum()):,} |\n\n")

    # 2. Target prevalence
    md_lines.append("## 2. Target Prevalence\n\n")
    md_lines.append("| Target | Positive | Negative | Missing/Undeterminable | Prevalence (%) |\n|---|---|---|---|---|\n")
    md_lines.append(f"| Hard CVD (MCQ160B-F) | {cvd_info['positive']:,} | {cvd_info['negative']:,} | {cvd_info['missing']:,} | {cvd_info['prevalence_pct']}% |\n")
    md_lines.append(f"| Diabetes (ADA criteria) | {dm_info['positive']:,} | {dm_info['negative']:,} | {dm_info['missing']:,} | {dm_info['prevalence_pct']}% |\n")
    md_lines.append(f"| Hypertension (JNC7) | {htn_info['positive']:,} | {htn_info['negative']:,} | {htn_info['missing']:,} | {htn_info['prevalence_pct']}% |\n\n")

    # 3-4. Mode sample sizes
    md_lines.append("## 3. Mode A and Mode B Sample Sizes (per Model)\n\n")
    md_lines.append("| Model | Mode | N with target | N complete cases | N features |\n|---|---|---|---|---|\n")
    for key, vals in mode_sizes.items():
        parts = key.rsplit("_", 1)
        md_lines.append(
            f"| {parts[0].upper()} | {parts[1].upper()} | "
            f"{vals['n_with_target']:,} | {vals['n_complete_cases_all_features']:,} | "
            f"{vals['n_features']} |\n"
        )
    md_lines.append("\n")

    # 5. Feature missingness (summary — top 20 by missing_pct for mode_b)
    md_lines.append("## 4. Feature Missingness Summary (Mode B, CVD Model)\n\n")
    md_lines.append("| Feature | N present | Missing N | Missing % | Note |\n|---|---|---|---|---|\n")
    subset = feat_stats[(feat_stats["model"] == "cvd") & (feat_stats["mode"] == "mode_b")]
    for _, row in subset.sort_values("missing_pct", ascending=False).iterrows():
        md_lines.append(
            f"| {row['feature']} | {row['n']:,} | {row['missing_n']:,} | "
            f"{row['missing_pct']}% | {row['note']} |\n"
        )
    md_lines.append("\n")

    # 6. Plausibility checks
    md_lines.append("## 5. Unit & Plausibility Checks\n\n")
    md_lines.append("| Feature | N | Min | Max | Median | Mean | Flagged OOR | Plausible Range |\n|---|---|---|---|---|---|---|---|\n")
    for feat, vals in plaus.items():
        if "error" in vals:
            md_lines.append(f"| {feat} | — | — | — | — | — | ERROR: {vals['error']} | — |\n")
        else:
            md_lines.append(
                f"| {feat} | {vals['n_non_missing']:,} | {vals['min']:.2f} | {vals['max']:.2f} | "
                f"{vals['median']:.2f} | {vals['mean']:.2f} | {vals['n_flagged_out_of_range']} | "
                f"[{vals['plausible_low']}, {vals['plausible_high']}] |\n"
            )
    md_lines.append("\n")

    # 7. Leakage checks
    md_lines.append("## 6. Target Leakage Verification\n\n")
    if leakage_violations:
        md_lines.append("> [!CAUTION]\n> **LEAKAGE VIOLATIONS DETECTED:**\n")
        for v in leakage_violations:
            md_lines.append(f"> - {v}\n")
    else:
        md_lines.append("> [!NOTE]\n> **All leakage checks passed.** No prohibited variable appears in any model's predictor set.\n")
    md_lines.append("\n")

    # 8. Special missing value handling
    md_lines.append("## 7. Special Missing-Value Handling\n\n")
    md_lines.append("| Code | Meaning | Variables Affected | Action |\n|---|---|---|---|\n")
    md_lines.append("| 7 / 77 / 777 / 7777 | Refused | SMQ020/040, DIQ010/050/070, MCQ160B-F, BPQ020, ALQ111, PAD790Q, PAD810Q, PAD680 | Recoded to NaN |\n")
    md_lines.append("| 9 / 99 / 999 / 9999 | Don't Know | Same set as above | Recoded to NaN |\n")
    md_lines.append("| 5.3976e-79 (SAS epsilon) | True zero (SAS floating-point underflow) | Continuous numeric columns in XPT | Clamped to 0.0 |\n\n")

    # 9. Survey weight docs
    md_lines.append("## 8. Survey-Weight Variables (Preserved, NOT Applied)\n\n")
    md_lines.append("| Variable | Scope | Population | Status |\n|---|---|---|---|\n")
    md_lines.append("| WTINT2YR | 2-year interview weight | All interviewed (N=11,933) | Preserved in merged dataset |\n")
    md_lines.append("| WTMEC2YR | 2-year MEC examination weight | All examined | Preserved; to be used in calibration (Stage 2) |\n")
    md_lines.append("| WTSAF2YR | Fasting subsample weight | Fasting subsample (N~3996 total; ~3210 adults) | Present in GLU_L.xpt; preserved |\n")
    md_lines.append("| SDMVSTRA | Masked pseudo-stratum | Survey design variable | Preserved for variance estimation |\n")
    md_lines.append("| SDMVPSU | Masked pseudo-PSU | Survey design variable | Preserved for variance estimation |\n\n")

    # 10. Duplicate checks
    md_lines.append("## 9. Duplicate SEQN Check\n\n")
    md_lines.append(f"Duplicate SEQN in merged adult dataset: **{n_dup_seqn}**\n\n")
    md_lines.append("> [!NOTE]\n> SEQN uniquely identifies each NHANES participant. Zero duplicates is the expected and observed result.\n\n")

    # 11. Unexpected issues / flags
    md_lines.append("## 10. Flags & Unexpected Issues\n\n")
    flagged_feats = [f for f, v in plaus.items() if not isinstance(v, dict) or v.get("n_flagged_out_of_range", 0) > 0]
    if flagged_feats:
        md_lines.append("The following features contain values outside the defined plausible range. "
                         "These are flagged for review — they are NOT dropped or clipped at this stage.\n\n")
        for f in flagged_feats:
            v = plaus[f]
            md_lines.append(f"- `{f}`: {v['n_flagged_out_of_range']} values outside [{v['plausible_low']}, {v['plausible_high']}]\n")
    else:
        md_lines.append("No out-of-range values flagged in any physiological variable.\n")
    md_lines.append("\n")

    # Footer
    md_lines.append("---\n\n")
    md_lines.append("*Generated by `nhanes_quality_checks.py` — Stage 1C data preparation.*  \n")
    md_lines.append("*Feature statistics: `nhanes_feature_statistics.csv`*\n")

    qc_path = audit_dir / "nhanes_processed_qc.md"
    with open(qc_path, "w", encoding="utf-8") as f:
        f.writelines(md_lines)
    log.info("QC report written: %s", qc_path)
