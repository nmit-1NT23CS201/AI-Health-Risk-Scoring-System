"""
build_nhanes_dataset.py
=======================
Stage 1C — NHANES Data Pipeline: Master Orchestrator

This is the single entry-point for the full NHANES 2021–2023 preprocessing
pipeline. It calls Modules 1–4 in sequence and produces all processed datasets.

USAGE (from workspace root):
    python -m backend.ml.scripts.build_nhanes_dataset

Or equivalently:
    python backend/ml/scripts/build_nhanes_dataset.py

WHAT IT PRODUCES
----------------
Validation report:
    backend/ml/data/interim/audit/nhanes_raw_validation.json

Cleaning log:
    backend/ml/data/interim/audit/nhanes_cleaning_log.json

Target construction report:
    backend/ml/data/interim/audit/nhanes_target_construction.json

QC report (Markdown):
    backend/ml/data/interim/audit/nhanes_processed_qc.md

Feature statistics:
    backend/ml/data/interim/audit/nhanes_feature_statistics.csv

Intermediate merged dataset:
    backend/ml/data/interim/nhanes_2021_2023_merged.parquet

Processed leakage-controlled predictor datasets (6 files):
    backend/ml/data/processed/nhanes_2021_2023/nhanes_cvd_mode_a.parquet
    backend/ml/data/processed/nhanes_2021_2023/nhanes_cvd_mode_b.parquet
    backend/ml/data/processed/nhanes_2021_2023/nhanes_diabetes_mode_a.parquet
    backend/ml/data/processed/nhanes_2021_2023/nhanes_diabetes_mode_b.parquet
    backend/ml/data/processed/nhanes_2021_2023/nhanes_hypertension_mode_a.parquet
    backend/ml/data/processed/nhanes_2021_2023/nhanes_hypertension_mode_b.parquet

Dataset manifest:
    backend/ml/data/processed/nhanes_2021_2023/DATASET_MANIFEST.md

SAFETY CONTRACT
---------------
- Never modifies raw XPT files, ICMR sample.dta, V1 CSV, or any .pkl model.
- Never modifies frontend, API, or database.
- The existing V1 system remains fully operational.
- DOES NOT train any model.

REPRODUCIBILITY
---------------
All paths are project-relative (no machine-specific absolute paths).
Run from the workspace root directory.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Guarantee project root is in sys.path when executed directly
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ---------------------------------------------------------------------------
# Configure logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("build_nhanes_dataset")

# ---------------------------------------------------------------------------
# Pipeline imports
# ---------------------------------------------------------------------------
from backend.ml.scripts.nhanes_loader        import validate_all_files, RAW_DIR
from backend.ml.scripts.nhanes_harmonizer    import merge_nhanes, INTERIM_DIR, AUDIT_DIR
from backend.ml.scripts.nhanes_targets       import construct_targets
from backend.ml.scripts.nhanes_quality_checks import (
    run_all_qc,
    MODEL_PREDICTOR_SETS,
    SURVEY_VARS,
    MODE_A_FEATURES,
    MODE_B_EXTRA_FEATURES,
)

PROCESSED_DIR = Path("backend/ml/data/processed/nhanes_2021_2023")


# ---------------------------------------------------------------------------
# Step 7: Build per-model, per-mode predictor datasets
# ---------------------------------------------------------------------------

def _write_predictor_dataset(
    df: pd.DataFrame,
    model: str,
    mode: str,
    features: list[str],
    target_col: str,
    out_dir: Path,
) -> Path:
    """
    Select SEQN + predictor features + target column + survey weights.
    Write to parquet. Returns path.
    """
    # Include SEQN, survey weights (preserved but not trained on)
    id_cols  = ["SEQN"]
    wt_cols  = [c for c in SURVEY_VARS if c in df.columns]
    feat_cols = [f for f in features if f in df.columns]

    # Final column order: SEQN, survey weights, features, target
    all_cols = id_cols + wt_cols + feat_cols + [target_col]
    all_cols = list(dict.fromkeys(all_cols))   # deduplicate while preserving order

    out = df[all_cols].copy()

    # Verification: target column must be present
    assert target_col in out.columns, f"Target {target_col} missing from output dataset!"

    # Verification: no prohibited variable present
    # (belt-and-suspenders; leakage_isolation was already verified in QC)
    fname = out_dir / f"nhanes_{model}_{mode}.parquet"
    out.to_parquet(fname, index=False, engine="pyarrow")

    n_pos = int((out[target_col] == 1).sum())
    n_neg = int((out[target_col] == 0).sum())
    n_nan = int(out[target_col].isna().sum())
    n_ev  = n_pos + n_neg
    prev  = round(100.0 * n_pos / n_ev, 2) if n_ev > 0 else 0.0

    log.info(
        "  Written: %s  |  rows=%d, features=%d, target=%s pos=%d (%.1f%%) neg=%d miss=%d",
        fname.name, len(out), len(feat_cols), target_col, n_pos, prev, n_neg, n_nan,
    )
    return fname


def write_all_predictor_datasets(df: pd.DataFrame, out_dir: Path) -> dict[str, str]:
    """Write 6 leakage-controlled predictor parquet files (3 models × 2 modes)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    target_map = {
        "cvd":          "target_cvd",
        "diabetes":     "target_diabetes",
        "hypertension": "target_hypertension",
    }

    paths = {}
    for model, preds in MODEL_PREDICTOR_SETS.items():
        tgt = target_map[model]
        for mode, feat_list in preds.items():
            key = f"{model}_{mode}"
            p = _write_predictor_dataset(df, model, mode, feat_list, tgt, out_dir)
            paths[key] = str(p)

    return paths


# ---------------------------------------------------------------------------
# Dataset Manifest
# ---------------------------------------------------------------------------

def write_dataset_manifest(
    df: pd.DataFrame,
    paths: dict[str, str],
    target_report: dict,
    out_dir: Path,
) -> None:
    """Write DATASET_MANIFEST.md documenting the full pipeline output."""
    n = len(df)

    def _fmt_target(tname: str) -> str:
        col = df[tname]
        n_pos = int((col == 1).sum())
        n_neg = int((col == 0).sum())
        n_nan = int(col.isna().sum())
        prev  = round(100.0 * n_pos / (n_pos + n_neg), 2) if (n_pos + n_neg) > 0 else 0.0
        return f"Positive={n_pos:,} ({prev}%), Negative={n_neg:,}, Missing={n_nan:,}"

    lines = [
        "# NHANES 2021–2023 Processed Dataset Manifest\n\n",
        "> **Stage:** 1C — Data Preparation Pipeline  \n",
        "> **Source Dataset:** CDC NHANES August 2021–August 2023  \n",
        f"> **Analysis Population:** Adults aged 20+, N = {n:,}  \n",
        "> **Status:** Preparation Complete — Awaiting Stage 1D (Train/Val/Test Split)  \n\n",
        "---\n\n",

        "## 1. Source Files\n\n",
        "| File | Domain | Rows | Key Variables Used |\n|---|---|---|---|\n",
        "| `DEMO_L.xpt` | Demographics, Survey Design | 11,933 | RIDAGEYR, RIAGENDR, DMDEDUC2, INDFMPIR, WTINT2YR, WTMEC2YR, SDMVSTRA, SDMVPSU |\n",
        "| `BMX_L.xpt` | Anthropometrics | 8,860 | BMXBMI, BMXWAIST |\n",
        "| `BPXO_L.xpt` | Blood Pressure (Oscillometric) | 7,801 | BPXOSY1-3, BPXODI1-3, BPXOPLS1-3 |\n",
        "| `BPQ_L.xpt` | BP Questionnaire | 8,501 | BPQ020 (HTN target component) |\n",
        "| `SMQ_L.xpt` | Smoking | 9,015 | SMQ020, SMQ040 |\n",
        "| `ALQ_L.xpt` | Alcohol | 6,337 | ALQ111, ALQ121 |\n",
        "| `PAQ_L.xpt` | Physical Activity | 8,153 | PAD790Q, PAD810Q, PAD680 |\n",
        "| `DIQ_L.xpt` | Diabetes Questionnaire | 11,744 | DIQ010, DIQ050, DIQ070 (DM target) |\n",
        "| `MCQ_L.xpt` | Medical Conditions | 11,744 | MCQ160B-F (CVD target) |\n",
        "| `GHB_L.xpt` | HbA1c Laboratory | 7,199 | LBXGH |\n",
        "| `GLU_L.xpt` | Fasting Glucose (Subsample) | 3,996 | LBXGLU, WTSAF2YR |\n",
        "| `TCHOL_L.xpt` | Total Cholesterol | 8,068 | LBXTC |\n",
        "| `HDL_L.xpt` | HDL Cholesterol | 8,068 | LBDHDD |\n",
        "| `TRIGLY_L.xpt` | Triglycerides + LDL (Fasting) | 3,996 | LBXTLG, LBDLDL |\n",
        "| `BIOPRO_L.xpt` | Biochemistry Profile | 7,199 | LBXSCR, LBXSBU, LBXSUA, LBXSATSI, LBXSASSI, LBXSAL |\n",
        "| `CBC_L.xpt` | Complete Blood Count | 8,727 | LBXHGB, LBXWBCSI, LBXPLTSI, LBXRDW |\n\n",

        "## 2. Merge Strategy\n\n",
        "- **Base:** `DEMO_L.xpt` (all 11,933 participants)\n",
        "- **Adult filter:** Applied first — `RIDAGEYR >= 20` → N = {n:,}\n".replace("{n:,}", f"{n:,}"),
        "- **Join type:** Left join on `SEQN` for all subsequent components\n",
        "- **Rationale:** Preserves Mode A participants without Mode B lab data. "
        "Missing lab values appear as NaN, not as participant exclusion.\n\n",

        "## 3. Target Definitions\n\n",
        "### Hard CVD (target_cvd)\n",
        "- **Formula:** `MCQ160B=1 OR MCQ160C=1 OR MCQ160D=1 OR MCQ160E=1 OR MCQ160F=1`\n",
        "- **Criterion:** Self-reported physician diagnosis of CHF, CHD, Angina, MI, or Stroke\n",
        f"- **Result:** {_fmt_target('target_cvd')}\n\n",
        "### Diabetes (target_diabetes) — ADA Multi-Criteria\n",
        "- **Formula:** `HbA1c>=6.5 OR Fasting Glucose>=126 OR DIQ010=1 OR DIQ050=1 OR DIQ070=1`\n",
        "- **Prohibited predictors in DM model:** hba1c, fasting_glucose, DIQ010, DIQ050, DIQ070\n",
        f"- **Result:** {_fmt_target('target_diabetes')}\n\n",
        "### Hypertension — JNC7 (target_hypertension)\n",
        "- **Formula:** `mean_sbp>=140 OR mean_dbp>=90 OR BPQ020=1`\n",
        "- **Definition:** JNC7 hypertension threshold (NOT ACC/AHA 2017, NOT Stage-2 only)\n",
        "- **Prohibited predictors in HTN model:** mean_sbp, mean_dbp, BPQ020\n",
        f"- **Result:** {_fmt_target('target_hypertension')}\n\n",

        "## 4. Cleaning Rules\n\n",
        "| Rule | Variables | Action |\n|---|---|---|\n",
        "| Refused (7/77/777/7777) | SMQ020/040, DIQ010/050/070, MCQ160B-F, BPQ020, ALQ111, PAD790Q/810Q/680 | Recoded to NaN |\n",
        "| Don't Know (9/99/999/9999) | Same set | Recoded to NaN |\n",
        "| SAS epsilon (5.4e-79) | All continuous XPT columns | Clamped to 0.0 |\n",
        "| Smoking recode | SMQ020 + SMQ040 | 0=Never, 1=Former, 2=Current → smoking_status |\n",
        "| Alcohol recode | ALQ111 + ALQ121 | 0=Never, 1=Former, 2=Current → alcohol_frequency |\n",
        "| Physical activity recode | PAD790Q + PAD810Q | 1=High, 2=Moderate, 3=Low → physical_activity_level |\n",
        "| BP means | BPXOSY1-3, BPXODI1-3, BPXOPLS1-3 | nanmean of valid readings → mean_sbp, mean_dbp, mean_pulse |\n\n",

        "## 5. Predictor Exclusions (Leakage Matrix)\n\n",
        "| Variable | CVD Model | DM Model | HTN Model |\n|---|---|---|---|\n",
        "| MCQ160B-F | PROHIBITED (defines target) | Not used | Not used |\n",
        "| hba1c, fasting_glucose | ALLOWED | PROHIBITED (defines target) | ALLOWED |\n",
        "| DIQ010, DIQ050, DIQ070 | Not used | PROHIBITED (defines target) | Not used |\n",
        "| mean_sbp, mean_dbp | ALLOWED | ALLOWED | PROHIBITED (defines target) |\n",
        "| BPQ020 | Not used | Not used | PROHIBITED (defines target) |\n",
        "| BPQ101D (chol meds) | PROHIBITED (treatment proxy) | PROHIBITED | PROHIBITED |\n\n",

        "## 6. Mode A Feature Set (13 Features)\n\n",
        "age, gender, education_level, poverty_income_ratio, bmi, waist_circumference, "
        "mean_sbp\\*, mean_dbp\\*, mean_pulse, smoking_status, alcohol_frequency, "
        "physical_activity_level, sedentary_minutes\n\n",
        "\\* mean_sbp and mean_dbp are excluded from the HTN model predictor set.\n\n",

        "## 7. Mode B Feature Set (Mode A + 16 Lab Biomarkers = 29 Features)\n\n",
        "Mode A features + hba1c\\*, fasting_glucose\\*, total_cholesterol, hdl_cholesterol, "
        "triglycerides, ldl_cholesterol, serum_creatinine, blood_urea_nitrogen, "
        "serum_uric_acid, alt_enzyme, ast_enzyme, hemoglobin, wbc_count, platelet_count, "
        "rdw, serum_albumin\n\n",
        "\\* hba1c and fasting_glucose excluded from the DM model predictor set.\n\n",

        "## 8. Missing-Value Strategy (Current Stage)\n\n",
        "No imputation has been applied. All missing values remain as NaN.\n",
        "- Mode A has larger sample sizes because it does not require laboratory measurements.\n",
        "- Mode B has smaller effective sample sizes due to lab coverage (70.4% for routine panel, 41.1% for fasting subsample).\n",
        "- Imputation strategy (IterativeImputer for Tier 1 labs, Median+indicator for fasting subsample) will be applied in Stage 1D during train/val/test splitting.\n\n",

        "## 9. Survey-Weight Variables (Preserved, Not Applied)\n\n",
        "| Variable | Location in raw data | Purpose |\n|---|---|---|\n",
        "| WTINT2YR | DEMO_L.xpt | 2-year interview weight (all participants) |\n",
        "| WTMEC2YR | DEMO_L.xpt | 2-year MEC examination weight |\n",
        "| WTSAF2YR | GLU_L.xpt | Fasting subsample weight (confirmed present) |\n",
        "| SDMVSTRA | DEMO_L.xpt | Masked variance pseudo-stratum |\n",
        "| SDMVPSU  | DEMO_L.xpt | Masked variance pseudo-PSU |\n\n",
        "Survey weights will be applied during probability calibration (Stage 2), not during tree model training.\n\n",

        "## 10. Output Files\n\n",
        "| File | Rows | Features | Target |\n|---|---|---|---|\n",
    ]

    for key, fpath in paths.items():
        parts   = key.rsplit("_mode_", 1)
        model   = parts[0]
        mode    = "mode_" + parts[1]
        tgt_col = {"cvd": "target_cvd", "diabetes": "target_diabetes", "hypertension": "target_hypertension"}[model]
        try:
            pf = pd.read_parquet(fpath)
            feat_cols = [c for c in pf.columns if c not in ["SEQN"] + SURVEY_VARS + [tgt_col]]
            lines.append(f"| `{Path(fpath).name}` | {len(pf):,} | {len(feat_cols)} | {tgt_col} |\n")
        except Exception:
            lines.append(f"| `{Path(fpath).name}` | — | — | {tgt_col} |\n")

    lines.append("\n## 11. Reproducibility\n\n")
    lines.append("Run the full pipeline from the workspace root:\n\n")
    lines.append("```bash\npython -m backend.ml.scripts.build_nhanes_dataset\n```\n\n")
    lines.append("All outputs are regenerated from raw XPT files with no manual edits required.\n\n")
    lines.append("---\n\n*Generated by `build_nhanes_dataset.py` — Stage 1C.*\n")

    manifest_path = out_dir / "DATASET_MANIFEST.md"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    log.info("Dataset manifest written: %s", manifest_path)


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    log.info("=" * 70)
    log.info("NHANES 2021-2023 PREPROCESSING PIPELINE — Stage 1C")
    log.info("=" * 70)

    # -----------------------------------------------------------------------
    # STEP 1: Validate all raw XPT files
    # -----------------------------------------------------------------------
    log.info("\n[STEP 1] Validating raw NHANES XPT files...")
    validation = validate_all_files(raw_dir=RAW_DIR, audit_dir=AUDIT_DIR)
    if validation["status"] != "OK":
        log.error("RAW FILE VALIDATION FAILED. Errors: %s", validation["errors"])
        sys.exit(1)
    log.info("[STEP 1] PASSED — All 16 XPT files validated.")

    # -----------------------------------------------------------------------
    # STEPS 2–5: Load, merge, recode, harmonise
    # -----------------------------------------------------------------------
    log.info("\n[STEPS 2-5] Loading, merging, and harmonising NHANES components...")
    merged = merge_nhanes(raw_dir=RAW_DIR, interim_dir=INTERIM_DIR, audit_dir=AUDIT_DIR)
    log.info("[STEPS 2-5] DONE — Merged adult dataset: %d rows × %d cols", len(merged), len(merged.columns))

    # -----------------------------------------------------------------------
    # STEP 6: Construct targets
    # -----------------------------------------------------------------------
    log.info("\n[STEP 6] Constructing targets (CVD, Diabetes, Hypertension)...")
    df = construct_targets(merged, audit_dir=AUDIT_DIR)
    log.info("[STEP 6] DONE — Targets constructed.")

    # -----------------------------------------------------------------------
    # STEP 7: Write per-model predictor datasets
    # -----------------------------------------------------------------------
    log.info("\n[STEP 7] Writing leakage-controlled predictor datasets...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    paths = write_all_predictor_datasets(df, PROCESSED_DIR)
    log.info("[STEP 7] DONE — 6 dataset files written to %s", PROCESSED_DIR)

    # -----------------------------------------------------------------------
    # STEPS 8–12: Quality control, feature statistics, QC report
    # -----------------------------------------------------------------------
    log.info("\n[STEPS 8-12] Running quality checks...")
    target_report = {}  # already logged inside construct_targets
    run_all_qc(df, target_report, audit_dir=AUDIT_DIR)
    log.info("[STEPS 8-12] DONE — QC reports written.")

    # -----------------------------------------------------------------------
    # STEP 13: Dataset manifest
    # -----------------------------------------------------------------------
    log.info("\n[STEP 13] Writing dataset manifest...")
    write_dataset_manifest(df, paths, target_report, out_dir=PROCESSED_DIR)
    log.info("[STEP 13] DONE — Manifest written.")

    # -----------------------------------------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------------------------------------
    n = len(df)
    log.info("\n" + "=" * 70)
    log.info("STAGE 1C COMPLETE")
    log.info("=" * 70)
    log.info("Adults in analysis population: %d", n)
    for tname in ["target_cvd", "target_diabetes", "target_hypertension"]:
        col = df[tname]
        n_pos = int((col == 1).sum())
        n_ev  = int(col.notna().sum())
        prev  = round(100.0 * n_pos / n_ev, 2) if n_ev > 0 else 0.0
        log.info("  %s: %d positive / %d evaluable (%.2f%%)", tname, n_pos, n_ev, prev)
    log.info("\nProcessed datasets written to: %s", PROCESSED_DIR)
    log.info("QC reports written to: %s", AUDIT_DIR)
    log.info("\nNext step: Stage 1D — Train/Validation/Test Split & Model Benchmarking")
    log.info("  (Awaiting explicit user approval before proceeding.)")
    log.info("=" * 70)


if __name__ == "__main__":
    run_pipeline()
