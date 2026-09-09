"""
optimize_random_forest.py
=========================
Stage 1E-3 -- Controlled Random Forest Optimization & Final Model Comparison

Performs RandomizedSearchCV (StratifiedKFold-5) for RandomForestClassifier
across all 6 dataset/mode configurations using the same participant-level
dev/test split methodology established in Stage 1E-2.

Compares:
  - V1 Random Forest baseline (Stage 1D results)
  - Optimized Random Forest (this stage)
  - Stage 1E-2 winning model

CV DESIGN (identical to Stage 1E-2):
  - Dev pool = Stage 1D train + validation SEQNs (85%)
  - StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  - RandomizedSearchCV(n_iter=40, scoring="roc_auc")
  - Test set (15%) NEVER touched during tuning
  - All preprocessing fitted inside CV fold training sets

USAGE:
    $env:PYTHONIOENCODING="utf-8"
    .venv\\Scripts\\python.exe backend/ml/scripts/optimize_random_forest.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Force UTF-8 on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("optimize_rf")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROCESSED_DIR   = Path("backend/ml/data/processed/nhanes_2021_2023")
SPLITS_DIR      = PROCESSED_DIR / "splits"
STAGE_1D_CSV    = Path("backend/outputs/model_v2_baseline/benchmark_results.csv")
STAGE_1E2_CSV   = Path("backend/ml/evaluation/stage_1e2/stage_1e2_comparison.csv")
OUTPUT_DIR      = Path("backend/ml/evaluation/stage_1e3")
MODELS_DIR      = OUTPUT_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXCLUDE_COLS = ["SEQN", "WTINT2YR", "WTMEC2YR", "WTSAF2YR", "SDMVSTRA", "SDMVPSU"]
RANDOM_STATE = 42
CV_FOLDS     = 5
N_ITER       = 40

CONFIGS = [
    {"target_name": "cvd",          "target_col": "target_cvd",          "mode": "mode_a", "file": "nhanes_cvd_mode_a.parquet",          "split_file": "cvd_splits.csv"},
    {"target_name": "cvd",          "target_col": "target_cvd",          "mode": "mode_b", "file": "nhanes_cvd_mode_b.parquet",          "split_file": "cvd_splits.csv"},
    {"target_name": "diabetes",     "target_col": "target_diabetes",     "mode": "mode_a", "file": "nhanes_diabetes_mode_a.parquet",     "split_file": "diabetes_splits.csv"},
    {"target_name": "diabetes",     "target_col": "target_diabetes",     "mode": "mode_b", "file": "nhanes_diabetes_mode_b.parquet",     "split_file": "diabetes_splits.csv"},
    {"target_name": "hypertension", "target_col": "target_hypertension", "mode": "mode_a", "file": "nhanes_hypertension_mode_a.parquet", "split_file": "hypertension_splits.csv"},
    {"target_name": "hypertension", "target_col": "target_hypertension", "mode": "mode_b", "file": "nhanes_hypertension_mode_b.parquet", "split_file": "hypertension_splits.csv"},
]

# ---------------------------------------------------------------------------
# RF hyperparameter space
# ---------------------------------------------------------------------------
RF_PARAM_SPACE = [
    {
        # Default class weighting
        "clf__n_estimators":    [300, 500, 800],
        "clf__max_depth":       [None, 8, 12, 16, 20],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf":  [1, 2, 5, 10],
        "clf__max_features":    ["sqrt", "log2", 0.5, 0.8],
        "clf__class_weight":    [None],
    },
    {
        # Balanced class weighting
        "clf__n_estimators":    [300, 500, 800],
        "clf__max_depth":       [None, 8, 12, 16, 20],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf":  [1, 2, 5, 10],
        "clf__max_features":    ["sqrt", "log2", 0.5, 0.8],
        "clf__class_weight":    ["balanced"],
    },
    {
        # Balanced subsample class weighting
        "clf__n_estimators":    [300, 500, 800],
        "clf__max_depth":       [None, 8, 12, 16, 20],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf":  [1, 2, 5, 10],
        "clf__max_features":    ["sqrt", "log2", 0.5, 0.8],
        "clf__class_weight":    ["balanced_subsample"],
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_no_leakage(feature_cols: list[str], target_name: str, target_col: str) -> None:
    for c in feature_cols:
        assert c not in EXCLUDE_COLS,    f"LEAKAGE: survey metadata '{c}' in features!"
        assert c != target_col,          f"LEAKAGE: target '{c}' in features!"
        if target_name == "cvd":
            assert not c.startswith("MCQ160"), f"LEAKAGE: CVD-defining var '{c}'!"
        elif target_name == "diabetes":
            assert c not in ["hba1c", "fasting_glucose", "DIQ010", "DIQ050", "DIQ070"], \
                f"LEAKAGE: Diabetes-defining var '{c}'!"
        elif target_name == "hypertension":
            assert c not in ["mean_sbp", "mean_dbp", "BPQ020"], \
                f"LEAKAGE: Hypertension-defining var '{c}'!"


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.50) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "roc_auc":     float(roc_auc_score(y_true, y_prob)),
        "pr_auc":      float(average_precision_score(y_true, y_prob)),
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "precision":   float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":      float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":          float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(spec),
        "brier":       float(brier_score_loss(y_true, y_prob)),
        "log_loss":    float(log_loss(y_true, y_prob)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold":   threshold,
    }


def make_rf_pipeline() -> Pipeline:
    """RF wrapped in Pipeline with median imputation (same as Stage 1D/1E-2 LR approach)."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf",     RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
    ])


def load_stage1d_rf_baselines(target_name: str, mode: str) -> dict:
    """Return best V1 RF and balanced-RF row from Stage 1D CSV."""
    df = pd.read_csv(STAGE_1D_CSV)
    sub = df[
        (df["target"] == target_name) &
        (df["mode"]   == mode) &
        (df["model"].str.startswith("random_forest"))
    ].copy()
    if sub.empty:
        return {"default": None, "balanced": None}

    def _row(model_name: str) -> dict | None:
        r = sub[sub["model"] == model_name]
        if r.empty:
            return None
        r = r.iloc[0]
        return {
            "model":        model_name,
            "val_roc_auc":  float(r["roc_auc_validation"]),
            "val_pr_auc":   float(r["pr_auc_validation"]),
            "test_roc_auc": float(r["roc_auc_test"]),
            "test_pr_auc":  float(r["pr_auc_test"]),
            "test_recall":  float(r["recall_test"]),
            "test_brier":   float(r["brier_test"]),
        }

    # Best V1 RF (pick whichever has higher test ROC-AUC)
    candidates = []
    for m in ["random_forest", "random_forest_balanced"]:
        row = _row(m)
        if row:
            candidates.append(row)
    best_v1 = max(candidates, key=lambda x: x["test_roc_auc"]) if candidates else None

    return {
        "best_v1": best_v1,
        "all":     candidates,
    }


def load_stage1e2_winner(target_name: str, mode: str) -> dict | None:
    if not STAGE_1E2_CSV.exists():
        return None
    df = pd.read_csv(STAGE_1E2_CSV)
    sub = df[(df["target"] == target_name) & (df["mode"] == mode)]
    if sub.empty:
        return None
    r = sub.iloc[0]
    return {
        "candidate":   str(r["stage1e2_candidate"]),
        "cv_roc":      float(r["stage1e2_cv_roc"]),
        "test_roc":    float(r["stage1e2_test_roc"]),
        "test_pr":     float(r["stage1e2_test_pr"]),
        "test_recall": float(r["stage1e2_test_recall"]),
        "test_brier":  float(r["stage1e2_test_brier"]),
        "recommendation": str(r["recommendation"]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_stage_1e3() -> None:
    log.info("=" * 70)
    log.info("Stage 1E-3 -- Random Forest Optimization & Final Model Comparison")
    log.info(f"sklearn={sklearn.__version__}  seed={RANDOM_STATE}  cv_folds={CV_FOLDS}  n_iter={N_ITER}")
    log.info("=" * 70)

    wall_start = time.time()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    all_results: list[dict]    = []
    comparison_rows: list[dict] = []

    for config in CONFIGS:
        target_name = config["target_name"]
        target_col  = config["target_col"]
        mode        = config["mode"]
        filename    = config["file"]
        split_file  = config["split_file"]
        config_key  = f"{target_name}_{mode}"

        log.info(f"\n{'='*70}")
        log.info(f"CONFIG: {config_key}")
        log.info(f"{'='*70}")

        # -- Load dataset ---------------------------------------------------
        df = pd.read_parquet(PROCESSED_DIR / filename)
        df = df.dropna(subset=[target_col]).copy()

        splits_df   = pd.read_csv(SPLITS_DIR / split_file)
        train_seqns = set(splits_df[splits_df["split"] == "train"]["SEQN"])
        val_seqns   = set(splits_df[splits_df["split"] == "validation"]["SEQN"])
        test_seqns  = set(splits_df[splits_df["split"] == "test"]["SEQN"])
        dev_seqns   = train_seqns | val_seqns

        df_dev  = df[df["SEQN"].isin(dev_seqns)].copy()
        df_test = df[df["SEQN"].isin(test_seqns)].copy()

        feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS and c != target_col]
        assert_no_leakage(feature_cols, target_name, target_col)

        X_dev  = df_dev[feature_cols]
        y_dev  = df_dev[target_col].values.astype(int)
        X_test = df_test[feature_cols]
        y_test = df_test[target_col].values.astype(int)

        n_pos = int(y_dev.sum())
        n_neg = int(len(y_dev) - n_pos)

        log.info(f"  Features : {len(feature_cols)}")
        log.info(f"  Dev      : N={len(y_dev)} (pos={n_pos}, neg={n_neg})")
        log.info(f"  Test     : N={len(y_test)} (pos={int(y_test.sum())})")

        # -- Load baselines -------------------------------------------------
        v1_rf   = load_stage1d_rf_baselines(target_name, mode)
        e2_best = load_stage1e2_winner(target_name, mode)

        # -- RandomizedSearchCV --------------------------------------------
        log.info(f"  -> RandomizedSearchCV n_iter={N_ITER}, cv={CV_FOLDS} ...")
        t0 = time.time()

        searcher = RandomizedSearchCV(
            estimator=make_rf_pipeline(),
            param_distributions=RF_PARAM_SPACE,
            n_iter=N_ITER,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            refit=True,
            verbose=0,
            error_score="raise",
        )
        searcher.fit(X_dev, y_dev)
        elapsed = time.time() - t0

        cv_results = searcher.cv_results_
        best_idx   = searcher.best_index_
        mean_cv    = float(searcher.best_score_)
        std_cv     = float(cv_results["std_test_score"][best_idx])
        best_params = searcher.best_params_

        log.info(f"     Best CV ROC-AUC : {mean_cv:.4f} +/- {std_cv:.4f}")
        log.info(f"     Best params     : {best_params}")
        log.info(f"     Elapsed         : {elapsed:.1f}s")

        # -- Evaluate on test (once, after tuning is complete) ------------
        best_model  = searcher.best_estimator_
        test_probs  = best_model.predict_proba(X_test)[:, 1]
        test_m      = compute_metrics(y_test, test_probs)

        log.info(f"     Test ROC-AUC    : {test_m['roc_auc']:.4f}")
        log.info(f"     Test PR-AUC     : {test_m['pr_auc']:.4f}")
        log.info(f"     Test Recall     : {test_m['recall']:.4f}  Spec: {test_m['specificity']:.4f}")
        log.info(f"     Test F1         : {test_m['f1']:.4f}  Brier: {test_m['brier']:.4f}")

        # -- Save model ----------------------------------------------------
        model_path = MODELS_DIR / f"{config_key}_rf_v2_optimized.joblib"
        joblib.dump(best_model, model_path)
        # Also save as named pkl for consistency with 1E-2 convention
        named_path = MODELS_DIR / f"{config_key}_rf_v2.pkl"
        joblib.dump(best_model, named_path)
        log.info(f"     Saved model -> {model_path.name}")

        # -- Build result record -------------------------------------------
        record = {
            "stage":            "1E-3",
            "target":           target_name,
            "mode":             mode,
            "config_key":       config_key,
            "dev_n":            int(len(y_dev)),
            "dev_n_pos":        n_pos,
            "test_n":           int(len(y_test)),
            "n_features":       len(feature_cols),
            "feature_cols":     feature_cols,
            "cv_folds":         CV_FOLDS,
            "n_iter_search":    N_ITER,
            "mean_cv_roc_auc":  mean_cv,
            "std_cv_roc_auc":   std_cv,
            "best_params":      {str(k): str(v) for k, v in best_params.items()},
            "test_metrics":     test_m,
            "model_file":       str(model_path),
            "elapsed_s":        round(elapsed, 1),
            "random_state":     RANDOM_STATE,
            "sklearn_version":  sklearn.__version__,
            "timestamp":        datetime.now().isoformat(),
        }
        all_results.append(record)

        # -- Comparison row ------------------------------------------------
        v1_best      = v1_rf.get("best_v1")
        v1_roc       = v1_best["test_roc_auc"] if v1_best else None
        v1_pr        = v1_best["test_pr_auc"]  if v1_best else None
        v1_model_name = v1_best["model"]        if v1_best else "N/A"

        e2_roc   = e2_best["test_roc"] if e2_best else None
        e2_pr    = e2_best["test_pr"]  if e2_best else None
        e2_cand  = e2_best["candidate"] if e2_best else "N/A"

        delta_vs_v1  = test_m["roc_auc"] - v1_roc  if v1_roc  is not None else None
        delta_vs_e2  = test_m["roc_auc"] - e2_roc  if e2_roc  is not None else None

        # Recommendation logic
        if e2_roc is not None:
            if delta_vs_e2 > 0.002:
                recommendation = "OPTIMIZED RF IS BEST"
            elif delta_vs_e2 >= -0.002:
                if test_m["roc_auc"] >= (v1_roc or 0) - 0.001:
                    recommendation = "ROUGHLY EQUIVALENT -- COMPARE CAREFULLY"
                else:
                    recommendation = f"KEEP STAGE 1E-2 ({e2_cand})"
            else:
                recommendation = f"KEEP STAGE 1E-2 ({e2_cand})"
        elif v1_roc is not None:
            if delta_vs_v1 > 0.005:
                recommendation = "OPTIMIZED RF > V1 RF"
            else:
                recommendation = "KEEP V1 RF"
        else:
            recommendation = "NO BASELINE AVAILABLE"

        comparison_rows.append({
            "target":               target_name,
            "mode":                 mode,
            # V1 RF baseline
            "v1_rf_model":          v1_model_name,
            "v1_rf_test_roc":       round(v1_roc, 4)  if v1_roc  is not None else None,
            "v1_rf_test_pr":        round(v1_pr, 4)   if v1_pr   is not None else None,
            # Stage 1E-2 best
            "stage1e2_candidate":   e2_cand,
            "stage1e2_test_roc":    round(e2_roc, 4)  if e2_roc  is not None else None,
            "stage1e2_test_pr":     round(e2_pr, 4)   if e2_pr   is not None else None,
            # Optimized RF (this stage)
            "opt_rf_cv_roc":        round(mean_cv, 4),
            "opt_rf_cv_roc_std":    round(std_cv, 4),
            "opt_rf_test_roc":      round(test_m["roc_auc"],  4),
            "opt_rf_test_pr":       round(test_m["pr_auc"],   4),
            "opt_rf_test_recall":   round(test_m["recall"],   4),
            "opt_rf_test_spec":     round(test_m["specificity"], 4),
            "opt_rf_test_f1":       round(test_m["f1"],       4),
            "opt_rf_test_brier":    round(test_m["brier"],    4),
            # Deltas
            "delta_roc_vs_v1":      round(delta_vs_v1, 4)  if delta_vs_v1  is not None else None,
            "delta_roc_vs_1e2":     round(delta_vs_e2, 4)  if delta_vs_e2  is not None else None,
            # Best params (key summary)
            "best_n_estimators":    best_params.get("clf__n_estimators"),
            "best_max_depth":       best_params.get("clf__max_depth"),
            "best_max_features":    best_params.get("clf__max_features"),
            "best_class_weight":    best_params.get("clf__class_weight"),
            "recommendation":       recommendation,
        })

        v1_roc_s = f"{v1_roc:.4f}" if v1_roc is not None else "N/A"
        e2_roc_s = f"{e2_roc:.4f}" if e2_roc is not None else "N/A"
        log.info(f"\n  Result: Test ROC={test_m['roc_auc']:.4f} | "
                 f"V1 RF ROC={v1_roc_s} | "
                 f"1E-2 ROC={e2_roc_s} | "
                 f"{recommendation}")

    total_elapsed = time.time() - wall_start
    log.info(f"\n{'='*70}")
    log.info(f"All 6 configs complete. Total time: {total_elapsed:.1f}s")

    # -- Save results.json -------------------------------------------------
    results_path = OUTPUT_DIR / "random_forest_optimization_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"Saved: {results_path}")

    # -- Save comparison.csv -----------------------------------------------
    df_comp = pd.DataFrame(comparison_rows)
    comp_path = OUTPUT_DIR / "random_forest_comparison.csv"
    df_comp.to_csv(comp_path, index=False)
    log.info(f"Saved: {comp_path}")

    # -- Build & save report -----------------------------------------------
    report = _build_report(all_results, comparison_rows, total_elapsed)
    report_path = OUTPUT_DIR / "RANDOM_FOREST_STAGE_1E3_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    log.info(f"Saved: {report_path}")

    log.info("\nStage 1E-3 COMPLETE.")
    _print_summary_table(comparison_rows)


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _print_summary_table(rows: list[dict]) -> None:
    log.info("\n" + "=" * 110)
    log.info("FINAL COMPARISON TABLE -- Stage 1E-3")
    log.info("=" * 110)
    header = f"{'Config':<24} {'V1 RF ROC':>10} {'1E-2 ROC':>10} {'Opt RF ROC':>12} {'dv1':>8} {'d1E2':>8}  Recommendation"
    log.info(header)
    log.info("-" * 110)
    for r in rows:
        v1  = f"{r['v1_rf_test_roc']:.4f}" if r["v1_rf_test_roc"] is not None else "  N/A  "
        e2  = f"{r['stage1e2_test_roc']:.4f}" if r["stage1e2_test_roc"] is not None else "  N/A  "
        dv1 = f"{r['delta_roc_vs_v1']:+.4f}" if r["delta_roc_vs_v1"] is not None else "   N/A"
        de2 = f"{r['delta_roc_vs_1e2']:+.4f}" if r["delta_roc_vs_1e2"] is not None else "   N/A"
        log.info(f"{r['target']}_{r['mode']:<18} {v1:>10} {e2:>10} {r['opt_rf_test_roc']:>12.4f} {dv1:>8} {de2:>8}  {r['recommendation']}")


def _build_report(
    all_results: list[dict],
    comparison_rows: list[dict],
    total_elapsed: float,
) -> str:
    lines: list[str] = []

    def h(level: int, text: str) -> None:
        lines.append(f"\n{'#' * level} {text}\n")

    def p(text: str) -> None:
        lines.append(text + "\n")

    h(1, "STAGE 1E-3 -- Random Forest Optimization & Final Model Comparison")

    p(f"> **Stage:** 1E-3 -- Controlled Random Forest Hyperparameter Optimization")
    p(f"> **Source Population:** NHANES 2021-2023 Adults (Age >= 20)")
    p(f"> **scikit-learn Version:** `{sklearn.__version__}`")
    p(f"> **Random Seed:** {RANDOM_STATE}")
    p(f"> **CV Folds:** {CV_FOLDS} (StratifiedKFold, shuffle=True)")
    p(f"> **RandomizedSearchCV n_iter:** {N_ITER}")
    p(f"> **Total Wall-Clock Time:** {total_elapsed:.1f}s")
    p(f"> **Status:** Complete")

    h(2, "1. Methodology")

    p("Stage 1E-3 performs a focused Random Forest hyperparameter optimization using **the identical train/test methodology as Stage 1E-2**, allowing direct performance comparison.")
    p("")
    p("**Data partitioning (same as Stage 1E-2):**")
    p("- **Dev set** = Stage 1D `train` + `validation` SEQNs (70% + 15% = 85%)")
    p("- **Test set** (15%) is **never accessed during hyperparameter search**")
    p("- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the dev set")
    p("- `RandomizedSearchCV(n_iter=40, scoring='roc_auc')` selects the best parameter set")
    p("- Best estimator (refitted on full dev set by scikit-learn) is evaluated **once** on the held-out test set")
    p("")
    p("**Preprocessing pipeline:**")
    p("- `SimpleImputer(strategy='median')` + `RandomForestClassifier`")
    p("- Imputer fitted inside each CV fold (no data leakage across folds)")
    p("- `class_weight` treated as a hyperparameter (`None`, `'balanced'`, `'balanced_subsample'`)")

    h(2, "2. Hyperparameter Search Space")

    p("| Parameter | Values Explored |")
    p("|---|---|")
    p("| `n_estimators` | 300, 500, 800 |")
    p("| `max_depth` | None, 8, 12, 16, 20 |")
    p("| `min_samples_split` | 2, 5, 10 |")
    p("| `min_samples_leaf` | 1, 2, 5, 10 |")
    p("| `max_features` | 'sqrt', 'log2', 0.5, 0.8 |")
    p("| `class_weight` | None, 'balanced', 'balanced_subsample' |")
    p("")
    p(f"Search is conducted as `RandomizedSearchCV(n_iter={N_ITER})` across all three class-weight variants, "
      f"with `random_state={RANDOM_STATE}` ensuring full reproducibility.")

    h(2, "3. Per-Configuration Results")

    for res in all_results:
        config_key = res["config_key"]
        m = res["test_metrics"]
        h(3, config_key)
        p(f"**Dev set:** {res['dev_n']} participants ({res['dev_n_pos']} positive)  |  "
          f"**Test set:** {res['test_n']}  |  **Features:** {res['n_features']}")
        p("")
        p(f"**CV ROC-AUC (5-fold):** {res['mean_cv_roc_auc']:.4f} +/- {res['std_cv_roc_auc']:.4f}")
        p("")
        p("**Best hyperparameters:**")
        p("```")
        for k, v in res["best_params"].items():
            p(f"  {k}: {v}")
        p("```")
        p("")
        p("**Test-set metrics (threshold = 0.50, accessed once):**")
        p("")
        p("| Metric | Value |")
        p("|---|---|")
        p(f"| ROC-AUC | **{m['roc_auc']:.4f}** |")
        p(f"| PR-AUC | {m['pr_auc']:.4f} |")
        p(f"| Accuracy | {m['accuracy']:.4f} |")
        p(f"| Precision | {m['precision']:.4f} |")
        p(f"| Recall | {m['recall']:.4f} |")
        p(f"| F1 | {m['f1']:.4f} |")
        p(f"| Specificity | {m['specificity']:.4f} |")
        p(f"| Brier Score | {m['brier']:.4f} |")
        p(f"| TP / FP / FN / TN | {m['tp']} / {m['fp']} / {m['fn']} / {m['tn']} |")
        p("")

    h(2, "4. Final Comparison: V1 RF vs Optimized RF vs Stage 1E-2 Winner")

    p("*All ROC-AUC values are from the same held-out test set (15% of participants). "
      "Lower test Brier = better calibration at threshold 0.50.*")
    p("")
    p("| Configuration | V1 RF (best) | Opt RF (1E-3) | Stage 1E-2 Winner | RF vs V1 | RF vs 1E-2 | Recommended |")
    p("|---|---|---|---|---|---|---|")
    for r in comparison_rows:
        cfg      = f"{r['target']}_{r['mode']}"
        v1_s     = f"{r['v1_rf_test_roc']:.4f}"  if r["v1_rf_test_roc"]  is not None else "N/A"
        e2_s     = f"{r['stage1e2_test_roc']:.4f}" if r["stage1e2_test_roc"] is not None else "N/A"
        e2_cand  = r["stage1e2_candidate"] if r["stage1e2_candidate"] != "N/A" else "-"
        dv1_s    = f"{r['delta_roc_vs_v1']:+.4f}" if r["delta_roc_vs_v1"] is not None else "N/A"
        de2_s    = f"{r['delta_roc_vs_1e2']:+.4f}" if r["delta_roc_vs_1e2"] is not None else "N/A"
        p(f"| **{cfg}** | {v1_s} (`{r['v1_rf_model']}`) | **{r['opt_rf_test_roc']:.4f}** | {e2_s} (`{e2_cand}`) | {dv1_s} | {de2_s} | **{r['recommendation']}** |")

    h(2, "5. Per-Configuration Recommendation")

    for r in comparison_rows:
        cfg = f"{r['target'].upper()} {r['mode'].upper()}"
        h(3, cfg)
        p(f"- **V1 RF best test ROC-AUC:** {r['v1_rf_test_roc']:.4f} (`{r['v1_rf_model']}`)")
        p(f"- **Optimized RF test ROC-AUC:** {r['opt_rf_test_roc']:.4f} (CV: {r['opt_rf_cv_roc']:.4f} +/- {r['opt_rf_cv_roc_std']:.4f})")
        p(f"- **Stage 1E-2 winner test ROC-AUC:** {r['stage1e2_test_roc']:.4f} (`{r['stage1e2_candidate']}`)")
        p(f"- **Best RF params:** n_estimators={r['best_n_estimators']}, max_depth={r['best_max_depth']}, "
          f"max_features={r['best_max_features']}, class_weight={r['best_class_weight']}")
        p(f"- **Recommendation:** **{r['recommendation']}**")
        p("")

    h(2, "6. Warnings & Limitations")

    p("1. **Random Forest and tree depth:** `max_depth=None` allows fully-grown trees; verify that "
      "it does not indicate overfitting by comparing CV vs test ROC-AUC gaps.")
    p("2. **No probability calibration:** RF probabilities from `predict_proba` are not calibrated. "
      "Platt scaling or isotonic regression may be beneficial before clinical use.")
    p("3. **Threshold fixed at 0.50:** All recall/precision/F1/specificity are at 0.50. "
      "Clinical deployments may require threshold adjustment.")
    p("4. **Test set accessed once:** The final test set was touched exactly once per config, "
      "after all CV-based model selection was complete. No further iterations were performed.")
    p("5. **Small dataset:** ~7,800 NHANES participants is modest for RF. Differences less than "
      "~0.003 ROC-AUC points should not be treated as meaningful improvements.")

    h(2, "7. Reproduction Command")

    p("Run from the project root with the `.venv` activated:")
    p("```powershell")
    p("$env:PYTHONIOENCODING=\"utf-8\"")
    p(".venv\\Scripts\\python.exe backend/ml/scripts/optimize_random_forest.py")
    p("```")

    h(2, "8. Output Artifact Paths")

    p(f"- `backend/ml/evaluation/stage_1e3/random_forest_optimization_results.json`")
    p(f"- `backend/ml/evaluation/stage_1e3/random_forest_comparison.csv`")
    p(f"- `backend/ml/evaluation/stage_1e3/RANDOM_FOREST_STAGE_1E3_REPORT.md`")
    p(f"- `backend/ml/evaluation/stage_1e3/models/{{config}}_rf_v2_optimized.joblib`")
    p(f"- `backend/ml/evaluation/stage_1e3/models/{{config}}_rf_v2.pkl`")

    return "\n".join(lines)


if __name__ == "__main__":
    run_stage_1e3()
