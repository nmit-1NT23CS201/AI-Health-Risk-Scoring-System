"""
optimize_models.py
==================
Stage 1E-2 — Targeted Hyperparameter Optimization

Performs controlled RandomizedSearchCV (StratifiedKFold, 5 folds) for the
strongest candidate models identified in Stage 1D / Stage 1E-1.

MODELS TO OPTIMIZE PER CONFIGURATION:
  CVD Mode A       -> LogisticRegression
  CVD Mode B       -> LogisticRegression
  Diabetes Mode A  -> HistGradientBoostingClassifier
  Diabetes Mode B  -> HistGradientBoostingClassifier, XGBClassifier
  Hypertension A   -> LogisticRegression
  Hypertension B   -> HistGradientBoostingClassifier, LogisticRegression

CV DESIGN:
  - Hyperparameter search uses ONLY the Stage 1D dev set (train+validation SEQNs).
  - StratifiedKFold(n_splits=5, shuffle=True, random_state=42).
  - The held-out test set (15%) is NEVER touched during tuning.
  - All preprocessing (imputation, scaling) is fitted inside CV folds.
  - random_state=42 everywhere.

USAGE:
    python backend/ml/scripts/optimize_models.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Force UTF-8 stdout on Windows to avoid cp1252 encoding errors
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
import xgboost as xgb  # type: ignore  # pyright: ignore[reportMissingImports]

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    accuracy_score,
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

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("optimize_models")

# ─── Paths ────────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("backend/ml/data/processed/nhanes_2021_2023")
SPLITS_DIR    = PROCESSED_DIR / "splits"
STAGE_1D_DIR  = Path("backend/outputs/model_v2_baseline")
STAGE_1E1_DIR = Path("backend/outputs/model_v2_xgboost_baseline")
OUTPUT_DIR    = Path("backend/ml/evaluation/stage_1e2")
MODELS_DIR    = Path("backend/ml/evaluation/stage_1e2/models")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Constants ────────────────────────────────────────────────────────────────
EXCLUDE_COLS = ["SEQN", "WTINT2YR", "WTMEC2YR", "WTSAF2YR", "SDMVSTRA", "SDMVPSU"]
RANDOM_STATE = 42
CV_FOLDS     = 5
N_ITER       = 40   # RandomizedSearchCV iterations per candidate — reasonable for ~6600-row dev set

CONFIGS = [
    {
        "target_name": "cvd",
        "target_col":  "target_cvd",
        "mode":        "mode_a",
        "file":        "nhanes_cvd_mode_a.parquet",
        "split_file":  "cvd_splits.csv",
        "candidates":  ["logistic_regression"],
    },
    {
        "target_name": "cvd",
        "target_col":  "target_cvd",
        "mode":        "mode_b",
        "file":        "nhanes_cvd_mode_b.parquet",
        "split_file":  "cvd_splits.csv",
        "candidates":  ["logistic_regression"],
    },
    {
        "target_name": "diabetes",
        "target_col":  "target_diabetes",
        "mode":        "mode_a",
        "file":        "nhanes_diabetes_mode_a.parquet",
        "split_file":  "diabetes_splits.csv",
        "candidates":  ["hist_gradient_boosting"],
    },
    {
        "target_name": "diabetes",
        "target_col":  "target_diabetes",
        "mode":        "mode_b",
        "file":        "nhanes_diabetes_mode_b.parquet",
        "split_file":  "diabetes_splits.csv",
        "candidates":  ["hist_gradient_boosting", "xgboost"],
    },
    {
        "target_name": "hypertension",
        "target_col":  "target_hypertension",
        "mode":        "mode_a",
        "file":        "nhanes_hypertension_mode_a.parquet",
        "split_file":  "hypertension_splits.csv",
        "candidates":  ["logistic_regression"],
    },
    {
        "target_name": "hypertension",
        "target_col":  "target_hypertension",
        "mode":        "mode_b",
        "file":        "nhanes_hypertension_mode_b.parquet",
        "split_file":  "hypertension_splits.csv",
        "candidates":  ["hist_gradient_boosting", "logistic_regression"],
    },
]

# ─── Hyperparameter Spaces ────────────────────────────────────────────────────

def get_lr_param_space(n_pos: int, n_neg: int) -> list[dict]:
    """Two variants: default and balanced."""
    return [
        {
            # Variant 1: default class weighting
            "clf__C":            [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
            "clf__solver":       ["lbfgs", "liblinear"],
            "clf__class_weight": [None],
            "clf__max_iter":     [2000],
        },
        {
            # Variant 2: balanced class weighting
            "clf__C":            [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
            "clf__solver":       ["lbfgs", "liblinear"],
            "clf__class_weight": ["balanced"],
            "clf__max_iter":     [2000],
        },
    ]


def get_hgb_param_space() -> list[dict]:
    """HistGradientBoosting search space — two class-weight variants."""
    base = {
        "learning_rate":    [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
        "max_iter":         [100, 200, 300, 500],
        "max_leaf_nodes":   [15, 31, 63, 127],
        "min_samples_leaf": [10, 20, 30, 50],
        "l2_regularization":[0.0, 0.01, 0.1, 1.0, 10.0],
    }
    return [
        {**base, "class_weight": [None]},
        {**base, "class_weight": ["balanced"]},
    ]


def get_xgb_param_space(n_pos: int, n_neg: int) -> list[dict]:
    """XGBoost search space — default and balanced (scale_pos_weight)."""
    spw = n_neg / n_pos if n_pos > 0 else 1.0
    return [
        {
            # Variant 1: default
            "n_estimators":     [100, 200, 300, 500],
            "max_depth":        [3, 4, 5, 6, 7],
            "learning_rate":    [0.01, 0.05, 0.1, 0.2, 0.3],
            "subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "min_child_weight": [1, 3, 5, 10],
            "reg_lambda":       [0.1, 1.0, 5.0, 10.0],
            "scale_pos_weight": [1.0],
        },
        {
            # Variant 2: balanced via scale_pos_weight
            "n_estimators":     [100, 200, 300, 500],
            "max_depth":        [3, 4, 5, 6, 7],
            "learning_rate":    [0.01, 0.05, 0.1, 0.2, 0.3],
            "subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "min_child_weight": [1, 3, 5, 10],
            "reg_lambda":       [0.1, 1.0, 5.0, 10.0],
            "scale_pos_weight": [round(spw, 4)],
        },
    ]

# ─── Helpers ─────────────────────────────────────────────────────────────────

def assert_no_leakage(feature_cols: list[str], target_name: str, target_col: str) -> None:
    """Strict leakage guard matching Stage 1D."""
    for c in feature_cols:
        assert c not in EXCLUDE_COLS,    f"LEAKAGE: survey metadata '{c}' in features!"
        assert c != target_col,          f"LEAKAGE: target column '{c}' in features!"
        if target_name == "cvd":
            assert not c.startswith("MCQ160"), f"LEAKAGE: CVD-defining var '{c}' in features!"
        elif target_name == "diabetes":
            assert c not in ["hba1c", "fasting_glucose", "DIQ010", "DIQ050", "DIQ070"], \
                f"LEAKAGE: Diabetes-defining var '{c}' in features!"
        elif target_name == "hypertension":
            assert c not in ["mean_sbp", "mean_dbp", "BPQ020"], \
                f"LEAKAGE: Hypertension-defining var '{c}' in features!"


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.50) -> dict:
    """Compute all evaluation metrics at a fixed threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "roc_auc":    float(roc_auc_score(y_true, y_prob)),
        "pr_auc":     float(average_precision_score(y_true, y_prob)),
        "accuracy":   float(accuracy_score(y_true, y_pred)),
        "precision":  float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":     float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":         float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(spec),
        "brier":      float(brier_score_loss(y_true, y_prob)),
        "log_loss":   float(log_loss(y_true, y_prob)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold":  threshold,
    }


def make_lr_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     LogisticRegression(random_state=RANDOM_STATE)),
    ])


def make_hgb_model() -> HistGradientBoostingClassifier:
    # HGB has native NaN handling — no imputer needed
    return HistGradientBoostingClassifier(random_state=RANDOM_STATE, early_stopping=False)


def make_xgb_model() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def run_randomized_search(
    estimator,
    param_distributions: list[dict],
    X_dev: pd.DataFrame,
    y_dev: np.ndarray,
    cv: StratifiedKFold,
    n_iter: int,
    candidate_label: str,
) -> tuple[RandomizedSearchCV, dict]:
    """Run RandomizedSearchCV over a list of param grids (handles multiple dicts cleanly)."""
    log.info(f"  -> RandomizedSearchCV n_iter={n_iter}, cv={cv.n_splits} folds ...")

    searcher = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,
        verbose=0,
        error_score="raise",
    )
    searcher.fit(X_dev, y_dev)

    cv_results = searcher.cv_results_
    best_idx   = searcher.best_index_

    cv_summary = {
        "mean_cv_roc_auc": float(searcher.best_score_),
        "std_cv_roc_auc":  float(cv_results["std_test_score"][best_idx]),
        "best_params":     searcher.best_params_,
        "n_iter":          n_iter,
    }

    log.info(f"    Best CV ROC-AUC: {cv_summary['mean_cv_roc_auc']:.4f} ± {cv_summary['std_cv_roc_auc']:.4f}")
    log.info(f"    Best params: {searcher.best_params_}")

    return searcher, cv_summary


# ─── Stage 1D / 1E-1 Baseline Loaders ────────────────────────────────────────

def load_stage1d_best(target_name: str, mode: str) -> dict:
    """Return best Stage 1D model row for a config."""
    df = pd.read_csv(STAGE_1D_DIR / "benchmark_results.csv")
    sub = df[(df["target"] == target_name) & (df["mode"] == mode)].copy()
    sub = sub.sort_values(["roc_auc_validation", "pr_auc_validation"], ascending=False)
    row = sub.iloc[0]
    return {
        "model":            row["model"],
        "val_roc_auc":      row["roc_auc_validation"],
        "val_pr_auc":       row["pr_auc_validation"],
        "test_roc_auc":     row["roc_auc_test"],
        "test_pr_auc":      row["pr_auc_test"],
        "test_recall":      row["recall_test"],
        "test_specificity": row["specificity_test"],
        "test_brier":       row["brier_test"],
    }


def load_stage1e1_best(target_name: str, mode: str) -> dict | None:
    """Return best Stage 1E-1 XGBoost row for a config, or None if not found."""
    p = STAGE_1E1_DIR / "xgboost_benchmark_results.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    sub = df[(df["target"] == target_name) & (df["mode"] == mode)].copy()
    if sub.empty:
        return None
    sub = sub.sort_values(["roc_auc_validation", "pr_auc_validation"], ascending=False)
    row = sub.iloc[0]
    return {
        "model":        row["model"],
        "val_roc_auc":  row["roc_auc_validation"],
        "val_pr_auc":   row["pr_auc_validation"],
        "test_roc_auc": row["roc_auc_test"],
        "test_pr_auc":  row["pr_auc_test"],
        "test_brier":   row["brier_test"],
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_stage_1e2() -> None:
    log.info("=" * 70)
    log.info("Stage 1E-2 — Targeted Hyperparameter Optimization")
    log.info(f"sklearn={sklearn.__version__}  xgboost={xgb.__version__}  seed={RANDOM_STATE}")
    log.info("=" * 70)

    start_wall = time.time()

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    all_results   : list[dict] = []
    comparison_rows: list[dict] = []

    for config in CONFIGS:
        target_name = config["target_name"]
        target_col  = config["target_col"]
        mode        = config["mode"]
        filename    = config["file"]
        split_file  = config["split_file"]
        candidates  = config["candidates"]
        config_key  = f"{target_name}_{mode}"

        log.info(f"\n{'='*70}")
        log.info(f"CONFIG: {config_key}  |  candidates: {candidates}")
        log.info(f"{'='*70}")

        # ── Load data ──────────────────────────────────────────────────────
        df = pd.read_parquet(PROCESSED_DIR / filename)
        df = df.dropna(subset=[target_col]).copy()

        splits_df  = pd.read_csv(SPLITS_DIR / split_file)
        train_seqns = set(splits_df[splits_df["split"] == "train"]["SEQN"])
        val_seqns   = set(splits_df[splits_df["split"] == "validation"]["SEQN"])
        test_seqns  = set(splits_df[splits_df["split"] == "test"]["SEQN"])

        # dev set = train + validation (used for CV tuning)
        dev_seqns = train_seqns | val_seqns

        df_dev  = df[df["SEQN"].isin(dev_seqns)].copy()
        df_test = df[df["SEQN"].isin(test_seqns)].copy()

        feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS and c != target_col]

        # Leakage check
        assert_no_leakage(feature_cols, target_name, target_col)

        X_dev  = df_dev[feature_cols]
        y_dev  = df_dev[target_col].values.astype(int)
        X_test = df_test[feature_cols]
        y_test = df_test[target_col].values.astype(int)

        n_pos_dev = int(y_dev.sum())
        n_neg_dev = int(len(y_dev) - n_pos_dev)

        log.info(f"  Dev set:  N={len(y_dev)} (pos={n_pos_dev}, neg={n_neg_dev})")
        log.info(f"  Test set: N={len(y_test)} (pos={int(y_test.sum())}, neg={len(y_test)-int(y_test.sum())})")
        log.info(f"  Features: {len(feature_cols)}")

        # ── Load Stage 1D baseline ─────────────────────────────────────────
        baseline_1d  = load_stage1d_best(target_name, mode)
        baseline_1e1 = load_stage1e1_best(target_name, mode)

        config_winner: dict | None = None  # best across all candidates for this config

        for cand in candidates:
            log.info(f"\n  -- Candidate: {cand} ----------------------------------------")

            t0 = time.time()

            if cand == "logistic_regression":
                estimator   = make_lr_pipeline()
                param_space = get_lr_param_space(n_pos_dev, n_neg_dev)

            elif cand == "hist_gradient_boosting":
                estimator   = make_hgb_model()
                param_space = get_hgb_param_space()

            elif cand == "xgboost":
                estimator   = make_xgb_model()
                param_space = get_xgb_param_space(n_pos_dev, n_neg_dev)

            else:
                log.warning(f"Unknown candidate '{cand}', skipping.")
                continue

            searcher, cv_summary = run_randomized_search(
                estimator, param_space, X_dev, y_dev, cv,
                n_iter=N_ITER, candidate_label=cand,
            )
            elapsed = time.time() - t0

            # ── Refit best estimator on FULL dev set & evaluate on test ──
            best_model = searcher.best_estimator_
            # best_estimator_ is already refit on full dev set by RandomizedSearchCV(refit=True)

            test_probs   = best_model.predict_proba(X_test)[:, 1]
            test_metrics = compute_metrics(y_test, test_probs)

            log.info(f"    Test ROC-AUC: {test_metrics['roc_auc']:.4f}  PR-AUC: {test_metrics['pr_auc']:.4f}  Brier: {test_metrics['brier']:.4f}")
            log.info(f"    Test Recall: {test_metrics['recall']:.4f}  Spec: {test_metrics['specificity']:.4f}  (elapsed: {elapsed:.1f}s)")

            # ── Save model ────────────────────────────────────────────────
            model_fname  = f"{config_key}_{cand}_v2_optimized.joblib"
            model_path   = MODELS_DIR / model_fname
            joblib.dump(best_model, model_path)
            log.info(f"    Saved model -> {model_path}")

            # ── Build result record ───────────────────────────────────────
            record = {
                "stage":              "1E-2",
                "target":             target_name,
                "mode":               mode,
                "candidate":          cand,
                "dev_n":              int(len(y_dev)),
                "dev_n_pos":          n_pos_dev,
                "test_n":             int(len(y_test)),
                "feature_cols":       feature_cols,
                "n_features":         len(feature_cols),
                "cv_folds":           CV_FOLDS,
                "n_iter_search":      N_ITER,
                "mean_cv_roc_auc":    cv_summary["mean_cv_roc_auc"],
                "std_cv_roc_auc":     cv_summary["std_cv_roc_auc"],
                "best_params":        {str(k): str(v) for k, v in cv_summary["best_params"].items()},
                "test_metrics":       test_metrics,
                "model_file":         str(model_path),
                "elapsed_s":          round(elapsed, 1),
                "random_state":       RANDOM_STATE,
                "sklearn_version":    sklearn.__version__,
                "xgboost_version":    xgb.__version__,
                "tuning_timestamp":   datetime.utcnow().isoformat(),
            }
            all_results.append(record)

            # Track best for this config (by mean CV ROC-AUC primary, test ROC-AUC secondary)
            if config_winner is None or cv_summary["mean_cv_roc_auc"] > config_winner["mean_cv_roc_auc"]:
                config_winner = {**record}

        # ── Comparison row for this config ─────────────────────────────────
        if config_winner is not None:
            opt_test    = config_winner["test_metrics"]
            delta_vs_1d = opt_test["roc_auc"] - baseline_1d["test_roc_auc"]
            delta_pr_1d = opt_test["pr_auc"]  - baseline_1d["test_pr_auc"]

            delta_vs_1e1_roc = None
            delta_vs_1e1_pr  = None
            if baseline_1e1:
                delta_vs_1e1_roc = opt_test["roc_auc"] - baseline_1e1["test_roc_auc"]
                delta_vs_1e1_pr  = opt_test["pr_auc"]  - baseline_1e1["test_pr_auc"]

            if delta_vs_1d > 0.002:
                recommendation = "OPTIMIZED MODEL IS BETTER"
            elif delta_vs_1d >= -0.002:
                recommendation = "ROUGHLY EQUIVALENT — PREFER SIMPLER"
            else:
                recommendation = "KEEP STAGE 1D MODEL"

            comparison_rows.append({
                "target":              target_name,
                "mode":                mode,
                # Stage 1D baseline
                "stage1d_best_model":  baseline_1d["model"],
                "stage1d_test_roc":    round(baseline_1d["test_roc_auc"], 4),
                "stage1d_test_pr":     round(baseline_1d["test_pr_auc"],  4),
                # Stage 1E-1 XGBoost baseline
                "stage1e1_model":      baseline_1e1["model"] if baseline_1e1 else "N/A",
                "stage1e1_test_roc":   round(baseline_1e1["test_roc_auc"], 4) if baseline_1e1 else None,
                "stage1e1_test_pr":    round(baseline_1e1["test_pr_auc"],  4) if baseline_1e1 else None,
                # Stage 1E-2 optimized
                "stage1e2_candidate":  config_winner["candidate"],
                "stage1e2_cv_roc":     round(config_winner["mean_cv_roc_auc"], 4),
                "stage1e2_cv_roc_std": round(config_winner["std_cv_roc_auc"],  4),
                "stage1e2_test_roc":   round(opt_test["roc_auc"], 4),
                "stage1e2_test_pr":    round(opt_test["pr_auc"],  4),
                "stage1e2_test_recall":round(opt_test["recall"],  4),
                "stage1e2_test_spec":  round(opt_test["specificity"], 4),
                "stage1e2_test_brier": round(opt_test["brier"],   4),
                # Deltas
                "delta_roc_vs_1d":     round(delta_vs_1d,  4),
                "delta_pr_vs_1d":      round(delta_pr_1d,  4),
                "delta_roc_vs_1e1":    round(delta_vs_1e1_roc, 4) if delta_vs_1e1_roc is not None else None,
                "delta_pr_vs_1e1":     round(delta_vs_1e1_pr,  4) if delta_vs_1e1_pr  is not None else None,
                "recommendation":      recommendation,
                "best_params":         config_winner["best_params"],
            })

            log.info(f"\n  CONFIG WINNER: {config_winner['candidate']} | "
                     f"CV ROC-AUC={config_winner['mean_cv_roc_auc']:.4f} | "
                     f"Test ROC-AUC={opt_test['roc_auc']:.4f} | "
                     f"Delta vs 1D={delta_vs_1d:+.4f} | {recommendation}")

    total_elapsed = time.time() - start_wall
    log.info(f"\n{'='*70}")
    log.info(f"All configurations complete. Total time: {total_elapsed:.1f}s")

    # ── Save stage_1e2_results.json ───────────────────────────────────────────
    results_path = OUTPUT_DIR / "stage_1e2_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"Saved: {results_path}")

    # ── Save stage_1e2_comparison.csv ─────────────────────────────────────────
    df_comp = pd.DataFrame(comparison_rows)
    # Drop nested best_params column from CSV (keep in JSON)
    csv_cols = [c for c in df_comp.columns if c != "best_params"]
    comp_path = OUTPUT_DIR / "stage_1e2_comparison.csv"
    df_comp[csv_cols].to_csv(comp_path, index=False)
    log.info(f"Saved: {comp_path}")

    # ── Save named v2 model copies ────────────────────────────────────────────
    _save_named_v2_models(comparison_rows)

    # ── Generate stage_1e2_report.md ──────────────────────────────────────────
    report = _build_report(all_results, comparison_rows, df_comp, total_elapsed)
    report_path = OUTPUT_DIR / "stage_1e2_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    log.info(f"Saved: {report_path}")

    log.info("\nStage 1E-2 COMPLETE.")
    _print_summary_table(comparison_rows)


def _save_named_v2_models(comparison_rows: list[dict]) -> None:
    """Copy best optimized models to named v2 pkl files."""
    name_map = {
        ("cvd",          "mode_a"): "cvd_mode_a_v2.pkl",
        ("cvd",          "mode_b"): "cvd_mode_b_v2.pkl",
        ("diabetes",     "mode_a"): "diabetes_mode_a_v2.pkl",
        ("diabetes",     "mode_b"): "diabetes_mode_b_v2.pkl",
        ("hypertension", "mode_a"): "hypertension_mode_a_v2.pkl",
        ("hypertension", "mode_b"): "hypertension_mode_b_v2.pkl",
    }

    for row in comparison_rows:
        key = (row["target"], row["mode"])
        if key not in name_map:
            continue
        cand      = row["stage1e2_candidate"]
        src_path  = MODELS_DIR / f"{row['target']}_{row['mode']}_{cand}_v2_optimized.joblib"
        dest_path = MODELS_DIR / name_map[key]
        if src_path.exists():
            model = joblib.load(src_path)
            joblib.dump(model, dest_path)
            log.info(f"  Named v2 model saved: {dest_path.name}")
        else:
            log.warning(f"  Source model not found: {src_path}")


def _print_summary_table(comparison_rows: list[dict]) -> None:
    log.info("\n" + "="*70)
    log.info("FINAL SUMMARY TABLE — Stage 1E-2 vs Stage 1D")
    log.info("="*70)
    log.info(f"{'Config':<24} {'1D Best':<32} {'1D ROC':>8} {'1E2 ROC':>8} {'Delta':>8}  {'Recommendation'}")
    log.info("-" * 100)
    for r in comparison_rows:
        config = f"{r['target']}_{r['mode']}"
        log.info(f"{config:<24} {r['stage1d_best_model']:<32} {r['stage1d_test_roc']:>8.4f} "
                 f"{r['stage1e2_test_roc']:>8.4f} {r['delta_roc_vs_1d']:>+8.4f}  {r['recommendation']}")


def _build_report(
    all_results: list[dict],
    comparison_rows: list[dict],
    df_comp: pd.DataFrame,
    total_elapsed: float,
) -> str:
    """Build the full Stage 1E-2 Markdown report."""

    lines: list[str] = []

    def h(level: int, text: str) -> None:
        lines.append(f"\n{'#' * level} {text}\n")

    def p(text: str) -> None:
        lines.append(text + "\n")

    h(1, "STAGE 1E-2 — Targeted Model Optimization Report")

    p(f"> **Stage:** 1E-2 — Controlled Hyperparameter Optimization")
    p(f"> **Source Population:** NHANES 2021–2023 Adults (Age ≥ 20)")
    p(f"> **scikit-learn Version:** `{sklearn.__version__}`")
    p(f"> **XGBoost Version:** `{xgb.__version__}`")
    p(f"> **Random Seed:** {RANDOM_STATE}")
    p(f"> **Total Wall-Clock Time:** {total_elapsed:.1f}s")
    p(f"> **Status:** Complete")

    h(2, "1. Methodology")
    p("Stage 1E-2 performs controlled, cross-validated hyperparameter optimization for the strongest model candidates identified from Stage 1D and Stage 1E-1 baselines. The objective is a fair, CV-driven comparison without any test-set leakage.")
    p("**Key design decisions:**")
    p("- Hyperparameter search uses `RandomizedSearchCV` with `n_iter=40` and `scoring='roc_auc'`.")
    p(f"- Cross-validation uses `StratifiedKFold(n_splits={CV_FOLDS}, shuffle=True, random_state={RANDOM_STATE})`.")
    p("- The CV pool is the **development set** = Stage 1D training split + validation split (70% + 15% = 85% of participants).")
    p("- The **held-out test set** (15%) is **never accessed** during hyperparameter search. It is used only once for the final performance report.")
    p("- All preprocessing (imputation, scaling) is fitted **inside** CV fold training sets — no global fit on the full dev set before CV.")
    p("- `random_state=42` is used everywhere applicable.")

    h(2, "2. CV Design & Data Partitioning")
    p("| Split | Participants | Used For |")
    p("|---|---|---|")
    p("| Train (70%) | 5,460–5,464 | Part of CV pool (dev set) |")
    p("| Validation (15%) | 1,170–1,171 | Part of CV pool (dev set) |")
    p("| **Dev set (85%)** | **6,630–6,635** | **Hyperparameter search via 5-fold stratified CV** |")
    p("| Test (15%) | 1,170–1,172 | Final evaluation only — **never touched during tuning** |")

    h(2, "3. Models & Hyperparameter Search Spaces")

    h(3, "Logistic Regression (CVD Mode A/B, Hypertension Mode A/B)")
    p("Implemented as `Pipeline([SimpleImputer(median), StandardScaler, LogisticRegression])`.")
    p("Preprocessing is entirely inside the pipeline, ensuring correct CV behavior.")
    p("| Parameter | Values Explored |")
    p("|---|---|")
    p("| `C` | 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0 |")
    p("| `solver` | lbfgs, liblinear |")
    p("| `class_weight` | None, 'balanced' |")
    p("| `max_iter` | 2000 |")

    h(3, "HistGradientBoostingClassifier (Diabetes Mode A/B, Hypertension Mode B)")
    p("Native NaN handling — no imputation step required.")
    p("| Parameter | Values Explored |")
    p("|---|---|")
    p("| `learning_rate` | 0.01, 0.05, 0.1, 0.15, 0.2, 0.3 |")
    p("| `max_iter` | 100, 200, 300, 500 |")
    p("| `max_leaf_nodes` | 15, 31, 63, 127 |")
    p("| `min_samples_leaf` | 10, 20, 30, 50 |")
    p("| `l2_regularization` | 0.0, 0.01, 0.1, 1.0, 10.0 |")
    p("| `class_weight` | None, 'balanced' |")

    h(3, "XGBClassifier (Diabetes Mode B only)")
    p("| Parameter | Values Explored |")
    p("|---|---|")
    p("| `n_estimators` | 100, 200, 300, 500 |")
    p("| `max_depth` | 3, 4, 5, 6, 7 |")
    p("| `learning_rate` | 0.01, 0.05, 0.1, 0.2, 0.3 |")
    p("| `subsample` | 0.6, 0.7, 0.8, 0.9, 1.0 |")
    p("| `colsample_bytree` | 0.6, 0.7, 0.8, 0.9, 1.0 |")
    p("| `min_child_weight` | 1, 3, 5, 10 |")
    p("| `reg_lambda` | 0.1, 1.0, 5.0, 10.0 |")
    p("| `scale_pos_weight` | 1.0 (default), computed N_neg/N_pos (balanced) |")

    h(2, "4. Best Parameters & CV Performance")
    for res in all_results:
        key = f"{res['target']}_{res['mode']} — {res['candidate']}"
        h(3, key)
        p(f"**Mean CV ROC-AUC:** {res['mean_cv_roc_auc']:.4f} ± {res['std_cv_roc_auc']:.4f}")
        p(f"**Best hyperparameters found:**")
        p("```")
        for k, v in res["best_params"].items():
            p(f"  {k}: {v}")
        p("```")

    h(2, "5. Final Test-Set Performance (Optimized Models)")
    p("*Threshold = 0.50 throughout. Test set was accessed exactly once, after CV was complete.*")
    p("")
    p("| Config | Candidate | Test ROC-AUC | Test PR-AUC | Test Recall | Test Spec | Test Brier |")
    p("|---|---|---|---|---|---|---|")
    for res in all_results:
        m = res["test_metrics"]
        p(f"| {res['target']}_{res['mode']} | `{res['candidate']}` | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} | {m['recall']:.4f} | {m['specificity']:.4f} | {m['brier']:.4f} |")

    h(2, "6. Comparison Against Stage 1E-1 Baseline")
    p("Comparing the best Stage 1E-2 optimized model against Stage 1D winner and Stage 1E-1 XGBoost baseline:")
    p("")
    p("| Target | Mode | Stage 1D ROC | 1E-1 XGB ROC | 1E-2 Opt ROC | Delta vs 1D | Delta vs 1E-1 | Recommendation |")
    p("|---|---|---|---|---|---|---|---|")
    for r in comparison_rows:
        e1_roc = f"{r['stage1e1_test_roc']:.4f}" if r['stage1e1_test_roc'] is not None else "N/A"
        d1e1   = f"{r['delta_roc_vs_1e1']:+.4f}" if r['delta_roc_vs_1e1'] is not None else "N/A"
        p(f"| **{r['target'].upper()}** | **{r['mode']}** | {r['stage1d_test_roc']:.4f} | {e1_roc} | {r['stage1e2_test_roc']:.4f} | {r['delta_roc_vs_1d']:+.4f} | {d1e1} | **{r['recommendation']}** |")

    h(2, "7. Winner for Each Target/Mode")
    for r in comparison_rows:
        h(3, f"{r['target'].upper()} {r['mode'].upper()}")
        p(f"- **Optimized candidate:** `{r['stage1e2_candidate']}`")
        p(f"- **CV ROC-AUC:** {r['stage1e2_cv_roc']:.4f} ± {r['stage1e2_cv_roc_std']:.4f}")
        p(f"- **Test ROC-AUC:** {r['stage1e2_test_roc']:.4f} (Stage 1D: {r['stage1d_test_roc']:.4f}, Delta: {r['delta_roc_vs_1d']:+.4f})")
        p(f"- **Test PR-AUC:** {r['stage1e2_test_pr']:.4f} (Stage 1D: {r['stage1d_test_pr']:.4f}, Delta: {r['delta_pr_vs_1d']:+.4f})")
        p(f"- **Recommendation:** **{r['recommendation']}**")

    h(2, "8. Warnings & Limitations")
    p("1. **Small dataset size:** With ~7,800 participants total (6,600 in dev set), small CV AUC differences (< 0.005) should be interpreted cautiously — they may be within noise.")
    p("2. **No probability calibration:** Reported probabilities are raw model outputs. If these are to be used as risk scores, Platt scaling or isotonic regression calibration should be applied in a subsequent stage.")
    p("3. **Threshold = 0.50 fixed:** All reported precision/recall/F1/specificity are at threshold 0.50. Optimal thresholds for clinical use may differ significantly.")
    p("4. **Test set used once:** The final test set was accessed exactly once after all CV-based model selection was complete. No further iterations or threshold tuning were performed against it.")
    p("5. **XGBoost vs HGB:** For Diabetes Mode B, both were optimized and compared; the CV-best was saved as the config winner. Check `stage_1e2_results.json` for both candidate metrics.")

    h(2, "9. Output Artifact Paths")
    p(f"- `backend/ml/evaluation/stage_1e2/stage_1e2_results.json` — Full experiment JSON")
    p(f"- `backend/ml/evaluation/stage_1e2/stage_1e2_comparison.csv` — Comparison CSV")
    p(f"- `backend/ml/evaluation/stage_1e2/stage_1e2_report.md` — This report")
    p(f"- `backend/ml/evaluation/stage_1e2/models/` — All optimized model `.joblib` files")
    p(f"- `backend/ml/evaluation/stage_1e2/models/{{target}}_{{mode}}_v2.pkl` — Named v2 model files")

    return "\n".join(lines)


if __name__ == "__main__":
    run_stage_1e2()
