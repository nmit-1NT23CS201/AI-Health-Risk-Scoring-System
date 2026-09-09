"""
benchmark_models.py
===================
Stage 1D — Model Benchmarking & Honest Baselines

This script performs participant-level splitting, trains baseline ML models across
6 dataset/mode configurations, computes metrics on validation and final test sets,
generates feature importance logs, creates visualizations, and saves all outputs.

USAGE:
    python backend/ml/scripts/benchmark_models.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("benchmark_models")

# Define directories
PROCESSED_DIR = Path("backend/ml/data/processed/nhanes_2021_2023")
SPLITS_DIR = PROCESSED_DIR / "splits"
OUTPUT_DIR = Path("backend/outputs/model_v2_baseline")
MODELS_DIR = OUTPUT_DIR / "models"
FI_DIR = OUTPUT_DIR / "feature_importance"

SPLITS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FI_DIR.mkdir(parents=True, exist_ok=True)

# Survey metadata & identifier columns to exclude from predictors
EXCLUDE_COLS = ["SEQN", "WTINT2YR", "WTMEC2YR", "WTSAF2YR", "SDMVSTRA", "SDMVPSU"]
TARGET_COLS = ["target_cvd", "target_diabetes", "target_hypertension"]

CONFIGS = [
    {"target_name": "cvd", "target_col": "target_cvd", "mode": "mode_a", "file": "nhanes_cvd_mode_a.parquet"},
    {"target_name": "cvd", "target_col": "target_cvd", "mode": "mode_b", "file": "nhanes_cvd_mode_b.parquet"},
    {"target_name": "diabetes", "target_col": "target_diabetes", "mode": "mode_a", "file": "nhanes_diabetes_mode_a.parquet"},
    {"target_name": "diabetes", "target_col": "target_diabetes", "mode": "mode_b", "file": "nhanes_diabetes_mode_b.parquet"},
    {"target_name": "hypertension", "target_col": "target_hypertension", "mode": "mode_a", "file": "nhanes_hypertension_mode_a.parquet"},
    {"target_name": "hypertension", "target_col": "target_hypertension", "mode": "mode_b", "file": "nhanes_hypertension_mode_b.parquet"},
]


def get_models():
    """Return dictionary of models to benchmark."""
    models = {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "logistic_regression_balanced": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        "random_forest_balanced": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42))
        ]),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=42),
        "hist_gradient_boosting_balanced": HistGradientBoostingClassifier(class_weight="balanced", random_state=42),
    }
    return models


def compute_metrics(y_true, y_prob, threshold=0.50):
    """Compute classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    logloss = log_loss(y_true, y_prob)
    
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "specificity": spec,
        "brier": brier,
        "log_loss": logloss,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "mean_pred_prob": float(np.mean(y_prob)),
        "observed_pos_rate": float(np.mean(y_true)),
    }


def extract_feature_importance(model, model_name, feature_names, X_val, y_val):
    """Extract or compute feature importances."""
    if "logistic_regression" in model_name:
        clf = model.named_steps["clf"]
        coefs = np.abs(clf.coef_[0])
        df_fi = pd.DataFrame({"feature": feature_names, "importance": coefs})
    elif "random_forest" in model_name:
        clf = model.named_steps["clf"]
        importances = clf.feature_importances_
        df_fi = pd.DataFrame({"feature": feature_names, "importance": importances})
    elif "hist_gradient_boosting" in model_name:
        res = permutation_importance(model, X_val, y_val, n_repeats=5, random_state=42, n_jobs=-1)
        importances = res.importances_mean
        df_fi = pd.DataFrame({"feature": feature_names, "importance": importances})
    else:
        df_fi = pd.DataFrame({"feature": feature_names, "importance": np.zeros(len(feature_names))})
        
    df_fi = df_fi.sort_values(by="importance", ascending=False).reset_index(drop=True)
    return df_fi


def generate_plots(config_name, results_map, X_val, y_val, X_test, y_test, best_model_name, best_model):
    """Generate ROC, PR, Calibration, and Confusion Matrix plots."""
    # 1. ROC Curves
    plt.figure(figsize=(8, 6))
    for mname, res in results_map.items():
        fpr, tpr, _ = roc_curve(y_val, res["val_probs"])
        plt.plot(fpr, tpr, label=f"{mname} (AUC = {res['metrics_val']['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title(f"ROC Curves (Validation) — {config_name.upper()}")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{config_name}_roc.png", dpi=300)
    plt.close()

    # 2. PR Curves
    plt.figure(figsize=(8, 6))
    for mname, res in results_map.items():
        prec, rec, _ = precision_recall_curve(y_val, res["val_probs"])
        plt.plot(rec, prec, label=f"{mname} (PR-AUC = {res['metrics_val']['pr_auc']:.3f})")
    baseline = y_val.mean()
    plt.axhline(baseline, color="k", linestyle="--", alpha=0.5, label=f"Prevalence ({baseline:.3f})")
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision (PPV)")
    plt.title(f"Precision-Recall Curves (Validation) — {config_name.upper()}")
    plt.legend(loc="upper right", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{config_name}_pr.png", dpi=300)
    plt.close()

    # 3. Calibration Curves
    plt.figure(figsize=(8, 6))
    for mname, res in results_map.items():
        prob_true, prob_pred = calibration_curve(y_val, res["val_probs"], n_bins=10)
        plt.plot(prob_pred, prob_true, marker="o", label=f"{mname} (Brier = {res['metrics_val']['brier']:.4f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly Calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Fraction of Positives")
    plt.title(f"Calibration Curves (Validation) — {config_name.upper()}")
    plt.legend(loc="upper left", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{config_name}_calibration.png", dpi=300)
    plt.close()

    # 4. Confusion Matrix for Best Model on Final Test Set
    test_probs = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else best_model.decision_function(X_test)
    test_preds = (test_probs >= 0.50).astype(int)
    cm = confusion_matrix(y_test, test_preds, labels=[0, 1])
    
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.title(f"Confusion Matrix (Test Set)\nBest: {best_model_name} — {config_name.upper()}")
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ["Negative (0)", "Positive (1)"])
    plt.yticks(tick_marks, ["Negative (0)", "Positive (1)"])
    
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=14, fontweight="bold")
            
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label (Threshold = 0.50)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{config_name}_confusion_matrix.png", dpi=300)
    plt.close()


def run_stage_1d():
    log.info("Starting Stage 1D — Model Benchmarking & Honest Baselines")
    
    # Store split SEQN assignments per target
    target_splits = {}
    
    # Generate reproducible splits for each target first
    for target_info in [("cvd", "target_cvd", "nhanes_cvd_mode_a.parquet"),
                        ("diabetes", "target_diabetes", "nhanes_diabetes_mode_a.parquet"),
                        ("hypertension", "target_hypertension", "nhanes_hypertension_mode_a.parquet")]:
        t_name, t_col, f_name = target_info
        df = pd.read_parquet(PROCESSED_DIR / f_name)
        df_valid = df.dropna(subset=[t_col]).copy()
        
        # Perform 70/15/15 stratified split on SEQN
        seqns = df_valid[["SEQN", t_col]]
        train_seqn, temp_seqn = train_test_split(
            seqns, test_size=0.30, random_state=42, stratify=seqns[t_col]
        )
        val_seqn, test_seqn = train_test_split(
            temp_seqn, test_size=0.50, random_state=42, stratify=temp_seqn[t_col]
        )
        
        split_df = pd.DataFrame({
            "SEQN": df_valid["SEQN"],
            t_col: df_valid[t_col],
            "split": "train"
        })
        split_df.loc[split_df["SEQN"].isin(val_seqn["SEQN"]), "split"] = "validation"
        split_df.loc[split_df["SEQN"].isin(test_seqn["SEQN"]), "split"] = "test"
        
        split_path = SPLITS_DIR / f"{t_name}_splits.csv"
        split_df.to_csv(split_path, index=False)
        target_splits[t_name] = split_df
        log.info(f"Splits saved for {t_name}: Train={len(train_seqn)}, Val={len(val_seqn)}, Test={len(test_seqn)}")

    results_rows = []
    summary_markdown_sections = []
    best_models_dict = {}

    for config in CONFIGS:
        target_name = config["target_name"]
        target_col = config["target_col"]
        mode = config["mode"]
        filename = config["file"]
        config_key = f"{target_name}_{mode}"
        
        log.info(f"\n==========================================")
        log.info(f"Benchmarking Dataset: {filename} ({config_key})")
        log.info(f"==========================================")
        
        df = pd.read_parquet(PROCESSED_DIR / filename)
        df = df.dropna(subset=[target_col]).copy()
        
        # Load split SEQN map
        splits_df = target_splits[target_name]
        train_seqns = set(splits_df[splits_df["split"] == "train"]["SEQN"])
        val_seqns = set(splits_df[splits_df["split"] == "validation"]["SEQN"])
        test_seqns = set(splits_df[splits_df["split"] == "test"]["SEQN"])
        
        df_train = df[df["SEQN"].isin(train_seqns)].copy()
        df_val = df[df["SEQN"].isin(val_seqns)].copy()
        df_test = df[df["SEQN"].isin(test_seqns)].copy()
        
        # Predictor columns
        feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS and c != target_col]
        
        # Critical Leakage Check
        for c in feature_cols:
            assert c not in EXCLUDE_COLS, f"LEAKAGE ERROR: Excluded column {c} in features!"
            assert c != target_col, f"LEAKAGE ERROR: Target {c} in features!"
            assert not c.startswith("MCQ160") if target_name == "cvd" else True
            assert c not in ["hba1c", "fasting_glucose", "DIQ010", "DIQ050", "DIQ070"] if target_name == "diabetes" else True
            assert c not in ["mean_sbp", "mean_dbp", "BPQ020"] if target_name == "hypertension" else True

        X_train = df_train[feature_cols]
        y_train = df_train[target_col].values
        
        X_val = df_val[feature_cols]
        y_val = df_val[target_col].values
        
        X_test = df_test[feature_cols]
        y_test = df_test[target_col].values
        
        log.info(f"Feature count: {len(feature_cols)}. Train N: {len(X_train)}, Val N: {len(X_val)}, Test N: {len(X_test)}")
        
        models_dict = get_models()
        config_results = {}
        
        for mname, model in models_dict.items():
            # Fit strictly on train
            model.fit(X_train, y_train)
            
            # Save model artifact
            model_filename = f"{config_key}_{mname}.joblib"
            joblib.dump(model, MODELS_DIR / model_filename)
            
            # Predict on val
            val_probs = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_val)
            val_metrics = compute_metrics(y_val, val_probs)
            
            # Predict on test
            test_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
            test_metrics = compute_metrics(y_test, test_probs)
            
            config_results[mname] = {
                "model": model,
                "val_probs": val_probs,
                "test_probs": test_probs,
                "metrics_val": val_metrics,
                "metrics_test": test_metrics,
            }
            
            # Feature importance
            df_fi = extract_feature_importance(model, mname, feature_cols, X_val, y_val)
            df_fi.head(15).to_csv(FI_DIR / f"{config_key}_{mname}_top15_fi.csv", index=False)
            
            # Record result row
            results_rows.append({
                "target": target_name,
                "mode": mode,
                "model": mname,
                "train_n": len(X_train),
                "validation_n": len(X_val),
                "test_n": len(X_test),
                "roc_auc_validation": val_metrics["roc_auc"],
                "pr_auc_validation": val_metrics["pr_auc"],
                "accuracy_validation": val_metrics["accuracy"],
                "precision_validation": val_metrics["precision"],
                "recall_validation": val_metrics["recall"],
                "f1_validation": val_metrics["f1"],
                "specificity_validation": val_metrics["specificity"],
                "brier_validation": val_metrics["brier"],
                "roc_auc_test": test_metrics["roc_auc"],
                "pr_auc_test": test_metrics["pr_auc"],
                "accuracy_test": test_metrics["accuracy"],
                "precision_test": test_metrics["precision"],
                "recall_test": test_metrics["recall"],
                "f1_test": test_metrics["f1"],
                "specificity_test": test_metrics["specificity"],
                "brier_test": test_metrics["brier"],
            })
            
            log.info(f"[{mname:32s}] Val ROC-AUC: {val_metrics['roc_auc']:.4f} | PR-AUC: {val_metrics['pr_auc']:.4f} | Test ROC-AUC: {test_metrics['roc_auc']:.4f} | PR-AUC: {test_metrics['pr_auc']:.4f}")

        # Pick best model by validation PR-AUC / ROC-AUC sum or ROC-AUC
        # We rank by validation ROC-AUC primary, PR-AUC secondary
        best_mname = max(config_results.keys(), key=lambda k: (config_results[k]["metrics_val"]["roc_auc"], config_results[k]["metrics_val"]["pr_auc"]))
        best_model = config_results[best_mname]["model"]
        best_models_dict[config_key] = (best_mname, config_results[best_mname])
        
        log.info(f"--> BEST model for {config_key}: {best_mname} (Val ROC-AUC: {config_results[best_mname]['metrics_val']['roc_auc']:.4f})")
        
        # Generate visual plots
        generate_plots(config_key, config_results, X_val, y_val, X_test, y_test, best_mname, best_model)

    # Save benchmark results CSV
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(OUTPUT_DIR / "benchmark_results.csv", index=False)
    log.info(f"\nSaved benchmark_results.csv ({len(results_df)} benchmark runs)")

    # Build Summary Markdown
    summary_md = "# Model V2 Baseline Benchmark Summary\n\n"
    summary_md += "> Stage 1D — Model Benchmarking across 6 configurations (NHANES 2021–2023)\n\n"
    
    for config in CONFIGS:
        ckey = f"{config['target_name']}_{config['mode']}"
        cdf = results_df[(results_df["target"] == config["target_name"]) & (results_df["mode"] == config["mode"])].copy()
        cdf = cdf.sort_values(by="roc_auc_validation", ascending=False)
        
        summary_md += f"## {ckey.upper()}\n\n"
        summary_md += "| Rank | Model | Val ROC-AUC | Val PR-AUC | Val Recall | Val Specificity | Test ROC-AUC | Test PR-AUC |\n"
        summary_md += "|---|---|---|---|---|---|---|---|\n"
        for rank, (_, r) in enumerate(cdf.iterrows(), start=1):
            summary_md += f"| {rank} | {r['model']} | {r['roc_auc_validation']:.4f} | {r['pr_auc_validation']:.4f} | {r['recall_validation']:.4f} | {r['specificity_validation']:.4f} | {r['roc_auc_test']:.4f} | {r['pr_auc_test']:.4f} |\n"
        summary_md += "\n"

    with open(OUTPUT_DIR / "benchmark_summary.md", "w") as f:
        f.write(summary_md)
    log.info("Saved benchmark_summary.md")

    # Create EXPERIMENT_MANIFEST.json
    import sklearn
    manifest = {
        "stage": "1D",
        "random_state": 42,
        "environment": {
            "python_version": sys.version,
            "sklearn_version": sklearn.__version__,
            "xgboost": "SKIPPED (not installed)",
            "lightgbm": "SKIPPED (not installed)",
        },
        "configs_benchmarked": [f"{c['target_name']}_{c['mode']}" for c in CONFIGS],
        "best_models_val_roc_auc": {k: {"model": v[0], "val_roc_auc": v[1]["metrics_val"]["roc_auc"], "test_roc_auc": v[1]["metrics_test"]["roc_auc"]} for k, v in best_models_dict.items()}
    }
    with open(OUTPUT_DIR / "EXPERIMENT_MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Saved EXPERIMENT_MANIFEST.json")

    return results_df, best_models_dict


if __name__ == "__main__":
    run_stage_1d()
