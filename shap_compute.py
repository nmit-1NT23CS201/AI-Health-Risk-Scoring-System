"""
shap_compute.py
================
Computes REAL mean(|SHAP value|) global feature importance for the six final
V2 disease-mode pipelines on the held-out 15% participant-level test set, and
writes shap_global_values.csv and shap_global_values.json.

METHODOLOGICAL PRINCIPLES:
1. Evaluation is conducted strictly on the 15% participant-level held-out test
   partitions (NHANES 2021-2023), matching the exact test evaluations reported
   in the research paper. Training and validation observations are never used.
2. The SHAP explanations evaluate the underlying predictive base classifiers
   (Random Forest, HistGradientBoosting, XGBoost, Logistic Regression) rather
   than the monotonic post-hoc calibration mapping (Platt/sigmoid scaling).
3. Base estimators wrapped in CalibratedClassifierCV -> FrozenEstimator (scikit-learn 1.9.0)
   are extracted cleanly to access the underlying models.
4. For pipelines containing scikit-learn Pipeline objects (CVD A/B, Hypertension A/B),
   preprocessing steps (median imputation, standard scaling) are applied to the
   held-out test predictors prior to tree/linear explanation, preserving exact
   1-to-1 feature alignment without altering original feature names.
5. Standalone tree estimators with native missing-value support (Diabetes A HistGB,
   Diabetes B XGBoost) receive the raw predictor matrix directly.
6. The exact 6 survey/metadata design columns (SEQN, WTINT2YR, WTMEC2YR, WTSAF2YR,
   SDMVSTRA, SDMVPSU) and disease target columns are excluded from predictors.

OUTPUTS:
  shap_global_values.csv   (columns: target, mode, feature, mean_abs_shap, rank, n_test)
  shap_global_values.json  (list of records, 60 items: 6 pipelines x top 10 features)
"""

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

try:
    import shap
except ImportError:
    sys.exit("Missing dependency: pip install shap")

# Base project paths
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "backend" / "ml" / "data" / "processed" / "nhanes_2021_2023"
SPLITS_DIR = PROCESSED_DIR / "splits"
MODELS_DIR = BASE_DIR / "backend" / "ml" / "evaluation" / "stage_1f" / "models"

# 6 non-predictor survey / participant identifiers
EXCLUDE_COLS = [
    "SEQN",
    "WTINT2YR",
    "WTMEC2YR",
    "WTSAF2YR",
    "SDMVSTRA",
    "SDMVPSU",
]

# ============================================================================
# CONFIG -- Locked final V2 artifacts and test datasets verified by repository audit
# ============================================================================
CONFIG = [
    dict(
        target="CVD",
        mode="A",
        model_name="Random Forest",
        model_path=MODELS_DIR / "cvd_mode_a_calibrated.joblib",
        preprocessor_path=None,  # Embedded in Pipeline
        data_path=PROCESSED_DIR / "nhanes_cvd_mode_a.parquet",
        split_path=SPLITS_DIR / "cvd_splits.csv",
        target_col="target_cvd",
        id_cols=EXCLUDE_COLS,
        model_kind="tree",
        calibration="calibrated_sigmoid",
    ),
    dict(
        target="CVD",
        mode="B",
        model_name="Random Forest",
        model_path=MODELS_DIR / "cvd_mode_b_calibrated.joblib",
        preprocessor_path=None,  # Embedded in Pipeline
        data_path=PROCESSED_DIR / "nhanes_cvd_mode_b.parquet",
        split_path=SPLITS_DIR / "cvd_splits.csv",
        target_col="target_cvd",
        id_cols=EXCLUDE_COLS,
        model_kind="tree",
        calibration="calibrated_sigmoid",
    ),
    dict(
        target="Diabetes",
        mode="A",
        model_name="HistGradientBoosting",
        model_path=MODELS_DIR / "diabetes_mode_a_calibrated.joblib",
        preprocessor_path=None,  # Native missing-value support
        data_path=PROCESSED_DIR / "nhanes_diabetes_mode_a.parquet",
        split_path=SPLITS_DIR / "diabetes_splits.csv",
        target_col="target_diabetes",
        id_cols=EXCLUDE_COLS,
        model_kind="tree",
        calibration="calibrated_sigmoid",
    ),
    dict(
        target="Diabetes",
        mode="B",
        model_name="XGBoost",
        model_path=MODELS_DIR / "diabetes_mode_b_calibrated.joblib",
        preprocessor_path=None,  # Native missing-value support
        data_path=PROCESSED_DIR / "nhanes_diabetes_mode_b.parquet",
        split_path=SPLITS_DIR / "diabetes_splits.csv",
        target_col="target_diabetes",
        id_cols=EXCLUDE_COLS,
        model_kind="tree",
        calibration="calibrated_sigmoid",
    ),
    dict(
        target="Hypertension",
        mode="A",
        model_name="Logistic Regression",
        model_path=MODELS_DIR / "hypertension_mode_a_calibrated.joblib",
        preprocessor_path=None,  # Embedded in Pipeline
        data_path=PROCESSED_DIR / "nhanes_hypertension_mode_a.parquet",
        split_path=SPLITS_DIR / "hypertension_splits.csv",
        target_col="target_hypertension",
        id_cols=EXCLUDE_COLS,
        model_kind="linear",
        calibration="calibrated_sigmoid",
    ),
    dict(
        target="Hypertension",
        mode="B",
        model_name="Random Forest",
        model_path=MODELS_DIR / "hypertension_mode_b_calibrated.joblib",
        preprocessor_path=None,  # Embedded in Pipeline
        data_path=PROCESSED_DIR / "nhanes_hypertension_mode_b.parquet",
        split_path=SPLITS_DIR / "hypertension_splits.csv",
        target_col="target_hypertension",
        id_cols=EXCLUDE_COLS,
        model_kind="tree",
        calibration="calibrated_sigmoid",
    ),
]

TOP_N = 10
BACKGROUND_SAMPLE = 200  # background samples for LinearExplainer masker
RANDOM_STATE = 42


def extract_base_estimator(model, calibration: str):
    """
    Return the underlying predictive estimator SHAP should actually explain.

    Per the research methodology: SHAP explains the base predictive classifier
    prior to post-hoc calibration. In scikit-learn 1.9.0, CalibratedClassifierCV
    wraps the base estimator inside a FrozenEstimator object:
      CalibratedClassifierCV -> calibrated_classifiers_[0] -> FrozenEstimator -> base_estimator
    This function cleanly unwraps any FrozenEstimator layers to access the actual
    Pipeline or standalone classifier.
    """
    if calibration == "none":
        return model

    if calibration == "calibrated_sigmoid":
        if hasattr(model, "calibrated_classifiers_") and len(model.calibrated_classifiers_) > 0:
            cc = model.calibrated_classifiers_[0]
            est = None
            for attr in ("estimator", "base_estimator"):
                if hasattr(cc, attr):
                    est = getattr(cc, attr)
                    break
            if est is None:
                raise AttributeError(
                    "Could not find base estimator on calibrated_classifiers_[0]."
                )

            # Unwrap scikit-learn FrozenEstimator wrapper(s)
            while hasattr(est, "estimator") and type(est).__name__ == "FrozenEstimator":
                est = est.estimator

            return est

        raise AttributeError(
            "calibration='calibrated_sigmoid' but model has no "
            "'calibrated_classifiers_' attribute."
        )

    raise ValueError(f"Unknown calibration setting: {calibration}")


def load_feature_matrix(cfg: dict):
    """
    Load the exact 15% held-out test partition for the given pipeline.

    - Reads the processed Parquet dataset.
    - Reads the verified disease-specific split CSV.
    - Subsets strictly to rows where split == 'test'.
    - Excludes the 6 survey/metadata design columns and disease target column.
    - Preserves exact feature column ordering and names.
    - Returns predictor DataFrame X_raw, list of feature names, and ground-truth y.
    """
    data_path = Path(cfg["data_path"])
    split_path = Path(cfg["split_path"])

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    target_col = cfg["target_col"]
    df = pd.read_parquet(data_path).dropna(subset=[target_col]).copy()
    splits_df = pd.read_csv(split_path)

    test_seqns = set(splits_df[splits_df["split"] == "test"]["SEQN"])
    df_test = df[df["SEQN"].isin(test_seqns)].copy()

    drop_cols = set(cfg["id_cols"]) | {target_col}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X_raw = df_test[feature_cols].copy()
    y = df_test[target_col].values.astype(int)
    feature_names = list(feature_cols)

    return X_raw, feature_names, y


def prepare_inputs_and_classifier(est, X_raw: pd.DataFrame, feature_names: list[str]):
    """
    Separate preprocessing from classifier if the base estimator is a Pipeline.

    - For Pipeline models (CVD A/B, Hypertension A/B):
      Extracts est[:-1] as the preprocessor, transforms X_raw to imputed/scaled
      matrix X_in, and uses est[-1] as the classifier for SHAP.
      Because SimpleImputer and StandardScaler operate element-wise on features
      without dropping or generating one-hot encodings, original feature_names
      remain 1-to-1 preserved.
    - For standalone models (Diabetes A HistGB, Diabetes B XGBoost):
      Returns the classifier and raw predictor matrix directly.
    """
    if isinstance(est, Pipeline):
        preprocessor = est[:-1]
        clf = est[-1]
        X_in = preprocessor.transform(X_raw)
    else:
        clf = est
        X_in = X_raw

    return clf, X_in, feature_names


def compute_one(cfg: dict):
    """Compute SHAP global feature importances for a single pipeline."""
    model_path = Path(cfg["model_path"])
    if not model_path.exists():
        print(f"[SKIP] Model file not found: {model_path}")
        return None

    print(f"\n[RUN ] {cfg['target']} - Mode {cfg['mode']} ({cfg['model_name']})")
    model = joblib.load(model_path)
    X_raw, feature_names, y = load_feature_matrix(cfg)
    n_test = len(X_raw)

    est = extract_base_estimator(model, cfg["calibration"])
    clf, X_in, feature_names = prepare_inputs_and_classifier(est, X_raw, feature_names)

    print(f"       Classifier: {type(clf).__name__} | Held-out test rows: {n_test} | Features: {len(feature_names)}")

    if cfg["model_kind"] == "tree":
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_in)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]  # Positive class SHAP
        sv = np.asarray(sv)
        if sv.ndim == 3:  # (n_samples, n_features, n_classes)
            sv = sv[:, :, 1]
    elif cfg["model_kind"] == "linear":
        bg_n = min(BACKGROUND_SAMPLE, n_test)
        bg = shap.sample(X_in, bg_n, random_state=RANDOM_STATE)
        explainer = shap.LinearExplainer(clf, bg)
        sv = np.asarray(explainer.shap_values(X_in))
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        if sv.ndim == 3:
            sv = sv[:, :, 1]
    else:
        raise ValueError(f"Unknown model_kind: {cfg['model_kind']}")

    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(-mean_abs)[:TOP_N]

    rows = []
    for rank, idx in enumerate(order, start=1):
        rows.append(dict(
            target=cfg["target"],
            mode=cfg["mode"],
            feature=feature_names[idx],
            mean_abs_shap=float(mean_abs[idx]),
            rank=rank,
            n_test=n_test,
        ))

    print(f"       Top 3 features: {[r['feature'] for r in rows[:3]]}")
    return rows


def main():
    print("=" * 80)
    print("AI HEALTH RISK SCORING SYSTEM: GLOBAL SHAP FEATURE IMPORTANCE COMPUTATION")
    print("=" * 80)

    all_rows = []
    for cfg in CONFIG:
        rows = compute_one(cfg)
        if rows:
            all_rows.extend(rows)

    if not all_rows:
        sys.exit("Error: No pipelines produced output. Check file paths.")

    df = pd.DataFrame(all_rows)
    csv_out = BASE_DIR / "shap_global_values.csv"
    json_out = BASE_DIR / "shap_global_values.json"

    df.to_csv(csv_out, index=False)
    with open(json_out, "w") as f:
        json.dump(all_rows, f, indent=2)

    n_models = df.groupby(["target", "mode"]).ngroups
    print("\n" + "=" * 80)
    print(f"COMPLETED: Wrote {csv_out.name} and {json_out.name}")
    print(f"Total rows: {len(df)} across {n_models}/6 disease-mode pipelines.")
    print("=" * 80)

    if n_models < 6 or len(df) != 60:
        print("WARNING: Expected exactly 60 rows across 6 pipelines. Review skipped items above.")


if __name__ == "__main__":
    main()