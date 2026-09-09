"""Quick inspection script for Stage 1E-2 planning."""
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("backend/ml/data/processed/nhanes_2021_2023")
SPLITS_DIR = PROCESSED_DIR / "splits"
EXCLUDE_COLS = ["SEQN", "WTINT2YR", "WTMEC2YR", "WTSAF2YR", "SDMVSTRA", "SDMVPSU"]

configs = [
    ("cvd", "target_cvd", "nhanes_cvd_mode_a.parquet", "cvd_splits.csv", "mode_a"),
    ("cvd", "target_cvd", "nhanes_cvd_mode_b.parquet", "cvd_splits.csv", "mode_b"),
    ("diabetes", "target_diabetes", "nhanes_diabetes_mode_a.parquet", "diabetes_splits.csv", "mode_a"),
    ("diabetes", "target_diabetes", "nhanes_diabetes_mode_b.parquet", "diabetes_splits.csv", "mode_b"),
    ("hypertension", "target_hypertension", "nhanes_hypertension_mode_a.parquet", "hypertension_splits.csv", "mode_a"),
    ("hypertension", "target_hypertension", "nhanes_hypertension_mode_b.parquet", "hypertension_splits.csv", "mode_b"),
]

for t_name, t_col, fname, split_fname, mode in configs:
    df = pd.read_parquet(PROCESSED_DIR / fname)
    df = df.dropna(subset=[t_col])
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS and c != t_col]

    splits_df = pd.read_csv(SPLITS_DIR / split_fname)
    train_seqns = set(splits_df[splits_df["split"] == "train"]["SEQN"])
    val_seqns   = set(splits_df[splits_df["split"] == "validation"]["SEQN"])
    test_seqns  = set(splits_df[splits_df["split"] == "test"]["SEQN"])

    df_train = df[df["SEQN"].isin(train_seqns)]
    df_val   = df[df["SEQN"].isin(val_seqns)]
    df_test  = df[df["SEQN"].isin(test_seqns)]
    df_dev   = df[df["SEQN"].isin(train_seqns | val_seqns)]

    y_train = df_train[t_col].values
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)

    X_sample = df_train[feature_cols]
    has_object = any(d == object for d in X_sample.dtypes)
    nan_pct = X_sample.isnull().mean().mean()

    config_key = f"{t_name}_{mode}"
    print(f"\n{config_key}: features={len(feature_cols)}, train={len(df_train)}, val={len(df_val)}, dev={len(df_dev)}, test={len(df_test)}, pos={n_pos}, neg={n_neg}, has_object={has_object}, nan_pct={nan_pct:.3f}")
    print(f"  Columns: {feature_cols}")
    if has_object:
        obj_cols = X_sample.select_dtypes(include=["object"]).columns.tolist()
        print(f"  Object cols: {obj_cols}")
