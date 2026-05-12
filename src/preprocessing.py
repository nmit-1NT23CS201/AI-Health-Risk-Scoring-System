from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _normalize_yes_no(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .replace({"yes": "Yes", "no": "No"})
    )


def build_preprocessor(
    X: pd.DataFrame, categorical_cols: List[str], numeric_cols: List[str]
) -> ColumnTransformer:
    X = X.copy()

    for col in categorical_cols:
        X[col] = _normalize_yes_no(X[col])

    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical_cols),
            ("numeric", numeric_pipeline, numeric_cols),
        ]
    )

    return preprocessor


def get_feature_groups(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = [col for col in X.columns if col not in categorical_cols]
    return categorical_cols, numeric_cols
