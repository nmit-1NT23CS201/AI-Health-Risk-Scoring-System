import os
from typing import Tuple

import pandas as pd


def load_dataset(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    print("Dataset loaded successfully.")
    print(f"Shape: {df.shape}")
    print("Columns:")
    print(df.columns.tolist())
    print("Summary:")
    print(df.describe(include="all"))

    missing_count = df.isna().sum().sum()
    if missing_count:
        print(f"Warning: dataset contains {missing_count} missing values.")

    return df
