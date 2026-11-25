"""One-off builder for the Hypertension Mode A SHAP background asset (training split only).

Reads (read-only) the frozen processed parquet + split file and writes
``backend/api/v2/assets/hypertension_mode_a_shap_background.parquet``: 200 rows drawn with
``random_state=42`` from rows with ``split == "train"`` and a non-missing target, in the
feature order from stage_1f_results.json, plus SEQN for the startup split-hygiene assertion.

Usage (from the repo root): python -m backend.api.v2.tools.build_hypertension_a_background
"""
from __future__ import annotations

import json

import pandas as pd

from backend.api.v2 import config

TARGET = "target_hypertension"


def main() -> None:
    results = json.loads(config.results_json_path().read_text(encoding="utf-8"))["results"]
    features = next(r["features"] for r in results if r["config_key"] == "hypertension_mode_a")
    processed = config.SPLITS_DIR.parent
    df = pd.read_parquet(processed / "nhanes_hypertension_mode_a.parquet").dropna(subset=[TARGET])
    splits = pd.read_csv(config.SPLITS_DIR / "hypertension_splits.csv")
    train_seqns = set(splits.loc[splits["split"] == "train", "SEQN"])
    train = df[df["SEQN"].isin(train_seqns)]
    bg = train.sample(n=config.HTN_A_BACKGROUND_SIZE, random_state=config.HTN_A_BACKGROUND_SEED)
    out = bg[["SEQN", *features]].reset_index(drop=True)
    config.ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(config.HTN_A_BACKGROUND_FILE, index=False)
    print(f"wrote {config.HTN_A_BACKGROUND_FILE} ({len(out)} rows from {len(train)} training rows)")


if __name__ == "__main__":
    main()
