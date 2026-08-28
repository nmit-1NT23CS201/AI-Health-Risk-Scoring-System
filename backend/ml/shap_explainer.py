import os
from typing import List, Tuple

import joblib
import numpy as np
import shap
from sklearn.pipeline import Pipeline


def load_model(model_path: str) -> Pipeline:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    try:
        model = joblib.load(model_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model: {exc}") from exc

    return model


def _get_feature_names(preprocessor, fallback_count: int) -> List[str]:
    try:
        names = preprocessor.get_feature_names_out()
        return [str(name) for name in names]
    except Exception:
        return [f"feature_{idx}" for idx in range(fallback_count)]


def prepare_shap_inputs(
    pipeline: Pipeline, X
) -> Tuple[np.ndarray, List[str]]:
    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is None:
        raise ValueError("Pipeline missing 'preprocessor' step.")

    X_processed = preprocessor.transform(X)
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    feature_names = _get_feature_names(preprocessor, X_processed.shape[1])
    return X_processed, feature_names


def compute_shap_values(
    pipeline: Pipeline, X_processed: np.ndarray
) -> Tuple[np.ndarray, float]:
    model = pipeline.named_steps.get("model")
    if model is None:
        raise ValueError("Pipeline missing 'model' step.")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_processed)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(np.array(base_value).ravel()[0])

    return np.array(shap_values), float(base_value)
