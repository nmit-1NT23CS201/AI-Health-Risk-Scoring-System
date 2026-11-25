"""Live, per-request SHAP on the unwrapped base estimator (never the calibrated output).

Unwrapping mirrors ``shap_compute.py`` (CalibratedClassifierCV -> calibrated_classifiers_[0]
-> FrozenEstimator -> Pipeline | bare estimator). Values are additive in the base model's own
output space (RF: uncalibrated probability; HGB/XGB/LR: log-odds). They are NOT additive in the
calibrated probability. No matplotlib/pyplot is used and no precomputed SHAP files are read.
"""
from __future__ import annotations

import threading
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from backend.api.v2 import config

_RF = "RandomForestClassifier"
_LR = "LogisticRegression"
_HGB = "HistGradientBoostingClassifier"
_XGB = "XGBClassifier"


def unwrap_base_estimator(model):
    cc = model.calibrated_classifiers_[0]
    est = getattr(cc, "estimator", None)
    if est is None:
        est = getattr(cc, "base_estimator")
    while type(est).__name__ == "FrozenEstimator":
        est = est.estimator
    return est


def split_pipeline(est):
    if isinstance(est, Pipeline):
        return est[:-1], est[-1]
    return None, est


def reference_output(clf, x_in) -> np.ndarray:
    """The base model's own output in the space SHAP values add up in."""
    name = type(clf).__name__
    if name == _RF:
        return clf.predict_proba(x_in)[:, 1]
    if name in (_LR, _HGB):
        return np.asarray(clf.decision_function(x_in))
    if name == _XGB:
        return np.asarray(clf.predict(x_in, output_margin=True))
    raise ValueError(f"Unsupported base classifier for SHAP: {name}")


def _class1(sv) -> np.ndarray:
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv


def load_htn_a_background(features: list[str]) -> pd.DataFrame:
    """Persisted training-split background (raw, pre-imputation). Verifies split hygiene."""
    path = config.HTN_A_BACKGROUND_FILE
    if not path.exists():
        raise FileNotFoundError(f"SHAP background asset missing: {path}")
    bg = pd.read_parquet(path)
    if list(bg.columns) != ["SEQN", *features]:
        raise ValueError("Background asset columns do not match registry features for hypertension_mode_a")
    if len(bg) != config.HTN_A_BACKGROUND_SIZE:
        raise ValueError(f"Background asset has {len(bg)} rows, expected {config.HTN_A_BACKGROUND_SIZE}")
    splits = pd.read_csv(config.SPLITS_DIR / "hypertension_splits.csv")
    split_of = dict(zip(splits["SEQN"], splits["split"]))
    labels = {split_of.get(s) for s in bg["SEQN"]}
    if labels != {"train"}:
        raise ValueError(f"Background rows must all be in the training split; found {sorted(map(str, labels))}")
    test_seqns = set(splits.loc[splits["split"] == "test", "SEQN"])
    if test_seqns & set(bg["SEQN"]):
        raise ValueError("Background intersects the test split")
    return bg[features].astype("float64")


@dataclass(frozen=True)
class RawExplanation:
    values: np.ndarray  # one SHAP value per feature (features order)
    base_value: float
    base_model_output: float
    method: str
    explained_estimator: str
    output_space: str


class ConfigExplainer:
    def __init__(self, model, features: list[str], background_raw: pd.DataFrame | None = None):
        self._features = list(features)
        est = unwrap_base_estimator(model)
        self._preproc, self._clf = split_pipeline(est)
        self.explained_estimator = type(self._clf).__name__
        self._lock = threading.Lock()
        if self.explained_estimator == _LR:
            if background_raw is None:
                raise ValueError("LinearExplainer requires a training-split background")
            bg = self._transform(background_raw)
            masker = shap.maskers.Independent(bg, max_samples=len(bg))  # default would subsample to 100
            self._explainer = shap.LinearExplainer(self._clf, masker)
            self.method = "LinearExplainer"
        elif self.explained_estimator in (_RF, _HGB, _XGB):
            self._explainer = shap.TreeExplainer(self._clf)
            self.method = "TreeExplainer"
        else:
            raise ValueError(f"Unsupported base classifier for SHAP: {self.explained_estimator}")
        self.output_space = "uncalibrated_rf_probability" if self.explained_estimator == _RF else "log_odds_margin"

    def _transform(self, x_raw: pd.DataFrame):
        return self._preproc.transform(x_raw) if self._preproc is not None else x_raw

    def explain(self, x_raw: pd.DataFrame) -> RawExplanation:
        with self._lock, warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            x_in = self._transform(x_raw)
            values = _class1(self._explainer.shap_values(x_in))[0].astype(np.float64)  # XGBoost returns float32
            # Read AFTER shap_values: for XGBoost, shap updates expected_value on the first call
            # (observed: -1.5345 at construction vs -1.5177 afterwards, the latter matching XGBoost's bias).
            base_value = float(np.ravel(self._explainer.expected_value)[-1])
        base_out = base_value + float(values.sum())
        return RawExplanation(values, base_value, base_out, self.method, self.explained_estimator, self.output_space)

    def additivity_error(self, x_raw: pd.DataFrame) -> float:
        """|base_value + sum(shap) - base model output| for one row."""
        raw = self.explain(x_raw)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            ref = float(reference_output(self._clf, self._transform(x_raw))[0])
        return abs(raw.base_model_output - ref)

