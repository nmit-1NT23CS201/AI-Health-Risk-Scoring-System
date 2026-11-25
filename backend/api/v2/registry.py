"""Model registry: loads the six calibrated artifacts described by stage_1f_results.json.

Everything model-specific (features, order, thresholds, offline metrics) comes from that JSON.
Hard assertion failures raise ``RegistryLoadError`` (V2 becomes unavailable; V1 is unaffected).
Per-config SHAP problems are soft: predictions stay available, explanations are disabled.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any

import joblib
import numpy as np
import pandas as pd

from backend.api.v2 import config
from backend.api.v2.errors import RegistryLoadError
from backend.api.v2.explain import ConfigExplainer, load_htn_a_background
from backend.api.v2.preprocessing import FEATURE_SPECS
from backend.api.v2.schemas import ModeBInputs

log = logging.getLogger("backend.api.v2")

DISEASES = ("cvd", "diabetes", "hypertension")
MODES = ("A", "B")
EXPECTED_CONFIG_KEYS = {f"{d}_mode_{m.lower()}" for d in DISEASES for m in MODES}


def log_event(event: str, level: int = logging.INFO, **fields: Any) -> None:
    log.log(level, json.dumps({"event": event, **fields}, default=str))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_artifact(backend_dir: Path, recorded: str) -> Path:
    """artifact_paths in the JSON are backend-relative with Windows separators."""
    return backend_dir.joinpath(*PureWindowsPath(recorded).parts)


@dataclass(frozen=True)
class ModelBundle:
    config_key: str
    disease: str
    mode: str
    model_name: str
    features: tuple[str, ...]
    threshold: float
    model: Any
    sha256: str
    artifact_path: Path
    record: dict  # JSON record (metrics etc.)
    explainer: ConfigExplainer | None = None
    explanation_error: str | None = None

    @property
    def model_version(self) -> str:
        return f"{config.MODEL_VERSION_PREFIX}/{self.config_key}"


@dataclass
class ModelRegistry:
    bundles: dict[tuple[str, str], ModelBundle]
    results_path: Path
    results_sha256: str
    sklearn_expected: str
    versions_runtime: dict[str, str]
    loaded_at: float = field(default_factory=time.time)

    def get(self, disease: str, mode: str) -> ModelBundle:
        return self.bundles[(disease, mode)]

    def all(self) -> list[ModelBundle]:
        return [self.bundles[(d, m)] for d in DISEASES for m in MODES]


def _runtime_versions() -> dict[str, str]:
    import fastapi
    import pydantic
    import shap
    import sklearn
    import xgboost

    return {
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "shap": shap.__version__,
        "fastapi": fastapi.__version__,
        "pydantic": pydantic.__version__,
    }


def _check_record(rec: dict, reasons: list[str]) -> bool:
    key = rec.get("config_key", "<missing config_key>")
    ok = True

    def fail(msg: str) -> None:
        nonlocal ok
        ok = False
        reasons.append(f"{key}: {msg}")

    feats = rec.get("features")
    if not isinstance(feats, list) or not feats:
        fail("missing/empty 'features'")
        return False
    if rec.get("n_features") != len(feats):
        fail(f"n_features={rec.get('n_features')} != len(features)={len(feats)}")
    if len(set(feats)) != len(feats):
        fail("duplicate feature names")
    bad = [f for f in feats if f in config.SURVEY_METADATA_COLUMNS or f.startswith("target_")]
    if bad:
        fail(f"survey/target columns present in features: {bad}")
    no_encoder = [f for f in feats if f not in FEATURE_SPECS or f not in ModeBInputs.model_fields]
    if no_encoder:
        fail(f"no encoder/schema field for features: {no_encoder}")
    thr = rec.get("selected_threshold")
    if not isinstance(thr, (int, float)) or not (0 < thr < 1) or math.isnan(thr):
        fail(f"invalid selected_threshold: {thr!r}")
    if rec.get("selected_calibration_method") != "sigmoid":
        fail(f"selected_calibration_method={rec.get('selected_calibration_method')!r}, expected 'sigmoid'")
    if not isinstance(rec.get("artifact_paths", {}).get("joblib"), str):
        fail("missing artifact_paths.joblib")
    return ok


def load_registry(results_path: Path | None = None, backend_dir: Path | None = None) -> ModelRegistry:
    results_path = Path(results_path) if results_path else config.results_json_path()
    backend_dir = Path(backend_dir) if backend_dir else config.BACKEND_DIR
    reasons: list[str] = []

    if not results_path.exists():
        raise RegistryLoadError([f"results file not found: {results_path}"])
    try:
        doc = json.loads(results_path.read_text(encoding="utf-8"))
    except Exception as exc:  # corrupt JSON
        raise RegistryLoadError([f"results file unreadable: {type(exc).__name__}: {exc}"]) from exc
    records = doc.get("results")
    if doc.get("stage") != "1F" or not isinstance(records, list):
        raise RegistryLoadError(["results file is not a Stage 1F results document"])
    keys = {r.get("config_key") for r in records}
    if len(records) != 6 or keys != EXPECTED_CONFIG_KEYS:
        raise RegistryLoadError([f"expected exactly the six configs {sorted(EXPECTED_CONFIG_KEYS)}, got {sorted(map(str, keys))}"])

    valid = [r for r in records if _check_record(r, reasons)]
    if reasons:
        raise RegistryLoadError(reasons)

    versions = _runtime_versions()
    expected_sk = str(doc.get("sklearn_version"))
    if versions["scikit-learn"] != expected_sk:
        log_event("sklearn_version_mismatch", logging.WARNING, expected=expected_sk, runtime=versions["scikit-learn"])

    bundles: dict[tuple[str, str], ModelBundle] = {}
    for rec in valid:
        key = rec["config_key"]
        disease = rec["target"]
        mode = rec["mode"].split("_")[-1].upper()
        features = list(rec["features"])
        path = resolve_artifact(backend_dir, rec["artifact_paths"]["joblib"])
        if not path.exists():
            reasons.append(f"{key}: artifact not found: {path}")
            continue
        sha = sha256_file(path)
        try:
            model = joblib.load(path)
        except Exception as exc:
            reasons.append(f"{key}: failed to deserialize {path.name}: {type(exc).__name__}: {exc}")
            continue
        problems = _check_model(model, rec, features)
        if problems:
            reasons.extend(f"{key}: {p}" for p in problems)
            continue

        explainer, expl_err = _build_explainer(model, features, key)
        bundles[(disease, mode)] = ModelBundle(
            config_key=key, disease=disease, mode=mode, model_name=rec["model_name"],
            features=tuple(features), threshold=float(rec["selected_threshold"]), model=model,
            sha256=sha, artifact_path=path, record=rec, explainer=explainer, explanation_error=expl_err,
        )
        log_event("model_loaded", config=key, sha256=sha, threshold=rec["selected_threshold"],
                  n_features=len(features), explanations=expl_err is None)
    if reasons:
        raise RegistryLoadError(reasons)

    return ModelRegistry(
        bundles=bundles, results_path=results_path, results_sha256=sha256_file(results_path),
        sklearn_expected=expected_sk, versions_runtime=versions,
    )


def _check_model(model, rec: dict, features: list[str]) -> list[str]:
    from sklearn.calibration import CalibratedClassifierCV

    p: list[str] = []
    if not isinstance(model, CalibratedClassifierCV):
        return [f"artifact is {type(model).__name__}, expected CalibratedClassifierCV"]
    if len(model.calibrated_classifiers_) != 1:
        p.append(f"expected 1 calibrated classifier, got {len(model.calibrated_classifiers_)}")
    if [int(c) for c in model.classes_] != [0, 1]:
        p.append(f"classes_ = {list(model.classes_)}, expected [0, 1]")
    if getattr(model.calibrated_classifiers_[0], "method", "sigmoid") != "sigmoid":
        p.append("calibrator is not sigmoid")
    if getattr(model, "n_features_in_", len(features)) != len(features):
        p.append(f"n_features_in_={model.n_features_in_} != {len(features)}")
    names = getattr(model, "feature_names_in_", None)
    if names is not None and list(names) != features:
        p.append("feature_names_in_ differs from registry features (order/name mismatch)")
    if p:
        return p
    # smoke test: all-NaN row is valid for native-NaN models and for median-imputing pipelines
    try:
        x = pd.DataFrame([[np.nan] * len(features)], columns=features, dtype="float64")
        pr = model.predict_proba(x)
        if pr.shape != (1, 2) or not np.isfinite(pr).all() or abs(float(pr.sum()) - 1.0) > 1e-9:
            p.append(f"smoke predict_proba returned unexpected output {pr!r}")
    except Exception as exc:
        p.append(f"smoke predict_proba failed: {type(exc).__name__}: {exc}")
    return p


def _build_explainer(model, features: list[str], key: str) -> tuple[ConfigExplainer | None, str | None]:
    try:
        bg = load_htn_a_background(features) if key == "hypertension_mode_a" else None
        explainer = ConfigExplainer(model, features, bg)
        x = pd.DataFrame([[np.nan] * len(features)], columns=features, dtype="float64")
        err = explainer.additivity_error(x)
        if err > config.ADDITIVITY_TOLERANCE:
            raise AssertionError(f"additivity self-check failed (error {err:.3e})")
        return explainer, None
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        log_event("explainer_unavailable", logging.ERROR, config=key, reason=reason)
        return None, reason
