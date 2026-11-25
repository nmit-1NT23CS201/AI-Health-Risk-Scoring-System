"""Orchestration: encode -> calibrated predict_proba -> locked threshold -> optional live SHAP."""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from backend.api.v2 import config, history
from backend.api.v2.explain import RawExplanation
from backend.api.v2.preprocessing import FEATURE_SPECS, EncodedInputs, build_frame, encode_inputs
from backend.api.v2.registry import ModelBundle, ModelRegistry
from backend.api.v2.schemas import (
    Disease,
    DiseaseResult,
    Explanation,
    FeatureAttribution,
    PredictRequestA,
    PredictRequestB,
    PredictResponse,
)

log = logging.getLogger("backend.api.v2")

_NEUTRAL_EPS = 1e-12
_NOTICE = (
    "Model-behaviour attributions for this submitted input, expressed in the explained base model's "
    "output space. Not causal, not probability percentage points, and not additive to the calibrated probability."
)
_REFERENCE = "NHANES 2021-2023 training split (model-internal reference)"


def predict_proba_frame(bundle: ModelBundle, frame: pd.DataFrame) -> np.ndarray:
    """Calibrated P(positive). The artifact already contains the sigmoid; never re-calibrate or use predict()."""
    return bundle.model.predict_proba(frame[list(bundle.features)])[:, 1]


def _attributions(bundle: ModelBundle, raw: RawExplanation, frame: pd.DataFrame, probability: float) -> Explanation:
    feats = list(bundle.features)
    order = np.argsort(-np.abs(raw.values))
    row = frame.iloc[0]
    items: list[FeatureAttribution] = []
    for rank, idx in enumerate(order[: config.TOP_K_FEATURES], start=1):
        name = feats[idx]
        v = float(raw.values[idx])
        val = row[name]
        missing = bool(np.isnan(val))
        items.append(
            FeatureAttribution(
                feature=name,
                label=FEATURE_SPECS[name].label,
                input_value=None if missing else float(val),
                was_imputed=missing,
                shap_value=v,
                direction="neutral" if abs(v) <= _NEUTRAL_EPS else ("raises_model_output" if v > 0 else "lowers_model_output"),
                rank=rank,
            )
        )
    shown = {i for i in order[: config.TOP_K_FEATURES]}
    other = float(sum(raw.values[i] for i in range(len(feats)) if i not in shown))
    return Explanation(
        method=raw.method,  # type: ignore[arg-type]
        explained_estimator=raw.explained_estimator,
        output_space=raw.output_space,  # type: ignore[arg-type]
        base_value=raw.base_value,
        sum_of_attributions=float(raw.values.sum()),
        base_model_output=raw.base_model_output,
        calibrated_probability=probability,
        top_features=items,
        other_features_sum=other,
        reference_population=_REFERENCE,
        notice=_NOTICE,
    )


def disease_result(bundle: ModelBundle, enc: EncodedInputs, include_explanation: bool) -> DiseaseResult:
    return _evaluate(bundle, enc, include_explanation)[0]


def _evaluate(bundle: ModelBundle, enc: EncodedInputs, include_explanation: bool) -> tuple[DiseaseResult, pd.DataFrame]:
    frame, report = build_frame(enc, bundle.features)
    probability = float(predict_proba_frame(bundle, frame)[0])
    predicted = probability >= bundle.threshold
    explanation = None
    reason = None
    if include_explanation:
        if bundle.explainer is None:
            reason = bundle.explanation_error or "explainer unavailable"
        else:
            explanation = _attributions(bundle, bundle.explainer.explain(frame), frame, probability)
    info = config.DISEASE_INFO[bundle.disease]
    result = DiseaseResult(
        disease=Disease(bundle.disease),
        mode=bundle.mode,  # type: ignore[arg-type]
        config_key=bundle.config_key,
        model_name=bundle.model_name,
        probability=probability,
        threshold=bundle.threshold,
        predicted_positive=bool(predicted),
        decision_label="screen_positive" if predicted else "screen_negative",
        target_definition=info["target_definition"],
        probability_note=config.PROBABILITY_NOTE,
        imputed_features=report.imputed_features,
        ignored_inputs=report.ignored_inputs,
        warnings=report.warnings,
        model_version=bundle.model_version,
        artifact_sha256=bundle.sha256,
        explanation=explanation,
        explanation_unavailable_reason=reason,
    )
    return result, frame


def predict(registry: ModelRegistry, request: PredictRequestA | PredictRequestB) -> PredictResponse:
    enc = encode_inputs(request.inputs)
    evaluated = [
        _evaluate(registry.get(d.value, request.mode), enc, request.include_explanation)
        for d in request.diseases
    ]
    response = PredictResponse(
        submission_id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc),
        mode=request.mode,
        results=[r for r, _ in evaluated],
        history_saved=False,
        input_adjustments=enc.adjustments,
        disclaimer=config.DISCLAIMER,
    )
    if request.save_to_history:
        # Persist only after every prediction succeeded, in one transaction. A persistence failure
        # must never turn a successful prediction into an error: degrade to a warning instead.
        try:
            history.save_submission(request, response, [f for _, f in evaluated])
            response.history_saved = True
        except Exception:
            log.exception("v2_history_save_failed submission_id=%s", response.submission_id)
            response.warnings.append(
                "The prediction succeeded but could not be saved to history; this result is not stored."
            )
    return response
