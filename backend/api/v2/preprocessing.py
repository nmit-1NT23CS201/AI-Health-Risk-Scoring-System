"""Inference-time encoding contract. Mirrors ``backend/ml/scripts/nhanes_harmonizer.py``.

The model feature names/order come from the registry (stage_1f_results.json); this module
only knows how to turn each *named* feature from the API payload into the value the frozen
models were trained on. NaN is passed through unchanged (imputation lives inside the
artifacts: median imputers for RF/LR, native NaN handling for HGB/XGB).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from backend.api.v2.schemas import (
    ActivityLevel,
    AlcoholStatus,
    Gender,
    ModeAInputs,
    ModeBInputs,
    SmokingStatus,
)

AGE_TOP_CODE = 80

# harmonizer: RIAGENDR 1 = male, 2 = female (L54, L366)
GENDER_CODE = {Gender.MALE: 1.0, Gender.FEMALE: 2.0}
# harmonizer _recode_smoking L80-102
SMOKING_CODE = {SmokingStatus.NEVER: 0.0, SmokingStatus.FORMER: 1.0, SmokingStatus.CURRENT: 2.0}
# harmonizer _recode_alcohol L105-123
ALCOHOL_CODE = {AlcoholStatus.NEVER: 0.0, AlcoholStatus.FORMER: 1.0, AlcoholStatus.CURRENT: 2.0}
# harmonizer _recode_physical_activity L126-158 (note: 1 = HIGH, 3 = LOW)
ACTIVITY_CODE = {ActivityLevel.HIGH: 1.0, ActivityLevel.MODERATE: 2.0, ActivityLevel.LOW: 3.0}

_ENUM_CODES: dict[str, dict] = {
    "gender": GENDER_CODE,
    "smoking_status": SMOKING_CODE,
    "alcohol_frequency": ALCOHOL_CODE,
    "physical_activity_level": ACTIVITY_CODE,
}


@dataclass(frozen=True)
class FeatureSpec:
    label: str
    unit: str | None
    harmonizer_source: str
    training_range: tuple[float, float] | None  # observed min/max in the processed NHANES data


# training_range = min/max of the six processed parquet files (verified by test_v2_preprocessing).
FEATURE_SPECS: dict[str, FeatureSpec] = {
    "age": FeatureSpec("Age", "years", "RIDAGEYR (L365)", (20.0, 80.0)),
    "gender": FeatureSpec("Sex", None, "RIAGENDR (L366): male=1, female=2", None),
    "education_level": FeatureSpec("Education level", "code 1-5", "DMDEDUC2 (L367)", (1.0, 5.0)),
    "poverty_income_ratio": FeatureSpec("Income-to-poverty ratio", "ratio", "INDFMPIR (L368)", (0.0, 5.0)),
    "bmi": FeatureSpec("Body mass index", "kg/m2", "BMXBMI (L369)", (11.1, 74.8)),
    "waist_circumference": FeatureSpec("Waist circumference", "cm", "BMXWAIST (L370)", (60.0, 187.0)),
    "mean_sbp": FeatureSpec("Systolic blood pressure (mean)", "mmHg", "nanmean BPXOSY1-3 (L161-185)", (70.0, 232.33333333333334)),
    "mean_dbp": FeatureSpec("Diastolic blood pressure (mean)", "mmHg", "nanmean BPXODI1-3 (L161-185)", (34.0, 139.0)),
    "mean_pulse": FeatureSpec("Pulse (mean)", "bpm", "nanmean BPXOPLS1-3 (L161-185)", (34.0, 129.33333333333333)),
    "smoking_status": FeatureSpec("Smoking status", None, "SMQ020/SMQ040 (L80-102): 0/1/2", None),
    "alcohol_frequency": FeatureSpec("Alcohol use status", None, "ALQ111/ALQ121 (L105-123): 0/1/2", None),
    "physical_activity_level": FeatureSpec("Physical activity level", None, "PAD790Q/PAD810Q (L126-158): 1 high, 2 moderate, 3 low", None),
    "sedentary_minutes": FeatureSpec("Sedentary time", "min/day", "PAD680 (L372)", (0.0, 1380.0)),
    "hba1c": FeatureSpec("HbA1c", "%", "LBXGH (L373)", (3.2, 17.1)),
    "fasting_glucose": FeatureSpec("Fasting glucose", "mg/dL", "LBXGLU (L374)", (59.0, 561.0)),
    "total_cholesterol": FeatureSpec("Total cholesterol", "mg/dL", "LBXTC (L375)", (62.0, 438.0)),
    "hdl_cholesterol": FeatureSpec("HDL cholesterol", "mg/dL", "LBDHDD (L376)", (22.0, 159.0)),
    "triglycerides": FeatureSpec("Triglycerides", "mg/dL", "LBXTLG (L377)", (19.0, 1745.0)),
    "ldl_cholesterol": FeatureSpec("LDL cholesterol", "mg/dL", "LBDLDL (L378)", (3.0, 314.0)),
    "serum_creatinine": FeatureSpec("Serum creatinine", "mg/dL", "LBXSCR (L379)", (0.35, 15.17)),
    "blood_urea_nitrogen": FeatureSpec("Blood urea nitrogen", "mg/dL", "LBXSBU (L380)", (4.0, 74.0)),
    "serum_uric_acid": FeatureSpec("Serum uric acid", "mg/dL", "LBXSUA (L381)", (1.1, 13.2)),
    "alt_enzyme": FeatureSpec("ALT", "U/L", "LBXSATSI (L382)", (3.0, 350.0)),
    "ast_enzyme": FeatureSpec("AST", "U/L", "LBXSASSI (L383)", (6.0, 363.0)),
    "hemoglobin": FeatureSpec("Hemoglobin", "g/dL", "LBXHGB (L384)", (6.3, 18.6)),
    "wbc_count": FeatureSpec("White blood cell count", "10^3/uL", "LBXWBCSI (L385)", (2.1, 18.4)),
    "platelet_count": FeatureSpec("Platelet count", "10^3/uL", "LBXPLTSI (L386)", (46.0, 787.0)),
    "rdw": FeatureSpec("Red cell distribution width", "%", "LBXRDW (L387)", (11.5, 37.5)),
    "serum_albumin": FeatureSpec("Serum albumin", "g/dL", "LBXSAL (L388)", (2.2, 5.5)),
}

REQUIRED_FIELDS = ("age", "gender", "bmi")


def input_field_names(mode: str) -> list[str]:
    model = ModeBInputs if mode == "B" else ModeAInputs
    return list(model.model_fields)


@dataclass
class EncodedInputs:
    """All API fields encoded to model values (NaN where absent)."""

    values: dict[str, float]
    provided: set[str]
    adjustments: list[str] = field(default_factory=list)


@dataclass
class FrameReport:
    imputed_features: list[str]
    ignored_inputs: list[str]
    warnings: list[str]


def _to_float(name: str, raw) -> float:
    if raw is None:
        return math.nan
    if isinstance(raw, Enum):
        return _ENUM_CODES[name][raw]
    return float(raw)


def encode_inputs(inputs: ModeAInputs) -> EncodedInputs:
    values: dict[str, float] = {}
    provided: set[str] = set()
    adjustments: list[str] = []
    for name in type(inputs).model_fields:
        raw = getattr(inputs, name)
        if raw is not None:
            provided.add(name)
        values[name] = _to_float(name, raw)
    if values["age"] > AGE_TOP_CODE:
        adjustments.append(f"age {values['age']:g} top-coded to {AGE_TOP_CODE} (NHANES top-codes age at 80)")
        values["age"] = float(AGE_TOP_CODE)
    return EncodedInputs(values=values, provided=provided, adjustments=adjustments)


def build_frame(enc: EncodedInputs, features: list[str] | tuple[str, ...]) -> tuple[pd.DataFrame, FrameReport]:
    """One-row float64 DataFrame in exactly ``features`` order, plus a per-disease report."""
    missing_encoders = [f for f in features if f not in enc.values]
    if missing_encoders:
        raise KeyError(f"No encoder/input for model features: {missing_encoders}")
    row = [enc.values[f] for f in features]
    frame = pd.DataFrame([row], columns=list(features), dtype="float64")

    feature_set = set(features)
    imputed = [f for f in features if math.isnan(enc.values[f])]
    ignored = sorted(n for n in enc.provided if n not in feature_set)
    warnings: list[str] = []
    for f in features:
        spec = FEATURE_SPECS.get(f)
        v = enc.values[f]
        if spec and spec.training_range and not math.isnan(v):
            lo, hi = spec.training_range
            if v < lo or v > hi:
                warnings.append(
                    f"{f}={v:g} is outside the NHANES training range [{lo:g}, {hi:g}]; the model is extrapolating."
                )
    return frame, FrameReport(imputed_features=imputed, ignored_inputs=ignored, warnings=warnings)


def frame_from_values(values: dict[str, float], features: list[str] | tuple[str, ...]) -> pd.DataFrame:
    """Helper for tests/benchmarks: encoded values -> ordered frame."""
    return pd.DataFrame([[values.get(f, np.nan) for f in features]], columns=list(features), dtype="float64")
