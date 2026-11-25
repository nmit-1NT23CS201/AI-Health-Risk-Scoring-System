"""Pydantic v2 request/response models for /v2.

Hard bounds below are plausibility limits (reject impossible values). Training-range
warnings live in ``preprocessing.py``. There is deliberately no aggregate / composite score.
"""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Gender(StrEnum):
    MALE = "male"
    FEMALE = "female"


class SmokingStatus(StrEnum):
    NEVER = "never"
    FORMER = "former"
    CURRENT = "current"


class AlcoholStatus(StrEnum):
    NEVER = "never"
    FORMER = "former"
    CURRENT = "current"


class ActivityLevel(StrEnum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"


class Disease(StrEnum):
    CVD = "cvd"
    DIABETES = "diabetes"
    HYPERTENSION = "hypertension"


Mode = Literal["A", "B"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModeAInputs(_Strict):
    # Required
    age: float = Field(ge=20, le=100, description="Years. Values above 80 are top-coded to 80 (NHANES).")
    gender: Gender
    bmi: float = Field(ge=10, le=80, description="kg/m^2")
    # Optional (absent -> NaN)
    waist_circumference: float | None = Field(None, ge=40, le=200, description="cm")
    mean_sbp: float | None = Field(None, ge=60, le=260, description="mmHg; mean of up to 3 readings")
    mean_dbp: float | None = Field(None, ge=30, le=160, description="mmHg; mean of up to 3 readings")
    mean_pulse: float | None = Field(None, ge=30, le=220, description="bpm; mean of up to 3 readings")
    education_level: int | None = Field(None, ge=1, le=5, description="NHANES DMDEDUC2 code 1-5")
    poverty_income_ratio: float | None = Field(None, ge=0, le=5, description="Family income / poverty line (0-5)")
    smoking_status: SmokingStatus | None = None
    alcohol_frequency: AlcoholStatus | None = Field(
        None, description="Never / former / current drinker (a status, despite the training column name)."
    )
    physical_activity_level: ActivityLevel | None = Field(
        None,
        description=(
            "Training definition: high = vigorous >=3 or moderate >=5 sessions/week; "
            "low = none of either; moderate otherwise."
        ),
    )
    sedentary_minutes: float | None = Field(None, ge=0, le=1440, description="Minutes per day")


class ModeBInputs(ModeAInputs):
    hba1c: float | None = Field(None, ge=3, le=20, description="%")
    fasting_glucose: float | None = Field(None, ge=40, le=700, description="mg/dL")
    total_cholesterol: float | None = Field(None, ge=50, le=500, description="mg/dL")
    hdl_cholesterol: float | None = Field(None, ge=10, le=200, description="mg/dL")
    triglycerides: float | None = Field(None, ge=10, le=2000, description="mg/dL")
    ldl_cholesterol: float | None = Field(None, ge=1, le=400, description="mg/dL")
    serum_creatinine: float | None = Field(None, ge=0.2, le=20, description="mg/dL")
    blood_urea_nitrogen: float | None = Field(None, ge=2, le=150, description="mg/dL")
    serum_uric_acid: float | None = Field(None, ge=0.5, le=20, description="mg/dL")
    alt_enzyme: float | None = Field(None, ge=1, le=1000, description="U/L")
    ast_enzyme: float | None = Field(None, ge=1, le=1000, description="U/L")
    hemoglobin: float | None = Field(None, ge=4, le=25, description="g/dL")
    wbc_count: float | None = Field(None, ge=0.5, le=100, description="10^3 cells/uL")
    platelet_count: float | None = Field(None, ge=10, le=1500, description="10^3 cells/uL")
    rdw: float | None = Field(None, ge=8, le=40, description="%")
    serum_albumin: float | None = Field(None, ge=1, le=7, description="g/dL")


class _PredictBase(_Strict):
    diseases: list[Disease] = Field(default_factory=lambda: list(Disease), min_length=1)
    include_explanation: bool = True
    # Accepted now so clients do not change when Phase 2C lands; currently a no-op.
    save_to_history: bool = True
    # Optional label stored with the history entry only; never used by the models.
    patient_name: str | None = Field(default=None, max_length=120)

    @field_validator("patient_name", mode="before")
    @classmethod
    def _clean_name(cls, v):
        if isinstance(v, str):
            v = v.strip()
            return v or None
        return v

    @field_validator("diseases")
    @classmethod
    def _unique(cls, v: list[Disease]) -> list[Disease]:
        if len(set(v)) != len(v):
            raise ValueError("diseases must be unique")
        return v


class PredictRequestA(_PredictBase):
    mode: Literal["A"]
    inputs: ModeAInputs


class PredictRequestB(_PredictBase):
    mode: Literal["B"]
    inputs: ModeBInputs


PredictRequest = Annotated[Union[PredictRequestA, PredictRequestB], Field(discriminator="mode")]


# ----------------------------------------------------------------------------- responses
class FeatureAttribution(BaseModel):
    feature: str
    label: str
    input_value: float | None = Field(description="Value supplied by the caller (after encoding); null if missing")
    was_imputed: bool
    shap_value: float = Field(description="In `output_space` units; NOT a change in calibrated probability")
    direction: Literal["raises_model_output", "lowers_model_output", "neutral"]
    rank: int


class Explanation(BaseModel):
    method: Literal["TreeExplainer", "LinearExplainer"]
    explained_estimator: str
    output_space: Literal["uncalibrated_rf_probability", "log_odds_margin"]
    base_value: float
    sum_of_attributions: float
    base_model_output: float
    additive_to_calibrated_probability: Literal[False] = False
    calibrated_probability: float
    top_features: list[FeatureAttribution]
    other_features_sum: float
    reference_population: str
    display_hint: Literal["relative_magnitude_only"] = "relative_magnitude_only"
    notice: str


class DiseaseResult(BaseModel):
    disease: Disease
    mode: Mode
    config_key: str
    model_name: str
    probability: float
    threshold: float
    predicted_positive: bool
    decision_label: Literal["screen_positive", "screen_negative"]
    target_definition: str
    probability_note: str
    imputed_features: list[str]
    ignored_inputs: list[str]
    warnings: list[str]
    model_version: str
    artifact_sha256: str
    explanation: Explanation | None = None
    explanation_unavailable_reason: str | None = None


class PredictResponse(BaseModel):
    submission_id: str
    created_at: datetime
    mode: Mode
    results: list[DiseaseResult]
    history_saved: bool = Field(False, description="True when the submission was persisted to V2 history")
    input_adjustments: list[str]
    warnings: list[str] = Field(default_factory=list, description="Submission-level warnings (e.g. history not saved)")
    disclaimer: str


# ----------------------------------------------------------------------------- history
class HistoryResult(BaseModel):
    disease: Disease
    config_key: str
    probability: float
    threshold: float
    predicted_positive: bool
    model_version: str


class HistorySubmission(BaseModel):
    submission_id: str
    patient_name: str | None = None
    created_at: str
    mode: Mode
    results: list[HistoryResult]


class HistoryList(BaseModel):
    items: list[HistorySubmission]
    next_before_id: int | None = Field(None, description="Pass as `before_id` to fetch the next (older) page")


class HistoryDetailResult(HistoryResult):
    model_input: dict[str, float | None]
    imputed_features: list[str]
    artifact_sha256: str


class HistoryDetail(BaseModel):
    submission_id: str
    patient_name: str | None = None
    created_at: str
    mode: Mode
    inputs: dict
    results: list[HistoryDetailResult]


class HistoryDeleteResponse(BaseModel):
    deleted_rows: int
    submission_id: str | None = None
