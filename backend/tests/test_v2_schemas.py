from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from backend.api.v2.schemas import ModeAInputs, ModeBInputs, PredictRequest

ADAPTER = TypeAdapter(PredictRequest)
MIN = {"age": 40, "gender": "male", "bmi": 25}


def test_only_age_gender_bmi_required():
    ModeAInputs(**MIN)
    for missing in MIN:
        bad = {k: v for k, v in MIN.items() if k != missing}
        with pytest.raises(ValidationError):
            ModeAInputs(**bad)


def test_mode_a_rejects_lab_fields_and_unknown_keys():
    with pytest.raises(ValidationError):
        ModeAInputs(**MIN, hba1c=5.5)
    with pytest.raises(ValidationError):
        ModeAInputs(**MIN, favourite_colour="blue")
    ModeBInputs(**MIN, hba1c=5.5)


@pytest.mark.parametrize(
    "field,value",
    [("age", 19), ("age", 101), ("bmi", 9), ("bmi", 81), ("mean_sbp", 300), ("mean_dbp", 20),
     ("education_level", 6), ("poverty_income_ratio", 6), ("sedentary_minutes", 1500),
     ("gender", "other"), ("smoking_status", "sometimes"), ("physical_activity_level", "extreme")],
)
def test_hard_bounds_and_enums_rejected(field, value):
    with pytest.raises(ValidationError):
        ModeAInputs(**{**MIN, field: value})


def test_nan_and_inf_rejected():
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValidationError):
            ModeAInputs(**{**MIN, "bmi": bad})


def test_explicit_null_means_missing():
    m = ModeAInputs(**MIN, mean_sbp=None, smoking_status=None)
    assert m.mean_sbp is None and m.smoking_status is None


def test_request_discriminated_by_mode_and_defaults():
    a = ADAPTER.validate_python({"mode": "A", "inputs": MIN})
    assert [d.value for d in a.diseases] == ["cvd", "diabetes", "hypertension"]
    assert a.include_explanation is True and a.save_to_history is True
    with pytest.raises(ValidationError):  # lab field under mode A
        ADAPTER.validate_python({"mode": "A", "inputs": {**MIN, "hba1c": 5.5}})
    with pytest.raises(ValidationError):
        ADAPTER.validate_python({"mode": "C", "inputs": MIN})
    ADAPTER.validate_python({"mode": "B", "inputs": {**MIN, "hba1c": 5.5}, "save_to_history": False})


def test_diseases_unique_nonempty_and_extra_forbidden():
    with pytest.raises(ValidationError):
        ADAPTER.validate_python({"mode": "A", "inputs": MIN, "diseases": []})
    with pytest.raises(ValidationError):
        ADAPTER.validate_python({"mode": "A", "inputs": MIN, "diseases": ["cvd", "cvd"]})
    with pytest.raises(ValidationError):
        ADAPTER.validate_python({"mode": "A", "inputs": MIN, "risk_score": 1})


def test_response_has_no_composite_score():
    from backend.api.v2.schemas import DiseaseResult, PredictResponse

    banned = {"risk_score", "overall_score", "composite", "risk_level", "score"}
    assert not banned & set(PredictResponse.model_fields)
    assert not banned & set(DiseaseResult.model_fields)
