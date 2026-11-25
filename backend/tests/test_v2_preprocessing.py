from __future__ import annotations

import math

import pandas as pd
import pytest

from backend.api.v2 import preprocessing as pp
from backend.api.v2.schemas import ModeAInputs, ModeBInputs
from conftest import PROCESSED, load_split_frame  # noqa: F401  (conftest is on sys.path via rootdir)

CONFIGS = [f"{d}_mode_{m}" for d in ("cvd", "diabetes", "hypertension") for m in ("a", "b")]


def enc(**kw):
    return pp.encode_inputs(ModeBInputs(age=40, gender="male", bmi=25, **kw))


def test_enum_codes_match_harmonizer():
    assert pp.encode_inputs(ModeAInputs(age=40, gender="male", bmi=25)).values["gender"] == 1.0
    assert pp.encode_inputs(ModeAInputs(age=40, gender="female", bmi=25)).values["gender"] == 2.0
    for label, code in (("never", 0.0), ("former", 1.0), ("current", 2.0)):
        assert enc(smoking_status=label).values["smoking_status"] == code
        assert enc(alcohol_frequency=label).values["alcohol_frequency"] == code
    for label, code in (("high", 1.0), ("moderate", 2.0), ("low", 3.0)):
        assert enc(physical_activity_level=label).values["physical_activity_level"] == code


def test_age_top_coding_with_adjustment_note():
    e = pp.encode_inputs(ModeAInputs(age=93, gender="male", bmi=25))
    assert e.values["age"] == 80.0 and len(e.adjustments) == 1 and "93" in e.adjustments[0]
    e = pp.encode_inputs(ModeAInputs(age=80, gender="male", bmi=25))
    assert e.values["age"] == 80.0 and e.adjustments == []


def test_absent_fields_are_nan_not_zero():
    e = pp.encode_inputs(ModeAInputs(age=40, gender="male", bmi=25))
    assert e.provided == {"age", "gender", "bmi"}
    assert all(math.isnan(v) for k, v in e.values.items() if k not in e.provided)


def test_passthrough_values_unchanged():
    e = enc(hba1c=5.7, mean_sbp=120.33333333333333, education_level=4)
    assert e.values["hba1c"] == 5.7 and e.values["mean_sbp"] == 120.33333333333333 and e.values["education_level"] == 4.0


def test_every_spec_matches_a_schema_field_and_vice_versa():
    assert set(pp.FEATURE_SPECS) == set(ModeBInputs.model_fields)
    assert set(pp.input_field_names("A")) == set(ModeAInputs.model_fields)


@pytest.mark.artifacts
@pytest.mark.parametrize("key", CONFIGS)
def test_frame_order_and_dtype_match_registry(registry, key):
    b = next(x for x in registry.all() if x.config_key == key)
    frame, rep = pp.build_frame(enc(hba1c=5.5), b.features)
    assert list(frame.columns) == list(b.features) and frame.shape == (1, len(b.features))
    assert (frame.dtypes == "float64").all()
    assert set(rep.imputed_features) == {f for f in b.features if math.isnan(frame.iloc[0][f])}


@pytest.mark.artifacts
def test_leakage_exclusions_become_ignored_inputs(registry):
    e = enc(hba1c=5.5, fasting_glucose=99, mean_sbp=120, mean_dbp=80)
    _, htn_b = pp.build_frame(e, registry.get("hypertension", "B").features)
    assert {"mean_sbp", "mean_dbp"} <= set(htn_b.ignored_inputs)
    _, dm_b = pp.build_frame(e, registry.get("diabetes", "B").features)
    assert {"hba1c", "fasting_glucose"} <= set(dm_b.ignored_inputs)
    _, cvd_b = pp.build_frame(e, registry.get("cvd", "B").features)
    assert cvd_b.ignored_inputs == []


@pytest.mark.artifacts
def test_out_of_training_range_warns(registry):
    e = pp.encode_inputs(ModeAInputs(age=40, gender="male", bmi=75.5))  # hard-valid, outside 11.1-74.8
    _, rep = pp.build_frame(e, registry.get("cvd", "A").features)
    assert any(w.startswith("bmi=") for w in rep.warnings)


@pytest.mark.artifacts
def test_training_ranges_equal_processed_data_min_max():
    """The hard-coded training_range table must equal the frozen processed data."""
    df = pd.concat(
        [pd.read_parquet(PROCESSED / f"nhanes_{k}.parquet") for k in ("cvd_mode_b", "diabetes_mode_b")]
    )
    for name, spec in pp.FEATURE_SPECS.items():
        if spec.training_range is None:
            continue
        lo, hi = spec.training_range
        assert df[name].min() == pytest.approx(lo, abs=1e-9), name
        assert df[name].max() == pytest.approx(hi, abs=1e-9), name


@pytest.mark.artifacts
def test_categorical_codes_in_data_are_covered_by_contract():
    df = pd.read_parquet(PROCESSED / "nhanes_cvd_mode_b.parquet")
    assert set(df["gender"].dropna()) == set(pp.GENDER_CODE.values())
    assert set(df["smoking_status"].dropna()) == set(pp.SMOKING_CODE.values())
    assert set(df["alcohol_frequency"].dropna()) == set(pp.ALCOHOL_CODE.values())
    assert set(df["physical_activity_level"].dropna()) == set(pp.ACTIVITY_CODE.values())
