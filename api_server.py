from __future__ import annotations

import base64
import io
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple
from src.database import (
    create_tables,
    save_assessment,
    get_assessment_history
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.feature_engineering import FEATURE_COLUMNS
from src.risk_interpreter import (
    classify_risk,
    generate_insights,
    top_contributors,
    _pretty_feature_name,
)
from src.shap_explainer import compute_shap_values, load_model, prepare_shap_inputs


class PredictionInput(BaseModel):
    age: int = Field(ge=18, le=80)
    gender: str
    bmi: float = Field(ge=15, le=45)
    systolic_bp: int = Field(ge=90, le=200)
    diastolic_bp: int = Field(ge=60, le=130)
    cholesterol_mg_dl: int = Field(ge=120, le=320)
    smoking: str
    alcohol_consumption: str
    physical_activity: str
    family_history: str
    heart_rate_bpm: int = Field(ge=50, le=130)
    sdnn_hrv: float = Field(ge=10, le=140)
    rmssd_hrv: float = Field(ge=10, le=140)
    spo2: float = Field(ge=90, le=100)


class Contributor(BaseModel):
    feature: str
    value: float


class PredictionResponse(BaseModel):
    risk_score: float
    risk_level: str
    positive_contributors: List[Contributor]
    negative_contributors: List[Contributor]
    insights: List[str]
    recommendations: List[str]
    summary_plot: str
    bar_plot: str
    waterfall_plot: str


create_tables()

app = FastAPI(title="AI Health Risk API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@lru_cache
def _get_paths() -> Tuple[Path, Path]:
    project_root = Path(__file__).resolve().parent
    dataset_path = project_root / "dataset" / "indian_health_risk_dataset.csv"
    model_path = project_root / "models" / "random_forest_model.pkl"
    return dataset_path, model_path


@lru_cache
def _get_pipeline():
    _, model_path = _get_paths()
    return load_model(str(model_path))


def _prepare_input_df(payload: PredictionInput) -> pd.DataFrame:
    data = payload.dict()
    return pd.DataFrame([data], columns=FEATURE_COLUMNS)


def _generate_recommendations(values: Dict[str, object], risk_level: str) -> List[str]:
    recommendations: List[str] = []

    if values["smoking"] == "Yes":
        recommendations.append("Reduce or stop smoking to lower cardiovascular risk.")
    if values["bmi"] >= 27:
        recommendations.append("Aim for gradual weight reduction with balanced nutrition.")
    if values["cholesterol_mg_dl"] >= 200:
        recommendations.append("Reduce saturated fats and monitor cholesterol levels.")
    if values["systolic_bp"] >= 130 or values["diastolic_bp"] >= 85:
        recommendations.append("Monitor blood pressure and limit excess sodium.")
    if values["physical_activity"] == "Low":
        recommendations.append("Add at least 30 minutes of moderate activity most days.")
    if values["spo2"] < 95:
        recommendations.append("Prioritize breathing exercises and check oxygen levels.")

    if not recommendations:
        recommendations.append("Maintain your current healthy lifestyle routines.")

    if risk_level == "High Risk":
        recommendations.append("Consult a healthcare professional for a detailed checkup.")

    return recommendations


def _fig_to_base64() -> str:
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
    plt.close()
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


@lru_cache
def _get_reference_shap() -> Tuple[np.ndarray, List[str], np.ndarray]:
    dataset_path, _ = _get_paths()
    pipeline = _get_pipeline()

    df = pd.read_csv(dataset_path)
    sample = df[FEATURE_COLUMNS].sample(min(200, len(df)), random_state=42)
    X_processed, feature_names = prepare_shap_inputs(pipeline, sample)
    shap_values, _ = compute_shap_values(pipeline, X_processed)
    return X_processed, feature_names, shap_values


def _render_summary_plot() -> str:
    X_processed, feature_names, shap_values = _get_reference_shap()
    shap.summary_plot(shap_values, X_processed, feature_names=feature_names, show=False,plot_size=None)
    return _fig_to_base64()


def _render_bar_plot() -> str:
    X_processed, feature_names, shap_values = _get_reference_shap()
    shap.summary_plot(
        shap_values, X_processed, feature_names=feature_names, plot_type="bar", show=False
    )
    return _fig_to_base64()


def _render_waterfall_plot(
    shap_row: np.ndarray, base_value: float, data_row: np.ndarray, feature_names: List[str]
) -> str:
    explanation = shap.Explanation(
        values=shap_row,
        base_values=base_value,
        data=data_row,
        feature_names=feature_names,
    )
    shap.plots.waterfall(explanation, show=False)
    return _fig_to_base64()


@app.get("/")
def root() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}
@app.get("/history")
def history():

    rows = get_assessment_history()

    return [
        {
            "id": row[0],
            "age": row[1],
            "risk_score": row[2],
            "risk_level": row[3],
            "created_at": row[4]
        }
        for row in rows
    ]


@app.get("/model-metrics")
def get_model_metrics() -> Dict[str, object]:
    project_root = Path(__file__).resolve().parent
    metrics_path = project_root / "outputs" / "metrics.json"
    cm_path = project_root / "outputs" / "confusion_matrix.png"

    if not metrics_path.exists() or not cm_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Model metrics or confusion matrix image not found. Please run the evaluation pipeline script manually first."
        )

    try:
        import json
        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)

        with open(cm_path, "rb") as f:
            cm_base64 = base64.b64encode(f.read()).decode("utf-8")

        return {
            "regression_metrics": metrics_data["regression_metrics"],
            "classification_metrics": metrics_data["classification_metrics"],
            "confusion_matrix": cm_base64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model metrics: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionInput) -> PredictionResponse:
    try:
        pipeline = _get_pipeline()
        input_df = _prepare_input_df(payload)

        prediction = float(pipeline.predict(input_df)[0])
        risk_level = classify_risk(prediction)
        assessment_id = save_assessment(
            payload,
            prediction,
            risk_level
        )
        X_processed, feature_names = prepare_shap_inputs(pipeline, input_df)
        shap_values, base_value = compute_shap_values(pipeline, X_processed)
        shap_row = shap_values[0]
        print(feature_names)

        positive, negative = top_contributors(shap_row, feature_names, top_n=6)
        insights = generate_insights(positive, negative, payload)
        recommendations = _generate_recommendations(payload.dict(), risk_level)

        summary_plot = _render_summary_plot()
        bar_plot = _render_bar_plot()
        waterfall_plot = _render_waterfall_plot(
            shap_row, base_value, X_processed[0], feature_names
        )

        return PredictionResponse(
            risk_score=prediction,
            risk_level=risk_level,
            positive_contributors=[
    Contributor(
        feature=_pretty_feature_name(name),
        value=value
    )
    for name, value in positive
],
negative_contributors=[
    Contributor(
        feature=_pretty_feature_name(name),
        value=value
    )
    for name, value in negative
],
            insights=insights,
            recommendations=recommendations,
            summary_plot=summary_plot,
            bar_plot=bar_plot,
            waterfall_plot=waterfall_plot,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
