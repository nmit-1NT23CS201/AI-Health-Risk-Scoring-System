from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from api_server import app as api_app
from src.feature_engineering import FEATURE_COLUMNS
from src.risk_interpreter import (
    classify_risk,
    format_contributors,
    generate_insights,
    top_contributors,
)
from src.shap_explainer import compute_shap_values, load_model, prepare_shap_inputs


def apply_theme() -> None:
    st.set_page_config(
        page_title="Health Risk Scoring Dashboard",
        page_icon="🩺",
        layout="wide",
    )

    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;500;700&family=Space+Grotesk:wght@500;700&display=swap');
            :root {
                --bg: #070b18;
                --bg-alt: #0b1226;
                --glass: rgba(14, 22, 44, 0.7);
                --glass-strong: rgba(12, 20, 40, 0.88);
                --stroke: rgba(84, 112, 255, 0.25);
                --text: #e7f1ff;
                --muted: #9fb0d8;
                --accent: #3df0ff;
                --accent-2: #6f8bff;
                --accent-3: #7c4dff;
                --success: #20c997;
                --warning: #f4a93c;
                --danger: #ff5c7c;
            }
            .stApp {
                background: radial-gradient(circle at 15% 20%, #0d1b36 0%, #070b18 45%, #05060f 100%);
                color: var(--text);
                font-family: 'Manrope', sans-serif;
            }
            h1, h2, h3, h4 {
                font-family: 'Space Grotesk', sans-serif;
                color: var(--text);
            }
            .block-container { padding-top: 1.5rem; }
            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, rgba(8, 14, 30, 0.95) 0%, rgba(8, 10, 20, 0.98) 100%);
                border-right: 1px solid rgba(61, 240, 255, 0.08);
            }
            section[data-testid="stSidebar"] .stMarkdown,
            section[data-testid="stSidebar"] label,
            section[data-testid="stSidebar"] span {
                color: var(--text);
            }
            .glass-card {
                background: var(--glass);
                border: 1px solid var(--stroke);
                box-shadow: 0 20px 50px rgba(0, 0, 0, 0.35);
                border-radius: 20px;
                padding: 18px 20px;
                margin-bottom: 18px;
            }
            .glass-hero {
                background: linear-gradient(135deg, rgba(23, 36, 68, 0.85) 0%, rgba(10, 16, 34, 0.95) 100%);
                border: 1px solid rgba(61, 240, 255, 0.2);
                border-radius: 24px;
                padding: 28px 30px;
                box-shadow: 0 30px 60px rgba(0, 0, 0, 0.45);
            }
            .nav-bar {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 20px;
                padding: 16px 22px;
                border-radius: 18px;
                background: var(--glass-strong);
                border: 1px solid rgba(111, 139, 255, 0.3);
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
                margin-bottom: 18px;
            }
            .nav-title {
                font-size: 1.1rem;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                color: var(--muted);
            }
            .nav-value {
                font-size: 1.3rem;
                font-weight: 700;
                color: var(--text);
            }
            .metric-card {
                background: linear-gradient(145deg, rgba(30, 42, 80, 0.85), rgba(11, 18, 40, 0.95));
                border: 1px solid rgba(61, 240, 255, 0.25);
                border-radius: 24px;
                padding: 24px;
                box-shadow: 0 24px 50px rgba(0, 0, 0, 0.45);
            }
            .metric-score {
                font-size: 3.6rem;
                font-weight: 700;
                line-height: 1;
                color: var(--text);
            }
            .metric-label {
                font-size: 0.9rem;
                letter-spacing: 0.14em;
                text-transform: uppercase;
                color: var(--muted);
            }
            .risk-pill {
                padding: 6px 14px;
                border-radius: 999px;
                display: inline-block;
                font-weight: 600;
                color: #05060f;
                margin-top: 10px;
                box-shadow: 0 0 20px rgba(61, 240, 255, 0.3);
            }
            .pill-low { background: linear-gradient(135deg, #1cd59b, #7fffd4); }
            .pill-medium { background: linear-gradient(135deg, #f4a93c, #ffd166); }
            .pill-high { background: linear-gradient(135deg, #ff5c7c, #ff8fab); }
            .risk-bar {
                height: 10px;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.08);
                overflow: hidden;
                margin-top: 18px;
            }
            .risk-bar-fill {
                height: 100%;
                border-radius: 999px;
                background: linear-gradient(90deg, #3df0ff, #6f8bff, #7c4dff);
                box-shadow: 0 0 15px rgba(61, 240, 255, 0.5);
                transition: width 0.6s ease;
            }
            .insight-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 12px;
            }
            .insight-card {
                padding: 12px 14px;
                border-radius: 16px;
                border: 1px solid rgba(124, 77, 255, 0.3);
                background: rgba(12, 18, 38, 0.7);
                color: var(--text);
            }
            .insight-card.positive { border-color: rgba(255, 92, 124, 0.5); box-shadow: 0 0 20px rgba(255, 92, 124, 0.2); }
            .insight-card.negative { border-color: rgba(32, 201, 151, 0.4); box-shadow: 0 0 20px rgba(32, 201, 151, 0.18); }
            .tag {
                display: inline-block;
                font-size: 0.72rem;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                color: var(--muted);
            }
            .cta-button button {
                width: 100%;
                border-radius: 14px;
                border: none;
                background: linear-gradient(135deg, #3df0ff, #6f8bff, #7c4dff);
                color: #05060f;
                font-weight: 700;
                padding: 0.85rem 1rem;
                box-shadow: 0 0 24px rgba(61, 240, 255, 0.5);
            }
            .cta-button button:hover {
                box-shadow: 0 0 30px rgba(111, 139, 255, 0.6);
                transform: translateY(-1px);
            }
            footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_pipeline(model_path: str):
    return load_model(model_path)


@st.cache_data
def load_feature_data(dataset_path: str) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    return df[FEATURE_COLUMNS].copy()


def get_demo_defaults(preset: str) -> Dict[str, object]:
    if preset == "High Risk Sample":
        return {
            "age": 55,
            "gender": "Male",
            "bmi": 31.0,
            "systolic_bp": 145,
            "diastolic_bp": 92,
            "cholesterol_mg_dl": 250,
            "smoking": "Yes",
            "alcohol_consumption": "Yes",
            "physical_activity": "Low",
            "family_history": "Yes",
            "heart_rate_bpm": 92,
            "sdnn_hrv": 35.0,
            "rmssd_hrv": 28.0,
            "spo2": 94.0,
        }
    if preset == "Low Risk Sample":
        return {
            "age": 24,
            "gender": "Female",
            "bmi": 22.0,
            "systolic_bp": 112,
            "diastolic_bp": 72,
            "cholesterol_mg_dl": 160,
            "smoking": "No",
            "alcohol_consumption": "No",
            "physical_activity": "High",
            "family_history": "No",
            "heart_rate_bpm": 68,
            "sdnn_hrv": 78.0,
            "rmssd_hrv": 65.0,
            "spo2": 99.0,
        }

    return {
        "age": 40,
        "gender": "Male",
        "bmi": 25.0,
        "systolic_bp": 125,
        "diastolic_bp": 80,
        "cholesterol_mg_dl": 190,
        "smoking": "No",
        "alcohol_consumption": "No",
        "physical_activity": "Medium",
        "family_history": "No",
        "heart_rate_bpm": 76,
        "sdnn_hrv": 55.0,
        "rmssd_hrv": 50.0,
        "spo2": 97.0,
    }


def build_input_form(defaults: Dict[str, object]) -> Tuple[Dict[str, object], bool]:
    with st.sidebar.form("health_input_form"):
        st.markdown("### Patient Inputs")

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("**Demographics**")
        age = st.slider("Age", 18, 80, int(defaults["age"]))
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"],
            index=0 if defaults["gender"] == "Male" else 1,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("**Lifestyle**")
        smoking = st.radio(
            "Smoking",
            ["Yes", "No"],
            index=0 if defaults["smoking"] == "Yes" else 1,
        )
        alcohol_consumption = st.radio(
            "Alcohol Consumption",
            ["Yes", "No"],
            index=0 if defaults["alcohol_consumption"] == "Yes" else 1,
        )
        physical_activity = st.selectbox(
            "Physical Activity",
            ["Low", "Medium", "High"],
            index=["Low", "Medium", "High"].index(defaults["physical_activity"]),
        )
        family_history = st.radio(
            "Family History",
            ["Yes", "No"],
            index=0 if defaults["family_history"] == "Yes" else 1,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("**Clinical Indicators**")
        bmi = st.number_input("BMI", 15.0, 45.0, float(defaults["bmi"]), 0.1)
        systolic_bp = st.slider("Systolic BP", 90, 200, int(defaults["systolic_bp"]))
        diastolic_bp = st.slider(
            "Diastolic BP", 60, 130, int(defaults["diastolic_bp"])
        )
        cholesterol_mg_dl = st.slider(
            "Cholesterol (mg/dL)", 120, 320, int(defaults["cholesterol_mg_dl"])
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("**Physiological Metrics**")
        heart_rate_bpm = st.slider(
            "Heart Rate (BPM)", 50, 130, int(defaults["heart_rate_bpm"])
        )
        sdnn_hrv = st.number_input(
            "SDNN (HRV)", 10.0, 140.0, float(defaults["sdnn_hrv"]), 0.5
        )
        rmssd_hrv = st.number_input(
            "RMSSD (HRV)", 10.0, 140.0, float(defaults["rmssd_hrv"]), 0.5
        )
        spo2 = st.slider("SpO2 (%)", 90.0, 100.0, float(defaults["spo2"]), 0.5)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='cta-button'>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Predict Health Risk")
        st.markdown("</div>", unsafe_allow_html=True)

    values = {
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "cholesterol_mg_dl": cholesterol_mg_dl,
        "smoking": smoking,
        "alcohol_consumption": alcohol_consumption,
        "physical_activity": physical_activity,
        "family_history": family_history,
        "heart_rate_bpm": heart_rate_bpm,
        "sdnn_hrv": sdnn_hrv,
        "rmssd_hrv": rmssd_hrv,
        "spo2": spo2,
    }

    return values, submitted


def generate_recommendations(values: Dict[str, object], risk_level: str) -> List[str]:
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


def render_risk_level(risk_level: str) -> None:
    if risk_level == "Low Risk":
        pill_class = "pill-low"
    elif risk_level == "Medium Risk":
        pill_class = "pill-medium"
    else:
        pill_class = "pill-high"

    st.markdown(
        f"<div class='risk-pill {pill_class}'>{risk_level}</div>",
        unsafe_allow_html=True,
    )


def render_navbar(prediction_state: str) -> None:
    st.markdown(
        f"""
        <div class="nav-bar">
            <div>
                <div class="nav-title">AI Health Risk System</div>
                <div class="nav-value">Preventive Intelligence Dashboard</div>
            </div>
            <div>
                <div class="nav-title">AI Status</div>
                <div class="nav-value">Online</div>
            </div>
            <div>
                <div class="nav-title">Prediction State</div>
                <div class="nav-value">{prediction_state}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="glass-hero">
            <div class="nav-title">AI-Powered Preventive Healthcare Intelligence</div>
            <h1>Personalized Health Risk Scoring</h1>
            <p style="color: var(--muted); font-size: 1.05rem; max-width: 720px;">
                Predict cardiovascular risk with explainable AI, visualize the top drivers,
                and deliver personalized recommendations in seconds.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary_plot(
    shap_values: np.ndarray, X_processed: np.ndarray, feature_names: List[str]
) -> None:
    shap.summary_plot(
        shap_values, X_processed, feature_names=feature_names, show=False
    )
    st.pyplot(plt.gcf(), clear_figure=True)


def render_bar_plot(
    shap_values: np.ndarray, X_processed: np.ndarray, feature_names: List[str]
) -> None:
    shap.summary_plot(
        shap_values,
        X_processed,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
    )
    st.pyplot(plt.gcf(), clear_figure=True)


def render_waterfall_plot(
    shap_row: np.ndarray,
    base_value: float,
    data_row: np.ndarray,
    feature_names: List[str],
) -> None:
    explanation = shap.Explanation(
        values=shap_row,
        base_values=base_value,
        data=data_row,
        feature_names=feature_names,
    )
    shap.plots.waterfall(explanation, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)


def run_streamlit() -> None:
    apply_theme()

    project_root = Path(__file__).resolve().parent
    dataset_path = project_root / "dataset" / "indian_health_risk_dataset.csv"
    model_path = project_root / "models" / "random_forest_model.pkl"

    prediction_state = st.session_state.get("prediction_state", "Awaiting Input")
    render_navbar(prediction_state)
    render_hero()

    st.sidebar.markdown("## Demo Presets")
    preset = st.sidebar.radio(
        "",
        ["Custom", "High Risk Sample", "Low Risk Sample"],
        help="Select a ready-to-demo input configuration.",
    )
    defaults = get_demo_defaults(preset)

    values, submitted = build_input_form(defaults)

    top_col_left, top_col_right = st.columns([1.1, 1.2], gap="large")

    with top_col_left:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Risk Score</div>", unsafe_allow_html=True)
        score_placeholder = st.empty()
        risk_level_placeholder = st.empty()
        bar_placeholder = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

    with top_col_right:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("AI Explanation")
        contributor_placeholder = st.empty()
        insight_placeholder = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

    recommendations_card = st.container()

    if submitted:
        try:
            pipeline = get_pipeline(str(model_path))
            input_df = pd.DataFrame([values], columns=FEATURE_COLUMNS)

            prediction = float(pipeline.predict(input_df)[0])
            risk_level = classify_risk(prediction)

            X_processed, feature_names = prepare_shap_inputs(pipeline, input_df)
            shap_values, base_value = compute_shap_values(pipeline, X_processed)
            shap_row = shap_values[0]

            positive, negative = top_contributors(shap_row, feature_names, top_n=6)
            insights = generate_insights(positive, negative)
            recommendations = generate_recommendations(values, risk_level)

            st.session_state["prediction_state"] = "Prediction Complete"

            with top_col_left:
                score_placeholder.markdown(
                    f"<div class='metric-score'>{prediction:.2f}</div>",
                    unsafe_allow_html=True,
                )
                render_risk_level(risk_level)
                bar_width = max(0, min(100, prediction))
                bar_placeholder.markdown(
                    f"<div class='risk-bar'><div class='risk-bar-fill' style='width: {bar_width:.0f}%;'></div></div>",
                    unsafe_allow_html=True,
                )

            with top_col_right:
                contributor_lines = format_contributors(positive, "+") + format_contributors(
                    negative, "-"
                )
                contributor_markup = "".join(
                    f"<div class='insight-card {'positive' if line.startswith('+') else 'negative'}'>{line}</div>"
                    for line in contributor_lines
                )
                contributor_placeholder.markdown(
                    "<div class='tag'>Top Contributors</div>"
                    f"<div class='insight-grid'>{contributor_markup}</div>",
                    unsafe_allow_html=True,
                )

                if insights:
                    insight_list = "".join(
                        f"<div class='insight-card'>{insight}</div>" for insight in insights
                    )
                    insight_placeholder.markdown(
                        "<div class='tag'>Insights</div>"
                        f"<div class='insight-grid'>{insight_list}</div>",
                        unsafe_allow_html=True,
                    )

            with recommendations_card:
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.subheader("Personalized Recommendations")
                for rec in recommendations:
                    st.markdown(f"- {rec}")
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("## SHAP Visualizations")

            X_reference = load_feature_data(str(dataset_path))
            sample = X_reference.sample(min(200, len(X_reference)), random_state=42)
            X_ref_processed, ref_feature_names = prepare_shap_inputs(pipeline, sample)
            ref_shap_values, _ = compute_shap_values(pipeline, X_ref_processed)

            viz_col1, viz_col2 = st.columns(2)
            with viz_col1:
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.markdown("**Summary Plot**")
                render_summary_plot(ref_shap_values, X_ref_processed, ref_feature_names)
                st.markdown("</div>", unsafe_allow_html=True)
            with viz_col2:
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.markdown("**Bar Plot**")
                render_bar_plot(ref_shap_values, X_ref_processed, ref_feature_names)
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("**Waterfall Plot (Selected Patient)**")
            render_waterfall_plot(
                shap_row,
                base_value,
                X_processed[0],
                feature_names,
            )
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception as exc:
            st.error(f"Unable to generate prediction: {exc}")

    st.markdown(
        """
        <div class="glass-card" style="text-align:center;">
            <div class="nav-title">AI Healthcare System</div>
            <div class="nav-value">Sensor-Augmented Personalized Health Risk Scoring</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    run_streamlit()


def main():
    return api_app
