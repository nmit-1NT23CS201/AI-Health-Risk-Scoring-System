from typing import Dict, Iterable, List, Tuple

import numpy as np


RISK_BANDS = [
    (0, 34, "Low Risk"),
    (35, 69, "Medium Risk"),
    (70, 100, "High Risk"),
]


def classify_risk(score: float) -> str:
    for low, high, label in RISK_BANDS:
        if low <= score <= high:
            return label
    return "Unknown"


def _pretty_feature_name(raw_name: str) -> str:
    name = raw_name.replace("categorical__", "").replace("numeric__", "")

    if "smoking_" in name:
        return "Smoking"

    if "family_history_" in name:
        return "Family History"

    if "physical_activity_" in name:
        return "Physical Activity"

    name = name.replace("_", " ")
    return " ".join(word.capitalize() for word in name.split())


def top_contributors(
    shap_row: np.ndarray, feature_names: List[str], top_n: int = 5
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:

    indices = np.argsort(np.abs(shap_row))[::-1]

    seen = set()
    top_items = []

    for i in indices:
        name = feature_names[i]

        base_name = name.replace("categorical__", "").replace("numeric__", "")
        base_name = base_name.split("_")[0]

        if base_name in seen:
            continue

        seen.add(base_name)
        top_items.append((name, float(shap_row[i])))

        if len(top_items) == top_n:
            break

    positive = [(name, value) for name, value in top_items if value > 0]
    negative = [(name, value) for name, value in top_items if value < 0]

    return positive, negative


def generate_insights(
    positive: Iterable[Tuple[str, float]],
    negative: Iterable[Tuple[str, float]],
    payload
) -> List[str]:

    insights: List[str] = []
    seen = set()

    def add_insight(feature_name: str):
        if feature_name in seen:
            return

        seen.add(feature_name)

        # AGE
                # AGE
        if feature_name == "age":
            if payload.age < 30:
                insights.append(
                    f"Age {payload.age} years is not considered a significant cardiovascular risk factor."
                )
            elif payload.age < 50:
                insights.append(
                    f"Age {payload.age} may moderately influence cardiovascular risk."
                )
            else:
                insights.append(
                    f"Age {payload.age} is a recognized cardiovascular risk factor."
                )

        # BMI
        elif feature_name == "bmi":
            if payload.bmi < 25:
                insights.append(
                    f"BMI of {payload.bmi:.1f} is within the healthy range."
                )
            elif payload.bmi < 30:
                insights.append(
                    f"BMI of {payload.bmi:.1f} falls in the overweight range."
                )
            else:
                insights.append(
                    f"BMI of {payload.bmi:.1f} falls in the obese range."
                )

        # CHOLESTEROL
        elif feature_name == "cholesterol":
            if payload.cholesterol_mg_dl < 200:
                insights.append(
                    f"Cholesterol level of {payload.cholesterol_mg_dl} mg/dL is within the desirable range."
                )
            elif payload.cholesterol_mg_dl < 240:
                insights.append(
                    f"Cholesterol level of {payload.cholesterol_mg_dl} mg/dL is borderline high."
                )
            else:
                insights.append(
                    f"Cholesterol level of {payload.cholesterol_mg_dl} mg/dL is elevated."
                )

        # SYSTOLIC BP
        elif feature_name == "systolic":
            if payload.systolic_bp < 90:
                insights.append(
                    f"Systolic blood pressure of {payload.systolic_bp} mmHg is below the normal range."
                )

            elif payload.systolic_bp <= 119:
                insights.append(
                    f"Systolic blood pressure of {payload.systolic_bp} mmHg is within the normal range."
                )

            elif payload.systolic_bp <= 129:
                insights.append(
                    f"Systolic blood pressure of {payload.systolic_bp} mmHg is elevated."
                )

            elif payload.systolic_bp <= 139:
                insights.append(
                    f"Systolic blood pressure of {payload.systolic_bp} mmHg falls in the Stage 1 hypertension range."
                )

            else:
                insights.append(
                    f"Systolic blood pressure of {payload.systolic_bp} mmHg falls in the Stage 2 hypertension range."
                )
        # DIASTOLIC BP
        elif feature_name == "diastolic":
            if payload.diastolic_bp < 70:
                insights.append(
                    f"Diastolic blood pressure of {payload.diastolic_bp} mmHg is below the normal range."
                )

            elif payload.diastolic_bp <= 80:
                insights.append(
                    f"Diastolic blood pressure of {payload.diastolic_bp} mmHg is within the normal range."
                )

            elif payload.diastolic_bp <= 89:
                insights.append(
                    f"Diastolic blood pressure of {payload.diastolic_bp} mmHg is slightly elevated."
                )

            else:
                insights.append(
                    f"Diastolic blood pressure of {payload.diastolic_bp} mmHg is in the hypertension range."
        )
        # SMOKING
        elif feature_name == "smoking":
            if str(payload.smoking).lower() == "yes":
                insights.append(
                    "Smoking is a significant cardiovascular risk factor."
                )
            else:
                insights.append(
                    "Non-smoking helps reduce long-term cardiovascular risk."
                )

        # PHYSICAL ACTIVITY
        elif feature_name == "physical_activity":
            if str(payload.physical_activity).lower() == "low":
                insights.append(
                    "Low physical activity may increase cardiovascular risk."
                )
            elif str(payload.physical_activity).lower() == "moderate":
                insights.append(
                    "Moderate physical activity supports cardiovascular health."
                )
            else:
                insights.append(
                    "High physical activity supports cardiovascular health."
                )

        # FAMILY HISTORY
        elif feature_name == "family_history":
            if str(payload.family_history).lower() == "yes":
                insights.append(
                    "Family history may increase susceptibility to cardiovascular disease."
                )

        # SPO2
        elif feature_name == "spo2":
            if payload.spo2 >= 95:
                insights.append(
                    f"SpO₂ level of {payload.spo2}% is within the healthy range."
                )

            elif payload.spo2 >= 90:
                insights.append(
                    f"SpO₂ level of {payload.spo2}% is lower than ideal and should be monitored."
                )

            else:
                insights.append(
                    f"SpO₂ level of {payload.spo2}% is significantly below the normal range."
                )

        # SDNN
        elif feature_name == "sdnn":
            insights.append(
                f"SDNN HRV value of {payload.sdnn_hrv} indicates healthy heart-rate variability."
            )

        # RMSSD
        elif feature_name == "rmssd":
            insights.append(
                f"RMSSD HRV value of {payload.rmssd_hrv} indicates healthy heart-rate variability."
            )

    # Positive contributors
    for name, _ in positive:
        lower = name.lower()

        if "age" in lower:
            add_insight("age")
        elif "bmi" in lower:
            add_insight("bmi")
        elif "cholesterol" in lower:
            add_insight("cholesterol")
        elif "systolic" in lower:
            add_insight("systolic")
        elif "diastolic" in lower:
            add_insight("diastolic")
        elif "smoking" in lower:
            add_insight("smoking")
        elif "physical_activity" in lower:
            add_insight("physical_activity")
        elif "family_history" in lower:
            add_insight("family_history")
        elif "spo2" in lower:
            add_insight("spo2")
        elif "sdnn" in lower:
            add_insight("sdnn")
        elif "rmssd" in lower:
            add_insight("rmssd")

    # Negative contributors
    for name, _ in negative:
        lower = name.lower()

        if "age" in lower:
            add_insight("age")
        elif "bmi" in lower:
            add_insight("bmi")
        elif "cholesterol" in lower:
            add_insight("cholesterol")
        elif "systolic" in lower:
            add_insight("systolic")
        elif "diastolic" in lower:
            add_insight("diastolic")
        elif "smoking" in lower:
            add_insight("smoking")
        elif "physical_activity" in lower:
            add_insight("physical_activity")
        elif "family_history" in lower:
            add_insight("family_history")
        elif "spo2" in lower:
            add_insight("spo2")
        elif "sdnn" in lower:
            add_insight("sdnn")
        elif "rmssd" in lower:
            add_insight("rmssd")

    return insights

def format_contributors(contributors: Iterable[Tuple[str, float]], sign: str) -> List[str]:
    lines = []
    for name, _ in contributors:
        lines.append(f"{sign} {_pretty_feature_name(name)}")
    return lines


def build_report(
    risk_score: float,
    risk_level: str,
    positive: Iterable[Tuple[str, float]],
    negative: Iterable[Tuple[str, float]],
    insights: Iterable[str],
) -> List[str]:
    lines = [
        "-----------------------------------",
        "Prediction Summary",
        "-----------------------------------",
        "",
        f"Risk Score: {risk_score:.2f}",
        f"Risk Level: {risk_level}",
        "",
        "Top Contributors:",
    ]

    lines.extend(format_contributors(positive, "+"))
    lines.extend(format_contributors(negative, "-"))
    lines.append("")
    lines.append("Insights:")
    for insight in insights:
        lines.append(f"- {insight}")

    return lines
