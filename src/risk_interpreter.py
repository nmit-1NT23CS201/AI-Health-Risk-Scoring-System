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
    name = name.replace("_", " ")
    return " ".join(word.capitalize() for word in name.split())


def top_contributors(
    shap_row: np.ndarray, feature_names: List[str], top_n: int = 5
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    indices = np.argsort(np.abs(shap_row))[::-1][:top_n]
    top_items = [(feature_names[i], float(shap_row[i])) for i in indices]

    positive = [(name, value) for name, value in top_items if value > 0]
    negative = [(name, value) for name, value in top_items if value < 0]
    return positive, negative


def generate_insights(
    positive: Iterable[Tuple[str, float]], negative: Iterable[Tuple[str, float]]
) -> List[str]:
    insights: List[str] = []

    for name, _ in positive:
        pretty = _pretty_feature_name(name)
        insights.append(f"{pretty} is increasing the predicted risk.")

    for name, _ in negative:
        pretty = _pretty_feature_name(name)
        insights.append(f"{pretty} is helping reduce the predicted risk.")

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
