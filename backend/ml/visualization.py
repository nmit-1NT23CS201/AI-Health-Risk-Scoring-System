from typing import List

import matplotlib.pyplot as plt
import numpy as np
import shap


def save_summary_plot(
    shap_values: np.ndarray,
    X_processed: np.ndarray,
    feature_names: List[str],
    output_path: str,
) -> None:
    shap.summary_plot(
        shap_values, X_processed, feature_names=feature_names, show=False
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_bar_plot(
    shap_values: np.ndarray,
    X_processed: np.ndarray,
    feature_names: List[str],
    output_path: str,
) -> None:
    shap.summary_plot(
        shap_values,
        X_processed,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_waterfall_plot(
    shap_row: np.ndarray,
    base_value: float,
    data_row: np.ndarray,
    feature_names: List[str],
    output_path: str,
) -> None:
    explanation = shap.Explanation(
        values=shap_row,
        base_values=base_value,
        data=data_row,
        feature_names=feature_names,
    )
    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
