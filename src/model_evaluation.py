import os
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    ConfusionMatrixDisplay,
)


def score_to_class(score: float) -> str:
    if score < 35:
        return "Low Risk"
    if score < 70:
        return "Medium Risk"
    return "High Risk"


def evaluate_classification(y_true_scores, y_pred_scores) -> Dict[str, Any]:
    y_true = [score_to_class(v) for v in y_true_scores]
    y_pred = [score_to_class(v) for v in y_pred_scores]
    labels = ["Low Risk", "Medium Risk", "High Risk"]

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(
            y_true,
            y_pred,
            labels=labels,
            average="weighted",
            zero_division=0
        )),
        "recall": float(recall_score(
            y_true,
            y_pred,
            labels=labels,
            average="weighted",
            zero_division=0
        )),
        "f1": float(f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="weighted",
            zero_division=0
        )),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=labels
        ).tolist()
    }


def run_evaluation_pipeline(
    dataset_path: str,
    outputs_dir: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    from src.data_loader import load_dataset
    from src.feature_engineering import select_features
    from src.preprocessing import get_feature_groups, build_preprocessor
    from src.train_model import split_data, train_model
    from src.utils import ensure_dir

    # 1. Load dataset
    df = load_dataset(dataset_path)

    # 2. Select features
    X, y = select_features(df)

    # 3. Splits train/test data
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=random_state
    )

    # 4. Train RandomForestRegressor
    cat_cols, num_cols = get_feature_groups(X)
    preprocessor = build_preprocessor(X_train, cat_cols, num_cols)
    pipeline = train_model(preprocessor, X_train, y_train, random_state=random_state)

    # 5. Evaluate regression metrics
    predictions = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    regression_metrics = {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2)
    }

    # 6. Converts predicted scores into risk levels & computes classification metrics
    classification_results = evaluate_classification(y_test, predictions)
    classification_metrics = {
        "accuracy": classification_results["accuracy"],
        "precision": classification_results["precision"],
        "recall": classification_results["recall"],
        "f1": classification_results["f1"]
    }

    # 7. Save outputs
    ensure_dir(outputs_dir)
    
    # Save metrics.json
    metrics_path = os.path.join(outputs_dir, "metrics.json")
    metrics_data = {
        "regression_metrics": regression_metrics,
        "classification_metrics": classification_metrics
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=4)

    # Save confusion matrix plot
    y_true_labels = [score_to_class(v) for v in y_test]
    y_pred_labels = [score_to_class(v) for v in predictions]
    labels = ["Low Risk", "Medium Risk", "High Risk"]

    plt.figure(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true_labels,
        y_pred_labels,
        labels=labels,
        cmap=plt.cm.Blues,
        ax=plt.gca()
    )
    plt.title("Confusion Matrix (Risk Levels)")
    cm_path = os.path.join(outputs_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"Evaluation pipeline completed. Outputs saved to {outputs_dir}/")
    return metrics_data


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    dataset_csv = project_root / "dataset" / "indian_health_risk_dataset.csv"
    outputs_path = project_root / "outputs"
    run_evaluation_pipeline(
        dataset_path=str(dataset_csv),
        outputs_dir=str(outputs_path)
    )