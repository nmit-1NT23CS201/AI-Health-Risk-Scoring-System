import argparse
from pathlib import Path

from src.data_loader import load_dataset
from src.feature_engineering import select_features
from src.predict import generate_predictions
from src.risk_interpreter import (
    build_report,
    classify_risk,
    generate_insights,
    top_contributors,
)
from src.shap_explainer import compute_shap_values, load_model, prepare_shap_inputs
from src.train_model import split_data
from src.utils import ensure_dir, save_dataframe, save_text
from src.visualization import save_bar_plot, save_summary_plot, save_waterfall_plot


def run_pipeline(test_size: float, random_state: int, sample_index: int) -> None:
    project_root = Path(__file__).resolve().parent
    dataset_path = project_root / "dataset" / "indian_health_risk_dataset.csv"
    model_path = project_root / "models" / "random_forest_model.pkl"
    outputs_dir = project_root / "outputs"

    ensure_dir(str(outputs_dir))

    summary_plot_path = outputs_dir / "shap_summary_plot.png"
    bar_plot_path = outputs_dir / "shap_bar_plot.png"
    waterfall_plot_path = outputs_dir / "shap_waterfall_plot.png"
    interpretations_path = outputs_dir / "risk_interpretations.txt"
    predictions_path = outputs_dir / "predictions.csv"

    df = load_dataset(str(dataset_path))
    X, y = select_features(df)

    _, X_test, _, y_test = split_data(X, y, test_size=test_size, random_state=random_state)

    pipeline = load_model(str(model_path))

    predictions_df = generate_predictions(pipeline, X_test, y_test)
    save_dataframe(predictions_df, str(predictions_path))

    X_processed, feature_names = prepare_shap_inputs(pipeline, X_test)
    shap_values, base_value = compute_shap_values(pipeline, X_processed)

    if sample_index < 0 or sample_index >= X_processed.shape[0]:
        raise ValueError(
            f"Sample index {sample_index} is out of range (0-{X_processed.shape[0] - 1})."
        )

    sample_score = float(predictions_df.iloc[sample_index]["predicted_risk_score"])
    risk_level = classify_risk(sample_score)

    positive, negative = top_contributors(
        shap_values[sample_index], feature_names, top_n=6
    )
    insights = generate_insights(positive, negative)
    report_lines = build_report(sample_score, risk_level, positive, negative, insights)
    save_text(str(interpretations_path), report_lines)

    save_summary_plot(shap_values, X_processed, feature_names, str(summary_plot_path))
    save_bar_plot(shap_values, X_processed, feature_names, str(bar_plot_path))
    save_waterfall_plot(
        shap_values[sample_index],
        base_value,
        X_processed[sample_index],
        feature_names,
        str(waterfall_plot_path),
    )

    print("Risk Interpretation:")
    for line in report_lines:
        print(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Phase 2 SHAP explainability pipeline."
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data used for analysis (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Index of the test sample used for waterfall plot (default: 0)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        test_size=args.test_size,
        random_state=args.random_state,
        sample_index=args.sample_index,
    )
