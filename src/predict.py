import pandas as pd


def generate_predictions(pipeline, X_test, y_test) -> pd.DataFrame:
    predicted = pipeline.predict(X_test)

    result = pd.DataFrame(
        {
            "actual_risk_score": y_test.values,
            "predicted_risk_score": predicted,
        }
    )

    return result
