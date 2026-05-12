from typing import Dict, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def split_data(X, y, test_size: float, random_state: int):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_model(preprocessor, X_train, y_train, random_state: int) -> Pipeline:
    model = RandomForestRegressor(random_state=random_state)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(pipeline: Pipeline, X_test, y_test) -> Dict[str, float]:
    predictions = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    }


def save_model(pipeline: Pipeline, path: str) -> None:
    joblib.dump(pipeline, path)
