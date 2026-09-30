import pandas as pd

from backend.ml.evaluation.metrics import (
    calculate_mae,
    calculate_rmse,
    calculate_wape,
)


def evaluate_forecast(
    test_df: pd.DataFrame,
    predictions,
) -> dict:
    """
    Evaluate forecast predictions against the actual revenue values.
    """

    actual = test_df["Revenue"].to_numpy()
    predicted = predictions

    errors = actual - predicted

    return {
        "MAE": calculate_mae(actual, predicted),
        "RMSE": calculate_rmse(actual, predicted),
        "WAPE": calculate_wape(actual, predicted),
        "NegativePredictions": int((predicted < 0).sum()),
        "MinPrediction": float(predicted.min()),
        "MaxPrediction": float(predicted.max()),
        "LargestAbsoluteError": float(abs(errors).max()),
    }
