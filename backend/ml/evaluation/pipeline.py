import pandas as pd

from backend.ml.evaluation.forecast_evaluation import evaluate_forecast


def run_forecast_evaluation(
    test_df: pd.DataFrame,
    predictions,
) -> dict:
    """
    Evaluate forecast predictions against the test dataset.
    """

    return evaluate_forecast(
        test_df=test_df,
        predictions=predictions,
    )