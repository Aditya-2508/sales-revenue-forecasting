from pathlib import Path

import pandas as pd
from backend.ml.services.result import ForecastResult

from backend.ml.evaluation.pipeline import run_forecast_evaluation
from backend.ml.features.forecasting_features import create_forecasting_features
from backend.ml.features.model_dataset import (
    prepare_model_dataset,
    split_model_dataset,
)
from backend.ml.features.time_series import prepare_daily_revenue_series
from backend.ml.inference.forecast import generate_forecast
from backend.ml.inference.output import save_forecast
from backend.ml.inference.validation import validate_forecast_output
from backend.ml.models.xgboost_model import train_xgboost_forecaster
from backend.ml.preprocessing.cleaning import clean_sales_data


def run_forecasting_service(
    raw_df: pd.DataFrame,
    output_path: str | Path,
) -> ForecastResult:
    """
    Execute the complete revenue forecasting workflow.
    """

    cleaned_df = clean_sales_data(raw_df)

    daily_df = prepare_daily_revenue_series(cleaned_df)

    feature_df = create_forecasting_features(daily_df)

    model_df, feature_columns = prepare_model_dataset(feature_df)

    train_df, test_df = split_model_dataset(model_df)

    model = train_xgboost_forecaster(
        train_df=train_df,
        feature_columns=feature_columns,
    )

    predictions = model.predict(test_df[feature_columns])

    forecast_df = generate_forecast(
        model=model,
        data=test_df,
        feature_columns=feature_columns,
    )

    validation = validate_forecast_output(forecast_df)

    metrics = run_forecast_evaluation(
        test_df=test_df,
        predictions=predictions,
    )

    output_file = save_forecast(
        forecast_df=forecast_df,
        output_path=output_path,
    )

    return ForecastResult(
        forecast=forecast_df,
        metrics=metrics,
        validation=validation,
        output_path=output_file,
        training_rows=len(train_df),
        test_rows=len(test_df),
    )