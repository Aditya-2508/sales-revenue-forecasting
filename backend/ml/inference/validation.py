import pandas as pd


def validate_forecast_output(
    forecast_df: pd.DataFrame,
) -> dict:
    """
    Validate the structure and basic integrity of a forecast output.
    """

    required_columns = {
        "Date",
        "Revenue",
        "PredictedRevenue",
    }

    missing_columns = sorted(
        required_columns - set(forecast_df.columns)
    )

    has_required_columns = len(missing_columns) == 0

    if has_required_columns:
        missing_predictions = int(
            forecast_df["PredictedRevenue"].isna().sum()
        )
        duplicate_dates = int(
            forecast_df["Date"].duplicated().sum()
        )
        negative_predictions = int(
            (forecast_df["PredictedRevenue"] < 0).sum()
        )
        dates_valid = bool(
            forecast_df["Date"].notna().all()
            and forecast_df["Date"].is_monotonic_increasing
        )
    else:
        missing_predictions = 0
        duplicate_dates = 0
        negative_predictions = 0
        dates_valid = False

    valid = bool(
        has_required_columns
        and len(forecast_df) > 0
        and dates_valid
        and missing_predictions == 0
        and duplicate_dates == 0
    )

    return {
        "Valid": valid,
        "Rows": int(len(forecast_df)),
        "MissingColumns": missing_columns,
        "MissingPredictions": missing_predictions,
        "DuplicateDates": duplicate_dates,
        "NegativePredictions": negative_predictions,
    }