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

    return {
        "Valid": (
            len(missing_columns) == 0
            and len(forecast_df) > 0
            and forecast_df["Date"].notna().all()
            and forecast_df["PredictedRevenue"].notna().all()
            and forecast_df["Date"].is_monotonic_increasing
        ),
        "Rows": int(len(forecast_df)),
        "MissingColumns": missing_columns,
        "MissingPredictions": int(
            forecast_df["PredictedRevenue"].isna().sum()
        ),
        "DuplicateDates": int(
            forecast_df["Date"].duplicated().sum()
        ),
        "NegativePredictions": int(
            (forecast_df["PredictedRevenue"] < 0).sum()
        ),
    }