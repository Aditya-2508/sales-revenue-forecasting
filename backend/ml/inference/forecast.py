import pandas as pd
from xgboost import XGBRegressor


def generate_forecast(
    model: XGBRegressor,
    data: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Generate revenue predictions for model-ready data.
    """

    predictions = model.predict(data[feature_columns])

    forecast_df = data[["Date", "Revenue"]].copy()
    forecast_df["PredictedRevenue"] = predictions

    return forecast_df