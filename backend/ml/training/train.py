from xgboost import XGBRegressor

from backend.ml.models.xgboost_model import train_xgboost_forecaster


def train_forecasting_model(
    train_df,
    feature_columns,
) -> XGBRegressor:
    """
    Train the configured revenue forecasting model.
    """

    return train_xgboost_forecaster(
        train_df=train_df,
        feature_columns=feature_columns,
    )