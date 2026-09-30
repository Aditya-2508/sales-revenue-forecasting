from xgboost import XGBRegressor

from backend.ml.models.config import FORECAST_MODEL_CONFIG


def train_xgboost_forecaster(
    train_df,
    feature_columns,
) -> XGBRegressor:
    X_train = train_df[feature_columns]
    y_train = train_df["Revenue"]

    model = XGBRegressor(**FORECAST_MODEL_CONFIG)

    model.fit(X_train, y_train)

    return model