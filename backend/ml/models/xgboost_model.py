from xgboost import XGBRegressor
import pandas as pd


def train_xgboost_forecaster(
    train_df: pd.DataFrame,
    feature_columns: list[str],
) -> XGBRegressor:
    """
    Train the XGBoost revenue forecasting model.
    """

    X_train = train_df[feature_columns]
    y_train = train_df["Revenue"]

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    return model