import pandas as pd


def create_forecasting_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create calendar, lag, and rolling features for daily revenue forecasting.
    """

    feature_df = daily_df.copy()

    # Calendar features
    feature_df["DayOfWeek"] = feature_df["Date"].dt.dayofweek
    feature_df["DayOfMonth"] = feature_df["Date"].dt.day
    feature_df["Month"] = feature_df["Date"].dt.month
    feature_df["WeekOfYear"] = feature_df["Date"].dt.isocalendar().week.astype(int)
    feature_df["IsWeekend"] = (
        feature_df["DayOfWeek"] >= 5
    ).astype(int)

    # Lag features
    feature_df["Lag1"] = feature_df["Revenue"].shift(1)
    feature_df["Lag7"] = feature_df["Revenue"].shift(7)
    feature_df["Lag14"] = feature_df["Revenue"].shift(14)
    feature_df["Lag28"] = feature_df["Revenue"].shift(28)

    # Exclude the current day's revenue from rolling features.
    historical_revenue = feature_df["Revenue"].shift(1)

    feature_df["RollingMean7"] = (
        historical_revenue.rolling(window=7).mean()
    )

    feature_df["RollingMean28"] = (
        historical_revenue.rolling(window=28).mean()
    )

    return feature_df