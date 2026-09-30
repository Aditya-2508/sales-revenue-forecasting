import pandas as pd

from backend.ml.features.forecasting_features import create_forecasting_features


def test_forecasting_features_create_expected_columns():
    dates = pd.date_range(
        start="2011-01-01",
        periods=35,
        freq="D",
    )

    daily_df = pd.DataFrame(
        {
            "Date": dates,
            "Revenue": range(1, 36),
        }
    )

    feature_df = create_forecasting_features(daily_df)

    expected_columns = [
        "Date",
        "Revenue",
        "DayOfWeek",
        "DayOfMonth",
        "Month",
        "WeekOfYear",
        "IsWeekend",
        "Lag1",
        "Lag7",
        "Lag14",
        "Lag28",
        "RollingMean7",
        "RollingMean28",
    ]

    assert feature_df.columns.tolist() == expected_columns
    assert len(feature_df) == 35


def test_lag_features_use_previous_revenue():
    dates = pd.date_range(
        start="2011-01-01",
        periods=35,
        freq="D",
    )

    daily_df = pd.DataFrame(
        {
            "Date": dates,
            "Revenue": range(1, 36),
        }
    )

    feature_df = create_forecasting_features(daily_df)

    assert pd.isna(feature_df.loc[0, "Lag1"])
    assert feature_df.loc[1, "Lag1"] == 1
    assert feature_df.loc[7, "Lag7"] == 1
    assert feature_df.loc[14, "Lag14"] == 1
    assert feature_df.loc[28, "Lag28"] == 1


def test_rolling_features_exclude_current_revenue():
    dates = pd.date_range(
        start="2011-01-01",
        periods=35,
        freq="D",
    )

    daily_df = pd.DataFrame(
        {
            "Date": dates,
            "Revenue": range(1, 36),
        }
    )

    feature_df = create_forecasting_features(daily_df)

    # On day 8, the 7-day rolling mean must use days 1-7,
    # not the current day's revenue.
    assert feature_df.loc[7, "RollingMean7"] == 4.0

    # On day 29, the 28-day rolling mean must use days 1-28.
    assert feature_df.loc[28, "RollingMean28"] == 14.5


def test_calendar_features_are_correct():
    dates = pd.to_datetime(
        [
            "2011-01-01",  # Saturday
            "2011-01-02",  # Sunday
            "2011-01-03",  # Monday
        ]
    )

    daily_df = pd.DataFrame(
        {
            "Date": dates,
            "Revenue": [100.0, 200.0, 300.0],
        }
    )

    feature_df = create_forecasting_features(daily_df)

    assert feature_df.loc[0, "DayOfWeek"] == 5
    assert feature_df.loc[1, "DayOfWeek"] == 6
    assert feature_df.loc[2, "DayOfWeek"] == 0

    assert feature_df.loc[0, "IsWeekend"] == 1
    assert feature_df.loc[1, "IsWeekend"] == 1
    assert feature_df.loc[2, "IsWeekend"] == 0