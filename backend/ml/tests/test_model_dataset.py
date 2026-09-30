import pandas as pd

from backend.ml.features.model_dataset import (
    prepare_model_dataset,
    split_model_dataset,
)


def test_prepare_model_dataset_removes_warmup_rows():
    dates = pd.date_range(
        start="2010-12-01",
        periods=35,
        freq="D",
    )

    feature_df = pd.DataFrame(
        {
            "Date": dates,
            "Revenue": range(35),
            "DayOfWeek": [0] * 35,
            "DayOfMonth": list(range(1, 29)) + list(range(1, 8)),
            "Month": [12] * 31 + [1] * 4,
            "WeekOfYear": [48] * 35,
            "IsWeekend": [0] * 35,
            "Lag1": [None] + list(range(34)),
            "Lag7": [None] * 7 + list(range(28)),
            "Lag14": [None] * 14 + list(range(21)),
            "Lag28": [None] * 28 + list(range(7)),
            "RollingMean7": [None] * 7 + [1.0] * 28,
            "RollingMean28": [None] * 28 + [1.0] * 7,
        }
    )

    model_df, feature_columns = prepare_model_dataset(feature_df)

    assert len(model_df) == 7
    assert len(feature_columns) == 11
    assert model_df["Date"].min() == pd.Timestamp("2010-12-29")
    assert model_df["Date"].is_monotonic_increasing
    assert model_df[feature_columns].isna().sum().sum() == 0


def test_split_model_dataset_is_chronological():
    dates = pd.date_range(
        start="2011-01-01",
        periods=10,
        freq="D",
    )

    model_df = pd.DataFrame(
        {
            "Date": dates,
            "Revenue": range(10),
        }
    )

    train_df, test_df = split_model_dataset(
        model_df,
        train_end_date="2011-01-06",
    )

    assert len(train_df) == 6
    assert len(test_df) == 4

    assert train_df["Date"].max() == pd.Timestamp("2011-01-06")
    assert test_df["Date"].min() == pd.Timestamp("2011-01-07")

    assert train_df["Date"].max() < test_df["Date"].min()
    assert train_df["Date"].is_monotonic_increasing
    assert test_df["Date"].is_monotonic_increasing