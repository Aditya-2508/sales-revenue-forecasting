import pandas as pd


FEATURE_COLUMNS = [
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


def prepare_model_dataset(
    feature_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Prepare the model-ready forecasting dataset by removing
    rows without complete forecasting features.
    """

    model_df = feature_df.copy()

    model_df = model_df.dropna(
        subset=FEATURE_COLUMNS
    ).copy()

    model_df = model_df.sort_values("Date").reset_index(drop=True)

    return model_df, FEATURE_COLUMNS.copy()

def split_model_dataset(
    model_df: pd.DataFrame,
    train_end_date: str = "2011-09-25",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the model-ready dataset chronologically into training
    and test datasets.
    """

    train_end = pd.Timestamp(train_end_date)

    train_df = model_df.loc[
        model_df["Date"] <= train_end
    ].copy()

    test_df = model_df.loc[
        model_df["Date"] > train_end
    ].copy()

    train_df = train_df.sort_values("Date").reset_index(drop=True)
    test_df = test_df.sort_values("Date").reset_index(drop=True)

    return train_df, test_df