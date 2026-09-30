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