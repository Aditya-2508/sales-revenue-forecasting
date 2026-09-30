from pathlib import Path

import pandas as pd


def save_forecast(
    forecast_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """
    Save forecast results as a CSV artifact.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    forecast_df.to_csv(output_path, index=False)

    return output_path