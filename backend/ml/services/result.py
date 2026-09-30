from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class ForecastResult:
    forecast: pd.DataFrame
    metrics: dict
    validation: dict
    output_path: Path
    training_rows: int
    test_rows: int