from datetime import datetime, timezone

from backend.ml.models.config import FORECAST_MODEL_CONFIG
from backend.ml.features.model_dataset import FEATURE_COLUMNS


def build_model_metadata(
    train_df,
    model_name: str = "xgboost_revenue_forecaster",
) -> dict:
    """
    Build descriptive metadata for a trained forecasting model.
    """

    return {
        "model_name": model_name,
        "model_type": "XGBRegressor",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(len(train_df)),
        "training_start": train_df["Date"].min().isoformat(),
        "training_end": train_df["Date"].max().isoformat(),
        "feature_columns": FEATURE_COLUMNS.copy(),
        "model_config": FORECAST_MODEL_CONFIG.copy(),
    }