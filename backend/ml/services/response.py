from backend.ml.services.result import ForecastResult


def build_forecast_response(result: ForecastResult) -> dict:
    """
    Convert a ForecastResult into a JSON-friendly response structure.
    """

    forecast_records = result.forecast.copy()

    forecast_records["Date"] = forecast_records["Date"].dt.strftime(
        "%Y-%m-%d"
    )

    return {
        "forecast": forecast_records.to_dict(orient="records"),
        "metrics": result.metrics,
        "validation": result.validation,
        "training_rows": result.training_rows,
        "test_rows": result.test_rows,
    }