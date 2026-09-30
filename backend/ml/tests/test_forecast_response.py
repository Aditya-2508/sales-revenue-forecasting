import pandas as pd

from backend.ml.services.response import build_forecast_response
from backend.ml.services.result import ForecastResult


def test_build_forecast_response():
    forecast_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2011-09-26",
                    "2011-09-27",
                ]
            ),
            "Revenue": [100.0, 200.0],
            "PredictedRevenue": [110.0, 190.0],
        }
    )

    result = ForecastResult(
        forecast=forecast_df,
        metrics={
            "MAE": 10.0,
            "RMSE": 10.0,
            "WAPE": 0.05,
        },
        validation={
            "Valid": True,
        },
        output_path=__import__("pathlib").Path(
            "data/forecasts/test.csv"
        ),
        training_rows=271,
        test_rows=2,
    )

    response = build_forecast_response(result)

    assert isinstance(response, dict)
    assert set(response.keys()) == {
        "forecast",
        "metrics",
        "validation",
        "training_rows",
        "test_rows",
    }

    assert len(response["forecast"]) == 2
    assert response["forecast"][0]["Date"] == "2011-09-26"
    assert response["forecast"][1]["Date"] == "2011-09-27"

    assert response["forecast"][0]["Revenue"] == 100.0
    assert response["forecast"][0]["PredictedRevenue"] == 110.0

    assert response["metrics"]["MAE"] == 10.0
    assert response["validation"]["Valid"] is True
    assert response["training_rows"] == 271
    assert response["test_rows"] == 2