import pandas as pd

from backend.ml.services.forecasting import run_forecasting_service
from backend.ml.services.result import ForecastResult


def test_forecasting_service_returns_valid_result(tmp_path):
    dates = pd.date_range(
        start="2010-12-01",
        periods=320,
        freq="D",
    )

    raw_df = pd.DataFrame(
        {
            "InvoiceNo": [str(10000 + i) for i in range(320)],
            "StockCode": ["100"] * 320,
            "Description": ["Product A"] * 320,
            "Quantity": [2] * 320,
            "InvoiceDate": dates,
            "UnitPrice": [10.0] * 320,
            "CustomerID": list(range(1, 321)),
            "Country": ["United Kingdom"] * 320,
        }
    )

    output_path = tmp_path / "forecast.csv"

    result = run_forecasting_service(
        raw_df,
        output_path,
    )

    assert isinstance(result, ForecastResult)
    assert result.output_path == output_path
    assert result.output_path.exists()
    assert len(result.forecast) > 0
    assert result.validation["Valid"] is True
    assert result.forecast["PredictedRevenue"].notna().all()