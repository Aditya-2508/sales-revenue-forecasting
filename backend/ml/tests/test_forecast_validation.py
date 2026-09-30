import pandas as pd

from backend.ml.inference.validation import validate_forecast_output


def test_valid_forecast_output():
    forecast_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2011-09-26",
                    "2011-09-27",
                    "2011-09-28",
                ]
            ),
            "Revenue": [100.0, 200.0, 150.0],
            "PredictedRevenue": [110.0, 190.0, 160.0],
        }
    )

    result = validate_forecast_output(forecast_df)

    assert result["Valid"] is True
    assert result["Rows"] == 3
    assert result["MissingColumns"] == []
    assert result["MissingPredictions"] == 0
    assert result["DuplicateDates"] == 0
    assert result["NegativePredictions"] == 0


def test_forecast_with_missing_prediction_is_invalid():
    forecast_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2011-09-26",
                    "2011-09-27",
                ]
            ),
            "Revenue": [100.0, 200.0],
            "PredictedRevenue": [110.0, None],
        }
    )

    result = validate_forecast_output(forecast_df)

    assert result["Valid"] is False
    assert result["MissingPredictions"] == 1


def test_forecast_with_duplicate_dates_is_invalid():
    forecast_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2011-09-26",
                    "2011-09-26",
                ]
            ),
            "Revenue": [100.0, 200.0],
            "PredictedRevenue": [110.0, 190.0],
        }
    )

    result = validate_forecast_output(forecast_df)

    assert result["Valid"] is False
    assert result["DuplicateDates"] == 1


def test_forecast_with_missing_column_is_invalid():
    forecast_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2011-09-26",
                    "2011-09-27",
                ]
            ),
            "Revenue": [100.0, 200.0],
        }
    )

    result = validate_forecast_output(forecast_df)

    assert result["Valid"] is False
    assert result["MissingColumns"] == ["PredictedRevenue"]