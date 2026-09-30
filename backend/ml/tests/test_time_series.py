import pandas as pd

from backend.ml.features.time_series import prepare_daily_revenue_series


def test_prepare_daily_revenue_series_aggregates_revenue():
    df = pd.DataFrame(
        {
            "InvoiceDate": pd.to_datetime(
                [
                    "2011-01-01 09:00:00",
                    "2011-01-01 12:00:00",
                    "2011-01-03 10:00:00",
                ]
            ),
            "Revenue": [100.0, 50.0, 200.0],
        }
    )

    result = prepare_daily_revenue_series(df)

    assert len(result) == 3
    assert result.loc[
        result["Date"] == pd.Timestamp("2011-01-01"),
        "Revenue",
    ].iloc[0] == 150.0

    assert result.loc[
        result["Date"] == pd.Timestamp("2011-01-03"),
        "Revenue",
    ].iloc[0] == 200.0


def test_prepare_daily_revenue_series_fills_missing_dates_with_zero():
    df = pd.DataFrame(
        {
            "InvoiceDate": pd.to_datetime(
                [
                    "2011-01-01 09:00:00",
                    "2011-01-03 10:00:00",
                ]
            ),
            "Revenue": [100.0, 200.0],
        }
    )

    result = prepare_daily_revenue_series(df)

    assert result["Date"].tolist() == [
        pd.Timestamp("2011-01-01"),
        pd.Timestamp("2011-01-02"),
        pd.Timestamp("2011-01-03"),
    ]

    assert result.loc[
        result["Date"] == pd.Timestamp("2011-01-02"),
        "Revenue",
    ].iloc[0] == 0.0


def test_prepare_daily_revenue_series_preserves_total_revenue():
    df = pd.DataFrame(
        {
            "InvoiceDate": pd.to_datetime(
                [
                    "2011-01-01 09:00:00",
                    "2011-01-01 12:00:00",
                    "2011-01-03 10:00:00",
                ]
            ),
            "Revenue": [100.0, 50.0, 200.0],
        }
    )

    result = prepare_daily_revenue_series(df)

    assert result["Revenue"].sum() == df["Revenue"].sum()


def test_prepare_daily_revenue_series_returns_chronological_dates():
    df = pd.DataFrame(
        {
            "InvoiceDate": pd.to_datetime(
                [
                    "2011-01-03 10:00:00",
                    "2011-01-01 09:00:00",
                ]
            ),
            "Revenue": [200.0, 100.0],
        }
    )

    result = prepare_daily_revenue_series(df)

    assert result["Date"].is_monotonic_increasing