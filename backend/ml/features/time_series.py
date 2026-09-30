import pandas as pd


def prepare_daily_revenue_series(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a complete daily revenue time series.

    Missing calendar dates are included with zero revenue.
    """

    daily_df = (
        df.assign(
            Date=df["InvoiceDate"].dt.normalize()
        )
        .groupby("Date", as_index=False)["Revenue"]
        .sum()
    )

    full_date_range = pd.date_range(
        start=daily_df["Date"].min(),
        end=daily_df["Date"].max(),
        freq="D",
    )

    daily_df = (
        daily_df.set_index("Date")
        .reindex(full_date_range, fill_value=0)
        .rename_axis("Date")
        .reset_index()
    )

    daily_df["Revenue"] = daily_df["Revenue"].astype(float)

    return daily_df