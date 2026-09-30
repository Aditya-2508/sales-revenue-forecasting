import pandas as pd


def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """Create the primary cleaned customer/product sales dataset."""

    non_standard_stock_codes = {
        "AMAZONFEE",
        "POST",
        "M",
        "D",
        "S",
        "BANK CHARGES",
        "CRUK",
        "B",
        "DOT",
    }

    cleaned_df = df.copy()

    # Remove exact duplicate records.
    cleaned_df = cleaned_df.drop_duplicates()

    # Convert transaction dates safely.
    cleaned_df["InvoiceDate"] = pd.to_datetime(
        cleaned_df["InvoiceDate"],
        errors="coerce",
    )

    invoice_numbers = cleaned_df["InvoiceNo"].astype(str)

    valid_sale_mask = (
        cleaned_df["CustomerID"].notna()
        & cleaned_df["InvoiceDate"].notna()
        & ~invoice_numbers.str.startswith("C")
        & (cleaned_df["Quantity"] > 0)
        & (cleaned_df["UnitPrice"] > 0)
        & ~cleaned_df["StockCode"].isin(non_standard_stock_codes)
    )

    cleaned_df = cleaned_df.loc[valid_sale_mask].copy()

    # Calculate transaction revenue.
    cleaned_df["Revenue"] = (
        cleaned_df["Quantity"] * cleaned_df["UnitPrice"]
    )

    return cleaned_df