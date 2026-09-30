from pathlib import Path

import pandas as pd
import numpy as np
from xgboost import XGBRegressor


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "Online Retail.xlsx"


# ============================================================
# Dataset loading
# ============================================================

def load_dataset() -> pd.DataFrame:
    """
    Load the original UCI Online Retail dataset.

    The raw dataset is never modified or overwritten.
    """

    print(f"Loading dataset from: {DATASET_PATH}")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATASET_PATH}\n"
            "Please download 'Online Retail.xlsx' from the "
            "UCI Machine Learning Repository and place it "
            "inside data/raw/."
        )

    return pd.read_excel(DATASET_PATH)


# ============================================================
# Schema profiling
# ============================================================

def profile_schema(df: pd.DataFrame) -> None:
    """Display detailed information about the dataset schema."""

    print("\n" + "=" * 70)
    print("DATASET SCHEMA")
    print("=" * 70)

    schema = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "non_null_count": df.notna().sum().values,
        "null_count": df.isna().sum().values,
        "unique_values": df.nunique(dropna=True).values,
    })

    schema["null_percentage"] = (
        schema["null_count"] / len(df) * 100
    ).round(2)

    print("\n")
    print(schema.to_string(index=False))

    print("\n" + "-" * 70)
    print("PANDAS INFO")
    print("-" * 70)

    df.info()

# ============================================================
# Missing-value profiling
# ============================================================

def profile_missing_values(df: pd.DataFrame) -> None:
    """Analyze missing values in the dataset."""

    print("\n" + "=" * 70)
    print("MISSING-VALUE ANALYSIS")
    print("=" * 70)

    missing = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isna().sum().values,
        "missing_percentage": (
            df.isna().mean().values * 100
        ).round(2),
    })

    missing = missing.sort_values(
        by="missing_count",
        ascending=False,
    )

    print("\nMissing values by column:")
    print(missing.to_string(index=False))

    # --------------------------------------------------------
    # Description
    # --------------------------------------------------------

    description_missing = df[df["Description"].isna()]

    print("\n" + "-" * 70)
    print("MISSING DESCRIPTION")
    print("-" * 70)

    print(
        f"Rows with missing Description: "
        f"{len(description_missing)}"
    )

    print("\nSample records:")
    print(
        description_missing[
            [
                "InvoiceNo",
                "StockCode",
                "Quantity",
                "InvoiceDate",
                "UnitPrice",
                "CustomerID",
                "Country",
            ]
        ].head(10).to_string(index=False)
    )

    # --------------------------------------------------------
    # CustomerID
    # --------------------------------------------------------

    customer_missing = df[df["CustomerID"].isna()]

    print("\n" + "-" * 70)
    print("MISSING CUSTOMER ID")
    print("-" * 70)

    print(
        f"Rows with missing CustomerID: "
        f"{len(customer_missing)}"
    )

    print(
        f"Percentage of dataset: "
        f"{len(customer_missing) / len(df) * 100:.2f}%"
    )

    print("\nSample records:")
    print(
        customer_missing[
            [
                "InvoiceNo",
                "StockCode",
                "Description",
                "Quantity",
                "InvoiceDate",
                "UnitPrice",
                "Country",
            ]
        ].head(10).to_string(index=False)
    )

# ============================================================
# Duplicate profiling
# ============================================================

def profile_duplicates(df: pd.DataFrame) -> None:
    """Analyze duplicate records in the dataset."""

    print("\n" + "=" * 70)
    print("DUPLICATE RECORD ANALYSIS")
    print("=" * 70)

    duplicate_mask = df.duplicated(keep=False)
    duplicate_rows = df[duplicate_mask]

    duplicate_count = df.duplicated().sum()

    print(f"\nDuplicate rows excluding first occurrence: {duplicate_count}")

    print(
        f"Duplicate percentage of dataset: "
        f"{duplicate_count / len(df) * 100:.2f}%"
    )

    print(
        f"\nRows belonging to duplicate groups: "
        f"{len(duplicate_rows)}"
    )

    print(
        f"Duplicate-group row percentage: "
        f"{len(duplicate_rows) / len(df) * 100:.2f}%"
    )

    if duplicate_count > 0:
        print("\nSample duplicate groups:")

        duplicate_rows = duplicate_rows.sort_values(
            by=[
                "InvoiceNo",
                "StockCode",
                "InvoiceDate",
            ]
        )

        print(
            duplicate_rows.head(20).to_string(index=False)
        )
    else:
        print("\nNo duplicate rows found.")


# ============================================================
# Cancelled invoice profiling
# ============================================================

def profile_cancelled_invoices(df: pd.DataFrame) -> None:
    """Analyze cancelled invoices."""

    print("\n" + "=" * 70)
    print("CANCELLED INVOICE ANALYSIS")
    print("=" * 70)

    invoice_numbers = df["InvoiceNo"].astype(str)

    cancelled_mask = invoice_numbers.str.upper().str.startswith("C")

    cancelled_rows = df[cancelled_mask]
    normal_rows = df[~cancelled_mask]

    # --------------------------------------------------------
    # Basic counts
    # --------------------------------------------------------

    cancelled_row_count = len(cancelled_rows)
    normal_row_count = len(normal_rows)

    cancelled_invoice_count = (
        cancelled_rows["InvoiceNo"].nunique()
    )

    normal_invoice_count = (
        normal_rows["InvoiceNo"].nunique()
    )

    print(
        f"\nCancelled transaction rows: "
        f"{cancelled_row_count}"
    )

    print(
        f"Cancelled row percentage: "
        f"{cancelled_row_count / len(df) * 100:.2f}%"
    )

    print(
        f"\nUnique cancelled invoices: "
        f"{cancelled_invoice_count}"
    )

    print(
        f"Unique non-cancelled invoices: "
        f"{normal_invoice_count}"
    )

    # --------------------------------------------------------
    # Quantity analysis
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("CANCELLED QUANTITY ANALYSIS")
    print("-" * 70)

    print("\nQuantity statistics for cancelled rows:")
    print(
        cancelled_rows["Quantity"].describe().to_string()
    )

    negative_cancelled = (
        cancelled_rows["Quantity"] < 0
    ).sum()

    zero_cancelled = (
        cancelled_rows["Quantity"] == 0
    ).sum()

    positive_cancelled = (
        cancelled_rows["Quantity"] > 0
    ).sum()

    print(
        f"\nCancelled rows with negative quantity: "
        f"{negative_cancelled}"
    )

    print(
        f"Cancelled rows with zero quantity: "
        f"{zero_cancelled}"
    )

    print(
        f"Cancelled rows with positive quantity: "
        f"{positive_cancelled}"
    )

    # --------------------------------------------------------
    # Price analysis
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("CANCELLED PRICE ANALYSIS")
    print("-" * 70)

    print("\nUnitPrice statistics for cancelled rows:")
    print(
        cancelled_rows["UnitPrice"].describe().to_string()
    )

    # --------------------------------------------------------
    # Sample cancelled records
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("SAMPLE CANCELLED RECORDS")
    print("-" * 70)

    print(
        cancelled_rows[
            [
                "InvoiceNo",
                "StockCode",
                "Description",
                "Quantity",
                "InvoiceDate",
                "UnitPrice",
                "CustomerID",
                "Country",
            ]
        ].head(20).to_string(index=False)
    )

    # --------------------------------------------------------
    # Non-cancelled comparison
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("CANCELLED VS NON-CANCELLED")
    print("-" * 70)

    comparison = pd.DataFrame({
        "group": [
            "Cancelled",
            "Non-cancelled",
        ],
        "rows": [
            cancelled_row_count,
            normal_row_count,
        ],
        "unique_invoices": [
            cancelled_invoice_count,
            normal_invoice_count,
        ],
    })

    print(comparison.to_string(index=False))

# ============================================================
# Quantity profiling
# ============================================================

def profile_quantity(df: pd.DataFrame) -> None:
    """Analyze quantity values and extreme quantities."""

    print("\n" + "=" * 70)
    print("QUANTITY ANALYSIS")
    print("=" * 70)

    negative = df[df["Quantity"] < 0]
    zero = df[df["Quantity"] == 0]
    positive = df[df["Quantity"] > 0]

    print("\nQuantity statistics:")
    print(df["Quantity"].describe().to_string())

    print("\n" + "-" * 70)
    print("QUANTITY GROUPS")
    print("-" * 70)

    print(f"\nNegative quantity rows: {len(negative)}")
    print(
        f"Negative percentage: "
        f"{len(negative) / len(df) * 100:.2f}%"
    )

    print(f"\nZero quantity rows: {len(zero)}")
    print(
        f"Zero percentage: "
        f"{len(zero) / len(df) * 100:.2f}%"
    )

    print(f"\nPositive quantity rows: {len(positive)}")
    print(
        f"Positive percentage: "
        f"{len(positive) / len(df) * 100:.2f}%"
    )

    print("\n" + "-" * 70)
    print("LARGEST POSITIVE QUANTITIES")
    print("-" * 70)

    print(
        df.nlargest(10, "Quantity")[
            [
                "InvoiceNo",
                "StockCode",
                "Description",
                "Quantity",
                "InvoiceDate",
                "UnitPrice",
                "CustomerID",
                "Country",
            ]
        ].to_string(index=False)
    )

    print("\n" + "-" * 70)
    print("LARGEST NEGATIVE QUANTITIES")
    print("-" * 70)

    print(
        df.nsmallest(10, "Quantity")[
            [
                "InvoiceNo",
                "StockCode",
                "Description",
                "Quantity",
                "InvoiceDate",
                "UnitPrice",
                "CustomerID",
                "Country",
            ]
        ].to_string(index=False)
    )

    print("\n" + "-" * 70)
    print("NEGATIVE QUANTITIES NOT MARKED AS CANCELLATIONS")
    print("-" * 70)

    invoice_numbers = df["InvoiceNo"].astype(str)

    non_cancelled_negative = df[
        (df["Quantity"] < 0)
        & (~invoice_numbers.str.upper().str.startswith("C"))
    ]

    print(
        f"\nRows with negative quantity but "
        f"non-cancellation InvoiceNo: "
        f"{len(non_cancelled_negative)}"
    )

    if len(non_cancelled_negative) > 0:
        print("\nSample:")
        print(
            non_cancelled_negative[
                [
                    "InvoiceNo",
                    "StockCode",
                    "Description",
                    "Quantity",
                    "InvoiceDate",
                    "UnitPrice",
                    "CustomerID",
                    "Country",
                ]
            ].head(20).to_string(index=False)
        )


# ============================================================
# Unit price profiling
# ============================================================

def profile_unit_price(df: pd.DataFrame) -> None:
    """Analyze UnitPrice values and extreme prices."""

    print("\n" + "=" * 70)
    print("UNIT PRICE ANALYSIS")
    print("=" * 70)

    negative = df[df["UnitPrice"] < 0]
    zero = df[df["UnitPrice"] == 0]
    positive = df[df["UnitPrice"] > 0]

    print("\nUnitPrice statistics:")
    print(df["UnitPrice"].describe().to_string())

    print("\n" + "-" * 70)
    print("PRICE GROUPS")
    print("-" * 70)

    print(f"\nNegative UnitPrice rows: {len(negative)}")
    print(
        f"Negative percentage: "
        f"{len(negative) / len(df) * 100:.2f}%"
    )

    print(f"\nZero UnitPrice rows: {len(zero)}")
    print(
        f"Zero percentage: "
        f"{len(zero) / len(df) * 100:.2f}%"
    )

    print(f"\nPositive UnitPrice rows: {len(positive)}")
    print(
        f"Positive percentage: "
        f"{len(positive) / len(df) * 100:.2f}%"
    )

    print("\n" + "-" * 70)
    print("LARGEST UNIT PRICES")
    print("-" * 70)

    print(
        df.nlargest(10, "UnitPrice")[
            [
                "InvoiceNo",
                "StockCode",
                "Description",
                "Quantity",
                "InvoiceDate",
                "UnitPrice",
                "CustomerID",
                "Country",
            ]
        ].to_string(index=False)
    )

    print("\n" + "-" * 70)
    print("ZERO-PRICE TRANSACTIONS")
    print("-" * 70)

    print(
        zero[
            [
                "InvoiceNo",
                "StockCode",
                "Description",
                "Quantity",
                "InvoiceDate",
                "UnitPrice",
                "CustomerID",
                "Country",
            ]
        ].head(20).to_string(index=False)
    )

    print("\n" + "-" * 70)
    print("NEGATIVE-PRICE TRANSACTIONS")
    print("-" * 70)

    if len(negative) > 0:
        print(
            negative[
                [
                    "InvoiceNo",
                    "StockCode",
                    "Description",
                    "Quantity",
                    "InvoiceDate",
                    "UnitPrice",
                    "CustomerID",
                    "Country",
                ]
            ].head(20).to_string(index=False)
        )
    else:
        print("\nNo negative UnitPrice rows found.")

# ============================================================
# Date profiling
# ============================================================

def profile_dates(df: pd.DataFrame) -> None:
    """Analyze the InvoiceDate field."""

    print("\n" + "=" * 70)
    print("DATE ANALYSIS")
    print("=" * 70)

    invoice_dates = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce",
    )

    invalid_dates = invoice_dates.isna().sum()

    print(f"\nInvalid InvoiceDate values: {invalid_dates}")

    if invalid_dates == 0:
        print("All InvoiceDate values are valid datetimes.")

    min_date = invoice_dates.min()
    max_date = invoice_dates.max()

    print(f"\nEarliest transaction: {min_date}")
    print(f"Latest transaction:   {max_date}")

    print(
        f"\nUnique transaction dates: "
        f"{invoice_dates.dt.date.nunique()}"
    )

    print(
        f"Unique months: "
        f"{invoice_dates.dt.to_period('M').nunique()}"
    )

    print(
        f"Unique years: "
        f"{invoice_dates.dt.year.nunique()}"
    )

    print("\nTransactions by year:")

    yearly_counts = (
        invoice_dates
        .dt.year
        .value_counts()
        .sort_index()
    )

    print(yearly_counts.to_string())

    print("\nTransactions by month:")

    monthly_counts = (
        invoice_dates
        .dt.to_period("M")
        .value_counts()
        .sort_index()
    )

    print(monthly_counts.to_string())

def profile_entities(df: pd.DataFrame) -> None:
    """Analyze unique customers, products, and countries."""

    print("\n" + "=" * 70)
    print("ENTITY ANALYSIS")
    print("=" * 70)

    print("\nUnique customers:")

    unique_customers = df["CustomerID"].nunique(dropna=True)

    print(f"Unique CustomerIDs: {unique_customers}")

    print(
        f"Rows with CustomerID: "
        f"{df['CustomerID'].notna().sum()}"
    )

    print(
        f"Rows without CustomerID: "
        f"{df['CustomerID'].isna().sum()}"
    )

    print("\nUnique products:")

    unique_stock_codes = df["StockCode"].nunique()

    print(f"Unique StockCodes: {unique_stock_codes}")

    print("\nUnique countries:")

    unique_countries = df["Country"].nunique()

    print(f"Unique Countries: {unique_countries}")

    print("\nCountries:")

    country_counts = (
        df["Country"]
        .value_counts()
        .sort_values(ascending=False)
    )

    print(country_counts.to_string())

    print("\nTop 10 products by transaction rows:")

    product_counts = (
        df["StockCode"]
        .value_counts()
        .head(10)
    )

    print(product_counts.to_string())

def profile_revenue(df: pd.DataFrame) -> None:
    """Analyze transaction-level revenue."""

    print("\n" + "=" * 70)
    print("REVENUE ANALYSIS")
    print("=" * 70)

    revenue = df["Quantity"] * df["UnitPrice"]

    print("\nRevenue statistics:")

    print(f"Count:  {revenue.count()}")
    print(f"Mean:   {revenue.mean():.2f}")
    print(f"Median: {revenue.median():.2f}")
    print(f"Min:    {revenue.min():.2f}")
    print(f"Max:    {revenue.max():.2f}")

    print(f"\nNegative revenue rows: {(revenue < 0).sum()}")
    print(f"Zero revenue rows: {(revenue == 0).sum()}")
    print(f"Positive revenue rows: {(revenue > 0).sum()}")

    print("\nLargest positive revenue transactions:")

    positive_revenue = (
        df.assign(Revenue=revenue)
        .sort_values("Revenue", ascending=False)
        .head(10)
    )

    print(
        positive_revenue[
            [
                "InvoiceNo",
                "StockCode",
                "Quantity",
                "UnitPrice",
                "Revenue",
            ]
        ].to_string(index=False)
    )

    print("\nLargest negative revenue transactions:")

    negative_revenue = (
        df.assign(Revenue=revenue)
        .sort_values("Revenue")
        .head(10)
    )

    print(
        negative_revenue[
            [
                "InvoiceNo",
                "StockCode",
                "Quantity",
                "UnitPrice",
                "Revenue",
            ]
        ].to_string(index=False)
    )

def profile_non_standard_transactions(df: pd.DataFrame) -> None:
    """Profile potentially non-standard transaction records."""

    print("\n" + "=" * 70)
    print("NON-STANDARD TRANSACTION ANALYSIS")
    print("=" * 70)

    special_codes = [
        "AMAZONFEE",
        "POST",
        "M",
        "B",
        "BANK CHARGES",
        "D",
        "S",
        "DOT",
        "CRUK",
    ]

    print("\nSelected non-standard StockCodes:")

    special_mask = df["StockCode"].isin(special_codes)

    special_rows = df.loc[special_mask].copy()

    print(f"Rows matching selected codes: {len(special_rows)}")

    if not special_rows.empty:
        print("\nCounts by StockCode:")

        print(
            special_rows["StockCode"]
            .value_counts()
            .to_string()
        )

        print("\nRevenue by StockCode:")

        special_rows["Revenue"] = (
            special_rows["Quantity"]
            * special_rows["UnitPrice"]
        )

        revenue_by_code = (
            special_rows
            .groupby("StockCode")["Revenue"]
            .agg(["count", "sum", "min", "max"])
            .sort_values("sum", ascending=False)
        )

        print(revenue_by_code.to_string())

    print("\nStockCodes with missing descriptions:")

    missing_description = df["Description"].isna()

    print(
        f"Rows with missing Description: "
        f"{missing_description.sum()}"
    )

    print(
        f"Unique StockCodes among missing descriptions: "
        f"{df.loc[missing_description, 'StockCode'].nunique()}"
    )

    print("\nRows with zero UnitPrice:")

    zero_price = df["UnitPrice"] == 0

    print(f"Zero-price rows: {zero_price.sum()}")

    print("\nRows with negative Quantity but non-cancelled InvoiceNo:")

    non_cancelled_negative = (
        (df["Quantity"] < 0)
        & (~df["InvoiceNo"].astype(str).str.startswith("C"))
    )

    print(
        f"Negative-quantity non-cancellation rows: "
        f"{non_cancelled_negative.sum()}"
    )

    if non_cancelled_negative.any():
        print("\nTop descriptions for these rows:")

        print(
            df.loc[non_cancelled_negative, "Description"]
            .value_counts(dropna=False)
            .head(15)
            .to_string()
        )

def profile_invoice_structure(df: pd.DataFrame) -> None:
    """Analyze invoice-level transaction structure."""

    print("\n" + "=" * 70)
    print("INVOICE STRUCTURE ANALYSIS")
    print("=" * 70)

    invoice_numbers = df["InvoiceNo"].astype(str)

    cancelled_mask = invoice_numbers.str.startswith("C")

    print("\nInvoice counts:")

    print(
        f"Unique invoices: "
        f"{df['InvoiceNo'].nunique()}"
    )

    print(
        f"Unique cancelled invoices: "
        f"{df.loc[cancelled_mask, 'InvoiceNo'].nunique()}"
    )

    print(
        f"Unique non-cancelled invoices: "
        f"{df.loc[~cancelled_mask, 'InvoiceNo'].nunique()}"
    )

    print("\nRows per invoice:")

    rows_per_invoice = df.groupby("InvoiceNo").size()

    print(
        f"Mean rows per invoice: "
        f"{rows_per_invoice.mean():.2f}"
    )

    print(
        f"Median rows per invoice: "
        f"{rows_per_invoice.median():.2f}"
    )

    print(
        f"Minimum rows per invoice: "
        f"{rows_per_invoice.min()}"
    )

    print(
        f"Maximum rows per invoice: "
        f"{rows_per_invoice.max()}"
    )

    print("\nInvoice row-count distribution:")

    print(
        rows_per_invoice
        .value_counts()
        .sort_index()
        .head(20)
        .to_string()
    )

    print("\nCancelled invoices by number of rows:")

    cancelled_rows_per_invoice = (
        df.loc[cancelled_mask]
        .groupby("InvoiceNo")
        .size()
    )

    print(
        cancelled_rows_per_invoice
        .describe()
        .to_string()
    )

    print("\nNon-cancelled invoices by number of rows:")

    normal_rows_per_invoice = (
        df.loc[~cancelled_mask]
        .groupby("InvoiceNo")
        .size()
    )

    print(
        normal_rows_per_invoice
        .describe()
        .to_string()
    )

    print("\nInvoices containing multiple StockCodes:")

    stock_codes_per_invoice = (
        df.groupby("InvoiceNo")["StockCode"]
        .nunique()
    )

    print(
        f"Invoices with more than one StockCode: "
        f"{(stock_codes_per_invoice > 1).sum()}"
    )

    print(
        f"Invoices with exactly one StockCode: "
        f"{(stock_codes_per_invoice == 1).sum()}"
    )

def profile_customer_activity(df: pd.DataFrame) -> None:
    """Analyze transaction activity for identified customers."""

    print("\n" + "=" * 70)
    print("CUSTOMER ACTIVITY ANALYSIS")
    print("=" * 70)

    customer_df = df[df["CustomerID"].notna()].copy()

    print("\nCustomer-level coverage:")

    print(
        f"Rows with CustomerID: "
        f"{len(customer_df)}"
    )

    print(
        f"Unique customers: "
        f"{customer_df['CustomerID'].nunique()}"
    )

    customer_transactions = (
        customer_df
        .groupby("CustomerID")
        .size()
    )

    print("\nTransactions per customer:")

    print(
        customer_transactions
        .describe()
        .to_string()
    )

    print("\nCustomer transaction distribution:")

    print(
        customer_transactions
        .value_counts()
        .sort_index()
        .head(20)
        .to_string()
    )

    print("\nTop 10 customers by transaction rows:")

    print(
        customer_transactions
        .sort_values(ascending=False)
        .head(10)
        .to_string()
    )

    customer_revenue = (
        customer_df
        .assign(
            Revenue=(
                customer_df["Quantity"]
                * customer_df["UnitPrice"]
            )
        )
        .groupby("CustomerID")["Revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\nTop 10 customers by total transaction revenue:")

    print(
        customer_revenue
        .head(10)
        .to_string()
    )

    print("\nCustomer-level revenue statistics:")

    print(
        customer_revenue
        .describe()
        .to_string()
    )

def profile_product_activity(df: pd.DataFrame) -> None:
    """Analyze transaction activity for products."""

    print("\n" + "=" * 70)
    print("PRODUCT ACTIVITY ANALYSIS")
    print("=" * 70)

    product_transactions = (
        df.groupby("StockCode")
        .size()
        .sort_values(ascending=False)
    )

    print("\nProduct-level coverage:")

    print(
        f"Unique StockCodes: "
        f"{df['StockCode'].nunique()}"
    )

    print("\nTransactions per product:")

    print(
        product_transactions
        .describe()
        .to_string()
    )

    print("\nTop 10 products by transaction rows:")

    print(
        product_transactions
        .head(10)
        .to_string()
    )

    product_summary = (
        df.assign(
            Revenue=(
                df["Quantity"]
                * df["UnitPrice"]
            )
        )
        .groupby("StockCode")
        .agg(
            transactions=("StockCode", "size"),
            total_quantity=("Quantity", "sum"),
            total_revenue=("Revenue", "sum"),
            average_unit_price=("UnitPrice", "mean"),
        )
    )

    print("\nTop 10 products by total quantity:")

    print(
        product_summary
        .sort_values("total_quantity", ascending=False)
        .head(10)
        ["total_quantity"]
        .to_string()
    )

    print("\nTop 10 products by total revenue:")

    print(
        product_summary
        .sort_values("total_revenue", ascending=False)
        .head(10)
        ["total_revenue"]
        .to_string()
    )

    print("\nProduct-level revenue statistics:")

    print(
        product_summary["total_revenue"]
        .describe()
        .to_string()
    )

def profile_country_activity(df: pd.DataFrame) -> None:
    """Analyze transaction activity by country."""

    print("\n" + "=" * 70)
    print("COUNTRY ACTIVITY ANALYSIS")
    print("=" * 70)

    country_summary = (
        df.assign(
            Revenue=(
                df["Quantity"]
                * df["UnitPrice"]
            )
        )
        .groupby("Country")
        .agg(
            transactions=("Country", "size"),
            unique_customers=("CustomerID", "nunique"),
            unique_products=("StockCode", "nunique"),
            total_quantity=("Quantity", "sum"),
            total_revenue=("Revenue", "sum"),
        )
        .sort_values("transactions", ascending=False)
    )

    print("\nCountry-level coverage:")

    print(
        f"Unique countries: "
        f"{df['Country'].nunique()}"
    )

    print("\nTop 10 countries by transaction rows:")

    print(
        country_summary
        .head(10)
        .to_string()
    )

    print("\nTop 10 countries by total revenue:")

    print(
        country_summary
        .sort_values("total_revenue", ascending=False)
        .head(10)
        .to_string()
    )

    print("\nCountry-level revenue statistics:")

    print(
        country_summary["total_revenue"]
        .describe()
        .to_string()
    )

    print("\nCountries with negative total revenue:")

    negative_country_revenue = (
        country_summary[
            country_summary["total_revenue"] < 0
        ]
    )

    print(
        negative_country_revenue
        .to_string()
    )

def profile_cleaning_impact(df: pd.DataFrame) -> None:
    """Estimate the impact of proposed cleaning rules."""

    print("\n" + "=" * 70)
    print("CLEANING IMPACT ANALYSIS")
    print("=" * 70)

    invoice_numbers = df["InvoiceNo"].astype(str)

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

    rules = {
        "Missing CustomerID": df["CustomerID"].isna(),
        "Cancelled Invoice": invoice_numbers.str.startswith("C"),
        "Non-positive Quantity": df["Quantity"] <= 0,
        "Non-positive UnitPrice": df["UnitPrice"] <= 0,
        "Non-standard StockCode": df["StockCode"].isin(
            non_standard_stock_codes
        ),
        "Invalid InvoiceDate": pd.to_datetime(
            df["InvoiceDate"],
            errors="coerce",
        ).isna(),
    }

    print("\nRows affected by individual rules:")

    for rule_name, mask in rules.items():
        count = mask.sum()
        percentage = count / len(df) * 100

        print(
            f"{rule_name}: "
            f"{count} rows ({percentage:.2f}%)"
        )

    valid_sale_mask = (
        df["CustomerID"].notna()
        & ~invoice_numbers.str.startswith("C")
        & (df["Quantity"] > 0)
        & (df["UnitPrice"] > 0)
        & ~df["StockCode"].isin(non_standard_stock_codes)
        & pd.to_datetime(
            df["InvoiceDate"],
            errors="coerce",
        ).notna()
    )

    valid_count = valid_sale_mask.sum()
    removed_count = len(df) - valid_count

    print("\nCombined proposed valid-sale dataset:")

    print(
        f"Valid sale rows: "
        f"{valid_count}"
    )

    print(
        f"Rows excluded: "
        f"{removed_count}"
    )

    print(
        f"Percentage retained: "
        f"{valid_count / len(df) * 100:.2f}%"
    )

    print(
        f"Percentage excluded: "
        f"{removed_count / len(df) * 100:.2f}%"
    )

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

    # Convert InvoiceDate safely.
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
        & ~cleaned_df["StockCode"].isin(
            non_standard_stock_codes
        )
    )

    cleaned_df = cleaned_df.loc[valid_sale_mask].copy()

    # Calculate transaction-level revenue.
    cleaned_df["Revenue"] = (
        cleaned_df["Quantity"]
        * cleaned_df["UnitPrice"]
    )

    return cleaned_df

def save_cleaned_sales(df: pd.DataFrame) -> None:
    """Save the cleaned sales dataset."""

    output_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "cleaned_sales.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("CLEANED DATASET")
    print("=" * 70)

    print(f"\nOutput path: {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nRevenue statistics:")

    print(
        df["Revenue"]
        .describe()
        .to_string()
    )

def validate_cleaned_sales(df: pd.DataFrame) -> None:
    """Validate the cleaned sales dataset."""

    print("\n" + "=" * 70)
    print("CLEANED DATA VALIDATION")
    print("=" * 70)

    invoice_numbers = df["InvoiceNo"].astype(str)

    duplicate_rows = df.duplicated().sum()
    missing_customer_ids = df["CustomerID"].isna().sum()
    cancelled_invoices = invoice_numbers.str.startswith("C").sum()
    non_positive_quantity = (df["Quantity"] <= 0).sum()
    non_positive_price = (df["UnitPrice"] <= 0).sum()
    invalid_dates = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce",
    ).isna()

    revenue_check = (
        df["Revenue"]
        == df["Quantity"] * df["UnitPrice"]
    ).all()

    print("\nValidation checks:")

    print(f"Duplicate rows: {duplicate_rows}")
    print(f"Missing CustomerID: {missing_customer_ids}")
    print(f"Cancelled invoices: {cancelled_invoices}")
    print(f"Non-positive Quantity: {non_positive_quantity}")
    print(f"Non-positive UnitPrice: {non_positive_price}")
    print(f"Invalid InvoiceDate: {invalid_dates.sum()}")
    print(f"Revenue calculation correct: {revenue_check}")

    print("\nDate coverage:")

    print(
        f"Earliest InvoiceDate: "
        f"{df['InvoiceDate'].min()}"
    )

    print(
        f"Latest InvoiceDate: "
        f"{df['InvoiceDate'].max()}"
    )

    print("\nCustomer coverage:")

    print(
        f"Unique customers: "
        f"{df['CustomerID'].nunique()}"
    )

    print("\nProduct coverage:")

    print(
        f"Unique StockCodes: "
        f"{df['StockCode'].nunique()}"
    )

    print("\nCountry coverage:")

    print(
        f"Unique countries: "
        f"{df['Country'].nunique()}"
    )

    print("\nRevenue totals:")

    print(
        f"Total revenue: "
        f"{df['Revenue'].sum():.2f}"
    )

    print(
        f"Average transaction revenue: "
        f"{df['Revenue'].mean():.2f}"
    )

    print(
        f"Median transaction revenue: "
        f"{df['Revenue'].median():.2f}"
    )

    validation_passed = (
        duplicate_rows == 0
        and missing_customer_ids == 0
        and cancelled_invoices == 0
        and non_positive_quantity == 0
        and non_positive_price == 0
        and invalid_dates.sum() == 0
        and revenue_check
    )

    print("\nValidation status:")

    if validation_passed:
        print("PASSED - cleaned dataset satisfies all validation rules.")
    else:
        print("FAILED - one or more validation rules were violated.")

def analyze_monthly_sales(df: pd.DataFrame) -> None:
    """Analyze monthly sales revenue trends."""

    print("\n" + "=" * 70)
    print("MONTHLY SALES ANALYSIS")
    print("=" * 70)

    sales_df = df.copy()

    sales_df["InvoiceDate"] = pd.to_datetime(
        sales_df["InvoiceDate"],
        errors="coerce",
    )

    sales_df["YearMonth"] = (
        sales_df["InvoiceDate"]
        .dt.to_period("M")
    )

    monthly_sales = (
        sales_df
        .groupby("YearMonth")
        .agg(
            Revenue=("Revenue", "sum"),
            Transactions=("InvoiceNo", "nunique"),
            UnitsSold=("Quantity", "sum"),
            Customers=("CustomerID", "nunique"),
        )
        .reset_index()
    )

    print("\nMonthly sales:")
    print(monthly_sales.to_string(index=False))

    print("\nMonthly revenue statistics:")
    print(
        monthly_sales["Revenue"].describe().to_string()
    )

    highest_month = monthly_sales.loc[
        monthly_sales["Revenue"].idxmax()
    ]

    lowest_month = monthly_sales.loc[
        monthly_sales["Revenue"].idxmin()
    ]

    print("\nHighest revenue month:")
    print(
        f"{highest_month['YearMonth']}: "
        f"{highest_month['Revenue']:.2f}"
    )

    print("\nLowest revenue month:")
    print(
        f"{lowest_month['YearMonth']}: "
        f"{lowest_month['Revenue']:.2f}"
    )

    print("\nNote:")
    print(
        "December 2011 is incomplete because the dataset ends "
        "on 2011-12-09."
    )

def analyze_daily_sales(df: pd.DataFrame) -> None:
    """Analyze daily sales revenue and transaction trends."""

    print("\n" + "=" * 70)
    print("DAILY SALES ANALYSIS")
    print("=" * 70)

    sales_df = df.copy()

    sales_df["InvoiceDate"] = pd.to_datetime(
        sales_df["InvoiceDate"],
        errors="coerce",
    )

    sales_df["Date"] = sales_df["InvoiceDate"].dt.date

    daily_sales = (
        sales_df
        .groupby("Date")
        .agg(
            Revenue=("Revenue", "sum"),
            Transactions=("InvoiceNo", "nunique"),
            UnitsSold=("Quantity", "sum"),
            Customers=("CustomerID", "nunique"),
        )
        .reset_index()
    )

    print("\nDaily sales summary:")
    print(daily_sales.head(10).to_string(index=False))

    print("\nDaily revenue statistics:")
    print(
        daily_sales["Revenue"].describe().to_string()
    )

    highest_day = daily_sales.loc[
        daily_sales["Revenue"].idxmax()
    ]

    lowest_day = daily_sales.loc[
        daily_sales["Revenue"].idxmin()
    ]

    print("\nHighest revenue day:")
    print(
        f"{highest_day['Date']}: "
        f"{highest_day['Revenue']:.2f}"
    )

    print("\nLowest revenue day:")
    print(
        f"{lowest_day['Date']}: "
        f"{lowest_day['Revenue']:.2f}"
    )

    sales_df["DayOfWeek"] = (
        sales_df["InvoiceDate"]
        .dt.day_name()
    )

    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    weekday_sales = (
        sales_df
        .groupby("DayOfWeek")
        .agg(
            Revenue=("Revenue", "sum"),
            Transactions=("InvoiceNo", "nunique"),
            UnitsSold=("Quantity", "sum"),
        )
        .reindex(weekday_order)
        .reset_index()
    )

    print("\nSales by day of week:")
    print(
        weekday_sales.to_string(index=False)
    )

    print("\nAverage daily revenue by day of week:")

    daily_weekday_sales = (
        daily_sales.assign(
            DayOfWeek=pd.to_datetime(
                daily_sales["Date"]
            ).dt.day_name()
        )
        .groupby("DayOfWeek")["Revenue"]
        .mean()
        .reindex(weekday_order)
    )

    print(
        daily_weekday_sales
        .to_string()
    )

    print("\nNote:")
    print(
        "The dataset contains 305 unique transaction dates, "
        "and the final month (December 2011) is incomplete."
    )

def analyze_daily_revenue_outliers(df: pd.DataFrame) -> None:
    """Identify unusually large daily revenue observations."""

    print("\n" + "=" * 70)
    print("DAILY REVENUE OUTLIER ANALYSIS")
    print("=" * 70)

    sales_df = df.copy()

    sales_df["InvoiceDate"] = pd.to_datetime(
        sales_df["InvoiceDate"],
        errors="coerce",
    )

    sales_df["Date"] = sales_df["InvoiceDate"].dt.date

    daily_sales = (
        sales_df
        .groupby("Date")
        .agg(
            Revenue=("Revenue", "sum"),
            Transactions=("InvoiceNo", "nunique"),
            UnitsSold=("Quantity", "sum"),
            Customers=("CustomerID", "nunique"),
        )
        .reset_index()
    )

    q1 = daily_sales["Revenue"].quantile(0.25)
    q3 = daily_sales["Revenue"].quantile(0.75)
    iqr = q3 - q1

    upper_bound = q3 + (1.5 * iqr)

    outliers = daily_sales[
        daily_sales["Revenue"] > upper_bound
    ].copy()

    outliers = outliers.sort_values(
        "Revenue",
        ascending=False,
    )

    print("\nOutlier threshold:")
    print(f"Q1: {q1:.2f}")
    print(f"Q3: {q3:.2f}")
    print(f"IQR: {iqr:.2f}")
    print(f"Upper bound: {upper_bound:.2f}")

    print("\nNumber of high-revenue outlier days:")
    print(len(outliers))

    print("\nHigh-revenue outlier days:")
    if len(outliers) > 0:
        print(
            outliers.to_string(index=False)
        )
    else:
        print("No high-revenue outlier days detected.")

    print("\nTop 10 revenue days:")
    print(
        daily_sales
        .sort_values("Revenue", ascending=False)
        .head(10)
        .to_string(index=False)
    )

def investigate_revenue_spikes(df: pd.DataFrame) -> None:
    """Investigate transactions contributing to extreme revenue days."""

    print("\n" + "=" * 70)
    print("REVENUE SPIKE INVESTIGATION")
    print("=" * 70)

    sales_df = df.copy()

    sales_df["InvoiceDate"] = pd.to_datetime(
        sales_df["InvoiceDate"],
        errors="coerce",
    )

    sales_df["Date"] = sales_df["InvoiceDate"].dt.date

    daily_revenue = (
        sales_df
        .groupby("Date")["Revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    top_spike_dates = daily_revenue.head(3).index

    print("\nTop 3 revenue spike dates:")
    for date in top_spike_dates:
        print(
            f"  - {date}: "
            f"{daily_revenue.loc[date]:.2f}"
        )

    for date in top_spike_dates:
        print("\n" + "-" * 70)
        print(f"SPIKE DATE: {date}")
        print("-" * 70)

        day_df = sales_df[
            sales_df["Date"] == date
        ].copy()

        print("\nDaily totals:")
        print(
            f"Revenue: {day_df['Revenue'].sum():.2f}"
        )
        print(
            f"Transactions: "
            f"{day_df['InvoiceNo'].nunique()}"
        )
        print(
            f"Units sold: "
            f"{day_df['Quantity'].sum()}"
        )
        print(
            f"Customers: "
            f"{day_df['CustomerID'].nunique()}"
        )

        print("\nTop 10 transaction lines by revenue:")

        top_transactions = (
            day_df[
                [
                    "InvoiceNo",
                    "StockCode",
                    "Description",
                    "Quantity",
                    "UnitPrice",
                    "CustomerID",
                    "Revenue",
                ]
            ]
            .sort_values(
                "Revenue",
                ascending=False,
            )
            .head(10)
        )

        print(
            top_transactions.to_string(
                index=False
            )
        )

        print("\nTop 10 products by revenue:")

        top_products = (
            day_df
            .groupby(
                ["StockCode", "Description"],
                dropna=False,
            )
            .agg(
                Revenue=("Revenue", "sum"),
                Quantity=("Quantity", "sum"),
                Transactions=("InvoiceNo", "nunique"),
            )
            .reset_index()
            .sort_values(
                "Revenue",
                ascending=False,
            )
            .head(10)
        )

        print(
            top_products.to_string(
                index=False
            )
        )

        print("\nTop 10 invoices by revenue:")

        top_invoices = (
            day_df
            .groupby("InvoiceNo")
            .agg(
                Revenue=("Revenue", "sum"),
                Quantity=("Quantity", "sum"),
                Products=("StockCode", "nunique"),
                CustomerID=("CustomerID", "first"),
            )
            .reset_index()
            .sort_values(
                "Revenue",
                ascending=False,
            )
            .head(10)
        )

        print(
            top_invoices.to_string(
                index=False
            )
        )

def analyze_customer_revenue_concentration(
    df: pd.DataFrame,
) -> None:
    """Analyze customer revenue concentration."""

    print("\n" + "=" * 70)
    print("CUSTOMER REVENUE CONCENTRATION")
    print("=" * 70)

    customer_revenue = (
        df
        .groupby("CustomerID")
        .agg(
            Revenue=("Revenue", "sum"),
            Transactions=("InvoiceNo", "nunique"),
            UnitsSold=("Quantity", "sum"),
        )
        .reset_index()
        .sort_values(
            "Revenue",
            ascending=False,
        )
    )

    total_revenue = customer_revenue["Revenue"].sum()

    customer_revenue["RevenueShare"] = (
        customer_revenue["Revenue"]
        / total_revenue
    )

    customer_revenue["CumulativeRevenueShare"] = (
        customer_revenue["RevenueShare"]
        .cumsum()
    )

    customer_count = len(customer_revenue)

    print("\nCustomer count:")
    print(customer_count)

    print("\nTotal customer revenue:")
    print(f"{total_revenue:.2f}")

    print("\nTop 10 customers by revenue:")

    print(
        customer_revenue
        .head(10)
        .to_string(index=False)
    )

    print("\nRevenue concentration:")

    for percentage in [0.01, 0.05, 0.10, 0.20]:

        count = max(
            1,
            int(customer_count * percentage),
        )

        revenue_share = (
            customer_revenue
            .head(count)["Revenue"]
            .sum()
            / total_revenue
        )

        print(
            f"Top {percentage:.0%} of customers "
            f"({count} customers): "
            f"{revenue_share:.2%} of revenue"
        )

    revenue_80_position = (
        customer_revenue[
            customer_revenue["CumulativeRevenueShare"] >= 0.80
        ]
        .index[0]
        + 1
    )

    revenue_80_percentage = (
        revenue_80_position / customer_count
    )

    print("\nCustomers required to reach 80% of revenue:")

    print(
        f"{revenue_80_position} customers "
        f"({revenue_80_percentage:.2%} of customers)"
    )

    print("\nCustomer revenue statistics:")

    print(
        customer_revenue["Revenue"]
        .describe()
        .to_string()
    )

def analyze_product_revenue_concentration(
    df: pd.DataFrame,
) -> None:
    """Analyze product revenue concentration."""

    print("\n" + "=" * 70)
    print("PRODUCT REVENUE CONCENTRATION")
    print("=" * 70)

    product_revenue = (
        df
        .groupby(
            ["StockCode", "Description"],
            dropna=False,
        )
        .agg(
            Revenue=("Revenue", "sum"),
            Transactions=("InvoiceNo", "nunique"),
            UnitsSold=("Quantity", "sum"),
        )
        .reset_index()
        .sort_values(
            "Revenue",
            ascending=False,
        )
    )

    total_revenue = product_revenue["Revenue"].sum()

    product_revenue["RevenueShare"] = (
        product_revenue["Revenue"]
        / total_revenue
    )

    product_revenue["CumulativeRevenueShare"] = (
        product_revenue["RevenueShare"]
        .cumsum()
    )

    product_count = len(product_revenue)

    print("\nProduct count:")
    print(product_count)

    print("\nTotal product revenue:")
    print(f"{total_revenue:.2f}")

    print("\nTop 10 products by revenue:")

    print(
        product_revenue
        .head(10)
        .to_string(index=False)
    )

    print("\nRevenue concentration:")

    for percentage in [0.01, 0.05, 0.10, 0.20]:

        count = max(
            1,
            int(product_count * percentage),
        )

        revenue_share = (
            product_revenue
            .head(count)["Revenue"]
            .sum()
            / total_revenue
        )

        print(
            f"Top {percentage:.0%} of products "
            f"({count} products): "
            f"{revenue_share:.2%} of revenue"
        )

    revenue_80_position = (
        product_revenue[
            product_revenue["CumulativeRevenueShare"] >= 0.80
        ]
        .index[0]
        + 1
    )

    revenue_80_percentage = (
        revenue_80_position / product_count
    )

    print("\nProducts required to reach 80% of revenue:")

    print(
        f"{revenue_80_position} products "
        f"({revenue_80_percentage:.2%} of products)"
    )

    print("\nProduct revenue statistics:")

    print(
        product_revenue["Revenue"]
        .describe()
        .to_string()
    )

def analyze_country_revenue(df: pd.DataFrame) -> None:
    """Analyze revenue and sales activity by country."""

    print("\n" + "=" * 70)
    print("COUNTRY REVENUE ANALYSIS")
    print("=" * 70)

    country_sales = (
        df
        .groupby("Country")
        .agg(
            Revenue=("Revenue", "sum"),
            Customers=("CustomerID", "nunique"),
            Transactions=("InvoiceNo", "nunique"),
            UnitsSold=("Quantity", "sum"),
        )
        .reset_index()
        .sort_values(
            "Revenue",
            ascending=False,
        )
    )

    total_revenue = country_sales["Revenue"].sum()

    country_sales["RevenueShare"] = (
        country_sales["Revenue"]
        / total_revenue
    )

    country_sales["AverageRevenuePerTransaction"] = (
        country_sales["Revenue"]
        / country_sales["Transactions"]
    )

    print("\nCountry count:")
    print(len(country_sales))

    print("\nTotal revenue:")
    print(f"{total_revenue:.2f}")

    print("\nCountry revenue summary:")

    print(
        country_sales.to_string(index=False)
    )

    print("\nTop 10 countries by revenue:")

    print(
        country_sales
        .head(10)
        .to_string(index=False)
    )

    print("\nRevenue concentration by country:")

    for percentage in [0.10, 0.25, 0.50]:

        count = max(
            1,
            int(len(country_sales) * percentage),
        )

        revenue_share = (
            country_sales
            .head(count)["Revenue"]
            .sum()
            / total_revenue
        )

        print(
            f"Top {percentage:.0%} of countries "
            f"({count} countries): "
            f"{revenue_share:.2%} of revenue"
        )

def analyze_transaction_value(df: pd.DataFrame) -> None:
    """Analyze invoice-level transaction value and basket size."""

    print("\n" + "=" * 70)
    print("TRANSACTION VALUE AND BASKET SIZE ANALYSIS")
    print("=" * 70)

    invoice_sales = (
        df
        .groupby("InvoiceNo")
        .agg(
            Revenue=("Revenue", "sum"),
            UnitsSold=("Quantity", "sum"),
            Products=("StockCode", "nunique"),
            CustomerID=("CustomerID", "first"),
            Country=("Country", "first"),
        )
        .reset_index()
    )

    invoice_sales["RevenuePerProduct"] = (
        invoice_sales["Revenue"]
        / invoice_sales["Products"]
    )

    print("\nInvoice count:")
    print(len(invoice_sales))

    print("\nInvoice-level revenue statistics:")
    print(
        invoice_sales["Revenue"]
        .describe()
        .to_string()
    )

    print("\nUnits per invoice statistics:")
    print(
        invoice_sales["UnitsSold"]
        .describe()
        .to_string()
    )

    print("\nProducts per invoice statistics:")
    print(
        invoice_sales["Products"]
        .describe()
        .to_string()
    )

    print("\nTop 10 invoices by revenue:")

    print(
        invoice_sales
        .sort_values(
            "Revenue",
            ascending=False,
        )
        .head(10)
        .to_string(index=False)
    )

    print("\nTop 10 invoices by units sold:")

    print(
        invoice_sales
        .sort_values(
            "UnitsSold",
            ascending=False,
        )
        .head(10)
        .to_string(index=False)
    )

    print("\nTop 10 invoices by product count:")

    print(
        invoice_sales
        .sort_values(
            "Products",
            ascending=False,
        )
        .head(10)
        .to_string(index=False)
    )

    print("\nInvoices with revenue above 10,000:")

    high_value_count = (
        invoice_sales["Revenue"] > 10000
    ).sum()

    print(high_value_count)

    print("\nInvoices with revenue above 5,000:")

    medium_high_value_count = (
        invoice_sales["Revenue"] > 5000
    ).sum()

    print(medium_high_value_count)

def analyze_customer_purchase_frequency(
    df: pd.DataFrame,
) -> None:
    """Analyze customer purchase frequency."""

    print("\n" + "=" * 70)
    print("CUSTOMER PURCHASE FREQUENCY ANALYSIS")
    print("=" * 70)

    customer_activity = (
        df
        .groupby("CustomerID")
        .agg(
            Transactions=("InvoiceNo", "nunique"),
            Revenue=("Revenue", "sum"),
            UnitsSold=("Quantity", "sum"),
            FirstPurchase=("InvoiceDate", "min"),
            LastPurchase=("InvoiceDate", "max"),
        )
        .reset_index()
    )

    customer_activity["PurchaseSpanDays"] = (
        pd.to_datetime(
            customer_activity["LastPurchase"]
        )
        - pd.to_datetime(
            customer_activity["FirstPurchase"]
        )
    ).dt.days

    customer_count = len(customer_activity)

    one_time_customers = (
        customer_activity["Transactions"] == 1
    ).sum()

    repeat_customers = (
        customer_activity["Transactions"] > 1
    ).sum()

    print("\nCustomer count:")
    print(customer_count)

    print("\nOne-time customers:")
    print(
        f"{one_time_customers} "
        f"({one_time_customers / customer_count:.2%})"
    )

    print("\nRepeat customers:")
    print(
        f"{repeat_customers} "
        f"({repeat_customers / customer_count:.2%})"
    )

    print("\nTransactions per customer statistics:")
    print(
        customer_activity["Transactions"]
        .describe()
        .to_string()
    )

    print("\nPurchase span in days statistics:")
    print(
        customer_activity["PurchaseSpanDays"]
        .describe()
        .to_string()
    )

    print("\nCustomer frequency distribution:")

    frequency_distribution = (
        customer_activity["Transactions"]
        .value_counts()
        .sort_index()
    )

    print(
        frequency_distribution
        .head(20)
        .to_string()
    )

    one_time_revenue = customer_activity.loc[
        customer_activity["Transactions"] == 1,
        "Revenue",
    ].sum()

    repeat_customer_revenue = customer_activity.loc[
        customer_activity["Transactions"] > 1,
        "Revenue",
    ].sum()

    total_revenue = customer_activity["Revenue"].sum()

    print("\nRevenue by customer type:")

    print(
        f"One-time customer revenue: "
        f"{one_time_revenue:.2f}"
    )

    print(
        f"Repeat customer revenue: "
        f"{repeat_customer_revenue:.2f}"
    )

    print(
        f"One-time customer revenue share: "
        f"{one_time_revenue / total_revenue:.2%}"
    )

    print(
        f"Repeat customer revenue share: "
        f"{repeat_customer_revenue / total_revenue:.2%}"
    )

    print("\nTop 10 customers by transaction count:")

    print(
        customer_activity
        .sort_values(
            "Transactions",
            ascending=False,
        )
        .head(10)
        .to_string(index=False)
    )

def analyze_customer_recency_and_value(
    df: pd.DataFrame,
) -> None:
    """Analyze customer recency and observed customer value."""

    print("\n" + "=" * 70)
    print("CUSTOMER RECENCY AND VALUE ANALYSIS")
    print("=" * 70)

    sales_df = df.copy()

    sales_df["InvoiceDate"] = pd.to_datetime(
        sales_df["InvoiceDate"],
        errors="coerce",
    )

    analysis_date = sales_df["InvoiceDate"].max()

    customer_value = (
        sales_df
        .groupby("CustomerID")
        .agg(
            FirstPurchase=("InvoiceDate", "min"),
            LastPurchase=("InvoiceDate", "max"),
            Transactions=("InvoiceNo", "nunique"),
            UnitsSold=("Quantity", "sum"),
            Revenue=("Revenue", "sum"),
        )
        .reset_index()
    )

    customer_value["RecencyDays"] = (
        analysis_date
        - customer_value["LastPurchase"]
    ).dt.days

    customer_value["ObservedLifespanDays"] = (
        customer_value["LastPurchase"]
        - customer_value["FirstPurchase"]
    ).dt.days

    customer_value["AverageRevenuePerTransaction"] = (
        customer_value["Revenue"]
        / customer_value["Transactions"]
    )

    customer_value["AverageRevenuePerUnit"] = (
        customer_value["Revenue"]
        / customer_value["UnitsSold"]
    )

    active_days = (
        customer_value["ObservedLifespanDays"]
        .clip(lower=1)
    )

    customer_value["RevenuePerObservedDay"] = (
        customer_value["Revenue"]
        / active_days
    )

    print("\nAnalysis date:")
    print(analysis_date)

    print("\nCustomer value statistics:")

    print(
        customer_value[
            [
                "Revenue",
                "Transactions",
                "UnitsSold",
                "RecencyDays",
                "ObservedLifespanDays",
                "AverageRevenuePerTransaction",
                "AverageRevenuePerUnit",
                "RevenuePerObservedDay",
            ]
        ]
        .describe()
        .to_string()
    )

    print("\nTop 10 customers by observed revenue:")

    print(
        customer_value
        .sort_values(
            "Revenue",
            ascending=False,
        )
        .head(10)
        .to_string(index=False)
    )

    print("\nTop 10 most recent customers:")

    print(
        customer_value
        .sort_values(
            "RecencyDays",
            ascending=True,
        )
        .head(10)
        .to_string(index=False)
    )

    print("\nCustomers with longest observed lifespan:")

    print(
        customer_value
        .sort_values(
            "ObservedLifespanDays",
            ascending=False,
        )
        .head(10)
        .to_string(index=False)
    )

    print("\nRecency distribution:")

    print(
        customer_value["RecencyDays"]
        .describe()
        .to_string()
    )

    print("\nCustomers by recency bucket:")

    recency_bins = [
        -1,
        30,
        90,
        180,
        365,
        float("inf"),
    ]

    recency_labels = [
        "0-30 days",
        "31-90 days",
        "91-180 days",
        "181-365 days",
        "366+ days",
    ]

    recency_distribution = pd.cut(
        customer_value["RecencyDays"],
        bins=recency_bins,
        labels=recency_labels,
    ).value_counts().sort_index()

    print(
        recency_distribution.to_string()
    )

def analyze_product_demand(df: pd.DataFrame) -> None:
    """Analyze product demand and sales volume."""

    print("\n" + "=" * 70)
    print("PRODUCT DEMAND AND SALES VOLUME ANALYSIS")
    print("=" * 70)

    product_demand = (
        df
        .groupby(
            ["StockCode", "Description"],
            dropna=False,
        )
        .agg(
            UnitsSold=("Quantity", "sum"),
            Transactions=("InvoiceNo", "nunique"),
            Revenue=("Revenue", "sum"),
            AverageUnitPrice=("UnitPrice", "mean"),
        )
        .reset_index()
    )

    product_demand["RevenuePerTransaction"] = (
        product_demand["Revenue"]
        / product_demand["Transactions"]
    )

    total_units = product_demand["UnitsSold"].sum()
    total_revenue = product_demand["Revenue"].sum()

    product_demand["UnitShare"] = (
        product_demand["UnitsSold"]
        / total_units
    )

    product_demand["RevenueShare"] = (
        product_demand["Revenue"]
        / total_revenue
    )

    print("\nProduct count:")
    print(len(product_demand))

    print("\nTotal units sold:")
    print(total_units)

    print("\nTotal revenue:")
    print(f"{total_revenue:.2f}")

    print("\nTop 10 products by units sold:")

    print(
        product_demand
        .sort_values(
            "UnitsSold",
            ascending=False,
        )
        .head(10)
        .to_string(index=False)
    )

    print("\nTop 10 products by transaction count:")

    print(
        product_demand
        .sort_values(
            "Transactions",
            ascending=False,
        )
        .head(10)
        .to_string(index=False)
    )

    print("\nTop 10 products by revenue:")

    print(
        product_demand
        .sort_values(
            "Revenue",
            ascending=False,
        )
        .head(10)
        .to_string(index=False)
    )

    print("\nHighest average unit prices:")

    print(
        product_demand
        .sort_values(
            "AverageUnitPrice",
            ascending=False,
        )
        .head(10)
        .to_string(index=False)
    )

    print("\nTop 10 products by units sold - unit share:")

    top_units = (
        product_demand
        .sort_values(
            "UnitsSold",
            ascending=False,
        )
        .head(10)
    )

    print(
        top_units[
            [
                "StockCode",
                "Description",
                "UnitsSold",
                "UnitShare",
            ]
        ]
        .to_string(index=False)
    )

def analyze_product_demand_concentration(df: pd.DataFrame) -> None:
    """Analyze concentration of product demand and revenue."""

    print("\n" + "=" * 70)
    print("PRODUCT DEMAND CONCENTRATION AND PARETO ANALYSIS")
    print("=" * 70)

    product_summary = (
        df.groupby(
            ["StockCode", "Description"],
            dropna=False,
        )
        .agg(
            UnitsSold=("Quantity", "sum"),
            Revenue=("Revenue", "sum"),
            Transactions=("InvoiceNo", "nunique"),
        )
        .reset_index()
    )

    total_products = len(product_summary)
    total_units = product_summary["UnitsSold"].sum()
    total_revenue = product_summary["Revenue"].sum()

    print("\nTotal products:")
    print(total_products)

    print("\nTotal units sold:")
    print(total_units)

    print("\nTotal revenue:")
    print(f"{total_revenue:.2f}")

    # Unit concentration
    units_ranked = product_summary.sort_values(
        "UnitsSold",
        ascending=False,
    ).reset_index(drop=True)

    units_ranked["CumulativeUnits"] = (
        units_ranked["UnitsSold"].cumsum()
    )

    units_ranked["CumulativeUnitShare"] = (
        units_ranked["CumulativeUnits"]
        / total_units
    )

    # Revenue concentration
    revenue_ranked = product_summary.sort_values(
        "Revenue",
        ascending=False,
    ).reset_index(drop=True)

    revenue_ranked["CumulativeRevenue"] = (
        revenue_ranked["Revenue"].cumsum()
    )

    revenue_ranked["CumulativeRevenueShare"] = (
        revenue_ranked["CumulativeRevenue"]
        / total_revenue
    )

    def products_required_for_share(
        ranked_df: pd.DataFrame,
        share_column: str,
        target_share: float,
    ) -> int:
        """Return number of products required to reach a target share."""

        matching_indices = ranked_df.index[
            ranked_df[share_column] >= target_share
        ]

        if len(matching_indices) == 0:
            return len(ranked_df)

        return int(matching_indices[0] + 1)

    print("\nProducts required to reach unit-volume thresholds:")

    for target in [0.50, 0.80, 0.90]:
        count = products_required_for_share(
            units_ranked,
            "CumulativeUnitShare",
            target,
        )

        percentage = count / total_products

        print(
            f"{target:.0%} of units: "
            f"{count} products "
            f"({percentage:.2%} of products)"
        )

    print("\nProducts required to reach revenue thresholds:")

    for target in [0.50, 0.80, 0.90]:
        count = products_required_for_share(
            revenue_ranked,
            "CumulativeRevenueShare",
            target,
        )

        percentage = count / total_products

        print(
            f"{target:.0%} of revenue: "
            f"{count} products "
            f"({percentage:.2%} of products)"
        )

    print("\nTop 20 products by cumulative unit share:")

    print(
        units_ranked[
            [
                "StockCode",
                "Description",
                "UnitsSold",
                "CumulativeUnitShare",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\nTop 20 products by cumulative revenue share:")

    print(
        revenue_ranked[
            [
                "StockCode",
                "Description",
                "Revenue",
                "CumulativeRevenueShare",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

def analyze_product_demand_revenue_relationship(df: pd.DataFrame) -> None:
    """Analyze the relationship between product demand and revenue."""

    print("\n" + "=" * 70)
    print("PRODUCT DEMAND VS REVENUE RELATIONSHIP")
    print("=" * 70)

    product_summary = (
        df.groupby(
            ["StockCode", "Description"],
            dropna=False,
        )
        .agg(
            UnitsSold=("Quantity", "sum"),
            Revenue=("Revenue", "sum"),
            Transactions=("InvoiceNo", "nunique"),
        )
        .reset_index()
    )

    product_summary["RevenuePerUnit"] = (
        product_summary["Revenue"]
        / product_summary["UnitsSold"]
    )

    correlation = product_summary[
        ["UnitsSold", "Revenue"]
    ].corr().loc["UnitsSold", "Revenue"]

    print("\nProduct count:")
    print(len(product_summary))

    print("\nCorrelation between units sold and revenue:")
    print(f"{correlation:.6f}")

    print("\nTop 10 products by units sold with revenue:")

    print(
        product_summary
        .sort_values("UnitsSold", ascending=False)
        .head(10)
        .to_string(index=False)
    )

    print("\nTop 10 products by revenue with units sold:")

    print(
        product_summary
        .sort_values("Revenue", ascending=False)
        .head(10)
        .to_string(index=False)
    )

    units_top_10 = set(
        product_summary
        .nlargest(10, "UnitsSold")["StockCode"]
    )

    revenue_top_10 = set(
        product_summary
        .nlargest(10, "Revenue")["StockCode"]
    )

    overlap = units_top_10.intersection(revenue_top_10)

    print("\nTop-10 ranking overlap:")
    print(len(overlap))

    print("\nProducts appearing in both top-10 groups:")
    print(sorted(overlap, key=str))

    print("\nTop 10 products by revenue per unit:")

    print(
        product_summary
        .sort_values("RevenuePerUnit", ascending=False)
        .head(10)
        .to_string(index=False)
    )

    print("\nTop 10 high-volume products by revenue per unit:")

    high_volume_products = product_summary[
        product_summary["UnitsSold"] >= product_summary["UnitsSold"].quantile(0.90)
    ]

    print(
        high_volume_products
        .sort_values("RevenuePerUnit", ascending=False)
        .head(10)
        .to_string(index=False)
    )

def analyze_customer_frequency_revenue_relationship(
    df: pd.DataFrame,
) -> None:
    """Analyze the relationship between customer purchase frequency and revenue."""

    print("\n" + "=" * 70)
    print("CUSTOMER PURCHASE FREQUENCY VS REVENUE RELATIONSHIP")
    print("=" * 70)

    customer_summary = (
        df.groupby("CustomerID")
        .agg(
            Revenue=("Revenue", "sum"),
            Transactions=("InvoiceNo", "nunique"),
            UnitsPurchased=("Quantity", "sum"),
        )
        .reset_index()
    )

    customer_summary["AverageRevenuePerTransaction"] = (
        customer_summary["Revenue"]
        / customer_summary["Transactions"]
    )

    customer_summary["RevenuePerUnit"] = (
        customer_summary["Revenue"]
        / customer_summary["UnitsPurchased"]
    )

    frequency_revenue_correlation = customer_summary[
        ["Transactions", "Revenue"]
    ].corr().loc["Transactions", "Revenue"]

    frequency_units_correlation = customer_summary[
        ["Transactions", "UnitsPurchased"]
    ].corr().loc["Transactions", "UnitsPurchased"]

    print("\nCustomer count:")
    print(len(customer_summary))

    print("\nCorrelation between transactions and revenue:")
    print(f"{frequency_revenue_correlation:.6f}")

    print("\nCorrelation between transactions and units purchased:")
    print(f"{frequency_units_correlation:.6f}")

    print("\nTop 10 customers by transaction count:")

    print(
        customer_summary
        .sort_values("Transactions", ascending=False)
        .head(10)
        .to_string(index=False)
    )

    print("\nTop 10 customers by revenue:")

    print(
        customer_summary
        .sort_values("Revenue", ascending=False)
        .head(10)
        .to_string(index=False)
    )

    print("\nTop 10 customers by average revenue per transaction:")

    print(
        customer_summary
        .sort_values(
            "AverageRevenuePerTransaction",
            ascending=False,
        )
        .head(10)
        .to_string(index=False)
    )

    print("\nTop 10 customers by revenue per unit:")

    print(
        customer_summary
        .sort_values(
            "RevenuePerUnit",
            ascending=False,
        )
        .head(10)
        .to_string(index=False)
    )

def analyze_customer_revenue_segments(df: pd.DataFrame) -> None:
    """Analyze customer revenue distribution across percentile segments."""

    print("\n" + "=" * 70)
    print("CUSTOMER REVENUE DISTRIBUTION AND SEGMENTATION")
    print("=" * 70)

    customer_summary = (
        df.groupby("CustomerID")
        .agg(
            Revenue=("Revenue", "sum"),
            Transactions=("InvoiceNo", "nunique"),
            UnitsPurchased=("Quantity", "sum"),
        )
        .reset_index()
        .sort_values("Revenue", ascending=False)
        .reset_index(drop=True)
    )

    total_customers = len(customer_summary)
    total_revenue = customer_summary["Revenue"].sum()

    print("\nTotal customers:")
    print(total_customers)

    print("\nTotal revenue:")
    print(f"{total_revenue:.2f}")

    segment_definitions = [
        ("Top 1%", 0.01),
        ("Top 5%", 0.05),
        ("Top 10%", 0.10),
        ("Top 20%", 0.20),
    ]

    print("\nCustomer revenue segments:")

    for segment_name, proportion in segment_definitions:
        customer_count = max(
            1,
            int(total_customers * proportion),
        )

        segment = customer_summary.head(customer_count)

        segment_revenue = segment["Revenue"].sum()
        segment_transactions = segment["Transactions"].sum()
        segment_units = segment["UnitsPurchased"].sum()

        revenue_share = segment_revenue / total_revenue

        print(f"\n{segment_name}:")
        print(f"Customers: {customer_count}")
        print(f"Customer share: {customer_count / total_customers:.2%}")
        print(f"Revenue: {segment_revenue:.2f}")
        print(f"Revenue share: {revenue_share:.2%}")
        print(f"Transactions: {segment_transactions}")
        print(f"Units purchased: {segment_units}")

    top_20_count = max(
        1,
        int(total_customers * 0.20),
    )

    top_20 = customer_summary.head(top_20_count)
    remaining = customer_summary.iloc[top_20_count:]

    print("\nTop 20% vs remaining customers:")

    comparison = pd.DataFrame(
        {
            "Segment": [
                "Top 20%",
                "Remaining 80%",
            ],
            "Customers": [
                len(top_20),
                len(remaining),
            ],
            "Revenue": [
                top_20["Revenue"].sum(),
                remaining["Revenue"].sum(),
            ],
            "Transactions": [
                top_20["Transactions"].sum(),
                remaining["Transactions"].sum(),
            ],
            "UnitsPurchased": [
                top_20["UnitsPurchased"].sum(),
                remaining["UnitsPurchased"].sum(),
            ],
        }
    )

    comparison["RevenueShare"] = (
        comparison["Revenue"] / total_revenue
    )

    comparison["RevenuePerCustomer"] = (
        comparison["Revenue"]
        / comparison["Customers"]
    )

    print(comparison.to_string(index=False))

def analyze_customer_frequency_segments(df: pd.DataFrame) -> None:
    """Analyze customer revenue across purchase-frequency segments."""

    print("\n" + "=" * 70)
    print("CUSTOMER PURCHASE FREQUENCY SEGMENTATION")
    print("=" * 70)

    customer_summary = (
        df.groupby("CustomerID")
        .agg(
            Revenue=("Revenue", "sum"),
            Transactions=("InvoiceNo", "nunique"),
            UnitsPurchased=("Quantity", "sum"),
        )
        .reset_index()
    )

    total_revenue = customer_summary["Revenue"].sum()

    def assign_frequency_segment(transactions: int) -> str:
        if transactions == 1:
            return "1 transaction"
        if transactions <= 3:
            return "2-3 transactions"
        if transactions <= 5:
            return "4-5 transactions"
        if transactions <= 10:
            return "6-10 transactions"
        if transactions <= 20:
            return "11-20 transactions"
        return "21+ transactions"

    customer_summary["FrequencySegment"] = (
        customer_summary["Transactions"]
        .apply(assign_frequency_segment)
    )

    segment_order = [
        "1 transaction",
        "2-3 transactions",
        "4-5 transactions",
        "6-10 transactions",
        "11-20 transactions",
        "21+ transactions",
    ]

    segment_summary = (
        customer_summary
        .groupby("FrequencySegment", observed=False)
        .agg(
            Customers=("CustomerID", "count"),
            Revenue=("Revenue", "sum"),
            Transactions=("Transactions", "sum"),
            UnitsPurchased=("UnitsPurchased", "sum"),
        )
        .reindex(segment_order)
        .reset_index()
    )

    segment_summary["CustomerShare"] = (
        segment_summary["Customers"]
        / len(customer_summary)
    )

    segment_summary["RevenueShare"] = (
        segment_summary["Revenue"]
        / total_revenue
    )

    segment_summary["RevenuePerCustomer"] = (
        segment_summary["Revenue"]
        / segment_summary["Customers"]
    )

    segment_summary["AverageRevenuePerTransaction"] = (
        segment_summary["Revenue"]
        / segment_summary["Transactions"]
    )

    print("\nCustomer frequency segments:")

    print(
        segment_summary.to_string(index=False)
    )

    print("\nRevenue share by frequency segment:")

    print(
        segment_summary[
            [
                "FrequencySegment",
                "Customers",
                "CustomerShare",
                "Revenue",
                "RevenueShare",
                "RevenuePerCustomer",
            ]
        ].to_string(index=False)
    )

def analyze_customer_recency_frequency_segments(
    df: pd.DataFrame,
) -> None:
    """Analyze customer segments using recency and purchase frequency."""

    print("\n" + "=" * 70)
    print("CUSTOMER RECENCY VS FREQUENCY SEGMENTATION")
    print("=" * 70)

    analysis_date = df["InvoiceDate"].max()

    customer_summary = (
        df.groupby("CustomerID")
        .agg(
            LastPurchase=("InvoiceDate", "max"),
            Revenue=("Revenue", "sum"),
            Transactions=("InvoiceNo", "nunique"),
            UnitsPurchased=("Quantity", "sum"),
        )
        .reset_index()
    )

    customer_summary["RecencyDays"] = (
        analysis_date - customer_summary["LastPurchase"]
    ).dt.days

    def assign_recency_segment(recency_days: int) -> str:
        if recency_days <= 30:
            return "0-30 days"
        if recency_days <= 90:
            return "31-90 days"
        if recency_days <= 180:
            return "91-180 days"
        if recency_days <= 365:
            return "181-365 days"
        return "366+ days"

    def assign_frequency_segment(transactions: int) -> str:
        if transactions == 1:
            return "1 transaction"
        if transactions <= 3:
            return "2-3 transactions"
        if transactions <= 5:
            return "4-5 transactions"
        if transactions <= 10:
            return "6-10 transactions"
        if transactions <= 20:
            return "11-20 transactions"
        return "21+ transactions"

    customer_summary["RecencySegment"] = (
        customer_summary["RecencyDays"]
        .apply(assign_recency_segment)
    )

    customer_summary["FrequencySegment"] = (
        customer_summary["Transactions"]
        .apply(assign_frequency_segment)
    )

    recency_order = [
        "0-30 days",
        "31-90 days",
        "91-180 days",
        "181-365 days",
        "366+ days",
    ]

    frequency_order = [
        "1 transaction",
        "2-3 transactions",
        "4-5 transactions",
        "6-10 transactions",
        "11-20 transactions",
        "21+ transactions",
    ]

    segment_summary = (
        customer_summary
        .groupby(
            ["RecencySegment", "FrequencySegment"],
            observed=False,
        )
        .agg(
            Customers=("CustomerID", "count"),
            Revenue=("Revenue", "sum"),
            Transactions=("Transactions", "sum"),
            UnitsPurchased=("UnitsPurchased", "sum"),
        )
        .reset_index()
    )

    segment_summary["RecencySegment"] = pd.Categorical(
        segment_summary["RecencySegment"],
        categories=recency_order,
        ordered=True,
    )

    segment_summary["FrequencySegment"] = pd.Categorical(
        segment_summary["FrequencySegment"],
        categories=frequency_order,
        ordered=True,
    )

    segment_summary = segment_summary.sort_values(
        ["RecencySegment", "FrequencySegment"]
    )

    print("\nAnalysis date:")
    print(analysis_date)

    print("\nCustomer segment matrix:")

    print(
        segment_summary.to_string(index=False)
    )

    print("\nCustomer counts by recency segment:")

    print(
        customer_summary["RecencySegment"]
        .value_counts()
        .reindex(recency_order, fill_value=0)
        .to_string()
    )

    print("\nCustomer counts by frequency segment:")

    print(
        customer_summary["FrequencySegment"]
        .value_counts()
        .reindex(frequency_order, fill_value=0)
        .to_string()
    )

def analyze_customer_cohorts(df: pd.DataFrame) -> None:
    """Analyze customers by their first-purchase month."""

    print("\n" + "=" * 70)
    print("CUSTOMER COHORT ANALYSIS")
    print("=" * 70)

    customer_first_purchase = (
        df.groupby("CustomerID")["InvoiceDate"]
        .min()
        .reset_index(name="FirstPurchaseDate")
    )

    customer_first_purchase["CohortMonth"] = (
        customer_first_purchase["FirstPurchaseDate"]
        .dt.to_period("M")
        .astype(str)
    )

    customer_summary = (
        df.groupby("CustomerID")
        .agg(
            Revenue=("Revenue", "sum"),
            Transactions=("InvoiceNo", "nunique"),
            UnitsPurchased=("Quantity", "sum"),
            LastPurchaseDate=("InvoiceDate", "max"),
        )
        .reset_index()
    )

    customer_summary = customer_summary.merge(
        customer_first_purchase,
        on="CustomerID",
        how="left",
    )

    cohort_summary = (
        customer_summary
        .groupby("CohortMonth")
        .agg(
            Customers=("CustomerID", "count"),
            Revenue=("Revenue", "sum"),
            Transactions=("Transactions", "sum"),
            UnitsPurchased=("UnitsPurchased", "sum"),
        )
        .reset_index()
    )

    total_revenue = cohort_summary["Revenue"].sum()
    total_customers = cohort_summary["Customers"].sum()

    cohort_summary["CustomerShare"] = (
        cohort_summary["Customers"]
        / total_customers
    )

    cohort_summary["RevenueShare"] = (
        cohort_summary["Revenue"]
        / total_revenue
    )

    cohort_summary["RevenuePerCustomer"] = (
        cohort_summary["Revenue"]
        / cohort_summary["Customers"]
    )

    cohort_summary["TransactionsPerCustomer"] = (
        cohort_summary["Transactions"]
        / cohort_summary["Customers"]
    )

    print("\nCustomer first-purchase cohorts:")

    print(
        cohort_summary.to_string(index=False)
    )

    print("\nTotal cohorts:")
    print(len(cohort_summary))

    print("\nLargest cohorts by customer count:")

    print(
        cohort_summary
        .sort_values("Customers", ascending=False)
        .head(10)
        .to_string(index=False)
    )

    print("\nLargest cohorts by revenue:")

    print(
        cohort_summary
        .sort_values("Revenue", ascending=False)
        .head(10)
        .to_string(index=False)
    )

def analyze_customer_cohort_retention(df: pd.DataFrame) -> None:
    """Analyze customer retention by months since first purchase."""

    print("\n" + "=" * 70)
    print("CUSTOMER COHORT RETENTION ANALYSIS")
    print("=" * 70)

    customer_first_purchase = (
        df.groupby("CustomerID")["InvoiceDate"]
        .min()
        .reset_index(name="FirstPurchaseDate")
    )

    customer_first_purchase["CohortMonth"] = (
        customer_first_purchase["FirstPurchaseDate"]
        .dt.to_period("M")
    )

    transaction_months = (
        df[["CustomerID", "InvoiceDate"]]
        .copy()
    )

    transaction_months["PurchaseMonth"] = (
        transaction_months["InvoiceDate"]
        .dt.to_period("M")
    )

    transaction_months = transaction_months.merge(
        customer_first_purchase[
            ["CustomerID", "CohortMonth"]
        ],
        on="CustomerID",
        how="left",
    )

    transaction_months["MonthOffset"] = (
        (
            transaction_months["PurchaseMonth"].dt.year
            - transaction_months["CohortMonth"].dt.year
        ) * 12
        + (
            transaction_months["PurchaseMonth"].dt.month
            - transaction_months["CohortMonth"].dt.month
        )
    )

    customer_month_activity = (
        transaction_months[
            ["CustomerID", "CohortMonth", "MonthOffset"]
        ]
        .drop_duplicates()
    )

    cohort_sizes = (
        customer_first_purchase
        .groupby("CohortMonth")
        .size()
        .rename("CohortCustomers")
    )

    retention = (
        customer_month_activity
        .groupby(
            ["CohortMonth", "MonthOffset"]
        )
        .size()
        .rename("ActiveCustomers")
        .reset_index()
    )

    retention = retention.merge(
        cohort_sizes.reset_index(),
        on="CohortMonth",
        how="left",
    )

    retention["RetentionRate"] = (
        retention["ActiveCustomers"]
        / retention["CohortCustomers"]
    )

    print("\nCohort retention records:")

    print(
        retention
        .sort_values(
            ["CohortMonth", "MonthOffset"]
        )
        .to_string(index=False)
    )

    retention_matrix = (
        retention
        .pivot(
            index="CohortMonth",
            columns="MonthOffset",
            values="RetentionRate",
        )
        .sort_index()
    )

    print("\nCohort retention matrix:")

    print(
        retention_matrix.to_string()
    )

    print("\nRetention at month 1:")

    month_1 = retention[
        retention["MonthOffset"] == 1
    ].copy()

    print(
        month_1[
            [
                "CohortMonth",
                "CohortCustomers",
                "ActiveCustomers",
                "RetentionRate",
            ]
        ].to_string(index=False)
    )

def analyze_cohort_monetary_value(df: pd.DataFrame) -> None:
    """Analyze monetary value across customer acquisition cohorts."""

    print("\n" + "=" * 70)
    print("CUSTOMER COHORT MONETARY VALUE ANALYSIS")
    print("=" * 70)

    customer_first_purchase = (
        df.groupby("CustomerID")["InvoiceDate"]
        .min()
        .reset_index(name="FirstPurchaseDate")
    )

    customer_first_purchase["CohortMonth"] = (
        customer_first_purchase["FirstPurchaseDate"]
        .dt.to_period("M")
        .astype(str)
    )

    customer_summary = (
        df.groupby("CustomerID")
        .agg(
            Revenue=("Revenue", "sum"),
            Transactions=("InvoiceNo", "nunique"),
            UnitsPurchased=("Quantity", "sum"),
        )
        .reset_index()
    )

    customer_summary = customer_summary.merge(
        customer_first_purchase[
            ["CustomerID", "CohortMonth"]
        ],
        on="CustomerID",
        how="left",
    )

    cohort_summary = (
        customer_summary
        .groupby("CohortMonth")
        .agg(
            Customers=("CustomerID", "count"),
            Revenue=("Revenue", "sum"),
            Transactions=("Transactions", "sum"),
            UnitsPurchased=("UnitsPurchased", "sum"),
        )
        .reset_index()
    )

    cohort_summary["RevenuePerCustomer"] = (
        cohort_summary["Revenue"]
        / cohort_summary["Customers"]
    )

    cohort_summary["TransactionsPerCustomer"] = (
        cohort_summary["Transactions"]
        / cohort_summary["Customers"]
    )

    cohort_summary["RevenuePerTransaction"] = (
        cohort_summary["Revenue"]
        / cohort_summary["Transactions"]
    )

    cohort_summary["UnitsPerCustomer"] = (
        cohort_summary["UnitsPurchased"]
        / cohort_summary["Customers"]
    )

    print("\nCohort monetary value:")

    print(
        cohort_summary.to_string(index=False)
    )

    print("\nHighest revenue per customer cohorts:")

    print(
        cohort_summary
        .sort_values(
            "RevenuePerCustomer",
            ascending=False,
        )
        .head(10)
        .to_string(index=False)
    )

    print("\nHighest revenue per transaction cohorts:")

    print(
        cohort_summary
        .sort_values(
            "RevenuePerTransaction",
            ascending=False,
        )
        .head(10)
        .to_string(index=False)
    )

def analyze_time_series_structure(df: pd.DataFrame) -> None:
    """Analyze temporal structure of daily revenue for forecasting."""

    daily_revenue = (
        df.assign(SaleDate=df["InvoiceDate"].dt.normalize())
        .groupby("SaleDate")["Revenue"]
        .sum()
    )

    full_date_index = pd.date_range(
        start=daily_revenue.index.min(),
        end=daily_revenue.index.max(),
        freq="D",
    )

    daily_revenue = daily_revenue.reindex(
        full_date_index,
        fill_value=0,
    )

    daily_revenue.name = "Revenue"

    print("\n=== TIME-SERIES STRUCTURE ===")

    print(f"Calendar days: {len(daily_revenue)}")
    print(f"Days with recorded sales: {(daily_revenue > 0).sum()}")
    print(f"Days with zero recorded sales: {(daily_revenue == 0).sum()}")
    print(f"First date: {daily_revenue.index.min().date()}")
    print(f"Last date: {daily_revenue.index.max().date()}")

    print("\nDaily revenue statistics:")
    print(daily_revenue.describe())

    print("\nWeekly seasonality:")
    weekday_summary = (
        daily_revenue.groupby(daily_revenue.index.day_name())
        .agg(
            AverageRevenue="mean",
            TotalRevenue="sum",
            ActiveDays=lambda x: (x > 0).sum(),
        )
        .reindex(
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
        )
    )

    print(weekday_summary)

    print("\nRevenue autocorrelation:")
    for lag in [1, 2, 3, 7, 14, 28]:
        print(f"Lag {lag:>2}: {daily_revenue.autocorr(lag=lag):.6f}")

    rolling_7 = daily_revenue.rolling(window=7).mean()
    rolling_30 = daily_revenue.rolling(window=30).mean()

    print("\nRolling revenue:")
    print(f"Latest 7-day average: {rolling_7.iloc[-1]:.2f}")
    print(f"Latest 30-day average: {rolling_30.iloc[-1]:.2f}")

    print("\nRolling volatility:")
    rolling_7_std = daily_revenue.rolling(window=7).std()
    rolling_30_std = daily_revenue.rolling(window=30).std()

    print(f"Latest 7-day std: {rolling_7_std.iloc[-1]:.2f}")
    print(f"Latest 30-day std: {rolling_30_std.iloc[-1]:.2f}")

    print("\nDaily revenue changes:")
    daily_change = daily_revenue.diff()

    print(f"Average daily change: {daily_change.mean():.2f}")
    print(f"Median daily change: {daily_change.median():.2f}")

    print("\nTime-series analysis complete.")

def prepare_daily_revenue_series(df: pd.DataFrame) -> pd.DataFrame:
    """Create a complete chronological daily revenue series."""

    daily_revenue = (
        df.assign(Date=df["InvoiceDate"].dt.normalize())
        .groupby("Date", as_index=True)["Revenue"]
        .sum()
    )

    full_date_index = pd.date_range(
        start=daily_revenue.index.min(),
        end=daily_revenue.index.max(),
        freq="D",
    )

    daily_revenue = daily_revenue.reindex(
        full_date_index,
        fill_value=0,
    )

    daily_revenue.index.name = "Date"

    daily_df = daily_revenue.reset_index()

    print("\n=== DAILY REVENUE TARGET DATASET ===")
    print(f"Rows: {len(daily_df)}")
    print(f"Columns: {list(daily_df.columns)}")
    print(f"First date: {daily_df['Date'].min().date()}")
    print(f"Last date: {daily_df['Date'].max().date()}")
    print(f"Missing dates: {daily_df['Date'].isna().sum()}")
    print(f"Missing revenue: {daily_df['Revenue'].isna().sum()}")
    print(f"Duplicate dates: {daily_df['Date'].duplicated().sum()}")
    print(f"Zero-revenue days: {(daily_df['Revenue'] == 0).sum()}")

    print("\nTarget preview:")
    print(daily_df.head())

    print("\nTarget tail:")
    print(daily_df.tail())

    print("\nTarget statistics:")
    print(daily_df["Revenue"].describe())

    return daily_df


def create_forecasting_split(
    daily_df: pd.DataFrame,
    train_ratio: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a chronological train/test split for forecasting."""

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    daily_df = daily_df.sort_values("Date").reset_index(drop=True)

    split_index = int(len(daily_df) * train_ratio)

    train_df = daily_df.iloc[:split_index].copy()
    test_df = daily_df.iloc[split_index:].copy()

    print("\n=== FORECASTING TRAIN/TEST SPLIT ===")

    print(f"Total rows: {len(daily_df)}")
    print(f"Training rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Training ratio: {len(train_df) / len(daily_df):.4f}")
    print(f"Test ratio: {len(test_df) / len(daily_df):.4f}")

    print("\nTraining period:")
    print(f"Start: {train_df['Date'].min().date()}")
    print(f"End: {train_df['Date'].max().date()}")

    print("\nTest period:")
    print(f"Start: {test_df['Date'].min().date()}")
    print(f"End: {test_df['Date'].max().date()}")

    print("\nSplit validation:")
    print(f"Training dates sorted: {train_df['Date'].is_monotonic_increasing}")
    print(f"Test dates sorted: {test_df['Date'].is_monotonic_increasing}")
    print(
        "No temporal overlap:",
        train_df["Date"].max() < test_df["Date"].min(),
    )

    return train_df, test_df

def evaluate_naive_forecasts(
    daily_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Evaluate simple forecasting baselines on the test period."""

    actual = test_df["Revenue"].to_numpy()

    # Baseline 1: training-set mean.
    mean_prediction = np.full(
        shape=len(test_df),
        fill_value=train_df["Revenue"].mean(),
    )

    # Baseline 2: last observed training value.
    last_value_prediction = np.full(
        shape=len(test_df),
        fill_value=train_df["Revenue"].iloc[-1],
    )

    # Baseline 3: seasonal naive using the same weekday from 7 days earlier.
    daily_series = daily_df.set_index("Date")["Revenue"]

    seasonal_prediction = daily_series.shift(7).loc[
        test_df["Date"]
    ].to_numpy()

    predictions = {
        "Training Mean": mean_prediction,
        "Last Observed Value": last_value_prediction,
        "Seasonal Naive (7-day)": seasonal_prediction,
    }

    print("\n=== NAIVE FORECAST BASELINES ===")

    print(f"Test observations: {len(actual)}")

    for name, prediction in predictions.items():
        if np.isnan(prediction).any():
            raise ValueError(
                f"{name} contains missing predictions."
            )

        errors = actual - prediction

        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))

        actual_total = np.sum(np.abs(actual))

        if actual_total == 0:
            wape = np.nan
        else:
            wape = np.sum(np.abs(errors)) / actual_total

        print(f"\n{name}:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"WAPE: {wape:.4f}")

    print("\nBaseline evaluation complete.")

def create_forecasting_features(
    daily_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create leakage-safe calendar, lag, and rolling features."""

    feature_df = daily_df.copy()
    feature_df = feature_df.sort_values("Date").reset_index(drop=True)

    feature_df["DayOfWeek"] = feature_df["Date"].dt.dayofweek
    feature_df["DayOfMonth"] = feature_df["Date"].dt.day
    feature_df["Month"] = feature_df["Date"].dt.month
    feature_df["WeekOfYear"] = feature_df["Date"].dt.isocalendar().week.astype(int)
    feature_df["IsWeekend"] = (
        feature_df["DayOfWeek"] >= 5
    ).astype(int)

    feature_df["Lag1"] = feature_df["Revenue"].shift(1)
    feature_df["Lag7"] = feature_df["Revenue"].shift(7)
    feature_df["Lag14"] = feature_df["Revenue"].shift(14)
    feature_df["Lag28"] = feature_df["Revenue"].shift(28)

    historical_revenue = feature_df["Revenue"].shift(1)

    feature_df["RollingMean7"] = (
        historical_revenue.rolling(window=7).mean()
    )

    feature_df["RollingMean28"] = (
        historical_revenue.rolling(window=28).mean()
    )

    print("\n=== FORECASTING FEATURES ===")

    print(f"Rows: {len(feature_df)}")
    print(f"Columns: {list(feature_df.columns)}")

    print("\nFeature missing values:")
    print(feature_df.isna().sum())

    print("\nFeature preview:")
    print(feature_df.head(35))

    return feature_df

def prepare_model_dataset(
    feature_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Prepare the leakage-safe feature dataset for forecasting models."""

    feature_columns = [
        "DayOfWeek",
        "DayOfMonth",
        "Month",
        "WeekOfYear",
        "IsWeekend",
        "Lag1",
        "Lag7",
        "Lag14",
        "Lag28",
        "RollingMean7",
        "RollingMean28",
    ]

    required_columns = [
        "Date",
        "Revenue",
        *feature_columns,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in feature_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    model_df = feature_df[
        required_columns
    ].copy()

    initial_rows = len(model_df)

    model_df = model_df.dropna(
        subset=feature_columns
    ).reset_index(drop=True)

    removed_rows = initial_rows - len(model_df)

    print("\n=== MODEL-READY DATASET ===")

    print(f"Initial rows: {initial_rows}")
    print(f"Rows removed for feature warm-up: {removed_rows}")
    print(f"Final rows: {len(model_df)}")

    print(f"\nFeature columns ({len(feature_columns)}):")
    print(feature_columns)

    print("\nMissing values:")
    print(model_df.isna().sum())

    print("\nDuplicate dates:")
    print(model_df["Date"].duplicated().sum())

    print("\nChronological order:")
    print(model_df["Date"].is_monotonic_increasing)

    print("\nModel-ready date range:")
    print(f"Start: {model_df['Date'].min().date()}")
    print(f"End: {model_df['Date'].max().date()}")

    print("\nModel-ready preview:")
    print(model_df.head())

    print("\nModel-ready tail:")
    print(model_df.tail())

    return model_df, feature_columns

def split_model_dataset(
    model_df: pd.DataFrame,
    train_end_date: str = "2011-09-25",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the model-ready dataset using a fixed chronological boundary."""

    split_date = pd.Timestamp(train_end_date)

    train_df = model_df.loc[
        model_df["Date"] <= split_date
    ].copy()

    test_df = model_df.loc[
        model_df["Date"] > split_date
    ].copy()

    if train_df.empty:
        raise ValueError("Training dataset is empty.")

    if test_df.empty:
        raise ValueError("Test dataset is empty.")

    if train_df["Date"].max() >= test_df["Date"].min():
        raise ValueError("Training and test periods overlap.")

    if not train_df["Date"].is_monotonic_increasing:
        raise ValueError("Training data is not chronological.")

    if not test_df["Date"].is_monotonic_increasing:
        raise ValueError("Test data is not chronological.")

    print("\n=== MODEL DATASET SPLIT ===")

    print(f"Total rows: {len(model_df)}")
    print(f"Training rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")

    print("\nTraining period:")
    print(f"Start: {train_df['Date'].min().date()}")
    print(f"End: {train_df['Date'].max().date()}")

    print("\nTest period:")
    print(f"Start: {test_df['Date'].min().date()}")
    print(f"End: {test_df['Date'].max().date()}")

    print("\nSplit validation:")
    print(
        "Training dates sorted:",
        train_df["Date"].is_monotonic_increasing,
    )
    print(
        "Test dates sorted:",
        test_df["Date"].is_monotonic_increasing,
    )
    print(
        "No temporal overlap:",
        train_df["Date"].max() < test_df["Date"].min(),
    )

    return train_df, test_df
def train_xgboost_forecaster(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
):
    """Train a first-pass XGBoost revenue forecasting model."""

    from xgboost import XGBRegressor

    X_train = train_df[feature_columns]
    y_train = train_df["Revenue"]

    X_test = test_df[feature_columns]
    y_test = test_df["Revenue"]

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    print("\n=== XGBOOST FORECASTER ===")

    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Features: {len(feature_columns)}")

    print("\nTraining prediction summary:")
    print(
        pd.Series(train_predictions).describe()
    )

    print("\nTest prediction summary:")
    print(
        pd.Series(test_predictions).describe()
    )

    print("\nXGBoost training complete.")

    return model, train_predictions, test_predictions

def evaluate_xgboost_forecast(
    test_df: pd.DataFrame,
    test_predictions: np.ndarray,
) -> None:
    """Evaluate XGBoost predictions on the chronological test period."""

    actual = test_df["Revenue"].to_numpy()

    if len(actual) != len(test_predictions):
        raise ValueError(
            "Actual and prediction lengths do not match."
        )

    errors = actual - test_predictions

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))

    actual_total = np.sum(np.abs(actual))

    if actual_total == 0:
        wape = np.nan
    else:
        wape = np.sum(np.abs(errors)) / actual_total

    negative_prediction_count = np.sum(
        test_predictions < 0
    )

    print("\n=== XGBOOST EVALUATION ===")

    print(f"Test observations: {len(actual)}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"WAPE: {wape:.4f}")

    print("\nPrediction diagnostics:")
    print(
        f"Negative predictions: "
        f"{negative_prediction_count}"
    )
    print(
        f"Minimum prediction: "
        f"{test_predictions.min():.2f}"
    )
    print(
        f"Maximum prediction: "
        f"{test_predictions.max():.2f}"
    )

    print("\nLargest absolute errors:")

    evaluation_df = test_df[
        ["Date", "Revenue"]
    ].copy()

    evaluation_df["Prediction"] = test_predictions
    evaluation_df["AbsoluteError"] = np.abs(errors)

    print(
        evaluation_df
        .sort_values(
            "AbsoluteError",
            ascending=False,
        )
        .head(10)
        .to_string(index=False)
    )

    print("\nXGBoost evaluation complete.")
def analyze_xgboost_feature_importance(model, feature_columns):
    """Analyze feature importance from the trained XGBoost model."""

    importance_df = pd.DataFrame(
        {
            "Feature": feature_columns,
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    print("\n=== XGBOOST FEATURE IMPORTANCE ===")
    print(importance_df.to_string(index=False))

    print("\nTotal importance:", importance_df["Importance"].sum())

    return importance_df

def analyze_negative_predictions(test_df, test_predictions):
    """Inspect negative predictions produced by the forecasting model."""

    negative_mask = test_predictions < 0

    negative_df = test_df.loc[
        negative_mask,
        ["Date", "Revenue"],
    ].copy()

    negative_df["Prediction"] = test_predictions[negative_mask]
    negative_df["Error"] = (
        negative_df["Revenue"] - negative_df["Prediction"]
    )

    print("\n=== NEGATIVE PREDICTION ANALYSIS ===")
    print("Negative predictions:", negative_mask.sum())

    if negative_df.empty:
        print("No negative predictions found.")
    else:
        print(negative_df.to_string(index=False))

    return negative_df

def create_model_validation_split(model_df, validation_days=30):
    """Create chronological train and validation datasets."""

    if validation_days <= 0:
        raise ValueError("validation_days must be positive.")

    if validation_days >= len(model_df):
        raise ValueError("validation_days must be smaller than the dataset size.")

    split_index = len(model_df) - validation_days

    train_df = model_df.iloc[:split_index].copy()
    validation_df = model_df.iloc[split_index:].copy()

    print("\n=== MODEL VALIDATION SPLIT ===")
    print("Training rows:", len(train_df))
    print("Validation rows:", len(validation_df))

    print(
        "Training date range:",
        train_df["Date"].min(),
        "to",
        train_df["Date"].max(),
    )

    print(
        "Validation date range:",
        validation_df["Date"].min(),
        "to",
        validation_df["Date"].max(),
    )

    print(
        "Chronological:",
        train_df["Date"].max() < validation_df["Date"].min(),
    )

    return train_df, validation_df

def evaluate_xgboost_validation(
    train_df,
    validation_df,
    feature_columns,
):
    """Train XGBoost on the training subset and evaluate validation data."""

    model, train_predictions, validation_predictions = (
        train_xgboost_forecaster(
            train_df,
            validation_df,
            feature_columns,
        )
    )

    print("\n=== XGBOOST VALIDATION EVALUATION ===")

    evaluate_xgboost_forecast(
        validation_df,
        validation_predictions,
    )

    return (
        model,
        train_predictions,
        validation_predictions,
    )

def evaluate_validation_baselines(
    daily_df,
    train_df,
    validation_df,
):
    """Evaluate naive forecasting baselines on the validation period."""

    print("\n=== VALIDATION BASELINE EVALUATION ===")

    evaluate_naive_forecasts(
        daily_df,
        train_df,
        validation_df,
    )

def run_xgboost_validation_experiments(
    train_df,
    validation_df,
    feature_columns,
):
    """Compare a small set of XGBoost configurations on validation data."""

    from xgboost import XGBRegressor

    configurations = [
        {
            "name": "baseline",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        {
            "name": "shallower",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        {
            "name": "lower_learning_rate",
            "n_estimators": 500,
            "learning_rate": 0.03,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
    ]

    results = []

    X_train = train_df[feature_columns]
    y_train = train_df["Revenue"]

    X_validation = validation_df[feature_columns]
    y_validation = validation_df["Revenue"]

    for config in configurations:
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=config["n_estimators"],
            learning_rate=config["learning_rate"],
            max_depth=config["max_depth"],
            subsample=config["subsample"],
            colsample_bytree=config["colsample_bytree"],
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_validation)

        errors = y_validation.to_numpy() - predictions

        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))

        actual_sum = np.sum(np.abs(y_validation.to_numpy()))

        if actual_sum == 0:
            wape = np.nan
        else:
            wape = np.sum(np.abs(errors)) / actual_sum

        results.append(
            {
                "Configuration": config["name"],
                "MAE": mae,
                "RMSE": rmse,
                "WAPE": wape,
                "NegativePredictions": int(np.sum(predictions < 0)),
            }
        )

    results_df = pd.DataFrame(results)

    print("\n=== XGBOOST VALIDATION EXPERIMENTS ===")
    print(results_df.to_string(index=False))

    return results_df

def run_walk_forward_validation(
    model_df,
    feature_columns,
    validation_days=30,
    windows=3,
):
    """Run chronological walk-forward validation."""

    results = []

    total_rows = len(model_df)

    for window in range(windows):
        validation_end = total_rows - (windows - window - 1) * validation_days
        validation_start = validation_end - validation_days

        train_end = validation_start

        if train_end <= 0:
            raise ValueError(
                "Not enough data for the requested walk-forward windows."
            )

        train_df = model_df.iloc[:train_end].copy()
        validation_df = model_df.iloc[
            validation_start:validation_end
        ].copy()

        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )

        X_train = train_df[feature_columns]
        y_train = train_df["Revenue"]

        X_validation = validation_df[feature_columns]
        y_validation = validation_df["Revenue"]

        model.fit(X_train, y_train)

        predictions = model.predict(X_validation)

        errors = y_validation.to_numpy() - predictions

        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))

        actual_sum = np.sum(
            np.abs(y_validation.to_numpy())
        )

        if actual_sum == 0:
            wape = np.nan
        else:
            wape = np.sum(np.abs(errors)) / actual_sum

        results.append(
            {
                "Window": window + 1,
                "TrainRows": len(train_df),
                "ValidationRows": len(validation_df),
                "TrainStart": train_df["Date"].min(),
                "TrainEnd": train_df["Date"].max(),
                "ValidationStart": validation_df["Date"].min(),
                "ValidationEnd": validation_df["Date"].max(),
                "MAE": mae,
                "RMSE": rmse,
                "WAPE": wape,
                "NegativePredictions": int(
                    np.sum(predictions < 0)
                ),
            }
        )

    results_df = pd.DataFrame(results)

    print("\n=== WALK-FORWARD VALIDATION ===")
    print(results_df.to_string(index=False))

    print("\nAverage metrics:")
    print(
        "MAE:",
        results_df["MAE"].mean(),
    )
    print(
        "RMSE:",
        results_df["RMSE"].mean(),
    )
    print(
        "WAPE:",
        results_df["WAPE"].mean(),
    )

    return results_df

def run_walk_forward_baselines(
    daily_df,
    model_df,
    validation_days=30,
    windows=3,
):
    """Evaluate naive baselines across the same walk-forward windows."""

    results = []

    total_rows = len(model_df)

    for window in range(windows):
        validation_end = (
            total_rows
            - (windows - window - 1) * validation_days
        )

        validation_start = validation_end - validation_days
        train_end = validation_start

        train_model_df = model_df.iloc[:train_end].copy()
        validation_model_df = model_df.iloc[
            validation_start:validation_end
        ].copy()

        train_end_date = train_model_df["Date"].max()
        validation_end_date = validation_model_df["Date"].max()

        daily_train_df = daily_df[
            daily_df["Date"] <= train_end_date
        ].copy()

        daily_validation_df = daily_df[
            (daily_df["Date"] > train_end_date)
            & (daily_df["Date"] <= validation_end_date)
        ].copy()

        actual = daily_validation_df["Revenue"].to_numpy()

        last_value_predictions = np.full(
            len(actual),
            daily_train_df["Revenue"].iloc[-1],
        )

        seasonal_predictions = daily_validation_df["Date"].apply(
            lambda date: daily_df.loc[
                daily_df["Date"] == date - pd.Timedelta(days=7),
                "Revenue",
            ].iloc[0]
        ).to_numpy()

        last_errors = actual - last_value_predictions
        seasonal_errors = actual - seasonal_predictions

        actual_sum = np.sum(np.abs(actual))

        last_wape = (
            np.sum(np.abs(last_errors)) / actual_sum
            if actual_sum != 0
            else np.nan
        )

        seasonal_wape = (
            np.sum(np.abs(seasonal_errors)) / actual_sum
            if actual_sum != 0
            else np.nan
        )

        results.append(
            {
                "Window": window + 1,
                "LastObservedMAE": np.mean(
                    np.abs(last_errors)
                ),
                "LastObservedRMSE": np.sqrt(
                    np.mean(last_errors ** 2)
                ),
                "LastObservedWAPE": last_wape,
                "SeasonalNaiveMAE": np.mean(
                    np.abs(seasonal_errors)
                ),
                "SeasonalNaiveRMSE": np.sqrt(
                    np.mean(seasonal_errors ** 2)
                ),
                "SeasonalNaiveWAPE": seasonal_wape,
            }
        )

    results_df = pd.DataFrame(results)

    print("\n=== WALK-FORWARD BASELINE VALIDATION ===")
    print(results_df.to_string(index=False))

    print("\nAverage baseline metrics:")

    print(
        "Last Observed WAPE:",
        results_df["LastObservedWAPE"].mean(),
    )

    print(
        "Seasonal Naive WAPE:",
        results_df["SeasonalNaiveWAPE"].mean(),
    )

    return results_df


# ============================================================
# Main
# ============================================================

def main() -> None:
    """Load the dataset and run the current forecasting workflow."""

    df = load_dataset()

    print("\n" + "=" * 70)
    print("UCI ONLINE RETAIL DATASET")
    print("=" * 70)

    print(f"\nDataset shape: {df.shape}")

    print("\nColumn names:")
    for column in df.columns:
        print(f"  - {column}")

    # --------------------------------------------------------
    # Dataset profiling
    # --------------------------------------------------------

    # profile_schema(df)
    # profile_missing_values(df)
    # profile_duplicates(df)
    # profile_cancelled_invoices(df)
    # profile_quantity(df)
    # profile_unit_price(df)
    # profile_dates(df)
    # profile_entities(df)
    # profile_revenue(df)
    # profile_non_standard_transactions(df)
    # profile_invoice_structure(df)
    # profile_customer_activity(df)
    # profile_product_activity(df)
    # profile_country_activity(df)
    # profile_cleaning_impact(df)

    # --------------------------------------------------------
    # Data cleaning
    # --------------------------------------------------------

    cleaned_df = clean_sales_data(df)

    # save_cleaned_sales(cleaned_df)
    # validate_cleaned_sales(cleaned_df)

    # --------------------------------------------------------
    # Exploratory data analysis
    # --------------------------------------------------------

    # analyze_monthly_sales(cleaned_df)
    # analyze_daily_sales(cleaned_df)
    # analyze_daily_revenue_outliers(cleaned_df)
    # investigate_revenue_spikes(cleaned_df)
    # analyze_customer_revenue_concentration(cleaned_df)
    # analyze_product_revenue_concentration(cleaned_df)
    # analyze_country_revenue(cleaned_df)
    # analyze_transaction_value(cleaned_df)
    # analyze_customer_purchase_frequency(cleaned_df)
    # analyze_customer_recency_and_value(cleaned_df)
    # analyze_product_demand(cleaned_df)
    # analyze_product_demand_concentration(cleaned_df)
    # analyze_product_demand_revenue_relationship(cleaned_df)
    # analyze_customer_frequency_revenue_relationship(cleaned_df)
    # analyze_customer_revenue_segments(cleaned_df)
    # analyze_customer_frequency_segments(cleaned_df)
    # analyze_customer_recency_frequency_segments(cleaned_df)
    # analyze_customer_cohorts(cleaned_df)
    # analyze_customer_cohort_retention(cleaned_df)
    # analyze_cohort_monetary_value(cleaned_df)
    # analyze_time_series_structure(cleaned_df)

    # --------------------------------------------------------
    # Daily forecasting dataset
    # --------------------------------------------------------

    daily_df = prepare_daily_revenue_series(cleaned_df)

    # create_forecasting_split(daily_df)

    # train_df, test_df = create_forecasting_split(daily_df)

    # evaluate_naive_forecasts(
    #     daily_df,
    #     train_df,
    #     test_df,
    # )

    # --------------------------------------------------------
    # Forecasting feature engineering
    # --------------------------------------------------------

    feature_df = create_forecasting_features(daily_df)

    model_df, feature_columns = prepare_model_dataset(
        feature_df
    )

    # --------------------------------------------------------
    # Chronological model/test split
    # --------------------------------------------------------

    train_df, test_df = split_model_dataset(
        model_df
    )

    # --------------------------------------------------------
    # Validation split inside training data
    # --------------------------------------------------------

    validation_train_df, validation_df = (
        create_model_validation_split(
            train_df,
            validation_days=30,
        )
    )
    run_walk_forward_validation(
        train_df,
        feature_columns,
        validation_days=30,
        windows=3,
    )
    run_walk_forward_baselines(
        daily_df,
        train_df,
        validation_days=30,
        windows=3,
    )
    evaluate_validation_baselines(
        daily_df,
        validation_train_df,
        validation_df,
    )
    run_xgboost_validation_experiments(
        validation_train_df,
        validation_df,
        feature_columns,
    )
    validation_model, validation_train_predictions, validation_predictions = (
        evaluate_xgboost_validation(
            validation_train_df,
            validation_df,
            feature_columns,
        )
    )
    # --------------------------------------------------------
    # Initial XGBoost model
    # --------------------------------------------------------

    model, train_predictions, test_predictions = (
        train_xgboost_forecaster(
            train_df,
            test_df,
            feature_columns,
        )
    )

    # --------------------------------------------------------
    # XGBoost evaluation
    # --------------------------------------------------------

    evaluate_xgboost_forecast(
        test_df,
        test_predictions,
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    analyze_xgboost_feature_importance(
        model,
        [
            "DayOfWeek",
            "DayOfMonth",
            "Month",
            "WeekOfYear",
            "IsWeekend",
            "Lag1",
            "Lag7",
            "Lag14",
            "Lag28",
            "RollingMean7",
            "RollingMean28",
        ],
    )

    # --------------------------------------------------------
    # Negative prediction analysis
    # --------------------------------------------------------

    analyze_negative_predictions(
        test_df,
        test_predictions,
    )

    # --------------------------------------------------------
    # Final dataset preview
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FIRST 5 RECORDS")
    print("=" * 70)

    print(df.head().to_string(index=False))

    print("\nDataset profiling completed successfully.")



if __name__ == "__main__":
    main()