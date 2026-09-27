from pathlib import Path

import pandas as pd


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




# ============================================================
# Main
# ============================================================

def main() -> None:
    """Load the dataset and perform initial profiling."""

    df = load_dataset()

    print("\n" + "=" * 70)
    print("UCI ONLINE RETAIL DATASET")
    print("=" * 70)

    print(f"\nDataset shape: {df.shape}")

    print("\nColumn names:")
    for column in df.columns:
        print(f"  - {column}")

    profile_schema(df)

    profile_missing_values(df)

    profile_duplicates(df)

    profile_cancelled_invoices(df)

    profile_quantity(df)

    profile_unit_price(df)

    profile_dates(df)

    profile_entities(df)

    profile_revenue(df)

    profile_non_standard_transactions(df)

    profile_invoice_structure(df)

    profile_customer_activity(df)

    profile_product_activity(df)

    profile_country_activity(df)

    print("\n" + "=" * 70)
    print("FIRST 5 RECORDS")
    print("=" * 70)

    print(df.head().to_string(index=False))

    print("\nDataset profiling completed successfully.")


if __name__ == "__main__":
    main()