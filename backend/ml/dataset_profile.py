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

    cleaned_df = clean_sales_data(df)

    # save_cleaned_sales(cleaned_df)

    # validate_cleaned_sales(cleaned_df)

    # analyze_monthly_sales(cleaned_df)

    # analyze_daily_sales(cleaned_df)

    # analyze_daily_revenue_outliers(cleaned_df)

    # investigate_revenue_spikes(cleaned_df)

    # analyze_customer_revenue_concentration(cleaned_df)

    analyze_product_revenue_concentration(cleaned_df)
    
    
    print("\n" + "=" * 70)
    print("FIRST 5 RECORDS")
    print("=" * 70)

    print(df.head().to_string(index=False))

    print("\nDataset profiling completed successfully.")


if __name__ == "__main__":
    main()