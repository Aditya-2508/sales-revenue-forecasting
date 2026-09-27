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

    print("\n" + "=" * 70)
    print("FIRST 5 RECORDS")
    print("=" * 70)

    print(df.head().to_string(index=False))

    print("\nDataset profiling completed successfully.")


if __name__ == "__main__":
    main()