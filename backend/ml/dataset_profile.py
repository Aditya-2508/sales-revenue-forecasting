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

    print("\n" + "=" * 70)
    print("FIRST 5 RECORDS")
    print("=" * 70)

    print(df.head().to_string(index=False))

    print("\nDataset profiling completed successfully.")


if __name__ == "__main__":
    main()