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

    df = pd.read_excel(DATASET_PATH)

    return df


# ============================================================
# Main
# ============================================================

def main() -> None:
    """Load the dataset and display basic information."""

    df = load_dataset()

    print("\n" + "=" * 70)
    print("UCI ONLINE RETAIL DATASET")
    print("=" * 70)

    print(f"\nDataset shape: {df.shape}")

    print("\nColumn names:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst 5 records:")
    print(df.head())

    print("\nDataset loaded successfully.")


if __name__ == "__main__":
    main()