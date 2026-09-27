from pathlib import Path

import pandas as pd


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Raw dataset path
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "Online Retail.xlsx"


def load_dataset():
    """Load the original UCI Online Retail dataset."""
    print(f"Loading dataset from: {DATASET_PATH}")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATASET_PATH}\n"
            "Please download 'Online Retail.xlsx' from the UCI "
            "Machine Learning Repository and place it in data/raw/."
        )

    df = pd.read_excel(DATASET_PATH)

    return df


def main():
    df = load_dataset()

    print("\n" + "=" * 60)
    print("DATASET LOADED SUCCESSFULLY")
    print("=" * 60)

    print(f"\nShape: {df.shape}")

    print("\nColumns:")
    for column in df.columns:
        print(f" - {column}")

    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()