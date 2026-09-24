"""Load and preprocess the Ethereum Fraud Detection Dataset.

Dataset source: https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset
Expected file: data/transaction_dataset.csv
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
RAW_FILE = DATA_DIR / "transaction_dataset.csv"


def load_raw_dataset(filepath: Path | str = RAW_FILE) -> pd.DataFrame:
    """Load the raw CSV dataset.

    Args:
        filepath: Path to the CSV file. Defaults to data/transaction_dataset.csv.

    Returns:
        Raw DataFrame with all columns from the dataset.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at {filepath}.\n"
            "Download it from: https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset\n"
            "Place the CSV file at: data/transaction_dataset.csv"
        )
    return pd.read_csv(filepath)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the dataset.

    Steps:
    - Rename columns to snake_case
    - Drop the unnamed index column if present
    - Convert FLAG to int (0 = legitimate, 1 = fraud)
    - Drop rows with all-NaN feature values
    - Fill remaining NaN with 0 (addresses with no activity in that category)
    """
    # Drop unnamed index columns
    unnamed_cols = [c for c in df.columns if "Unnamed" in c]
    df = df.drop(columns=unnamed_cols)

    # Normalize column names: lowercase, replace spaces with underscores
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace(".", "_", regex=False)
    )

    # Ensure flag column is integer
    if "flag" in df.columns:
        df["flag"] = df["flag"].astype(int)

    # Drop rows where all feature columns are NaN
    feature_cols = [c for c in df.columns if c not in ("address", "flag")]
    df = df.dropna(subset=feature_cols, how="all")

    # Fill remaining NaN with 0
    df = df.fillna(0)

    return df


def load_dataset(filepath: Path | str = RAW_FILE) -> pd.DataFrame:
    """Load and clean the dataset in one step.

    Returns:
        Cleaned DataFrame ready for feature engineering.
    """
    raw = load_raw_dataset(filepath)
    return clean_dataset(raw)


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """Return a summary of the dataset for quick inspection.

    Returns:
        Dict with total_addresses, fraud_count, legitimate_count,
        fraud_ratio, num_features, and feature_names.
    """
    feature_cols = [c for c in df.columns if c not in ("address", "flag")]
    return {
        "total_addresses": len(df),
        "fraud_count": int(df["flag"].sum()),
        "legitimate_count": int((df["flag"] == 0).sum()),
        "fraud_ratio": float(df["flag"].mean()),
        "num_features": len(feature_cols),
        "feature_names": feature_cols,
    }


if __name__ == "__main__":
    print("Loading dataset...")
    try:
        df = load_dataset()
        summary = get_dataset_summary(df)
        print(f"Total addresses: {summary['total_addresses']}")
        print(f"Fraud: {summary['fraud_count']} ({summary['fraud_ratio']:.1%})")
        print(f"Legitimate: {summary['legitimate_count']}")
        print(f"Features: {summary['num_features']}")
        print(f"\nFirst 5 rows:")
        print(df.head())
    except FileNotFoundError as e:
        print(e)
