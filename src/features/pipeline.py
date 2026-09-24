"""Feature pipeline: combine all feature extractors into a unified interface.

This module ties together transaction, graph, and statistical features
into a single pipeline. It can process:
1. A single address (fetches transactions via Etherscan, returns feature dict)
2. The Kaggle dataset (uses pre-computed features, adds graph features where possible)
"""

import logging

import pandas as pd

from src.collector.etherscan import EtherscanClient
from src.features.graph import extract_graph_features
from src.features.statistical import extract_statistical_features
from src.features.transaction import extract_transaction_features

logger = logging.getLogger(__name__)


def extract_features_for_address(
    address: str,
    client: EtherscanClient | None = None,
) -> dict[str, float]:
    """Extract all features for a single Ethereum address.

    Fetches transaction data from Etherscan and runs all three
    feature extractors: transaction, graph, and statistical.

    Args:
        address: Ethereum address to analyze.
        client: Optional EtherscanClient instance. Creates one if not provided.

    Returns:
        Dict of all features merged together.
    """
    if client is None:
        client = EtherscanClient()

    logger.info(f"Fetching transactions for {address}")
    txn_data = client.get_all_transactions(address)

    # Use normal transactions as the primary dataset
    normal_txns = txn_data.get("normal", [])
    logger.info(f"Got {len(normal_txns)} normal transactions")

    # Extract features from each module
    txn_features = extract_transaction_features(normal_txns, address)
    graph_features = extract_graph_features(normal_txns, address)
    stat_features = extract_statistical_features(normal_txns, address)

    # Merge all features
    features = {}
    features.update(txn_features)
    features.update(graph_features)
    features.update(stat_features)

    return features


def extract_features_from_dataset(
    df: pd.DataFrame,
    label_column: str = "flag",
) -> pd.DataFrame:
    """Build a feature matrix from the Kaggle dataset.

    The Kaggle dataset already has pre-computed features per address.
    This function selects and renames the relevant columns to match
    our feature naming convention.

    Args:
        df: Cleaned Kaggle dataset DataFrame.
        label_column: Name of the fraud label column.

    Returns:
        DataFrame with features and label column.
    """
    # Map Kaggle dataset columns to our feature names
    # The Kaggle dataset uses different naming, so we map what we can
    kaggle_feature_map = {
        # Transaction features
        "avg_val_received": "avg_value_received",
        "avg_val_sent": "avg_value_sent",
        "total_ether_received": "total_ether_received",
        "total_ether_sent": "total_ether_sent",
        "total_ether_balance": "total_ether_balance",
        "total_transactions_including_tnx_to_create_contract": "total_transactions",
        "sent_tnx": "sent_transactions",
        "received_tnx": "received_transactions",
        "number_of_unique_received_from_addresses": "unique_received_from_addresses",
        "number_of_unique_sent_to_addresses": "unique_sent_to_addresses",
        "min_value_received": "min_value_received",
        "max_value_received": "max_value_received",
        "min_val_sent": "min_value_sent",
        "max_val_sent": "max_value_sent",
        "avg_min_between_sent_tnx": "avg_minutes_between_sent_tnx",
        "avg_min_between_received_tnx": "avg_minutes_between_received_tnx",
        "time_diff_between_first_and_last_mins": "time_span_minutes",
        # ERC20 features (already in dataset)
        "total_erc20_tnxs": "total_erc20_transactions",
        "erc20_total_ether_received": "erc20_total_ether_received",
        "erc20_total_ether_sent": "erc20_total_ether_sent",
        "erc20_uniq_sent_addr": "erc20_unique_sent_addresses",
        "erc20_uniq_rec_addr": "erc20_unique_received_addresses",
        "erc20_avg_val_rec": "erc20_avg_value_received",
        "erc20_avg_val_sent": "erc20_avg_value_sent",
        "erc20_uniq_sent_token_name": "erc20_unique_sent_tokens",
        "erc20_uniq_rec_token_name": "erc20_unique_received_tokens",
        "erc20_most_sent_token_type": "erc20_most_sent_token",
        "erc20_most_rec_token_type": "erc20_most_received_token",
    }

    # Find which columns actually exist in the dataset
    available_features = {}
    for kaggle_col, our_col in kaggle_feature_map.items():
        if kaggle_col in df.columns:
            available_features[kaggle_col] = our_col

    # Select and rename columns
    feature_cols = list(available_features.keys())
    if label_column in df.columns:
        feature_cols.append(label_column)

    result = df[feature_cols].copy()
    result = result.rename(columns=available_features)

    # Add derived features we can compute from existing columns
    if "sent_transactions" in result.columns and "received_transactions" in result.columns:
        result["sent_received_ratio"] = (
            result["sent_transactions"] / result["received_transactions"].replace(0, 1)
        )
        result["total_unique_addresses"] = (
            result.get("unique_sent_to_addresses", 0)
            + result.get("unique_received_from_addresses", 0)
        )

    logger.info(
        f"Feature matrix: {result.shape[0]} addresses, "
        f"{result.shape[1] - (1 if label_column in result.columns else 0)} features"
    )

    return result


def get_feature_names() -> list[str]:
    """Return the full list of feature names produced by the live pipeline.

    Useful for aligning the Kaggle-trained model with live predictions.
    """
    from src.features.graph import _empty_graph_features
    from src.features.statistical import _empty_statistical_features
    from src.features.transaction import _empty_transaction_features

    names = []
    names.extend(_empty_transaction_features().keys())
    names.extend(_empty_graph_features().keys())
    names.extend(_empty_statistical_features().keys())
    return names


if __name__ == "__main__":
    import json

    # Quick test: extract features for a known address
    test_address = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"  # vitalik.eth
    print(f"Extracting features for {test_address}...")

    features = extract_features_for_address(test_address)
    print(f"\nTotal features: {len(features)}")
    print(json.dumps(features, indent=2, default=str))
