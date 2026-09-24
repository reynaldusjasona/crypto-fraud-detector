"""Extract transaction-level features from raw Etherscan transaction data.

These features capture basic wallet behavior: how much ETH moves,
how often, and over what time span.
"""

from datetime import datetime

import numpy as np
import pandas as pd


def _txns_to_dataframe(txns: list[dict], address: str) -> pd.DataFrame:
    """Convert raw Etherscan transaction list to a cleaned DataFrame."""
    if not txns:
        return pd.DataFrame()

    df = pd.DataFrame(txns)

    # Convert types
    df["value_eth"] = df["value"].astype(float) / 1e18
    df["timestamp"] = df["timeStamp"].astype(int)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df["gas_used"] = pd.to_numeric(df.get("gasUsed", 0), errors="coerce").fillna(0)
    df["gas_price"] = pd.to_numeric(df.get("gasPrice", 0), errors="coerce").fillna(0)

    # Tag direction
    addr_lower = address.lower()
    df["direction"] = df.apply(
        lambda r: "sent" if r["from"].lower() == addr_lower else "received",
        axis=1,
    )

    return df


def extract_transaction_features(txns: list[dict], address: str) -> dict[str, float]:
    """Compute transaction-level features for a single address.

    Args:
        txns: List of transaction dicts from Etherscan API.
        address: The Ethereum address being analyzed.

    Returns:
        Dict of feature_name -> value.
    """
    df = _txns_to_dataframe(txns, address)

    if df.empty:
        return _empty_transaction_features()

    sent = df[df["direction"] == "sent"]
    received = df[df["direction"] == "received"]

    # Time features
    time_span_minutes = 0.0
    avg_time_between_txns = 0.0
    if len(df) > 1:
        timestamps = df["timestamp"].sort_values()
        time_span_minutes = (timestamps.iloc[-1] - timestamps.iloc[0]) / 60.0
        diffs = timestamps.diff().dropna()
        avg_time_between_txns = diffs.mean() / 60.0  # in minutes

    # Unique counterparties
    unique_sent_to = sent["to"].nunique() if not sent.empty else 0
    unique_received_from = received["from"].nunique() if not received.empty else 0

    return {
        # Counts
        "total_transactions": len(df),
        "sent_transactions": len(sent),
        "received_transactions": len(received),
        "sent_received_ratio": len(sent) / max(len(received), 1),
        # Values (ETH)
        "total_ether_sent": sent["value_eth"].sum() if not sent.empty else 0.0,
        "total_ether_received": received["value_eth"].sum() if not received.empty else 0.0,
        "avg_value_sent": sent["value_eth"].mean() if not sent.empty else 0.0,
        "avg_value_received": received["value_eth"].mean() if not received.empty else 0.0,
        "max_value_sent": sent["value_eth"].max() if not sent.empty else 0.0,
        "max_value_received": received["value_eth"].max() if not received.empty else 0.0,
        "min_value_sent": sent["value_eth"].min() if not sent.empty else 0.0,
        "min_value_received": received["value_eth"].min() if not received.empty else 0.0,
        # Time
        "time_span_minutes": time_span_minutes,
        "avg_minutes_between_txns": avg_time_between_txns,
        # Counterparties
        "unique_sent_to_addresses": unique_sent_to,
        "unique_received_from_addresses": unique_received_from,
        "total_unique_addresses": unique_sent_to + unique_received_from,
        # Gas
        "avg_gas_used": df["gas_used"].mean(),
        "avg_gas_price_gwei": df["gas_price"].mean() / 1e9,
    }


def _empty_transaction_features() -> dict[str, float]:
    """Return zeroed features when no transactions exist."""
    return {
        "total_transactions": 0,
        "sent_transactions": 0,
        "received_transactions": 0,
        "sent_received_ratio": 0.0,
        "total_ether_sent": 0.0,
        "total_ether_received": 0.0,
        "avg_value_sent": 0.0,
        "avg_value_received": 0.0,
        "max_value_sent": 0.0,
        "max_value_received": 0.0,
        "min_value_sent": 0.0,
        "min_value_received": 0.0,
        "time_span_minutes": 0.0,
        "avg_minutes_between_txns": 0.0,
        "unique_sent_to_addresses": 0,
        "unique_received_from_addresses": 0,
        "total_unique_addresses": 0,
        "avg_gas_used": 0.0,
        "avg_gas_price_gwei": 0.0,
    }
