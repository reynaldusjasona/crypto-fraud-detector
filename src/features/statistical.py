"""Extract statistical pattern features from transaction data.

These features capture behavioral patterns that are hard to see
from raw counts alone: how regular or bursty the activity is,
how skewed the value distribution is, and whether the wallet
operates at unusual hours.
"""

from collections import Counter

import numpy as np
import pandas as pd


def _skewness(arr: np.ndarray) -> float:
    """Compute skewness without scipy."""
    n = len(arr)
    if n < 3:
        return 0.0
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    if std == 0:
        return 0.0
    return float((n / ((n - 1) * (n - 2))) * np.sum(((arr - mean) / std) ** 3))


def _kurtosis(arr: np.ndarray) -> float:
    """Compute excess kurtosis without scipy."""
    n = len(arr)
    if n < 4:
        return 0.0
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    if std == 0:
        return 0.0
    m4 = np.mean((arr - mean) ** 4)
    return float(m4 / (std ** 4) - 3.0)


def _entropy(probs: np.ndarray) -> float:
    """Compute Shannon entropy without scipy."""
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    return float(-np.sum(probs * np.log(probs)))


def extract_statistical_features(txns: list[dict], address: str) -> dict[str, float]:
    """Compute statistical pattern features for a single address.

    Args:
        txns: List of transaction dicts from Etherscan API.
        address: The Ethereum address being analyzed.

    Returns:
        Dict of feature_name -> value.
    """
    if not txns:
        return _empty_statistical_features()

    df = _prepare_dataframe(txns, address)

    if df.empty:
        return _empty_statistical_features()

    features = {}

    # Value distribution features
    features.update(_value_distribution_features(df))

    # Temporal pattern features
    features.update(_temporal_features(df))

    # Counterparty pattern features
    features.update(_counterparty_features(df, address))

    # Burst detection features
    features.update(_burst_features(df))

    return features


def _prepare_dataframe(txns: list[dict], address: str) -> pd.DataFrame:
    """Convert raw transactions to a DataFrame with derived columns."""
    df = pd.DataFrame(txns)

    df["value_eth"] = df["value"].astype(float) / 1e18
    df["timestamp"] = df["timeStamp"].astype(int)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek  # 0=Monday

    addr_lower = address.lower()
    df["direction"] = df.apply(
        lambda r: "sent" if r["from"].lower() == addr_lower else "received",
        axis=1,
    )

    return df.sort_values("timestamp").reset_index(drop=True)


def _value_distribution_features(df: pd.DataFrame) -> dict[str, float]:
    """Features from the distribution of transaction values."""
    values = df["value_eth"]

    # Filter out zero-value transactions for distribution stats
    nonzero = values[values > 0].values

    skewness = _skewness(nonzero)
    kurtosis_val = _kurtosis(nonzero)

    # Coefficient of variation: std / mean
    cv = 0.0
    if len(nonzero) > 1 and np.mean(nonzero) > 0:
        cv = float(np.std(nonzero, ddof=1) / np.mean(nonzero))

    # Percentile features
    p25, p50, p75 = 0.0, 0.0, 0.0
    if len(nonzero) > 0:
        p25 = float(np.percentile(nonzero, 25))
        p50 = float(np.percentile(nonzero, 50))
        p75 = float(np.percentile(nonzero, 75))

    # Ratio of zero-value transactions (common in token approvals, spam)
    zero_ratio = (values == 0).sum() / max(len(values), 1)

    return {
        "value_skewness": skewness,
        "value_kurtosis": kurtosis_val,
        "value_coefficient_of_variation": cv,
        "value_p25": p25,
        "value_p50_median": p50,
        "value_p75": p75,
        "zero_value_txn_ratio": zero_ratio,
    }


def _temporal_features(df: pd.DataFrame) -> dict[str, float]:
    """Features from transaction timing patterns."""
    # Hour-of-day distribution entropy
    hour_counts = df["hour"].value_counts()
    hour_probs = (hour_counts / hour_counts.sum()).values
    hour_entropy = _entropy(hour_probs)

    # Day-of-week distribution entropy
    dow_counts = df["day_of_week"].value_counts()
    dow_probs = (dow_counts / dow_counts.sum()).values
    dow_entropy = _entropy(dow_probs)

    # Peak hour (most active hour)
    peak_hour = int(hour_counts.idxmax())

    # Night activity ratio (UTC 0-6)
    night_txns = df[df["hour"].between(0, 5)]
    night_ratio = len(night_txns) / max(len(df), 1)

    # Weekend activity ratio
    weekend_txns = df[df["day_of_week"].isin([5, 6])]
    weekend_ratio = len(weekend_txns) / max(len(df), 1)

    # Inter-transaction time statistics
    timestamps = df["timestamp"].values
    if len(timestamps) > 1:
        diffs = np.diff(timestamps).astype(float)
        itx_std = float(np.std(diffs)) / 60.0  # minutes
        itx_median = float(np.median(diffs)) / 60.0
        # Coefficient of variation for inter-tx time
        mean_diff = np.mean(diffs)
        itx_cv = float(np.std(diffs) / max(mean_diff, 1))
    else:
        itx_std = 0.0
        itx_median = 0.0
        itx_cv = 0.0

    return {
        "hour_entropy": hour_entropy,
        "day_of_week_entropy": dow_entropy,
        "peak_hour": peak_hour,
        "night_activity_ratio": night_ratio,
        "weekend_activity_ratio": weekend_ratio,
        "inter_txn_time_std_minutes": itx_std,
        "inter_txn_time_median_minutes": itx_median,
        "inter_txn_time_cv": itx_cv,
    }


def _counterparty_features(df: pd.DataFrame, address: str) -> dict[str, float]:
    """Features from counterparty interaction patterns."""
    addr_lower = address.lower()

    sent = df[df["direction"] == "sent"]
    received = df[df["direction"] == "received"]

    # Concentration: does the wallet interact with many addresses or few?
    all_counterparties = []
    if not sent.empty:
        all_counterparties.extend(sent["to"].str.lower().tolist())
    if not received.empty:
        all_counterparties.extend(received["from"].str.lower().tolist())

    # Remove self-references
    all_counterparties = [c for c in all_counterparties if c != addr_lower]

    if not all_counterparties:
        return {
            "counterparty_concentration": 0.0,
            "top_counterparty_txn_ratio": 0.0,
            "unique_counterparty_ratio": 0.0,
            "repeat_counterparty_ratio": 0.0,
        }

    counter = Counter(all_counterparties)
    total_interactions = sum(counter.values())
    unique_counterparties = len(counter)

    # Concentration: Herfindahl index (sum of squared shares)
    shares = [c / total_interactions for c in counter.values()]
    concentration = sum(s ** 2 for s in shares)

    # Top counterparty ratio
    top_count = counter.most_common(1)[0][1]
    top_ratio = top_count / total_interactions

    # Unique counterparty ratio (unique / total interactions)
    unique_ratio = unique_counterparties / total_interactions

    # Repeat counterparty ratio (counterparties seen more than once)
    repeat_count = sum(1 for c in counter.values() if c > 1)
    repeat_ratio = repeat_count / max(unique_counterparties, 1)

    return {
        "counterparty_concentration": concentration,
        "top_counterparty_txn_ratio": top_ratio,
        "unique_counterparty_ratio": unique_ratio,
        "repeat_counterparty_ratio": repeat_ratio,
    }


def _burst_features(df: pd.DataFrame) -> dict[str, float]:
    """Detect burst patterns: sudden spikes in transaction activity.

    Bursts can indicate coordinated activity, pump-and-dump exits,
    or automated bot behavior.
    """
    if len(df) < 5:
        return {
            "burst_count": 0,
            "max_burst_size": 0,
            "burst_txn_ratio": 0.0,
        }

    # Group transactions into 1-hour windows
    df = df.copy()
    df["hour_bin"] = df["datetime"].dt.floor("h")
    hourly_counts = df.groupby("hour_bin").size()

    if len(hourly_counts) < 2:
        return {
            "burst_count": 0,
            "max_burst_size": 0,
            "burst_txn_ratio": 0.0,
        }

    # A burst is any hour with more than 2 standard deviations above mean
    mean_rate = hourly_counts.mean()
    std_rate = hourly_counts.std()
    threshold = mean_rate + 2 * std_rate

    bursts = hourly_counts[hourly_counts > threshold]
    burst_count = len(bursts)
    max_burst_size = int(bursts.max()) if not bursts.empty else 0
    burst_txn_count = bursts.sum() if not bursts.empty else 0
    burst_txn_ratio = burst_txn_count / len(df)

    return {
        "burst_count": burst_count,
        "max_burst_size": max_burst_size,
        "burst_txn_ratio": burst_txn_ratio,
    }


def _empty_statistical_features() -> dict[str, float]:
    """Return zeroed features when no data exists."""
    return {
        "value_skewness": 0.0,
        "value_kurtosis": 0.0,
        "value_coefficient_of_variation": 0.0,
        "value_p25": 0.0,
        "value_p50_median": 0.0,
        "value_p75": 0.0,
        "zero_value_txn_ratio": 0.0,
        "hour_entropy": 0.0,
        "day_of_week_entropy": 0.0,
        "peak_hour": 0,
        "night_activity_ratio": 0.0,
        "weekend_activity_ratio": 0.0,
        "inter_txn_time_std_minutes": 0.0,
        "inter_txn_time_median_minutes": 0.0,
        "inter_txn_time_cv": 0.0,
        "counterparty_concentration": 0.0,
        "top_counterparty_txn_ratio": 0.0,
        "unique_counterparty_ratio": 0.0,
        "repeat_counterparty_ratio": 0.0,
        "burst_count": 0,
        "max_burst_size": 0,
        "burst_txn_ratio": 0.0,
    }
