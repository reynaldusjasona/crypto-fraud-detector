"""Tests for the data collector modules.

The Etherscan client is tested with requests mocked, so no real HTTP
calls are made. Dataset cleaning is tested with a tiny in-memory frame.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.collector.dataset import clean_dataset, get_dataset_summary
from src.collector.etherscan import EtherscanClient
from src.models.evaluate import _risk_level


# --- dataset cleaning -------------------------------------------------------

def _raw_frame():
    return pd.DataFrame(
        {
            "Unnamed: 0": [0, 1, 2],
            "Address": ["0xa", "0xb", "0xc"],
            "FLAG": [0, 1, 0],
            "Sent tnx": [10, 2, None],
            "Avg min between sent tnx": [5.0, None, 1.0],
        }
    )


def test_clean_drops_unnamed_and_lowercases():
    cleaned = clean_dataset(_raw_frame())
    assert not any("Unnamed" in c for c in cleaned.columns)
    assert "sent_tnx" in cleaned.columns
    assert "avg_min_between_sent_tnx" in cleaned.columns


def test_clean_flag_is_int_and_nan_filled():
    cleaned = clean_dataset(_raw_frame())
    assert cleaned["flag"].dtype == int
    # No NaN should remain after cleaning
    assert cleaned.isna().sum().sum() == 0


def test_dataset_summary_counts():
    cleaned = clean_dataset(_raw_frame())
    summary = get_dataset_summary(cleaned)
    assert summary["total_addresses"] == 3
    assert summary["fraud_count"] == 1
    assert summary["legitimate_count"] == 2
    assert 0.0 <= summary["fraud_ratio"] <= 1.0


# --- risk level thresholds --------------------------------------------------

@pytest.mark.parametrize(
    "prob,expected",
    [
        (0.95, "CRITICAL"),
        (0.7, "HIGH"),
        (0.5, "MEDIUM"),
        (0.3, "LOW"),
        (0.05, "MINIMAL"),
    ],
)
def test_risk_level_thresholds(prob, expected):
    assert _risk_level(prob) == expected


# --- Etherscan client (mocked) ----------------------------------------------

@patch("src.collector.etherscan.requests.get")
def test_client_returns_result_list(mock_get):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"status": "1", "result": [{"hash": "0x1"}]}
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    client = EtherscanClient(api_key="dummy")
    txns = client.get_normal_transactions("0xabc")

    assert txns == [{"hash": "0x1"}]
    assert mock_get.called


@patch("src.collector.etherscan.requests.get")
def test_client_no_transactions_is_not_an_error(mock_get):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "status": "0",
        "message": "No transactions found",
        "result": [],
    }
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    client = EtherscanClient(api_key="dummy")
    assert client.get_normal_transactions("0xabc") == []


@patch("src.collector.etherscan.requests.get")
def test_client_api_error_raises(mock_get):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "status": "0",
        "message": "NOTOK",
        "result": "Invalid API Key",
    }
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    client = EtherscanClient(api_key="dummy")
    with pytest.raises(Exception):
        client.get_normal_transactions("0xabc")


def test_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("ETHERSCAN_API_KEY", raising=False)
    with pytest.raises(Exception):
        EtherscanClient(api_key=None)
