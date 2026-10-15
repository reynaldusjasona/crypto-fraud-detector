"""Tests for the feature engineering modules.

Uses small synthetic transaction lists so tests run offline and
deterministically. No Etherscan calls.
"""

from src.features.graph import (
    build_transaction_graph,
    extract_graph_features,
)
from src.features.statistical import extract_statistical_features
from src.features.transaction import extract_transaction_features

ADDR = "0xaaaa000000000000000000000000000000000001"
B = "0xbbbb000000000000000000000000000000000002"
C = "0xcccc000000000000000000000000000000000003"

ONE_ETH = str(10**18)
TWO_ETH = str(2 * 10**18)


def _tx(frm, to, value_wei, ts):
    return {
        "from": frm,
        "to": to,
        "value": value_wei,
        "timeStamp": str(ts),
        "gasUsed": "21000",
        "gasPrice": "1000000000",
    }


# --- transaction features ---------------------------------------------------

def test_transaction_empty_returns_zeros():
    feats = extract_transaction_features([], ADDR)
    assert feats["total_transactions"] == 0
    assert feats["total_ether_sent"] == 0.0
    assert feats["total_unique_addresses"] == 0


def test_transaction_counts_and_values():
    txns = [
        _tx(ADDR, B, ONE_ETH, 1_000),   # sent 1 ETH
        _tx(ADDR, C, ONE_ETH, 2_000),   # sent 1 ETH
        _tx(B, ADDR, TWO_ETH, 3_000),   # received 2 ETH
    ]
    feats = extract_transaction_features(txns, ADDR)
    assert feats["total_transactions"] == 3
    assert feats["sent_transactions"] == 2
    assert feats["received_transactions"] == 1
    assert feats["total_ether_sent"] == 2.0
    assert feats["total_ether_received"] == 2.0
    assert feats["unique_sent_to_addresses"] == 2
    assert feats["unique_received_from_addresses"] == 1


def test_transaction_time_span():
    txns = [
        _tx(ADDR, B, ONE_ETH, 0),
        _tx(ADDR, C, ONE_ETH, 600),  # 600s = 10 minutes later
    ]
    feats = extract_transaction_features(txns, ADDR)
    assert feats["time_span_minutes"] == 10.0


# --- graph features ---------------------------------------------------------

def test_graph_build_merges_parallel_edges():
    txns = [
        _tx(ADDR, B, ONE_ETH, 1),
        _tx(ADDR, B, ONE_ETH, 2),  # same edge again
    ]
    g = build_transaction_graph(txns)
    assert g.has_edge(ADDR.lower(), B.lower())
    assert g[ADDR.lower()][B.lower()]["count"] == 2
    assert g[ADDR.lower()][B.lower()]["weight"] == 2.0


def test_graph_empty_returns_zeros():
    feats = extract_graph_features([], ADDR)
    assert feats["in_degree"] == 0
    assert feats["out_degree"] == 0
    assert feats["has_self_loop"] == 0


def test_graph_self_loop_detected():
    txns = [_tx(ADDR, ADDR, ONE_ETH, 1)]
    feats = extract_graph_features(txns, ADDR)
    assert feats["has_self_loop"] == 1
    assert feats["self_loop_count"] == 1


def test_graph_circular_path_detected():
    # A -> B -> A is a 2-hop circle
    txns = [
        _tx(ADDR, B, ONE_ETH, 1),
        _tx(B, ADDR, ONE_ETH, 2),
    ]
    feats = extract_graph_features(txns, ADDR)
    assert feats["circular_path_count"] >= 1
    assert feats["in_degree"] == 1
    assert feats["out_degree"] == 1


def test_graph_degree_counts():
    txns = [
        _tx(ADDR, B, ONE_ETH, 1),
        _tx(ADDR, C, ONE_ETH, 2),
        _tx(B, ADDR, ONE_ETH, 3),
    ]
    feats = extract_graph_features(txns, ADDR)
    assert feats["out_degree"] == 2
    assert feats["in_degree"] == 1
    assert feats["num_neighbors"] == 2


# --- statistical features ---------------------------------------------------

def test_statistical_empty_returns_zeros():
    feats = extract_statistical_features([], ADDR)
    assert feats["value_skewness"] == 0.0
    assert feats["burst_count"] == 0
    assert feats["hour_entropy"] == 0.0


def test_statistical_zero_value_ratio():
    txns = [
        _tx(ADDR, B, "0", 1),
        _tx(ADDR, C, ONE_ETH, 2),
    ]
    feats = extract_statistical_features(txns, ADDR)
    assert feats["zero_value_txn_ratio"] == 0.5


def test_statistical_returns_all_expected_keys():
    txns = [_tx(ADDR, B, ONE_ETH, i * 100) for i in range(10)]
    feats = extract_statistical_features(txns, ADDR)
    # A representative set of keys the pipeline depends on
    for key in [
        "value_skewness",
        "value_kurtosis",
        "hour_entropy",
        "counterparty_concentration",
        "burst_count",
    ]:
        assert key in feats
