"""Extract graph-based features from Ethereum transaction data using NetworkX.

Graph features capture structural relationships between wallets:
who transacts with whom, how central a wallet is in the network,
and whether circular or self-referencing patterns exist.
These are the features that distinguish this project from basic
tabular ML approaches.
"""

import networkx as nx
import numpy as np


def build_transaction_graph(txns: list[dict]) -> nx.DiGraph:
    """Build a directed graph from raw transaction data.

    Each node is an Ethereum address. Each edge represents at least
    one transaction between two addresses, weighted by total ETH
    transferred and annotated with transaction count.

    Args:
        txns: List of transaction dicts from Etherscan API.

    Returns:
        Directed graph with weighted edges.
    """
    G = nx.DiGraph()

    for tx in txns:
        sender = tx["from"].lower()
        receiver = tx["to"].lower() if tx.get("to") else None

        if receiver is None:
            # Contract creation transaction, skip
            continue

        value_eth = float(tx.get("value", 0)) / 1e18

        if G.has_edge(sender, receiver):
            G[sender][receiver]["weight"] += value_eth
            G[sender][receiver]["count"] += 1
        else:
            G.add_edge(sender, receiver, weight=value_eth, count=1)

    return G


def extract_graph_features(txns: list[dict], address: str) -> dict[str, float]:
    """Compute graph-based features for a single address.

    Args:
        txns: List of transaction dicts from Etherscan API.
        address: The Ethereum address being analyzed.

    Returns:
        Dict of feature_name -> value.
    """
    if not txns:
        return _empty_graph_features()

    G = build_transaction_graph(txns)
    addr = address.lower()

    if addr not in G:
        return _empty_graph_features()

    # Degree features
    in_degree = G.in_degree(addr)
    out_degree = G.out_degree(addr)
    total_degree = in_degree + out_degree

    # Weighted degree (total ETH flowing in/out)
    in_weight = sum(d["weight"] for _, _, d in G.in_edges(addr, data=True))
    out_weight = sum(d["weight"] for _, _, d in G.out_edges(addr, data=True))

    # Degree ratio: high out vs in can indicate draining behavior
    degree_ratio = out_degree / max(in_degree, 1)

    # Clustering coefficient
    # For directed graphs, use the undirected version for clustering
    G_undirected = G.to_undirected()
    clustering_coeff = nx.clustering(G_undirected, addr)

    # PageRank: how "important" is this address in the transaction network
    try:
        pagerank_scores = nx.pagerank(G, alpha=0.85, max_iter=100)
        pagerank = pagerank_scores.get(addr, 0.0)
    except nx.PowerIterationFailedConvergence:
        pagerank = 0.0

    # Self-loops: address sending ETH to itself (wash trading indicator)
    self_loop = 1 if G.has_edge(addr, addr) else 0
    self_loop_count = G[addr][addr]["count"] if self_loop else 0

    # Circular paths: A -> B -> A patterns (another wash trading indicator)
    circular_count = _count_circular_paths(G, addr)

    # Neighbor statistics
    neighbors = set(G.successors(addr)) | set(G.predecessors(addr))
    neighbors.discard(addr)  # exclude self
    num_neighbors = len(neighbors)

    # Average neighbor degree
    avg_neighbor_degree_val = 0.0
    if num_neighbors > 0:
        neighbor_degrees = [G.degree(n) for n in neighbors]
        avg_neighbor_degree_val = np.mean(neighbor_degrees)

    # Community detection via Louvain
    # Louvain works on undirected graphs
    try:
        communities = nx.community.louvain_communities(G_undirected, seed=42)
        community_id = -1
        community_size = 0
        for i, comm in enumerate(communities):
            if addr in comm:
                community_id = i
                community_size = len(comm)
                break
    except Exception:
        community_id = -1
        community_size = 0

    # Hub and authority scores (HITS algorithm)
    try:
        hubs, authorities = nx.hits(G, max_iter=100)
        hub_score = hubs.get(addr, 0.0)
        authority_score = authorities.get(addr, 0.0)
    except nx.PowerIterationFailedConvergence:
        hub_score = 0.0
        authority_score = 0.0

    return {
        "in_degree": in_degree,
        "out_degree": out_degree,
        "total_degree": total_degree,
        "degree_ratio": degree_ratio,
        "in_weight_eth": in_weight,
        "out_weight_eth": out_weight,
        "clustering_coefficient": clustering_coeff,
        "pagerank": pagerank,
        "has_self_loop": self_loop,
        "self_loop_count": self_loop_count,
        "circular_path_count": circular_count,
        "num_neighbors": num_neighbors,
        "avg_neighbor_degree": avg_neighbor_degree_val,
        "community_id": community_id,
        "community_size": community_size,
        "hub_score": hub_score,
        "authority_score": authority_score,
    }


def _count_circular_paths(G: nx.DiGraph, addr: str, max_depth: int = 2) -> int:
    """Count short circular paths (A -> ... -> A) up to max_depth hops.

    Looks for patterns where ETH leaves the address and comes back
    within a few hops. Common in wash trading and layering schemes.

    Args:
        G: Transaction graph.
        addr: Address to check.
        max_depth: Maximum path length to search (default 2 hops).

    Returns:
        Number of circular paths found.
    """
    count = 0
    successors = list(G.successors(addr))

    for succ in successors:
        if succ == addr:
            continue  # self-loop counted separately

        if max_depth >= 1 and G.has_edge(succ, addr):
            # A -> B -> A (2-hop circle)
            count += 1

        if max_depth >= 2:
            # A -> B -> C -> A (3-hop circle)
            for succ2 in G.successors(succ):
                if succ2 == addr:
                    continue
                if G.has_edge(succ2, addr):
                    count += 1

    return count


def _empty_graph_features() -> dict[str, float]:
    """Return zeroed features when no graph data exists."""
    return {
        "in_degree": 0,
        "out_degree": 0,
        "total_degree": 0,
        "degree_ratio": 0.0,
        "in_weight_eth": 0.0,
        "out_weight_eth": 0.0,
        "clustering_coefficient": 0.0,
        "pagerank": 0.0,
        "has_self_loop": 0,
        "self_loop_count": 0,
        "circular_path_count": 0,
        "num_neighbors": 0,
        "avg_neighbor_degree": 0.0,
        "community_id": -1,
        "community_size": 0,
        "hub_score": 0.0,
        "authority_score": 0.0,
    }
