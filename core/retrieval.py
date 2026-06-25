"""BM25 + vector cosine + graph-hop retrieval with token-budget enforcement.

Scoring weights (from architecture.md):
    vector : 0.5
    bm25   : 0.3
    graph  : 0.2

Tier-3 (convention) nodes are always included regardless of score.
Remaining slots are filled by top-k candidates (default k=8) up to 600 tokens.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from rank_bm25 import BM25Okapi

if TYPE_CHECKING:
    import tiktoken

from core.embedder import cosine_similarity, embed
from core.graph import Graph, Node

logger = logging.getLogger(__name__)

VECTOR_WEIGHT: float = 0.5
BM25_WEIGHT: float = 0.3
GRAPH_WEIGHT: float = 0.2

TOP_K: int = 8
TOKEN_BUDGET: int = 600


@dataclass
class ScoredNode:
    """A graph node paired with its retrieval score."""

    node: Node
    score: float


def build_bm25_index(nodes: list[Node]) -> BM25Okapi:
    """Build a BM25Okapi index from node texts.

    Args:
        nodes: List of nodes whose texts form the corpus.

    Returns:
        A fitted BM25Okapi instance over the node texts.
    """
    tokenized = [node.text.lower().split() for node in nodes]
    return BM25Okapi(tokenized)


def vector_channel(
    query_embedding: np.ndarray,
    nodes: list[Node],
) -> list[ScoredNode]:
    """Score nodes by cosine similarity with the query embedding.

    Nodes without an embedding receive a score of 0.0.

    Args:
        query_embedding: Float32 query vector (384-dim).
        nodes: Candidate nodes to score.

    Returns:
        One ScoredNode per input node, ordered to match input order.
    """
    results: list[ScoredNode] = []
    for node in nodes:
        if node.embedding is not None:
            sim = cosine_similarity(query_embedding, node.embedding)
            results.append(ScoredNode(node=node, score=max(0.0, float(sim))))
        else:
            results.append(ScoredNode(node=node, score=0.0))
    return results


def bm25_channel(
    query: str,
    nodes: list[Node],
    index: BM25Okapi,
) -> list[ScoredNode]:
    """Score nodes using BM25 keyword matching, normalised to [0, 1].

    Args:
        query: Raw query string; tokenised by whitespace.
        nodes: Same nodes that were used to build the index (same order).
        index: Pre-built BM25Okapi index.

    Returns:
        One ScoredNode per input node in the same order as nodes.
    """
    tokens = query.lower().split()
    raw = index.get_scores(tokens)
    max_score = float(np.max(raw)) if raw.size > 0 else 0.0
    results: list[ScoredNode] = []
    for i, node in enumerate(nodes):
        score = float(raw[i]) / max_score if max_score > 0.0 else 0.0
        results.append(ScoredNode(node=node, score=score))
    return results


def graph_channel(
    seed_nodes: list[Node],
    graph: Graph,
    all_nodes: list[Node],
) -> list[ScoredNode]:
    """Score nodes by proximity to seed nodes in the edge graph.

    Traversal is limited to 2 hops:
    - 1-hop neighbour: score 0.5
    - 2-hop neighbour: score 0.25
    - Seed nodes themselves: score 0.0 (handled by other channels)
    - Unreachable nodes: score 0.0

    Uses get_edges_for_nodes() to fetch all required edges in two SQL queries
    instead of one query per node.

    Args:
        seed_nodes: Starting nodes (typically top-3 by vector score).
        graph: Graph instance for edge lookups.
        all_nodes: Full candidate set to score.

    Returns:
        One ScoredNode per node in all_nodes, in the same order.
    """
    seed_ids = {n.id for n in seed_nodes}
    node_map = {n.id: n for n in all_nodes}
    hop_scores: dict[str, float] = {}

    # Single query for all seed edges
    seed_edge_map = graph.get_edges_for_nodes([n.id for n in seed_nodes])

    for seed in seed_nodes:
        for edge in seed_edge_map.get(seed.id, []):
            neighbor_id = (
                edge.target_id if edge.source_id == seed.id else edge.source_id
            )
            if neighbor_id not in seed_ids:
                hop_scores[neighbor_id] = max(hop_scores.get(neighbor_id, 0.0), 0.5)

    hop1_ids = [nid for nid in hop_scores if nid in node_map]

    # Single query for all hop-1 edges
    hop1_edge_map = graph.get_edges_for_nodes(hop1_ids)

    for hop1_id in hop1_ids:
        for edge in hop1_edge_map.get(hop1_id, []):
            neighbor_id = (
                edge.target_id if edge.source_id == hop1_id else edge.source_id
            )
            if neighbor_id not in seed_ids and neighbor_id not in hop_scores:
                hop_scores[neighbor_id] = 0.25

    return [
        ScoredNode(node=node, score=hop_scores.get(node.id, 0.0)) for node in all_nodes
    ]


def fuse_scores(
    vector_scores: list[ScoredNode],
    bm25_scores: list[ScoredNode],
    graph_scores: list[ScoredNode],
) -> list[ScoredNode]:
    """Fuse three channel scores into a single ranked list.

    Formula: score = 0.5 * vector + 0.3 * BM25 + 0.2 * graph

    All three input lists must cover the same set of nodes (same ids).

    Args:
        vector_scores: Output of vector_channel.
        bm25_scores: Output of bm25_channel.
        graph_scores: Output of graph_channel.

    Returns:
        ScoredNodes sorted by fused score descending.
    """
    fused: list[ScoredNode] = [
        ScoredNode(
            node=v.node,
            score=VECTOR_WEIGHT * v.score
            + BM25_WEIGHT * b.score
            + GRAPH_WEIGHT * g.score,
        )
        for v, b, g in zip(vector_scores, bm25_scores, graph_scores, strict=True)
    ]
    return sorted(fused, key=lambda sn: sn.score, reverse=True)


@lru_cache(maxsize=1)
def _get_tokenizer() -> tiktoken.Encoding:
    """Lazy-load the cl100k_base tiktoken encoder (cached after first call)."""
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count cl100k_base tokens in a string.

    Args:
        text: Input string to tokenize.

    Returns:
        Number of cl100k_base tokens.
    """
    return len(_get_tokenizer().encode(text))


def _enforce_budget(
    tier3: list[Node],
    fused_candidates: list[ScoredNode],
    budget_tokens: int,
) -> list[Node]:
    """Select nodes up to the token budget.

    Tier-3 nodes are always included first. Remaining budget is filled by
    scored candidates in descending score order.

    Args:
        tier3: Convention nodes (always kept).
        fused_candidates: Scored non-tier-3 nodes, descending by score.
        budget_tokens: Hard upper limit on total text tokens.

    Returns:
        List of nodes within the token budget.
    """
    result: list[Node] = []
    used_tokens = 0
    tier3_ids = {n.id for n in tier3}

    for node in tier3:
        tokens = count_tokens(node.text)
        result.append(node)
        used_tokens += tokens

    for sn in fused_candidates:
        if sn.node.id in tier3_ids:
            continue
        tokens = count_tokens(sn.node.text)
        if used_tokens + tokens <= budget_tokens:
            result.append(sn.node)
            used_tokens += tokens

    return result


def retrieve(
    query: str,
    project: str,
    graph: Graph,
    budget_tokens: int = TOKEN_BUDGET,
) -> list[Node]:
    """Retrieve the most relevant nodes for a query within the token budget.

    Algorithm:
    1. All tier-3 (convention) nodes are always included.
    2. Tier-1/2 candidates are ranked by fused score (vector + BM25 + graph-hop).
    3. Top-k candidates (default 8) are kept.
    4. The combined set is trimmed to budget_tokens using tiktoken.

    Args:
        query: Natural-language description of the current task.
        project: Absolute project path to scope the retrieval.
        graph: Graph instance to query.
        budget_tokens: Token ceiling for the injected context block.

    Returns:
        Ordered list of nodes: tier-3 first, then scored candidates.
    """
    all_nodes = graph.get_all_nodes(project=project)
    if not all_nodes:
        return []

    tier3 = [n for n in all_nodes if n.tier == 3]
    candidates = [n for n in all_nodes if n.tier < 3]

    if not candidates:
        return list(tier3)

    query_embedding = embed(query)
    index = build_bm25_index(candidates)

    v_scores = vector_channel(query_embedding, candidates)
    b_scores = bm25_channel(query, candidates, index)

    top3_seeds = [
        sn.node for sn in sorted(v_scores, key=lambda s: s.score, reverse=True)[:3]
    ]
    g_scores = graph_channel(top3_seeds, graph, candidates)

    fused = fuse_scores(v_scores, b_scores, g_scores)
    top_fused = fused[:TOP_K]

    return _enforce_budget(tier3, top_fused, budget_tokens)


def format_injection_block(
    nodes: list[Node],
    project: str,
    budget_tokens: int = TOKEN_BUDGET,
) -> str:
    """Format retrieved nodes as the Cortex injection block.

    Tier-3 nodes appear in a [CONVENTIONS] section; all others in [RELEVANT CONTEXT].
    The footer includes the actual token count and the active budget ceiling.

    Args:
        nodes: Nodes to format (typically the output of retrieve()).
        project: Absolute project path shown in the header.
        budget_tokens: Token ceiling used during retrieval; shown in the footer
            so callers can verify how close to the limit the block is.

    Returns:
        A human-readable multi-line string ready for stdout injection.
    """
    conventions = [n for n in nodes if n.tier == 3]
    context = [n for n in nodes if n.tier < 3]

    lines: list[str] = [f"=== CORTEX MEMORY (PROJECT: {project}) ==="]

    if conventions:
        lines.append("")
        lines.append("[CONVENTIONS — always apply]")
        for node in conventions:
            line = f"• {node.text}"
            if node.rationale:
                line += f" ({node.rationale})"
            lines.append(line)

    if context:
        lines.append("")
        lines.append("[RELEVANT CONTEXT — this session]")
        for node in context:
            line = f"• {node.text}"
            if node.rationale:
                line += f" ({node.rationale})"
            lines.append(line)

    body = "\n".join(lines)
    token_count = count_tokens(body)
    footer = (
        f"\n=== END CORTEX MEMORY (tokens: {token_count} / budget: {budget_tokens}) ==="
    )
    return body + footer
