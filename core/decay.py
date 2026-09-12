"""Weight decay, tier promotion, and eviction for the Cortex knowledge graph.

Decay constants live here; do not hardcode them in other modules.
Tier-3 nodes are NEVER decayed or evicted — only manual pruning via CLI.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from core.embedder import serialize
from core.graph import Graph, Node

# ---------------------------------------------------------------------------
# Decay constants (see CLAUDE.md for the authoritative table)
# ---------------------------------------------------------------------------

TIER1_DECAY_RATE: float = 0.85
TIER2_DECAY_RATE: float = 0.95

TIER1_EVICTION_THRESHOLD: float = 0.3
TIER2_EVICTION_THRESHOLD: float = 0.5

TIER1_PROMOTION_WEIGHT: float = 8.0
TIER1_PROMOTION_SESSIONS: int = 3

TIER2_PROMOTION_WEIGHT: float = 20.0
TIER2_PROMOTION_SESSIONS: int = 8
TIER2_PROMOTION_AGE_DAYS: int = 14

SECONDS_PER_DAY: int = 86_400

# Per-tier lookup tables — avoids repeated if/else in hot loops
_DECAY_RATE: dict[int, float] = {
    1: TIER1_DECAY_RATE,
    2: TIER2_DECAY_RATE,
}

_EVICTION_THRESHOLD_MAP: dict[int, float] = {
    1: TIER1_EVICTION_THRESHOLD,
    2: TIER2_EVICTION_THRESHOLD,
}

# Promotion targets: {from_tier: (to_tier, precision_bits)}
_PROMOTION_TARGET: dict[int, tuple[int, int]] = {
    1: (2, 8),
    2: (3, 2),
}


@dataclass
class DecayResult:
    """Summary of a single decay run for one project."""

    project: str
    nodes_decayed: int
    nodes_evicted: int
    nodes_promoted: int
    dry_run: bool = False


@dataclass
class NodeDecayPreview:
    """What would happen to one node during a dry-run decay pass."""

    node: Node
    decayed_weight: float
    action: str  # "decay" | "evict" | "promote" | "skip"
    new_tier: int | None  # set when action == "promote"


def _decay_rate(tier: int) -> float:
    return _DECAY_RATE[tier]


def _eviction_threshold(tier: int) -> float:
    return _EVICTION_THRESHOLD_MAP[tier]


def _meets_promotion_criteria(node: Node, decayed_weight: float, now: int) -> bool:
    """Return True if node qualifies for promotion to the next tier.

    Tier-1 → Tier-2: weight ≥ 8 AND session_count ≥ 3
    Tier-2 → Tier-3: weight ≥ 20 AND session_count ≥ 8 AND age ≥ 14 days
    """
    if node.tier == 1:
        return (
            decayed_weight >= TIER1_PROMOTION_WEIGHT
            and node.session_count >= TIER1_PROMOTION_SESSIONS
        )
    if node.tier == 2:
        age_days = (now - node.created_at) / SECONDS_PER_DAY
        return (
            decayed_weight >= TIER2_PROMOTION_WEIGHT
            and node.session_count >= TIER2_PROMOTION_SESSIONS
            and age_days >= TIER2_PROMOTION_AGE_DAYS
        )
    return False


def run_decay(graph: Graph, project: str) -> DecayResult:
    """Apply weight decay, eviction, and promotion to all nodes for a project.

    Rules applied in order:
    1. Tier-1 nodes: weight * TIER1_DECAY_RATE; evict if weight < TIER1_EVICTION_THRESHOLD
    2. Tier-2 nodes: weight * TIER2_DECAY_RATE; evict if weight < TIER2_EVICTION_THRESHOLD
    3. Tier-3 nodes: no decay, no eviction
    4. Tier-1 → Tier-2 promotion if weight ≥ 8 AND session_count ≥ 3
    5. Tier-2 → Tier-3 promotion if weight ≥ 20 AND session_count ≥ 8 AND age ≥ 14 days

    Precision is downcast on promotion:
    - Tier-1 → Tier-2: precision_bits = 8
    - Tier-2 → Tier-3: precision_bits = 2

    Weight updates and evictions are batched into single SQL transactions.
    Promotions are applied individually because each requires an embedding
    precision downcast (a separate UPDATE per node).

    Args:
        graph: Graph instance to operate on.
        project: Absolute project path — only nodes for this project are touched.

    Returns:
        DecayResult with counts of decayed, evicted, and promoted nodes.
    """
    nodes = graph.get_all_nodes(project=project)
    now = int(time.time())

    weight_deltas: list[tuple[str, float]] = []
    evict_ids: list[str] = []
    to_promote: list[tuple[Node, float]] = []

    for node in nodes:
        if node.tier == 3:
            continue

        delta = node.weight * _decay_rate(node.tier) - node.weight
        decayed_weight = max(0.0, node.weight + delta)
        weight_deltas.append((node.id, delta))

        if decayed_weight < _eviction_threshold(node.tier):
            evict_ids.append(node.id)
        elif _meets_promotion_criteria(node, decayed_weight, now):
            to_promote.append((node, decayed_weight))

    # Batch all weight updates in one transaction
    graph.update_weights_bulk(weight_deltas)

    # Batch all evictions in one DELETE ... IN (...)
    graph.delete_nodes_bulk(evict_ids)

    # Promotions are rare and each needs an embedding downcast — keep individual
    for node, _decayed_weight in to_promote:
        new_tier, new_precision = _PROMOTION_TARGET[node.tier]
        _promote_node(
            graph, node, new_tier=new_tier, new_precision=new_precision, now=now
        )

    return DecayResult(
        project=project,
        nodes_decayed=len(weight_deltas),
        nodes_evicted=len(evict_ids),
        nodes_promoted=len(to_promote),
    )


def preview_decay(graph: Graph, project: str) -> list[NodeDecayPreview]:
    """Return what run_decay() would do without modifying the graph.

    Iterates all project nodes and computes the expected action for each:
    - 'skip'    — tier-3 node (never touched)
    - 'evict'   — decayed weight falls below eviction threshold
    - 'promote' — meets promotion criteria after decay
    - 'decay'   — weight reduced but node survives

    Args:
        graph: Graph instance to query (read-only; no writes).
        project: Absolute project path.

    Returns:
        List of NodeDecayPreview, one per non-tier-3 node.
    """
    nodes = graph.get_all_nodes(project=project)
    now = int(time.time())
    previews: list[NodeDecayPreview] = []

    for node in nodes:
        if node.tier == 3:
            previews.append(
                NodeDecayPreview(
                    node=node, decayed_weight=node.weight, action="skip", new_tier=None
                )
            )
            continue

        delta = node.weight * _decay_rate(node.tier) - node.weight
        decayed_weight = max(0.0, node.weight + delta)

        if decayed_weight < _eviction_threshold(node.tier):
            action = "evict"
            new_tier = None
        elif _meets_promotion_criteria(node, decayed_weight, now):
            action = "promote"
            new_tier = _PROMOTION_TARGET[node.tier][0]
        else:
            action = "decay"
            new_tier = None

        previews.append(
            NodeDecayPreview(
                node=node,
                decayed_weight=decayed_weight,
                action=action,
                new_tier=new_tier,
            )
        )

    return previews


def _promote_node(
    graph: Graph, node: Node, new_tier: int, new_precision: int, now: int
) -> None:
    """Promote a node to a higher tier and downcast embedding precision.

    Args:
        graph: Graph instance.
        node: The node to promote.
        new_tier: Target tier (2 or 3).
        new_precision: Target precision_bits (8 or 2).
        now: Current unix timestamp.
    """
    new_blob: bytes | None = None
    if node.embedding is not None:
        new_blob = serialize(node.embedding.astype(np.float32), new_precision)

    graph.update_node_tier(
        node_id=node.id,
        new_tier=new_tier,
        new_precision=new_precision,
        embedding_blob=new_blob,
        now=now,
    )
