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


@dataclass
class DecayResult:
    """Summary of a single decay run for one project."""

    project: str
    nodes_decayed: int
    nodes_evicted: int
    nodes_promoted: int


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

    Args:
        graph: Graph instance to operate on.
        project: Absolute project path — only nodes for this project are touched.

    Returns:
        DecayResult with counts of decayed, evicted, and promoted nodes.
    """
    nodes = graph.get_all_nodes(project=project)
    now = int(time.time())

    decayed = 0
    evicted = 0
    promoted = 0

    for node in nodes:
        if node.tier == 3:
            continue

        # Apply decay
        if node.tier == 1:
            new_weight = node.weight * TIER1_DECAY_RATE
        else:
            new_weight = node.weight * TIER2_DECAY_RATE

        delta = new_weight - node.weight
        graph.update_weight(node.id, delta=delta)
        decayed += 1

        decayed_weight = max(0.0, node.weight + delta)

        # Eviction
        if node.tier == 1 and decayed_weight < TIER1_EVICTION_THRESHOLD:
            graph.delete_node(node.id)
            evicted += 1
            continue

        if node.tier == 2 and decayed_weight < TIER2_EVICTION_THRESHOLD:
            graph.delete_node(node.id)
            evicted += 1
            continue

        # Promotion checks
        if node.tier == 1:
            if (
                decayed_weight >= TIER1_PROMOTION_WEIGHT
                and node.session_count >= TIER1_PROMOTION_SESSIONS
            ):
                _promote_node(graph, node, new_tier=2, new_precision=8, now=now)
                promoted += 1

        elif node.tier == 2:
            age_days = (now - node.created_at) / SECONDS_PER_DAY
            if (
                decayed_weight >= TIER2_PROMOTION_WEIGHT
                and node.session_count >= TIER2_PROMOTION_SESSIONS
                and age_days >= TIER2_PROMOTION_AGE_DAYS
            ):
                _promote_node(graph, node, new_tier=3, new_precision=2, now=now)
                promoted += 1

    return DecayResult(
        project=project,
        nodes_decayed=decayed,
        nodes_evicted=evicted,
        nodes_promoted=promoted,
    )


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
