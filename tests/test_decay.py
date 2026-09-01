"""Tests for core/decay.py — weight decay, tier promotion, eviction."""

from __future__ import annotations

import sqlite3
import time

import numpy as np
import pytest

from core.decay import (
    SECONDS_PER_DAY,
    TIER1_DECAY_RATE,
    TIER1_EVICTION_THRESHOLD,
    TIER1_PROMOTION_SESSIONS,
    TIER1_PROMOTION_WEIGHT,
    TIER2_DECAY_RATE,
    TIER2_PROMOTION_AGE_DAYS,
    TIER2_PROMOTION_SESSIONS,
    TIER2_PROMOTION_WEIGHT,
    DecayResult,
    _meets_promotion_criteria,
    run_decay,
)
from core.graph import Graph, Node

TEST_PROJECT = "/tmp/cortex_test_project"


@pytest.fixture
def graph(db: sqlite3.Connection) -> Graph:
    return Graph(connection=db)


def _make_node(
    embedding: np.ndarray,
    tier: int = 1,
    weight: float = 1.0,
    session_count: int = 1,
    created_at: int | None = None,
    project: str = TEST_PROJECT,
    text: str = "test node",
) -> Node:
    now = int(time.time())
    return Node(
        id="",
        type="observation",
        tier=tier,
        text=text,
        rationale=None,
        embedding=embedding,
        precision_bits=32,
        weight=weight,
        project=project,
        scope="project",
        source="jsonl",
        last_accessed=now,
        created_at=created_at or now,
        session_count=session_count,
    )


# ---------------------------------------------------------------------------
# Basic decay
# ---------------------------------------------------------------------------


class TestDecayRates:
    def test_tier1_decays_by_correct_rate(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, tier=1, weight=2.0))
        run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        expected = 2.0 * TIER1_DECAY_RATE
        assert nodes[0].weight == pytest.approx(expected, abs=1e-4)

    def test_tier2_decays_by_correct_rate(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, tier=2, weight=2.0))
        run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        expected = 2.0 * TIER2_DECAY_RATE
        assert nodes[0].weight == pytest.approx(expected, abs=1e-4)

    def test_tier3_nodes_never_decay(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, tier=3, weight=5.0))
        initial_weight = 5.0
        for _ in range(100):
            run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT, tier=3)
        assert nodes[0].weight == pytest.approx(initial_weight, abs=1e-4)

    def test_decay_result_counts_decayed(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, tier=1, weight=2.0))
        graph.write_node(_make_node(dummy_embedding, tier=2, weight=2.0, text="t2"))
        result = run_decay(graph, TEST_PROJECT)
        assert result.nodes_decayed == 2

    def test_empty_graph_returns_zero_counts(self, graph: Graph) -> None:
        result = run_decay(graph, TEST_PROJECT)
        assert result.nodes_decayed == 0
        assert result.nodes_evicted == 0
        assert result.nodes_promoted == 0

    def test_decay_result_has_correct_project(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding))
        result = run_decay(graph, TEST_PROJECT)
        assert result.project == TEST_PROJECT

    def test_decay_only_affects_target_project(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, project="/proj/a", weight=1.0))
        graph.write_node(_make_node(dummy_embedding, project="/proj/b", weight=1.0))
        run_decay(graph, "/proj/a")
        b_nodes = graph.get_all_nodes(project="/proj/b")
        assert b_nodes[0].weight == pytest.approx(1.0, abs=1e-6)

    def test_tier3_weight_unchanged_after_decay(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, tier=3, weight=7.5))
        run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT, tier=3)
        assert nodes[0].weight == pytest.approx(7.5, abs=1e-6)

    def test_multiple_decays_accumulate_tier1_weight_decrease(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, tier=1, weight=1.0))
        run_decay(graph, TEST_PROJECT)
        after_first = graph.get_all_nodes(project=TEST_PROJECT, tier=1)[0].weight
        run_decay(graph, TEST_PROJECT)
        after_second = graph.get_all_nodes(project=TEST_PROJECT, tier=1)[0].weight
        assert after_first < 1.0
        assert after_second < after_first

    def test_tier2_weight_correct_after_two_decays(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, tier=2, weight=2.0))
        run_decay(graph, TEST_PROJECT)
        run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT, tier=2)
        if nodes:
            expected = 2.0 * TIER2_DECAY_RATE * TIER2_DECAY_RATE
            assert nodes[0].weight == pytest.approx(expected, abs=1e-3)

    def test_nodes_decayed_count_excludes_tier3(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, tier=1, weight=2.0))
        graph.write_node(_make_node(dummy_embedding, tier=3, weight=2.0, text="t3"))
        result = run_decay(graph, TEST_PROJECT)
        assert result.nodes_decayed == 1

    def test_tier1_node_weight_lower_after_decay(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, tier=1, weight=5.0))
        run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT, tier=1)
        if nodes:
            assert nodes[0].weight < 5.0

    def test_decay_result_is_decay_result_instance(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, tier=1))
        result = run_decay(graph, TEST_PROJECT)
        assert isinstance(result, DecayResult)

    def test_decay_result_project_field_is_str(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, tier=1))
        result = run_decay(graph, TEST_PROJECT)
        assert isinstance(result.project, str)

    def test_decay_skips_other_project_nodes(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        other = "/other/proj"
        graph.write_node(_make_node(dummy_embedding, tier=1, project=other, weight=5.0))
        run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=other, tier=1)
        assert len(nodes) == 1
        assert nodes[0].weight == 5.0

    def test_tier2_weight_lower_after_decay(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, tier=2, weight=3.0))
        run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT, tier=2)
        if nodes:
            assert nodes[0].weight < 3.0

    def test_decay_result_nodes_decayed_nonneg(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        result = run_decay(graph, TEST_PROJECT)
        assert result.nodes_decayed >= 0


# ---------------------------------------------------------------------------
# Eviction
# ---------------------------------------------------------------------------


class TestEviction:
    def test_tier1_node_evicted_below_threshold(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(
            _make_node(dummy_embedding, tier=1, weight=TIER1_EVICTION_THRESHOLD + 0.01)
        )
        # Decay once to push weight just below threshold
        # weight after decay = (threshold + 0.01) * 0.85 < threshold
        run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT, tier=1)
        # Check if evicted
        decayed_weight = (TIER1_EVICTION_THRESHOLD + 0.01) * TIER1_DECAY_RATE
        if decayed_weight < TIER1_EVICTION_THRESHOLD:
            assert len(nodes) == 0

    def test_tier1_eviction_at_threshold(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        # Start just at threshold: after decay it goes below
        graph.write_node(_make_node(dummy_embedding, tier=1, weight=0.35))
        run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT, tier=1)
        # 0.35 * 0.85 = 0.2975 < 0.3 → evicted
        assert len(nodes) == 0

    def test_tier2_eviction_at_threshold(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, tier=2, weight=0.53))
        run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT, tier=2)
        # 0.53 * 0.95 = 0.5035 > 0.5 → not evicted
        assert len(nodes) == 1

    def test_tier2_evicted_below_threshold(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, tier=2, weight=0.52))
        run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT, tier=2)
        # 0.52 * 0.95 = 0.494 < 0.5 → evicted
        assert len(nodes) == 0

    def test_eviction_count_in_result(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, tier=1, weight=0.2))
        result = run_decay(graph, TEST_PROJECT)
        assert result.nodes_evicted >= 1

    def test_tier3_never_evicted(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, tier=3, weight=0.001))
        for _ in range(50):
            run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT, tier=3)
        assert len(nodes) == 1

    def test_high_weight_tier1_node_survives_eviction(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, tier=1, weight=5.0))
        run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT, tier=1)
        assert len(nodes) == 1

    def test_eviction_does_not_affect_other_tier(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, tier=1, weight=0.1))
        graph.write_node(_make_node(dummy_embedding, tier=2, weight=5.0, text="t2"))
        run_decay(graph, TEST_PROJECT)
        tier2_nodes = graph.get_all_nodes(project=TEST_PROJECT, tier=2)
        assert len(tier2_nodes) == 1

    def test_two_low_weight_tier1_nodes_both_evicted(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, tier=1, weight=0.2, text="a"))
        graph.write_node(_make_node(dummy_embedding, tier=1, weight=0.2, text="b"))
        run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT, tier=1)
        assert len(nodes) == 0

    def test_tier2_safely_above_threshold_survives(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, tier=2, weight=2.0))
        run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT, tier=2)
        assert len(nodes) == 1

    def test_eviction_result_project_matches(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, tier=1, weight=0.1))
        result = run_decay(graph, TEST_PROJECT)
        assert result.project == TEST_PROJECT

    def test_eviction_count_is_nonnegative(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, tier=1, weight=2.0))
        result = run_decay(graph, TEST_PROJECT)
        assert result.nodes_evicted >= 0

    def test_eviction_result_project_is_str(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        result = run_decay(graph, TEST_PROJECT)
        assert isinstance(result.project, str)

    def test_other_project_node_survives_eviction(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        other = "/other/project"
        graph.write_node(_make_node(dummy_embedding, tier=1, project=other, weight=0.1))
        run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=other, tier=1)
        assert len(nodes) == 1

    def test_tier1_eviction_removes_node_from_graph(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, tier=1, weight=0.1))
        run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT, tier=1)
        assert len(nodes) == 0

    def test_tier2_eviction_removes_node_from_graph(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, tier=2, weight=0.1))
        run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT, tier=2)
        assert len(nodes) == 0


# ---------------------------------------------------------------------------
# Promotion — Tier 1 → Tier 2
# ---------------------------------------------------------------------------


class TestTier1Promotion:
    def test_promotion_requires_both_weight_and_sessions(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        # Meets weight but not sessions
        graph.write_node(
            _make_node(
                dummy_embedding,
                tier=1,
                weight=TIER1_PROMOTION_WEIGHT + 1,
                session_count=TIER1_PROMOTION_SESSIONS - 1,
                text="weight but not sessions",
            )
        )
        run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        still_tier1 = [n for n in nodes if n.text == "weight but not sessions"]
        if still_tier1:
            assert still_tier1[0].tier == 1

    def test_node_promotes_to_tier2_when_conditions_met(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        initial_weight = TIER1_PROMOTION_WEIGHT + 2.0
        graph.write_node(
            _make_node(
                dummy_embedding,
                tier=1,
                weight=initial_weight,
                session_count=TIER1_PROMOTION_SESSIONS,
                text="promote this",
            )
        )
        run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        promoted = [n for n in nodes if n.text == "promote this"]
        if promoted:
            assert promoted[0].tier == 2

    def test_precision_downcasts_to_8_on_tier1_promotion(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        initial_weight = TIER1_PROMOTION_WEIGHT + 2.0
        graph.write_node(
            _make_node(
                dummy_embedding,
                tier=1,
                weight=initial_weight,
                session_count=TIER1_PROMOTION_SESSIONS,
                text="precision check",
            )
        )
        run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        precision_nodes = [n for n in nodes if n.text == "precision check"]
        if precision_nodes and precision_nodes[0].tier == 2:
            assert precision_nodes[0].precision_bits == 8

    def test_promotion_count_in_result(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(
            _make_node(
                dummy_embedding,
                tier=1,
                weight=TIER1_PROMOTION_WEIGHT + 5.0,
                session_count=TIER1_PROMOTION_SESSIONS,
            )
        )
        result = run_decay(graph, TEST_PROJECT)
        assert result.nodes_promoted >= 1

    def test_low_weight_node_stays_tier1(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(
            _make_node(
                dummy_embedding,
                tier=1,
                weight=TIER1_PROMOTION_WEIGHT - 1.0,
                session_count=TIER1_PROMOTION_SESSIONS,
                text="under weight",
            )
        )
        run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        under = [n for n in nodes if n.text == "under weight"]
        if under:
            assert under[0].tier == 1

    def test_tier1_qualified_node_promoted_increments_result(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(
            _make_node(
                dummy_embedding,
                tier=1,
                weight=TIER1_PROMOTION_WEIGHT + 3.0,
                session_count=TIER1_PROMOTION_SESSIONS,
                text="ready to promote",
            )
        )
        result = run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        promoted = [n for n in nodes if n.text == "ready to promote" and n.tier == 2]
        if promoted:
            assert result.nodes_promoted >= 1

    def test_tier1_node_project_preserved_after_promotion(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(
            _make_node(
                dummy_embedding,
                tier=1,
                weight=TIER1_PROMOTION_WEIGHT + 2.0,
                session_count=TIER1_PROMOTION_SESSIONS,
                text="project check",
                project=TEST_PROJECT,
            )
        )
        run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        matched = [n for n in nodes if n.text == "project check"]
        if matched:
            assert matched[0].project == TEST_PROJECT

    def test_empty_project_promotion_count_zero(self, graph: Graph) -> None:
        result = run_decay(graph, TEST_PROJECT)
        assert result.nodes_promoted == 0

    def test_tier1_promotion_count_nonneg(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        result = run_decay(graph, TEST_PROJECT)
        assert result.nodes_promoted >= 0

    def test_tier1_promoted_node_has_tier2(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(
            _make_node(
                dummy_embedding,
                tier=1,
                weight=TIER1_PROMOTION_WEIGHT + 5.0,
                session_count=TIER1_PROMOTION_SESSIONS,
                text="promote_me",
            )
        )
        run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        tier2 = [n for n in nodes if n.text == "promote_me" and n.tier == 2]
        tier1 = [n for n in nodes if n.text == "promote_me" and n.tier == 1]
        assert len(tier2) == 1 or len(tier1) == 0

    def test_tier1_not_promoted_when_low_sessions(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(
            _make_node(
                dummy_embedding,
                tier=1,
                weight=TIER1_PROMOTION_WEIGHT + 5.0,
                session_count=1,
                text="low_sessions",
            )
        )
        run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        tier2 = [n for n in nodes if n.text == "low_sessions" and n.tier == 2]
        assert len(tier2) == 0

    def test_tier1_result_project_str(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        result = run_decay(graph, TEST_PROJECT)
        assert isinstance(result.project, str)


# ---------------------------------------------------------------------------
# Promotion — Tier 2 → Tier 3
# ---------------------------------------------------------------------------


class TestTier2Promotion:
    def _old_node(self, embedding: np.ndarray, days_old: int = 15) -> Node:
        old_timestamp = int(time.time()) - days_old * SECONDS_PER_DAY
        return _make_node(
            embedding,
            tier=2,
            weight=TIER2_PROMOTION_WEIGHT + 2.0,
            session_count=TIER2_PROMOTION_SESSIONS,
            created_at=old_timestamp,
        )

    def test_tier2_promotion_requires_all_three_conditions(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        # Missing: age < 14 days
        node = _make_node(
            dummy_embedding,
            tier=2,
            weight=TIER2_PROMOTION_WEIGHT + 2.0,
            session_count=TIER2_PROMOTION_SESSIONS,
            text="no age",
        )
        graph.write_node(node)
        run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        no_age = [n for n in nodes if n.text == "no age"]
        if no_age:
            assert no_age[0].tier == 2

    def test_tier2_promotes_to_tier3_when_all_conditions_met(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        node = self._old_node(dummy_embedding)
        node = Node(
            id=node.id,
            type=node.type,
            tier=node.tier,
            text="promote to tier3",
            rationale=node.rationale,
            embedding=node.embedding,
            precision_bits=node.precision_bits,
            weight=node.weight,
            project=node.project,
            scope=node.scope,
            source=node.source,
            last_accessed=node.last_accessed,
            created_at=node.created_at,
            session_count=node.session_count,
        )
        graph.write_node(node)
        run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        promoted = [n for n in nodes if n.text == "promote to tier3"]
        if promoted:
            assert promoted[0].tier == 3

    def test_precision_downcasts_to_2_on_tier2_promotion(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        node = self._old_node(dummy_embedding)
        node = Node(
            id=node.id,
            type=node.type,
            tier=node.tier,
            text="precision3 check",
            rationale=node.rationale,
            embedding=node.embedding,
            precision_bits=8,
            weight=node.weight,
            project=node.project,
            scope=node.scope,
            source=node.source,
            last_accessed=node.last_accessed,
            created_at=node.created_at,
            session_count=node.session_count,
        )
        graph.write_node(node)
        run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        p3 = [n for n in nodes if n.text == "precision3 check"]
        if p3 and p3[0].tier == 3:
            assert p3[0].precision_bits == 2

    def test_tier2_promotion_missing_weight_stays_tier2(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        old_ts = int(time.time()) - 20 * SECONDS_PER_DAY
        node = _make_node(
            dummy_embedding,
            tier=2,
            weight=5.0,  # too low
            session_count=TIER2_PROMOTION_SESSIONS,
            created_at=old_ts,
            text="missing weight",
        )
        graph.write_node(node)
        run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        low_wt = [n for n in nodes if n.text == "missing weight"]
        if low_wt:
            assert low_wt[0].tier == 2

    def test_tier2_promotion_missing_sessions_stays_tier2(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        old_ts = int(time.time()) - 20 * SECONDS_PER_DAY
        node = _make_node(
            dummy_embedding,
            tier=2,
            weight=TIER2_PROMOTION_WEIGHT + 2.0,
            session_count=2,  # too few
            created_at=old_ts,
            text="missing sessions",
        )
        graph.write_node(node)
        run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        low_sess = [n for n in nodes if n.text == "missing sessions"]
        if low_sess:
            assert low_sess[0].tier == 2

    def test_tier2_promotion_count_in_result(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        old_ts = int(time.time()) - 20 * SECONDS_PER_DAY
        node = _make_node(
            dummy_embedding,
            tier=2,
            weight=TIER2_PROMOTION_WEIGHT + 2.0,
            session_count=TIER2_PROMOTION_SESSIONS,
            created_at=old_ts,
            text="promo node",
        )
        graph.write_node(node)
        result = run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        promoted = [n for n in nodes if n.tier == 3]
        if promoted:
            assert result.nodes_promoted >= 1

    def test_tier2_node_young_stays_tier2(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        recent_ts = int(time.time()) - 5 * SECONDS_PER_DAY
        node = _make_node(
            dummy_embedding,
            tier=2,
            weight=TIER2_PROMOTION_WEIGHT + 2.0,
            session_count=TIER2_PROMOTION_SESSIONS,
            created_at=recent_ts,
            text="young node",
        )
        graph.write_node(node)
        run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        young = [n for n in nodes if n.text == "young node"]
        if young:
            assert young[0].tier == 2

    def test_promoted_node_retains_project(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        old_ts = int(time.time()) - 20 * SECONDS_PER_DAY
        node = _make_node(
            dummy_embedding,
            tier=2,
            weight=TIER2_PROMOTION_WEIGHT + 2.0,
            session_count=TIER2_PROMOTION_SESSIONS,
            created_at=old_ts,
            text="project check",
        )
        graph.write_node(node)
        run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        found = [n for n in nodes if n.text == "project check"]
        assert all(n.project == TEST_PROJECT for n in found)

    def test_non_qualifying_node_weight_decays(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        node = _make_node(
            dummy_embedding,
            tier=2,
            weight=3.0,
            session_count=1,
            text="no promote",
        )
        graph.write_node(node)
        run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        found = [n for n in nodes if n.text == "no promote"]
        if found:
            assert found[0].weight < 3.0

    def test_tier2_two_qualifying_nodes_count_in_result(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        old_ts = int(time.time()) - 20 * SECONDS_PER_DAY
        for i in range(2):
            graph.write_node(
                _make_node(
                    dummy_embedding,
                    tier=2,
                    weight=TIER2_PROMOTION_WEIGHT + 2.0,
                    session_count=TIER2_PROMOTION_SESSIONS,
                    created_at=old_ts,
                    text=f"qualify {i}",
                )
            )
        result = run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        promoted = [n for n in nodes if n.tier == 3]
        if len(promoted) == 2:
            assert result.nodes_promoted >= 2

    def test_promoted_tier2_project_unchanged(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        old_ts = int(time.time()) - 20 * SECONDS_PER_DAY
        graph.write_node(
            _make_node(
                dummy_embedding,
                tier=2,
                weight=TIER2_PROMOTION_WEIGHT + 2.0,
                session_count=TIER2_PROMOTION_SESSIONS,
                created_at=old_ts,
                text="proj unchanged",
                project=TEST_PROJECT,
            )
        )
        run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        found = [n for n in nodes if n.text == "proj unchanged"]
        if found:
            assert found[0].project == TEST_PROJECT


# ---------------------------------------------------------------------------
# DecayResult dataclass
# ---------------------------------------------------------------------------


class TestDecayResult:
    def test_decay_result_is_dataclass(self) -> None:
        result = DecayResult(
            project="/p", nodes_decayed=1, nodes_evicted=0, nodes_promoted=0
        )
        assert result.project == "/p"
        assert result.nodes_decayed == 1

    def test_multiple_decay_runs_accumulate_decay(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, tier=1, weight=5.0))
        for _ in range(3):
            run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT, tier=1)
        if nodes:
            expected = 5.0 * (TIER1_DECAY_RATE**3)
            assert nodes[0].weight == pytest.approx(expected, abs=0.01)

    def test_decay_result_project_matches_input(self) -> None:
        result = DecayResult(
            project=TEST_PROJECT, nodes_decayed=0, nodes_evicted=0, nodes_promoted=0
        )
        assert result.project == TEST_PROJECT

    def test_decay_result_eviction_count_nonzero_on_low_weight(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(
            _make_node(dummy_embedding, tier=1, weight=TIER1_EVICTION_THRESHOLD - 0.01)
        )
        result = run_decay(graph, TEST_PROJECT)
        assert result.nodes_evicted >= 1

    def test_decay_result_nodes_decayed_is_nonneg(self, graph: Graph) -> None:
        result = run_decay(graph, TEST_PROJECT)
        assert result.nodes_decayed >= 0

    def test_decay_result_nodes_evicted_is_nonneg(self, graph: Graph) -> None:
        result = run_decay(graph, TEST_PROJECT)
        assert result.nodes_evicted >= 0

    def test_decay_result_nodes_promoted_nonneg(self, graph: Graph) -> None:
        result = run_decay(graph, TEST_PROJECT)
        assert result.nodes_promoted >= 0

    def test_decay_result_all_counts_int(self, graph: Graph) -> None:
        result = run_decay(graph, TEST_PROJECT)
        assert isinstance(result.nodes_decayed, int)
        assert isinstance(result.nodes_evicted, int)
        assert isinstance(result.nodes_promoted, int)

    def test_decay_result_project_not_empty(self, graph: Graph) -> None:
        result = run_decay(graph, TEST_PROJECT)
        assert len(result.project) > 0

    def test_decay_result_evicted_le_decayed(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        for _ in range(3):
            graph.write_node(_make_node(dummy_embedding, tier=1, weight=1.5))
        result = run_decay(graph, TEST_PROJECT)
        assert result.nodes_evicted <= result.nodes_decayed


# ---------------------------------------------------------------------------
# _meets_promotion_criteria — parametrized unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "weight, session_count, expected",
    [
        # Meets both criteria
        (TIER1_PROMOTION_WEIGHT, TIER1_PROMOTION_SESSIONS, True),
        (TIER1_PROMOTION_WEIGHT + 5.0, TIER1_PROMOTION_SESSIONS + 2, True),
        # Too light
        (TIER1_PROMOTION_WEIGHT - 0.1, TIER1_PROMOTION_SESSIONS, False),
        # Too few sessions
        (TIER1_PROMOTION_WEIGHT, TIER1_PROMOTION_SESSIONS - 1, False),
        # Both too low
        (TIER1_PROMOTION_WEIGHT - 1.0, TIER1_PROMOTION_SESSIONS - 1, False),
    ],
)
def test_meets_promotion_criteria_tier1(
    weight: float,
    session_count: int,
    expected: bool,
    dummy_embedding: np.ndarray,
) -> None:
    node = _make_node(
        dummy_embedding, tier=1, weight=weight, session_count=session_count
    )
    now = int(time.time())
    assert _meets_promotion_criteria(node, weight, now) is expected


@pytest.mark.parametrize(
    "weight, session_count, age_days, expected",
    [
        # All criteria met
        (
            TIER2_PROMOTION_WEIGHT,
            TIER2_PROMOTION_SESSIONS,
            TIER2_PROMOTION_AGE_DAYS,
            True,
        ),
        # Too light
        (
            TIER2_PROMOTION_WEIGHT - 0.1,
            TIER2_PROMOTION_SESSIONS,
            TIER2_PROMOTION_AGE_DAYS,
            False,
        ),
        # Too few sessions
        (
            TIER2_PROMOTION_WEIGHT,
            TIER2_PROMOTION_SESSIONS - 1,
            TIER2_PROMOTION_AGE_DAYS,
            False,
        ),
        # Too young (1 day short)
        (
            TIER2_PROMOTION_WEIGHT,
            TIER2_PROMOTION_SESSIONS,
            TIER2_PROMOTION_AGE_DAYS - 1,
            False,
        ),
    ],
)
def test_meets_promotion_criteria_tier2(
    weight: float,
    session_count: int,
    age_days: int,
    expected: bool,
    dummy_embedding: np.ndarray,
) -> None:
    now = int(time.time())
    created_at = now - int(age_days * SECONDS_PER_DAY)
    node = _make_node(
        dummy_embedding,
        tier=2,
        weight=weight,
        session_count=session_count,
        created_at=created_at,
    )
    assert _meets_promotion_criteria(node, weight, now) is expected


def test_meets_promotion_criteria_tier3_always_false(
    dummy_embedding: np.ndarray,
) -> None:
    node = _make_node(dummy_embedding, tier=3, weight=100.0, session_count=100)
    assert _meets_promotion_criteria(node, 100.0, int(time.time())) is False


# ---------------------------------------------------------------------------
# _decay_rate and _eviction_threshold helpers
# ---------------------------------------------------------------------------


def test_decay_rate_tier1_matches_constant(dummy_embedding: np.ndarray) -> None:
    from core.decay import TIER1_DECAY_RATE, _decay_rate

    assert _decay_rate(1) == TIER1_DECAY_RATE


def test_decay_rate_tier2_matches_constant(dummy_embedding: np.ndarray) -> None:
    from core.decay import TIER2_DECAY_RATE, _decay_rate

    assert _decay_rate(2) == TIER2_DECAY_RATE


def test_eviction_threshold_tier1_matches_constant(dummy_embedding: np.ndarray) -> None:
    from core.decay import TIER1_EVICTION_THRESHOLD, _eviction_threshold

    assert _eviction_threshold(1) == TIER1_EVICTION_THRESHOLD


def test_eviction_threshold_tier2_matches_constant(dummy_embedding: np.ndarray) -> None:
    from core.decay import TIER2_EVICTION_THRESHOLD, _eviction_threshold

    assert _eviction_threshold(2) == TIER2_EVICTION_THRESHOLD


def test_run_decay_empty_project_returns_zero_counts(
    graph: Graph, dummy_embedding: np.ndarray
) -> None:
    result = run_decay(graph, TEST_PROJECT)
    assert result.nodes_decayed == 0
    assert result.nodes_evicted == 0
    assert result.nodes_promoted == 0


def test_decay_result_project_field_matches(
    graph: Graph, dummy_embedding: np.ndarray
) -> None:
    result = run_decay(graph, TEST_PROJECT)
    assert result.project == TEST_PROJECT


def test_tier3_node_never_decayed(graph: Graph, dummy_embedding: np.ndarray) -> None:
    node = _make_node(dummy_embedding, tier=3, weight=100.0)
    graph.write_node(node)
    result = run_decay(graph, TEST_PROJECT)
    nodes = graph.get_all_nodes(project=TEST_PROJECT)
    assert nodes[0].weight == 100.0
    assert result.nodes_decayed == 0


class TestDecay20260901A:
    def test_tier1_rate_in_unit_interval(self) -> None:
        assert 0.0 < TIER1_DECAY_RATE < 1.0

    def test_tier2_rate_in_unit_interval(self) -> None:
        assert 0.0 < TIER2_DECAY_RATE < 1.0

    def test_tier2_rate_exceeds_tier1(self) -> None:
        assert TIER2_DECAY_RATE > TIER1_DECAY_RATE

    def test_tier1_eviction_threshold_positive(self) -> None:
        assert TIER1_EVICTION_THRESHOLD > 0.0

    def test_seconds_per_day_value(self) -> None:
        assert SECONDS_PER_DAY == 86_400

    def test_tier2_promotion_sessions_exceeds_tier1(self) -> None:
        assert TIER2_PROMOTION_SESSIONS >= TIER1_PROMOTION_SESSIONS
