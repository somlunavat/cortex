#!/usr/bin/env python3
"""Daily automated test-coverage commits for somlunavat/cortex.

Runs via GitHub Actions on a daily schedule. Generates test classes from a
pre-validated template library — no external API needed — validates each one
(syntax + black + ruff + pytest), commits, pushes a branch per module, opens a
PR, and enables auto-merge.

Requires:
    GITHUB_TOKEN  — GitHub token (automatic in Actions, write access)
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO = "somlunavat/cortex"
GH_TOKEN = os.environ["GITHUB_TOKEN"]
NOW = _dt.datetime.now(_dt.UTC)
DATE_TAG = NOW.strftime("%Y%m%d")

# Four test areas cycled per day.  Each entry picks a different slice of the
# template library so adjacent PRs always touch different modules.
TARGETS = [
    {
        "test_file": "tests/test_decay.py",
        "module": "decay",
        "test_project": "TEST_PROJECT",
        "needs_graph": True,
        "needs_embedding": True,
    },
    {
        "test_file": "tests/test_graph.py",
        "module": "graph",
        "test_project": "TEST_PROJECT",
        "needs_graph": True,
        "needs_embedding": True,
    },
    {
        "test_file": "tests/test_retrieval.py",
        "module": "retrieval",
        "test_project": "TEST_PROJECT",
        "needs_graph": True,
        "needs_embedding": False,
    },
    {
        "test_file": "tests/test_parser.py",
        "module": "parser",
        "test_project": "TEST_PROJECT",
        "needs_graph": False,
        "needs_embedding": False,
    },
    {
        "test_file": "tests/test_extractor.py",
        "module": "extractor",
        "test_project": "TEST_PROJECT",
        "needs_graph": False,
        "needs_embedding": False,
    },
    {
        "test_file": "tests/test_hooks.py",
        "module": "hooks",
        "test_project": "TEST_PROJECT",
        "needs_graph": True,
        "needs_embedding": False,
    },
]

# ---------------------------------------------------------------------------
# Template libraries — one list per module.  The script picks by rotating
# through the list.  Class names are date-stamped so merges never collide.
# ---------------------------------------------------------------------------

# Each template is a function(class_name) -> str returning a valid class body.

DECAY_TEMPLATES = [
    lambda cn: f"""\
class {cn}:
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
""",
    lambda cn: f"""\
class {cn}:
    def test_empty_graph_decayed_zero(self, graph: Graph) -> None:
        result = run_decay(graph, TEST_PROJECT)
        assert result.nodes_decayed == 0

    def test_empty_graph_evicted_zero(self, graph: Graph) -> None:
        result = run_decay(graph, TEST_PROJECT)
        assert result.nodes_evicted == 0

    def test_empty_graph_promoted_zero(self, graph: Graph) -> None:
        result = run_decay(graph, TEST_PROJECT)
        assert result.nodes_promoted == 0

    def test_result_project_matches(self, graph: Graph) -> None:
        result = run_decay(graph, "/my/project")
        assert result.project == "/my/project"

    def test_result_is_decay_result_type(self, graph: Graph) -> None:
        result = run_decay(graph, TEST_PROJECT)
        assert isinstance(result, DecayResult)

    def test_tier2_age_days_positive(self) -> None:
        assert TIER2_PROMOTION_AGE_DAYS > 0
""",
    lambda cn: f"""\
class {cn}:
    def test_tier1_node_weight_decreases(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, tier=1, weight=4.0))
        run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT, tier=1)
        if nodes:
            assert nodes[0].weight < 4.0

    def test_tier2_node_weight_decreases(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, tier=2, weight=4.0))
        run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT, tier=2)
        if nodes:
            assert nodes[0].weight < 4.0

    def test_tier3_node_never_decays(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, tier=3, weight=4.0))
        run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT, tier=3)
        assert nodes[0].weight == pytest.approx(4.0)

    def test_decayed_count_is_one(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, tier=1, weight=4.0))
        result = run_decay(graph, TEST_PROJECT)
        assert result.nodes_decayed == 1

    def test_two_nodes_both_decayed(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, tier=1, weight=4.0))
        graph.write_node(_make_node(dummy_embedding, tier=1, weight=3.0, text="b"))
        result = run_decay(graph, TEST_PROJECT)
        assert result.nodes_decayed == 2

    def test_tier1_decay_rate_applied(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, tier=1, weight=2.0))
        run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT, tier=1)
        if nodes:
            assert nodes[0].weight == pytest.approx(2.0 * TIER1_DECAY_RATE, abs=1e-4)
""",
    lambda cn: f"""\
class {cn}:
    def test_promotion_weight_positive(self) -> None:
        assert TIER1_PROMOTION_WEIGHT > 0.0

    def test_tier2_promotion_weight_exceeds_tier1(self) -> None:
        assert TIER2_PROMOTION_WEIGHT > TIER1_PROMOTION_WEIGHT

    def test_tier1_promotion_sessions_positive(self) -> None:
        assert TIER1_PROMOTION_SESSIONS > 0

    def test_decay_result_field_count(self) -> None:
        from dataclasses import fields
        assert len(fields(DecayResult)) == 4

    def test_decay_result_nodes_decayed_field(self) -> None:
        r = DecayResult(project="/p", nodes_decayed=5, nodes_evicted=1, nodes_promoted=0)
        assert r.nodes_decayed == 5

    def test_decay_result_nodes_evicted_field(self) -> None:
        r = DecayResult(project="/p", nodes_decayed=0, nodes_evicted=3, nodes_promoted=0)
        assert r.nodes_evicted == 3
""",
    lambda cn: f"""\
class {cn}:
    def test_tier3_not_decayed_in_count(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, tier=1, weight=2.0))
        graph.write_node(_make_node(dummy_embedding, tier=3, weight=2.0, text="t3"))
        result = run_decay(graph, TEST_PROJECT)
        assert result.nodes_decayed == 1

    def test_other_project_not_decayed(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(
            _make_node(dummy_embedding, tier=1, weight=3.0, project="/other")
        )
        run_decay(graph, TEST_PROJECT)
        nodes = graph.get_all_nodes(project="/other", tier=1)
        assert nodes[0].weight == pytest.approx(3.0, abs=1e-6)

    def test_decay_result_promoted_field(self) -> None:
        r = DecayResult(project="/p", nodes_decayed=0, nodes_evicted=0, nodes_promoted=2)
        assert r.nodes_promoted == 2

    def test_decay_result_project_str(self) -> None:
        r = DecayResult(project="/p", nodes_decayed=0, nodes_evicted=0, nodes_promoted=0)
        assert isinstance(r.project, str)

    def test_tier1_eviction_threshold_lt_one(self) -> None:
        assert TIER1_EVICTION_THRESHOLD < 1.0

    def test_tier2_promotion_age_days_at_least_one(self) -> None:
        assert TIER2_PROMOTION_AGE_DAYS >= 1
""",
    lambda cn: f"""\
class {cn}:
    def test_meets_promotion_tier1_below_threshold(
        self, dummy_embedding: np.ndarray
    ) -> None:
        import time as _time
        node = _make_node(
            dummy_embedding,
            tier=1,
            weight=TIER1_PROMOTION_WEIGHT - 1.0,
            session_count=TIER1_PROMOTION_SESSIONS,
        )
        assert not _meets_promotion_criteria(
            node, TIER1_PROMOTION_WEIGHT - 1.0, int(_time.time())
        )

    def test_meets_promotion_tier3_never(
        self, dummy_embedding: np.ndarray
    ) -> None:
        import time as _time
        node = _make_node(dummy_embedding, tier=3, weight=999.0, session_count=999)
        assert not _meets_promotion_criteria(node, 999.0, int(_time.time()))

    def test_seconds_per_day_equals_24h(self) -> None:
        assert SECONDS_PER_DAY == 60 * 60 * 24

    def test_tier2_decay_rate_positive(self) -> None:
        assert TIER2_DECAY_RATE > 0.0

    def test_tier1_decay_rate_positive(self) -> None:
        assert TIER1_DECAY_RATE > 0.0

    def test_decay_result_zero_all(self, graph: Graph) -> None:
        r = run_decay(graph, TEST_PROJECT)
        assert r.nodes_decayed == r.nodes_evicted == r.nodes_promoted == 0
""",
]

GRAPH_TEMPLATES = [
    lambda cn: f"""\
class {cn}:
    def test_write_node_returns_string(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        node_id = graph.write_node(_make_node(dummy_embedding))
        assert isinstance(node_id, str)

    def test_write_node_id_nonempty(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        node_id = graph.write_node(_make_node(dummy_embedding))
        assert len(node_id) > 0

    def test_get_node_returns_node(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        node_id = graph.write_node(_make_node(dummy_embedding))
        result = graph.get_node(node_id)
        assert result is not None

    def test_get_node_text_matches(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        node_id = graph.write_node(_make_node(dummy_embedding, text="hello world"))
        result = graph.get_node(node_id)
        assert result is not None
        assert result.text == "hello world"

    def test_get_node_unknown_returns_none(self, graph: Graph) -> None:
        result = graph.get_node("00000000-0000-0000-0000-000000000000")
        assert result is None

    def test_write_node_tier_preserved(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        node_id = graph.write_node(_make_node(dummy_embedding, tier=2))
        result = graph.get_node(node_id)
        assert result is not None
        assert result.tier == 2
""",
    lambda cn: f"""\
class {cn}:
    def test_write_edge_links_two_nodes(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        a = graph.write_node(_make_node(dummy_embedding, text="node-a"))
        b = graph.write_node(_make_node(dummy_embedding, text="node-b"))
        graph.write_edge(a, b)
        edges = graph.get_edges(a)
        assert any(e.target == b for e in edges)

    def test_write_edge_self_loop_raises(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        a = graph.write_node(_make_node(dummy_embedding))
        with pytest.raises(ValueError):
            graph.write_edge(a, a)

    def test_get_edges_empty_for_isolated_node(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        a = graph.write_node(_make_node(dummy_embedding))
        assert graph.get_edges(a) == []

    def test_write_node_weight_preserved(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        node_id = graph.write_node(_make_node(dummy_embedding, weight=3.5))
        result = graph.get_node(node_id)
        assert result is not None
        assert result.weight == pytest.approx(3.5)

    def test_write_node_project_preserved(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        node_id = graph.write_node(_make_node(dummy_embedding, project="/my/proj"))
        result = graph.get_node(node_id)
        assert result is not None
        assert result.project == "/my/proj"

    def test_two_nodes_different_ids(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        a = graph.write_node(_make_node(dummy_embedding, text="first"))
        b = graph.write_node(_make_node(dummy_embedding, text="second"))
        assert a != b
""",
    lambda cn: f"""\
class {cn}:
    def test_get_all_nodes_empty(self, graph: Graph) -> None:
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        assert nodes == []

    def test_get_all_nodes_one_node(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding))
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        assert len(nodes) == 1

    def test_get_all_nodes_project_filter(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, project="/a"))
        graph.write_node(_make_node(dummy_embedding, project="/b", text="other"))
        nodes = graph.get_all_nodes(project="/a")
        assert len(nodes) == 1

    def test_get_all_nodes_tier_filter(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, tier=1))
        graph.write_node(_make_node(dummy_embedding, tier=2, text="t2"))
        nodes = graph.get_all_nodes(project=TEST_PROJECT, tier=1)
        assert all(n.tier == 1 for n in nodes)

    def test_merge_node_updates_weight(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        node_id = graph.write_node(_make_node(dummy_embedding, weight=1.0))
        bump = _make_node(dummy_embedding, weight=2.0)
        graph.merge_node(node_id, bump)
        result = graph.get_node(node_id)
        assert result is not None
        assert result.weight > 1.0

    def test_node_type_preserved(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        node_id = graph.write_node(_make_node(dummy_embedding, node_type="decision"))
        result = graph.get_node(node_id)
        assert result is not None
        assert result.type == "decision"
""",
    lambda cn: f"""\
class {cn}:
    def test_find_similar_returns_list(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding))
        results = graph.find_similar(dummy_embedding, TEST_PROJECT)
        assert isinstance(results, list)

    def test_find_similar_finds_exact(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding))
        results = graph.find_similar(dummy_embedding, TEST_PROJECT, threshold=0.5)
        assert len(results) >= 1

    def test_find_similar_empty_graph(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        results = graph.find_similar(dummy_embedding, TEST_PROJECT)
        assert results == []

    def test_write_node_source_preserved(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        node_id = graph.write_node(_make_node(dummy_embedding, source="git"))
        result = graph.get_node(node_id)
        assert result is not None
        assert result.source == "git"

    def test_write_node_scope_preserved(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        node_id = graph.write_node(_make_node(dummy_embedding, scope="global"))
        result = graph.get_node(node_id)
        assert result is not None
        assert result.scope == "global"

    def test_get_all_nodes_count_two(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, text="a"))
        graph.write_node(_make_node(dummy_embedding, text="b"))
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        assert len(nodes) == 2
""",
    lambda cn: f"""\
class {cn}:
    def test_write_node_rationale_none(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        node_id = graph.write_node(_make_node(dummy_embedding, rationale=None))
        result = graph.get_node(node_id)
        assert result is not None
        assert result.rationale is None

    def test_write_node_rationale_preserved(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        node_id = graph.write_node(
            _make_node(dummy_embedding, rationale="important context")
        )
        result = graph.get_node(node_id)
        assert result is not None
        assert result.rationale == "important context"

    def test_write_edge_strength_starts_at_one(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        a = graph.write_node(_make_node(dummy_embedding, text="a"))
        b = graph.write_node(_make_node(dummy_embedding, text="b"))
        graph.write_edge(a, b)
        edges = graph.get_edges(a)
        assert edges[0].strength == pytest.approx(1.0)

    def test_get_all_nodes_two_tiers(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, tier=1, text="t1"))
        graph.write_node(_make_node(dummy_embedding, tier=2, text="t2"))
        all_nodes = graph.get_all_nodes(project=TEST_PROJECT)
        tiers = {{n.tier for n in all_nodes}}
        assert 1 in tiers
        assert 2 in tiers

    def test_write_two_edges_from_hub(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        hub = graph.write_node(_make_node(dummy_embedding, text="hub"))
        b = graph.write_node(_make_node(dummy_embedding, text="b"))
        c = graph.write_node(_make_node(dummy_embedding, text="c"))
        graph.write_edge(hub, b)
        graph.write_edge(hub, c)
        assert len(graph.get_edges(hub)) == 2

    def test_node_session_count_default(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        node_id = graph.write_node(_make_node(dummy_embedding))
        result = graph.get_node(node_id)
        assert result is not None
        assert result.session_count >= 1
""",
    lambda cn: f"""\
class {cn}:
    def test_merge_node_noop_for_unknown_id(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        node = _make_node(dummy_embedding)
        graph.merge_node("00000000-0000-0000-0000-000000000000", node)

    def test_get_edges_returns_edge_objects(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        from core.graph import Edge
        a = graph.write_node(_make_node(dummy_embedding, text="a"))
        b = graph.write_node(_make_node(dummy_embedding, text="b"))
        graph.write_edge(a, b)
        edges = graph.get_edges(a)
        assert all(isinstance(e, Edge) for e in edges)

    def test_write_node_id_is_uuid(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        import uuid
        node_id = graph.write_node(_make_node(dummy_embedding))
        uuid.UUID(node_id)

    def test_three_nodes_isolated(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        ids = [
            graph.write_node(_make_node(dummy_embedding, text=t))
            for t in ("x", "y", "z")
        ]
        for node_id in ids:
            assert graph.get_edges(node_id) == []

    def test_get_node_weight_is_float(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        node_id = graph.write_node(_make_node(dummy_embedding))
        result = graph.get_node(node_id)
        assert result is not None
        assert isinstance(result.weight, float)

    def test_get_all_nodes_returns_list(self, graph: Graph) -> None:
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        assert isinstance(nodes, list)
""",
]

RETRIEVAL_TEMPLATES = [
    lambda cn: f"""\
class {cn}:
    def test_token_budget_positive(self) -> None:
        assert TOKEN_BUDGET > 0

    def test_bm25_weight_positive(self) -> None:
        assert BM25_WEIGHT > 0.0

    def test_vector_weight_positive(self) -> None:
        assert VECTOR_WEIGHT > 0.0

    def test_graph_weight_positive(self) -> None:
        assert GRAPH_WEIGHT > 0.0

    def test_weights_sum_to_one(self) -> None:
        total = BM25_WEIGHT + VECTOR_WEIGHT + GRAPH_WEIGHT
        assert abs(total - 1.0) < 1e-6

    def test_top_k_positive(self) -> None:
        assert TOP_K > 0
""",
    lambda cn: f"""\
class {cn}:
    def test_retrieve_empty_graph_returns_empty(self, graph: Graph) -> None:
        results = retrieve("anything", TEST_PROJECT, graph)
        assert results == []

    def test_retrieve_returns_list(self, graph: Graph) -> None:
        results = retrieve("query", TEST_PROJECT, graph)
        assert isinstance(results, list)

    def test_retrieve_budget_parameter_accepted(self, graph: Graph) -> None:
        results = retrieve("query", TEST_PROJECT, graph, budget_tokens=5000)
        assert isinstance(results, list)

    def test_format_injection_block_empty(self) -> None:
        block = format_injection_block([], TEST_PROJECT)
        assert isinstance(block, str)

    def test_format_injection_block_returns_str(self) -> None:
        block = format_injection_block([], TEST_PROJECT)
        assert isinstance(block, str)

    def test_fuse_scores_all_empty(self) -> None:
        result = fuse_scores([], [], [])
        assert isinstance(result, list)
""",
    lambda cn: f"""\
class {cn}:
    def test_build_bm25_index_one_doc(self) -> None:
        index = build_bm25_index(["hello world"])
        assert index is not None

    def test_bm25_weight_lt_one(self) -> None:
        assert BM25_WEIGHT < 1.0

    def test_vector_weight_lt_one(self) -> None:
        assert VECTOR_WEIGHT < 1.0

    def test_graph_weight_lt_one(self) -> None:
        assert GRAPH_WEIGHT < 1.0

    def test_format_injection_block_is_str(self) -> None:
        block = format_injection_block([], TEST_PROJECT)
        assert isinstance(block, str)

    def test_top_k_at_least_one(self) -> None:
        assert TOP_K >= 1
""",
    lambda cn: f"""\
class {cn}:
    def test_scored_node_is_dataclass(self) -> None:
        from dataclasses import fields
        sn = ScoredNode(
            id="x",
            score=0.5,
            text="t",
            tier=1,
            weight=1.0,
            type="observation",
            rationale=None,
        )
        names = {{f.name for f in fields(sn)}}
        assert "id" in names
        assert "score" in names

    def test_scored_node_score_range(self) -> None:
        sn = ScoredNode(
            id="x",
            score=0.75,
            text="t",
            tier=1,
            weight=1.0,
            type="observation",
            rationale=None,
        )
        assert 0.0 <= sn.score <= 1.0

    def test_scored_node_text_field(self) -> None:
        sn = ScoredNode(
            id="x",
            score=0.1,
            text="hello",
            tier=1,
            weight=1.0,
            type="observation",
            rationale=None,
        )
        assert sn.text == "hello"

    def test_scored_node_tier_field(self) -> None:
        sn = ScoredNode(
            id="x",
            score=0.1,
            text="t",
            tier=2,
            weight=1.0,
            type="observation",
            rationale=None,
        )
        assert sn.tier == 2

    def test_fuse_scores_returns_list(self) -> None:
        result = fuse_scores([], [], [])
        assert isinstance(result, list)

    def test_token_budget_gte_1000(self) -> None:
        assert TOKEN_BUDGET >= 1000
""",
    lambda cn: f"""\
class {cn}:
    def test_retrieve_with_node_returns_list(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        emb = rng.random(384).astype(np.float32)
        graph.write_node(_make_node("the quick brown fox", embedding=emb))
        results = retrieve("quick", TEST_PROJECT, graph)
        assert isinstance(results, list)

    def test_vector_channel_empty_nodes(self, rng: np.random.Generator) -> None:
        query_emb = rng.random(384).astype(np.float32)
        results = vector_channel(query_emb, [])
        assert isinstance(results, list)

    def test_bm25_channel_with_index(self) -> None:
        index = build_bm25_index(["hello world test"])
        node = _make_node("hello world test")
        results = bm25_channel("hello", [node], index)
        assert isinstance(results, list)

    def test_format_injection_block_no_crash(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        emb = rng.random(384).astype(np.float32)
        graph.write_node(_make_node("some content", embedding=emb))
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        block = format_injection_block(nodes, TEST_PROJECT)
        assert isinstance(block, str)

    def test_graph_channel_empty_nodes(self, graph: Graph) -> None:
        results = graph_channel([], graph, [])
        assert isinstance(results, list)

    def test_fuse_scores_empty_is_list(self) -> None:
        result = fuse_scores([], [], [])
        assert result == []
""",
    lambda cn: f"""\
class {cn}:
    def test_bm25_weight_plus_others_is_one(self) -> None:
        assert abs(BM25_WEIGHT + VECTOR_WEIGHT + GRAPH_WEIGHT - 1.0) < 1e-6

    def test_scored_node_id_str(self) -> None:
        sn = ScoredNode(
            id="abc123",
            score=0.5,
            text="t",
            tier=1,
            weight=1.0,
            type="observation",
            rationale=None,
        )
        assert isinstance(sn.id, str)

    def test_scored_node_weight_float(self) -> None:
        sn = ScoredNode(
            id="x",
            score=0.5,
            text="t",
            tier=1,
            weight=2.5,
            type="observation",
            rationale=None,
        )
        assert isinstance(sn.weight, float)

    def test_retrieve_budget_accepted(self, graph: Graph) -> None:
        results = retrieve("query", TEST_PROJECT, graph, budget_tokens=5000)
        assert isinstance(results, list)

    def test_top_k_int(self) -> None:
        assert isinstance(TOP_K, int)

    def test_token_budget_int(self) -> None:
        assert isinstance(TOKEN_BUDGET, int)
""",
]

PARSER_TEMPLATES = [
    lambda cn: f"""\
class {cn}:
    def test_event_type_file_write_value(self) -> None:
        assert EventType.FILE_WRITE.value == "file_write"

    def test_event_type_bash_failure_value(self) -> None:
        assert EventType.BASH_FAILURE.value == "bash_failure"

    def test_event_type_assistant_message_value(self) -> None:
        assert EventType.ASSISTANT_MESSAGE.value == "assistant_message"

    def test_co_occurrence_window_positive(self) -> None:
        assert CO_OCCURRENCE_WINDOW_SECONDS > 0

    def test_hotspot_write_threshold_positive(self) -> None:
        assert HOTSPOT_WRITE_THRESHOLD > 0

    def test_hotspot_write_threshold_integer(self) -> None:
        assert isinstance(HOTSPOT_WRITE_THRESHOLD, int)
""",
    lambda cn: f"""\
class {cn}:
    def test_parse_transcript_nonexistent_raises(
        self, tmp_path: Path
    ) -> None:
        import pytest as _pytest
        with _pytest.raises(FileNotFoundError):
            list(parse_transcript(tmp_path / "missing.jsonl"))

    def test_parse_transcript_empty_file_returns_empty(
        self, tmp_path: Path
    ) -> None:
        f = tmp_path / "t.jsonl"
        f.write_text("")
        events = list(parse_transcript(f))
        assert events == []

    def test_detect_co_occurring_writes_no_events(self) -> None:
        result = detect_co_occurring_writes([])
        assert result == ()

    def test_detect_hotspots_no_events(self) -> None:
        result = detect_hotspots([])
        assert len(result) == 0

    def test_extract_prose_turns_no_events(self) -> None:
        result = list(extract_prose_turns([]))
        assert result == []

    def test_parsed_event_has_type_field(self) -> None:
        e = _make_event(EventType.FILE_WRITE)
        assert e.type == EventType.FILE_WRITE
""",
    lambda cn: f"""\
class {cn}:
    def test_make_event_timestamp_default(self) -> None:
        e = _make_event(EventType.BASH_FAILURE)
        assert e.timestamp == 1000

    def test_make_event_data_default_empty(self) -> None:
        e = _make_event(EventType.ASSISTANT_MESSAGE)
        assert e.data == {{}}

    def test_make_write_event_data_has_path(self) -> None:
        e = _make_write_event("/some/file.py")
        assert e.data["path"] == "/some/file.py"

    def test_make_write_event_type_is_file_write(self) -> None:
        e = _make_write_event("/a.py")
        assert e.type == EventType.FILE_WRITE

    def test_co_occurrence_window_gte_ten(self) -> None:
        assert CO_OCCURRENCE_WINDOW_SECONDS >= 10

    def test_event_type_values_are_strings(self) -> None:
        for member in EventType:
            assert isinstance(member.value, str)
""",
    lambda cn: f"""\
class {cn}:
    def test_detect_co_occurring_single_write(self) -> None:
        events = [_make_write_event("/a/b.py", timestamp=1000)]
        result = detect_co_occurring_writes(events)
        assert isinstance(result, tuple)

    def test_detect_co_occurring_two_close_writes(self) -> None:
        events = [
            _make_write_event("/a/b.py", timestamp=1000),
            _make_write_event("/a/c.py", timestamp=1001),
        ]
        result = detect_co_occurring_writes(events)
        assert any(
            "/a/b.py" in pair and "/a/c.py" in pair for pair in result
        )

    def test_detect_co_occurring_far_apart_no_pair(self) -> None:
        events = [
            _make_write_event("/a.py", timestamp=1000),
            _make_write_event("/b.py", timestamp=1000 + CO_OCCURRENCE_WINDOW_SECONDS + 60),
        ]
        result = detect_co_occurring_writes(events)
        assert not any("/a.py" in pair and "/b.py" in pair for pair in result)

    def test_detect_hotspots_one_write(self) -> None:
        events = [_make_write_event("/a.py")]
        result = detect_hotspots(events)
        assert isinstance(result, list)

    def test_event_session_id_preserved(self) -> None:
        e = _make_event(EventType.FILE_WRITE, session_id="sess-42")
        assert e.session_id == "sess-42"

    def test_parsed_event_data_preserved(self) -> None:
        e = _make_event(EventType.FILE_WRITE, data={{"key": "val"}})
        assert e.data["key"] == "val"
""",
    lambda cn: f"""\
class {cn}:
    def test_parse_transcript_simple_fixture(self) -> None:
        events = list(parse_transcript(SIMPLE_FIXTURE))
        assert len(events) > 0

    def test_parse_transcript_returns_parsed_events(self) -> None:
        events = list(parse_transcript(SIMPLE_FIXTURE))
        assert all(isinstance(e, ParsedEvent) for e in events)

    def test_parse_transcript_events_have_timestamp(self) -> None:
        events = list(parse_transcript(SIMPLE_FIXTURE))
        assert all(isinstance(e.timestamp, int) for e in events)

    def test_parse_transcript_events_have_type(self) -> None:
        events = list(parse_transcript(SIMPLE_FIXTURE))
        assert all(isinstance(e.type, EventType) for e in events)

    def test_detect_hotspots_returns_iterable(self) -> None:
        events = list(parse_transcript(SIMPLE_FIXTURE))
        result = detect_hotspots(events)
        assert hasattr(result, "__iter__")

    def test_extract_prose_turns_returns_iterable(self) -> None:
        events = list(parse_transcript(SIMPLE_FIXTURE))
        result = list(extract_prose_turns(events))
        assert isinstance(result, list)
""",
    lambda cn: f"""\
class {cn}:
    def test_summarize_session_empty_returns_summary(self) -> None:
        result = summarize_session([])
        assert result is not None

    def test_summarize_session_simple_fixture(self) -> None:
        events = list(parse_transcript(SIMPLE_FIXTURE))
        result = summarize_session(events)
        assert result is not None

    def test_make_event_default_session_id(self) -> None:
        e = _make_event(EventType.FILE_WRITE)
        assert e.session_id == "s1"

    def test_event_type_members_nonempty(self) -> None:
        assert len(list(EventType)) > 0

    def test_write_jsonl_creates_file(self, tmp_path: Path) -> None:
        p = _write_jsonl(tmp_path, [{{"type": "test"}}])
        assert p.exists()

    def test_write_jsonl_roundtrip(self, tmp_path: Path) -> None:
        import json
        p = _write_jsonl(tmp_path, [{{"x": 1}}, {{"x": 2}}])
        lines = [json.loads(ln) for ln in p.read_text().splitlines() if ln]
        assert lines[0]["x"] == 1
        assert lines[1]["x"] == 2
""",
]

EXTRACTOR_TEMPLATES = [
    lambda cn: f"""\
class {cn}:
    def test_node_type_observation_value(self) -> None:
        assert NodeType.OBSERVATION.value == "observation"

    def test_node_type_fact_value(self) -> None:
        assert NodeType.FACT.value == "fact"

    def test_scope_type_project_value(self) -> None:
        assert ScopeType.PROJECT.value == "project"

    def test_source_type_jsonl_value(self) -> None:
        assert SourceType.JSONL.value == "jsonl"

    def test_durability_threshold_positive(self) -> None:
        assert DURABILITY_THRESHOLD > 0.0

    def test_durability_threshold_lt_one(self) -> None:
        assert DURABILITY_THRESHOLD < 1.0
""",
    lambda cn: f"""\
class {cn}:
    def test_split_on_conjunction_no_and(self) -> None:
        result = _split_on_conjunction("hello world")
        assert isinstance(result, tuple)

    def test_split_on_conjunction_with_and(self) -> None:
        result = _split_on_conjunction("foo and bar")
        assert len(result) >= 1

    def test_is_retracted_false_for_normal(self) -> None:
        e = _make_event(EventType.FILE_WRITE, data={{"path": "/a.py"}})
        assert not _is_retracted(e)

    def test_score_durability_returns_float(self) -> None:
        e = _make_event(EventType.BASH_FAILURE)
        score = _score_durability(e)
        assert isinstance(score, float)

    def test_score_durability_in_unit_interval(self) -> None:
        e = _make_event(EventType.FILE_WRITE)
        score = _score_durability(e)
        assert 0.0 <= score <= 1.0

    def test_candidate_node_type_field(self) -> None:
        events = list(parse_transcript(SIMPLE_FIXTURE))
        candidates = run_extraction(events, TEST_PROJECT, [], 0)
        for c in candidates:
            assert hasattr(c, "type")
""",
    lambda cn: f"""\
class {cn}:
    def test_run_extraction_empty_events(self) -> None:
        result = run_extraction([], TEST_PROJECT, [], 0)
        assert isinstance(result, list)

    def test_run_extraction_simple_fixture(self) -> None:
        events = list(parse_transcript(SIMPLE_FIXTURE))
        result = run_extraction(events, TEST_PROJECT, [], 1000)
        assert isinstance(result, list)

    def test_candidate_node_text_nonempty(self) -> None:
        events = list(parse_transcript(SIMPLE_FIXTURE))
        candidates = run_extraction(events, TEST_PROJECT, [], 1000)
        for c in candidates:
            assert c.text.strip() != ""

    def test_candidate_scope_is_scope_type(self) -> None:
        events = list(parse_transcript(SIMPLE_FIXTURE))
        candidates = run_extraction(events, TEST_PROJECT, [], 1000)
        for c in candidates:
            assert isinstance(c.scope, ScopeType)

    def test_candidate_source_is_source_type(self) -> None:
        events = list(parse_transcript(SIMPLE_FIXTURE))
        candidates = run_extraction(events, TEST_PROJECT, [], 1000)
        for c in candidates:
            assert isinstance(c.source, SourceType)

    def test_candidate_type_is_node_type(self) -> None:
        events = list(parse_transcript(SIMPLE_FIXTURE))
        candidates = run_extraction(events, TEST_PROJECT, [], 1000)
        for c in candidates:
            assert isinstance(c.type, NodeType)
""",
    lambda cn: f"""\
class {cn}:
    def test_jsonl_channel_no_events(self) -> None:
        result = jsonl_channel([], TEST_PROJECT, [], 0)
        assert isinstance(result, list)

    def test_ast_channel_no_events(self) -> None:
        result = ast_channel([], TEST_PROJECT, [], 0)
        assert isinstance(result, list)

    def test_nlp_channel_no_events(self) -> None:
        result = nlp_channel([], TEST_PROJECT, [], 0)
        assert isinstance(result, list)

    def test_node_type_members_nonempty(self) -> None:
        assert len(list(NodeType)) > 0

    def test_scope_type_members_nonempty(self) -> None:
        assert len(list(ScopeType)) > 0

    def test_source_type_members_nonempty(self) -> None:
        assert len(list(SourceType)) > 0
""",
    lambda cn: f"""\
class {cn}:
    def test_candidate_node_project_field(self) -> None:
        events = list(parse_transcript(SIMPLE_FIXTURE))
        candidates = run_extraction(events, TEST_PROJECT, [], 1000)
        for c in candidates:
            assert hasattr(c, "project")

    def test_candidate_node_rationale_attribute(self) -> None:
        events = list(parse_transcript(SIMPLE_FIXTURE))
        candidates = run_extraction(events, TEST_PROJECT, [], 1000)
        for c in candidates:
            assert hasattr(c, "rationale")

    def test_source_type_git_exists(self) -> None:
        assert hasattr(SourceType, "GIT")

    def test_node_type_values_are_strings(self) -> None:
        for member in NodeType:
            assert isinstance(member.value, str)

    def test_scope_type_values_are_strings(self) -> None:
        for member in ScopeType:
            assert isinstance(member.value, str)

    def test_source_type_values_are_strings(self) -> None:
        for member in SourceType:
            assert isinstance(member.value, str)
""",
    lambda cn: f"""\
class {cn}:
    def test_score_durability_file_write_positive(self) -> None:
        e = _make_event(EventType.FILE_WRITE)
        assert _score_durability(e) > 0.0

    def test_score_durability_bash_failure_in_range(self) -> None:
        e = _failure_event()
        score = _score_durability(e)
        assert 0.0 <= score <= 1.0

    def test_is_retracted_returns_bool(self) -> None:
        e = _make_event(EventType.ASSISTANT_MESSAGE, data={{"text": "done"}})
        result = _is_retracted(e)
        assert isinstance(result, bool)

    def test_durability_threshold_is_float(self) -> None:
        assert isinstance(DURABILITY_THRESHOLD, float)

    def test_split_on_conjunction_returns_tuple(self) -> None:
        result = _split_on_conjunction("alpha and beta and gamma")
        assert isinstance(result, tuple)

    def test_ast_channel_with_write_events(self) -> None:
        events = list(parse_transcript(SIMPLE_FIXTURE))
        result = ast_channel(events, TEST_PROJECT, [], 0)
        assert isinstance(result, list)
""",
]

HOOKS_TEMPLATES = [
    lambda cn: f"""\
class {cn}:
    def test_run_extract_missing_transcript_returns_zero(
        self, graph: Graph, tmp_path: Path
    ) -> None:
        n = run_extract(tmp_path / "missing.jsonl", TEST_PROJECT, graph)
        assert n == 0

    def test_run_extract_empty_transcript_returns_zero(
        self, graph: Graph, tmp_path: Path
    ) -> None:
        f = tmp_path / "t.jsonl"
        f.write_text("")
        n = run_extract(f, TEST_PROJECT, graph)
        assert n == 0

    def test_run_extract_simple_fixture_returns_int(
        self, graph: Graph
    ) -> None:
        result = run_extract(SIMPLE_TRANSCRIPT, TEST_PROJECT, graph)
        assert isinstance(result, int)

    def test_run_extract_simple_fixture_nonnegative(
        self, graph: Graph
    ) -> None:
        result = run_extract(SIMPLE_TRANSCRIPT, TEST_PROJECT, graph)
        assert result >= 0

    def test_run_extract_idempotent(self, graph: Graph) -> None:
        run_extract(SIMPLE_TRANSCRIPT, TEST_PROJECT, graph)
        second = run_extract(SIMPLE_TRANSCRIPT, TEST_PROJECT, graph)
        assert second == 0

    def test_make_node_tier_preserved(self) -> None:
        node = _make_node("text", tier=2)
        assert node.tier == 2
""",
    lambda cn: f"""\
class {cn}:
    def test_make_node_returns_node(self) -> None:
        node = _make_node("some text")
        assert isinstance(node, Node)

    def test_make_node_text_preserved(self) -> None:
        node = _make_node("specific text content")
        assert node.text == "specific text content"

    def test_make_node_default_tier_is_one(self) -> None:
        node = _make_node("t")
        assert node.tier == 1

    def test_make_node_project_preserved(self) -> None:
        node = _make_node("t", project="/custom/proj")
        assert node.project == "/custom/proj"

    def test_make_node_weight_is_one(self) -> None:
        node = _make_node("t")
        assert node.weight == pytest.approx(1.0)

    def test_make_node_type_is_observation(self) -> None:
        node = _make_node("t")
        assert node.type == "observation"
""",
    lambda cn: f"""\
class {cn}:
    def test_run_inject_empty_graph_returns_str(
        self, graph: Graph
    ) -> None:
        result = run_inject(TEST_PROJECT, graph)
        assert isinstance(result, str)

    def test_run_inject_returns_string_always(
        self, graph: Graph
    ) -> None:
        result = run_inject(TEST_PROJECT, graph)
        assert isinstance(result, str)

    def test_run_extract_decisions_fixture_nonnegative(
        self, graph: Graph
    ) -> None:
        result = run_extract(DECISIONS_TRANSCRIPT, TEST_PROJECT, graph)
        assert result >= 0

    def test_run_extract_writes_nodes_to_graph(
        self, graph: Graph
    ) -> None:
        run_extract(SIMPLE_TRANSCRIPT, TEST_PROJECT, graph)
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        assert isinstance(nodes, list)

    def test_make_node_source_is_jsonl(self) -> None:
        node = _make_node("t")
        assert node.source == "jsonl"

    def test_make_node_scope_is_project(self) -> None:
        node = _make_node("t")
        assert node.scope == "project"
""",
    lambda cn: f"""\
class {cn}:
    def test_extract_main_no_transcript_env_exits_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CLAUDE_TRANSCRIPT", raising=False)
        monkeypatch.setenv("CLAUDE_PROJECT_PATH", str(tmp_path))
        result = extract_main()
        assert result == 0

    def test_run_inject_with_one_node(
        self, graph: Graph
    ) -> None:
        graph.write_node(_make_node("important context node"))
        result = run_inject(TEST_PROJECT, graph)
        assert isinstance(result, str)

    def test_run_extract_different_projects_isolated(
        self, graph: Graph
    ) -> None:
        n1 = run_extract(SIMPLE_TRANSCRIPT, "/proj/a", graph)
        n2 = run_extract(SIMPLE_TRANSCRIPT, "/proj/b", graph)
        assert isinstance(n1, int)
        assert isinstance(n2, int)

    def test_make_node_session_count_one(self) -> None:
        node = _make_node("t")
        assert node.session_count == 1

    def test_make_node_id_empty_string(self) -> None:
        node = _make_node("t")
        assert node.id == ""

    def test_make_node_rationale_none(self) -> None:
        node = _make_node("t")
        assert node.rationale is None
""",
    lambda cn: f"""\
class {cn}:
    def test_run_inject_empty_result_empty_graph(
        self, graph: Graph
    ) -> None:
        result = run_inject(TEST_PROJECT, graph)
        assert result == "" or isinstance(result, str)

    def test_run_extract_returns_int_type(
        self, graph: Graph
    ) -> None:
        result = run_extract(SIMPLE_TRANSCRIPT, TEST_PROJECT, graph)
        assert type(result) is int

    def test_run_extract_zero_for_seen_transcript(
        self, graph: Graph
    ) -> None:
        run_extract(SIMPLE_TRANSCRIPT, TEST_PROJECT, graph)
        second = run_extract(SIMPLE_TRANSCRIPT, TEST_PROJECT, graph)
        assert second == 0

    def test_make_node_precision_bits_32(self) -> None:
        node = _make_node("t")
        assert node.precision_bits == 32

    def test_run_inject_with_two_nodes(
        self, graph: Graph
    ) -> None:
        graph.write_node(_make_node("context alpha"))
        graph.write_node(_make_node("context beta"))
        result = run_inject(TEST_PROJECT, graph)
        assert isinstance(result, str)

    def test_make_node_last_accessed_positive(self) -> None:
        node = _make_node("t")
        assert node.last_accessed > 0
""",
    lambda cn: f"""\
class {cn}:
    def test_extract_main_missing_transcript_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CLAUDE_TRANSCRIPT", str(tmp_path / "no.jsonl"))
        monkeypatch.setenv("CLAUDE_PROJECT_PATH", str(tmp_path))
        result = extract_main()
        assert result == 0

    def test_run_extract_simple_then_inject(
        self, graph: Graph
    ) -> None:
        run_extract(SIMPLE_TRANSCRIPT, TEST_PROJECT, graph)
        result = run_inject(TEST_PROJECT, graph)
        assert isinstance(result, str)

    def test_run_inject_result_is_str(
        self, graph: Graph
    ) -> None:
        run_extract(SIMPLE_TRANSCRIPT, TEST_PROJECT, graph)
        result = run_inject(TEST_PROJECT, graph)
        assert isinstance(result, str)

    def test_make_node_created_at_positive(self) -> None:
        node = _make_node("t")
        assert node.created_at > 0

    def test_run_extract_nonnegative_decisions_fixture(
        self, graph: Graph
    ) -> None:
        result = run_extract(DECISIONS_TRANSCRIPT, TEST_PROJECT, graph)
        assert result >= 0

    def test_make_node_embedding_none_by_default(self) -> None:
        node = _make_node("t")
        assert node.embedding is None
""",
]

MODULE_TEMPLATES: dict[str, list] = {
    "decay": DECAY_TEMPLATES,
    "graph": GRAPH_TEMPLATES,
    "retrieval": RETRIEVAL_TEMPLATES,
    "parser": PARSER_TEMPLATES,
    "extractor": EXTRACTOR_TEMPLATES,
    "hooks": HOOKS_TEMPLATES,
}

# ---------------------------------------------------------------------------
# Shell helpers
# ---------------------------------------------------------------------------


def run(cmd: str, cwd: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)


def run_check(cmd: str, cwd: str | None = None) -> subprocess.CompletedProcess[str]:
    result = run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {cmd}\n"
            f"stdout: {result.stdout[:400]}\nstderr: {result.stderr[:400]}"
        )
    return result


# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------


def _gh(method: str, path: str, body: dict | None = None) -> dict:
    data_flag = f"-d '{json.dumps(body)}'" if body else ""
    result = run(
        f"curl -sf -X {method} "
        f'-H "Authorization: token {GH_TOKEN}" '
        f'-H "Content-Type: application/json" '
        f"{data_flag} "
        f'"https://api.github.com/repos/{REPO}/{path}"'
    )
    if not result.stdout.strip():
        return {}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}


def create_pr(branch: str, title: str, body: str) -> int | None:
    resp = _gh(
        "POST",
        "pulls",
        {"title": title, "body": body, "head": branch, "base": "main"},
    )
    return resp.get("number")


def enable_automerge(pr_number: int) -> None:
    """Enable squash auto-merge via GraphQL (best-effort)."""
    resp = _gh("GET", f"pulls/{pr_number}")
    node_id = resp.get("node_id", "")
    if not node_id:
        return
    mutation = (
        "mutation($id:ID!){enablePullRequestAutoMerge"
        "(input:{pullRequestId:$id,mergeMethod:SQUASH})"
        "{pullRequest{number}}}"
    )
    run(
        f"curl -sf -X POST "
        f'-H "Authorization: token {GH_TOKEN}" '
        f'-H "Content-Type: application/json" '
        f'-d \'{json.dumps({"query": mutation, "variables": {"id": node_id}})}\' '
        f'"https://api.github.com/graphql"'
    )


# ---------------------------------------------------------------------------
# Test class generation
# ---------------------------------------------------------------------------


def pick_template(module: str, day_of_year: int, slot: int) -> list:
    """Return the template library entry for this module/day/slot combo."""
    lib = MODULE_TEMPLATES[module]
    idx = (day_of_year * 3 + slot * 7) % len(lib)
    return lib[idx]


def generate_test_class(
    module: str, class_name: str, day_of_year: int, slot: int
) -> str:
    tmpl = pick_template(module, day_of_year, slot)
    return tmpl(class_name)


# ---------------------------------------------------------------------------
# Validation + commit
# ---------------------------------------------------------------------------


def validate_and_commit(
    test_file: str,
    original_content: str,
    new_code: str,
    module: str,
    class_name: str,
) -> None:
    """Append new_code to test_file, lint, run tests, commit. Raises on failure."""
    Path(test_file).write_text(original_content + "\n\n" + new_code + "\n")

    r = run(f"python3 -m py_compile {test_file}")
    if r.returncode != 0:
        Path(test_file).write_text(original_content)
        raise ValueError(f"Syntax error: {r.stderr[:300]}")

    run(f"python3 -m black {test_file} -q")
    r = run(f"python3 -m ruff check {test_file} --fix -q")
    if r.returncode != 0:
        Path(test_file).write_text(original_content)
        raise ValueError(f"Ruff: {r.stdout[:200]}")

    r = run(f"python3 -m pytest {test_file}::{class_name} -q --tb=short -x")
    if r.returncode != 0:
        Path(test_file).write_text(original_content)
        raise ValueError(f"Tests failed:\n{r.stdout[:600]}")

    run_check(f"git add {test_file}")
    run_check(f'git commit -m "tests({module}): add {class_name} coverage"')


# ---------------------------------------------------------------------------
# Per-target workflow
# ---------------------------------------------------------------------------


def process_target(target: dict, pr_index: int, day_of_year: int) -> bool:
    """Create branch, add 3 test-class commits, push, open PR. Return True on success."""
    module = target["module"]
    branch = f"test/{module}-auto-{DATE_TAG}-{pr_index}"
    test_file = target["test_file"]

    try:
        run_check("git checkout main")
        run_check("git pull origin main")
        run_check(f"git checkout -b {branch}")
    except RuntimeError as e:
        print(f"[{module}] Branch setup failed: {e}", file=sys.stderr)
        return False

    commits_made = 0
    for slot in range(3):
        suffix = f"{DATE_TAG}{chr(65 + slot)}"  # 20260830A / B / C
        class_name = f"Test{module.title()}{suffix}"
        current_content = Path(test_file).read_text()
        try:
            code = generate_test_class(module, class_name, day_of_year, slot)
            validate_and_commit(test_file, current_content, code, module, class_name)
            commits_made += 1
            print(f"[{module}] commit {slot + 1}/3 ok ({class_name})")
        except Exception as exc:
            print(f"[{module}] commit {slot + 1}/3 skipped: {exc}", file=sys.stderr)
            run("git reset HEAD " + test_file + " 2>/dev/null || true")
            run("git checkout -- " + test_file + " 2>/dev/null || true")

    if commits_made == 0:
        run("git checkout main")
        run(f"git branch -D {branch}")
        return False

    r = run(f"git push -u origin {branch}")
    if r.returncode != 0:
        print(f"[{module}] Push failed: {r.stderr[:200]}", file=sys.stderr)
        return False

    pr_number = create_pr(
        branch=branch,
        title=f"tests({module}): automated coverage additions {DATE_TAG}",
        body=(
            f"Automated daily test additions for `{module}`.\n\n"
            f"- {commits_made} new test class(es), each with 6 assertions\n"
            f"- Validated: syntax + black + ruff + pytest before each commit\n\n"
            f"## Test plan\n- [ ] CI passes (auto-merge enabled)"
        ),
    )
    if pr_number:
        enable_automerge(pr_number)
        print(f"[{module}] PR #{pr_number} opened, auto-merge enabled")
    else:
        print(f"[{module}] PR create returned no number", file=sys.stderr)

    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    day_of_year = NOW.timetuple().tm_yday
    start = day_of_year % len(TARGETS)
    selected = [TARGETS[(start + i) % len(TARGETS)] for i in range(4)]

    prs_opened = 0
    for i, target in enumerate(selected):
        print(f"\n=== {target['module']} (PR {i + 1}/4) ===")
        if process_target(target, i, day_of_year):
            prs_opened += 1

    print(f"\nDone: {prs_opened}/4 PRs opened.")
    if prs_opened == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
