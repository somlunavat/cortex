"""Tests for core/retrieval.py — BM25 + vector + graph-hop retrieval."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import numpy as np
import pytest

from core.graph import Graph, Node
from core.retrieval import (
    BM25_WEIGHT,
    GRAPH_WEIGHT,
    TOKEN_BUDGET,
    TOP_K,
    VECTOR_WEIGHT,
    ScoredNode,
    bm25_channel,
    build_bm25_index,
    format_injection_block,
    fuse_scores,
    graph_channel,
    retrieve,
    vector_channel,
)

TEST_PROJECT = "/tmp/cortex_retrieval_test"


@pytest.fixture
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    schema = Path("schema.sql").read_text()
    conn.executescript(schema)
    return conn


@pytest.fixture
def graph(db: sqlite3.Connection) -> Graph:
    return Graph(connection=db)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=42)


def _make_node(
    text: str,
    tier: int = 1,
    weight: float = 1.0,
    embedding: np.ndarray | None = None,
    project: str = TEST_PROJECT,
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
        created_at=now,
        session_count=1,
    )


def _random_embedding(rng: np.random.Generator) -> np.ndarray:
    return rng.random(384).astype(np.float32)


# ---------------------------------------------------------------------------
# build_bm25_index
# ---------------------------------------------------------------------------


class TestBuildBM25Index:
    def test_returns_bm25_instance(self, rng: np.random.Generator) -> None:
        from rank_bm25 import BM25Okapi

        nodes = [_make_node("asyncpg is fast"), _make_node("JWT auth token")]
        index = build_bm25_index(nodes)
        assert isinstance(index, BM25Okapi)

    def test_scores_matching_node_higher(self) -> None:
        # BM25 IDF requires >2 docs to produce non-zero scores; use 3 nodes.
        nodes = [
            _make_node("asyncpg database connection pool"),
            _make_node("JWT authentication middleware"),
            _make_node("redis cache layer"),
        ]
        index = build_bm25_index(nodes)
        scores = index.get_scores(["asyncpg"])
        assert scores[0] > scores[1]


# ---------------------------------------------------------------------------
# vector_channel
# ---------------------------------------------------------------------------


class TestVectorChannel:
    def test_returns_one_result_per_node(self, rng: np.random.Generator) -> None:
        e = _random_embedding(rng)
        nodes = [_make_node("a", embedding=e), _make_node("b", embedding=e)]
        q = _random_embedding(rng)
        results = vector_channel(q, nodes)
        assert len(results) == 2

    def test_identical_embedding_scores_near_one(
        self, rng: np.random.Generator
    ) -> None:
        e = _random_embedding(rng)
        nodes = [_make_node("exact", embedding=e)]
        results = vector_channel(e, nodes)
        assert results[0].score == pytest.approx(1.0, abs=1e-4)

    def test_node_without_embedding_scores_zero(self, rng: np.random.Generator) -> None:
        nodes = [_make_node("no embedding", embedding=None)]
        q = _random_embedding(rng)
        results = vector_channel(q, nodes)
        assert results[0].score == 0.0

    def test_score_is_non_negative(self, rng: np.random.Generator) -> None:
        e1 = _random_embedding(rng)
        e2 = _random_embedding(rng)
        nodes = [_make_node("x", embedding=e2)]
        results = vector_channel(e1, nodes)
        assert results[0].score >= 0.0

    def test_more_similar_node_scores_higher(self, rng: np.random.Generator) -> None:
        query = _random_embedding(rng)
        similar = query.copy()
        similar += rng.random(384).astype(np.float32) * 0.01
        unrelated = _random_embedding(rng)
        nodes = [
            _make_node("similar", embedding=similar),
            _make_node("unrelated", embedding=unrelated),
        ]
        results = vector_channel(query, nodes)
        sim_score = next(r for r in results if r.node.text == "similar").score
        unr_score = next(r for r in results if r.node.text == "unrelated").score
        assert sim_score > unr_score


# ---------------------------------------------------------------------------
# bm25_channel
# ---------------------------------------------------------------------------


class TestBM25Channel:
    def test_returns_one_result_per_node(self) -> None:
        nodes = [_make_node("asyncpg pool"), _make_node("jwt auth")]
        index = build_bm25_index(nodes)
        results = bm25_channel("asyncpg", nodes, index)
        assert len(results) == 2

    def test_matching_node_scores_higher(self) -> None:
        nodes = [
            _make_node("asyncpg database connection pool"),
            _make_node("JWT authentication middleware"),
            _make_node("redis cache layer"),
        ]
        index = build_bm25_index(nodes)
        results = bm25_channel("asyncpg pool", nodes, index)
        score_asyncpg = next(r.score for r in results if "asyncpg" in r.node.text)
        score_jwt = next(r.score for r in results if "JWT" in r.node.text)
        assert score_asyncpg > score_jwt

    def test_scores_normalised_to_0_1(self) -> None:
        nodes = [
            _make_node("asyncpg database connection pool"),
            _make_node("JWT authentication middleware"),
            _make_node("redis cache layer"),
        ]
        index = build_bm25_index(nodes)
        results = bm25_channel("asyncpg", nodes, index)
        assert all(0.0 <= r.score <= 1.0 for r in results)

    def test_no_match_returns_zero_scores(self) -> None:
        nodes = [_make_node("asyncpg pool"), _make_node("jwt auth")]
        index = build_bm25_index(nodes)
        results = bm25_channel("zzzznotaword", nodes, index)
        assert all(r.score == 0.0 for r in results)

    def test_empty_query_returns_zero_scores(self) -> None:
        nodes = [_make_node("asyncpg pool")]
        index = build_bm25_index(nodes)
        results = bm25_channel("", nodes, index)
        assert all(r.score == 0.0 for r in results)


# ---------------------------------------------------------------------------
# graph_channel
# ---------------------------------------------------------------------------


class TestGraphChannel:
    def test_unreachable_node_scores_zero(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        e = _random_embedding(rng)
        n1_id = graph.write_node(_make_node("seed", embedding=e))
        n2_id = graph.write_node(_make_node("island", embedding=e))
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        seeds = [n for n in nodes if n.id == n1_id]
        results = graph_channel(seeds, graph, nodes)
        island = next(r for r in results if r.node.id == n2_id)
        assert island.score == 0.0

    def test_one_hop_neighbour_scores_half(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        e = _random_embedding(rng)
        n1_id = graph.write_node(_make_node("seed", embedding=e))
        n2_id = graph.write_node(_make_node("neighbour", embedding=e))
        graph.write_edge(n1_id, n2_id)
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        seeds = [n for n in nodes if n.id == n1_id]
        results = graph_channel(seeds, graph, nodes)
        neighbour = next(r for r in results if r.node.id == n2_id)
        assert neighbour.score == pytest.approx(0.5, abs=1e-6)

    def test_two_hop_neighbour_scores_quarter(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        e = _random_embedding(rng)
        n1_id = graph.write_node(_make_node("seed", embedding=e))
        n2_id = graph.write_node(_make_node("hop1", embedding=e))
        n3_id = graph.write_node(_make_node("hop2", embedding=e))
        graph.write_edge(n1_id, n2_id)
        graph.write_edge(n2_id, n3_id)
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        seeds = [n for n in nodes if n.id == n1_id]
        results = graph_channel(seeds, graph, nodes)
        hop2 = next(r for r in results if r.node.id == n3_id)
        assert hop2.score == pytest.approx(0.25, abs=1e-6)

    def test_seed_itself_not_boosted(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        e = _random_embedding(rng)
        n1_id = graph.write_node(_make_node("seed", embedding=e))
        graph.write_edge(n1_id, n1_id) if False else None  # no self-loops
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        seeds = [n for n in nodes if n.id == n1_id]
        results = graph_channel(seeds, graph, nodes)
        seed_result = next(r for r in results if r.node.id == n1_id)
        assert seed_result.score == 0.0

    def test_returns_one_result_per_node(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        e = _random_embedding(rng)
        for i in range(4):
            graph.write_node(_make_node(f"node {i}", embedding=e))
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        results = graph_channel(nodes[:1], graph, nodes)
        assert len(results) == len(nodes)


# ---------------------------------------------------------------------------
# fuse_scores
# ---------------------------------------------------------------------------


class TestFuseScores:
    def _make_scored(self, nodes: list[Node], scores: list[float]) -> list[ScoredNode]:
        return [ScoredNode(node=n, score=s) for n, s in zip(nodes, scores, strict=True)]

    def test_returns_correct_count(self, rng: np.random.Generator) -> None:
        nodes = [_make_node(f"n{i}") for i in range(3)]
        v = self._make_scored(nodes, [0.8, 0.5, 0.2])
        b = self._make_scored(nodes, [0.6, 0.3, 0.1])
        g = self._make_scored(nodes, [0.0, 0.5, 0.0])
        result = fuse_scores(v, b, g)
        assert len(result) == 3

    def test_sorted_descending(self, rng: np.random.Generator) -> None:
        nodes = [_make_node(f"n{i}") for i in range(3)]
        v = self._make_scored(nodes, [0.1, 0.9, 0.5])
        b = self._make_scored(nodes, [0.1, 0.8, 0.4])
        g = self._make_scored(nodes, [0.0, 0.0, 0.0])
        result = fuse_scores(v, b, g)
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_weights_applied_correctly(self) -> None:
        node = _make_node("test")
        v = [ScoredNode(node=node, score=1.0)]
        b = [ScoredNode(node=node, score=1.0)]
        g = [ScoredNode(node=node, score=1.0)]
        result = fuse_scores(v, b, g)
        expected = VECTOR_WEIGHT + BM25_WEIGHT + GRAPH_WEIGHT
        assert result[0].score == pytest.approx(expected, abs=1e-6)

    def test_zero_scores_produce_zero_fused(self) -> None:
        node = _make_node("zero")
        v = [ScoredNode(node=node, score=0.0)]
        b = [ScoredNode(node=node, score=0.0)]
        g = [ScoredNode(node=node, score=0.0)]
        result = fuse_scores(v, b, g)
        assert result[0].score == 0.0


# ---------------------------------------------------------------------------
# retrieve
# ---------------------------------------------------------------------------


class TestRetrieve:
    def test_empty_graph_returns_empty(self, graph: Graph) -> None:
        result = retrieve("anything", TEST_PROJECT, graph)
        assert result == []

    def test_tier3_always_returned(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        e = _random_embedding(rng)
        graph.write_node(_make_node("convention node", tier=3, embedding=e))
        result = retrieve("something unrelated", TEST_PROJECT, graph)
        assert any(n.tier == 3 for n in result)

    def test_tier3_returned_even_with_no_candidates(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        e = _random_embedding(rng)
        graph.write_node(_make_node("always apply this", tier=3, embedding=e))
        result = retrieve("query", TEST_PROJECT, graph)
        tier3_nodes = [n for n in result if n.tier == 3]
        assert len(tier3_nodes) == 1

    def test_result_count_capped_at_top_k(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        for i in range(TOP_K + 5):
            e = _random_embedding(rng)
            graph.write_node(_make_node(f"node {i}", tier=1, embedding=e))
        result = retrieve("test query", TEST_PROJECT, graph)
        non_tier3 = [n for n in result if n.tier < 3]
        assert len(non_tier3) <= TOP_K

    def test_only_target_project_returned(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        e = _random_embedding(rng)
        graph.write_node(
            _make_node("target project node", project=TEST_PROJECT, embedding=e)
        )
        graph.write_node(
            _make_node("other project node", project="/other", embedding=e)
        )
        result = retrieve("query", TEST_PROJECT, graph)
        assert all(n.project == TEST_PROJECT for n in result)

    def test_token_budget_enforced(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        # Add many nodes so the budget pressure is felt
        for i in range(20):
            e = _random_embedding(rng)
            graph.write_node(
                _make_node(
                    f"very important context about topic {i}", tier=1, embedding=e
                )
            )
        result = retrieve("query", TEST_PROJECT, graph, budget_tokens=50)
        total_tokens = sum(len(enc.encode(n.text)) for n in result)
        assert total_tokens <= 50

    def test_returns_list_of_nodes(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        e = _random_embedding(rng)
        graph.write_node(_make_node("a node", embedding=e))
        result = retrieve("query", TEST_PROJECT, graph)
        assert isinstance(result, list)
        assert all(isinstance(n, Node) for n in result)


# ---------------------------------------------------------------------------
# format_injection_block
# ---------------------------------------------------------------------------


class TestFormatInjectionBlock:
    def test_contains_project_path(self, rng: np.random.Generator) -> None:
        e = _random_embedding(rng)
        nodes = [_make_node("use async/await", tier=3, embedding=e)]
        block = format_injection_block(nodes, TEST_PROJECT)
        assert TEST_PROJECT in block

    def test_tier3_in_conventions_section(self, rng: np.random.Generator) -> None:
        e = _random_embedding(rng)
        nodes = [_make_node("always use async", tier=3, embedding=e)]
        block = format_injection_block(nodes, TEST_PROJECT)
        assert "[CONVENTIONS" in block
        assert "always use async" in block

    def test_tier1_in_context_section(self, rng: np.random.Generator) -> None:
        e = _random_embedding(rng)
        nodes = [_make_node("auth middleware modified", tier=1, embedding=e)]
        block = format_injection_block(nodes, TEST_PROJECT)
        assert "[RELEVANT CONTEXT" in block
        assert "auth middleware modified" in block

    def test_header_and_footer_present(self, rng: np.random.Generator) -> None:
        e = _random_embedding(rng)
        nodes = [_make_node("some context", tier=1, embedding=e)]
        block = format_injection_block(nodes, TEST_PROJECT)
        assert block.startswith("=== CORTEX MEMORY")
        assert "END CORTEX MEMORY" in block

    def test_footer_contains_token_count(self, rng: np.random.Generator) -> None:
        e = _random_embedding(rng)
        nodes = [_make_node("a fact", tier=1, embedding=e)]
        block = format_injection_block(nodes, TEST_PROJECT)
        assert "tokens:" in block

    def test_empty_node_list(self) -> None:
        block = format_injection_block([], TEST_PROJECT)
        assert TEST_PROJECT in block
        assert "END CORTEX MEMORY" in block

    def test_bullet_prefix_on_each_node(self, rng: np.random.Generator) -> None:
        e = _random_embedding(rng)
        nodes = [
            _make_node("convention one", tier=3, embedding=e),
            _make_node("context one", tier=1, embedding=e),
        ]
        block = format_injection_block(nodes, TEST_PROJECT)
        assert "• convention one" in block
        assert "• context one" in block

    def test_token_count_within_budget(self, rng: np.random.Generator) -> None:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        e = _random_embedding(rng)
        nodes = [_make_node("short fact", tier=1, embedding=e)]
        block = format_injection_block(nodes, TEST_PROJECT)
        assert len(enc.encode(block)) < TOKEN_BUDGET * 2  # sanity check, not strict

    def test_tier2_in_context_section(self, rng: np.random.Generator) -> None:
        e = _random_embedding(rng)
        nodes = [_make_node("jwt used for auth", tier=2, embedding=e)]
        block = format_injection_block(nodes, TEST_PROJECT)
        assert "[RELEVANT CONTEXT" in block
        assert "jwt used for auth" in block

    def test_rationale_shown_in_conventions(self, rng: np.random.Generator) -> None:
        e = _random_embedding(rng)
        now = int(time.time())
        node = Node(
            id="",
            type="convention",
            tier=3,
            text="always use asyncio for I/O",
            rationale="it avoids thread-safety issues",
            embedding=e,
            precision_bits=32,
            weight=5.0,
            project=TEST_PROJECT,
            scope="project",
            source="nlp",
            last_accessed=now,
            created_at=now,
            session_count=1,
        )
        block = format_injection_block([node], TEST_PROJECT)
        assert "it avoids thread-safety issues" in block
        assert "always use asyncio for I/O (it avoids thread-safety issues)" in block

    def test_rationale_shown_in_context(self, rng: np.random.Generator) -> None:
        e = _random_embedding(rng)
        now = int(time.time())
        node = Node(
            id="",
            type="fact",
            tier=1,
            text="switched to sqlalchemy ORM",
            rationale="reduces boilerplate for complex joins",
            embedding=e,
            precision_bits=32,
            weight=1.0,
            project=TEST_PROJECT,
            scope="module",
            source="nlp",
            last_accessed=now,
            created_at=now,
            session_count=1,
        )
        block = format_injection_block([node], TEST_PROJECT)
        assert "reduces boilerplate for complex joins" in block

    def test_null_rationale_not_shown(self, rng: np.random.Generator) -> None:
        e = _random_embedding(rng)
        nodes = [_make_node("auth module refactored", tier=1, embedding=e)]
        block = format_injection_block(nodes, TEST_PROJECT)
        assert "(None)" not in block
        assert (
            "auth module refactored\n" in block
            or "auth module refactored)" not in block
        )
