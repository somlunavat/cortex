"""Tests for core/graph.py — SQLite graph read/write/merge layer."""

from __future__ import annotations

import sqlite3
import time

import numpy as np
import pytest

from core.graph import Graph, Node

TEST_PROJECT = "/tmp/cortex_test_project"


@pytest.fixture
def graph(db: sqlite3.Connection) -> Graph:
    return Graph(connection=db)


@pytest.fixture
def alt_embedding() -> np.ndarray:
    rng = np.random.default_rng(seed=99)
    return rng.random(384).astype(np.float32)


def _make_node(
    embedding: np.ndarray,
    text: str = "auth/middleware.py handles JWT validation",
    node_type: str = "observation",
    project: str = TEST_PROJECT,
    tier: int = 1,
    scope: str = "project",
    source: str = "jsonl",
    weight: float = 1.0,
    rationale: str | None = None,
) -> Node:
    now = int(time.time())
    return Node(
        id="",
        type=node_type,
        tier=tier,
        text=text,
        rationale=rationale,
        embedding=embedding,
        precision_bits=32,
        weight=weight,
        project=project,
        scope=scope,
        source=source,
        last_accessed=now,
        created_at=now,
        session_count=1,
    )


# ---------------------------------------------------------------------------
# write_node
# ---------------------------------------------------------------------------


class TestWriteNode:
    def test_returns_uuid(self, graph: Graph, dummy_embedding: np.ndarray) -> None:
        node = _make_node(dummy_embedding)
        node_id = graph.write_node(node)
        assert len(node_id) == 36
        assert node_id.count("-") == 4

    def test_creates_row_in_db(self, graph: Graph, dummy_embedding: np.ndarray) -> None:
        node = _make_node(dummy_embedding)
        node_id = graph.write_node(node)
        rows = graph.get_all_nodes(project=TEST_PROJECT)
        assert len(rows) == 1
        assert rows[0].id == node_id

    def test_all_schema_fields_populated(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        node = _make_node(dummy_embedding, text="test node", node_type="fact")
        node_id = graph.write_node(node)
        rows = graph.get_all_nodes(project=TEST_PROJECT)
        n = rows[0]
        assert n.id == node_id
        assert n.type == "fact"
        assert n.tier == 1
        assert n.text == "test node"
        assert n.precision_bits == 32
        assert n.weight == 1.0
        assert n.project == TEST_PROJECT
        assert n.scope == "project"
        assert n.source == "jsonl"
        assert n.session_count == 1
        assert n.last_accessed > 0
        assert n.created_at > 0

    def test_embedding_round_trips(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        node = _make_node(dummy_embedding)
        graph.write_node(node)
        rows = graph.get_all_nodes(project=TEST_PROJECT)
        recovered = rows[0].embedding
        assert recovered is not None
        np.testing.assert_allclose(recovered, dummy_embedding, atol=1e-5)

    def test_rationale_nullable(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        node = _make_node(dummy_embedding, rationale=None)
        graph.write_node(node)
        rows = graph.get_all_nodes(project=TEST_PROJECT)
        assert rows[0].rationale is None

    def test_rationale_stored_when_provided(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        node = _make_node(dummy_embedding, rationale="because performance")
        graph.write_node(node)
        rows = graph.get_all_nodes(project=TEST_PROJECT)
        assert rows[0].rationale == "because performance"

    @pytest.mark.parametrize(
        "node_type", ["observation", "fact", "convention", "error"]
    )
    def test_valid_node_types(
        self, graph: Graph, dummy_embedding: np.ndarray, node_type: str
    ) -> None:
        node = _make_node(dummy_embedding, node_type=node_type)
        node_id = graph.write_node(node)
        assert node_id

    @pytest.mark.parametrize("scope", ["project", "module", "session"])
    def test_valid_scopes(
        self, graph: Graph, dummy_embedding: np.ndarray, scope: str
    ) -> None:
        node = _make_node(dummy_embedding, scope=scope)
        node_id = graph.write_node(node)
        assert node_id

    @pytest.mark.parametrize("source", ["jsonl", "ast", "git", "nlp"])
    def test_valid_sources(
        self, graph: Graph, dummy_embedding: np.ndarray, source: str
    ) -> None:
        node = _make_node(dummy_embedding, source=source)
        node_id = graph.write_node(node)
        assert node_id


# ---------------------------------------------------------------------------
# merge_node
# ---------------------------------------------------------------------------


class TestMergeNode:
    def test_merge_increments_weight(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        node = _make_node(dummy_embedding)
        node_id = graph.write_node(node)
        graph.merge_node(existing_id=node_id, candidate=node)
        rows = graph.get_all_nodes(project=TEST_PROJECT)
        assert rows[0].weight == pytest.approx(1.5, abs=1e-5)

    def test_merge_does_not_duplicate(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        node = _make_node(dummy_embedding)
        node_id = graph.write_node(node)
        graph.merge_node(existing_id=node_id, candidate=node)
        rows = graph.get_all_nodes(project=TEST_PROJECT)
        assert len(rows) == 1

    def test_merge_increments_session_count(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        node = _make_node(dummy_embedding)
        node_id = graph.write_node(node)
        graph.merge_node(existing_id=node_id, candidate=node)
        rows = graph.get_all_nodes(project=TEST_PROJECT)
        assert rows[0].session_count == 2

    def test_merge_updates_last_accessed(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        node = _make_node(dummy_embedding)
        node_id = graph.write_node(node)
        before = int(time.time())
        time.sleep(0.01)
        graph.merge_node(existing_id=node_id, candidate=node)
        rows = graph.get_all_nodes(project=TEST_PROJECT)
        assert rows[0].last_accessed >= before

    def test_merge_keeps_existing_text_when_candidate_is_longer(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        existing_text = "JWT auth"
        node = _make_node(dummy_embedding, text=existing_text)
        node_id = graph.write_node(node)
        longer_node = _make_node(
            dummy_embedding,
            text="auth/middleware.py handles JWT validation and token refresh logic",
        )
        graph.merge_node(existing_id=node_id, candidate=longer_node)
        rows = graph.get_all_nodes(project=TEST_PROJECT)
        # candidate is longer → existing text is kept
        assert rows[0].text == existing_text

    def test_merge_uses_new_text_when_shorter(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        node = _make_node(
            dummy_embedding, text="a very long original text that says many things"
        )
        node_id = graph.write_node(node)
        short_node = _make_node(dummy_embedding, text="short")
        graph.merge_node(existing_id=node_id, candidate=short_node)
        rows = graph.get_all_nodes(project=TEST_PROJECT)
        assert rows[0].text == "short"

    def test_merge_keeps_existing_rationale_when_both_present(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        node = _make_node(dummy_embedding, rationale="original rationale")
        node_id = graph.write_node(node)
        candidate = _make_node(dummy_embedding, rationale="new rationale")
        graph.merge_node(existing_id=node_id, candidate=candidate)
        rows = graph.get_all_nodes(project=TEST_PROJECT)
        assert rows[0].rationale == "original rationale"

    def test_merge_fills_null_rationale(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        node = _make_node(dummy_embedding, rationale=None)
        node_id = graph.write_node(node)
        candidate = _make_node(dummy_embedding, rationale="new rationale")
        graph.merge_node(existing_id=node_id, candidate=candidate)
        rows = graph.get_all_nodes(project=TEST_PROJECT)
        assert rows[0].rationale == "new rationale"


# ---------------------------------------------------------------------------
# find_similar
# ---------------------------------------------------------------------------


class TestFindSimilar:
    def test_returns_identical_vector_at_threshold_1(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        node = _make_node(dummy_embedding)
        graph.write_node(node)
        results = graph.find_similar(
            embedding=dummy_embedding,
            project=TEST_PROJECT,
            threshold=1.0,
            limit=5,
        )
        assert len(results) == 1

    def test_excludes_nodes_below_threshold(
        self, graph: Graph, dummy_embedding: np.ndarray, alt_embedding: np.ndarray
    ) -> None:
        node = _make_node(alt_embedding)
        graph.write_node(node)
        results = graph.find_similar(
            embedding=dummy_embedding,
            project=TEST_PROJECT,
            threshold=0.99,
            limit=5,
        )
        assert len(results) == 0

    def test_threshold_boundary_exact(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        node = _make_node(dummy_embedding)
        graph.write_node(node)
        results_above = graph.find_similar(
            embedding=dummy_embedding,
            project=TEST_PROJECT,
            threshold=0.99,
            limit=5,
        )
        assert len(results_above) == 1

        results_exact = graph.find_similar(
            embedding=dummy_embedding,
            project=TEST_PROJECT,
            threshold=1.0,
            limit=5,
        )
        assert len(results_exact) == 1

    def test_respects_project_filter(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        node_a = _make_node(dummy_embedding, project="/project/a")
        node_b = _make_node(dummy_embedding, project="/project/b")
        graph.write_node(node_a)
        graph.write_node(node_b)
        results = graph.find_similar(
            embedding=dummy_embedding,
            project="/project/a",
            threshold=0.9,
            limit=5,
        )
        assert len(results) == 1
        assert results[0].project == "/project/a"

    def test_respects_limit(self, graph: Graph, dummy_embedding: np.ndarray) -> None:
        for i in range(5):
            node = _make_node(dummy_embedding, text=f"node {i}")
            graph.write_node(node)
        results = graph.find_similar(
            embedding=dummy_embedding,
            project=TEST_PROJECT,
            threshold=0.9,
            limit=3,
        )
        assert len(results) <= 3

    def test_returns_empty_when_no_nodes(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        results = graph.find_similar(
            embedding=dummy_embedding,
            project=TEST_PROJECT,
            threshold=0.9,
            limit=5,
        )
        assert results == []


# ---------------------------------------------------------------------------
# get_all_nodes
# ---------------------------------------------------------------------------


class TestGetAllNodes:
    def test_returns_all_nodes_for_project(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        for i in range(3):
            graph.write_node(_make_node(dummy_embedding, text=f"node {i}"))
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        assert len(nodes) == 3

    def test_filters_by_project(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, project="/proj/a"))
        graph.write_node(_make_node(dummy_embedding, project="/proj/b"))
        nodes = graph.get_all_nodes(project="/proj/a")
        assert len(nodes) == 1
        assert nodes[0].project == "/proj/a"

    def test_filters_by_tier(self, graph: Graph, dummy_embedding: np.ndarray) -> None:
        graph.write_node(_make_node(dummy_embedding, tier=1))
        graph.write_node(_make_node(dummy_embedding, tier=2, text="tier2"))
        graph.write_node(_make_node(dummy_embedding, tier=3, text="tier3"))
        tier1 = graph.get_all_nodes(project=TEST_PROJECT, tier=1)
        assert len(tier1) == 1
        assert tier1[0].tier == 1

    def test_returns_empty_list_when_none(self, graph: Graph) -> None:
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        assert nodes == []


# ---------------------------------------------------------------------------
# delete_node
# ---------------------------------------------------------------------------


class TestDeleteNode:
    def test_deletes_node(self, graph: Graph, dummy_embedding: np.ndarray) -> None:
        node_id = graph.write_node(_make_node(dummy_embedding))
        graph.delete_node(node_id)
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        assert len(nodes) == 0

    def test_cascade_delete_removes_edges(
        self, graph: Graph, dummy_embedding: np.ndarray, alt_embedding: np.ndarray
    ) -> None:
        id_a = graph.write_node(_make_node(dummy_embedding, text="node a"))
        id_b = graph.write_node(_make_node(alt_embedding, text="node b"))
        graph.write_edge(id_a, id_b)
        graph.delete_node(id_a)
        edges = graph.get_edges(id_b)
        assert len(edges) == 0

    def test_delete_nonexistent_is_silent(self, graph: Graph) -> None:
        graph.delete_node("nonexistent-id")


# ---------------------------------------------------------------------------
# write_edge / get_edges
# ---------------------------------------------------------------------------


class TestEdges:
    def test_write_edge_creates_edge(
        self, graph: Graph, dummy_embedding: np.ndarray, alt_embedding: np.ndarray
    ) -> None:
        id_a = graph.write_node(_make_node(dummy_embedding, text="node a"))
        id_b = graph.write_node(_make_node(alt_embedding, text="node b"))
        graph.write_edge(id_a, id_b)
        edges = graph.get_edges(id_a)
        assert len(edges) == 1
        assert edges[0].target_id == id_b

    def test_write_edge_increments_strength_on_repeat(
        self, graph: Graph, dummy_embedding: np.ndarray, alt_embedding: np.ndarray
    ) -> None:
        id_a = graph.write_node(_make_node(dummy_embedding, text="node a"))
        id_b = graph.write_node(_make_node(alt_embedding, text="node b"))
        graph.write_edge(id_a, id_b)
        graph.write_edge(id_a, id_b)
        edges = graph.get_edges(id_a)
        assert edges[0].strength == pytest.approx(2.0, abs=1e-5)

    def test_edge_strength_increments_on_co_retrieval(
        self, graph: Graph, dummy_embedding: np.ndarray, alt_embedding: np.ndarray
    ) -> None:
        id_a = graph.write_node(_make_node(dummy_embedding, text="node a"))
        id_b = graph.write_node(_make_node(alt_embedding, text="node b"))
        graph.write_edge(id_a, id_b)
        initial_edges = graph.get_edges(id_a)
        initial_strength = initial_edges[0].strength
        graph.write_edge(id_a, id_b)
        updated_edges = graph.get_edges(id_a)
        assert updated_edges[0].strength > initial_strength

    def test_get_edges_returns_both_directions(
        self, graph: Graph, dummy_embedding: np.ndarray, alt_embedding: np.ndarray
    ) -> None:
        id_a = graph.write_node(_make_node(dummy_embedding, text="node a"))
        id_b = graph.write_node(_make_node(alt_embedding, text="node b"))
        graph.write_edge(id_a, id_b)
        graph.write_edge(id_b, id_a)
        edges_a = graph.get_edges(id_a)
        edges_b = graph.get_edges(id_b)
        assert len(edges_a) >= 1
        assert len(edges_b) >= 1

    def test_get_edges_returns_empty_when_none(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        node_id = graph.write_node(_make_node(dummy_embedding))
        edges = graph.get_edges(node_id)
        assert edges == []

    def test_self_loop_rejected(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        node_id = graph.write_node(_make_node(dummy_embedding))
        with pytest.raises(ValueError):
            graph.write_edge(node_id, node_id)


# ---------------------------------------------------------------------------
# update_weight
# ---------------------------------------------------------------------------


class TestUpdateWeight:
    def test_increments_weight(self, graph: Graph, dummy_embedding: np.ndarray) -> None:
        node_id = graph.write_node(_make_node(dummy_embedding))
        graph.update_weight(node_id, delta=2.5)
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        assert nodes[0].weight == pytest.approx(3.5, abs=1e-5)

    def test_decrements_weight(self, graph: Graph, dummy_embedding: np.ndarray) -> None:
        node_id = graph.write_node(_make_node(dummy_embedding))
        graph.update_weight(node_id, delta=-0.15)
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        assert nodes[0].weight == pytest.approx(0.85, abs=1e-5)

    def test_update_nonexistent_is_silent(self, graph: Graph) -> None:
        graph.update_weight("nonexistent-id", delta=1.0)

    def test_weight_does_not_go_negative(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        node_id = graph.write_node(_make_node(dummy_embedding))
        graph.update_weight(node_id, delta=-999.0)
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        assert nodes[0].weight >= 0.0


# ---------------------------------------------------------------------------
# Coverage gap tests
# ---------------------------------------------------------------------------


class TestCoverageEdgeCases:
    def test_merge_nonexistent_node_is_silent(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        """merge_node with a nonexistent id returns without error."""
        node = _make_node(dummy_embedding)
        graph.merge_node(existing_id="nonexistent-uuid", candidate=node)

    def test_find_similar_skips_null_embedding_rows(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        """Nodes written without embedding are skipped in find_similar."""
        node = _make_node(dummy_embedding)
        node_id = graph.write_node(node)
        # Manually null out the embedding to simulate the coverage path.
        graph._conn.execute(
            "UPDATE nodes SET embedding = NULL WHERE id = ?", (node_id,)
        )
        graph._conn.commit()
        results = graph.find_similar(
            embedding=dummy_embedding,
            project=TEST_PROJECT,
            threshold=0.5,
            limit=5,
        )
        assert results == []

    def test_cosine_similarity_zero_vector(self, graph: Graph) -> None:
        """find_similar with a zero-norm query returns cosine 0.0 (below any positive threshold)."""
        zero = np.zeros(384, dtype=np.float32)
        rng = np.random.default_rng(seed=42)
        node = _make_node(rng.random(384).astype(np.float32))
        graph.write_node(node)
        # Zero norm → cosine = 0.0 → below threshold 0.1 → no results
        results = graph.find_similar(
            embedding=zero,
            project=TEST_PROJECT,
            threshold=0.1,
            limit=5,
        )
        assert results == []


# ---------------------------------------------------------------------------
# get_edges_for_nodes
# ---------------------------------------------------------------------------


class TestGetEdgesForNodes:
    def test_returns_empty_dict_for_empty_input(self, graph: Graph) -> None:
        assert graph.get_edges_for_nodes([]) == {}

    def test_returns_edges_for_single_node(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        n1_id = graph.write_node(_make_node(dummy_embedding, text="n1"))
        n2_id = graph.write_node(_make_node(dummy_embedding, text="n2"))
        graph.write_edge(n1_id, n2_id)
        result = graph.get_edges_for_nodes([n1_id])
        assert n1_id in result
        assert len(result[n1_id]) == 1

    def test_batch_matches_individual_get_edges(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        ids = [
            graph.write_node(_make_node(dummy_embedding, text=f"n{i}"))
            for i in range(3)
        ]
        graph.write_edge(ids[0], ids[1])
        graph.write_edge(ids[1], ids[2])

        batch = graph.get_edges_for_nodes(ids)
        for nid in ids:
            individual = graph.get_edges(nid)
            assert len(batch[nid]) == len(individual)

    def test_node_with_no_edges_returns_empty_list(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        n_id = graph.write_node(_make_node(dummy_embedding))
        result = graph.get_edges_for_nodes([n_id])
        assert result[n_id] == []

    def test_unknown_node_id_omitted_from_result(self, graph: Graph) -> None:
        result = graph.get_edges_for_nodes(["nonexistent-uuid"])
        assert "nonexistent-uuid" in result
        assert result["nonexistent-uuid"] == []


# ---------------------------------------------------------------------------
# write_session and touch_nodes
# ---------------------------------------------------------------------------


class TestWriteSession:
    def test_write_session_inserts_row(self, graph: Graph) -> None:
        import uuid as _uuid

        sid = str(_uuid.uuid4())
        graph.write_session(
            session_id=sid,
            project=TEST_PROJECT,
            started_at=1000,
            ended_at=2000,
            nodes_written=3,
            nodes_evicted=1,
            nodes_promoted=0,
            transcript_path="/tmp/t.jsonl",
        )
        row = graph._conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (sid,)
        ).fetchone()
        assert row is not None
        assert row["nodes_written"] == 3
        assert row["nodes_promoted"] == 0

    def test_write_session_records_promoted(self, graph: Graph) -> None:
        import uuid as _uuid

        sid = str(_uuid.uuid4())
        graph.write_session(
            session_id=sid,
            project=TEST_PROJECT,
            started_at=1000,
            ended_at=2000,
            nodes_written=0,
            nodes_evicted=0,
            nodes_promoted=2,
            transcript_path="/tmp/t.jsonl",
        )
        row = graph._conn.execute(
            "SELECT nodes_promoted FROM sessions WHERE id = ?", (sid,)
        ).fetchone()
        assert row["nodes_promoted"] == 2

    def test_write_session_stores_token_counts(self, graph: Graph) -> None:
        import uuid as _uuid

        sid = str(_uuid.uuid4())
        graph.write_session(
            session_id=sid,
            project=TEST_PROJECT,
            started_at=1000,
            ended_at=2000,
            nodes_written=5,
            nodes_evicted=0,
            nodes_promoted=0,
            transcript_path="/tmp/t.jsonl",
            tokens_raw=1200,
            tokens_injected=420,
        )
        row = graph._conn.execute(
            "SELECT tokens_raw, tokens_injected FROM sessions WHERE id = ?",
            (sid,),
        ).fetchone()
        assert row["tokens_raw"] == 1200
        assert row["tokens_injected"] == 420

    def test_write_session_token_counts_nullable(self, graph: Graph) -> None:
        import uuid as _uuid

        sid = str(_uuid.uuid4())
        graph.write_session(
            session_id=sid,
            project=TEST_PROJECT,
            started_at=1000,
            ended_at=2000,
            nodes_written=0,
            nodes_evicted=0,
            nodes_promoted=0,
            transcript_path="/tmp/t.jsonl",
        )
        row = graph._conn.execute(
            "SELECT tokens_raw, tokens_injected FROM sessions WHERE id = ?",
            (sid,),
        ).fetchone()
        assert row["tokens_raw"] is None
        assert row["tokens_injected"] is None


class TestTouchNodes:
    def test_touch_updates_last_accessed(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        n_id = graph.write_node(_make_node(dummy_embedding))
        import time as _t

        _t.sleep(1)
        now = int(_t.time())
        graph.touch_nodes([n_id], now=now)
        node = graph.get_node(n_id)
        assert node is not None
        assert node.last_accessed == now

    def test_touch_bumps_weight(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        n_id = graph.write_node(_make_node(dummy_embedding))
        node_before = graph.get_node(n_id)
        assert node_before is not None
        initial_weight = node_before.weight
        graph.touch_nodes([n_id], now=int(time.time()))
        node_after = graph.get_node(n_id)
        assert node_after is not None
        assert node_after.weight > initial_weight

    def test_touch_empty_list_is_noop(self, graph: Graph) -> None:
        graph.touch_nodes([], now=int(time.time()))  # must not raise

    def test_touch_multiple_nodes_in_one_call(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        ids = [
            graph.write_node(_make_node(dummy_embedding, text=f"n{i}"))
            for i in range(3)
        ]
        now = int(time.time()) + 100
        graph.touch_nodes(ids, now=now)
        for nid in ids:
            node = graph.get_node(nid)
            assert node is not None
            assert node.last_accessed == now

    def test_touch_increments_session_count(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        n_id = graph.write_node(_make_node(dummy_embedding, text="session count test"))
        node_before = graph.get_node(n_id)
        assert node_before is not None
        initial_count = node_before.session_count
        graph.touch_nodes([n_id], now=int(time.time()))
        node_after = graph.get_node(n_id)
        assert node_after is not None
        assert node_after.session_count == initial_count + 1

    def test_touch_session_count_cumulative(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        """Multiple touch calls (simulating multiple sessions) accumulate correctly."""
        n_id = graph.write_node(_make_node(dummy_embedding, text="cumulative count"))
        for i in range(4):
            graph.touch_nodes([n_id], now=int(time.time()) + i)
        node = graph.get_node(n_id)
        assert node is not None
        assert node.session_count == 1 + 4  # initial 1 + 4 touches


# ---------------------------------------------------------------------------
# get_nodes_by_source
# ---------------------------------------------------------------------------


class TestGetNodesBySource:
    def test_returns_only_matching_source(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, text="nlp node", source="nlp"))
        graph.write_node(_make_node(dummy_embedding, text="git node", source="git"))
        graph.write_node(_make_node(dummy_embedding, text="jsonl node", source="jsonl"))

        nlp_nodes = graph.get_nodes_by_source(TEST_PROJECT, source="nlp")
        assert len(nlp_nodes) == 1
        assert nlp_nodes[0].text == "nlp node"

    def test_returns_empty_when_no_match(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, text="git node", source="git"))
        result = graph.get_nodes_by_source(TEST_PROJECT, source="ast")
        assert result == []

    def test_tier_filter_applied_with_source(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(
            _make_node(dummy_embedding, text="nlp tier1", source="nlp", tier=1)
        )
        graph.write_node(
            _make_node(dummy_embedding, text="nlp tier3", source="nlp", tier=3)
        )
        tier3_only = graph.get_nodes_by_source(TEST_PROJECT, source="nlp", tier=3)
        assert len(tier3_only) == 1
        assert tier3_only[0].text == "nlp tier3"

    def test_ordered_by_weight_descending(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, text="low weight", source="nlp", weight=0.5))
        graph.write_node(_make_node(dummy_embedding, text="high weight", source="nlp", weight=9.0))
        graph.write_node(_make_node(dummy_embedding, text="mid weight", source="nlp", weight=3.0))

        nodes = graph.get_nodes_by_source(TEST_PROJECT, source="nlp")
        weights = [n.weight for n in nodes]
        assert weights == sorted(weights, reverse=True)

    def test_different_project_excluded(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, text="other project", source="nlp", project="/other"))
        result = graph.get_nodes_by_source(TEST_PROJECT, source="nlp")
        assert result == []


# ---------------------------------------------------------------------------
# update_node_rationale
# ---------------------------------------------------------------------------


class TestUpdateNodeRationale:
    def test_sets_rationale_on_existing_node(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        nid = graph.write_node(_make_node(dummy_embedding, rationale=None))
        result = graph.update_node_rationale(nid, "because it avoids locks")
        assert result is True
        node = graph.get_node(nid)
        assert node is not None
        assert node.rationale == "because it avoids locks"

    def test_clears_rationale_when_none(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        nid = graph.write_node(_make_node(dummy_embedding, rationale="old reason"))
        graph.update_node_rationale(nid, None)
        node = graph.get_node(nid)
        assert node is not None
        assert node.rationale is None

    def test_returns_false_for_unknown_id(self, graph: Graph) -> None:
        result = graph.update_node_rationale("non-existent-uuid", "some reason")
        assert result is False

    def test_overwrites_existing_rationale(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        nid = graph.write_node(_make_node(dummy_embedding, rationale="old"))
        graph.update_node_rationale(nid, "new reason")
        node = graph.get_node(nid)
        assert node is not None
        assert node.rationale == "new reason"


# ---------------------------------------------------------------------------
# set_node_weight
# ---------------------------------------------------------------------------


class TestSetNodeWeight:
    def test_sets_weight_to_new_value(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        nid = graph.write_node(_make_node(dummy_embedding, weight=1.0))
        graph.set_node_weight(nid, 15.0)
        node = graph.get_node(nid)
        assert node is not None
        assert abs(node.weight - 15.0) < 1e-6

    def test_floors_at_zero_for_negative(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        nid = graph.write_node(_make_node(dummy_embedding, weight=5.0))
        graph.set_node_weight(nid, -3.0)
        node = graph.get_node(nid)
        assert node is not None
        assert node.weight == 0.0

    def test_returns_false_for_unknown_id(self, graph: Graph) -> None:
        result = graph.set_node_weight("non-existent", 5.0)
        assert result is False

    def test_returns_true_on_success(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        nid = graph.write_node(_make_node(dummy_embedding))
        result = graph.set_node_weight(nid, 7.5)
        assert result is True
