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
# MERGE_WEIGHT_BUMP constant and precise merge arithmetic
# ---------------------------------------------------------------------------


class TestMergeWeightBump:
    def test_constant_value_is_point_five(self) -> None:
        from core.graph import MERGE_WEIGHT_BUMP

        assert pytest.approx(0.5) == MERGE_WEIGHT_BUMP

    def test_merge_increases_weight_by_constant(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        from core.graph import MERGE_WEIGHT_BUMP

        node = _make_node(dummy_embedding, weight=2.0)
        nid = graph.write_node(node)
        graph.merge_node(existing_id=nid, candidate=node)
        updated = graph.get_node(nid)
        assert updated is not None
        assert updated.weight == pytest.approx(2.0 + MERGE_WEIGHT_BUMP, abs=1e-5)

    def test_double_merge_increases_weight_by_two_bumps(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        from core.graph import MERGE_WEIGHT_BUMP

        node = _make_node(dummy_embedding, weight=1.0)
        nid = graph.write_node(node)
        graph.merge_node(existing_id=nid, candidate=node)
        graph.merge_node(existing_id=nid, candidate=node)
        updated = graph.get_node(nid)
        assert updated is not None
        assert updated.weight == pytest.approx(1.0 + 2 * MERGE_WEIGHT_BUMP, abs=1e-5)

    def test_touch_weight_bump_constant_value(self) -> None:
        from core.graph import TOUCH_WEIGHT_BUMP

        assert pytest.approx(0.1) == TOUCH_WEIGHT_BUMP


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
# delete_all_nodes
# ---------------------------------------------------------------------------


class TestDeleteAllNodes:
    def test_returns_count_of_deleted(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        for i in range(3):
            graph.write_node(_make_node(dummy_embedding, text=f"n{i}"))
        count = graph.delete_all_nodes(TEST_PROJECT)
        assert count == 3

    def test_clears_all_nodes_for_project(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, text="a"))
        graph.write_node(_make_node(dummy_embedding, text="b"))
        graph.delete_all_nodes(TEST_PROJECT)
        assert graph.get_all_nodes(TEST_PROJECT) == []

    def test_empty_project_returns_zero(self, graph: Graph) -> None:
        count = graph.delete_all_nodes(TEST_PROJECT)
        assert count == 0

    def test_does_not_delete_other_project_nodes(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        import uuid

        other_project = "/other/proj"
        graph.write_node(_make_node(dummy_embedding, text="mine"))
        other_node = _make_node(dummy_embedding, text="other")
        other_node = Node(
            id=str(uuid.uuid4()),
            type=other_node.type,
            tier=other_node.tier,
            text=other_node.text,
            rationale=other_node.rationale,
            embedding=other_node.embedding,
            precision_bits=other_node.precision_bits,
            weight=other_node.weight,
            project=other_project,
            scope=other_node.scope,
            source=other_node.source,
            last_accessed=other_node.last_accessed,
            created_at=other_node.created_at,
            session_count=other_node.session_count,
        )
        graph.write_node(other_node)
        graph.delete_all_nodes(TEST_PROJECT)
        remaining = graph.get_all_nodes(other_project)
        assert len(remaining) == 1


# ---------------------------------------------------------------------------
# update_node_tier
# ---------------------------------------------------------------------------


class TestUpdateNodeTier:
    def test_tier_is_updated(self, graph: Graph, dummy_embedding: np.ndarray) -> None:
        import time

        node_id = graph.write_node(_make_node(dummy_embedding, tier=1))
        graph.update_node_tier(
            node_id,
            new_tier=2,
            new_precision=8,
            embedding_blob=None,
            now=int(time.time()),
        )
        node = graph.get_node(node_id)
        assert node is not None
        assert node.tier == 2

    def test_precision_bits_updated(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        import time

        node_id = graph.write_node(_make_node(dummy_embedding, tier=1))
        graph.update_node_tier(
            node_id,
            new_tier=2,
            new_precision=8,
            embedding_blob=None,
            now=int(time.time()),
        )
        node = graph.get_node(node_id)
        assert node is not None
        assert node.precision_bits == 8

    def test_last_accessed_updated(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        import time

        now = int(time.time()) + 9999
        node_id = graph.write_node(_make_node(dummy_embedding, tier=1))
        graph.update_node_tier(
            node_id,
            new_tier=2,
            new_precision=8,
            embedding_blob=None,
            now=now,
        )
        node = graph.get_node(node_id)
        assert node is not None
        assert node.last_accessed == now

    def test_nonexistent_node_is_silent(self, graph: Graph) -> None:
        import time

        graph.update_node_tier(
            "nonexistent-uuid",
            new_tier=2,
            new_precision=8,
            embedding_blob=None,
            now=int(time.time()),
        )


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

    def test_zero_delta_leaves_weight_unchanged(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        node_id = graph.write_node(_make_node(dummy_embedding, weight=2.0))
        graph.update_weight(node_id, delta=0.0)
        node = graph.get_node(node_id)
        assert node is not None
        assert pytest.approx(2.0) == node.weight

    def test_two_increments_accumulate(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        node_id = graph.write_node(_make_node(dummy_embedding, weight=1.0))
        graph.update_weight(node_id, delta=0.5)
        graph.update_weight(node_id, delta=0.5)
        node = graph.get_node(node_id)
        assert node is not None
        assert pytest.approx(2.0) == node.weight


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

    def test_weight_bump_equals_touch_weight_bump_constant(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        from core.graph import TOUCH_WEIGHT_BUMP

        n_id = graph.write_node(_make_node(dummy_embedding, weight=1.0))
        graph.touch_nodes([n_id], now=int(time.time()))
        node = graph.get_node(n_id)
        assert node is not None
        assert pytest.approx(1.0 + TOUCH_WEIGHT_BUMP) == node.weight

    def test_untouched_node_weight_unchanged(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        id_touched = graph.write_node(
            _make_node(dummy_embedding, text="touched", weight=1.0)
        )
        id_other = graph.write_node(
            _make_node(dummy_embedding, text="other", weight=2.0)
        )
        graph.touch_nodes([id_touched], now=int(time.time()))
        other = graph.get_node(id_other)
        assert other is not None
        assert pytest.approx(2.0) == other.weight


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
        graph.write_node(
            _make_node(dummy_embedding, text="low weight", source="nlp", weight=0.5)
        )
        graph.write_node(
            _make_node(dummy_embedding, text="high weight", source="nlp", weight=9.0)
        )
        graph.write_node(
            _make_node(dummy_embedding, text="mid weight", source="nlp", weight=3.0)
        )

        nodes = graph.get_nodes_by_source(TEST_PROJECT, source="nlp")
        weights = [n.weight for n in nodes]
        assert weights == sorted(weights, reverse=True)

    def test_different_project_excluded(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(
            _make_node(
                dummy_embedding, text="other project", source="nlp", project="/other"
            )
        )
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


# ---------------------------------------------------------------------------
# get_stale_nodes
# ---------------------------------------------------------------------------


class TestGetStaleNodes:
    def test_returns_nodes_older_than_threshold(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        old_time = int(time.time()) - 10 * 86_400  # 10 days ago
        node = _make_node(dummy_embedding, text="old node")
        nid = graph.write_node(node)
        graph._conn.execute(
            "UPDATE nodes SET last_accessed = ? WHERE id = ?", (old_time, nid)
        )
        graph._conn.commit()

        stale = graph.get_stale_nodes(TEST_PROJECT, days=7)
        assert any(n.id == nid for n in stale)

    def test_excludes_recently_accessed_nodes(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        recent_time = int(time.time()) - 2 * 86_400  # 2 days ago
        node = _make_node(dummy_embedding, text="recent node")
        nid = graph.write_node(node)
        graph._conn.execute(
            "UPDATE nodes SET last_accessed = ? WHERE id = ?", (recent_time, nid)
        )
        graph._conn.commit()

        stale = graph.get_stale_nodes(TEST_PROJECT, days=7)
        assert not any(n.id == nid for n in stale)

    def test_excludes_tier_3_nodes(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        old_time = int(time.time()) - 30 * 86_400
        node = _make_node(dummy_embedding, text="permanent node", tier=3)
        nid = graph.write_node(node)
        graph._conn.execute(
            "UPDATE nodes SET last_accessed = ? WHERE id = ?", (old_time, nid)
        )
        graph._conn.commit()

        stale = graph.get_stale_nodes(TEST_PROJECT, days=7)
        assert not any(n.id == nid for n in stale)

    def test_tier_filter_respected(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        old_time = int(time.time()) - 10 * 86_400
        n1 = _make_node(dummy_embedding, text="tier1 old", tier=1)
        n2 = _make_node(dummy_embedding, text="tier2 old", tier=2)
        id1 = graph.write_node(n1)
        id2 = graph.write_node(n2)
        for nid in (id1, id2):
            graph._conn.execute(
                "UPDATE nodes SET last_accessed = ? WHERE id = ?", (old_time, nid)
            )
        graph._conn.commit()

        tier1_stale = graph.get_stale_nodes(TEST_PROJECT, days=7, tier=1)
        assert any(n.id == id1 for n in tier1_stale)
        assert not any(n.id == id2 for n in tier1_stale)

    def test_ordered_by_last_accessed_ascending(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        times = [
            int(time.time()) - 20 * 86_400,
            int(time.time()) - 15 * 86_400,
            int(time.time()) - 10 * 86_400,
        ]
        for i, t in enumerate(times):
            node = _make_node(dummy_embedding, text=f"old node {i}")
            nid = graph.write_node(node)
            graph._conn.execute(
                "UPDATE nodes SET last_accessed = ? WHERE id = ?", (t, nid)
            )
        graph._conn.commit()

        stale = graph.get_stale_nodes(TEST_PROJECT, days=7)
        accessed = [n.last_accessed for n in stale]
        assert accessed == sorted(accessed)

    def test_empty_when_no_stale_nodes(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        node = _make_node(dummy_embedding)
        graph.write_node(node)
        stale = graph.get_stale_nodes(TEST_PROJECT, days=7)
        assert stale == []

    def test_project_isolation(self, graph: Graph, dummy_embedding: np.ndarray) -> None:
        old_time = int(time.time()) - 10 * 86_400
        id_this = graph.write_node(
            _make_node(dummy_embedding, text="this project", tier=1)
        )
        id_other = graph.write_node(
            _make_node(dummy_embedding, text="other project", tier=1, project="/other")
        )
        for nid in (id_this, id_other):
            graph._conn.execute(
                "UPDATE nodes SET last_accessed = ? WHERE id = ?", (old_time, nid)
            )
        graph._conn.commit()
        stale = graph.get_stale_nodes(TEST_PROJECT, days=7)
        assert any(n.id == id_this for n in stale)
        assert all(n.project == TEST_PROJECT for n in stale)

    def test_days_boundary_one_second_before_cutoff(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        days = 7
        cutoff = int(time.time()) - days * 86_400
        barely_stale = cutoff - 1
        nid = graph.write_node(_make_node(dummy_embedding, text="boundary node"))
        graph._conn.execute(
            "UPDATE nodes SET last_accessed = ? WHERE id = ?", (barely_stale, nid)
        )
        graph._conn.commit()
        stale = graph.get_stale_nodes(TEST_PROJECT, days=days)
        assert any(n.id == nid for n in stale)


# ---------------------------------------------------------------------------
# delete_nodes_bulk
# ---------------------------------------------------------------------------


class TestDeleteNodesBulk:
    def test_deletes_all_specified_nodes(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        ids = [
            graph.write_node(_make_node(dummy_embedding, text=f"node {i}"))
            for i in range(3)
        ]
        deleted = graph.delete_nodes_bulk(ids)
        assert deleted == 3
        for nid in ids:
            assert graph.get_node(nid) is None

    def test_empty_list_returns_zero(self, graph: Graph) -> None:
        assert graph.delete_nodes_bulk([]) == 0

    def test_partial_delete_returns_actual_count(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        nid = graph.write_node(_make_node(dummy_embedding))
        deleted = graph.delete_nodes_bulk([nid, "nonexistent-uuid"])
        assert deleted == 1


# ---------------------------------------------------------------------------
# session_exists
# ---------------------------------------------------------------------------


class TestSessionExists:
    def test_returns_false_when_no_sessions(self, graph: Graph) -> None:
        assert graph.session_exists("/tmp/nonexistent.jsonl") is False

    def test_returns_true_after_write_session(self, graph: Graph) -> None:
        path = "/tmp/test_session.jsonl"
        now = int(time.time())
        graph.write_session(
            session_id="test-session-id",
            project=TEST_PROJECT,
            started_at=now - 60,
            ended_at=now,
            nodes_written=1,
            nodes_evicted=0,
            nodes_promoted=0,
            transcript_path=path,
        )
        assert graph.session_exists(path) is True

    def test_different_path_returns_false(self, graph: Graph) -> None:
        path = "/tmp/session_a.jsonl"
        now = int(time.time())
        graph.write_session(
            session_id="test-session-id",
            project=TEST_PROJECT,
            started_at=now - 60,
            ended_at=now,
            nodes_written=0,
            nodes_evicted=0,
            nodes_promoted=0,
            transcript_path=path,
        )
        assert graph.session_exists("/tmp/session_b.jsonl") is False

    def test_empty_string_path_returns_false(self, graph: Graph) -> None:
        assert graph.session_exists("") is False

    def test_returns_false_for_similar_but_different_path(self, graph: Graph) -> None:
        path = "/tmp/my_session.jsonl"
        now = int(time.time())
        graph.write_session(
            session_id="sim-id",
            project=TEST_PROJECT,
            started_at=now - 60,
            ended_at=now,
            nodes_written=0,
            nodes_evicted=0,
            nodes_promoted=0,
            transcript_path=path,
        )
        assert graph.session_exists("/tmp/my_session.jsonl.bak") is False


# ---------------------------------------------------------------------------
# get_tier_counts
# ---------------------------------------------------------------------------


class TestGetTierCounts:
    def test_empty_graph_returns_empty_dict(self, graph: Graph) -> None:
        assert graph.get_tier_counts(TEST_PROJECT) == {}

    def test_counts_match_written_nodes(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, text="t1a", tier=1))
        graph.write_node(_make_node(dummy_embedding, text="t1b", tier=1))
        graph.write_node(_make_node(dummy_embedding, text="t2a", tier=2))
        counts = graph.get_tier_counts(TEST_PROJECT)
        assert counts[1] == (2, pytest.approx(1.0))
        assert counts[2] == (1, pytest.approx(1.0))
        assert 3 not in counts

    def test_scoped_to_project(self, graph: Graph, dummy_embedding: np.ndarray) -> None:
        graph.write_node(
            _make_node(dummy_embedding, text="proj1", project=TEST_PROJECT)
        )
        graph.write_node(
            _make_node(dummy_embedding, text="proj2", project="/other/project")
        )
        counts = graph.get_tier_counts(TEST_PROJECT)
        total = sum(c for c, _ in counts.values())
        assert total == 1


# ---------------------------------------------------------------------------
# get_type_counts / get_source_counts
# ---------------------------------------------------------------------------


class TestGetTypeCounts:
    def test_returns_types_ordered_by_count(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        for i in range(3):
            graph.write_node(
                _make_node(dummy_embedding, text=f"obs{i}", node_type="observation")
            )
        graph.write_node(_make_node(dummy_embedding, text="fact1", node_type="fact"))
        pairs = graph.get_type_counts(TEST_PROJECT)
        assert pairs[0] == ("observation", 3)
        assert pairs[1] == ("fact", 1)

    def test_empty_returns_empty_list(self, graph: Graph) -> None:
        assert graph.get_type_counts(TEST_PROJECT) == []

    def test_project_isolation(self, graph: Graph, dummy_embedding: np.ndarray) -> None:
        graph.write_node(
            _make_node(dummy_embedding, text="obs", node_type="observation")
        )
        graph.write_node(
            _make_node(
                dummy_embedding,
                text="other obs",
                node_type="observation",
                project="/other",
            )
        )
        pairs = graph.get_type_counts(TEST_PROJECT)
        assert len(pairs) == 1
        assert pairs[0] == ("observation", 1)

    def test_single_type_returns_one_entry(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(
            _make_node(dummy_embedding, text="only fact", node_type="fact")
        )
        pairs = graph.get_type_counts(TEST_PROJECT)
        assert len(pairs) == 1
        assert pairs[0][0] == "fact"
        assert pairs[0][1] == 1


class TestGetSourceCounts:
    def test_groups_by_source(self, graph: Graph, dummy_embedding: np.ndarray) -> None:
        graph.write_node(_make_node(dummy_embedding, text="j1", source="jsonl"))
        graph.write_node(_make_node(dummy_embedding, text="j2", source="jsonl"))
        graph.write_node(_make_node(dummy_embedding, text="n1", source="nlp"))
        pairs = graph.get_source_counts(TEST_PROJECT)
        assert pairs[0] == ("jsonl", 2)
        assert pairs[1] == ("nlp", 1)

    def test_empty_returns_empty_list(self, graph: Graph) -> None:
        assert graph.get_source_counts(TEST_PROJECT) == []

    def test_project_isolation(self, graph: Graph, dummy_embedding: np.ndarray) -> None:
        graph.write_node(_make_node(dummy_embedding, text="git1", source="git"))
        graph.write_node(
            _make_node(
                dummy_embedding, text="other git", source="git", project="/other"
            )
        )
        pairs = graph.get_source_counts(TEST_PROJECT)
        assert len(pairs) == 1
        assert pairs[0] == ("git", 1)


# ---------------------------------------------------------------------------
# get_last_session
# ---------------------------------------------------------------------------


class TestGetLastSession:
    def test_returns_none_when_no_sessions(self, graph: Graph) -> None:
        assert graph.get_last_session(TEST_PROJECT) is None

    def test_returns_most_recent_session(self, graph: Graph) -> None:
        now = int(time.time())
        for i, ended in enumerate([now - 200, now - 100, now - 50]):
            graph.write_session(
                session_id=f"sess-{i}",
                project=TEST_PROJECT,
                started_at=ended - 60,
                ended_at=ended,
                nodes_written=i,
                nodes_evicted=0,
                nodes_promoted=0,
                transcript_path=f"/tmp/t{i}.jsonl",
            )
        last = graph.get_last_session(TEST_PROJECT)
        assert last is not None
        assert last["ended_at"] == now - 50

    def test_scoped_to_project(self, graph: Graph) -> None:
        now = int(time.time())
        graph.write_session(
            session_id="other-sess",
            project="/other/project",
            started_at=now - 60,
            ended_at=now,
            nodes_written=5,
            nodes_evicted=0,
            nodes_promoted=0,
            transcript_path="/tmp/other.jsonl",
        )
        assert graph.get_last_session(TEST_PROJECT) is None

    def test_includes_token_fields(self, graph: Graph) -> None:
        now = int(time.time())
        graph.write_session(
            session_id="tok-sess",
            project=TEST_PROJECT,
            started_at=now - 60,
            ended_at=now,
            nodes_written=3,
            nodes_evicted=1,
            nodes_promoted=0,
            transcript_path="/tmp/tok.jsonl",
            tokens_raw=800,
            tokens_injected=300,
        )
        last = graph.get_last_session(TEST_PROJECT)
        assert last is not None
        assert last["tokens_raw"] == 800
        assert last["tokens_injected"] == 300

    def test_includes_nodes_written_and_evicted(self, graph: Graph) -> None:
        now = int(time.time())
        graph.write_session(
            session_id="nw-sess",
            project=TEST_PROJECT,
            started_at=now - 60,
            ended_at=now,
            nodes_written=7,
            nodes_evicted=2,
            nodes_promoted=1,
            transcript_path="",
        )
        last = graph.get_last_session(TEST_PROJECT)
        assert last is not None
        assert last["nodes_written"] == 7
        assert last["nodes_evicted"] == 2
        assert last["nodes_promoted"] == 1


# ---------------------------------------------------------------------------
# get_recent_nodes
# ---------------------------------------------------------------------------


class TestGetRecentNodes:
    def test_ordered_by_last_accessed_desc(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        now = int(time.time())
        for i in range(3):
            node = _make_node(dummy_embedding, text=f"node {i}")
            node_id = graph.write_node(node)
            # Manually set different last_accessed values
            graph._conn.execute(
                "UPDATE nodes SET last_accessed = ? WHERE id = ?",
                (now - (3 - i) * 100, node_id),
            )
            graph._conn.commit()
        recent = graph.get_recent_nodes(TEST_PROJECT, limit=3)
        assert len(recent) == 3
        assert recent[0].text == "node 2"
        assert recent[1].text == "node 1"

    def test_limit_respected(self, graph: Graph, dummy_embedding: np.ndarray) -> None:
        for i in range(5):
            graph.write_node(_make_node(dummy_embedding, text=f"node {i}"))
        assert len(graph.get_recent_nodes(TEST_PROJECT, limit=2)) == 2

    def test_empty_graph_returns_empty(self, graph: Graph) -> None:
        assert graph.get_recent_nodes(TEST_PROJECT) == []

    def test_default_limit_is_ten(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        for i in range(15):
            graph.write_node(_make_node(dummy_embedding, text=f"node {i}"))
        recent = graph.get_recent_nodes(TEST_PROJECT)
        assert len(recent) == 10

    def test_scoped_to_project(self, graph: Graph, dummy_embedding: np.ndarray) -> None:
        graph.write_node(_make_node(dummy_embedding, text="mine", project=TEST_PROJECT))
        graph.write_node(
            _make_node(dummy_embedding, text="other", project="/other/proj")
        )
        recent = graph.get_recent_nodes(TEST_PROJECT)
        assert all(n.project == TEST_PROJECT for n in recent)


# ---------------------------------------------------------------------------
# get_aggregate_stats
# ---------------------------------------------------------------------------


class TestGetAggregateStats:
    def test_all_zeros_for_empty_graph(self, graph: Graph) -> None:
        agg = graph.get_aggregate_stats(TEST_PROJECT)
        assert agg["session_count"] == 0
        assert agg["total_written"] == 0
        assert agg["tier_counts"] == {}

    def test_accumulates_across_sessions(self, graph: Graph) -> None:
        now = int(time.time())
        for i in range(3):
            graph.write_session(
                session_id=f"s{i}",
                project=TEST_PROJECT,
                started_at=now - 60,
                ended_at=now - i,
                nodes_written=2,
                nodes_evicted=1,
                nodes_promoted=0,
                transcript_path=f"/tmp/s{i}.jsonl",
            )
        agg = graph.get_aggregate_stats(TEST_PROJECT)
        assert agg["session_count"] == 3
        assert agg["total_written"] == 6
        assert agg["total_evicted"] == 3

    def test_tier_counts_reflect_graph(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, text="t1", tier=1))
        graph.write_node(_make_node(dummy_embedding, text="t3", tier=3))
        agg = graph.get_aggregate_stats(TEST_PROJECT)
        tier_counts: dict[int, int] = agg["tier_counts"]
        assert tier_counts[1] == 1
        assert tier_counts[3] == 1

    def test_total_saved_sums_token_differences(self, graph: Graph) -> None:
        now = int(time.time())
        graph.write_session(
            session_id="s1",
            project=TEST_PROJECT,
            started_at=now - 60,
            ended_at=now,
            nodes_written=1,
            nodes_evicted=0,
            nodes_promoted=0,
            transcript_path="/tmp/s1.jsonl",
            tokens_raw=1000,
            tokens_injected=200,
        )
        graph.write_session(
            session_id="s2",
            project=TEST_PROJECT,
            started_at=now - 120,
            ended_at=now - 60,
            nodes_written=1,
            nodes_evicted=0,
            nodes_promoted=0,
            transcript_path="/tmp/s2.jsonl",
            tokens_raw=500,
            tokens_injected=100,
        )
        agg = graph.get_aggregate_stats(TEST_PROJECT)
        assert agg["total_saved"] == (1000 - 200) + (500 - 100)

    def test_first_and_last_session_timestamps(self, graph: Graph) -> None:
        now = int(time.time())
        graph.write_session(
            session_id="old",
            project=TEST_PROJECT,
            started_at=now - 200,
            ended_at=now - 100,
            nodes_written=0,
            nodes_evicted=0,
            nodes_promoted=0,
            transcript_path="",
        )
        graph.write_session(
            session_id="new",
            project=TEST_PROJECT,
            started_at=now - 60,
            ended_at=now,
            nodes_written=0,
            nodes_evicted=0,
            nodes_promoted=0,
            transcript_path="",
        )
        agg = graph.get_aggregate_stats(TEST_PROJECT)
        assert agg["first_session"] == now - 200
        assert agg["last_session"] == now

    def test_project_isolation(self, graph: Graph) -> None:
        now = int(time.time())
        graph.write_session(
            session_id="other",
            project="/other/project",
            started_at=now - 60,
            ended_at=now,
            nodes_written=99,
            nodes_evicted=0,
            nodes_promoted=0,
            transcript_path="",
        )
        agg = graph.get_aggregate_stats(TEST_PROJECT)
        assert agg["session_count"] == 0
        assert agg["total_written"] == 0


# ---------------------------------------------------------------------------
# query_nodes_page
# ---------------------------------------------------------------------------


class TestQueryNodesPage:
    def test_returns_all_when_no_filters(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        for i in range(4):
            graph.write_node(_make_node(dummy_embedding, text=f"node {i}"))
        nodes, total = graph.query_nodes_page(TEST_PROJECT)
        assert total == 4
        assert len(nodes) == 4

    def test_tier_filter(self, graph: Graph, dummy_embedding: np.ndarray) -> None:
        graph.write_node(_make_node(dummy_embedding, text="t1", tier=1))
        graph.write_node(_make_node(dummy_embedding, text="t2", tier=2))
        nodes, total = graph.query_nodes_page(TEST_PROJECT, tier=1)
        assert total == 1
        assert nodes[0].tier == 1

    def test_type_filter(self, graph: Graph, dummy_embedding: np.ndarray) -> None:
        graph.write_node(
            _make_node(dummy_embedding, text="obs", node_type="observation")
        )
        graph.write_node(_make_node(dummy_embedding, text="fact", node_type="fact"))
        nodes, total = graph.query_nodes_page(TEST_PROJECT, node_type="observation")
        assert total == 1
        assert nodes[0].type == "observation"

    def test_limit_respected(self, graph: Graph, dummy_embedding: np.ndarray) -> None:
        for i in range(10):
            graph.write_node(_make_node(dummy_embedding, text=f"n{i}", weight=float(i)))
        nodes, total = graph.query_nodes_page(TEST_PROJECT, limit=3)
        assert total == 10
        assert len(nodes) == 3

    def test_invalid_sort_col_raises(self, graph: Graph) -> None:
        with pytest.raises(ValueError, match="sort_col"):
            graph.query_nodes_page(TEST_PROJECT, sort_col="injected_column")

    def test_combined_type_and_source_filter(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(
            _make_node(
                dummy_embedding, text="a", node_type="observation", source="jsonl"
            )
        )
        graph.write_node(
            _make_node(dummy_embedding, text="b", node_type="fact", source="nlp")
        )
        nodes, total = graph.query_nodes_page(
            TEST_PROJECT, node_type="observation", source="jsonl"
        )
        assert total == 1
        assert nodes[0].text == "a"


# ---------------------------------------------------------------------------
# find_similar — validation guards (lines 266, 268)
# ---------------------------------------------------------------------------


class TestFindSimilarValidation:
    def test_threshold_below_zero_raises(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match="threshold"):
            graph.find_similar(dummy_embedding, TEST_PROJECT, threshold=-0.1, limit=10)

    def test_threshold_above_one_raises(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match="threshold"):
            graph.find_similar(dummy_embedding, TEST_PROJECT, threshold=1.1, limit=10)

    def test_limit_zero_raises(self, graph: Graph, dummy_embedding: np.ndarray) -> None:
        with pytest.raises(ValueError, match="limit"):
            graph.find_similar(dummy_embedding, TEST_PROJECT, threshold=0.9, limit=0)

    def test_limit_negative_raises(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match="limit"):
            graph.find_similar(dummy_embedding, TEST_PROJECT, threshold=0.9, limit=-1)

    def test_node_with_null_embedding_skipped(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        # Write a node without embedding; find_similar should skip it (line 279)
        import time
        import uuid

        node_id = str(uuid.uuid4())
        now = int(time.time())
        graph._conn.execute(
            """INSERT INTO nodes (id, type, tier, text, rationale, embedding,
               precision_bits, weight, project, scope, source, last_accessed,
               created_at, session_count)
               VALUES (?, 'observation', 1, 'no embed', NULL, NULL, 32, 1.0,
               ?, 'project', 'jsonl', ?, ?, 1)""",
            (node_id, TEST_PROJECT, now, now),
        )
        graph._conn.commit()
        results = graph.find_similar(
            dummy_embedding, TEST_PROJECT, threshold=0.0, limit=100
        )
        # Node without embedding should not appear in results
        assert all(n.id != node_id for n in results)


# ---------------------------------------------------------------------------
# delete_node — empty-id guard (line 354)
# ---------------------------------------------------------------------------


class TestDeleteNodeGuard:
    def test_delete_node_empty_string_is_noop(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        node_id = graph.write_node(_make_node(dummy_embedding))
        graph.delete_node("")
        # Original node should still exist
        assert graph.get_node(node_id) is not None

    def test_delete_node_valid_id_removes_node(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        node_id = graph.write_node(_make_node(dummy_embedding))
        graph.delete_node(node_id)
        assert graph.get_node(node_id) is None


# ---------------------------------------------------------------------------
# write_session — validation guards (lines 530, 534)
# ---------------------------------------------------------------------------


class TestWriteSessionValidation:
    def test_ended_before_started_raises(self, graph: Graph) -> None:
        import uuid

        with pytest.raises(ValueError, match="ended_at"):
            graph.write_session(
                session_id=str(uuid.uuid4()),
                project=TEST_PROJECT,
                started_at=2000,
                ended_at=1999,
                nodes_written=0,
                nodes_evicted=0,
                nodes_promoted=0,
                transcript_path="/tmp/t.jsonl",
            )

    def test_negative_nodes_written_raises(self, graph: Graph) -> None:
        import time
        import uuid

        now = int(time.time())
        with pytest.raises(ValueError, match="non-negative"):
            graph.write_session(
                session_id=str(uuid.uuid4()),
                project=TEST_PROJECT,
                started_at=now - 10,
                ended_at=now,
                nodes_written=-1,
                nodes_evicted=0,
                nodes_promoted=0,
                transcript_path="/tmp/t.jsonl",
            )

    def test_negative_nodes_evicted_raises(self, graph: Graph) -> None:
        import time
        import uuid

        now = int(time.time())
        with pytest.raises(ValueError, match="non-negative"):
            graph.write_session(
                session_id=str(uuid.uuid4()),
                project=TEST_PROJECT,
                started_at=now - 10,
                ended_at=now,
                nodes_written=0,
                nodes_evicted=-1,
                nodes_promoted=0,
                transcript_path="/tmp/t.jsonl",
            )


# ---------------------------------------------------------------------------
# _enforce_budget — tier-3 ids already included branch (line 245-246)
# ---------------------------------------------------------------------------


class TestEnforceBudgetSkipDuplicates:
    def test_tier3_nodes_not_duplicated_in_result(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        # Build a tier-3 node and a ScoredNode referencing the same object
        from core.graph import Node
        from core.retrieval import ScoredNode, _enforce_budget

        node = _make_node(dummy_embedding)
        node_id = graph.write_node(node)
        fetched = graph.get_node(node_id)
        assert fetched is not None
        fetched_tier3 = Node(
            id=fetched.id,
            type=fetched.type,
            tier=3,
            text=fetched.text,
            rationale=fetched.rationale,
            embedding=fetched.embedding,
            precision_bits=fetched.precision_bits,
            weight=fetched.weight,
            project=fetched.project,
            scope=fetched.scope,
            source=fetched.source,
            last_accessed=fetched.last_accessed,
            created_at=fetched.created_at,
            session_count=fetched.session_count,
        )
        # Put the same node into both tier3 and fused_candidates
        scored = ScoredNode(node=fetched_tier3, score=0.9)
        result = _enforce_budget(
            tier3=[fetched_tier3],
            fused_candidates=[scored],
            budget_tokens=10000,
        )
        count = sum(1 for n in result if n.id == fetched.id)
        assert count == 1, "same node should not appear twice"


# ---------------------------------------------------------------------------
# _meets_promotion_criteria — tier >= 3 returns False (line 75)
# ---------------------------------------------------------------------------


class TestMeetsPromotionCriteriaTier3:
    def test_tier3_node_never_meets_criteria(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        from core.decay import _meets_promotion_criteria
        from core.graph import Node

        node = _make_node(dummy_embedding)
        node_id = graph.write_node(node)
        fetched = graph.get_node(node_id)
        assert fetched is not None
        tier3_node = Node(
            id=fetched.id,
            type=fetched.type,
            tier=3,
            text=fetched.text,
            rationale=fetched.rationale,
            embedding=fetched.embedding,
            precision_bits=fetched.precision_bits,
            weight=fetched.weight,
            project=fetched.project,
            scope=fetched.scope,
            source=fetched.source,
            last_accessed=fetched.last_accessed,
            created_at=fetched.created_at,
            session_count=fetched.session_count,
        )
        assert (
            _meets_promotion_criteria(tier3_node, decayed_weight=99.0, now=0) is False
        )


# ---------------------------------------------------------------------------
# TestGetSessionRecords
# ---------------------------------------------------------------------------


class TestGetSessionRecords:
    def _write_session(
        self,
        graph: Graph,
        ended_at: int,
        nodes_written: int = 1,
        tokens_raw: int | None = None,
        tokens_injected: int | None = None,
    ) -> str:
        import uuid as _uuid

        sid = str(_uuid.uuid4())
        graph.write_session(
            session_id=sid,
            project=TEST_PROJECT,
            started_at=ended_at - 60,
            ended_at=ended_at,
            nodes_written=nodes_written,
            nodes_evicted=0,
            nodes_promoted=0,
            transcript_path="",
            tokens_raw=tokens_raw,
            tokens_injected=tokens_injected,
        )
        return sid

    def test_empty_returns_empty_list(self, graph: Graph) -> None:
        records = graph.get_session_records(TEST_PROJECT)
        assert records == []

    def test_returns_written_session(self, graph: Graph) -> None:
        sid = self._write_session(graph, ended_at=2000)
        records = graph.get_session_records(TEST_PROJECT)
        assert len(records) == 1
        assert records[0]["id"] == sid

    def test_ordered_by_ended_at_desc(self, graph: Graph) -> None:
        sid_old = self._write_session(graph, ended_at=1000)
        sid_new = self._write_session(graph, ended_at=2000)
        records = graph.get_session_records(TEST_PROJECT)
        assert records[0]["id"] == sid_new
        assert records[1]["id"] == sid_old

    def test_limit_respected(self, graph: Graph) -> None:
        for i in range(5):
            self._write_session(graph, ended_at=1000 + i * 100)
        records = graph.get_session_records(TEST_PROJECT, limit=2)
        assert len(records) == 2

    def test_tokens_raw_injected_stored(self, graph: Graph) -> None:
        self._write_session(graph, ended_at=3000, tokens_raw=800, tokens_injected=300)
        records = graph.get_session_records(TEST_PROJECT)
        assert records[0]["tokens_raw"] == 800
        assert records[0]["tokens_injected"] == 300

    def test_null_tokens_stored_as_none(self, graph: Graph) -> None:
        self._write_session(graph, ended_at=4000, tokens_raw=None, tokens_injected=None)
        records = graph.get_session_records(TEST_PROJECT)
        assert records[0]["tokens_raw"] is None
        assert records[0]["tokens_injected"] is None

    def test_different_project_excluded(self, graph: Graph) -> None:
        self._write_session(graph, ended_at=5000)
        import uuid as _uuid

        other_sid = str(_uuid.uuid4())
        graph.write_session(
            session_id=other_sid,
            project="/other/project",
            started_at=4940,
            ended_at=5000,
            nodes_written=1,
            nodes_evicted=0,
            nodes_promoted=0,
            transcript_path="",
        )
        records = graph.get_session_records(TEST_PROJECT)
        ids = [r["id"] for r in records]
        assert other_sid not in ids

    def test_records_include_nodes_written(self, graph: Graph) -> None:
        self._write_session(graph, ended_at=6000, nodes_written=7)
        records = graph.get_session_records(TEST_PROJECT)
        assert records[0]["nodes_written"] == 7


# ---------------------------------------------------------------------------
# TestQueryNodesPageSortCols
# ---------------------------------------------------------------------------


class TestQueryNodesPageSortCols:
    def test_sort_by_created_at(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, text="older node"))
        graph.write_node(_make_node(dummy_embedding, text="newer node"))
        nodes, total = graph.query_nodes_page(TEST_PROJECT, sort_col="created_at")
        assert total == 2
        assert len(nodes) == 2

    def test_sort_by_last_accessed(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, text="node a"))
        graph.write_node(_make_node(dummy_embedding, text="node b"))
        _nodes, total = graph.query_nodes_page(TEST_PROJECT, sort_col="last_accessed")
        assert total == 2

    def test_source_filter_in_page(
        self, graph: Graph, dummy_embedding: np.ndarray
    ) -> None:
        graph.write_node(_make_node(dummy_embedding, text="jsonl node", source="jsonl"))
        graph.write_node(_make_node(dummy_embedding, text="ast node", source="ast"))
        nodes, total = graph.query_nodes_page(TEST_PROJECT, source="jsonl")
        assert total == 1
        assert nodes[0].source == "jsonl"

    def test_empty_project_returns_zero_total(self, graph: Graph) -> None:
        nodes, total = graph.query_nodes_page(TEST_PROJECT)
        assert total == 0
        assert nodes == []
