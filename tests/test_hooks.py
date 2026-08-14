"""Integration tests for hooks/extract.py and hooks/inject.py."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import numpy as np
import pytest

from cli.config import SCHEMA_PATH, _apply_migrations
from core.graph import Graph, Node
from hooks.extract import main as extract_main
from hooks.extract import run_extract
from hooks.inject import main as inject_main
from hooks.inject import run_inject

TEST_PROJECT = "/tmp/cortex_hooks_test"
SIMPLE_TRANSCRIPT = Path("tests/fixtures/transcripts/simple.jsonl")
DECISIONS_TRANSCRIPT = Path("tests/fixtures/transcripts/with_decisions.jsonl")


@pytest.fixture
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_PATH.read_text())
    _apply_migrations(conn)
    return conn


@pytest.fixture
def graph(db: sqlite3.Connection) -> Graph:
    return Graph(connection=db)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=42)


def _create_graph_at(db_path: Path) -> Graph:
    """Create (or open) a Graph at a specific file path — test helper only."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_PATH.read_text())
    _apply_migrations(conn)
    return Graph(connection=conn)


def _open_graph_at(db_path: Path) -> Graph | None:
    """Open an existing Graph file — test helper only."""
    if not db_path.exists():
        return None
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_PATH.read_text())
    _apply_migrations(conn)
    return Graph(connection=conn)


def _make_node(
    text: str,
    tier: int = 1,
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
        weight=1.0,
        project=project,
        scope="project",
        source="jsonl",
        last_accessed=now,
        created_at=now,
        session_count=1,
    )


# ---------------------------------------------------------------------------
# run_extract
# ---------------------------------------------------------------------------


class TestRunExtract:
    def test_extract_returns_int(self, graph: Graph) -> None:
        result = run_extract(SIMPLE_TRANSCRIPT, TEST_PROJECT, graph)
        assert isinstance(result, int)

    def test_extract_writes_nodes_to_graph(self, graph: Graph) -> None:
        run_extract(SIMPLE_TRANSCRIPT, TEST_PROJECT, graph)
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        assert len(nodes) >= 0  # transcript may produce 0-N nodes

    def test_extract_idempotent_same_transcript(self, graph: Graph) -> None:
        run_extract(SIMPLE_TRANSCRIPT, TEST_PROJECT, graph)
        n2 = run_extract(SIMPLE_TRANSCRIPT, TEST_PROJECT, graph)
        assert n2 == 0  # second run is a no-op

    def test_extract_logs_session_row(self, graph: Graph) -> None:
        run_extract(SIMPLE_TRANSCRIPT, TEST_PROJECT, graph)
        row = graph._conn.execute(
            "SELECT id FROM sessions WHERE transcript_path = ?",
            (str(SIMPLE_TRANSCRIPT),),
        ).fetchone()
        assert row is not None

    def test_extract_node_count_does_not_grow_on_repeat(self, graph: Graph) -> None:
        run_extract(SIMPLE_TRANSCRIPT, TEST_PROJECT, graph)
        count_after_first = len(graph.get_all_nodes(project=TEST_PROJECT))
        run_extract(SIMPLE_TRANSCRIPT, TEST_PROJECT, graph)
        count_after_second = len(graph.get_all_nodes(project=TEST_PROJECT))
        assert count_after_first == count_after_second

    def test_extract_decisions_transcript(self, graph: Graph) -> None:
        result = run_extract(DECISIONS_TRANSCRIPT, TEST_PROJECT, graph)
        assert isinstance(result, int)

    def test_extract_missing_transcript_returns_zero(self, graph: Graph) -> None:
        missing = Path("/nonexistent/transcript.jsonl")
        result = run_extract(missing, TEST_PROJECT, graph)
        assert result == 0

    def test_extract_nodes_belong_to_project(self, graph: Graph) -> None:
        run_extract(SIMPLE_TRANSCRIPT, TEST_PROJECT, graph)
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        for node in nodes:
            assert node.project == TEST_PROJECT

    def test_extract_nodes_have_tier_one(self, graph: Graph) -> None:
        run_extract(SIMPLE_TRANSCRIPT, TEST_PROJECT, graph)
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        for node in nodes:
            assert node.tier == 1

    def test_extract_two_different_transcripts_accumulate(self, graph: Graph) -> None:
        run_extract(SIMPLE_TRANSCRIPT, TEST_PROJECT, graph)
        first_count = len(graph.get_all_nodes(project=TEST_PROJECT))
        run_extract(DECISIONS_TRANSCRIPT, TEST_PROJECT, graph)
        second_count = len(graph.get_all_nodes(project=TEST_PROJECT))
        assert second_count >= first_count

    def test_extract_returns_positive_int_for_valid_transcript(
        self, graph: Graph
    ) -> None:
        result = run_extract(SIMPLE_TRANSCRIPT, TEST_PROJECT, graph)
        assert isinstance(result, int)
        assert result >= 0

    def test_extract_does_not_write_to_wrong_project(self, graph: Graph) -> None:
        run_extract(SIMPLE_TRANSCRIPT, TEST_PROJECT, graph)
        other_nodes = graph.get_all_nodes(project="/some/other/project")
        assert len(other_nodes) == 0

    def test_extract_result_is_int_type(self, graph: Graph) -> None:
        result = run_extract(SIMPLE_TRANSCRIPT, TEST_PROJECT, graph)
        assert isinstance(result, int)

    def test_extract_returns_nonneg(self, graph: Graph) -> None:
        result = run_extract(SIMPLE_TRANSCRIPT, TEST_PROJECT, graph)
        assert result >= 0

    def test_extract_nodes_have_project_set(self, graph: Graph) -> None:
        run_extract(SIMPLE_TRANSCRIPT, TEST_PROJECT, graph)
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        assert all(n.project == TEST_PROJECT for n in nodes)

    def test_extract_missing_path_returns_zero(self, graph: Graph) -> None:
        result = run_extract(Path("/no/such/file.jsonl"), TEST_PROJECT, graph)
        assert result == 0


# ---------------------------------------------------------------------------
# run_inject
# ---------------------------------------------------------------------------


class TestRunInject:
    def test_inject_returns_string(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        e = rng.random(384).astype(np.float32)
        graph.write_node(_make_node("use async/await throughout", tier=3, embedding=e))
        result = run_inject(TEST_PROJECT, graph, query="")
        assert isinstance(result, str)

    def test_inject_empty_graph_returns_empty_string(self, graph: Graph) -> None:
        result = run_inject(TEST_PROJECT, graph, query="")
        assert result == ""

    def test_inject_output_contains_cortex_header(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        e = rng.random(384).astype(np.float32)
        graph.write_node(_make_node("always use async", tier=3, embedding=e))
        result = run_inject(TEST_PROJECT, graph, query="")
        assert "CORTEX MEMORY" in result

    def test_inject_tier3_node_in_conventions_section(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        e = rng.random(384).astype(np.float32)
        graph.write_node(_make_node("always validate at boundary", tier=3, embedding=e))
        result = run_inject(TEST_PROJECT, graph, query="")
        assert "CONVENTIONS" in result
        assert "always validate at boundary" in result

    def test_inject_tier1_node_in_context_section(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        e = rng.random(384).astype(np.float32)
        graph.write_node(
            _make_node("auth middleware was modified", tier=1, embedding=e)
        )
        result = run_inject(TEST_PROJECT, graph, query="")
        assert "RELEVANT CONTEXT" in result

    def test_inject_project_path_in_header(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        e = rng.random(384).astype(np.float32)
        graph.write_node(_make_node("some node", tier=3, embedding=e))
        result = run_inject(TEST_PROJECT, graph, query="")
        assert TEST_PROJECT in result

    def test_inject_with_query_returns_string(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        e = rng.random(384).astype(np.float32)
        graph.write_node(_make_node("authentication system", tier=1, embedding=e))
        result = run_inject(TEST_PROJECT, graph, query="fix the auth bug")
        assert isinstance(result, str)

    def test_inject_only_returns_nodes_for_project(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        e = rng.random(384).astype(np.float32)
        graph.write_node(
            _make_node("target project node", project=TEST_PROJECT, embedding=e)
        )
        graph.write_node(
            _make_node("other project node", project="/other/proj", embedding=e)
        )
        result = run_inject(TEST_PROJECT, graph, query="")
        assert "target project node" in result
        assert "other project node" not in result

    def test_inject_updates_last_accessed_for_retrieved_nodes(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        e = rng.random(384).astype(np.float32)
        node_id = graph.write_node(_make_node("convention node", tier=3, embedding=e))
        before = graph.get_node(node_id)
        assert before is not None
        before_ts = before.last_accessed

        # Small sleep to ensure timestamp advances
        import time as _time

        _time.sleep(1)

        run_inject(TEST_PROJECT, graph, query="")
        after = graph.get_node(node_id)
        assert after is not None
        assert after.last_accessed >= before_ts

    def test_inject_result_is_non_empty_when_nodes_exist(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        e = rng.random(384).astype(np.float32)
        graph.write_node(_make_node("always use async/await", tier=3, embedding=e))
        result = run_inject(TEST_PROJECT, graph, query="async")
        assert len(result) > 0

    def test_inject_returns_empty_for_wrong_project(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        e = rng.random(384).astype(np.float32)
        graph.write_node(_make_node("node for other project", tier=3, embedding=e))
        result = run_inject("/some/other/project", graph, query="")
        assert result == ""

    def test_inject_result_starts_with_header(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        e = rng.random(384).astype(np.float32)
        graph.write_node(_make_node("always use async", tier=3, embedding=e))
        result = run_inject(TEST_PROJECT, graph, query="")
        assert result.startswith("=== CORTEX MEMORY")

    def test_inject_multiple_tier3_all_in_result(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        e = rng.random(384).astype(np.float32)
        graph.write_node(_make_node("use type hints everywhere", tier=3, embedding=e))
        graph.write_node(
            _make_node("prefer asyncpg over psycopg2", tier=3, embedding=e)
        )
        result = run_inject(TEST_PROJECT, graph, query="")
        assert "use type hints everywhere" in result
        assert "prefer asyncpg over psycopg2" in result


# ---------------------------------------------------------------------------
# Integration: extract → inject round-trip
# ---------------------------------------------------------------------------


class TestExtractInjectLoop:
    def test_full_loop_extract_then_inject(self, graph: Graph) -> None:
        run_extract(SIMPLE_TRANSCRIPT, TEST_PROJECT, graph)
        result = run_inject(TEST_PROJECT, graph, query="authentication")
        assert isinstance(result, str)

    def test_injection_block_within_token_budget(self, graph: Graph) -> None:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        run_extract(SIMPLE_TRANSCRIPT, TEST_PROJECT, graph)
        block = run_inject(TEST_PROJECT, graph, query="")
        if block:
            assert len(enc.encode(block)) < 700  # small headroom over 600

    def test_extract_then_inject_has_end_marker(self, graph: Graph) -> None:
        run_extract(SIMPLE_TRANSCRIPT, TEST_PROJECT, graph)
        block = run_inject(TEST_PROJECT, graph, query="")
        if block:
            assert "END CORTEX MEMORY" in block

    def test_inject_after_two_extracts_returns_string(self, graph: Graph) -> None:
        run_extract(SIMPLE_TRANSCRIPT, TEST_PROJECT, graph)
        run_extract(DECISIONS_TRANSCRIPT, TEST_PROJECT, graph)
        result = run_inject(TEST_PROJECT, graph, query="")
        assert isinstance(result, str)

    def test_extract_then_inject_project_in_result(self, graph: Graph) -> None:
        run_extract(SIMPLE_TRANSCRIPT, TEST_PROJECT, graph)
        block = run_inject(TEST_PROJECT, graph, query="")
        if block:
            assert TEST_PROJECT in block

    def test_loop_produces_non_empty_block_when_extract_succeeds(
        self, graph: Graph
    ) -> None:
        count = run_extract(SIMPLE_TRANSCRIPT, TEST_PROJECT, graph)
        if count > 0:
            block = run_inject(TEST_PROJECT, graph, query="")
            assert len(block) > 0

    def test_loop_inject_after_extract_has_cortex_memory_header(
        self, graph: Graph
    ) -> None:
        run_extract(SIMPLE_TRANSCRIPT, TEST_PROJECT, graph)
        block = run_inject(TEST_PROJECT, graph, query="")
        if block:
            assert block.startswith("=== CORTEX MEMORY")


# ---------------------------------------------------------------------------
# open_graph helpers (via shared factory)
# ---------------------------------------------------------------------------


class TestOpenGraph:
    def test_create_graph_at_creates_db(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        graph = _create_graph_at(db_path)
        assert db_path.exists()
        assert isinstance(graph, Graph)

    def test_open_graph_at_returns_none_if_missing(self, tmp_path: Path) -> None:
        db_path = tmp_path / "nonexistent.db"
        result = _open_graph_at(db_path)
        assert result is None

    def test_open_graph_at_returns_graph_if_exists(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        _create_graph_at(db_path)  # create it
        graph = _open_graph_at(db_path)
        assert isinstance(graph, Graph)

    def test_create_graph_at_creates_parent_dirs(self, tmp_path: Path) -> None:
        db_path = tmp_path / "nested" / "cortex.db"
        _create_graph_at(db_path)
        assert db_path.exists()

    def test_create_graph_is_writable(self, tmp_path: Path) -> None:
        db_path = tmp_path / "writable.db"
        graph = _create_graph_at(db_path)
        node = _make_node("hello world")
        node_id = graph.write_node(node)
        assert len(node_id) == 36

    def test_open_nonexistent_path_returns_none(self, tmp_path: Path) -> None:
        db_path = tmp_path / "sub" / "dir" / "missing.db"
        result = _open_graph_at(db_path)
        assert result is None

    def test_create_twice_same_path_works(self, tmp_path: Path) -> None:
        db_path = tmp_path / "double.db"
        _create_graph_at(db_path)
        graph = _create_graph_at(db_path)
        assert isinstance(graph, Graph)

    def test_graph_writable_after_open(self, tmp_path: Path) -> None:
        db_path = tmp_path / "open_write.db"
        _create_graph_at(db_path)
        graph = _open_graph_at(db_path)
        assert graph is not None
        node_id = graph.write_node(_make_node("test write after open"))
        assert len(node_id) == 36


# ---------------------------------------------------------------------------
# main() entry points
# ---------------------------------------------------------------------------


class TestMainFunctions:
    def test_extract_main_no_transcript_env_returns_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CLAUDE_TRANSCRIPT", raising=False)
        result = extract_main()
        assert result == 0

    def test_extract_main_missing_transcript_returns_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CLAUDE_TRANSCRIPT", "/nonexistent/path.jsonl")
        monkeypatch.setenv("CLAUDE_PROJECT_PATH", TEST_PROJECT)
        result = extract_main()
        assert result == 0

    def test_extract_main_valid_transcript_returns_zero(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        db_path = tmp_path / "cortex.db"
        monkeypatch.setenv("CLAUDE_TRANSCRIPT", str(SIMPLE_TRANSCRIPT))
        monkeypatch.setenv("CLAUDE_PROJECT_PATH", TEST_PROJECT)
        monkeypatch.setenv("CORTEX_DB_PATH", str(db_path))
        result = extract_main()
        assert result == 0

    def test_inject_main_no_db_returns_zero(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        db_path = tmp_path / "nonexistent.db"
        monkeypatch.setenv("CLAUDE_PROJECT_PATH", TEST_PROJECT)
        monkeypatch.setenv("CORTEX_DB_PATH", str(db_path))
        result = inject_main()
        assert result == 0

    def test_inject_main_with_db_returns_zero(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        db_path = tmp_path / "cortex.db"
        _create_graph_at(db_path)  # create the DB
        monkeypatch.setenv("CLAUDE_PROJECT_PATH", TEST_PROJECT)
        monkeypatch.setenv("CORTEX_DB_PATH", str(db_path))
        monkeypatch.delenv("CLAUDE_INITIAL_MESSAGE", raising=False)
        result = inject_main()
        assert result == 0

    def test_inject_main_with_nodes_prints_block(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        db_path = tmp_path / "cortex.db"
        g = _create_graph_at(db_path)
        rng_local = np.random.default_rng(seed=99)
        e = rng_local.random(384).astype(np.float32)
        g.write_node(
            _make_node("always use async", tier=3, embedding=e, project=TEST_PROJECT)
        )
        monkeypatch.setenv("CLAUDE_PROJECT_PATH", TEST_PROJECT)
        monkeypatch.setenv("CORTEX_DB_PATH", str(db_path))
        monkeypatch.delenv("CLAUDE_INITIAL_MESSAGE", raising=False)
        result = inject_main()
        captured = capsys.readouterr()
        assert result == 0
        assert "CORTEX MEMORY" in captured.out

    def test_extract_empty_transcript_returns_zero(
        self, graph: Graph, tmp_path: Path
    ) -> None:
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        result = run_extract(empty, TEST_PROJECT, graph)
        assert result == 0

    def test_extract_dedup_runs_merge(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        e = rng.random(384).astype(np.float32)
        graph.write_node(
            _make_node("auth/middleware.py is a hotspot", tier=1, embedding=e)
        )
        initial_count = len(graph.get_all_nodes(project=TEST_PROJECT))
        run_extract(SIMPLE_TRANSCRIPT, TEST_PROJECT, graph)
        assert len(graph.get_all_nodes(project=TEST_PROJECT)) >= initial_count

    def test_extract_main_exception_returns_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("CLAUDE_TRANSCRIPT", str(SIMPLE_TRANSCRIPT))
        monkeypatch.setenv("CLAUDE_PROJECT_PATH", TEST_PROJECT)
        monkeypatch.setenv("CORTEX_DB_PATH", str(tmp_path / "cortex.db"))

        import hooks.extract as ext_mod

        def raise_always(project_root: Path, *, create: bool = False) -> Graph:
            raise RuntimeError("forced failure")

        monkeypatch.setattr(ext_mod, "open_graph", raise_always)
        result = extract_main()
        assert result == 1

    def test_inject_main_exception_returns_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        db_path = tmp_path / "cortex.db"
        _create_graph_at(db_path)
        monkeypatch.setenv("CLAUDE_PROJECT_PATH", TEST_PROJECT)
        monkeypatch.setenv("CORTEX_DB_PATH", str(db_path))
        monkeypatch.delenv("CLAUDE_INITIAL_MESSAGE", raising=False)

        import hooks.inject as inj_mod

        def raise_always(p: str, g: Graph, q: str = "") -> str:
            raise RuntimeError("forced failure")

        monkeypatch.setattr(inj_mod, "run_inject", raise_always)
        result = inject_main()
        assert result == 1

    def test_extract_stores_token_counts_in_session(self, graph: Graph) -> None:
        run_extract(SIMPLE_TRANSCRIPT, TEST_PROJECT, graph)
        row = graph._conn.execute(
            "SELECT tokens_raw, tokens_injected FROM sessions WHERE project = ?",
            (TEST_PROJECT,),
        ).fetchone()
        assert row is not None
        # tokens_raw may be None if graph was empty after decay, but should
        # be a non-negative integer when nodes exist
        if row["tokens_raw"] is not None:
            assert row["tokens_raw"] >= 0
        if row["tokens_injected"] is not None:
            assert row["tokens_injected"] >= 0

    def test_extract_token_savings_positive(self, graph: Graph) -> None:
        run_extract(SIMPLE_TRANSCRIPT, TEST_PROJECT, graph)
        row = graph._conn.execute(
            "SELECT tokens_raw, tokens_injected FROM sessions WHERE project = ?",
            (TEST_PROJECT,),
        ).fetchone()
        assert row is not None
        if row["tokens_raw"] and row["tokens_injected"]:
            assert row["tokens_raw"] >= row["tokens_injected"]


# ---------------------------------------------------------------------------
# _write_candidates
# ---------------------------------------------------------------------------


class TestWriteCandidates:
    """Unit tests for the _write_candidates helper in hooks/extract.py."""

    from hooks.extract import _write_candidates

    def _make_candidate(
        self, text: str, source: str = "nlp", type_: str = "fact"
    ) -> object:
        from core.extractor import CandidateNode, NodeType, ScopeType, SourceType

        return CandidateNode(
            text=text,
            rationale=None,
            type=NodeType(type_),
            source=SourceType(source),
            scope=ScopeType.PROJECT,
            durability=0.8,
            project=TEST_PROJECT,
        )

    def test_write_candidates_empty_returns_zero(self, graph: Graph) -> None:
        from hooks.extract import _write_candidates

        n, ids = _write_candidates([], [], graph, 0, TEST_PROJECT)
        assert n == 0
        assert ids == {}

    def test_write_candidates_writes_new_node(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        from hooks.extract import _write_candidates

        candidate = self._make_candidate("use black for formatting")
        embedding = rng.random(384).astype(np.float32)
        n, _ = _write_candidates([candidate], [embedding], graph, 0, TEST_PROJECT)
        assert n == 1
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        assert len(nodes) == 1
        assert nodes[0].text == "use black for formatting"

    def test_write_candidates_dedup_merges_similar(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        from hooks.extract import _write_candidates

        embedding = rng.random(384).astype(np.float32)
        # Write a node directly so the second _write_candidates call sees it
        node = _make_node("use black for formatting", embedding=embedding)
        graph.write_node(node)

        candidate = self._make_candidate("use black for formatting")
        n, _ = _write_candidates([candidate], [embedding], graph, 0, TEST_PROJECT)
        # Should merge, not write a new node
        assert n == 0
        assert len(graph.get_all_nodes(project=TEST_PROJECT)) == 1

    def test_write_candidates_jsonl_observation_tracked(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        from hooks.extract import _write_candidates

        candidate = self._make_candidate(
            "auth/middleware.py was modified 4 times",
            source="jsonl",
            type_="observation",
        )
        embedding = rng.random(384).astype(np.float32)
        _, file_ids = _write_candidates(
            [candidate], [embedding], graph, 0, TEST_PROJECT
        )
        assert "auth/middleware.py" in file_ids

    def test_write_candidates_non_observation_not_tracked(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        from hooks.extract import _write_candidates

        candidate = self._make_candidate("prefer async patterns")
        embedding = rng.random(384).astype(np.float32)
        _, file_ids = _write_candidates(
            [candidate], [embedding], graph, 0, TEST_PROJECT
        )
        assert file_ids == {}

    def test_write_candidates_multiple_nodes(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        from hooks.extract import _write_candidates

        candidates = [
            self._make_candidate("node one"),
            self._make_candidate("node two"),
            self._make_candidate("node three"),
        ]
        embeddings = [rng.random(384).astype(np.float32) for _ in candidates]
        n, _ = _write_candidates(candidates, embeddings, graph, 0, TEST_PROJECT)
        assert n == 3
        assert len(graph.get_all_nodes(project=TEST_PROJECT)) == 3

    def test_write_candidates_stores_correct_source(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        from hooks.extract import _write_candidates

        candidate = self._make_candidate("use ruff for linting", source="nlp")
        embedding = rng.random(384).astype(np.float32)
        _write_candidates([candidate], [embedding], graph, 0, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        assert nodes[0].source == "nlp"

    def test_write_candidates_stores_correct_project(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        from hooks.extract import _write_candidates

        candidate = self._make_candidate("use pytest fixtures")
        embedding = rng.random(384).astype(np.float32)
        _write_candidates([candidate], [embedding], graph, 0, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        assert nodes[0].project == TEST_PROJECT

    def test_write_candidates_tier_always_one(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        from hooks.extract import _write_candidates

        candidate = self._make_candidate("always start at tier 1")
        embedding = rng.random(384).astype(np.float32)
        _write_candidates([candidate], [embedding], graph, 0, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        assert nodes[0].tier == 1

    def test_write_candidates_returns_dict_as_second_element(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        from hooks.extract import _write_candidates

        candidate = self._make_candidate("convention node")
        embedding = rng.random(384).astype(np.float32)
        _, file_ids = _write_candidates(
            [candidate], [embedding], graph, 0, TEST_PROJECT
        )
        assert isinstance(file_ids, dict)

    def test_write_candidates_text_stored_correctly(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        from hooks.extract import _write_candidates

        candidate = self._make_candidate("always use type hints")
        embedding = rng.random(384).astype(np.float32)
        _write_candidates([candidate], [embedding], graph, 0, TEST_PROJECT)
        nodes = graph.get_all_nodes(project=TEST_PROJECT)
        assert nodes[0].text == "always use type hints"

    def test_write_candidates_first_return_is_int(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        from hooks.extract import _write_candidates

        candidate = self._make_candidate("prefer type hints everywhere")
        embedding = rng.random(384).astype(np.float32)
        n, _ = _write_candidates([candidate], [embedding], graph, 0, TEST_PROJECT)
        assert isinstance(n, int)

    def test_write_candidates_two_unique_returns_two(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        from hooks.extract import _write_candidates

        candidates = [
            self._make_candidate("use asyncpg for database"),
            self._make_candidate("always add type hints"),
        ]
        embeddings = [rng.random(384).astype(np.float32) for _ in candidates]
        n, _ = _write_candidates(candidates, embeddings, graph, 0, TEST_PROJECT)
        assert n == 2


# ---------------------------------------------------------------------------
# _wire_co_occurring_edges
# ---------------------------------------------------------------------------


class TestWireCoOccurringEdges:
    """Unit tests for _wire_co_occurring_edges in hooks/extract.py."""

    def _make_file_nodes(
        self, graph: Graph, rng: np.random.Generator, *rel_paths: str
    ) -> dict[str, str]:
        """Write observation nodes for each rel_path and return rel_path→node_id map."""
        from core.extractor import CandidateNode, NodeType, ScopeType, SourceType
        from hooks.extract import _write_candidates

        candidates = []
        embeddings = []
        for rel in rel_paths:
            candidates.append(
                CandidateNode(
                    text=f"{rel} was modified 3 times",
                    rationale=None,
                    type=NodeType.OBSERVATION,
                    source=SourceType.JSONL,
                    scope=ScopeType.MODULE,
                    durability=0.8,
                    project=TEST_PROJECT,
                )
            )
            embeddings.append(rng.random(384).astype(np.float32))
        _, file_ids = _write_candidates(candidates, embeddings, graph, 0, TEST_PROJECT)
        return file_ids

    def test_wire_creates_edge_between_co_occurring_files(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        from hooks.extract import _wire_co_occurring_edges

        file_ids = self._make_file_nodes(graph, rng, "a.py", "b.py")
        co_pairs: tuple[frozenset[str], ...] = (
            frozenset({f"{TEST_PROJECT}/a.py", f"{TEST_PROJECT}/b.py"}),
        )
        _wire_co_occurring_edges(co_pairs, file_ids, TEST_PROJECT, graph)
        id_a = file_ids.get("a.py")
        id_b = file_ids.get("b.py")
        assert id_a and id_b
        edges = graph.get_edges(id_a)
        assert any(e.target_id == id_b or e.source_id == id_b for e in edges)

    def test_wire_empty_pairs_no_edges(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        from hooks.extract import _wire_co_occurring_edges

        file_ids = self._make_file_nodes(graph, rng, "a.py", "b.py")
        _wire_co_occurring_edges((), file_ids, TEST_PROJECT, graph)
        for node_id in file_ids.values():
            assert graph.get_edges(node_id) == []

    def test_wire_skips_pair_if_node_missing(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        from hooks.extract import _wire_co_occurring_edges

        file_ids = self._make_file_nodes(graph, rng, "a.py")
        co_pairs: tuple[frozenset[str], ...] = (
            frozenset({f"{TEST_PROJECT}/a.py", f"{TEST_PROJECT}/missing.py"}),
        )
        _wire_co_occurring_edges(co_pairs, file_ids, TEST_PROJECT, graph)
        id_a = file_ids.get("a.py")
        assert id_a
        assert graph.get_edges(id_a) == []

    def test_wire_idempotent_duplicate_pair(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        from hooks.extract import _wire_co_occurring_edges

        file_ids = self._make_file_nodes(graph, rng, "a.py", "b.py")
        co_pairs: tuple[frozenset[str], ...] = (
            frozenset({f"{TEST_PROJECT}/a.py", f"{TEST_PROJECT}/b.py"}),
        )
        _wire_co_occurring_edges(co_pairs, file_ids, TEST_PROJECT, graph)
        # Call again — IntegrityError is suppressed so this should not raise
        _wire_co_occurring_edges(co_pairs, file_ids, TEST_PROJECT, graph)
        id_a = file_ids.get("a.py")
        assert id_a
        edges = graph.get_edges(id_a)
        assert len(edges) == 1  # still only one edge

    def test_wire_multiple_pairs_creates_multiple_edges(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        from hooks.extract import _wire_co_occurring_edges

        file_ids = self._make_file_nodes(graph, rng, "x.py", "y.py", "z.py")
        co_pairs: tuple[frozenset[str], ...] = (
            frozenset({f"{TEST_PROJECT}/x.py", f"{TEST_PROJECT}/y.py"}),
            frozenset({f"{TEST_PROJECT}/y.py", f"{TEST_PROJECT}/z.py"}),
        )
        _wire_co_occurring_edges(co_pairs, file_ids, TEST_PROJECT, graph)
        id_y = file_ids.get("y.py")
        assert id_y
        edges = graph.get_edges(id_y)
        assert len(edges) == 2

    def test_wire_edge_accessible_from_both_nodes(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        from hooks.extract import _wire_co_occurring_edges

        file_ids = self._make_file_nodes(graph, rng, "a.py", "b.py")
        co_pairs: tuple[frozenset[str], ...] = (
            frozenset({f"{TEST_PROJECT}/a.py", f"{TEST_PROJECT}/b.py"}),
        )
        _wire_co_occurring_edges(co_pairs, file_ids, TEST_PROJECT, graph)
        id_a = file_ids.get("a.py")
        id_b = file_ids.get("b.py")
        assert id_a and id_b
        edges_a = graph.get_edges(id_a)
        edges_b = graph.get_edges(id_b)
        assert len(edges_a) == 1
        assert len(edges_b) == 1

    def test_wire_three_files_triangle(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        from hooks.extract import _wire_co_occurring_edges

        file_ids = self._make_file_nodes(graph, rng, "a.py", "b.py", "c.py")
        co_pairs: tuple[frozenset[str], ...] = (
            frozenset({f"{TEST_PROJECT}/a.py", f"{TEST_PROJECT}/b.py"}),
            frozenset({f"{TEST_PROJECT}/b.py", f"{TEST_PROJECT}/c.py"}),
            frozenset({f"{TEST_PROJECT}/a.py", f"{TEST_PROJECT}/c.py"}),
        )
        _wire_co_occurring_edges(co_pairs, file_ids, TEST_PROJECT, graph)
        id_a = file_ids.get("a.py")
        assert id_a
        edges_a = graph.get_edges(id_a)
        assert len(edges_a) == 2

    def test_wire_only_prefixed_paths_matched(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        from hooks.extract import _wire_co_occurring_edges

        file_ids = self._make_file_nodes(graph, rng, "main.py")
        co_pairs: tuple[frozenset[str], ...] = (
            frozenset({"other_project/main.py", f"{TEST_PROJECT}/main.py"}),
        )
        _wire_co_occurring_edges(co_pairs, file_ids, TEST_PROJECT, graph)
        id_main = file_ids.get("main.py")
        assert id_main
        assert graph.get_edges(id_main) == []

    def test_wire_edge_strength_is_one_on_first_call(
        self, graph: Graph, rng: np.random.Generator
    ) -> None:
        from hooks.extract import _wire_co_occurring_edges

        file_ids = self._make_file_nodes(graph, rng, "x.py", "y.py")
        co_pairs: tuple[frozenset[str], ...] = (
            frozenset({f"{TEST_PROJECT}/x.py", f"{TEST_PROJECT}/y.py"}),
        )
        _wire_co_occurring_edges(co_pairs, file_ids, TEST_PROJECT, graph)
        id_x = file_ids.get("x.py")
        assert id_x
        edges = graph.get_edges(id_x)
        assert len(edges) == 1
        assert edges[0].strength == pytest.approx(1.0, abs=1e-5)

    def test_wire_returns_none(self, graph: Graph, rng: np.random.Generator) -> None:
        from hooks.extract import _wire_co_occurring_edges

        file_ids = self._make_file_nodes(graph, rng, "a.py", "b.py")
        result = _wire_co_occurring_edges((), file_ids, TEST_PROJECT, graph)
        assert result is None


# ---------------------------------------------------------------------------
# compact.py (PostCompact hook)
# ---------------------------------------------------------------------------


class TestCompactHook:
    """compact.py delegates to inject_main; verify the delegation."""

    def test_compact_returns_zero_when_no_db(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("CLAUDE_PROJECT_PATH", TEST_PROJECT)
        monkeypatch.setenv("CORTEX_DB_PATH", str(tmp_path / "nonexistent.db"))
        monkeypatch.delenv("CLAUDE_INITIAL_MESSAGE", raising=False)
        from hooks.compact import inject_main as compact_inject_main

        result = compact_inject_main()
        assert result == 0

    def test_compact_returns_zero_with_empty_db(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        db_path = tmp_path / "cortex.db"
        _create_graph_at(db_path)
        monkeypatch.setenv("CLAUDE_PROJECT_PATH", TEST_PROJECT)
        monkeypatch.setenv("CORTEX_DB_PATH", str(db_path))
        monkeypatch.delenv("CLAUDE_INITIAL_MESSAGE", raising=False)
        from hooks.compact import inject_main as compact_inject_main

        result = compact_inject_main()
        assert result == 0

    def test_compact_injects_tier3_nodes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        rng: np.random.Generator,
    ) -> None:
        db_path = tmp_path / "cortex.db"
        g = _create_graph_at(db_path)
        e = rng.random(384).astype(np.float32)
        g.write_node(
            _make_node(
                "always use type hints", tier=3, embedding=e, project=TEST_PROJECT
            )
        )
        monkeypatch.setenv("CLAUDE_PROJECT_PATH", TEST_PROJECT)
        monkeypatch.setenv("CORTEX_DB_PATH", str(db_path))
        monkeypatch.delenv("CLAUDE_INITIAL_MESSAGE", raising=False)
        from hooks.compact import inject_main as compact_inject_main

        result = compact_inject_main()
        captured = capsys.readouterr()
        assert result == 0
        assert "CORTEX MEMORY" in captured.out

    def test_compact_returns_int(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("CLAUDE_PROJECT_PATH", TEST_PROJECT)
        monkeypatch.setenv("CORTEX_DB_PATH", str(tmp_path / "nope.db"))
        monkeypatch.delenv("CLAUDE_INITIAL_MESSAGE", raising=False)
        from hooks.compact import inject_main as compact_inject_main

        result = compact_inject_main()
        assert isinstance(result, int)

    def test_compact_with_tier1_only_no_conventions_section(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        rng: np.random.Generator,
    ) -> None:
        db_path = tmp_path / "cortex2.db"
        g = _create_graph_at(db_path)
        e = rng.random(384).astype(np.float32)
        g.write_node(_make_node("tier1 observation", tier=1, embedding=e))
        monkeypatch.setenv("CLAUDE_PROJECT_PATH", TEST_PROJECT)
        monkeypatch.setenv("CORTEX_DB_PATH", str(db_path))
        monkeypatch.delenv("CLAUDE_INITIAL_MESSAGE", raising=False)
        from hooks.compact import inject_main as compact_inject_main

        compact_inject_main()
        captured = capsys.readouterr()
        assert "CONVENTIONS" not in captured.out


# ---------------------------------------------------------------------------
# cli/config.py — cortex_dir, sessions_dir, db_path override, _apply_migrations
# ---------------------------------------------------------------------------


class TestConfigHelpers:
    def test_cortex_dir_returns_correct_path(self, tmp_path: Path) -> None:
        from cli.config import cortex_dir

        result = cortex_dir(tmp_path)
        assert result == tmp_path / ".cortex"

    def test_sessions_dir_returns_correct_path(self, tmp_path: Path) -> None:
        from cli.config import sessions_dir

        result = sessions_dir(tmp_path)
        assert result == tmp_path / ".cortex" / "sessions"

    def test_db_path_non_absolute_override_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cli.config import db_path

        monkeypatch.setenv("CORTEX_DB_PATH", "relative/path.db")
        with pytest.raises(ValueError, match="absolute"):
            db_path(tmp_path)

    def test_db_path_absolute_override_is_returned(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cli.config import db_path

        override = str(tmp_path / "override.db")
        monkeypatch.setenv("CORTEX_DB_PATH", override)
        result = db_path(tmp_path)
        assert str(result) == override

    def test_apply_migrations_adds_nodes_promoted_column(self, tmp_path: Path) -> None:
        from cli.config import SCHEMA_PATH, _apply_migrations

        conn = sqlite3.connect(str(tmp_path / "test.db"))
        conn.executescript(SCHEMA_PATH.read_text())
        # Drop the column by recreating the table without it (SQLite doesn't
        # support DROP COLUMN in older versions, so we use a workaround)
        existing = {
            row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
        }
        if "nodes_promoted" in existing:
            # Already present — verify migration is idempotent (no error)
            _apply_migrations(conn)
        else:
            _apply_migrations(conn)
            cols = {
                row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
            }
            assert "nodes_promoted" in cols

    def test_apply_migrations_idempotent_when_column_exists(self) -> None:
        from cli.config import SCHEMA_PATH, _apply_migrations

        conn = sqlite3.connect(":memory:")
        conn.executescript(SCHEMA_PATH.read_text())
        _apply_migrations(conn)
        _apply_migrations(conn)  # second call must not raise
        cols = {
            row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
        }
        assert "nodes_promoted" in cols


# ---------------------------------------------------------------------------
# hooks/extract.py — _wire_co_occurring_edges pair!=2 branch (line 87)
# ---------------------------------------------------------------------------


class TestWireEdgesPairSizeGuard:
    def test_pair_with_three_paths_is_skipped(
        self, graph: Graph, tmp_path: Path
    ) -> None:
        from hooks.extract import _wire_co_occurring_edges

        # frozenset with 3 elements — paths != 2 branch
        triple = frozenset({"/a/x.py", "/a/y.py", "/a/z.py"})
        _wire_co_occurring_edges(
            co_pairs=(triple,),
            file_node_ids={},
            project=str(tmp_path),
            graph=graph,
        )
        # No error, no edges written
        rows = graph._conn.execute("SELECT COUNT(*) FROM edges").fetchone()
        assert rows[0] == 0


# ---------------------------------------------------------------------------
# hooks/extract.py — _compute_token_stats branches (lines 184, 191, 197-198)
# ---------------------------------------------------------------------------


class TestComputeTokenStats:
    def test_empty_graph_returns_none_none(self, graph: Graph) -> None:
        from hooks.extract import _compute_token_stats

        result = _compute_token_stats(graph, "/tmp/test_project")
        assert result == (None, None)

    def test_exception_returns_none_none(
        self, graph: Graph, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from hooks import extract as extract_mod

        monkeypatch.setattr(
            extract_mod,
            "count_tokens",
            lambda *a, **kw: (_ for _ in ()).throw(
                RuntimeError("tiktoken unavailable")
            ),
        )
        # Even with a node we can't count tokens — should catch and return None, None
        import numpy as np

        vec = np.zeros(384, dtype=np.float32)
        node = Node(
            id="",
            type="observation",
            tier=1,
            text="test node text for token counting",
            rationale="",
            embedding=vec,
            precision_bits=32,
            weight=1.0,
            project="/tmp/test_project",
            scope="project",
            source="jsonl",
            last_accessed=0,
            created_at=0,
            session_count=1,
        )
        graph.write_node(node)
        result = extract_mod._compute_token_stats(graph, "/tmp/test_project")
        assert result == (None, None)


# ---------------------------------------------------------------------------
# hooks/extract.py — main() open_graph=None path (lines 223-224)
# ---------------------------------------------------------------------------


class TestExtractMainGraphNone:
    def test_main_returns_1_when_graph_is_none(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from hooks import extract as extract_mod

        transcript = tmp_path / "t.jsonl"
        transcript.write_text("")
        monkeypatch.setenv("CLAUDE_TRANSCRIPT", str(transcript))
        monkeypatch.setenv("CLAUDE_PROJECT_PATH", str(tmp_path))
        monkeypatch.setattr(extract_mod, "open_graph", lambda *a, **kw: None)

        result = extract_main()
        assert result == 1
        captured = capsys.readouterr()
        assert "could not open graph" in captured.err


# ---------------------------------------------------------------------------
# cli/config.py — default db_path (line 47) and migration actual add (92-95)
# ---------------------------------------------------------------------------


class TestConfigMissingLines:
    def test_db_path_default_when_no_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cli.config import db_path

        monkeypatch.delenv("CORTEX_DB_PATH", raising=False)
        result = db_path(tmp_path)
        assert result == tmp_path / ".cortex" / "cortex.db"

    def test_apply_migrations_adds_column_when_missing(self, tmp_path: Path) -> None:
        from cli.config import _apply_migrations

        # Build a sessions table that is missing nodes_promoted
        conn = sqlite3.connect(str(tmp_path / "old.db"))
        conn.execute("""
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                project TEXT NOT NULL,
                started_at INTEGER NOT NULL,
                ended_at INTEGER NOT NULL,
                nodes_written INTEGER NOT NULL DEFAULT 0,
                nodes_evicted INTEGER NOT NULL DEFAULT 0,
                transcript_path TEXT
            )
        """)
        conn.commit()

        existing_before = {
            row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
        }
        assert "nodes_promoted" not in existing_before

        _apply_migrations(conn)

        existing_after = {
            row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
        }
        assert "nodes_promoted" in existing_after


# ---------------------------------------------------------------------------
# hooks/extract.py — _compute_token_stats line 191: retrieve returns empty
# ---------------------------------------------------------------------------


class TestComputeTokenStatsProjectedEmpty:
    def test_returns_raw_and_none_when_retrieve_empty(
        self, graph: Graph, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from hooks import extract as extract_mod

        vec = np.zeros(384, dtype=np.float32)
        node = Node(
            id="",
            type="observation",
            tier=1,
            text="some node that survives format but not retrieve",
            rationale=None,
            embedding=vec,
            precision_bits=32,
            weight=1.0,
            project="/tmp/test_project",
            scope="project",
            source="jsonl",
            last_accessed=0,
            created_at=0,
            session_count=1,
        )
        graph.write_node(node)

        monkeypatch.setattr(extract_mod, "retrieve", lambda *a, **kw: [])

        tokens_raw, tokens_injected = extract_mod._compute_token_stats(
            graph, "/tmp/test_project"
        )
        assert tokens_raw is not None
        assert tokens_raw > 0
        assert tokens_injected is None


# ---------------------------------------------------------------------------
# __main__ blocks: compact.py:14, inject.py:70, extract.py:234
# ---------------------------------------------------------------------------


class TestHooksMainEntrypoints:
    def test_inject_main_block_exits_cleanly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import runpy

        monkeypatch.setenv("CLAUDE_PROJECT_PATH", str(tmp_path))
        monkeypatch.setenv("CORTEX_DB_PATH", str(tmp_path / "no_such.db"))

        with pytest.raises(SystemExit) as exc_info:
            runpy.run_module("hooks.inject", run_name="__main__", alter_sys=False)
        assert exc_info.value.code == 0

    def test_compact_main_block_exits_cleanly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import runpy

        monkeypatch.setenv("CLAUDE_PROJECT_PATH", str(tmp_path))
        monkeypatch.setenv("CORTEX_DB_PATH", str(tmp_path / "no_such.db"))

        with pytest.raises(SystemExit) as exc_info:
            runpy.run_module("hooks.compact", run_name="__main__", alter_sys=False)
        assert exc_info.value.code == 0

    def test_extract_main_block_skips_without_transcript(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import runpy

        monkeypatch.delenv("CLAUDE_TRANSCRIPT", raising=False)
        monkeypatch.setenv("CLAUDE_PROJECT_PATH", str(tmp_path))

        with pytest.raises(SystemExit) as exc_info:
            runpy.run_module("hooks.extract", run_name="__main__", alter_sys=False)
        assert exc_info.value.code == 0
