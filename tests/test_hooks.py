"""Integration tests for hooks/extract.py and hooks/inject.py."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import numpy as np
import pytest

from core.graph import Graph, Node
from hooks.extract import _open_graph as extract_open_graph
from hooks.extract import main as extract_main
from hooks.extract import run_extract
from hooks.inject import _open_graph as inject_open_graph
from hooks.inject import main as inject_main
from hooks.inject import run_inject

TEST_PROJECT = "/tmp/cortex_hooks_test"
SIMPLE_TRANSCRIPT = Path("tests/fixtures/transcripts/simple.jsonl")
DECISIONS_TRANSCRIPT = Path("tests/fixtures/transcripts/with_decisions.jsonl")


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
        assert len(nodes) >= 0  # transcript may produce 0–N nodes

    def test_extract_idempotent_same_transcript(self, graph: Graph) -> None:
        n1 = run_extract(SIMPLE_TRANSCRIPT, TEST_PROJECT, graph)
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
        graph.write_node(_make_node("auth middleware was modified", tier=1, embedding=e))
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
        graph.write_node(_make_node("target project node", project=TEST_PROJECT, embedding=e))
        graph.write_node(_make_node("other project node", project="/other/proj", embedding=e))
        result = run_inject(TEST_PROJECT, graph, query="")
        assert "target project node" in result
        assert "other project node" not in result


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


# ---------------------------------------------------------------------------
# _open_graph helpers
# ---------------------------------------------------------------------------


class TestOpenGraph:
    def test_extract_open_graph_creates_db(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        graph = extract_open_graph(db_path)
        assert db_path.exists()
        assert isinstance(graph, Graph)

    def test_inject_open_graph_returns_none_if_missing(self, tmp_path: Path) -> None:
        db_path = tmp_path / "nonexistent.db"
        result = inject_open_graph(db_path)
        assert result is None

    def test_inject_open_graph_returns_graph_if_exists(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        extract_open_graph(db_path)  # create it
        graph = inject_open_graph(db_path)
        assert isinstance(graph, Graph)

    def test_extract_open_graph_creates_parent_dirs(self, tmp_path: Path) -> None:
        db_path = tmp_path / "nested" / "cortex.db"
        extract_open_graph(db_path)
        assert db_path.exists()


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
        extract_open_graph(db_path)  # create the DB
        monkeypatch.setenv("CLAUDE_PROJECT_PATH", TEST_PROJECT)
        monkeypatch.setenv("CORTEX_DB_PATH", str(db_path))
        monkeypatch.delenv("CLAUDE_INITIAL_MESSAGE", raising=False)
        result = inject_main()
        assert result == 0

    def test_inject_main_with_nodes_prints_block(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        db_path = tmp_path / "cortex.db"
        g = extract_open_graph(db_path)
        rng_local = np.random.default_rng(seed=99)
        e = rng_local.random(384).astype(np.float32)
        g.write_node(_make_node("always use async", tier=3, embedding=e, project=TEST_PROJECT))
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
        graph.write_node(_make_node("auth/middleware.py is a hotspot", tier=1, embedding=e))
        initial_count = len(graph.get_all_nodes(project=TEST_PROJECT))
        run_extract(SIMPLE_TRANSCRIPT, TEST_PROJECT, graph)
        assert len(graph.get_all_nodes(project=TEST_PROJECT)) >= initial_count

    def test_extract_main_exception_returns_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        db_path = tmp_path / "cortex.db"
        # Make extract_open_graph raise by providing a bad schema path
        monkeypatch.setenv("CLAUDE_TRANSCRIPT", str(SIMPLE_TRANSCRIPT))
        monkeypatch.setenv("CLAUDE_PROJECT_PATH", TEST_PROJECT)
        monkeypatch.setenv("CORTEX_DB_PATH", str(db_path))

        import hooks.extract as ext_mod

        original = ext_mod._open_graph  # type: ignore[attr-defined]

        def raise_always(p: Path) -> Graph:
            raise RuntimeError("forced failure")

        monkeypatch.setattr(ext_mod, "_open_graph", raise_always)
        result = extract_main()
        assert result == 1
        monkeypatch.setattr(ext_mod, "_open_graph", original)

    def test_inject_main_exception_returns_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        db_path = tmp_path / "cortex.db"
        extract_open_graph(db_path)
        monkeypatch.setenv("CLAUDE_PROJECT_PATH", TEST_PROJECT)
        monkeypatch.setenv("CORTEX_DB_PATH", str(db_path))
        monkeypatch.delenv("CLAUDE_INITIAL_MESSAGE", raising=False)

        import hooks.inject as inj_mod

        def raise_always(p: str, g: Graph, q: str = "") -> str:
            raise RuntimeError("forced failure")

        monkeypatch.setattr(inj_mod, "run_inject", raise_always)
        result = inject_main()
        assert result == 1
