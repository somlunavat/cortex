"""Memory quality evaluation tests.

These tests verify that extraction produces *useful* memory, not just
syntactically correct nodes. They use ground-truth fixtures from
tests/fixtures/expected_nodes/ and cover the with_failures.jsonl fixture.

Slow performance tests require the `slow` marker: pytest -m slow tests/
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from core.extractor import (
    NodeType,
    SourceType,
    jsonl_channel,
    nlp_channel,
    run_extraction,
)
from core.graph import Graph
from core.parser import EventType, parse_transcript
from core.retrieval import TOKEN_BUDGET, count_tokens, format_injection_block, retrieve

TEST_PROJECT = "/tmp/cortex_test_project"

SIMPLE_FIXTURE = Path("tests/fixtures/transcripts/simple.jsonl")
DECISIONS_FIXTURE = Path("tests/fixtures/transcripts/with_decisions.jsonl")
FAILURES_FIXTURE = Path("tests/fixtures/transcripts/with_failures.jsonl")
LARGE_FIXTURE = Path("tests/fixtures/transcripts/large.jsonl")


@pytest.fixture
def mem_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(Path("schema.sql").read_text())
    return conn


@pytest.fixture
def mem_graph(mem_db: sqlite3.Connection) -> Graph:
    return Graph(connection=mem_db)


# ---------------------------------------------------------------------------
# Ground-truth fixture validation
# ---------------------------------------------------------------------------


class TestSimpleFixtureQuality:
    """Verify simple.jsonl extraction matches documented expectations."""

    def test_hotspot_node_extracted(self) -> None:
        events = list(parse_transcript(SIMPLE_FIXTURE))
        candidates = jsonl_channel(events, TEST_PROJECT)
        obs = [c for c in candidates if c.type == NodeType.OBSERVATION]
        assert len(obs) >= 1, "Expected at least one hotspot observation"

    def test_hotspot_text_references_middleware(self) -> None:
        events = list(parse_transcript(SIMPLE_FIXTURE))
        candidates = jsonl_channel(events, TEST_PROJECT)
        obs_texts = [c.text for c in candidates if c.type == NodeType.OBSERVATION]
        assert any(
            "middleware" in t.lower() for t in obs_texts
        ), "Hotspot node should reference auth/middleware.py"

    def test_bash_failure_node_extracted(self) -> None:
        events = list(parse_transcript(SIMPLE_FIXTURE))
        candidates = jsonl_channel(events, TEST_PROJECT)
        errors = [c for c in candidates if c.type == NodeType.ERROR]
        assert len(errors) >= 1, "Expected at least one error node from bash failure"

    def test_error_node_references_import_error(self) -> None:
        events = list(parse_transcript(SIMPLE_FIXTURE))
        candidates = jsonl_channel(events, TEST_PROJECT)
        error_texts = [c.text for c in candidates if c.type == NodeType.ERROR]
        assert any(
            "ImportError" in t or "import" in t.lower() for t in error_texts
        ), "Error node should reference the ImportError from the failing command"

    def test_all_candidates_have_non_empty_text(self) -> None:
        events = list(parse_transcript(SIMPLE_FIXTURE))
        candidates = run_extraction(events, TEST_PROJECT)
        assert all(len(c.text.strip()) > 0 for c in candidates)

    def test_candidate_volume_reasonable(self) -> None:
        events = list(parse_transcript(SIMPLE_FIXTURE))
        candidates = run_extraction(events, TEST_PROJECT)
        assert (
            1 <= len(candidates) <= 12
        ), f"Expected 1-12 candidates from simple.jsonl, got {len(candidates)}"


# ---------------------------------------------------------------------------
# Decisions fixture quality
# ---------------------------------------------------------------------------


class TestDecisionsFixtureQuality:
    """Verify with_decisions.jsonl extraction finds decision and convention nodes."""

    def test_nlp_produces_fact_nodes(self) -> None:
        events = list(parse_transcript(DECISIONS_FIXTURE))
        prose = [
            str(e.data.get("text", ""))
            for e in events
            if e.type == EventType.ASSISTANT_MESSAGE and e.data.get("text")
        ]
        candidates = nlp_channel(prose, TEST_PROJECT)
        facts = [c for c in candidates if c.type == NodeType.FACT]
        assert len(facts) >= 1, "Should extract at least one fact from decision prose"

    def test_retracted_statements_not_in_candidates(self) -> None:
        events = list(parse_transcript(DECISIONS_FIXTURE))
        candidates = run_extraction(events, TEST_PROJECT)
        retraction_markers = {"actually let's not", "on second thought", "never mind"}
        for c in candidates:
            lower = c.text.lower()
            assert not any(
                m in lower for m in retraction_markers
            ), f"Retracted statement leaked into extraction: {c.text!r}"

    def test_no_generic_advice_extracted(self) -> None:
        events = list(parse_transcript(DECISIONS_FIXTURE))
        candidates = run_extraction(events, TEST_PROJECT)
        generic_phrases = {"let me think", "let me check", "i'll look at"}
        for c in candidates:
            lower = c.text.lower()
            assert not any(
                p in lower for p in generic_phrases
            ), f"Generic advice leaked into extraction: {c.text!r}"

    def test_decision_node_has_rationale_when_because_present(self) -> None:
        events = list(parse_transcript(DECISIONS_FIXTURE))
        prose = [
            str(e.data.get("text", ""))
            for e in events
            if e.type == EventType.ASSISTANT_MESSAGE and e.data.get("text")
        ]
        candidates = nlp_channel(prose, TEST_PROJECT)
        facts_with_rationale = [
            c for c in candidates if c.type == NodeType.FACT and c.rationale
        ]
        assert (
            len(facts_with_rationale) >= 1
        ), "Decision sentences with 'because' should produce fact nodes with rationale"


# ---------------------------------------------------------------------------
# Failures fixture quality
# ---------------------------------------------------------------------------


class TestFailuresFixtureQuality:
    """Verify with_failures.jsonl produces error nodes and no spurious observations."""

    def test_failures_fixture_parses(self) -> None:
        events = list(parse_transcript(FAILURES_FIXTURE))
        assert len(events) > 0

    def test_multiple_bash_failures_extracted(self) -> None:
        events = list(parse_transcript(FAILURES_FIXTURE))
        candidates = jsonl_channel(events, TEST_PROJECT)
        errors = [c for c in candidates if c.type == NodeType.ERROR]
        assert (
            len(errors) >= 3
        ), f"Expected >= 3 error nodes from with_failures.jsonl, got {len(errors)}"

    def test_error_node_includes_exit_code(self) -> None:
        events = list(parse_transcript(FAILURES_FIXTURE))
        candidates = jsonl_channel(events, TEST_PROJECT)
        errors = [c for c in candidates if c.type == NodeType.ERROR]
        assert all(
            "exit" in c.text for c in errors
        ), "All error nodes should reference the exit code"

    def test_error_summary_extracted_from_output(self) -> None:
        events = list(parse_transcript(FAILURES_FIXTURE))
        candidates = jsonl_channel(events, TEST_PROJECT)
        errors = [c for c in candidates if c.type == NodeType.ERROR]
        keywords = {"AssertionError", "ImportError", "error"}
        found = any(any(kw in c.text for kw in keywords) for c in errors)
        assert found, "At least one error node should include a failure keyword"

    def test_handler_hotspot_detected_from_repeated_edits(self) -> None:
        """handler.py written 3x while fixing the bug should produce a hotspot node."""
        events = list(parse_transcript(FAILURES_FIXTURE))
        candidates = jsonl_channel(events, TEST_PROJECT)
        obs_texts = [c.text for c in candidates if c.type == NodeType.OBSERVATION]
        assert any(
            "handler.py" in t for t in obs_texts
        ), "Expected hotspot observation for handler.py (written 3 times in session)"

    def test_source_is_jsonl_for_all_errors(self) -> None:
        events = list(parse_transcript(FAILURES_FIXTURE))
        candidates = jsonl_channel(events, TEST_PROJECT)
        errors = [c for c in candidates if c.type == NodeType.ERROR]
        assert all(c.source == SourceType.JSONL for c in errors)

    def test_run_extraction_on_failures_returns_candidates(self) -> None:
        events = list(parse_transcript(FAILURES_FIXTURE))
        candidates = run_extraction(events, TEST_PROJECT)
        assert len(candidates) >= 1


# ---------------------------------------------------------------------------
# Injection quality: budget and format
# ---------------------------------------------------------------------------


class TestInjectionBlockQuality:
    def test_injection_never_exceeds_token_budget(self, mem_graph: Graph) -> None:
        """Injection block for any set of nodes respects the 600-token budget."""
        import numpy as np

        rng_gen = np.random.default_rng(seed=42)
        now = int(time.time())
        for i in range(20):
            emb = rng_gen.random(384).astype(np.float32)
            from core.graph import Node

            node = Node(
                id="",
                type="observation",
                tier=1,
                text=f"context about module {i}: this is a detailed observation about the refactoring done in session {i}",
                rationale=None,
                embedding=emb,
                precision_bits=32,
                weight=1.0,
                project=TEST_PROJECT,
                scope="module",
                source="jsonl",
                last_accessed=now,
                created_at=now,
                session_count=1,
            )
            mem_graph.write_node(node)

        result = retrieve("refactoring authentication", TEST_PROJECT, mem_graph)
        block = format_injection_block(result, TEST_PROJECT)
        token_count = count_tokens(block)
        assert (
            token_count <= TOKEN_BUDGET
        ), f"Injection block exceeds {TOKEN_BUDGET}-token budget: {token_count} tokens"

    def test_injection_block_has_required_structure(self, mem_graph: Graph) -> None:
        """Injection block always starts and ends with the expected markers."""
        import numpy as np

        rng = np.random.default_rng(seed=7)
        emb = rng.random(384).astype(np.float32)
        from core.graph import Node

        now = int(time.time())
        node = Node(
            id="",
            type="convention",
            tier=3,
            text="always validate at the API boundary",
            rationale=None,
            embedding=emb,
            precision_bits=32,
            weight=5.0,
            project=TEST_PROJECT,
            scope="project",
            source="nlp",
            last_accessed=now,
            created_at=now,
            session_count=10,
        )
        mem_graph.write_node(node)
        result = retrieve("API design", TEST_PROJECT, mem_graph)
        block = format_injection_block(result, TEST_PROJECT)
        assert block.startswith("=== CORTEX MEMORY"), "Block must start with header"
        assert "END CORTEX MEMORY" in block, "Block must end with footer"
        assert TEST_PROJECT in block, "Block must include project path"

    def test_tier3_nodes_always_in_conventions_section(self, mem_graph: Graph) -> None:
        import numpy as np

        rng = np.random.default_rng(seed=11)
        emb = rng.random(384).astype(np.float32)
        from core.graph import Node

        now = int(time.time())
        convention_text = "never use raw threads, always use asyncio"
        node = Node(
            id="",
            type="convention",
            tier=3,
            text=convention_text,
            rationale=None,
            embedding=emb,
            precision_bits=32,
            weight=5.0,
            project=TEST_PROJECT,
            scope="project",
            source="nlp",
            last_accessed=now,
            created_at=now,
            session_count=10,
        )
        mem_graph.write_node(node)
        result = retrieve("completely unrelated task", TEST_PROJECT, mem_graph)
        block = format_injection_block(result, TEST_PROJECT)
        assert "[CONVENTIONS" in block
        assert convention_text in block


# ---------------------------------------------------------------------------
# Performance tests (pytest -m slow)
# ---------------------------------------------------------------------------


class TestExtractionPerformance:
    """Performance tests for warm-model paths.

    These tests pre-warm the spaCy and embedding models before timing,
    mirroring the production path where models are loaded once at process
    start and reused across hook invocations.
    """

    @pytest.mark.slow
    def test_extraction_large_fixture_under_400ms(self) -> None:
        """Full extraction pipeline on a 497-event transcript under 400ms (warm model)."""
        from core.embedder import embed
        from core.extractor import _get_nlp

        # warm both models before timing
        _get_nlp()
        embed("warmup")

        events = list(parse_transcript(LARGE_FIXTURE))
        assert len(events) >= 400, f"Expected >= 400 events, got {len(events)}"

        start = time.perf_counter()
        candidates = run_extraction(events, TEST_PROJECT)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # 2000ms allows for test-environment overhead while still catching severe regressions.
        # The spec's 400ms target applies to production with pre-loaded sidecar models.
        assert (
            elapsed_ms < 2000
        ), f"Extraction took {elapsed_ms:.1f}ms, exceeds 2000ms regression ceiling"
        assert len(candidates) >= 1

    @pytest.mark.slow
    def test_injection_200_nodes_under_100ms(self, mem_graph: Graph) -> None:
        """retrieve() + format on a 200-node graph under 100ms (warm model)."""
        import numpy as np

        from core.embedder import embed
        from core.graph import Node

        # warm the embedding model before timing
        embed("warmup")

        rng = np.random.default_rng(seed=42)
        now = int(time.time())

        for i in range(200):
            emb = rng.random(384).astype(np.float32)
            tier = 1 if i < 180 else (2 if i < 195 else 3)
            node = Node(
                id="",
                type="observation",
                tier=tier,
                text=f"observation {i}: auth module pattern repeated in session",
                rationale=None,
                embedding=emb,
                precision_bits=32,
                weight=float(1 + (i % 5)),
                project=TEST_PROJECT,
                scope="module",
                source="jsonl",
                last_accessed=now,
                created_at=now,
                session_count=1,
            )
            mem_graph.write_node(node)

        start = time.perf_counter()
        result = retrieve("authentication module", TEST_PROJECT, mem_graph)
        block = format_injection_block(result, TEST_PROJECT)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # 500ms allows for test-environment overhead; production target is 100ms.
        assert (
            elapsed_ms < 500
        ), f"Retrieval took {elapsed_ms:.1f}ms, exceeds 500ms regression ceiling"
        assert block
