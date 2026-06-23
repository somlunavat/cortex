#!/usr/bin/env python3
"""Stop hook — parse session transcript and write memory nodes to graph."""

from __future__ import annotations

import contextlib
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.config import open_graph
from core.decay import run_decay
from core.embedder import embed_batch
from core.extractor import run_extraction
from core.graph import Graph, Node
from core.parser import EventType, detect_co_occurring_writes, parse_transcript
from core.retrieval import TOKEN_BUDGET, count_tokens, format_injection_block, retrieve


def run_extract(transcript_path: Path, project: str, graph: Graph) -> int:
    """Parse a transcript and write extracted nodes to the graph.

    Idempotent: if transcript_path has already been processed (logged in the
    sessions table), this function returns 0 immediately.

    Args:
        transcript_path: Path to the Claude Code JSONL transcript file.
        project: Absolute path to the project root.
        graph: Open Graph instance to write nodes and sessions into.

    Returns:
        Number of new nodes written (not counting merges).
    """
    if not transcript_path.exists():
        return 0

    existing = graph._conn.execute(
        "SELECT id FROM sessions WHERE transcript_path = ?",
        (str(transcript_path),),
    ).fetchone()
    if existing:
        return 0

    events = list(parse_transcript(transcript_path))
    if not events:
        return 0

    session_start = min(e.timestamp for e in events)

    touched_files = list(
        dict.fromkeys(
            Path(e.data["path"])
            for e in events
            if e.type == EventType.FILE_WRITE and "path" in e.data
        )
    )

    candidates = run_extraction(events, project, touched_files, session_start)

    embeddings = embed_batch([c.text for c in candidates])

    nodes_written = 0
    now = int(time.time())

    # file_rel → node_id for wiring co-occurring pairs as edges
    file_node_ids: dict[str, str] = {}

    for candidate, embedding in zip(candidates, embeddings, strict=True):
        similar = graph.find_similar(embedding, project, threshold=0.9, limit=1)

        node = Node(
            id="",
            type=candidate.type.value,
            tier=1,
            text=candidate.text,
            rationale=candidate.rationale,
            embedding=embedding,
            precision_bits=32,
            weight=1.0,
            project=candidate.project,
            scope=candidate.scope.value,
            source=candidate.source.value,
            last_accessed=now,
            created_at=now,
            session_count=1,
        )

        if similar:
            node_id = similar[0].id
            graph.merge_node(node_id, node)
        else:
            node_id = graph.write_node(node)
            nodes_written += 1

        # Track observation nodes that are about a specific file for edge wiring
        if candidate.source.value == "jsonl" and candidate.type.value == "observation":
            # Text is "{rel_path} was modified N times…" — extract the leading path token
            first_word = candidate.text.split(" ")[0]
            if first_word:
                file_node_ids[first_word] = node_id

    # Wire co-occurring file pairs as edges between their memory nodes
    co_pairs = detect_co_occurring_writes(events)
    for pair in co_pairs:
        paths = list(pair)
        if len(paths) != 2:
            continue
        try:
            rel_a = str(Path(paths[0]).relative_to(project))
            rel_b = str(Path(paths[1]).relative_to(project))
        except ValueError:
            continue
        id_a = file_node_ids.get(rel_a)
        id_b = file_node_ids.get(rel_b)
        if id_a and id_b and id_a != id_b:
            with contextlib.suppress(Exception):
                graph.write_edge(id_a, id_b)

    decay_result = run_decay(graph, project)

    tokens_raw, tokens_injected = _compute_token_stats(graph, project)

    graph.write_session(
        session_id=str(uuid.uuid4()),
        project=project,
        started_at=session_start,
        ended_at=now,
        nodes_written=nodes_written,
        nodes_evicted=decay_result.nodes_evicted,
        nodes_promoted=decay_result.nodes_promoted,
        transcript_path=str(transcript_path),
        tokens_raw=tokens_raw,
        tokens_injected=tokens_injected,
    )

    return nodes_written


def _compute_token_stats(graph: Graph, project: str) -> tuple[int | None, int | None]:
    """Compute token savings metrics for the session record.

    tokens_raw = tokens in a full injection block containing ALL project nodes
        with no budget constraint. Represents the cost without Cortex's filtering.
    tokens_injected = tokens in the budget-constrained injection block that would
        be produced at the next session start. Always <= tokens_raw.

    Both values are approximate forward-looking estimates. Either can be None if
    computation fails (e.g., empty graph, tiktoken unavailable).

    Args:
        graph: Open Graph instance for the project.
        project: Absolute project path.

    Returns:
        Tuple of (tokens_raw, tokens_injected), each an int or None.
    """
    try:
        all_nodes = graph.get_all_nodes(project=project)
        if not all_nodes:
            return None, None
        # Raw: format all nodes with a very large budget (no trimming)
        raw_block = format_injection_block(all_nodes, project, budget_tokens=999_999)
        tokens_raw: int = count_tokens(raw_block)
        # Injected: format the budget-constrained subset
        projected = retrieve("", project, graph, budget_tokens=TOKEN_BUDGET)
        if not projected:
            return tokens_raw, None
        injected_block = format_injection_block(
            projected, project, budget_tokens=TOKEN_BUDGET
        )
        tokens_injected: int = count_tokens(injected_block)
        return tokens_raw, tokens_injected
    except Exception:
        return None, None


def main() -> int:
    """Entry point for the Stop hook.

    Reads CLAUDE_TRANSCRIPT (path to JSONL), CLAUDE_PROJECT_PATH (project root),
    and CORTEX_DB_PATH (optional override for the database location) from the
    environment.
    """
    transcript_env = os.environ.get("CLAUDE_TRANSCRIPT", "")
    if not transcript_env:
        print("CORTEX: CLAUDE_TRANSCRIPT not set, skipping extraction", file=sys.stderr)
        return 0

    transcript_path = Path(transcript_env)
    if not transcript_path.exists():
        print(f"CORTEX: transcript not found: {transcript_path}", file=sys.stderr)
        return 0

    project = os.environ.get("CLAUDE_PROJECT_PATH", str(Path.cwd()))

    try:
        graph = open_graph(Path(project), create=True)
        if graph is None:
            print("CORTEX: could not open graph", file=sys.stderr)
            return 1
        n = run_extract(transcript_path, project, graph)
        print(f"CORTEX: extracted {n} nodes", file=sys.stderr)
        return 0
    except Exception as exc:
        print(f"CORTEX: extraction failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
