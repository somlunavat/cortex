#!/usr/bin/env python3
"""Stop hook — parse session transcript and write memory nodes to graph."""

from __future__ import annotations

import os
import sqlite3
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.decay import run_decay
from core.embedder import embed
from core.extractor import run_extraction
from core.graph import Graph, Node
from core.parser import EventType, parse_transcript

_SCHEMA_PATH = Path(__file__).parent.parent / "schema.sql"


def _open_graph(db_path: Path) -> Graph:
    """Open (or create) the cortex SQLite database and apply schema."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA_PATH.read_text())
    return Graph(connection=conn)


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

    touched_files = [
        Path(e.data["path"])
        for e in events
        if e.type == EventType.FILE_WRITE and "path" in e.data
    ]

    candidates = run_extraction(events, project, touched_files, session_start)

    nodes_written = 0
    now = int(time.time())

    for candidate in candidates:
        embedding = embed(candidate.text)
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
            graph.merge_node(similar[0].id, node)
        else:
            graph.write_node(node)
            nodes_written += 1

    decay_result = run_decay(graph, project)

    graph._conn.execute(
        """
        INSERT INTO sessions (
            id, project, started_at, ended_at,
            nodes_written, nodes_evicted, transcript_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            project,
            session_start,
            now,
            nodes_written,
            decay_result.nodes_evicted,
            str(transcript_path),
        ),
    )
    graph._conn.commit()
    return nodes_written


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
    db_path = Path(
        os.environ.get(
            "CORTEX_DB_PATH",
            str(Path(project) / ".cortex" / "cortex.db"),
        )
    )

    try:
        graph = _open_graph(db_path)
        n = run_extract(transcript_path, project, graph)
        print(f"CORTEX: extracted {n} nodes", file=sys.stderr)
        return 0
    except Exception as exc:
        print(f"CORTEX: extraction failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
