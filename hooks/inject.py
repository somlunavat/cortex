#!/usr/bin/env python3
"""SessionStart hook — retrieve memory context and print it to stdout."""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.graph import Graph
from core.retrieval import format_injection_block, retrieve

_SCHEMA_PATH = Path(__file__).parent.parent / "schema.sql"


def _open_graph(db_path: Path) -> Graph | None:
    """Open an existing cortex database, or return None if not yet created.

    Args:
        db_path: Path to cortex.db.

    Returns:
        Open Graph instance, or None if the database file does not exist.
    """
    if not db_path.exists():
        return None
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA_PATH.read_text())
    return Graph(connection=conn)


def run_inject(project: str, graph: Graph, query: str = "") -> str:
    """Retrieve relevant nodes and return the formatted injection block.

    Args:
        project: Absolute project path used to scope retrieval.
        graph: Open Graph instance to query.
        query: Optional task description for scoring relevance. Empty string
               is safe — tier-3 nodes are always included regardless.

    Returns:
        Formatted multi-line injection block string (empty string if no nodes).
    """
    nodes = retrieve(query, project, graph)
    if not nodes:
        return ""
    return format_injection_block(nodes, project)


def main() -> int:
    """Entry point for the SessionStart hook.

    Reads CLAUDE_PROJECT_PATH (project root), CORTEX_DB_PATH (optional DB
    override), and CLAUDE_INITIAL_MESSAGE (optional query for scoring) from
    the environment.

    Prints the injection block to stdout if there are nodes to inject.
    """
    project = os.environ.get("CLAUDE_PROJECT_PATH", str(Path.cwd()))
    db_path = Path(
        os.environ.get(
            "CORTEX_DB_PATH",
            str(Path(project) / ".cortex" / "cortex.db"),
        )
    )

    graph = _open_graph(db_path)
    if graph is None:
        return 0

    query = os.environ.get("CLAUDE_INITIAL_MESSAGE", "")

    try:
        block = run_inject(project, graph, query)
        if block:
            print(block)
        return 0
    except Exception as exc:
        print(f"CORTEX: injection failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
