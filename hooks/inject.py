#!/usr/bin/env python3
"""SessionStart hook — retrieve memory context and print it to stdout."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.config import open_graph
from core.graph import Graph
from core.retrieval import TOKEN_BUDGET, format_injection_block, retrieve


def run_inject(project: str, graph: Graph, query: str = "") -> str:
    """Retrieve relevant nodes and return the formatted injection block.

    Nodes that are surfaced get their last_accessed timestamp and weight
    updated so retrieval frequency drives the memory weight signal.

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

    graph.touch_nodes([n.id for n in nodes], now=int(time.time()))

    return format_injection_block(nodes, project, budget_tokens=TOKEN_BUDGET)


def main() -> int:
    """Entry point for the SessionStart hook.

    Reads CLAUDE_PROJECT_PATH (project root), CORTEX_DB_PATH (optional DB
    override), and CLAUDE_INITIAL_MESSAGE (optional query for scoring) from
    the environment.

    Prints the injection block to stdout if there are nodes to inject.
    """
    project = os.environ.get("CLAUDE_PROJECT_PATH", str(Path.cwd()))

    graph = open_graph(Path(project))
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
