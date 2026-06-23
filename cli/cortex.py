"""Cortex CLI entry point.

Commands:
    status    — node counts by tier, token savings, last session
    graph     — ASCII adjacency summary
    inspect   — full metadata for a single node
    prune     — manually evict a node by id
    reset     — wipe all nodes for a project
    search    — BM25 text search across memory nodes
    list      — paginated table of all memory nodes with filters
    decay     — run decay/eviction/promotion manually
    install   — write plugin.json to Claude Code plugins directory
    dashboard — start the dashboard server on port 7000
    doctor    — verify that all Cortex dependencies are healthy
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.config import open_graph
from core.decay import run_decay

app = typer.Typer(
    name="cortex",
    help="Cortex — persistent memory layer for Claude Code.",
    add_completion=False,
)
console = Console()


def _project_root() -> Path:
    """Resolve project root from CWD."""
    return Path.cwd()


def _fmt_ts(ts: int | None) -> str:
    """Format a unix timestamp as a human-readable local datetime string."""
    if not ts:
        return "—"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def status(
    project_path: str = typer.Option("", "--project", help="Project path (defaults to CWD)"),
) -> None:
    """Show node counts by tier, type distribution, source breakdown, and last session."""
    root = Path(project_path) if project_path else _project_root()
    g = open_graph(root)

    if g is None:
        console.print("[yellow]No Cortex database found in this project.[/yellow]")
        raise typer.Exit(0)

    project = str(root)

    # Tier table with avg weight
    tier_rows = g._conn.execute(
        "SELECT tier, COUNT(*) AS cnt, AVG(weight) AS avg_w "
        "FROM nodes WHERE project = ? GROUP BY tier",
        (project,),
    ).fetchall()
    tier_map = {r["tier"]: (r["cnt"], r["avg_w"] or 0.0) for r in tier_rows}
    tier_labels = {1: "Ephemeral", 2: "Semantic", 3: "Procedural"}

    tier_table = Table(title=f"Cortex — {project}")
    tier_table.add_column("Tier", style="cyan")
    tier_table.add_column("Label")
    tier_table.add_column("Nodes", justify="right")
    tier_table.add_column("Avg weight", justify="right")
    for t in (1, 2, 3):
        cnt, avg_w = tier_map.get(t, (0, 0.0))
        tier_table.add_row(str(t), tier_labels[t], str(cnt), f"{avg_w:.2f}")
    console.print(tier_table)

    # Type distribution
    type_rows = g._conn.execute(
        "SELECT type, COUNT(*) AS cnt FROM nodes WHERE project = ? GROUP BY type ORDER BY cnt DESC",
        (project,),
    ).fetchall()
    if type_rows:
        type_table = Table(title="Node types")
        type_table.add_column("Type", style="dim")
        type_table.add_column("Count", justify="right")
        for r in type_rows:
            type_table.add_row(r["type"], str(r["cnt"]))
        console.print(type_table)

    # Source breakdown
    src_rows = g._conn.execute(
        "SELECT source, COUNT(*) AS cnt FROM nodes WHERE project = ? GROUP BY source ORDER BY cnt DESC",
        (project,),
    ).fetchall()
    if src_rows:
        src_table = Table(title="Extraction sources")
        src_table.add_column("Source", style="dim")
        src_table.add_column("Nodes", justify="right")
        for r in src_rows:
            src_table.add_row(r["source"], str(r["cnt"]))
        console.print(src_table)

    row = g._conn.execute(
        "SELECT ended_at, nodes_written, nodes_evicted, nodes_promoted, "
        "tokens_raw, tokens_injected "
        "FROM sessions WHERE project = ? ORDER BY ended_at DESC LIMIT 1",
        (project,),
    ).fetchone()

    if row:
        console.print(f"\nLast session: [green]{_fmt_ts(row['ended_at'])}[/green]")
        console.print(f"  Nodes written:   {row['nodes_written']}")
        console.print(f"  Nodes evicted:   {row['nodes_evicted']}")
        console.print(f"  Nodes promoted:  {row['nodes_promoted'] or 0}")
        if row["tokens_raw"] and row["tokens_injected"]:
            saved = row["tokens_raw"] - row["tokens_injected"]
            console.print(f"  Tokens saved:    {saved}")


@app.command()
def recent(
    limit: int = typer.Option(10, "--limit", "-n", help="Number of nodes to show"),
    project_path: str = typer.Option("", "--project", help="Project path (defaults to CWD)"),
) -> None:
    """Show the most recently accessed memory nodes.

    Useful after a session to see what Cortex surfaced to Claude.
    """
    root = Path(project_path) if project_path else _project_root()
    g = open_graph(root)

    if g is None:
        console.print("[yellow]No Cortex database found.[/yellow]")
        raise typer.Exit(0)

    project = str(root)
    rows = g._conn.execute(
        "SELECT id, type, tier, text, weight, source, last_accessed "
        "FROM nodes WHERE project = ? ORDER BY last_accessed DESC LIMIT ?",
        (project, limit),
    ).fetchall()

    if not rows:
        console.print("[dim]No nodes found.[/dim]")
        raise typer.Exit(0)

    table = Table(title=f"Recently accessed — {project}")
    table.add_column("Accessed", style="cyan", no_wrap=True)
    table.add_column("T", justify="center", width=3)
    table.add_column("Type", style="dim", width=12)
    table.add_column("Src", style="dim", width=5)
    table.add_column("Weight", justify="right", width=7)
    table.add_column("Text")
    table.add_column("ID", style="dim", width=10)

    for row in rows:
        table.add_row(
            _fmt_ts(row["last_accessed"]),
            str(row["tier"]),
            row["type"],
            row["source"],
            f"{row['weight']:.2f}",
            row["text"][:60],
            row["id"][:8] + "…",
        )

    console.print(table)


@app.command()
def graph() -> None:
    """Print an ASCII adjacency summary of the knowledge graph."""
    root = _project_root()
    g = open_graph(root)

    if g is None:
        console.print("[yellow]No Cortex database found.[/yellow]")
        raise typer.Exit(0)

    project = str(root)
    nodes = g.get_all_nodes(project=project)

    if not nodes:
        console.print("Graph is empty.")
        raise typer.Exit(0)

    node_map = {n.id: n for n in nodes}

    for node in nodes[:20]:
        edges = g.get_edges(node.id)
        neighbors = []
        for edge in edges:
            neighbor_id = (
                edge.target_id if edge.source_id == node.id else edge.source_id
            )
            neighbor = node_map.get(neighbor_id)
            if neighbor:
                neighbors.append(neighbor.text[:30])

        tier_marker = f"T{node.tier}"
        label = node.text[:50]
        neighbor_str = " → " + ", ".join(neighbors[:3]) if neighbors else ""
        console.print(f"[{tier_marker}] {label}{neighbor_str}")

    if len(nodes) > 20:
        console.print(f"  ... and {len(nodes) - 20} more nodes")


@app.command()
def inspect(node_id: str = typer.Argument(..., help="Node UUID to inspect")) -> None:
    """Display full metadata for a single node."""
    root = _project_root()
    g = open_graph(root)

    if g is None:
        console.print("[yellow]No Cortex database found.[/yellow]")
        raise typer.Exit(1)

    target = g.get_node(node_id)

    if target is None:
        console.print(f"[red]Node not found:[/red] {node_id}")
        raise typer.Exit(1)

    table = Table(title=f"Node {node_id[:8]}…")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("id", target.id)
    table.add_row("type", target.type)
    table.add_row("tier", str(target.tier))
    table.add_row("text", target.text)
    table.add_row("rationale", target.rationale or "—")
    table.add_row("weight", f"{target.weight:.4f}")
    table.add_row("session_count", str(target.session_count))
    table.add_row("precision_bits", str(target.precision_bits))
    table.add_row("scope", target.scope)
    table.add_row("source", target.source)
    table.add_row("project", target.project)
    table.add_row("last_accessed", _fmt_ts(target.last_accessed))
    table.add_row("created_at", _fmt_ts(target.created_at))
    console.print(table)


@app.command()
def prune(node_id: str = typer.Argument(..., help="Node UUID to evict")) -> None:
    """Manually evict a node from the graph."""
    root = _project_root()
    g = open_graph(root)

    if g is None:
        console.print("[yellow]No Cortex database found.[/yellow]")
        raise typer.Exit(1)

    target = g.get_node(node_id)

    if target is None:
        console.print(f"[red]Node not found:[/red] {node_id}")
        raise typer.Exit(1)

    g.delete_node(node_id)
    console.print(f"[green]Pruned node:[/green] {node_id[:8]}… ({target.text[:60]})")


@app.command()
def reset(
    project_path: str = typer.Option(
        "", "--project", help="Project path to reset (defaults to CWD)"
    ),
    confirm: bool = typer.Option(False, "--yes", help="Skip confirmation prompt"),
) -> None:
    """Wipe all nodes for a project. Irreversible."""
    root = Path(project_path) if project_path else _project_root()
    g = open_graph(root)

    if g is None:
        console.print("[yellow]No Cortex database found.[/yellow]")
        raise typer.Exit(0)

    project = str(root)
    count = len(g.get_all_nodes(project=project))

    if not confirm:
        typer.confirm(
            f"Delete all {count} nodes for project {project}?",
            abort=True,
        )

    deleted = g.delete_all_nodes(project)
    console.print(
        f"[green]Reset complete:[/green] removed {deleted} nodes for {project}"
    )


@app.command()
def search(
    query: str = typer.Argument(..., help="Text to search for in memory nodes"),
    tier: int = typer.Option(0, "--tier", "-t", help="Filter by tier (0 = all tiers)"),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum results to show"),
    source: str = typer.Option(
        "", "--source", "-s",
        help="Filter by extraction source: jsonl, ast, git, or nlp",
    ),
) -> None:
    """Search memory nodes by text using BM25 ranking."""
    root = _project_root()
    g = open_graph(root)

    if g is None:
        console.print("[yellow]No Cortex database found.[/yellow]")
        raise typer.Exit(0)

    project = str(root)
    valid_sources = {"jsonl", "ast", "git", "nlp"}
    if source and source not in valid_sources:
        console.print(f"[red]Invalid source '{source}'. Choose from: {', '.join(sorted(valid_sources))}[/red]")
        raise typer.Exit(1)

    if source:
        nodes = g.get_nodes_by_source(
            project=project, source=source, tier=tier if tier else None
        )
    else:
        nodes = g.get_all_nodes(project=project, tier=tier if tier else None)

    if not nodes:
        console.print("No nodes found.")
        raise typer.Exit(0)

    from rank_bm25 import BM25Okapi

    tokenized = [n.text.lower().split() for n in nodes]
    index = BM25Okapi(tokenized)
    tokens = query.lower().split()
    raw_scores = index.get_scores(tokens)

    scored = sorted(
        zip(raw_scores, nodes, strict=True),
        key=lambda x: x[0],
        reverse=True,
    )

    table = Table(title=f'Search: "{query}"')
    table.add_column("Score", style="yellow", justify="right")
    table.add_column("T", style="cyan", justify="center")
    table.add_column("Type", style="dim")
    table.add_column("Text")
    table.add_column("ID", style="dim")

    shown = 0
    for score, node in scored:
        if score <= 0.0:
            break
        if shown >= limit:
            break
        table.add_row(
            f"{score:.3f}",
            str(node.tier),
            node.type,
            node.text[:80],
            node.id[:8] + "…",
        )
        shown += 1

    if shown == 0:
        console.print("[dim]No matching nodes.[/dim]")
    else:
        console.print(table)


@app.command()
def decay(
    project_path: str = typer.Option(
        "", "--project", help="Project path (defaults to CWD)"
    ),
) -> None:
    """Run weight decay, eviction, and tier promotion for this project."""
    root = Path(project_path) if project_path else _project_root()
    g = open_graph(root)

    if g is None:
        console.print("[yellow]No Cortex database found.[/yellow]")
        raise typer.Exit(0)

    project = str(root)
    result = run_decay(g, project)

    console.print(f"[green]Decay complete[/green] for {project}")
    console.print(f"  Decayed:   {result.nodes_decayed}")
    console.print(f"  Evicted:   {result.nodes_evicted}")
    console.print(f"  Promoted:  {result.nodes_promoted}")


@app.command()
def export(
    output: str = typer.Option(
        "", "--out", "-o", help="Output file path (default: stdout)"
    ),
    project_path: str = typer.Option(
        "", "--project", help="Project path (defaults to CWD)"
    ),
    tier: int = typer.Option(0, "--tier", "-t", help="Filter by tier (0 = all tiers)"),
    fmt: str = typer.Option("json", "--format", "-f", help="Output format: json or csv"),
) -> None:
    """Export memory nodes to JSON or CSV for backup or inspection."""
    root = Path(project_path) if project_path else _project_root()
    g = open_graph(root)

    if g is None:
        console.print("[yellow]No Cortex database found.[/yellow]")
        raise typer.Exit(0)

    project = str(root)
    nodes = g.get_all_nodes(project=project, tier=tier if tier else None)

    if not nodes:
        console.print("[dim]No nodes to export.[/dim]")
        raise typer.Exit(0)

    if fmt == "csv":
        import csv
        import io

        buf = io.StringIO()
        writer = csv.DictWriter(
            buf,
            fieldnames=[
                "id", "type", "tier", "text", "rationale", "weight",
                "session_count", "precision_bits", "scope", "source",
                "last_accessed", "created_at",
            ],
        )
        writer.writeheader()
        for node in nodes:
            writer.writerow(
                {
                    "id": node.id,
                    "type": node.type,
                    "tier": node.tier,
                    "text": node.text,
                    "rationale": node.rationale or "",
                    "weight": f"{node.weight:.6f}",
                    "session_count": node.session_count,
                    "precision_bits": node.precision_bits,
                    "scope": node.scope,
                    "source": node.source,
                    "last_accessed": node.last_accessed,
                    "created_at": node.created_at,
                }
            )
        content = buf.getvalue()
    else:
        rows = [
            {
                "id": node.id,
                "type": node.type,
                "tier": node.tier,
                "text": node.text,
                "rationale": node.rationale,
                "weight": node.weight,
                "session_count": node.session_count,
                "precision_bits": node.precision_bits,
                "scope": node.scope,
                "source": node.source,
                "project": node.project,
                "last_accessed": node.last_accessed,
                "created_at": node.created_at,
            }
            for node in nodes
        ]
        content = json.dumps(rows, indent=2)

    if output:
        Path(output).write_text(content)
        console.print(
            f"[green]Exported {len(nodes)} nodes to:[/green] {output}"
        )
    else:
        print(content)


@app.command()
def sessions(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of sessions to show"),
    project_path: str = typer.Option(
        "", "--project", help="Project path (defaults to CWD)"
    ),
) -> None:
    """List recent extraction sessions with node and token stats."""
    root = Path(project_path) if project_path else _project_root()
    g = open_graph(root)

    if g is None:
        console.print("[yellow]No Cortex database found.[/yellow]")
        raise typer.Exit(0)

    rows = g._conn.execute(
        """
        SELECT id, started_at, ended_at, nodes_written, nodes_evicted,
               nodes_promoted, tokens_raw, tokens_injected, transcript_path
        FROM sessions
        WHERE project = ?
        ORDER BY ended_at DESC
        LIMIT ?
        """,
        (str(root), limit),
    ).fetchall()

    if not rows:
        console.print("[dim]No sessions recorded for this project.[/dim]")
        raise typer.Exit(0)

    table = Table(title=f"Sessions — {str(root)}")
    table.add_column("Ended", style="cyan", no_wrap=True)
    table.add_column("Written", justify="right")
    table.add_column("Evicted", justify="right")
    table.add_column("Promoted", justify="right")
    table.add_column("Tokens saved", justify="right")
    table.add_column("ID", style="dim")

    for row in rows:
        saved = ""
        if row["tokens_raw"] is not None and row["tokens_injected"] is not None:
            saved = str(row["tokens_raw"] - row["tokens_injected"])
        table.add_row(
            _fmt_ts(row["ended_at"]),
            str(row["nodes_written"] or 0),
            str(row["nodes_evicted"] or 0),
            str(row["nodes_promoted"] or 0),
            saved or "—",
            row["id"][:8] + "…",
        )

    console.print(table)


@app.command()
def stats(
    project_path: str = typer.Option(
        "", "--project", help="Project path (defaults to CWD)"
    ),
) -> None:
    """Show aggregate memory statistics across all sessions."""
    root = Path(project_path) if project_path else _project_root()
    g = open_graph(root)

    if g is None:
        console.print("[yellow]No Cortex database found.[/yellow]")
        raise typer.Exit(0)

    project = str(root)

    agg = g._conn.execute(
        """
        SELECT
            COUNT(*)              AS session_count,
            SUM(nodes_written)    AS total_written,
            SUM(nodes_evicted)    AS total_evicted,
            SUM(nodes_promoted)   AS total_promoted,
            SUM(CASE WHEN tokens_raw IS NOT NULL AND tokens_injected IS NOT NULL
                     THEN tokens_raw - tokens_injected ELSE 0 END) AS total_saved,
            MIN(started_at)       AS first_session,
            MAX(ended_at)         AS last_session
        FROM sessions
        WHERE project = ?
        """,
        (project,),
    ).fetchone()

    tier_counts = dict(
        g._conn.execute(
            "SELECT tier, COUNT(*) FROM nodes WHERE project = ? GROUP BY tier",
            (project,),
        ).fetchall()
    )

    node_total = sum(tier_counts.values())

    table = Table(title=f"Aggregate stats — {project}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Sessions recorded", str(agg["session_count"] or 0))
    table.add_row("First session", _fmt_ts(agg["first_session"]))
    table.add_row("Last session", _fmt_ts(agg["last_session"]))
    table.add_row("Nodes total", str(node_total))
    table.add_row("  Tier 1 (ephemeral)", str(tier_counts.get(1, 0)))
    table.add_row("  Tier 2 (semantic)", str(tier_counts.get(2, 0)))
    table.add_row("  Tier 3 (procedural)", str(tier_counts.get(3, 0)))
    table.add_row("Nodes written (all time)", str(agg["total_written"] or 0))
    table.add_row("Nodes evicted (all time)", str(agg["total_evicted"] or 0))
    table.add_row("Nodes promoted (all time)", str(agg["total_promoted"] or 0))
    table.add_row("Total tokens saved", str(agg["total_saved"] or 0))

    console.print(table)


@app.command()
def import_(
    input_file: str = typer.Argument(..., help="Path to JSON file from cortex export"),
    project_path: str = typer.Option(
        "", "--project", help="Project path (defaults to CWD)"
    ),
    skip_duplicates: bool = typer.Option(
        True, "--skip-duplicates/--allow-duplicates",
        help="Skip nodes whose text already exists in the graph (default: skip)",
    ),
) -> None:
    """Import memory nodes from a cortex export JSON file.

    Reads the JSON produced by 'cortex export' and writes each node into the
    graph. Embeddings are recomputed from node text since the export format
    omits binary blobs. Nodes whose text is already present are skipped by
    default to avoid duplicates.
    """
    root = Path(project_path) if project_path else _project_root()
    g = open_graph(root, create=True)

    if g is None:
        console.print("[red]Could not open or create Cortex database.[/red]")
        raise typer.Exit(1)

    src = Path(input_file)
    if not src.exists():
        console.print(f"[red]File not found:[/red] {src}")
        raise typer.Exit(1)

    try:
        raw = src.read_text(encoding="utf-8")
        records: list[dict] = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        console.print(f"[red]Invalid JSON:[/red] {exc}")
        raise typer.Exit(1)

    if not isinstance(records, list):
        console.print("[red]Expected a JSON array at the top level.[/red]")
        raise typer.Exit(1)

    project = str(root)
    existing_texts: set[str] = set()
    if skip_duplicates:
        nodes = g.get_all_nodes(project=project)
        existing_texts = {n.text for n in nodes}

    from core.embedder import embed
    from core.graph import Node
    import time as _time

    imported = 0
    skipped = 0
    now = int(_time.time())

    for rec in records:
        if not isinstance(rec, dict):
            skipped += 1
            continue

        text = str(rec.get("text", "")).strip()
        if not text:
            skipped += 1
            continue

        if skip_duplicates and text in existing_texts:
            skipped += 1
            continue

        tier = int(rec.get("tier", 1))
        if tier not in (1, 2, 3):
            tier = 1

        node_type = str(rec.get("type", "observation"))
        if node_type not in ("observation", "fact", "convention", "error"):
            node_type = "observation"

        scope = str(rec.get("scope", "project"))
        if scope not in ("project", "module", "session"):
            scope = "project"

        source = str(rec.get("source", "jsonl"))
        if source not in ("jsonl", "ast", "git", "nlp"):
            source = "jsonl"

        embedding = embed(text)
        node = Node(
            id="",
            type=node_type,
            tier=tier,
            text=text,
            rationale=rec.get("rationale"),
            embedding=embedding,
            precision_bits=32,
            weight=float(rec.get("weight", 1.0)),
            project=project,
            scope=scope,
            source=source,
            last_accessed=now,
            created_at=now,
            session_count=int(rec.get("session_count", 1)),
        )
        g.write_node(node)
        existing_texts.add(text)
        imported += 1

    console.print(
        f"[green]Imported {imported} nodes[/green] "
        f"({skipped} skipped) into {project}"
    )


@app.command()
def pin(
    node_id: str = typer.Argument(..., help="Node UUID to pin to tier 3"),
    project_path: str = typer.Option("", "--project", help="Project path (defaults to CWD)"),
) -> None:
    """Permanently pin a node to tier 3 (procedural memory).

    Tier-3 nodes are never decayed or evicted by the automatic scheduler.
    Embedding precision is downcast to 2 bits on promotion.
    """
    root = Path(project_path) if project_path else _project_root()
    g = open_graph(root)

    if g is None:
        console.print("[yellow]No Cortex database found.[/yellow]")
        raise typer.Exit(1)

    target = g.get_node(node_id)
    if target is None:
        console.print(f"[red]Node not found:[/red] {node_id}")
        raise typer.Exit(1)

    if target.tier == 3:
        console.print(f"[dim]Node {node_id[:8]}… is already tier 3.[/dim]")
        raise typer.Exit(0)

    import time as _time
    import numpy as np
    from core.embedder import serialize

    now = int(_time.time())
    new_blob = None
    if target.embedding is not None:
        new_blob = serialize(target.embedding.astype(np.float32), 2)

    g.update_node_tier(
        node_id=node_id,
        new_tier=3,
        new_precision=2,
        embedding_blob=new_blob,
        now=now,
    )
    console.print(
        f"[green]Pinned to tier 3:[/green] {node_id[:8]}… — {target.text[:60]}"
    )


@app.command()
def annotate(
    node_id: str = typer.Argument(..., help="Node UUID to annotate"),
    rationale: str = typer.Option(..., "--rationale", "-r", help="Rationale text to set"),
    project_path: str = typer.Option("", "--project", help="Project path (defaults to CWD)"),
) -> None:
    """Set or update the rationale for a node.

    The rationale is injected alongside the node text so Claude sees the
    'why' behind a convention or decision.
    """
    root = Path(project_path) if project_path else _project_root()
    g = open_graph(root)

    if g is None:
        console.print("[yellow]No Cortex database found.[/yellow]")
        raise typer.Exit(1)

    target = g.get_node(node_id)
    if target is None:
        console.print(f"[red]Node not found:[/red] {node_id}")
        raise typer.Exit(1)

    updated = g.update_node_rationale(node_id, rationale.strip() or None)
    if updated:
        console.print(f"[green]Rationale updated for[/green] {node_id[:8]}…")
        console.print(f"  [dim]{rationale[:120]}[/dim]")
    else:
        console.print(f"[red]Failed to update node:[/red] {node_id}")
        raise typer.Exit(1)


@app.command()
def bump(
    node_id: str = typer.Argument(..., help="Node UUID to bump"),
    amount: float = typer.Option(1.0, "--by", help="Amount to add to the node weight"),
    project_path: str = typer.Option("", "--project", help="Project path (defaults to CWD)"),
) -> None:
    """Increase a node's weight to reinforce its retrieval priority.

    Useful for manually signalling that a node is important without waiting
    for automatic access-based weighting.
    """
    root = Path(project_path) if project_path else _project_root()
    g = open_graph(root)

    if g is None:
        console.print("[yellow]No Cortex database found.[/yellow]")
        raise typer.Exit(1)

    target = g.get_node(node_id)
    if target is None:
        console.print(f"[red]Node not found:[/red] {node_id}")
        raise typer.Exit(1)

    if amount <= 0:
        console.print("[red]--by must be a positive number.[/red]")
        raise typer.Exit(1)

    g.update_weight(node_id, delta=amount)
    new_weight = target.weight + amount
    console.print(
        f"[green]Weight bumped[/green] {node_id[:8]}… "
        f"[dim]{target.weight:.2f} → {new_weight:.2f}[/dim]"
    )


@app.command()
def clean(
    days: int = typer.Option(7, "--days", "-d", help="Remove nodes inactive for at least this many days"),
    tier: int = typer.Option(0, "--tier", "-t", help="Limit to a specific tier (0 = all non-tier-3)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be removed without deleting"),
    project_path: str = typer.Option("", "--project", help="Project path (defaults to CWD)"),
) -> None:
    """Remove stale tier-1 and tier-2 nodes that have not been accessed recently.

    Tier-3 (procedural) nodes are never touched by this command; use
    `cortex prune <id>` to remove them individually.

    Use --dry-run to preview what would be deleted without committing.
    """
    root = Path(project_path) if project_path else _project_root()
    g = open_graph(root)

    if g is None:
        console.print("[yellow]No Cortex database found.[/yellow]")
        raise typer.Exit(0)

    project = str(root)
    stale = g.get_stale_nodes(project, days=days, tier=tier if tier else None)

    if not stale:
        console.print(f"[green]No stale nodes[/green] older than {days} days.")
        raise typer.Exit(0)

    table = Table(title=f"Stale nodes (inactive > {days} days){' [DRY RUN]' if dry_run else ''}")
    table.add_column("T", style="cyan", justify="center", width=3)
    table.add_column("Last accessed", style="dim")
    table.add_column("Weight", justify="right", width=7)
    table.add_column("Text")
    table.add_column("ID", style="dim", width=10)

    for node in stale:
        table.add_row(
            str(node.tier),
            _fmt_ts(node.last_accessed),
            f"{node.weight:.2f}",
            node.text[:60],
            node.id[:8] + "…",
        )

    console.print(table)

    if dry_run:
        console.print(f"\n[dim]Dry run: {len(stale)} nodes would be removed.[/dim]")
        raise typer.Exit(0)

    typer.confirm(f"Delete {len(stale)} stale nodes?", abort=True)
    deleted = g.delete_nodes_bulk([n.id for n in stale])
    console.print(f"[green]Removed {deleted} stale nodes.[/green]")


@app.command(name="list")
def list_nodes(
    tier: int = typer.Option(0, "--tier", "-t", help="Filter by tier (0 = all)"),
    node_type: str = typer.Option("", "--type", help="Filter by type: observation, fact, convention, error"),
    source: str = typer.Option("", "--source", "-s", help="Filter by source: jsonl, ast, git, nlp"),
    sort: str = typer.Option("weight", "--sort", help="Sort by: weight, created, accessed"),
    limit: int = typer.Option(25, "--limit", "-n", help="Maximum nodes to show"),
    project_path: str = typer.Option("", "--project", help="Project path (defaults to CWD)"),
) -> None:
    """List memory nodes in a paginated table with optional filters."""
    root = Path(project_path) if project_path else _project_root()
    g = open_graph(root)

    if g is None:
        console.print("[yellow]No Cortex database found.[/yellow]")
        raise typer.Exit(0)

    project = str(root)
    valid_types = {"observation", "fact", "convention", "error"}
    valid_sources = {"jsonl", "ast", "git", "nlp"}
    valid_sorts = {"weight", "created", "accessed"}

    if node_type and node_type not in valid_types:
        console.print(f"[red]Invalid type '{node_type}'. Choose from: {', '.join(sorted(valid_types))}[/red]")
        raise typer.Exit(1)
    if source and source not in valid_sources:
        console.print(f"[red]Invalid source '{source}'. Choose from: {', '.join(sorted(valid_sources))}[/red]")
        raise typer.Exit(1)
    if sort not in valid_sorts:
        console.print(f"[red]Invalid sort '{sort}'. Choose from: {', '.join(sorted(valid_sorts))}[/red]")
        raise typer.Exit(1)

    sort_col = {"weight": "weight", "created": "created_at", "accessed": "last_accessed"}[sort]

    clauses = ["project = ?"]
    params: list = [project]
    if tier:
        clauses.append("tier = ?")
        params.append(tier)
    if node_type:
        clauses.append("type = ?")
        params.append(node_type)
    if source:
        clauses.append("source = ?")
        params.append(source)

    where = " AND ".join(clauses)
    rows = g._conn.execute(
        f"SELECT id, type, tier, text, weight, session_count, source, created_at "
        f"FROM nodes WHERE {where} ORDER BY {sort_col} DESC LIMIT ?",
        params + [limit],
    ).fetchall()

    total = g._conn.execute(
        f"SELECT COUNT(*) FROM nodes WHERE {where}", params
    ).fetchone()[0]

    if not rows:
        console.print("[dim]No nodes found.[/dim]")
        raise typer.Exit(0)

    title = f"Nodes — {project}"
    if any([tier, node_type, source]):
        parts = []
        if tier:
            parts.append(f"tier={tier}")
        if node_type:
            parts.append(f"type={node_type}")
        if source:
            parts.append(f"source={source}")
        title += f" [{', '.join(parts)}]"

    table = Table(title=title)
    table.add_column("T", style="cyan", justify="center", width=3)
    table.add_column("Type", style="dim", width=12)
    table.add_column("Src", style="dim", width=5)
    table.add_column("Weight", justify="right", width=7)
    table.add_column("Sess", justify="right", width=5)
    table.add_column("Text")
    table.add_column("ID", style="dim", width=10)

    for row in rows:
        table.add_row(
            str(row["tier"]),
            row["type"],
            row["source"],
            f"{row['weight']:.2f}",
            str(row["session_count"]),
            row["text"][:72],
            row["id"][:8] + "…",
        )

    console.print(table)
    if total > limit:
        console.print(f"[dim]Showing {limit} of {total} nodes. Use --limit to see more.[/dim]")


@app.command()
def doctor(
    project_path: str = typer.Option("", "--project", help="Project path (defaults to CWD)"),
) -> None:
    """Verify that all Cortex runtime dependencies are healthy.

    Checks: spaCy model, sentence-transformers, SQLite schema, hook scripts,
    and the Claude Code plugin manifest.
    """
    root = Path(project_path) if project_path else _project_root()
    hooks_dir = Path(__file__).parent.parent / "hooks"
    plugin_path = Path.home() / ".claude" / "plugins" / "cortex.json"

    all_ok = True

    def check(label: str, ok: bool, detail: str = "") -> None:
        nonlocal all_ok
        icon = "[green]✓[/green]" if ok else "[red]✗[/red]"
        msg = f"  {icon}  {label}"
        if detail:
            msg += f"  [dim]{detail}[/dim]"
        console.print(msg)
        if not ok:
            all_ok = False

    console.print("\n[bold]Cortex Doctor[/bold]\n")

    # spaCy model
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        check("spaCy en_core_web_sm", True, f"v{nlp.meta.get('version', '?')}")
    except Exception as exc:
        check("spaCy en_core_web_sm", False, f"{exc}")

    # sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        SentenceTransformer.__new__(SentenceTransformer)  # don't load model, just import
        check("sentence-transformers", True, "import OK")
    except Exception as exc:
        check("sentence-transformers", False, str(exc))

    # tiktoken
    try:
        import tiktoken
        tiktoken.get_encoding("cl100k_base")
        check("tiktoken cl100k_base", True, "encoding loaded")
    except Exception as exc:
        check("tiktoken cl100k_base", False, str(exc))

    # SQLite schema
    db_exists = open_graph(root) is not None
    check(
        "Cortex database",
        db_exists,
        str(root / ".cortex" / "cortex.db") if db_exists else "not found (run a session first)",
    )

    # Hook scripts
    for hook_name in ("inject.py", "extract.py", "compact.py"):
        path = hooks_dir / hook_name
        check(f"Hook: {hook_name}", path.exists(), str(path) if path.exists() else "missing")

    # Plugin manifest
    check(
        "Plugin manifest",
        plugin_path.exists(),
        str(plugin_path) if plugin_path.exists() else "not installed — run: cortex install",
    )

    console.print()
    if all_ok:
        console.print("[green bold]All checks passed.[/green bold]")
    else:
        console.print("[yellow]Some checks failed. See details above.[/yellow]")
    console.print()


@app.command()
def install() -> None:
    """Write plugin.json to the Claude Code plugins directory."""
    plugins_dir = Path.home() / ".claude" / "plugins"
    plugins_dir.mkdir(parents=True, exist_ok=True)

    hooks_dir = Path(__file__).parent.parent / "hooks"
    plugin = {
        "name": "cortex",
        "version": "0.1.0",
        "hooks": {
            "SessionStart": [{"command": f"python3 {hooks_dir / 'inject.py'}"}],
            "Stop": [{"command": f"python3 {hooks_dir / 'extract.py'}"}],
            "PostCompact": [{"command": f"python3 {hooks_dir / 'compact.py'}"}],
        },
    }

    dest = plugins_dir / "cortex.json"
    dest.write_text(json.dumps(plugin, indent=2))
    console.print(f"[green]Installed:[/green] {dest}")
    console.print("Restart Claude Code to activate Cortex.")


@app.command()
def dashboard() -> None:
    """Start the Cortex dashboard server on port 7000."""
    try:
        import uvicorn

        dashboard_module = Path(__file__).parent.parent / "dashboard" / "server.py"
        if not dashboard_module.exists():
            console.print("[yellow]Dashboard not yet built.[/yellow]")
            raise typer.Exit(1)

        console.print(
            "[green]Starting Cortex dashboard on http://localhost:7000[/green]"
        )
        uvicorn.run("dashboard.server:app", host="127.0.0.1", port=7000, reload=False)
    except ImportError as exc:
        console.print("[red]uvicorn not installed. Run: pip install uvicorn[/red]")
        raise typer.Exit(1) from exc


if __name__ == "__main__":
    app()
