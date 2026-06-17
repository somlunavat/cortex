"""Cortex CLI entry point.

Commands:
    status    — node counts by tier, token savings, last session
    graph     — ASCII adjacency summary
    inspect   — full metadata for a single node
    prune     — manually evict a node by id
    reset     — wipe all nodes for a project
    search    — BM25 text search across memory nodes
    decay     — run decay/eviction/promotion manually
    install   — write plugin.json to Claude Code plugins directory
    dashboard — start the dashboard server on port 7000
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
def status() -> None:
    """Show node counts by tier, last session, and token savings."""
    root = _project_root()
    graph = open_graph(root)

    if graph is None:
        console.print("[yellow]No Cortex database found in this project.[/yellow]")
        raise typer.Exit(0)

    project = str(root)
    nodes = graph.get_all_nodes(project=project)

    tier_counts = {1: 0, 2: 0, 3: 0}
    for node in nodes:
        tier_counts[node.tier] = tier_counts.get(node.tier, 0) + 1

    table = Table(title=f"Cortex — {project}")
    table.add_column("Tier", style="cyan")
    table.add_column("Label")
    table.add_column("Nodes", justify="right")
    table.add_row("1", "Ephemeral", str(tier_counts[1]))
    table.add_row("2", "Semantic", str(tier_counts[2]))
    table.add_row("3", "Procedural", str(tier_counts[3]))
    console.print(table)

    row = graph._conn.execute(
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
) -> None:
    """Search memory nodes by text using BM25 ranking."""
    root = _project_root()
    g = open_graph(root)

    if g is None:
        console.print("[yellow]No Cortex database found.[/yellow]")
        raise typer.Exit(0)

    project = str(root)
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
