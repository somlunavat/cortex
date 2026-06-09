"""Cortex CLI entry point.

Commands:
    status   — node counts by tier, token savings, last session
    graph    — ASCII adjacency summary
    inspect  — full metadata for a single node
    prune    — manually evict a node by id
    reset    — wipe all nodes for a project
    install  — write plugin.json to Claude Code plugins directory
    dashboard — start the dashboard server on port 7000
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.config import SCHEMA_PATH, db_path
from core.graph import Graph

app = typer.Typer(
    name="cortex",
    help="Cortex — persistent memory layer for Claude Code.",
    add_completion=False,
)
console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _open_graph(project_root: Path) -> Graph | None:
    """Open the cortex database for a project, or return None if not found."""
    path = db_path(project_root)
    if not path.exists():
        return None
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_PATH.read_text())
    return Graph(connection=conn)


def _project_root() -> Path:
    """Resolve project root from CWD."""
    return Path.cwd()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def status() -> None:
    """Show node counts by tier, last session, and token savings."""
    root = _project_root()
    graph = _open_graph(root)

    if graph is None:
        console.print("[yellow]No Cortex database found in this project.[/yellow]")
        console.print(f"  Expected: {db_path(root)}")
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
        "SELECT ended_at, nodes_written, nodes_evicted, tokens_raw, tokens_injected "
        "FROM sessions WHERE project = ? ORDER BY ended_at DESC LIMIT 1",
        (project,),
    ).fetchone()

    if row:
        console.print(f"\nLast session: [green]{row['ended_at']}[/green]")
        console.print(f"  Nodes written: {row['nodes_written']}")
        console.print(f"  Nodes evicted: {row['nodes_evicted']}")
        if row["tokens_raw"] and row["tokens_injected"]:
            saved = row["tokens_raw"] - row["tokens_injected"]
            console.print(f"  Tokens saved:  {saved}")


@app.command()
def graph() -> None:
    """Print an ASCII adjacency summary of the knowledge graph."""
    root = _project_root()
    g = _open_graph(root)

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
    g = _open_graph(root)

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
    table.add_row("last_accessed", str(target.last_accessed))
    table.add_row("created_at", str(target.created_at))
    console.print(table)


@app.command()
def prune(node_id: str = typer.Argument(..., help="Node UUID to evict")) -> None:
    """Manually evict a node from the graph."""
    root = _project_root()
    g = _open_graph(root)

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
    g = _open_graph(root)

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
    console.print(f"[green]Reset complete:[/green] removed {deleted} nodes for {project}")


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
