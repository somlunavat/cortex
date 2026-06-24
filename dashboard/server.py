"""Cortex dashboard — FastAPI server with WebSocket and static frontend.

Reads cortex.db via SQLite WAL (read-only connection, no write locks).
Serves the React frontend from dashboard/app/.
WebSocket endpoint pushes graph snapshots to connected browsers.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.config import db_path as _db_path

_PACKAGE_ROOT = Path(__file__).parent.parent
_APP_DIR = Path(__file__).parent / "app"

app = FastAPI(title="Cortex Dashboard", version="0.1.0")


# ---------------------------------------------------------------------------
# Database helpers (read-only)
# ---------------------------------------------------------------------------


def _open_db_ro(project_root: Path) -> sqlite3.Connection | None:
    path = _db_path(project_root)
    if not path.exists():
        return None
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _project_root() -> Path:
    return Path(os.environ.get("CLAUDE_PROJECT_PATH", str(Path.cwd())))


# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------


@app.get("/api/status")
def api_status() -> dict[str, Any]:
    """Return tier counts and last-session summary for the current project."""
    root = _project_root()
    project = str(root)
    conn = _open_db_ro(root)

    if conn is None:
        return {"project": project, "db_exists": False}

    with contextlib.closing(conn):
        tier_counts = dict(
            conn.execute(
                "SELECT tier, COUNT(*) FROM nodes WHERE project = ? GROUP BY tier",
                (project,),
            ).fetchall()
        )

        last_session = conn.execute(
            "SELECT * FROM sessions WHERE project = ? ORDER BY ended_at DESC LIMIT 1",
            (project,),
        ).fetchone()

    result: dict[str, Any] = {
        "project": project,
        "db_exists": True,
        "tier_counts": {str(k): v for k, v in tier_counts.items()},
    }

    if last_session:
        result["last_session"] = dict(last_session)

    return result


@app.get("/api/nodes")
def api_nodes(
    project: str = "",
    tier: int | None = None,
) -> list[dict[str, Any]]:
    """Return all nodes (optionally filtered by tier) as JSON objects."""
    root = Path(project) if project else _project_root()
    conn = _open_db_ro(root)
    if conn is None:
        return []

    with contextlib.closing(conn):
        if tier is not None:
            rows = conn.execute(
                "SELECT id, type, tier, text, rationale, weight, scope, source, "
                "last_accessed, created_at, session_count, precision_bits "
                "FROM nodes WHERE project = ? AND tier = ?",
                (str(root), tier),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, type, tier, text, rationale, weight, scope, source, "
                "last_accessed, created_at, session_count, precision_bits "
                "FROM nodes WHERE project = ?",
                (str(root),),
            ).fetchall()

    return [dict(row) for row in rows]


@app.get("/api/edges")
def api_edges(project: str = "") -> list[dict[str, Any]]:
    """Return all edges for nodes in the current project."""
    root = Path(project) if project else _project_root()
    conn = _open_db_ro(root)
    if conn is None:
        return []

    with contextlib.closing(conn):
        rows = conn.execute(
            """
            SELECT e.source_id, e.target_id, e.strength, e.last_seen
            FROM edges e
            JOIN nodes n ON n.id = e.source_id
            WHERE n.project = ?
            """,
            (str(root),),
        ).fetchall()

    return [dict(row) for row in rows]


@app.get("/api/sessions")
def api_sessions(project: str = "", limit: int = 50) -> list[dict[str, Any]]:
    """Return recent session records for the current project."""
    root = Path(project) if project else _project_root()
    conn = _open_db_ro(root)
    if conn is None:
        return []

    with contextlib.closing(conn):
        rows = conn.execute(
            "SELECT id, project, started_at, ended_at, nodes_written, nodes_evicted, "
            "nodes_promoted, tokens_raw, tokens_injected FROM sessions WHERE project = ? "
            "ORDER BY ended_at DESC LIMIT ?",
            (str(root), limit),
        ).fetchall()

    return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# WebSocket — push graph snapshots
# ---------------------------------------------------------------------------


class _ConnectionManager:
    def __init__(self) -> None:
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self.active.remove(ws)

    async def broadcast(self, data: str) -> None:
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active.remove(ws)


_manager = _ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint — pushes a graph snapshot on connect and every 5s."""
    await _manager.connect(websocket)
    try:
        while True:
            snapshot = {
                "nodes": api_nodes(),
                "edges": api_edges(),
                "ts": int(time.time()),
            }
            await websocket.send_text(json.dumps(snapshot))
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        _manager.disconnect(websocket)


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------


if _APP_DIR.exists() and any(_APP_DIR.iterdir()):
    app.mount("/static", StaticFiles(directory=str(_APP_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def dashboard_ui() -> str:
    """Serve the dashboard HTML."""
    index = _APP_DIR / "index.html"
    if index.exists():
        return index.read_text()
    return _EMBEDDED_HTML


# ---------------------------------------------------------------------------
# Embedded single-file dashboard (no build step required)
# ---------------------------------------------------------------------------

_EMBEDDED_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Cortex Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: monospace; background: #0d0d0d; color: #e0e0e0; }
    header { padding: 1rem 2rem; border-bottom: 1px solid #333; display: flex; gap: 2rem; align-items: center; }
    header h1 { font-size: 1.1rem; color: #7dd3fc; }
    .badge { background: #1e293b; border: 1px solid #334; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.8rem; }
    main { display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: 340px 1fr; gap: 1rem; padding: 1rem; height: calc(100vh - 60px); }
    section { background: #111; border: 1px solid #222; border-radius: 6px; padding: 1rem; overflow: hidden; }
    section h2 { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.75rem; }
    #graph-section { grid-row: 1 / 3; }
    svg { width: 100%; height: 100%; }
    .node { cursor: pointer; }
    .link { stroke: #334; stroke-opacity: 0.5; }
    #inspector { padding: 1rem; font-size: 0.8rem; line-height: 1.6; }
    #inspector .field { display: flex; gap: 0.5rem; border-bottom: 1px solid #1e293b; padding: 0.25rem 0; }
    #inspector .key { color: #7dd3fc; min-width: 120px; }
    #inspector .val { color: #e2e8f0; word-break: break-all; }
    .tier-1 { fill: #ef4444; }
    .tier-2 { fill: #f59e0b; }
    .tier-3 { fill: #22c55e; }
  </style>
</head>
<body>
<header>
  <h1>⬡ CORTEX</h1>
  <span class="badge" id="project-badge">loading…</span>
  <span class="badge" id="node-count">0 nodes</span>
  <span class="badge" id="ws-status">connecting…</span>
</header>
<main>
  <section id="graph-section">
    <h2>Memory Graph</h2>
    <svg id="graph-svg"></svg>
  </section>
  <section>
    <h2>Token Savings</h2>
    <canvas id="savings-chart"></canvas>
  </section>
  <section>
    <h2>Node Inspector</h2>
    <div id="inspector"><p style="color:#64748b">Click a node to inspect it.</p></div>
  </section>
</main>
<script>
const WS_URL = `ws://${location.host}/ws`;
const API = '/api';

let chartInstance = null;
let simulation = null;

async function fetchSessions() {
  const r = await fetch(API + '/sessions');
  return r.ok ? r.json() : [];
}

async function renderChart(sessions) {
  const canvas = document.getElementById('savings-chart');
  const labels = sessions.map(s => new Date((s.ended_at || 0) * 1000).toLocaleDateString()).reverse();
  const savings = sessions.map(s => (s.tokens_raw || 0) - (s.tokens_injected || 0)).reverse();

  if (chartInstance) { chartInstance.destroy(); }
  chartInstance = new Chart(canvas, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Tokens Saved',
        data: savings,
        borderColor: '#7dd3fc',
        backgroundColor: 'rgba(125,211,252,0.1)',
        tension: 0.3,
        fill: true,
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: '#e0e0e0' } } },
      scales: {
        x: { ticks: { color: '#64748b' }, grid: { color: '#1e293b' } },
        y: { ticks: { color: '#64748b' }, grid: { color: '#1e293b' } },
      }
    }
  });
}

function renderGraph(nodes, edges) {
  const svg = d3.select('#graph-svg');
  svg.selectAll('*').remove();
  const rect = document.getElementById('graph-section').getBoundingClientRect();
  const W = rect.width - 32, H = rect.height - 48;
  svg.attr('viewBox', `0 0 ${W} ${H}`);

  const nodeMap = new Map(nodes.map(n => [n.id, n]));
  const links = edges.filter(e => nodeMap.has(e.source_id) && nodeMap.has(e.target_id))
    .map(e => ({ source: e.source_id, target: e.target_id, strength: e.strength }));

  if (simulation) simulation.stop();
  simulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).id(d => d.id).distance(80))
    .force('charge', d3.forceManyBody().strength(-120))
    .force('center', d3.forceCenter(W / 2, H / 2));

  const link = svg.append('g').selectAll('line').data(links).join('line').attr('class', 'link');

  const node = svg.append('g').selectAll('g').data(nodes).join('g')
    .attr('class', 'node')
    .call(d3.drag()
      .on('start', (e, d) => { if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y; })
      .on('end', (e, d) => { if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }))
    .on('click', (e, d) => inspectNode(d));

  node.append('circle')
    .attr('r', d => 4 + d.weight * 2)
    .attr('class', d => `tier-${d.tier}`)
    .attr('opacity', d => d.tier === 3 ? 1.0 : d.tier === 2 ? 0.75 : 0.5);

  node.append('title').text(d => d.text);

  simulation.on('tick', () => {
    link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    node.attr('transform', d => `translate(${d.x},${d.y})`);
  });

  document.getElementById('node-count').textContent = `${nodes.length} nodes`;
}

function inspectNode(d) {
  const fields = ['id','type','tier','text','rationale','weight','scope','source','session_count','precision_bits'];
  const html = fields.map(f =>
    `<div class="field"><span class="key">${f}</span><span class="val">${d[f] ?? '—'}</span></div>`
  ).join('');
  document.getElementById('inspector').innerHTML = html;
}

function initWS() {
  const ws = new WebSocket(WS_URL);
  const badge = document.getElementById('ws-status');
  ws.onopen = () => { badge.textContent = '● live'; badge.style.color = '#22c55e'; };
  ws.onclose = () => {
    badge.textContent = '○ reconnecting';
    badge.style.color = '#ef4444';
    setTimeout(initWS, 3000);
  };
  ws.onmessage = e => {
    const { nodes, edges } = JSON.parse(e.data);
    renderGraph(nodes, edges);
  };
}

async function init() {
  const r = await fetch(API + '/status');
  if (r.ok) {
    const s = await r.json();
    document.getElementById('project-badge').textContent = s.project || '—';
  }
  const sessions = await fetchSessions();
  renderChart(sessions);
  initWS();
}

init();
</script>
</body>
</html>
"""
