"""Tests for dashboard/server.py — FastAPI routes and helpers."""

from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "cortex.db"
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(Path("schema.sql").read_text())
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def project(tmp_path: Path) -> str:
    return str(tmp_path)


def _seed_node(db: Path, project: str, text: str = "a node", tier: int = 1) -> str:
    conn = sqlite3.connect(str(db))
    nid = str(uuid.uuid4())
    now = int(time.time())
    conn.execute(
        """INSERT INTO nodes (id, type, tier, text, rationale, embedding, precision_bits,
                              weight, project, scope, source, last_accessed, created_at, session_count)
           VALUES (?, 'observation', ?, ?, NULL, NULL, 32, 1.0, ?, 'project', 'jsonl', ?, ?, 1)""",
        (nid, tier, text, project, now, now),
    )
    conn.commit()
    conn.close()
    return nid


def _seed_session(db: Path, project: str) -> str:
    conn = sqlite3.connect(str(db))
    sid = str(uuid.uuid4())
    now = int(time.time())
    conn.execute(
        """INSERT INTO sessions (id, project, started_at, ended_at, nodes_written,
                                  nodes_evicted, nodes_promoted, tokens_raw,
                                  tokens_injected, transcript_path)
           VALUES (?, ?, ?, ?, 3, 0, 0, 500, 200, '')""",
        (sid, project, now - 60, now),
    )
    conn.commit()
    conn.close()
    return sid


def _seed_edge(db: Path, source_id: str, target_id: str) -> None:
    conn = sqlite3.connect(str(db))
    conn.execute(
        "INSERT OR IGNORE INTO edges (source_id, target_id, strength, last_seen) VALUES (?, ?, ?, ?)",
        (source_id, target_id, 1.0, int(time.time())),
    )
    conn.commit()
    conn.close()


@pytest.fixture
def client(db_path: Path, project: str, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from dashboard import server as srv

    monkeypatch.setenv("CLAUDE_PROJECT_PATH", project)
    monkeypatch.setattr("cli.config.db_path", lambda root: db_path)
    return TestClient(srv.app)


# ---------------------------------------------------------------------------
# _open_db_ro
# ---------------------------------------------------------------------------


class TestOpenDbRo:
    def test_returns_none_when_no_db(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dashboard.server import _open_db_ro

        monkeypatch.delenv("CORTEX_DB_PATH", raising=False)
        result = _open_db_ro(tmp_path)
        assert result is None

    def test_returns_connection_when_db_exists(
        self, db_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dashboard.server import _open_db_ro

        monkeypatch.setenv("CORTEX_DB_PATH", str(db_path))
        conn = _open_db_ro(db_path.parent)
        assert conn is not None
        conn.close()


# ---------------------------------------------------------------------------
# _resolve_root
# ---------------------------------------------------------------------------


class TestResolveRoot:
    def test_returns_given_path_when_nonempty(self, tmp_path: Path) -> None:
        from dashboard.server import _resolve_root

        assert _resolve_root(str(tmp_path)) == tmp_path

    def test_falls_back_to_env_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dashboard.server import _resolve_root

        monkeypatch.setenv("CLAUDE_PROJECT_PATH", str(tmp_path))
        assert _resolve_root("") == tmp_path

    def test_falls_back_to_cwd_when_no_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dashboard.server import _resolve_root

        monkeypatch.delenv("CLAUDE_PROJECT_PATH", raising=False)
        result = _resolve_root("")
        assert result == Path.cwd()


# ---------------------------------------------------------------------------
# GET /api/status
# ---------------------------------------------------------------------------


class TestApiStatus:
    def test_no_db_returns_db_exists_false(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dashboard import server as srv

        monkeypatch.setenv("CLAUDE_PROJECT_PATH", str(tmp_path))
        monkeypatch.delenv("CORTEX_DB_PATH", raising=False)
        c = TestClient(srv.app)
        resp = c.get("/api/status")
        assert resp.status_code == 200
        assert resp.json()["db_exists"] is False

    def test_with_db_returns_tier_counts(
        self,
        db_path: Path,
        project: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from dashboard import server as srv

        _seed_node(db_path, project, tier=1)
        _seed_node(db_path, project, tier=2)
        monkeypatch.setenv("CLAUDE_PROJECT_PATH", project)
        monkeypatch.setenv("CORTEX_DB_PATH", str(db_path))
        c = TestClient(srv.app)
        resp = c.get("/api/status")
        assert resp.status_code == 200
        d = resp.json()
        assert d["db_exists"] is True
        assert "tier_counts" in d

    def test_last_session_included_when_present(
        self,
        db_path: Path,
        project: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from dashboard import server as srv

        _seed_session(db_path, project)
        monkeypatch.setenv("CLAUDE_PROJECT_PATH", project)
        monkeypatch.setenv("CORTEX_DB_PATH", str(db_path))
        c = TestClient(srv.app)
        resp = c.get("/api/status")
        d = resp.json()
        assert "last_session" in d


# ---------------------------------------------------------------------------
# GET /api/nodes
# ---------------------------------------------------------------------------


class TestApiNodes:
    def test_no_db_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dashboard import server as srv

        monkeypatch.setenv("CLAUDE_PROJECT_PATH", str(tmp_path))
        monkeypatch.delenv("CORTEX_DB_PATH", raising=False)
        c = TestClient(srv.app)
        assert c.get("/api/nodes").json() == []

    def test_returns_all_nodes(
        self, db_path: Path, project: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dashboard import server as srv

        _seed_node(db_path, project, text="node a")
        _seed_node(db_path, project, text="node b")
        monkeypatch.setenv("CLAUDE_PROJECT_PATH", project)
        monkeypatch.setenv("CORTEX_DB_PATH", str(db_path))
        c = TestClient(srv.app)
        nodes = c.get(f"/api/nodes?project={project}").json()
        assert len(nodes) == 2

    def test_tier_filter(
        self, db_path: Path, project: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dashboard import server as srv

        _seed_node(db_path, project, tier=1)
        _seed_node(db_path, project, tier=2)
        monkeypatch.setenv("CLAUDE_PROJECT_PATH", project)
        monkeypatch.setenv("CORTEX_DB_PATH", str(db_path))
        c = TestClient(srv.app)
        tier1 = c.get(f"/api/nodes?project={project}&tier=1").json()
        assert all(n["tier"] == 1 for n in tier1)


# ---------------------------------------------------------------------------
# GET /api/edges
# ---------------------------------------------------------------------------


class TestApiEdges:
    def test_no_db_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dashboard import server as srv

        monkeypatch.setenv("CLAUDE_PROJECT_PATH", str(tmp_path))
        monkeypatch.delenv("CORTEX_DB_PATH", raising=False)
        c = TestClient(srv.app)
        assert c.get("/api/edges").json() == []

    def test_returns_edges_for_project(
        self, db_path: Path, project: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dashboard import server as srv

        n1 = _seed_node(db_path, project, text="source node")
        n2 = _seed_node(db_path, project, text="target node")
        _seed_edge(db_path, n1, n2)
        monkeypatch.setenv("CLAUDE_PROJECT_PATH", project)
        monkeypatch.setenv("CORTEX_DB_PATH", str(db_path))
        c = TestClient(srv.app)
        edges = c.get(f"/api/edges?project={project}").json()
        assert len(edges) == 1
        assert edges[0]["source_id"] == n1
        assert edges[0]["target_id"] == n2


# ---------------------------------------------------------------------------
# GET /api/sessions
# ---------------------------------------------------------------------------


class TestApiSessions:
    def test_no_db_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dashboard import server as srv

        monkeypatch.setenv("CLAUDE_PROJECT_PATH", str(tmp_path))
        monkeypatch.delenv("CORTEX_DB_PATH", raising=False)
        c = TestClient(srv.app)
        assert c.get("/api/sessions").json() == []

    def test_returns_sessions(
        self, db_path: Path, project: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dashboard import server as srv

        _seed_session(db_path, project)
        _seed_session(db_path, project)
        monkeypatch.setenv("CLAUDE_PROJECT_PATH", project)
        monkeypatch.setenv("CORTEX_DB_PATH", str(db_path))
        c = TestClient(srv.app)
        sessions = c.get(f"/api/sessions?project={project}").json()
        assert len(sessions) == 2

    def test_limit_respected(
        self, db_path: Path, project: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dashboard import server as srv

        for _ in range(5):
            _seed_session(db_path, project)
        monkeypatch.setenv("CLAUDE_PROJECT_PATH", project)
        monkeypatch.setenv("CORTEX_DB_PATH", str(db_path))
        c = TestClient(srv.app)
        sessions = c.get(f"/api/sessions?project={project}&limit=2").json()
        assert len(sessions) <= 2


# ---------------------------------------------------------------------------
# GET / — dashboard UI
# ---------------------------------------------------------------------------


class TestDashboardUi:
    def test_serves_html(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from dashboard import server as srv

        monkeypatch.setenv("CLAUDE_PROJECT_PATH", str(tmp_path))
        c = TestClient(srv.app)
        resp = c.get("/")
        assert resp.status_code == 200
        assert "html" in resp.text.lower()

    def test_serves_index_html_when_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dashboard import server as srv

        fake_app = tmp_path / "app"
        fake_app.mkdir()
        (fake_app / "index.html").write_text("<html><body>custom</body></html>")
        monkeypatch.setattr("dashboard.server._APP_DIR", fake_app)
        c = TestClient(srv.app)
        resp = c.get("/")
        assert "custom" in resp.text


# ---------------------------------------------------------------------------
# _ConnectionManager
# ---------------------------------------------------------------------------


class TestConnectionManager:
    def test_broadcast_removes_dead_connections(self) -> None:
        import asyncio

        from dashboard.server import _ConnectionManager

        mgr = _ConnectionManager()

        class FakeWS:
            async def accept(self) -> None:
                pass

            async def send_text(self, data: str) -> None:
                raise RuntimeError("dead")

        ws = FakeWS()

        async def run() -> None:
            await mgr.connect(ws)  # type: ignore[arg-type]
            assert len(mgr.active) == 1
            await mgr.broadcast("hello")
            assert len(mgr.active) == 0

        asyncio.run(run())

    def test_disconnect_removes_connection(self) -> None:
        import asyncio

        from dashboard.server import _ConnectionManager

        mgr = _ConnectionManager()

        class FakeWS:
            async def accept(self) -> None:
                pass

        ws = FakeWS()

        async def run() -> None:
            await mgr.connect(ws)  # type: ignore[arg-type]
            mgr.disconnect(ws)  # type: ignore[arg-type]
            assert len(mgr.active) == 0

        asyncio.run(run())


# ---------------------------------------------------------------------------
# _build_graph_snapshot
# ---------------------------------------------------------------------------


class TestBuildGraphSnapshot:
    def test_snapshot_has_required_keys(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dashboard.server import _build_graph_snapshot

        monkeypatch.setenv("CLAUDE_PROJECT_PATH", str(tmp_path))
        monkeypatch.delenv("CORTEX_DB_PATH", raising=False)
        snap = _build_graph_snapshot()
        assert "nodes" in snap
        assert "edges" in snap
        assert "ts" in snap
        assert isinstance(snap["ts"], int)
