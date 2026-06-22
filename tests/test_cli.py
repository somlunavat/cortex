"""Tests for cli/cortex.py — CLI command integration tests.

Uses typer's CliRunner to invoke commands against an in-memory database
injected via the CORTEX_DB_PATH environment variable pointing to a temp file.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from cli.cortex import app

runner = CliRunner()


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """Create a temporary cortex.db with schema applied."""
    db_file = tmp_path / "cortex.db"
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    conn.executescript(Path("schema.sql").read_text())
    conn.close()
    return db_file


@pytest.fixture
def project(tmp_path: Path) -> str:
    return str(tmp_path)


def _seed_node(db_path: Path, project: str, text: str = "auth.py hotspot", tier: int = 1) -> str:
    """Insert one node directly via SQL and return its id."""
    conn = sqlite3.connect(str(db_path))
    node_id = str(uuid.uuid4())
    now = int(time.time())
    emb = np.random.default_rng(seed=42).random(384).astype(np.float32)
    conn.execute(
        """
        INSERT INTO nodes (id, type, tier, text, rationale, embedding, precision_bits,
                           weight, project, scope, source, last_accessed, created_at, session_count)
        VALUES (?, 'observation', ?, ?, NULL, ?, 32, 1.5, ?, 'module', 'jsonl', ?, ?, 2)
        """,
        (node_id, tier, text, emb.tobytes(), project, now, now),
    )
    conn.commit()
    conn.close()
    return node_id


def _seed_session(
    db_path: Path,
    project: str,
    nodes_written: int = 3,
    nodes_evicted: int = 1,
    nodes_promoted: int = 0,
    tokens_raw: int | None = 800,
    tokens_injected: int | None = 320,
) -> str:
    """Insert one session row and return its id."""
    conn = sqlite3.connect(str(db_path))
    sid = str(uuid.uuid4())
    now = int(time.time())
    conn.execute(
        """
        INSERT INTO sessions (id, project, started_at, ended_at,
                              nodes_written, nodes_evicted, nodes_promoted,
                              tokens_raw, tokens_injected, transcript_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            sid, project, now - 300, now,
            nodes_written, nodes_evicted, nodes_promoted,
            tokens_raw, tokens_injected,
            f"/tmp/{sid}.jsonl",
        ),
    )
    conn.commit()
    conn.close()
    return sid


# ---------------------------------------------------------------------------
# cortex sessions
# ---------------------------------------------------------------------------


class TestSessionsCommand:
    def _invoke_sessions(self, tmp_db: Path, project: str, extra_args: list[str] | None = None) -> object:
        args = ["sessions", "--project", project] + (extra_args or [])
        return runner.invoke(app, args, env={"CORTEX_DB_PATH": str(tmp_db)})

    def test_no_db_exits_cleanly(self, tmp_path: Path, project: str) -> None:
        result = runner.invoke(
            app,
            ["sessions", "--project", project],
            env={"CORTEX_DB_PATH": str(tmp_path / "nonexistent.db")},
        )
        assert result.exit_code == 0
        assert "No Cortex" in result.output

    def test_no_sessions_shows_message(self, tmp_db: Path, project: str) -> None:
        result = self._invoke_sessions(tmp_db, project)
        assert result.exit_code == 0
        assert "No sessions" in result.output

    def test_shows_session_row(self, tmp_db: Path, project: str) -> None:
        _seed_session(tmp_db, project)
        result = self._invoke_sessions(tmp_db, project)
        assert result.exit_code == 0
        assert "Written" in result.output
        # Rich may wrap "Tokens saved" header; check for at least "saved" or "Tokens"
        assert "saved" in result.output.lower() or "Tokens" in result.output

    def test_token_savings_computed(self, tmp_db: Path, project: str) -> None:
        _seed_session(tmp_db, project, tokens_raw=800, tokens_injected=320)
        result = self._invoke_sessions(tmp_db, project)
        assert "480" in result.output  # 800 - 320

    def test_null_tokens_shows_dash(self, tmp_db: Path, project: str) -> None:
        _seed_session(tmp_db, project, tokens_raw=None, tokens_injected=None)
        result = self._invoke_sessions(tmp_db, project)
        assert result.exit_code == 0
        assert "—" in result.output

    def test_limit_flag_respected(self, tmp_db: Path, project: str) -> None:
        for _ in range(5):
            _seed_session(tmp_db, project)
        result = self._invoke_sessions(tmp_db, project, ["--limit", "2"])
        assert result.exit_code == 0
        lines_with_uuid = [ln for ln in result.output.splitlines() if "…" in ln]
        assert len(lines_with_uuid) == 2

    def test_multiple_sessions_most_recent_first(self, tmp_db: Path, project: str) -> None:
        """Most recent session (highest ended_at) should appear first in table."""
        conn = sqlite3.connect(str(tmp_db))
        now = int(time.time())
        sids = []
        for offset in [300, 200, 100]:
            sid = str(uuid.uuid4())
            sids.append(sid)
            conn.execute(
                """
                INSERT INTO sessions (id, project, started_at, ended_at, nodes_written,
                                      nodes_evicted, nodes_promoted, transcript_path)
                VALUES (?, ?, ?, ?, 1, 0, 0, '')
                """,
                (sid, project, now - offset - 60, now - offset),
            )
        conn.commit()
        conn.close()

        result = self._invoke_sessions(tmp_db, project)
        output = result.output
        # sids[2] has offset=100 (most recent), sids[0] has offset=300 (oldest)
        recent_prefix = sids[2][:8]
        oldest_prefix = sids[0][:8]
        assert recent_prefix in output
        assert oldest_prefix in output
        assert output.find(recent_prefix) < output.find(oldest_prefix)


# ---------------------------------------------------------------------------
# cortex stats
# ---------------------------------------------------------------------------


class TestStatsCommand:
    def _invoke_stats(self, tmp_db: Path, project: str) -> object:
        return runner.invoke(
            app, ["stats", "--project", project], env={"CORTEX_DB_PATH": str(tmp_db)}
        )

    def test_no_db_exits_cleanly(self, tmp_path: Path, project: str) -> None:
        result = runner.invoke(
            app,
            ["stats", "--project", project],
            env={"CORTEX_DB_PATH": str(tmp_path / "nonexistent.db")},
        )
        assert result.exit_code == 0
        assert "No Cortex" in result.output

    def test_shows_session_count(self, tmp_db: Path, project: str) -> None:
        _seed_session(tmp_db, project)
        _seed_session(tmp_db, project)
        result = self._invoke_stats(tmp_db, project)
        assert result.exit_code == 0
        assert "Sessions recorded" in result.output
        assert "2" in result.output

    def test_shows_node_counts_by_tier(self, tmp_db: Path, project: str) -> None:
        _seed_node(tmp_db, project, text="tier 1 node", tier=1)
        _seed_node(tmp_db, project, text="tier 2 node", tier=2)
        _seed_node(tmp_db, project, text="tier 3 node", tier=3)
        result = self._invoke_stats(tmp_db, project)
        assert "Tier 1" in result.output
        assert "Tier 2" in result.output
        assert "Tier 3" in result.output

    def test_total_tokens_saved_aggregated(self, tmp_db: Path, project: str) -> None:
        _seed_session(tmp_db, project, tokens_raw=600, tokens_injected=200)
        _seed_session(tmp_db, project, tokens_raw=400, tokens_injected=150)
        result = self._invoke_stats(tmp_db, project)
        # (600-200) + (400-150) = 400 + 250 = 650
        assert "650" in result.output

    def test_null_tokens_treated_as_zero(self, tmp_db: Path, project: str) -> None:
        _seed_session(tmp_db, project, tokens_raw=None, tokens_injected=None)
        result = self._invoke_stats(tmp_db, project)
        assert result.exit_code == 0
        assert "Total tokens saved" in result.output

    def test_empty_db_shows_zeros(self, tmp_db: Path, project: str) -> None:
        result = self._invoke_stats(tmp_db, project)
        assert result.exit_code == 0
        assert "0" in result.output
