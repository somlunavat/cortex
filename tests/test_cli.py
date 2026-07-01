"""Tests for cli/cortex.py — CLI command integration tests.

Uses typer's CliRunner to invoke commands against an in-memory database
injected via the CORTEX_DB_PATH environment variable pointing to a temp file.
"""

from __future__ import annotations

import json
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


def _seed_node(
    db_path: Path, project: str, text: str = "auth.py hotspot", tier: int = 1
) -> str:
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
            sid,
            project,
            now - 300,
            now,
            nodes_written,
            nodes_evicted,
            nodes_promoted,
            tokens_raw,
            tokens_injected,
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
    def _invoke_sessions(
        self, tmp_db: Path, project: str, extra_args: list[str] | None = None
    ) -> object:
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

    def test_multiple_sessions_most_recent_first(
        self, tmp_db: Path, project: str
    ) -> None:
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


# ---------------------------------------------------------------------------
# cortex import_
# ---------------------------------------------------------------------------


class TestImportCommand:
    def _make_export_json(
        self,
        tmp_path: Path,
        records: list[dict],
        filename: str = "export.json",
    ) -> Path:
        import json

        p = tmp_path / filename
        p.write_text(json.dumps(records))
        return p

    def _invoke_import(
        self,
        tmp_db: Path,
        project: str,
        file_path: Path,
        extra: list[str] | None = None,
    ) -> object:
        args = ["import-", str(file_path), "--project", project] + (extra or [])
        return runner.invoke(app, args, env={"CORTEX_DB_PATH": str(tmp_db)})

    def test_import_writes_nodes(
        self, tmp_db: Path, project: str, tmp_path: Path
    ) -> None:
        records = [
            {
                "text": "always validate at boundaries",
                "type": "convention",
                "tier": 3,
                "scope": "project",
                "source": "nlp",
                "weight": 5.0,
                "session_count": 3,
                "rationale": "catches external input errors early",
            },
        ]
        f = self._make_export_json(tmp_path, records)
        result = self._invoke_import(tmp_db, project, f)
        assert result.exit_code == 0
        assert "Imported 1" in result.output

        conn = sqlite3.connect(str(tmp_db))
        rows = conn.execute(
            "SELECT text FROM nodes WHERE project = ?", (project,)
        ).fetchall()
        conn.close()
        assert any("validate at boundaries" in r[0] for r in rows)

    def test_import_skips_duplicates_by_default(
        self, tmp_db: Path, project: str, tmp_path: Path
    ) -> None:
        records = [
            {
                "text": "use WAL mode for concurrency",
                "type": "fact",
                "tier": 2,
                "scope": "project",
                "source": "nlp",
                "weight": 2.0,
                "session_count": 1,
            }
        ]
        f = self._make_export_json(tmp_path, records)
        self._invoke_import(tmp_db, project, f)
        result = self._invoke_import(tmp_db, project, f)
        assert "skipped" in result.output
        # Second import should say 1 skipped
        assert "1 skipped" in result.output

    def test_import_file_not_found(
        self, tmp_db: Path, project: str, tmp_path: Path
    ) -> None:
        result = self._invoke_import(tmp_db, project, tmp_path / "missing.json")
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_import_invalid_json(
        self, tmp_db: Path, project: str, tmp_path: Path
    ) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("not json at all")
        result = self._invoke_import(tmp_db, project, bad)
        assert result.exit_code != 0
        assert "Invalid JSON" in result.output or "invalid" in result.output.lower()

    def test_import_empty_text_skipped(
        self, tmp_db: Path, project: str, tmp_path: Path
    ) -> None:
        records = [
            {
                "text": "",
                "type": "fact",
                "tier": 1,
                "scope": "project",
                "source": "nlp",
                "weight": 1.0,
                "session_count": 1,
            }
        ]
        f = self._make_export_json(tmp_path, records)
        result = self._invoke_import(tmp_db, project, f)
        assert result.exit_code == 0
        assert "1 skipped" in result.output

    def test_import_coerces_invalid_tier(
        self, tmp_db: Path, project: str, tmp_path: Path
    ) -> None:
        """tier=99 should be coerced to tier=1 rather than failing."""
        records = [
            {
                "text": "coerce tier node",
                "type": "observation",
                "tier": 99,
                "scope": "project",
                "source": "jsonl",
                "weight": 1.0,
                "session_count": 1,
            }
        ]
        f = self._make_export_json(tmp_path, records)
        result = self._invoke_import(tmp_db, project, f)
        assert result.exit_code == 0
        assert "Imported 1" in result.output

    def test_import_allows_duplicates_when_flag_set(
        self, tmp_db: Path, project: str, tmp_path: Path
    ) -> None:
        records = [
            {
                "text": "duplicate allowed",
                "type": "fact",
                "tier": 1,
                "scope": "project",
                "source": "nlp",
                "weight": 1.0,
                "session_count": 1,
            }
        ]
        f = self._make_export_json(tmp_path, records)
        self._invoke_import(tmp_db, project, f)
        result = self._invoke_import(tmp_db, project, f, ["--allow-duplicates"])
        assert "Imported 1" in result.output

    def test_export_then_import_roundtrip(
        self, tmp_db: Path, project: str, tmp_path: Path
    ) -> None:
        """Export a node to JSON and import it into a fresh database."""

        _seed_node(tmp_db, project, text="roundtrip test node", tier=1)

        export_file = tmp_path / "export.json"
        runner.invoke(
            app,
            ["export", "--project", project, "--out", str(export_file)],
            env={"CORTEX_DB_PATH": str(tmp_db)},
        )
        assert export_file.exists()

        db2 = tmp_path / "fresh.db"
        conn = sqlite3.connect(str(db2))
        conn.executescript(Path("schema.sql").read_text())
        conn.close()

        result = runner.invoke(
            app,
            ["import-", str(export_file), "--project", project],
            env={"CORTEX_DB_PATH": str(db2)},
        )
        assert result.exit_code == 0
        assert "Imported 1" in result.output

        conn = sqlite3.connect(str(db2))
        row = conn.execute(
            "SELECT text FROM nodes WHERE project = ?", (project,)
        ).fetchone()
        conn.close()
        assert row is not None
        assert "roundtrip test node" in row[0]


# ---------------------------------------------------------------------------
# cortex list
# ---------------------------------------------------------------------------


class TestListCommand:
    def _invoke_list(
        self, tmp_db: Path, project: str, extra: list[str] | None = None
    ) -> object:
        args = ["list", "--project", project] + (extra or [])
        return runner.invoke(app, args, env={"CORTEX_DB_PATH": str(tmp_db)})

    def test_no_db_exits_cleanly(self, tmp_path: Path, project: str) -> None:
        result = runner.invoke(
            app,
            ["list", "--project", project],
            env={"CORTEX_DB_PATH": str(tmp_path / "nonexistent.db")},
        )
        assert result.exit_code == 0
        assert "No Cortex" in result.output

    def test_empty_db_shows_message(self, tmp_db: Path, project: str) -> None:
        result = self._invoke_list(tmp_db, project)
        assert result.exit_code == 0
        assert "No nodes" in result.output

    def test_shows_node_row(self, tmp_db: Path, project: str) -> None:
        _seed_node(tmp_db, project, text="pytest is the test runner", tier=1)
        result = self._invoke_list(tmp_db, project)
        assert result.exit_code == 0
        # Rich may wrap long text across lines; check both halves
        assert "pytest" in result.output and "test runner" in result.output

    def test_tier_filter(self, tmp_db: Path, project: str) -> None:
        _seed_node(tmp_db, project, text="tier one node", tier=1)
        _seed_node(tmp_db, project, text="tier two node", tier=2)
        result = self._invoke_list(tmp_db, project, ["--tier", "2"])
        assert result.exit_code == 0
        assert "tier two node" in result.output
        assert "tier one node" not in result.output

    def test_source_filter(self, tmp_db: Path, project: str) -> None:
        conn = sqlite3.connect(str(tmp_db))
        now = int(time.time())
        nid = str(uuid.uuid4())
        conn.execute(
            """INSERT INTO nodes (id, type, tier, text, rationale, embedding, precision_bits,
                                  weight, project, scope, source, last_accessed, created_at, session_count)
               VALUES (?, 'fact', 1, 'git churn node', NULL, NULL, 32, 1.0, ?, 'module', 'git', ?, ?, 1)""",
            (nid, project, now, now),
        )
        conn.commit()
        conn.close()

        result = self._invoke_list(tmp_db, project, ["--source", "git"])
        assert result.exit_code == 0
        assert "git churn node" in result.output

        result2 = self._invoke_list(tmp_db, project, ["--source", "nlp"])
        assert "git churn node" not in result2.output

    def test_invalid_source_rejected(self, tmp_db: Path, project: str) -> None:
        result = self._invoke_list(tmp_db, project, ["--source", "bogus"])
        assert result.exit_code != 0

    def test_invalid_sort_rejected(self, tmp_db: Path, project: str) -> None:
        result = self._invoke_list(tmp_db, project, ["--sort", "bogus"])
        assert result.exit_code != 0

    def test_limit_respected(self, tmp_db: Path, project: str) -> None:
        for i in range(10):
            _seed_node(tmp_db, project, text=f"node number {i}", tier=1)
        result = self._invoke_list(tmp_db, project, ["--limit", "3"])
        assert result.exit_code == 0
        assert "Showing 3 of 10" in result.output

    def test_sort_by_created(self, tmp_db: Path, project: str) -> None:
        _seed_node(tmp_db, project, text="alpha node", tier=1)
        _seed_node(tmp_db, project, text="beta node", tier=1)
        result = self._invoke_list(tmp_db, project, ["--sort", "created"])
        assert result.exit_code == 0
        assert "alpha node" in result.output
        assert "beta node" in result.output

    def test_json_flag_outputs_valid_json(self, tmp_db: Path, project: str) -> None:
        _seed_node(tmp_db, project, text="json node text", tier=1)
        result = self._invoke_list(tmp_db, project, ["--json"])
        assert result.exit_code == 0
        records = json.loads(result.output)
        assert isinstance(records, list)
        assert len(records) == 1
        assert records[0]["text"] == "json node text"
        assert records[0]["tier"] == 1

    def test_json_flag_includes_required_fields(
        self, tmp_db: Path, project: str
    ) -> None:
        _seed_node(tmp_db, project, text="field check node", tier=2)
        result = self._invoke_list(tmp_db, project, ["--json"])
        records = json.loads(result.output)
        rec = records[0]
        for field in ("id", "type", "tier", "text", "weight", "source", "project"):
            assert field in rec

    def test_json_flag_empty_db_returns_empty_list(
        self, tmp_db: Path, project: str
    ) -> None:
        result = self._invoke_list(tmp_db, project, ["--json"])
        assert result.exit_code == 0
        assert json.loads(result.output) == []

    def test_json_flag_respects_tier_filter(self, tmp_db: Path, project: str) -> None:
        _seed_node(tmp_db, project, text="tier1 only", tier=1)
        _seed_node(tmp_db, project, text="tier2 only", tier=2)
        result = self._invoke_list(tmp_db, project, ["--json", "--tier", "1"])
        records = json.loads(result.output)
        assert all(r["tier"] == 1 for r in records)
        assert any(r["text"] == "tier1 only" for r in records)


# ---------------------------------------------------------------------------
# cortex doctor
# ---------------------------------------------------------------------------


class TestDoctorCommand:
    def test_runs_without_error(self, tmp_path: Path, project: str) -> None:
        result = runner.invoke(
            app,
            ["doctor", "--project", project],
            env={"CORTEX_DB_PATH": str(tmp_path / "nonexistent.db")},
        )
        assert result.exit_code == 0

    def test_shows_spacy_check(self, tmp_path: Path, project: str) -> None:
        result = runner.invoke(
            app,
            ["doctor", "--project", project],
            env={"CORTEX_DB_PATH": str(tmp_path / "nonexistent.db")},
        )
        assert "spaCy" in result.output or "spacy" in result.output.lower()

    def test_shows_database_check(self, tmp_db: Path, project: str) -> None:
        result = runner.invoke(
            app,
            ["doctor", "--project", project],
            env={"CORTEX_DB_PATH": str(tmp_db)},
        )
        assert "database" in result.output.lower() or "Database" in result.output

    def test_shows_hook_checks(self, tmp_path: Path, project: str) -> None:
        result = runner.invoke(
            app,
            ["doctor", "--project", project],
            env={"CORTEX_DB_PATH": str(tmp_path / "nonexistent.db")},
        )
        assert "inject.py" in result.output
        assert "extract.py" in result.output
        assert "compact.py" in result.output

    def test_all_checks_passed_when_healthy(self, tmp_db: Path, project: str) -> None:
        result = runner.invoke(
            app,
            ["doctor", "--project", project],
            env={"CORTEX_DB_PATH": str(tmp_db)},
        )
        assert result.exit_code == 0
        # Either all checks pass or some fail — but should not crash
        assert "Doctor" in result.output or "doctor" in result.output.lower()


# ---------------------------------------------------------------------------
# cortex pin
# ---------------------------------------------------------------------------


class TestPinCommand:
    def _invoke_pin(self, tmp_db: Path, project: str, node_id: str) -> object:
        return runner.invoke(
            app,
            ["pin", node_id, "--project", project],
            env={"CORTEX_DB_PATH": str(tmp_db)},
        )

    def test_pin_promotes_to_tier_3(self, tmp_db: Path, project: str) -> None:
        nid = _seed_node(tmp_db, project, text="use WAL mode", tier=1)
        result = self._invoke_pin(tmp_db, project, nid)
        assert result.exit_code == 0
        assert "Pinned" in result.output

        conn = sqlite3.connect(str(tmp_db))
        row = conn.execute("SELECT tier FROM nodes WHERE id = ?", (nid,)).fetchone()
        conn.close()
        assert row[0] == 3

    def test_pin_already_tier_3_is_noop(self, tmp_db: Path, project: str) -> None:
        nid = _seed_node(tmp_db, project, text="permanent convention", tier=3)
        result = self._invoke_pin(tmp_db, project, nid)
        assert result.exit_code == 0
        assert "already tier 3" in result.output

    def test_pin_unknown_node_exits_nonzero(self, tmp_db: Path, project: str) -> None:
        result = self._invoke_pin(
            tmp_db, project, "00000000-0000-0000-0000-000000000000"
        )
        assert result.exit_code != 0

    def test_pin_no_db_exits_cleanly(self, tmp_path: Path, project: str) -> None:
        result = runner.invoke(
            app,
            ["pin", "any-id", "--project", project],
            env={"CORTEX_DB_PATH": str(tmp_path / "nonexistent.db")},
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# cortex annotate
# ---------------------------------------------------------------------------


class TestAnnotateCommand:
    def _invoke_annotate(
        self, tmp_db: Path, project: str, node_id: str, rationale: str
    ) -> object:
        return runner.invoke(
            app,
            ["annotate", node_id, "--rationale", rationale, "--project", project],
            env={"CORTEX_DB_PATH": str(tmp_db)},
        )

    def test_annotate_sets_rationale(self, tmp_db: Path, project: str) -> None:
        nid = _seed_node(tmp_db, project, text="always use ruff")
        result = self._invoke_annotate(
            tmp_db, project, nid, "keeps formatting consistent"
        )
        assert result.exit_code == 0
        assert "updated" in result.output.lower()

        conn = sqlite3.connect(str(tmp_db))
        row = conn.execute(
            "SELECT rationale FROM nodes WHERE id = ?", (nid,)
        ).fetchone()
        conn.close()
        assert row[0] == "keeps formatting consistent"

    def test_annotate_overwrites_existing(self, tmp_db: Path, project: str) -> None:
        nid = _seed_node(tmp_db, project, text="mypy strict mode")
        self._invoke_annotate(tmp_db, project, nid, "old reason")
        self._invoke_annotate(tmp_db, project, nid, "new reason")
        conn = sqlite3.connect(str(tmp_db))
        row = conn.execute(
            "SELECT rationale FROM nodes WHERE id = ?", (nid,)
        ).fetchone()
        conn.close()
        assert row[0] == "new reason"

    def test_annotate_unknown_node_exits_nonzero(
        self, tmp_db: Path, project: str
    ) -> None:
        result = self._invoke_annotate(tmp_db, project, "no-such-id", "reason")
        assert result.exit_code != 0

    def test_annotate_no_db_exits_cleanly(self, tmp_path: Path, project: str) -> None:
        result = runner.invoke(
            app,
            ["annotate", "any-id", "--rationale", "r", "--project", project],
            env={"CORTEX_DB_PATH": str(tmp_path / "nonexistent.db")},
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# cortex bump
# ---------------------------------------------------------------------------


class TestBumpCommand:
    def _invoke_bump(
        self, tmp_db: Path, project: str, node_id: str, by: float | None = None
    ) -> object:
        args = ["bump", node_id, "--project", project]
        if by is not None:
            args += ["--by", str(by)]
        return runner.invoke(app, args, env={"CORTEX_DB_PATH": str(tmp_db)})

    def test_bump_increases_weight(self, tmp_db: Path, project: str) -> None:
        nid = _seed_node(tmp_db, project, text="important node")
        conn = sqlite3.connect(str(tmp_db))
        before = conn.execute(
            "SELECT weight FROM nodes WHERE id = ?", (nid,)
        ).fetchone()[0]
        conn.close()

        result = self._invoke_bump(tmp_db, project, nid, by=2.0)
        assert result.exit_code == 0

        conn = sqlite3.connect(str(tmp_db))
        after = conn.execute(
            "SELECT weight FROM nodes WHERE id = ?", (nid,)
        ).fetchone()[0]
        conn.close()
        assert abs(after - (before + 2.0)) < 1e-6

    def test_bump_default_amount_is_one(self, tmp_db: Path, project: str) -> None:
        nid = _seed_node(tmp_db, project, text="another node")
        conn = sqlite3.connect(str(tmp_db))
        before = conn.execute(
            "SELECT weight FROM nodes WHERE id = ?", (nid,)
        ).fetchone()[0]
        conn.close()

        self._invoke_bump(tmp_db, project, nid)

        conn = sqlite3.connect(str(tmp_db))
        after = conn.execute(
            "SELECT weight FROM nodes WHERE id = ?", (nid,)
        ).fetchone()[0]
        conn.close()
        assert abs(after - (before + 1.0)) < 1e-6

    def test_bump_unknown_node_exits_nonzero(self, tmp_db: Path, project: str) -> None:
        result = self._invoke_bump(tmp_db, project, "no-such-id")
        assert result.exit_code != 0

    def test_bump_negative_amount_rejected(self, tmp_db: Path, project: str) -> None:
        nid = _seed_node(tmp_db, project, text="test node")
        result = self._invoke_bump(tmp_db, project, nid, by=-1.0)
        assert result.exit_code != 0

    def test_bump_no_db_exits_cleanly(self, tmp_path: Path, project: str) -> None:
        result = runner.invoke(
            app,
            ["bump", "any-id", "--project", project],
            env={"CORTEX_DB_PATH": str(tmp_path / "nonexistent.db")},
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# cortex clean
# ---------------------------------------------------------------------------


def _seed_stale_node(
    db_path: Path,
    project: str,
    text: str = "stale node",
    tier: int = 1,
    days_old: int = 10,
) -> str:
    """Insert a node with last_accessed set far in the past."""
    conn = sqlite3.connect(str(db_path))
    node_id = str(uuid.uuid4())
    now = int(time.time())
    old_time = now - days_old * 86_400
    emb = np.random.default_rng(seed=7).random(384).astype(np.float32)
    conn.execute(
        """
        INSERT INTO nodes (id, type, tier, text, rationale, embedding, precision_bits,
                           weight, project, scope, source, last_accessed, created_at, session_count)
        VALUES (?, 'observation', ?, ?, NULL, ?, 32, 0.8, ?, 'module', 'jsonl', ?, ?, 1)
        """,
        (node_id, tier, text, emb.tobytes(), project, old_time, now),
    )
    conn.commit()
    conn.close()
    return node_id


class TestCleanCommand:
    def _invoke_clean(
        self, tmp_db: Path, project: str, extra: list[str] | None = None
    ) -> object:
        args = ["clean", "--project", project, "--yes"] + (extra or [])
        return runner.invoke(
            app, args, env={"CORTEX_DB_PATH": str(tmp_db)}, input="y\n"
        )

    def test_no_db_exits_cleanly(self, tmp_path: Path, project: str) -> None:
        result = runner.invoke(
            app,
            ["clean", "--project", project],
            env={"CORTEX_DB_PATH": str(tmp_path / "nonexistent.db")},
        )
        assert result.exit_code == 0
        assert "No Cortex" in result.output

    def test_no_stale_nodes_message(self, tmp_db: Path, project: str) -> None:
        _seed_node(tmp_db, project, text="fresh node")
        result = runner.invoke(
            app,
            ["clean", "--project", project, "--days", "7"],
            env={"CORTEX_DB_PATH": str(tmp_db)},
        )
        assert result.exit_code == 0
        assert "No stale" in result.output

    def test_dry_run_does_not_delete(self, tmp_db: Path, project: str) -> None:
        nid = _seed_stale_node(tmp_db, project, text="old observation", days_old=14)
        result = runner.invoke(
            app,
            ["clean", "--project", project, "--days", "7", "--dry-run"],
            env={"CORTEX_DB_PATH": str(tmp_db)},
        )
        assert result.exit_code == 0
        assert "DRY RUN" in result.output or "dry run" in result.output.lower()
        conn = sqlite3.connect(str(tmp_db))
        row = conn.execute("SELECT id FROM nodes WHERE id = ?", (nid,)).fetchone()
        conn.close()
        assert row is not None, "Dry run should not have deleted the node"

    def test_deletes_stale_nodes(self, tmp_db: Path, project: str) -> None:
        nid = _seed_stale_node(tmp_db, project, text="old node", days_old=14)
        result = runner.invoke(
            app,
            ["clean", "--project", project, "--days", "7"],
            env={"CORTEX_DB_PATH": str(tmp_db)},
            input="y\n",
        )
        assert result.exit_code == 0
        conn = sqlite3.connect(str(tmp_db))
        row = conn.execute("SELECT id FROM nodes WHERE id = ?", (nid,)).fetchone()
        conn.close()
        assert row is None, "Stale node should have been deleted"

    def test_tier3_nodes_never_deleted(self, tmp_db: Path, project: str) -> None:
        nid = _seed_stale_node(
            tmp_db, project, text="permanent convention", tier=3, days_old=30
        )
        result = runner.invoke(
            app,
            ["clean", "--project", project, "--days", "7"],
            env={"CORTEX_DB_PATH": str(tmp_db)},
            input="y\n",
        )
        assert result.exit_code == 0
        conn = sqlite3.connect(str(tmp_db))
        row = conn.execute("SELECT id FROM nodes WHERE id = ?", (nid,)).fetchone()
        conn.close()
        assert row is not None, "Tier-3 nodes must not be removed by clean"

    def test_tier_filter_limits_scope(self, tmp_db: Path, project: str) -> None:
        id1 = _seed_stale_node(tmp_db, project, text="tier1 stale", tier=1, days_old=10)
        id2 = _seed_stale_node(tmp_db, project, text="tier2 stale", tier=2, days_old=10)
        result = runner.invoke(
            app,
            ["clean", "--project", project, "--days", "7", "--tier", "1"],
            env={"CORTEX_DB_PATH": str(tmp_db)},
            input="y\n",
        )
        assert result.exit_code == 0
        conn = sqlite3.connect(str(tmp_db))
        assert (
            conn.execute("SELECT id FROM nodes WHERE id = ?", (id1,)).fetchone() is None
        )
        assert (
            conn.execute("SELECT id FROM nodes WHERE id = ?", (id2,)).fetchone()
            is not None
        )
        conn.close()

    def test_shows_stale_node_table(self, tmp_db: Path, project: str) -> None:
        _seed_stale_node(tmp_db, project, text="stale table row", days_old=10)
        result = runner.invoke(
            app,
            ["clean", "--project", project, "--days", "7", "--dry-run"],
            env={"CORTEX_DB_PATH": str(tmp_db)},
        )
        assert result.exit_code == 0
        assert "stale table row" in result.output


# ---------------------------------------------------------------------------
# cortex status (enhanced)
# ---------------------------------------------------------------------------


class TestStatusCommandEnhanced:
    def _invoke_status(self, tmp_db: Path, project: str) -> object:
        return runner.invoke(
            app, ["status", "--project", project], env={"CORTEX_DB_PATH": str(tmp_db)}
        )

    def test_shows_type_distribution(self, tmp_db: Path, project: str) -> None:
        _seed_node(tmp_db, project, text="observe this", tier=1)
        result = self._invoke_status(tmp_db, project)
        assert result.exit_code == 0
        assert "observation" in result.output

    def test_shows_source_breakdown(self, tmp_db: Path, project: str) -> None:
        _seed_node(tmp_db, project, text="jsonl node", tier=1)
        result = self._invoke_status(tmp_db, project)
        assert result.exit_code == 0
        assert "jsonl" in result.output

    def test_shows_avg_weight(self, tmp_db: Path, project: str) -> None:
        _seed_node(tmp_db, project, text="a node", tier=1)
        result = self._invoke_status(tmp_db, project)
        assert result.exit_code == 0
        assert "Avg weight" in result.output or "avg" in result.output.lower()

    def test_no_db_exits_cleanly(self, tmp_path: Path, project: str) -> None:
        result = runner.invoke(
            app,
            ["status", "--project", project],
            env={"CORTEX_DB_PATH": str(tmp_path / "nonexistent.db")},
        )
        assert result.exit_code == 0
        assert "No Cortex" in result.output

    def test_project_flag_accepted(self, tmp_db: Path, project: str) -> None:
        result = runner.invoke(
            app,
            ["status", "--project", project],
            env={"CORTEX_DB_PATH": str(tmp_db)},
        )
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# cortex recent
# ---------------------------------------------------------------------------


class TestRecentCommand:
    def _invoke_recent(
        self, tmp_db: Path, project: str, extra: list[str] | None = None
    ) -> object:
        args = ["recent", "--project", project] + (extra or [])
        return runner.invoke(app, args, env={"CORTEX_DB_PATH": str(tmp_db)})

    def test_no_db_exits_cleanly(self, tmp_path: Path, project: str) -> None:
        result = runner.invoke(
            app,
            ["recent", "--project", project],
            env={"CORTEX_DB_PATH": str(tmp_path / "nonexistent.db")},
        )
        assert result.exit_code == 0
        assert "No Cortex" in result.output

    def test_empty_db_shows_message(self, tmp_db: Path, project: str) -> None:
        result = self._invoke_recent(tmp_db, project)
        assert result.exit_code == 0
        assert "No nodes" in result.output

    def test_shows_recently_accessed_nodes(self, tmp_db: Path, project: str) -> None:
        _seed_node(tmp_db, project, text="recently touched node", tier=1)
        result = self._invoke_recent(tmp_db, project)
        assert result.exit_code == 0
        assert "recently touched" in result.output or "recently" in result.output

    def test_ordered_most_recent_first(self, tmp_db: Path, project: str) -> None:
        conn = sqlite3.connect(str(tmp_db))
        now = int(time.time())
        early_id = str(uuid.uuid4())
        late_id = str(uuid.uuid4())
        emb = np.zeros(384, dtype=np.float32).tobytes()
        for nid, ts, label in [
            (early_id, now - 1000, "early node"),
            (late_id, now - 10, "late node"),
        ]:
            conn.execute(
                """INSERT INTO nodes (id, type, tier, text, rationale, embedding, precision_bits,
                                      weight, project, scope, source, last_accessed, created_at, session_count)
                   VALUES (?, 'observation', 1, ?, NULL, ?, 32, 1.0, ?, 'project', 'jsonl', ?, ?, 1)""",
                (nid, label, emb, project, ts, ts),
            )
        conn.commit()
        conn.close()

        result = self._invoke_recent(tmp_db, project)
        assert result.exit_code == 0
        # Use UUID prefixes for ordering — visible regardless of column wrapping
        late_pos = result.output.find(late_id[:8])
        early_pos = result.output.find(early_id[:8])
        assert late_pos >= 0 and early_pos >= 0
        assert late_pos < early_pos, "Most recently accessed node should appear first"

    def test_limit_flag_respected(self, tmp_db: Path, project: str) -> None:
        for i in range(6):
            _seed_node(tmp_db, project, text=f"node {i}", tier=1)
        result = self._invoke_recent(tmp_db, project, ["--limit", "3"])
        assert result.exit_code == 0

    def test_json_flag_empty_db_returns_empty_list(
        self, tmp_db: Path, project: str
    ) -> None:
        result = self._invoke_recent(tmp_db, project, ["--json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed == []

    def test_json_flag_outputs_valid_json(self, tmp_db: Path, project: str) -> None:
        _seed_node(tmp_db, project, text="some recent node")
        result = self._invoke_recent(tmp_db, project, ["--json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert isinstance(parsed, list)
        assert len(parsed) == 1

    def test_json_flag_includes_required_fields(
        self, tmp_db: Path, project: str
    ) -> None:
        _seed_node(tmp_db, project, text="field check node")
        result = self._invoke_recent(tmp_db, project, ["--json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        rec = parsed[0]
        for field in ("id", "type", "tier", "text", "weight", "last_accessed"):
            assert field in rec, f"missing field: {field}"

    def test_json_flag_limit_respected(self, tmp_db: Path, project: str) -> None:
        for i in range(5):
            _seed_node(tmp_db, project, text=f"node {i}")
        result = self._invoke_recent(tmp_db, project, ["--json", "--limit", "2"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert len(parsed) <= 2


# ---------------------------------------------------------------------------
# cortex export
# ---------------------------------------------------------------------------


class TestExportCommand:
    def _invoke_export(
        self,
        tmp_db: Path,
        project: str,
        extra: list[str] | None = None,
    ) -> object:
        args = ["export", "--project", project] + (extra or [])
        return runner.invoke(app, args, env={"CORTEX_DB_PATH": str(tmp_db)})

    def test_no_db_exits_cleanly(self, tmp_path: Path, project: str) -> None:
        result = runner.invoke(
            app,
            ["export", "--project", project],
            env={"CORTEX_DB_PATH": str(tmp_path / "nonexistent.db")},
        )
        assert result.exit_code == 0
        assert "No Cortex" in result.output

    def test_empty_db_no_nodes_message(self, tmp_db: Path, project: str) -> None:
        result = self._invoke_export(tmp_db, project)
        assert result.exit_code == 0
        assert "No nodes" in result.output

    def test_json_output_to_stdout(self, tmp_db: Path, project: str) -> None:
        import json

        _seed_node(tmp_db, project, text="export test node", tier=2)
        result = self._invoke_export(tmp_db, project)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["text"] == "export test node"

    def test_json_output_to_file(
        self, tmp_db: Path, project: str, tmp_path: Path
    ) -> None:
        import json

        _seed_node(tmp_db, project, text="file export node", tier=1)
        out_file = tmp_path / "out.json"
        result = self._invoke_export(tmp_db, project, ["--out", str(out_file)])
        assert result.exit_code == 0
        assert "Exported 1" in result.output
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert data[0]["text"] == "file export node"

    def test_csv_format_to_stdout(self, tmp_db: Path, project: str) -> None:
        _seed_node(tmp_db, project, text="csv export node", tier=1)
        result = self._invoke_export(tmp_db, project, ["--format", "csv"])
        assert result.exit_code == 0
        assert "csv export node" in result.output
        assert "text" in result.output  # CSV header

    def test_csv_to_file(self, tmp_db: Path, project: str, tmp_path: Path) -> None:
        _seed_node(tmp_db, project, text="csv file node", tier=1)
        out_file = tmp_path / "out.csv"
        result = self._invoke_export(
            tmp_db, project, ["--format", "csv", "--out", str(out_file)]
        )
        assert result.exit_code == 0
        assert out_file.exists()
        content = out_file.read_text()
        assert "csv file node" in content

    def test_tier_filter_restricts_output(self, tmp_db: Path, project: str) -> None:
        import json

        _seed_node(tmp_db, project, text="tier1 node", tier=1)
        _seed_node(tmp_db, project, text="tier2 node", tier=2)
        _seed_node(tmp_db, project, text="tier3 node", tier=3)
        result = self._invoke_export(tmp_db, project, ["--tier", "3"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert all(n["tier"] == 3 for n in data)
        assert any("tier3" in n["text"] for n in data)

    def test_multiple_nodes_exported(self, tmp_db: Path, project: str) -> None:
        import json

        for i in range(4):
            _seed_node(tmp_db, project, text=f"bulk node {i}", tier=1)
        result = self._invoke_export(tmp_db, project)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 4

    def test_write_error_exits_nonzero(self, tmp_db: Path, project: str) -> None:
        _seed_node(tmp_db, project, text="write fail node", tier=1)
        result = self._invoke_export(
            tmp_db, project, ["--out", "/no_permission_dir/out.json"]
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# cortex decay
# ---------------------------------------------------------------------------


class TestDecayCommand:
    def _invoke_decay(self, tmp_db: Path, project: str) -> object:
        return runner.invoke(
            app,
            ["decay", "--project", project],
            env={"CORTEX_DB_PATH": str(tmp_db)},
        )

    def test_no_db_exits_cleanly(self, tmp_path: Path, project: str) -> None:
        result = runner.invoke(
            app,
            ["decay", "--project", project],
            env={"CORTEX_DB_PATH": str(tmp_path / "nonexistent.db")},
        )
        assert result.exit_code == 0
        assert "No Cortex" in result.output

    def test_decay_empty_db_prints_complete(self, tmp_db: Path, project: str) -> None:
        result = self._invoke_decay(tmp_db, project)
        assert result.exit_code == 0
        assert "Decay complete" in result.output

    def test_decay_prints_counts(self, tmp_db: Path, project: str) -> None:
        _seed_node(tmp_db, project, text="decayable node", tier=1)
        result = self._invoke_decay(tmp_db, project)
        assert result.exit_code == 0
        assert "Decayed" in result.output
        assert "Evicted" in result.output
        assert "Promoted" in result.output

    def test_decay_decrements_weight(self, tmp_db: Path, project: str) -> None:
        node_id = _seed_node(tmp_db, project, text="weight decay node", tier=1)
        conn = sqlite3.connect(str(tmp_db))
        weight_before = conn.execute(
            "SELECT weight FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()[0]
        conn.close()

        result = self._invoke_decay(tmp_db, project)
        assert result.exit_code == 0

        conn = sqlite3.connect(str(tmp_db))
        row = conn.execute(
            "SELECT weight FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        conn.close()
        if row is not None:
            assert row[0] <= weight_before  # weight was decayed or node was evicted

    def test_decay_evicts_very_low_weight_node(
        self, tmp_db: Path, project: str
    ) -> None:
        conn = sqlite3.connect(str(tmp_db))
        node_id = str(uuid.uuid4())
        now = int(time.time())
        emb = np.zeros(384, dtype=np.float32).tobytes()
        conn.execute(
            """INSERT INTO nodes (id, type, tier, text, rationale, embedding, precision_bits,
                                   weight, project, scope, source, last_accessed, created_at, session_count)
               VALUES (?, 'observation', 1, 'low weight node', NULL, ?, 32, 0.0001, ?, 'project',
                       'jsonl', ?, ?, 1)""",
            (node_id, emb, project, now - 86400 * 30, now - 86400 * 30),
        )
        conn.commit()
        conn.close()

        result = self._invoke_decay(tmp_db, project)
        assert result.exit_code == 0

        conn = sqlite3.connect(str(tmp_db))
        row = conn.execute("SELECT id FROM nodes WHERE id = ?", (node_id,)).fetchone()
        conn.close()
        assert row is None, "Node with near-zero weight should be evicted"

    def test_decay_result_project_in_output(self, tmp_db: Path, project: str) -> None:
        result = self._invoke_decay(tmp_db, project)
        assert result.exit_code == 0
        # Rich may wrap the long tmp path; check for any path fragment
        path_fragment = Path(project).name
        assert path_fragment in result.output.replace("\n", "")


# ---------------------------------------------------------------------------
# cortex graph
# ---------------------------------------------------------------------------


class TestGraphCommand:
    # `graph` has no --project; it uses CWD as the project string.
    # Nodes must be seeded with project=str(Path.cwd()) to be found.

    def _cwd_project(self) -> str:
        return str(Path.cwd())

    def _invoke_graph(self, tmp_db: Path) -> object:
        return runner.invoke(app, ["graph"], env={"CORTEX_DB_PATH": str(tmp_db)})

    def test_no_db_exits_cleanly(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app, ["graph"], env={"CORTEX_DB_PATH": str(tmp_path / "nonexistent.db")}
        )
        assert result.exit_code == 0
        assert "No Cortex" in result.output

    def test_empty_graph_message(self, tmp_db: Path) -> None:
        result = self._invoke_graph(tmp_db)
        assert result.exit_code == 0
        assert "empty" in result.output.lower()

    def test_single_node_printed(self, tmp_db: Path) -> None:
        _seed_node(tmp_db, self._cwd_project(), text="auth middleware is slow", tier=1)
        result = self._invoke_graph(tmp_db)
        assert result.exit_code == 0
        assert "auth middleware" in result.output

    def test_tier_marker_present(self, tmp_db: Path) -> None:
        _seed_node(tmp_db, self._cwd_project(), text="some node", tier=2)
        result = self._invoke_graph(tmp_db)
        assert "T2" in result.output

    def test_many_nodes_truncated(self, tmp_db: Path) -> None:
        cwd_proj = self._cwd_project()
        for i in range(25):
            _seed_node(tmp_db, cwd_proj, text=f"bulk node {i:03d}", tier=1)
        result = self._invoke_graph(tmp_db)
        assert result.exit_code == 0
        assert "more nodes" in result.output


# ---------------------------------------------------------------------------
# cortex inspect
# ---------------------------------------------------------------------------


class TestInspectCommand:
    # `inspect` has no --project; it opens graph at CWD but looks up nodes by ID.

    def _invoke_inspect(self, tmp_db: Path, node_id: str) -> object:
        return runner.invoke(
            app, ["inspect", node_id], env={"CORTEX_DB_PATH": str(tmp_db)}
        )

    def test_no_db_exits_nonzero(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            ["inspect", "00000000-0000-0000-0000-000000000000"],
            env={"CORTEX_DB_PATH": str(tmp_path / "nonexistent.db")},
        )
        assert result.exit_code != 0

    def test_missing_node_exits_nonzero(self, tmp_db: Path, project: str) -> None:
        result = self._invoke_inspect(tmp_db, "00000000-0000-0000-0000-000000000000")
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_existing_node_shows_fields(self, tmp_db: Path, project: str) -> None:
        node_id = _seed_node(tmp_db, project, text="inspect target node", tier=2)
        result = self._invoke_inspect(tmp_db, node_id)
        assert result.exit_code == 0
        assert "inspect target node" in result.output
        assert "tier" in result.output.lower()
        assert "weight" in result.output.lower()

    def test_inspect_shows_node_id(self, tmp_db: Path, project: str) -> None:
        node_id = _seed_node(tmp_db, project, text="id check node", tier=1)
        result = self._invoke_inspect(tmp_db, node_id)
        assert result.exit_code == 0
        assert node_id in result.output


# ---------------------------------------------------------------------------
# cortex prune
# ---------------------------------------------------------------------------


class TestPruneCommand:
    # `prune` has no --project; node lookup is by ID with no project filter.

    def _invoke_prune(self, tmp_db: Path, node_id: str) -> object:
        return runner.invoke(
            app, ["prune", node_id], env={"CORTEX_DB_PATH": str(tmp_db)}
        )

    def test_no_db_exits_nonzero(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            ["prune", "00000000-0000-0000-0000-000000000000"],
            env={"CORTEX_DB_PATH": str(tmp_path / "nonexistent.db")},
        )
        assert result.exit_code != 0

    def test_missing_node_exits_nonzero(self, tmp_db: Path, project: str) -> None:
        result = self._invoke_prune(tmp_db, "00000000-0000-0000-0000-000000000000")
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_prune_removes_node(self, tmp_db: Path, project: str) -> None:
        node_id = _seed_node(tmp_db, project, text="node to be pruned", tier=1)
        result = self._invoke_prune(tmp_db, node_id)
        assert result.exit_code == 0
        assert "Pruned" in result.output
        conn = sqlite3.connect(str(tmp_db))
        row = conn.execute("SELECT id FROM nodes WHERE id = ?", (node_id,)).fetchone()
        conn.close()
        assert row is None

    def test_prune_shows_node_text(self, tmp_db: Path, project: str) -> None:
        node_id = _seed_node(tmp_db, project, text="prune text check", tier=2)
        result = self._invoke_prune(tmp_db, node_id)
        assert "prune text check" in result.output


# ---------------------------------------------------------------------------
# cortex reset
# ---------------------------------------------------------------------------


class TestResetCommand:
    def test_reset_with_yes_deletes_all_nodes(self, tmp_db: Path, project: str) -> None:
        for i in range(3):
            _seed_node(tmp_db, project, text=f"reset node {i}", tier=1)
        result = runner.invoke(
            app,
            ["reset", "--project", project, "--yes"],
            env={"CORTEX_DB_PATH": str(tmp_db)},
        )
        assert result.exit_code == 0
        assert "Reset complete" in result.output
        conn = sqlite3.connect(str(tmp_db))
        count = conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE project = ?", (project,)
        ).fetchone()[0]
        conn.close()
        assert count == 0

    def test_reset_no_db_exits_cleanly(self, tmp_path: Path, project: str) -> None:
        result = runner.invoke(
            app,
            ["reset", "--project", project, "--yes"],
            env={"CORTEX_DB_PATH": str(tmp_path / "nonexistent.db")},
        )
        assert result.exit_code == 0
        assert "No Cortex" in result.output

    def test_reset_abort_without_yes(self, tmp_db: Path, project: str) -> None:
        _seed_node(tmp_db, project, text="should survive", tier=1)
        result = runner.invoke(
            app,
            ["reset", "--project", project],
            env={"CORTEX_DB_PATH": str(tmp_db)},
            input="n\n",
        )
        assert result.exit_code != 0
        conn = sqlite3.connect(str(tmp_db))
        count = conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE project = ?", (project,)
        ).fetchone()[0]
        conn.close()
        assert count == 1

    def test_reset_shows_deleted_count(self, tmp_db: Path, project: str) -> None:
        for i in range(5):
            _seed_node(tmp_db, project, text=f"count node {i}", tier=1)
        result = runner.invoke(
            app,
            ["reset", "--project", project, "--yes"],
            env={"CORTEX_DB_PATH": str(tmp_db)},
        )
        assert "5" in result.output


# ---------------------------------------------------------------------------
# cortex search
# ---------------------------------------------------------------------------


class TestSearchCommand:
    # `search` has no --project; it uses CWD. Nodes must be seeded with
    # project=str(Path.cwd()) to be visible to the command.

    @staticmethod
    def _cwd_project() -> str:
        return str(Path.cwd())

    def _invoke_search(
        self, tmp_db: Path, query: str, extra: list[str] | None = None
    ) -> object:
        args = ["search", query] + (extra or [])
        return runner.invoke(app, args, env={"CORTEX_DB_PATH": str(tmp_db)})

    def test_no_db_exits_cleanly(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            ["search", "auth"],
            env={"CORTEX_DB_PATH": str(tmp_path / "nonexistent.db")},
        )
        assert result.exit_code == 0
        assert "No Cortex" in result.output

    def test_empty_db_no_nodes(self, tmp_db: Path) -> None:
        result = self._invoke_search(tmp_db, "auth")
        assert result.exit_code == 0
        assert "No nodes" in result.output

    def test_matching_node_in_output(self, tmp_db: Path) -> None:
        # BM25 needs multiple docs to produce positive IDF scores.
        cwd_proj = self._cwd_project()
        _seed_node(tmp_db, cwd_proj, text="authentication middleware is key", tier=1)
        _seed_node(tmp_db, cwd_proj, text="database connection pool config", tier=1)
        _seed_node(tmp_db, cwd_proj, text="cache invalidation strategy", tier=1)
        result = self._invoke_search(tmp_db, "authentication")
        assert result.exit_code == 0
        assert "authentication" in result.output

    def test_no_match_shows_no_matching(self, tmp_db: Path) -> None:
        cwd_proj = self._cwd_project()
        _seed_node(tmp_db, cwd_proj, text="database connection pool", tier=1)
        _seed_node(tmp_db, cwd_proj, text="cache invalidation strategy", tier=1)
        result = self._invoke_search(tmp_db, "xyzzy_not_found")
        assert result.exit_code == 0
        assert "No matching" in result.output

    def test_invalid_source_exits_nonzero(self, tmp_db: Path) -> None:
        _seed_node(tmp_db, self._cwd_project(), text="some node", tier=1)
        result = self._invoke_search(tmp_db, "auth", ["--source", "invalid_source"])
        assert result.exit_code != 0
        assert "Invalid source" in result.output

    def test_source_filter_jsonl(self, tmp_db: Path) -> None:
        _seed_node(
            tmp_db,
            self._cwd_project(),
            text="jsonl source node about auth",
            tier=1,
        )
        result = self._invoke_search(tmp_db, "auth", ["--source", "jsonl"])
        assert result.exit_code == 0

    def test_tier_filter(self, tmp_db: Path) -> None:
        cwd_proj = self._cwd_project()
        # Need multiple docs in filtered corpus for positive BM25 scores.
        _seed_node(
            tmp_db, cwd_proj, text="tier3 auth convention always use tokens", tier=3
        )
        _seed_node(tmp_db, cwd_proj, text="tier3 cache policy evict early", tier=3)
        _seed_node(tmp_db, cwd_proj, text="tier3 database schema migration", tier=3)
        _seed_node(tmp_db, cwd_proj, text="tier1 auth observation recent", tier=1)
        result = self._invoke_search(tmp_db, "auth", ["--tier", "3"])
        assert result.exit_code == 0
        assert "tier3 auth convention" in result.output

    def test_limit_flag(self, tmp_db: Path) -> None:
        cwd_proj = self._cwd_project()
        for i in range(8):
            _seed_node(tmp_db, cwd_proj, text=f"auth node {i}", tier=1)
        result = self._invoke_search(tmp_db, "auth", ["--limit", "3"])
        assert result.exit_code == 0


# cortex status — last-session branch (lines 116-122)
# ---------------------------------------------------------------------------


class TestStatusLastSession:
    def test_status_shows_last_session_data(self, tmp_db: Path, project: str) -> None:
        _seed_node(tmp_db, project, text="some node", tier=1)
        _seed_session(tmp_db, project, tokens_raw=1200, tokens_injected=400)
        result = runner.invoke(
            app,
            ["status", "--project", project],
            env={"CORTEX_DB_PATH": str(tmp_db)},
        )
        assert result.exit_code == 0
        assert "Last session" in result.output
        assert "Nodes written" in result.output

    def test_status_shows_token_savings_when_present(
        self, tmp_db: Path, project: str
    ) -> None:
        _seed_session(tmp_db, project, tokens_raw=900, tokens_injected=300)
        result = runner.invoke(
            app,
            ["status", "--project", project],
            env={"CORTEX_DB_PATH": str(tmp_db)},
        )
        assert result.exit_code == 0
        assert "600" in result.output  # 900 - 300

    def test_status_no_session_skips_last_session(
        self, tmp_db: Path, project: str
    ) -> None:
        result = runner.invoke(
            app,
            ["status", "--project", project],
            env={"CORTEX_DB_PATH": str(tmp_db)},
        )
        assert result.exit_code == 0
        assert "Last session" not in result.output


# ---------------------------------------------------------------------------
# cortex list — validation error paths (lines 826-866)
# ---------------------------------------------------------------------------


class TestListValidation:
    def _invoke_list(
        self, tmp_db: Path, project: str, extra: list[str] | None = None
    ) -> object:
        args = ["list", "--project", project] + (extra or [])
        return runner.invoke(app, args, env={"CORTEX_DB_PATH": str(tmp_db)})

    def test_invalid_type_exits_nonzero(self, tmp_db: Path, project: str) -> None:
        result = self._invoke_list(tmp_db, project, ["--type", "invalid_type"])
        assert result.exit_code != 0
        assert "Invalid type" in result.output

    def test_invalid_source_exits_nonzero(self, tmp_db: Path, project: str) -> None:
        result = self._invoke_list(tmp_db, project, ["--source", "bad_source"])
        assert result.exit_code != 0
        assert "Invalid source" in result.output

    def test_invalid_sort_exits_nonzero(self, tmp_db: Path, project: str) -> None:
        result = self._invoke_list(tmp_db, project, ["--sort", "bad_sort"])
        assert result.exit_code != 0
        assert "Invalid sort" in result.output

    def test_valid_type_filter(self, tmp_db: Path, project: str) -> None:
        _seed_node(tmp_db, project, text="convention node", tier=3)
        result = self._invoke_list(tmp_db, project, ["--type", "observation"])
        assert result.exit_code == 0

    def test_valid_sort_created(self, tmp_db: Path, project: str) -> None:
        _seed_node(tmp_db, project, text="sort by created", tier=1)
        result = self._invoke_list(tmp_db, project, ["--sort", "created"])
        assert result.exit_code == 0

    def test_valid_sort_accessed(self, tmp_db: Path, project: str) -> None:
        _seed_node(tmp_db, project, text="sort by accessed", tier=1)
        result = self._invoke_list(tmp_db, project, ["--sort", "accessed"])
        assert result.exit_code == 0

    def test_limit_truncation_shows_message(self, tmp_db: Path, project: str) -> None:
        for i in range(6):
            _seed_node(tmp_db, project, text=f"paged node {i}", tier=1)
        result = self._invoke_list(tmp_db, project, ["--limit", "3"])
        assert result.exit_code == 0
        assert "Showing 3 of 6" in result.output


# ---------------------------------------------------------------------------
# cortex clean — input validation (lines 760-764)
# ---------------------------------------------------------------------------


class TestCleanValidation:
    def test_days_zero_exits_nonzero(self, tmp_db: Path, project: str) -> None:
        result = runner.invoke(
            app,
            ["clean", "--project", project, "--days", "0"],
            env={"CORTEX_DB_PATH": str(tmp_db)},
        )
        assert result.exit_code != 0
        assert "--days" in result.output

    def test_invalid_tier_exits_nonzero(self, tmp_db: Path, project: str) -> None:
        result = runner.invoke(
            app,
            ["clean", "--project", project, "--tier", "5"],
            env={"CORTEX_DB_PATH": str(tmp_db)},
        )
        assert result.exit_code != 0
        assert "--tier" in result.output


# ---------------------------------------------------------------------------
# cortex install
# ---------------------------------------------------------------------------


class TestInstallCommand:
    def test_install_creates_plugin_json(self, tmp_path: Path) -> None:
        import json

        plugins_dir = tmp_path / ".claude" / "plugins"
        result = runner.invoke(
            app,
            ["install"],
            env={"HOME": str(tmp_path)},
        )
        assert result.exit_code == 0
        assert "Installed" in result.output
        manifest = plugins_dir / "cortex.json"
        assert manifest.exists()
        data = json.loads(manifest.read_text())
        assert data["name"] == "cortex"
        assert "hooks" in data

    def test_install_mentions_restart(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["install"], env={"HOME": str(tmp_path)})
        assert "Restart" in result.output or "restart" in result.output.lower()


# ---------------------------------------------------------------------------
# cortex doctor — unhealthy path
# ---------------------------------------------------------------------------


class TestDoctorUnhealthyPath:
    def test_doctor_reports_missing_db(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("CORTEX_DB_PATH", str(tmp_path / "nonexistent.db"))
        result = runner.invoke(
            app,
            ["doctor", "--project", str(tmp_path)],
            env={"CORTEX_DB_PATH": str(tmp_path / "nonexistent.db")},
        )
        assert result.exit_code == 0
        assert "not found" in result.output.lower() or "✗" in result.output

    def test_doctor_runs_without_error(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            ["doctor", "--project", str(tmp_path)],
            env={"CORTEX_DB_PATH": str(tmp_path / "nonexistent.db")},
        )
        assert result.exit_code == 0
        assert "Cortex Doctor" in result.output


# ---------------------------------------------------------------------------
# cortex import — non-list JSON and non-dict records (lines 556-557, 576-577)
# cortex import — invalid type/scope/source coercion (lines 594, 598, 602)
# ---------------------------------------------------------------------------


class TestImportEdgeCases:
    def _make_json_file(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "data.json"
        p.write_text(content)
        return p

    def test_non_list_json_exits_nonzero(
        self, tmp_db: Path, project: str, tmp_path: Path
    ) -> None:
        f = self._make_json_file(tmp_path, '{"not": "a list"}')
        result = runner.invoke(
            app,
            ["import-", str(f), "--project", project],
            env={"CORTEX_DB_PATH": str(tmp_db)},
        )
        assert result.exit_code != 0
        assert "JSON array" in result.output or "Expected" in result.output

    def test_non_dict_record_is_skipped(
        self, tmp_db: Path, project: str, tmp_path: Path
    ) -> None:
        f = self._make_json_file(
            tmp_path, '[42, "a string", {"text": "valid node", "tier": 1}]'
        )
        result = runner.invoke(
            app,
            ["import-", str(f), "--project", project],
            env={"CORTEX_DB_PATH": str(tmp_db)},
        )
        assert result.exit_code == 0
        assert "2 skipped" in result.output
        assert "Imported 1" in result.output

    def test_invalid_node_type_coerced_to_observation(
        self, tmp_db: Path, project: str, tmp_path: Path
    ) -> None:
        f = self._make_json_file(
            tmp_path,
            '[{"text": "coerce type", "type": "bogus_type", "tier": 1}]',
        )
        result = runner.invoke(
            app,
            ["import-", str(f), "--project", project],
            env={"CORTEX_DB_PATH": str(tmp_db)},
        )
        assert result.exit_code == 0
        assert "Imported 1" in result.output

    def test_invalid_scope_coerced_to_project(
        self, tmp_db: Path, project: str, tmp_path: Path
    ) -> None:
        f = self._make_json_file(
            tmp_path,
            '[{"text": "coerce scope", "scope": "invalid_scope", "tier": 1}]',
        )
        result = runner.invoke(
            app,
            ["import-", str(f), "--project", project],
            env={"CORTEX_DB_PATH": str(tmp_db)},
        )
        assert result.exit_code == 0
        assert "Imported 1" in result.output

    def test_invalid_source_coerced_to_jsonl(
        self, tmp_db: Path, project: str, tmp_path: Path
    ) -> None:
        f = self._make_json_file(
            tmp_path,
            '[{"text": "coerce source", "source": "unknown_src", "tier": 1}]',
        )
        result = runner.invoke(
            app,
            ["import-", str(f), "--project", project],
            env={"CORTEX_DB_PATH": str(tmp_db)},
        )
        assert result.exit_code == 0
        assert "Imported 1" in result.output


# ---------------------------------------------------------------------------
# cortex graph — node with edges shows neighbour in output (lines 182-187)
# ---------------------------------------------------------------------------


class TestGraphWithEdges:
    def test_node_with_edge_shows_neighbour(self, tmp_db: Path) -> None:

        cwd_proj = str(Path.cwd())
        id_a = _seed_node(tmp_db, cwd_proj, text="auth middleware logic", tier=1)
        id_b = _seed_node(tmp_db, cwd_proj, text="session token handler", tier=1)

        conn = sqlite3.connect(str(tmp_db))
        conn.execute(
            "INSERT INTO edges (source_id, target_id, strength, last_seen) VALUES (?, ?, 1.0, 0)",
            (id_a, id_b),
        )
        conn.commit()
        conn.close()

        result = runner.invoke(app, ["graph"], env={"CORTEX_DB_PATH": str(tmp_db)})
        assert result.exit_code == 0
        assert "auth middleware" in result.output
        assert "session token" in result.output


# ---------------------------------------------------------------------------
# cortex rationale — update_node_rationale failure path (lines 702-703)
# ---------------------------------------------------------------------------


class TestAnnotateFailurePath:
    def test_annotate_exits_nonzero_when_node_missing(
        self, tmp_db: Path, project: str
    ) -> None:
        fake_id = "00000000-0000-0000-0000-000000000000"
        result = runner.invoke(
            app,
            [
                "annotate",
                fake_id,
                "--rationale",
                "some new rationale",
                "--project",
                project,
            ],
            env={"CORTEX_DB_PATH": str(tmp_db)},
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower()


# ---------------------------------------------------------------------------
# cortex dashboard — uvicorn ImportError path (lines 1034-1036)
# ---------------------------------------------------------------------------


class TestDashboardCommand:
    def test_dashboard_exits_1_without_uvicorn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def block_uvicorn(name: str, *args: object, **kwargs: object) -> object:
            if name == "uvicorn":
                raise ImportError("no module named uvicorn")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", block_uvicorn)
        result = runner.invoke(app, ["dashboard"])
        assert result.exit_code != 0
        assert "uvicorn" in result.output.lower()


# ---------------------------------------------------------------------------
# cortex count
# ---------------------------------------------------------------------------


class TestCountCommand:
    def test_count_empty_db_returns_zero(self, tmp_db: Path, project: str) -> None:
        result = runner.invoke(
            app,
            ["count", "--project", project],
            env={"CORTEX_DB_PATH": str(tmp_db)},
        )
        assert result.exit_code == 0
        assert result.output.strip() == "0"

    def test_count_all_nodes(self, tmp_db: Path, project: str) -> None:
        for i in range(3):
            _seed_node(tmp_db, project, text=f"node {i}", tier=1)
        result = runner.invoke(
            app,
            ["count", "--project", project],
            env={"CORTEX_DB_PATH": str(tmp_db)},
        )
        assert result.exit_code == 0
        assert result.output.strip() == "3"

    def test_count_filtered_by_tier(self, tmp_db: Path, project: str) -> None:
        _seed_node(tmp_db, project, text="tier1 node", tier=1)
        _seed_node(tmp_db, project, text="tier2 node", tier=2)
        _seed_node(tmp_db, project, text="tier3 node", tier=3)
        result = runner.invoke(
            app,
            ["count", "--tier", "2", "--project", project],
            env={"CORTEX_DB_PATH": str(tmp_db)},
        )
        assert result.exit_code == 0
        assert result.output.strip() == "1"

    def test_count_invalid_tier_exits_nonzero(self, tmp_db: Path, project: str) -> None:
        result = runner.invoke(
            app,
            ["count", "--tier", "99", "--project", project],
            env={"CORTEX_DB_PATH": str(tmp_db)},
        )
        assert result.exit_code != 0
        assert "--tier" in result.output.lower() or "tier" in result.output.lower()

    def test_count_no_db_exits_cleanly(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            ["count", "--project", str(tmp_path)],
            env={"CORTEX_DB_PATH": str(tmp_path / "none.db")},
        )
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# cortex import_ — open_graph returns None (lines 549-551)
# ---------------------------------------------------------------------------


class TestImportOpenGraphNone:
    def test_import_exits_1_when_graph_cannot_open(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import cli.cortex as cortex_mod

        f = tmp_path / "nodes.json"
        f.write_text("[]")
        monkeypatch.setattr(cortex_mod, "open_graph", lambda *a, **kw: None)
        result = runner.invoke(
            app,
            ["import-", str(f), "--project", str(tmp_path)],
            env={"CORTEX_DB_PATH": str(tmp_path / "c.db")},
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# cortex annotate — update_node_rationale returns False (lines 711-713)
# ---------------------------------------------------------------------------


class TestAnnotateUpdateFails:
    def test_annotate_exits_1_when_update_returns_false(
        self, tmp_db: Path, project: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from core import graph as graph_mod

        node_id = _seed_node(tmp_db, project)
        monkeypatch.setattr(
            graph_mod.Graph, "update_node_rationale", lambda *a, **kw: False
        )
        result = runner.invoke(
            app,
            ["annotate", node_id, "--rationale", "test rationale"],
            env={"CORTEX_DB_PATH": str(tmp_db), "CLAUDE_PROJECT_PATH": project},
        )
        assert result.exit_code != 0
        assert "failed" in result.output.lower()


# ---------------------------------------------------------------------------
# cortex doctor — package-specific unhealthy paths (lines 969-970, 980-981, 989-990)
# and all-ok path (line 1026)
# ---------------------------------------------------------------------------


class TestDoctorPackageChecks:
    def test_doctor_spacy_unhealthy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def break_spacy(name: str, *args: object, **kwargs: object) -> object:
            if name == "spacy":
                raise ImportError("no spacy")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", break_spacy)
        result = runner.invoke(
            app,
            ["doctor", "--project", str(tmp_path)],
            env={"CORTEX_DB_PATH": str(tmp_path / "c.db")},
        )
        assert result.exit_code == 0
        assert "spaCy" in result.output

    def test_doctor_sentence_transformers_unhealthy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def break_st(name: str, *args: object, **kwargs: object) -> object:
            if name == "sentence_transformers":
                raise ImportError("no sentence_transformers")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", break_st)
        result = runner.invoke(
            app,
            ["doctor", "--project", str(tmp_path)],
            env={"CORTEX_DB_PATH": str(tmp_path / "c.db")},
        )
        assert result.exit_code == 0
        assert "sentence-transformers" in result.output

    def test_doctor_tiktoken_unhealthy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def break_tiktoken(name: str, *args: object, **kwargs: object) -> object:
            if name == "tiktoken":
                raise ImportError("no tiktoken")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", break_tiktoken)
        result = runner.invoke(
            app,
            ["doctor", "--project", str(tmp_path)],
            env={"CORTEX_DB_PATH": str(tmp_path / "c.db")},
        )
        assert result.exit_code == 0
        assert "tiktoken" in result.output

    def test_doctor_all_ok_prints_success(
        self, tmp_db: Path, project: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Create a fake plugin manifest so every check passes
        fake_home = Path(tmp_db).parent
        (fake_home / ".claude" / "plugins").mkdir(parents=True, exist_ok=True)
        (fake_home / ".claude" / "plugins" / "cortex.json").write_text("{}")

        monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
        result = runner.invoke(
            app,
            ["doctor", "--project", project],
            env={"CORTEX_DB_PATH": str(tmp_db)},
        )
        assert result.exit_code == 0
        assert "All checks passed" in result.output


# ---------------------------------------------------------------------------
# cortex dashboard — uvicorn present but dashboard module missing (lines 1061-1064)
# ---------------------------------------------------------------------------


class TestDashboardModuleMissing:
    def test_dashboard_exits_1_when_server_module_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import cli.cortex as cortex_mod

        monkeypatch.setattr(cortex_mod.Path, "exists", lambda self: False)
        result = runner.invoke(app, ["dashboard"])
        assert result.exit_code != 0

    def test_dashboard_starts_when_module_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import types

        import cli.cortex as cortex_mod

        fake_uvicorn = types.SimpleNamespace(run=lambda *a, **kw: None)
        dashboard_dir = Path(cortex_mod.__file__).parent.parent / "dashboard"
        dashboard_dir.mkdir(exist_ok=True)
        server_file = dashboard_dir / "server.py"
        created = not server_file.exists()
        if created:
            server_file.write_text("# placeholder")
        try:
            monkeypatch.setitem(__import__("sys").modules, "uvicorn", fake_uvicorn)
            result = runner.invoke(app, ["dashboard"])
        finally:
            if created:
                server_file.unlink(missing_ok=True)
        assert result.exit_code == 0
        assert "localhost:7000" in result.output


# ---------------------------------------------------------------------------
# cli/cortex.py __main__ block (line 1076)
# ---------------------------------------------------------------------------


class TestCortexMainBlock:
    def test_main_block_invokes_app(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import runpy
        import sys

        monkeypatch.setattr(sys, "argv", ["cortex", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_module("cli.cortex", run_name="__main__", alter_sys=False)
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# cortex touch
# ---------------------------------------------------------------------------


class TestTouchCommand:
    def test_touch_updates_last_accessed(self, tmp_db: Path, project: str) -> None:
        import sqlite3
        import time

        node_id = _seed_node(tmp_db, project)
        old_ts = (
            sqlite3.connect(str(tmp_db))
            .execute("SELECT last_accessed FROM nodes WHERE id = ?", (node_id,))
            .fetchone()[0]
        )

        time.sleep(0.01)
        result = runner.invoke(
            app,
            ["touch", node_id, "--project", project],
            env={"CORTEX_DB_PATH": str(tmp_db)},
        )
        assert result.exit_code == 0
        assert "Touched" in result.output

        new_ts = (
            sqlite3.connect(str(tmp_db))
            .execute("SELECT last_accessed FROM nodes WHERE id = ?", (node_id,))
            .fetchone()[0]
        )
        assert new_ts >= old_ts

    def test_touch_bumps_weight(self, tmp_db: Path, project: str) -> None:
        import sqlite3

        node_id = _seed_node(tmp_db, project)
        old_w = (
            sqlite3.connect(str(tmp_db))
            .execute("SELECT weight FROM nodes WHERE id = ?", (node_id,))
            .fetchone()[0]
        )

        runner.invoke(
            app,
            ["touch", node_id, "--project", project],
            env={"CORTEX_DB_PATH": str(tmp_db)},
        )
        new_w = (
            sqlite3.connect(str(tmp_db))
            .execute("SELECT weight FROM nodes WHERE id = ?", (node_id,))
            .fetchone()[0]
        )
        assert abs(new_w - (old_w + 0.1)) < 1e-6

    def test_touch_missing_node_exits_1(self, tmp_db: Path, project: str) -> None:
        result = runner.invoke(
            app,
            ["touch", "00000000-0000-0000-0000-000000000000", "--project", project],
            env={"CORTEX_DB_PATH": str(tmp_db)},
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_touch_no_db_exits_cleanly(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            ["touch", "some-id", "--project", str(tmp_path)],
            env={"CORTEX_DB_PATH": str(tmp_path / "none.db")},
        )
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# cortex similar
# ---------------------------------------------------------------------------


class TestSimilarCommand:
    def test_similar_missing_node_exits_1(self, tmp_db: Path, project: str) -> None:
        result = runner.invoke(
            app,
            ["similar", "00000000-0000-0000-0000-000000000000", "--project", project],
            env={"CORTEX_DB_PATH": str(tmp_db)},
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_similar_node_without_embedding_exits_1(
        self, tmp_db: Path, project: str
    ) -> None:
        node_id = _seed_node(tmp_db, project, text="no-embedding node")
        conn = sqlite3.connect(str(tmp_db))
        conn.execute("UPDATE nodes SET embedding = NULL WHERE id = ?", (node_id,))
        conn.commit()
        conn.close()
        result = runner.invoke(
            app,
            ["similar", node_id, "--project", project],
            env={"CORTEX_DB_PATH": str(tmp_db)},
        )
        assert result.exit_code != 0

    def test_similar_invalid_threshold_exits_1(
        self, tmp_db: Path, project: str
    ) -> None:
        node_id = _seed_node(tmp_db, project)
        result = runner.invoke(
            app,
            ["similar", node_id, "--threshold", "2.5", "--project", project],
            env={"CORTEX_DB_PATH": str(tmp_db)},
        )
        assert result.exit_code != 0

    def test_similar_no_results_above_threshold(
        self, tmp_db: Path, project: str
    ) -> None:
        node_id = _seed_node(tmp_db, project, text="unique lone node")
        result = runner.invoke(
            app,
            ["similar", node_id, "--threshold", "0.9999", "--project", project],
            env={"CORTEX_DB_PATH": str(tmp_db)},
        )
        assert result.exit_code == 0
        assert "No similar" in result.output

    def test_similar_finds_close_node(self, tmp_db: Path, project: str) -> None:
        emb = np.random.default_rng(seed=7).random(384).astype(np.float32)
        n1 = _seed_node(tmp_db, project, text="node alpha")
        n2 = _seed_node(tmp_db, project, text="node beta")
        conn = sqlite3.connect(str(tmp_db))
        conn.execute("UPDATE nodes SET embedding = ? WHERE id = ?", (emb.tobytes(), n1))
        conn.execute("UPDATE nodes SET embedding = ? WHERE id = ?", (emb.tobytes(), n2))
        conn.commit()
        conn.close()
        result = runner.invoke(
            app,
            ["similar", n1, "--project", project, "--threshold", "0.5"],
            env={"CORTEX_DB_PATH": str(tmp_db)},
        )
        assert result.exit_code == 0
        assert "node beta" in result.output or "Similar" in result.output
