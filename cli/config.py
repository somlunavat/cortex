"""Runtime configuration: decay rates, thresholds, and filesystem paths."""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from core.graph import Graph

# ---------------------------------------------------------------------------
# Filesystem paths
# ---------------------------------------------------------------------------

CORTEX_DIR_NAME: str = ".cortex"
DB_FILE_NAME: str = "cortex.db"
SESSIONS_DIR_NAME: str = "sessions"

_PACKAGE_ROOT: Path = Path(__file__).parent.parent
SCHEMA_PATH: Path = _PACKAGE_ROOT / "schema.sql"


def cortex_dir(project_root: Path) -> Path:
    """Return the .cortex directory for a project root."""
    return project_root / CORTEX_DIR_NAME


def db_path(project_root: Path) -> Path:
    """Return the cortex.db path for a project root.

    Respects the CORTEX_DB_PATH environment variable if set.
    """
    override = os.environ.get("CORTEX_DB_PATH", "")
    if override:
        override_path = Path(override)
        if not override_path.is_absolute():
            raise ValueError(
                f"CORTEX_DB_PATH must be an absolute path, got: {override!r}"
            )
        logger.warning("CORTEX_DB_PATH override active: %s", override_path)
        return override_path
    return cortex_dir(project_root) / DB_FILE_NAME


def sessions_dir(project_root: Path) -> Path:
    """Return the sessions transcript directory for a project root."""
    return cortex_dir(project_root) / SESSIONS_DIR_NAME


def open_graph(project_root: Path, *, create: bool = False) -> Graph | None:
    """Open (or optionally create) the Cortex SQLite database for a project.

    Args:
        project_root: Absolute path to the project directory.
        create: If True, create the database and parent directories when they
            don't exist. If False, return None instead.

    Returns:
        An open Graph instance, or None if the database doesn't exist and
        create=False.
    """
    from core.graph import Graph

    path = db_path(project_root)
    if not path.exists():
        if not create:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    # Enable FK enforcement so ON DELETE CASCADE on edges fires correctly.
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_PATH.read_text())
    _apply_migrations(conn)
    return Graph(connection=conn)


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Apply schema migrations that cannot be expressed as idempotent DDL.

    Uses PRAGMA user_version as a migration counter so each migration runs
    exactly once per database, not on every open. Column-existence checks
    are kept as extra guards for databases created by older code that never
    set user_version.
    """
    version: int = conn.execute("PRAGMA user_version").fetchone()[0]

    # Migration 1: add nodes_promoted column to sessions
    if version < 1:
        existing_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
        }
        if "nodes_promoted" not in existing_cols:
            conn.execute(
                "ALTER TABLE sessions ADD COLUMN nodes_promoted INTEGER NOT NULL DEFAULT 0"
            )
        conn.commit()

    # Migration 2: purge dangling edges left by connections that ran without
    # PRAGMA foreign_keys = ON (guard on table existence for bare test schemas).
    if version < 2:
        has_edges = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='edges'"
        ).fetchone()
        if has_edges:
            conn.execute("""
                DELETE FROM edges
                WHERE source_id NOT IN (SELECT id FROM nodes)
                   OR target_id NOT IN (SELECT id FROM nodes)
                """)
        conn.commit()

    if version < 2:
        conn.execute("PRAGMA user_version = 2")
