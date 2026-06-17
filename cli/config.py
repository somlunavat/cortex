"""Runtime configuration: decay rates, thresholds, and filesystem paths."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

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
        return Path(override)
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
    conn.executescript(SCHEMA_PATH.read_text())
    _apply_migrations(conn)
    return Graph(connection=conn)


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Apply schema migrations that cannot be expressed as idempotent DDL.

    Each migration is guarded by a column-existence check so this function
    is safe to call on every database open.
    """
    existing_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
    }
    if "nodes_promoted" not in existing_cols:
        conn.execute(
            "ALTER TABLE sessions ADD COLUMN nodes_promoted INTEGER NOT NULL DEFAULT 0"
        )
        conn.commit()
