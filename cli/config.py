"""Runtime configuration: decay rates, thresholds, and filesystem paths."""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Decay constants — single source of truth (mirrors core/decay.py)
# ---------------------------------------------------------------------------

TIER1_DECAY_RATE: float = 0.85
TIER2_DECAY_RATE: float = 0.95

TIER1_EVICTION_THRESHOLD: float = 0.3
TIER2_EVICTION_THRESHOLD: float = 0.5

TIER1_PROMOTION_WEIGHT: float = 8.0
TIER1_PROMOTION_SESSIONS: int = 3

TIER2_PROMOTION_WEIGHT: float = 20.0
TIER2_PROMOTION_SESSIONS: int = 8
TIER2_PROMOTION_AGE_DAYS: int = 14

# ---------------------------------------------------------------------------
# Retrieval constants
# ---------------------------------------------------------------------------

TOKEN_BUDGET: int = 600
TOP_K: int = 8

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
