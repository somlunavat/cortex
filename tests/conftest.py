"""Shared pytest fixtures for all Cortex tests."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest

TEST_PROJECT = "/tmp/cortex_test_project"


@pytest.fixture
def db() -> Generator[sqlite3.Connection, None, None]:
    """In-memory SQLite connection with schema applied."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    schema = (Path(__file__).parent.parent / "schema.sql").read_text()
    conn.executescript(schema)
    yield conn
    conn.close()


@pytest.fixture
def dummy_embedding() -> np.ndarray:
    """Deterministic dummy 384-dim embedding for tests."""
    rng = np.random.default_rng(seed=42)
    return rng.random(384).astype(np.float32)
