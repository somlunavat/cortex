"""SQLite-backed knowledge graph for Cortex memory nodes and edges.

No ORM. Raw SQL only. schema.sql is the single source of truth.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass

import numpy as np


@dataclass
class Node:
    """A memory unit in the Cortex knowledge graph."""

    id: str
    type: str
    tier: int
    text: str
    rationale: str | None
    embedding: np.ndarray | None
    precision_bits: int
    weight: float
    project: str
    scope: str
    source: str
    last_accessed: int
    created_at: int
    session_count: int


@dataclass
class Edge:
    """A weighted relationship between two nodes."""

    source_id: str
    target_id: str
    strength: float
    last_seen: int


def _serialize_embedding(arr: np.ndarray) -> bytes:
    """Serialize a numpy array to bytes for SQLite BLOB storage."""
    return arr.tobytes()


def _deserialize_embedding(blob: bytes, precision_bits: int = 32) -> np.ndarray:
    """Deserialize a SQLite BLOB back to a numpy array."""
    match precision_bits:
        case 32:
            return np.frombuffer(blob, dtype=np.float32).copy()
        case 8 | 2:
            return np.frombuffer(blob, dtype=np.int8).copy()
        case _:
            return np.frombuffer(blob, dtype=np.float32).copy()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a_f = a.astype(np.float32)
    b_f = b.astype(np.float32)
    norm_a = float(np.linalg.norm(a_f))
    norm_b = float(np.linalg.norm(b_f))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a_f, b_f) / (norm_a * norm_b))


def _row_to_node(row: sqlite3.Row) -> Node:
    """Convert a sqlite3.Row to a Node dataclass."""
    embedding: np.ndarray | None = None
    if row["embedding"] is not None:
        embedding = _deserialize_embedding(row["embedding"], row["precision_bits"])
    return Node(
        id=row["id"],
        type=row["type"],
        tier=row["tier"],
        text=row["text"],
        rationale=row["rationale"],
        embedding=embedding,
        precision_bits=row["precision_bits"],
        weight=row["weight"],
        project=row["project"],
        scope=row["scope"],
        source=row["source"],
        last_accessed=row["last_accessed"],
        created_at=row["created_at"],
        session_count=row["session_count"],
    )


class Graph:
    """Wraps a sqlite3.Connection and exposes the Cortex graph API.

    All methods are synchronous. No ORM — raw SQL only.
    Connection injection allows in-memory SQLite in tests.
    """

    def __init__(self, connection: sqlite3.Connection) -> None:
        """Initialize Graph with an open SQLite connection.

        Args:
            connection: An open sqlite3.Connection with the Cortex schema applied.
        """
        self._conn = connection
        self._conn.row_factory = sqlite3.Row

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def write_node(self, node: Node) -> str:
        """Insert a new node into the graph and return its UUID.

        A fresh UUID4 is always generated, ignoring any id already set on
        the node dataclass. This guarantees uniqueness on every write.

        Args:
            node: Node dataclass populated with all required fields.

        Returns:
            The string UUID4 assigned to the persisted node.
        """
        node_id = str(uuid.uuid4())
        now = int(time.time())
        embedding_blob: bytes | None = None
        if node.embedding is not None:
            embedding_blob = _serialize_embedding(node.embedding.astype(np.float32))
        self._conn.execute(
            """
            INSERT INTO nodes (
                id, type, tier, text, rationale, embedding, precision_bits,
                weight, project, scope, source, last_accessed, created_at,
                session_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                node_id,
                node.type,
                node.tier,
                node.text,
                node.rationale,
                embedding_blob,
                node.precision_bits,
                node.weight,
                node.project,
                node.scope,
                node.source,
                now,
                node.created_at if node.created_at else now,
                node.session_count,
            ),
        )
        self._conn.commit()
        return node_id

    def merge_node(self, existing_id: str, candidate: Node) -> None:
        """Update an existing node without creating a duplicate.

        Merging rules (from extraction-pipeline.md):
        - weight += 0.5
        - last_accessed = now()
        - session_count += 1
        - text: keep existing unless new text is shorter
        - rationale: keep existing if non-null; fill from candidate if null

        Args:
            existing_id: The UUID of the node already in the graph.
            candidate: New candidate node carrying updated field values.
        """
        now = int(time.time())
        row = self._conn.execute(
            "SELECT text, rationale FROM nodes WHERE id = ?", (existing_id,)
        ).fetchone()
        if row is None:
            return

        existing_text: str = row["text"]
        existing_rationale: str | None = row["rationale"]

        new_text = (
            candidate.text
            if len(candidate.text) < len(existing_text)
            else existing_text
        )
        new_rationale = (
            existing_rationale
            if existing_rationale is not None
            else candidate.rationale
        )

        self._conn.execute(
            """
            UPDATE nodes
            SET weight = weight + 0.5,
                last_accessed = ?,
                session_count = session_count + 1,
                text = ?,
                rationale = ?
            WHERE id = ?
            """,
            (now, new_text, new_rationale, existing_id),
        )
        self._conn.commit()

    def find_similar(
        self,
        embedding: np.ndarray,
        project: str,
        threshold: float,
        limit: int,
    ) -> list[Node]:
        """Return nodes whose embedding cosine similarity meets the threshold.

        Loads all embeddings for the project and computes cosine similarity
        in Python (sqlite-vec is not assumed to be installed).

        Args:
            embedding: Query vector (float32, 384-dim).
            project: Absolute project path to filter candidates.
            threshold: Minimum cosine similarity (inclusive).
            limit: Maximum number of results.

        Returns:
            List of Node objects with similarity >= threshold, ordered by
            descending similarity, capped at limit.
        """
        rows = self._conn.execute(
            "SELECT * FROM nodes WHERE project = ? AND embedding IS NOT NULL",
            (project,),
        ).fetchall()

        scored: list[tuple[float, Node]] = []
        for row in rows:
            node = _row_to_node(row)
            if node.embedding is None:
                continue
            sim = _cosine_similarity(embedding, node.embedding)
            if sim >= threshold:
                scored.append((sim, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scored[:limit]]

    def get_all_nodes(self, project: str, tier: int | None = None) -> list[Node]:
        """Return all nodes for a project, optionally filtered by tier.

        Args:
            project: Absolute project path.
            tier: If provided, only return nodes at this tier level.

        Returns:
            List of Node objects for the project.
        """
        if tier is not None:
            rows = self._conn.execute(
                "SELECT * FROM nodes WHERE project = ? AND tier = ?",
                (project, tier),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM nodes WHERE project = ?",
                (project,),
            ).fetchall()
        return [_row_to_node(row) for row in rows]

    def delete_node(self, node_id: str) -> None:
        """Delete a node and all its edges (CASCADE).

        Silently does nothing if the node does not exist.

        Args:
            node_id: UUID of the node to delete.
        """
        self._conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
        self._conn.commit()

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------

    def write_edge(self, source_id: str, target_id: str) -> None:
        """Insert or increment an edge between two nodes.

        If the edge already exists, strength is incremented by 1.0 and
        last_seen is updated. Self-loops raise ValueError.

        Args:
            source_id: UUID of the source node.
            target_id: UUID of the target node.

        Raises:
            ValueError: If source_id == target_id.
            sqlite3.IntegrityError: If either node does not exist (FK constraint).
        """
        if source_id == target_id:
            raise ValueError("Self-loops are not allowed.")
        now = int(time.time())
        self._conn.execute(
            """
            INSERT INTO edges (source_id, target_id, strength, last_seen)
            VALUES (?, ?, 1.0, ?)
            ON CONFLICT (source_id, target_id) DO UPDATE SET
                strength = strength + 1.0,
                last_seen = excluded.last_seen
            """,
            (source_id, target_id, now),
        )
        self._conn.commit()

    def get_edges(self, node_id: str) -> list[Edge]:
        """Return all edges where node_id is source or target.

        Args:
            node_id: UUID of the node whose edges to retrieve.

        Returns:
            List of Edge objects connected to this node.
        """
        rows = self._conn.execute(
            """
            SELECT source_id, target_id, strength, last_seen
            FROM edges
            WHERE source_id = ? OR target_id = ?
            """,
            (node_id, node_id),
        ).fetchall()
        return [
            Edge(
                source_id=row["source_id"],
                target_id=row["target_id"],
                strength=row["strength"],
                last_seen=row["last_seen"],
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Weight
    # ------------------------------------------------------------------

    def update_weight(self, node_id: str, delta: float) -> None:
        """Apply a weight delta to a node, flooring at 0.0.

        Silently does nothing if the node does not exist.

        Args:
            node_id: UUID of the node to update.
            delta: Amount to add (positive) or subtract (negative).
        """
        self._conn.execute(
            """
            UPDATE nodes
            SET weight = MAX(0.0, weight + ?)
            WHERE id = ?
            """,
            (delta, node_id),
        )
        self._conn.commit()
