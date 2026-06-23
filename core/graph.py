"""SQLite-backed knowledge graph for Cortex memory nodes and edges.

No ORM. Raw SQL only. schema.sql is the single source of truth.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from core.embedder import cosine_similarity, deserialize, serialize


@lru_cache(maxsize=4096)
def _cached_deserialize(blob: bytes, precision_bits: int) -> np.ndarray:
    """Deserialize an embedding blob, caching the result by (blob, precision).

    The cache avoids repeated deserialization of the same node embedding
    across multiple find_similar calls within one session. The key is the
    raw blob bytes, which is immutable after write.
    """
    return deserialize(blob, precision_bits)


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


def _row_to_node(row: sqlite3.Row) -> Node:
    """Convert a sqlite3.Row to a Node dataclass."""
    embedding: np.ndarray | None = None
    if row["embedding"] is not None:
        embedding = _cached_deserialize(bytes(row["embedding"]), row["precision_bits"])
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
            embedding_blob = serialize(node.embedding, node.precision_bits)
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
        - text: keep existing unless candidate text is strictly shorter
        - rationale: keep existing if non-null; fill from candidate if null
        - embedding: update to candidate embedding when candidate text wins
          (the new text represents the latest phrasing, so the embedding
          should match it rather than the superseded phrasing)

        Args:
            existing_id: The UUID of the node already in the graph.
            candidate: New candidate node carrying updated field values.
        """
        now = int(time.time())
        row = self._conn.execute(
            "SELECT text, rationale, precision_bits FROM nodes WHERE id = ?",
            (existing_id,),
        ).fetchone()
        if row is None:
            return

        existing_text: str = row["text"]
        existing_rationale: str | None = row["rationale"]
        existing_precision: int = row["precision_bits"]

        candidate_wins = len(candidate.text) < len(existing_text)
        new_text = candidate.text if candidate_wins else existing_text
        new_rationale = (
            existing_rationale
            if existing_rationale is not None
            else candidate.rationale
        )

        # Refresh the embedding when the candidate text is adopted
        new_embedding_blob: bytes | None = None
        if candidate_wins and candidate.embedding is not None:
            new_embedding_blob = serialize(candidate.embedding, existing_precision)

        if new_embedding_blob is not None:
            self._conn.execute(
                """
                UPDATE nodes
                SET weight = weight + 0.5,
                    last_accessed = ?,
                    session_count = session_count + 1,
                    text = ?,
                    rationale = ?,
                    embedding = ?
                WHERE id = ?
                """,
                (now, new_text, new_rationale, new_embedding_blob, existing_id),
            )
        else:
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
            sim = cosine_similarity(embedding, node.embedding)
            if sim >= threshold:
                scored.append((sim, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scored[:limit]]

    def get_node(self, node_id: str) -> Node | None:
        """Return a single node by UUID, or None if not found.

        Args:
            node_id: UUID of the node to fetch.

        Returns:
            Node dataclass, or None.
        """
        row = self._conn.execute(
            "SELECT * FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        if row is None:
            return None
        return _row_to_node(row)

    def get_all_nodes(self, project: str, tier: int | None = None) -> list[Node]:
        """Return all nodes for a project ordered by weight descending.

        Ordering by weight ensures callers (CLI, retrieval) naturally see the
        most accessed/important nodes first without a secondary sort step.

        Args:
            project: Absolute project path.
            tier: If provided, only return nodes at this tier level.

        Returns:
            List of Node objects for the project, ordered by weight DESC.
        """
        if tier is not None:
            rows = self._conn.execute(
                "SELECT * FROM nodes WHERE project = ? AND tier = ? ORDER BY weight DESC",
                (project, tier),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM nodes WHERE project = ? ORDER BY weight DESC",
                (project,),
            ).fetchall()
        return [_row_to_node(row) for row in rows]

    def get_nodes_by_source(
        self, project: str, source: str, tier: int | None = None
    ) -> list[Node]:
        """Return nodes filtered by extraction source channel.

        Uses the idx_nodes_project_source composite index for efficiency.

        Args:
            project: Absolute project path.
            source: Extraction source: 'jsonl', 'ast', 'git', or 'nlp'.
            tier: Optional tier filter; None returns all tiers.

        Returns:
            List of Node objects, ordered by weight DESC.
        """
        if tier is not None:
            rows = self._conn.execute(
                "SELECT * FROM nodes WHERE project = ? AND source = ? AND tier = ? "
                "ORDER BY weight DESC",
                (project, source, tier),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM nodes WHERE project = ? AND source = ? "
                "ORDER BY weight DESC",
                (project, source),
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

    def delete_all_nodes(self, project: str) -> int:
        """Delete all nodes (and their edges) for a project in one statement.

        Args:
            project: Absolute project path.

        Returns:
            Number of nodes deleted.
        """
        cursor = self._conn.execute("DELETE FROM nodes WHERE project = ?", (project,))
        self._conn.commit()
        return cursor.rowcount

    def update_node_tier(
        self,
        node_id: str,
        new_tier: int,
        new_precision: int,
        embedding_blob: bytes | None,
        now: int,
    ) -> None:
        """Promote a node to a higher tier with downcast embedding precision.

        Args:
            node_id: UUID of the node to promote.
            new_tier: Target tier (2 or 3).
            new_precision: Target precision_bits (8 or 2).
            embedding_blob: Re-serialized embedding at the new precision, or None.
            now: Current unix timestamp for last_accessed.
        """
        self._conn.execute(
            """
            UPDATE nodes
            SET tier = ?,
                precision_bits = ?,
                embedding = ?,
                last_accessed = ?
            WHERE id = ?
            """,
            (new_tier, new_precision, embedding_blob, now, node_id),
        )
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

    def get_edges_for_nodes(self, node_ids: list[str]) -> dict[str, list[Edge]]:
        """Return all edges for a set of nodes in two SQL queries.

        Much more efficient than calling get_edges() in a loop when many
        nodes need their adjacency lists at once (e.g. graph_channel).

        Args:
            node_ids: UUIDs to fetch edges for.

        Returns:
            Mapping from node_id to the list of Edge objects touching it.
        """
        if not node_ids:
            return {}

        placeholders = ",".join("?" * len(node_ids))
        rows = self._conn.execute(
            f"""
            SELECT source_id, target_id, strength, last_seen
            FROM edges
            WHERE source_id IN ({placeholders})
               OR target_id IN ({placeholders})
            """,
            node_ids + node_ids,
        ).fetchall()

        result: dict[str, list[Edge]] = {nid: [] for nid in node_ids}
        for row in rows:
            edge = Edge(
                source_id=row["source_id"],
                target_id=row["target_id"],
                strength=row["strength"],
                last_seen=row["last_seen"],
            )
            if row["source_id"] in result:
                result[row["source_id"]].append(edge)
            if row["target_id"] in result:
                result[row["target_id"]].append(edge)

        return result

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def write_session(
        self,
        session_id: str,
        project: str,
        started_at: int,
        ended_at: int,
        nodes_written: int,
        nodes_evicted: int,
        nodes_promoted: int,
        transcript_path: str,
        tokens_raw: int | None = None,
        tokens_injected: int | None = None,
    ) -> None:
        """Insert a session record into the sessions table.

        Args:
            session_id: UUID4 for this session.
            project: Absolute project path.
            started_at: Unix timestamp of session start.
            ended_at: Unix timestamp of session end.
            nodes_written: Number of new nodes written.
            nodes_evicted: Number of nodes evicted during decay.
            nodes_promoted: Number of nodes promoted to a higher tier.
            transcript_path: Path to the JSONL transcript file.
            tokens_raw: Total tokens across all project node texts (no budget).
            tokens_injected: Tokens in the budget-constrained injection block.
        """
        self._conn.execute(
            """
            INSERT INTO sessions (
                id, project, started_at, ended_at,
                nodes_written, nodes_evicted, nodes_promoted, transcript_path,
                tokens_raw, tokens_injected
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                project,
                started_at,
                ended_at,
                nodes_written,
                nodes_evicted,
                nodes_promoted,
                transcript_path,
                tokens_raw,
                tokens_injected,
            ),
        )
        self._conn.commit()

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

    def update_node_rationale(self, node_id: str, rationale: str | None) -> bool:
        """Set or clear the rationale field on a node.

        Args:
            node_id: UUID of the node to update.
            rationale: New rationale string, or None to clear it.

        Returns:
            True if a row was updated, False if node not found.
        """
        cursor = self._conn.execute(
            "UPDATE nodes SET rationale = ? WHERE id = ?",
            (rationale, node_id),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def set_node_weight(self, node_id: str, weight: float) -> bool:
        """Set a node's weight to an absolute value (floored at 0.0).

        Args:
            node_id: UUID of the node.
            weight: New weight value (floored at 0.0).

        Returns:
            True if a row was updated, False if node not found.
        """
        cursor = self._conn.execute(
            "UPDATE nodes SET weight = MAX(0.0, ?) WHERE id = ?",
            (weight, node_id),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def get_stale_nodes(
        self, project: str, days: int, tier: int | None = None
    ) -> list[Node]:
        """Return nodes that have not been accessed in at least `days` days.

        Tier-3 (procedural) nodes are excluded from stale detection because
        they are permanent and must be pruned explicitly via `cortex prune`.

        Args:
            project: Absolute project path.
            days: Minimum inactivity threshold in days.
            tier: If provided, restrict to this tier only. Tier 3 is always
                excluded regardless of this parameter.

        Returns:
            List of stale Node objects ordered by last_accessed ascending
            (least recently accessed first).
        """
        import time

        cutoff = int(time.time()) - days * 86_400
        if tier is not None and tier != 3:
            rows = self._conn.execute(
                "SELECT * FROM nodes WHERE project = ? AND tier = ? "
                "AND last_accessed < ? ORDER BY last_accessed ASC",
                (project, tier, cutoff),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM nodes WHERE project = ? AND tier < 3 "
                "AND last_accessed < ? ORDER BY last_accessed ASC",
                (project, cutoff),
            ).fetchall()
        return [_row_to_node(row) for row in rows]

    def delete_nodes_bulk(self, node_ids: list[str]) -> int:
        """Delete multiple nodes by ID in a single SQL statement.

        Args:
            node_ids: UUIDs to delete. Empty list is a safe no-op.

        Returns:
            Number of rows actually deleted.
        """
        if not node_ids:
            return 0
        placeholders = ",".join("?" * len(node_ids))
        cursor = self._conn.execute(
            f"DELETE FROM nodes WHERE id IN ({placeholders})", node_ids
        )
        self._conn.commit()
        return cursor.rowcount

    def touch_nodes(self, node_ids: list[str], now: int) -> None:
        """Update last_accessed, bump weight, and increment session_count.

        Called by the inject hook to reinforce nodes that were actually
        retrieved and surfaced to Claude. session_count must be incremented
        here so tier-promotion thresholds (which check session_count) reflect
        how many distinct sessions have seen each node, not just how many
        sessions produced it via extraction.

        Args:
            node_ids: UUIDs of nodes that were retrieved this session.
            now: Current unix timestamp.
        """
        if not node_ids:
            return
        placeholders = ",".join("?" * len(node_ids))
        self._conn.execute(
            f"""
            UPDATE nodes
            SET last_accessed = ?,
                weight = weight + 0.1,
                session_count = session_count + 1
            WHERE id IN ({placeholders})
            """,
            [now, *node_ids],
        )
        self._conn.commit()
