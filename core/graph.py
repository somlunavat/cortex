"""SQLite-backed knowledge graph for Cortex memory nodes and edges.

No ORM. Raw SQL only. schema.sql is the single source of truth.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from core.embedder import cosine_similarity, deserialize, serialize

TOUCH_WEIGHT_BUMP: float = 0.1

_NODE_COLUMNS = (
    "id, type, tier, text, rationale, embedding, precision_bits, "
    "weight, project, scope, source, last_accessed, created_at, session_count"
)


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


def _node_filter(
    project: str,
    *,
    tier: int | None = None,
    source: str | None = None,
) -> tuple[str, list[Any]]:
    """Build a WHERE clause and params list for node queries.

    Always includes project = ?. Appends tier and/or source filters when
    provided. Returns (where_clause, params) ready for cursor.execute.
    """
    clauses = ["project = ?"]
    params: list[Any] = [project]
    if tier is not None:
        clauses.append("tier = ?")
        params.append(tier)
    if source is not None:
        clauses.append("source = ?")
        params.append(source)
    return " AND ".join(clauses), params


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

    def _exec_nodes(self, sql: str, params: list[Any]) -> list[Node]:
        """Execute a SELECT query and return the results as Node objects."""
        return [_row_to_node(row) for row in self._conn.execute(sql, params).fetchall()]

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
        created = node.created_at or now
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
                created,
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
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold}")
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")

        rows = self._conn.execute(
            f"SELECT {_NODE_COLUMNS} FROM nodes WHERE project = ? AND embedding IS NOT NULL",  # nosec B608
            (project,),
        ).fetchall()

        scored: list[tuple[float, Node]] = []
        for row in rows:
            node = _row_to_node(row)
            sim = cosine_similarity(embedding, node.embedding)  # type: ignore[arg-type]
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
            f"SELECT {_NODE_COLUMNS} FROM nodes WHERE id = ?",  # nosec B608
            (node_id,),
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
        where, params = _node_filter(project, tier=tier)
        return self._exec_nodes(
            f"SELECT {_NODE_COLUMNS} FROM nodes WHERE {where} ORDER BY weight DESC",  # nosec B608
            params,
        )

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
        where, params = _node_filter(project, source=source, tier=tier)
        return self._exec_nodes(
            f"SELECT {_NODE_COLUMNS} FROM nodes WHERE {where} ORDER BY weight DESC",  # nosec B608
            params,
        )

    def delete_node(self, node_id: str) -> None:
        """Delete a node and all its edges (CASCADE).

        Silently does nothing if the node does not exist or if node_id is empty.

        Args:
            node_id: UUID of the node to delete.
        """
        if not node_id:
            return
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
        sql = f"SELECT source_id, target_id, strength, last_seen FROM edges WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})"  # nosec B608
        rows = self._conn.execute(sql, node_ids + node_ids).fetchall()

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

        Raises:
            ValueError: If ended_at < started_at or any count is negative.
        """
        if ended_at < started_at:
            raise ValueError(
                f"ended_at ({ended_at}) must not be before started_at ({started_at})"
            )
        if nodes_written < 0 or nodes_evicted < 0 or nodes_promoted < 0:
            raise ValueError("node counts must be non-negative")
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
        cutoff = int(time.time()) - days * 86_400
        effective_tier = tier if (tier is not None and tier != 3) else None
        where, params = _node_filter(project, tier=effective_tier)
        if effective_tier is None:
            where += " AND tier < 3"
        where += " AND last_accessed < ?"
        params.append(cutoff)
        return self._exec_nodes(
            f"SELECT {_NODE_COLUMNS} FROM nodes WHERE {where} ORDER BY last_accessed ASC",  # nosec B608
            params,
        )

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
        sql = f"DELETE FROM nodes WHERE id IN ({placeholders})"  # nosec B608
        cursor = self._conn.execute(sql, node_ids)
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
        sql = f"UPDATE nodes SET last_accessed = ?, weight = weight + {TOUCH_WEIGHT_BUMP}, session_count = session_count + 1 WHERE id IN ({placeholders})"  # nosec B608
        self._conn.execute(sql, [now, *node_ids])
        self._conn.commit()

    # ---------------------------------------------------------------------------
    # Query helpers — keep all SQL inside this class
    # ---------------------------------------------------------------------------

    def session_exists(self, transcript_path: str) -> bool:
        """Return True if this transcript has already been processed.

        Args:
            transcript_path: Absolute path to the JSONL transcript file.
        """
        row = self._conn.execute(
            "SELECT id FROM sessions WHERE transcript_path = ?",
            (transcript_path,),
        ).fetchone()
        return row is not None

    def get_tier_counts(self, project: str) -> dict[int, tuple[int, float]]:
        """Return {tier: (count, avg_weight)} for all tiers in a project.

        Args:
            project: Absolute project path.
        """
        rows = self._conn.execute(
            "SELECT tier, COUNT(*) AS cnt, AVG(weight) AS avg_w "
            "FROM nodes WHERE project = ? GROUP BY tier",
            (project,),
        ).fetchall()
        return {int(r["tier"]): (int(r["cnt"]), float(r["avg_w"] or 0.0)) for r in rows}

    def get_type_counts(self, project: str) -> list[tuple[str, int]]:
        """Return [(type, count)] ordered by count DESC for a project.

        Args:
            project: Absolute project path.
        """
        rows = self._conn.execute(
            "SELECT type, COUNT(*) AS cnt FROM nodes "
            "WHERE project = ? GROUP BY type ORDER BY cnt DESC",
            (project,),
        ).fetchall()
        return [(str(r["type"]), int(r["cnt"])) for r in rows]

    def get_source_counts(self, project: str) -> list[tuple[str, int]]:
        """Return [(source, count)] ordered by count DESC for a project.

        Args:
            project: Absolute project path.
        """
        rows = self._conn.execute(
            "SELECT source, COUNT(*) AS cnt FROM nodes "
            "WHERE project = ? GROUP BY source ORDER BY cnt DESC",
            (project,),
        ).fetchall()
        return [(str(r["source"]), int(r["cnt"])) for r in rows]

    def get_last_session(self, project: str) -> dict[str, Any] | None:
        """Return the most recent session record for a project, or None.

        Args:
            project: Absolute project path.
        """
        row = self._conn.execute(
            "SELECT ended_at, nodes_written, nodes_evicted, nodes_promoted, "
            "tokens_raw, tokens_injected "
            "FROM sessions WHERE project = ? ORDER BY ended_at DESC LIMIT 1",
            (project,),
        ).fetchone()
        return dict(row) if row else None

    def get_recent_nodes(self, project: str, limit: int = 10) -> list[Node]:
        """Return nodes ordered by last_accessed DESC.

        Args:
            project: Absolute project path.
            limit: Maximum number of nodes to return.
        """
        return self._exec_nodes(
            "SELECT id, type, tier, text, rationale, embedding, precision_bits, "
            "weight, project, scope, source, last_accessed, created_at, session_count "
            "FROM nodes WHERE project = ? ORDER BY last_accessed DESC LIMIT ?",
            [project, limit],
        )

    def get_session_records(
        self, project: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Return recent session records ordered by ended_at DESC.

        Args:
            project: Absolute project path.
            limit: Maximum number of sessions to return.
        """
        rows = self._conn.execute(
            "SELECT id, started_at, ended_at, nodes_written, nodes_evicted, "
            "nodes_promoted, tokens_raw, tokens_injected, transcript_path "
            "FROM sessions WHERE project = ? ORDER BY ended_at DESC LIMIT ?",
            (project, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_aggregate_stats(self, project: str) -> dict[str, Any]:
        """Return aggregate session and node stats for a project.

        Args:
            project: Absolute project path.
        """
        agg = self._conn.execute(
            """
            SELECT
                COUNT(*)              AS session_count,
                SUM(nodes_written)    AS total_written,
                SUM(nodes_evicted)    AS total_evicted,
                SUM(nodes_promoted)   AS total_promoted,
                SUM(CASE WHEN tokens_raw IS NOT NULL AND tokens_injected IS NOT NULL
                         THEN tokens_raw - tokens_injected ELSE 0 END) AS total_saved,
                MIN(started_at)       AS first_session,
                MAX(ended_at)         AS last_session
            FROM sessions
            WHERE project = ?
            """,
            (project,),
        ).fetchone()
        tier_counts = dict(
            self._conn.execute(
                "SELECT tier, COUNT(*) FROM nodes WHERE project = ? GROUP BY tier",
                (project,),
            ).fetchall()
        )
        return {
            "session_count": agg["session_count"] or 0,
            "total_written": agg["total_written"] or 0,
            "total_evicted": agg["total_evicted"] or 0,
            "total_promoted": agg["total_promoted"] or 0,
            "total_saved": agg["total_saved"] or 0,
            "first_session": agg["first_session"],
            "last_session": agg["last_session"],
            "tier_counts": {int(k): int(v) for k, v in tier_counts.items()},
        }

    def query_nodes_page(
        self,
        project: str,
        *,
        tier: int | None = None,
        node_type: str | None = None,
        source: str | None = None,
        sort_col: str = "weight",
        limit: int = 25,
    ) -> tuple[list[Node], int]:
        """Return a page of nodes matching the given filters plus the total count.

        ``sort_col`` must be a valid column name from the allowlist
        ``{"weight", "created_at", "last_accessed"}``; the caller is responsible
        for validating it before calling this method.

        Args:
            project: Absolute project path.
            tier: If set, only return nodes in this tier.
            node_type: If set, only return nodes of this type.
            source: If set, only return nodes with this extraction source.
            sort_col: Column to sort by (must be pre-validated by caller).
            limit: Maximum nodes to return.

        Returns:
            (nodes, total_count) where total_count may be > len(nodes).
        """
        allowed_sort = {"weight", "created_at", "last_accessed"}
        if sort_col not in allowed_sort:
            raise ValueError(
                f"sort_col must be one of {allowed_sort}, got {sort_col!r}"
            )

        where, params = _node_filter(project, tier=tier, source=source)
        if node_type is not None:
            where += " AND type = ?"
            params.append(node_type)

        page_sql = f"SELECT id, type, tier, text, rationale, embedding, precision_bits, weight, project, scope, source, last_accessed, created_at, session_count FROM nodes WHERE {where} ORDER BY {sort_col} DESC LIMIT ?"  # nosec B608
        count_sql = f"SELECT COUNT(*) FROM nodes WHERE {where}"  # nosec B608
        total: int = self._conn.execute(count_sql, params).fetchone()[0]
        return self._exec_nodes(page_sql, [*params, limit]), total
