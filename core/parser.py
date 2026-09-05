"""JSONL transcript parser — converts raw Claude Code session files to typed events.

Reads one JSONL event per line. Skips malformed lines with a warning.
Never raises on bad input — logs and continues.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class EventType(StrEnum):
    """Canonical event types emitted by the Claude Code session transcript."""

    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    BASH_EXEC = "bash_exec"
    BASH_SUCCESS = "bash_success"
    BASH_FAILURE = "bash_failure"
    TOOL_USE = "tool_use"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    ASSISTANT_MESSAGE = "assistant_message"
    USER_MESSAGE = "user_message"


_KNOWN_TYPES: frozenset[str] = frozenset(e.value for e in EventType)

# Minimum number of writes to the same file within a session to flag as hotspot.
HOTSPOT_WRITE_THRESHOLD = 3

# Co-occurring write window in seconds.
CO_OCCURRENCE_WINDOW_SECONDS = 60


@dataclass(frozen=True)
class ParsedEvent:
    """A single extracted event from a Claude Code JSONL transcript."""

    type: EventType
    timestamp: int
    data: dict[str, Any]
    session_id: str


@dataclass(frozen=True)
class SessionSummary:
    """Aggregate signals derived from all events in a session."""

    session_id: str
    hotspot_files: frozenset[str]
    bash_failures: tuple[ParsedEvent, ...]
    co_occurring_pairs: tuple[frozenset[str], ...]
    prose_turns: tuple[str, ...]


def _normalize_path(raw: str) -> str:
    """Return an absolute POSIX path, resolving relative paths against cwd."""
    p = Path(raw)
    if not p.is_absolute():
        p = Path.cwd() / p
    return str(p)


def _parse_line(line: str, line_number: int) -> dict[str, Any] | None:
    """Parse a single JSONL line; return None and log a warning on failure."""
    stripped = line.strip()
    if not stripped:
        return None
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError as exc:
        logger.warning("Malformed JSONL on line %d: %s", line_number, exc)
        return None
    if not isinstance(obj, dict):
        logger.warning("Non-object JSONL on line %d; skipping", line_number)
        return None
    return obj


def _to_event_type(raw_type: str) -> EventType | None:
    """Map a raw type string to EventType; return None for unknown types."""
    if raw_type in _KNOWN_TYPES:
        return EventType(raw_type)
    return None


def _normalize_event_data(
    event_type: EventType, data: dict[str, Any]
) -> dict[str, Any]:
    """Normalize event-specific data fields (e.g., path to absolute path)."""
    out = dict(data)
    if (
        event_type in (EventType.FILE_READ, EventType.FILE_WRITE)
        and "path" in out
        and isinstance(out["path"], str)
    ):
        out["path"] = _normalize_path(out["path"])
    if event_type == EventType.TOOL_USE:
        tool_input = out.get("tool_input", {})
        if isinstance(tool_input, dict) and "file_path" in tool_input:
            out["tool_input"] = {
                **tool_input,
                "file_path": _normalize_path(str(tool_input["file_path"])),
            }
    return out


def parse_transcript(path: Path) -> Iterator[ParsedEvent]:
    """Parse a Claude Code JSONL transcript file into typed events.

    Skips user_message events (not project memory) and unknown event types.
    Malformed lines produce a warning log and are silently skipped.

    Args:
        path: Path to the JSONL transcript file.

    Yields:
        ParsedEvent for each valid, non-user event in the file.
    """
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            obj = _parse_line(line, line_number)
            if obj is None:
                continue

            raw_type = obj.get("type", "")
            event_type = _to_event_type(raw_type)
            if event_type is None:
                logger.debug(
                    "Unknown event type %r on line %d; skipping", raw_type, line_number
                )
                continue

            if event_type == EventType.USER_MESSAGE:
                continue

            timestamp = int(obj.get("timestamp", 0))
            session_id = str(obj.get("session_id", ""))
            data = _normalize_event_data(event_type, obj.get("data", {}))

            yield ParsedEvent(
                type=event_type,
                timestamp=timestamp,
                data=data,
                session_id=session_id,
            )


def detect_hotspots(events: list[ParsedEvent]) -> frozenset[str]:
    """Return file paths written >= HOTSPOT_WRITE_THRESHOLD times in the session.

    Args:
        events: All parsed events from one session.

    Returns:
        Frozenset of absolute file paths that are hotspots.
    """
    counts: Counter[str] = Counter()
    for event in events:
        if event.type == EventType.FILE_WRITE:
            path = event.data.get("path", "")
            if path:
                counts[path] += 1
    return frozenset(p for p, c in counts.items() if c >= HOTSPOT_WRITE_THRESHOLD)


def detect_co_occurring_writes(events: list[ParsedEvent]) -> tuple[frozenset[str], ...]:
    """Return pairs of files written within CO_OCCURRENCE_WINDOW_SECONDS of each other.

    Only includes pairs where both paths are different files.

    Uses a two-pointer sliding window over the sorted write list so the
    inner loop never visits events outside the time window. Complexity is
    O(n log n) for the sort plus O(n * w) where w is the average window
    width — much better than the O(n²) naive approach for large sessions.

    Args:
        events: All parsed events from one session.

    Returns:
        Tuple of frozensets, each containing two co-occurring file paths.
    """
    writes = sorted(
        (
            (e.timestamp, e.data.get("path", ""))
            for e in events
            if e.type == EventType.FILE_WRITE and e.data.get("path")
        ),
        key=lambda x: x[0],
    )

    pairs: list[frozenset[str]] = []
    seen: set[frozenset[str]] = set()
    left = 0

    for right, (ts_right, path_right) in enumerate(writes):
        # Advance left pointer to keep window within CO_OCCURRENCE_WINDOW_SECONDS
        while writes[left][0] < ts_right - CO_OCCURRENCE_WINDOW_SECONDS:
            left += 1

        for i in range(left, right):
            path_left = writes[i][1]
            if path_left == path_right:
                continue
            pair: frozenset[str] = frozenset({path_left, path_right})
            if pair not in seen:
                pairs.append(pair)
                seen.add(pair)

    return tuple(pairs)


def extract_prose_turns(events: list[ParsedEvent]) -> tuple[str, ...]:
    """Return Claude's prose output turns (assistant_message events only).

    Excludes tool outputs and user messages.

    Args:
        events: All parsed events from one session.

    Returns:
        Tuple of prose text strings from assistant turns.
    """
    return tuple(
        str(e.data.get("text", ""))
        for e in events
        if e.type == EventType.ASSISTANT_MESSAGE and e.data.get("text")
    )


def summarize_session(events: list[ParsedEvent]) -> SessionSummary:
    """Aggregate all high-value signals from a parsed session.

    Args:
        events: All parsed events from one session.

    Returns:
        SessionSummary with hotspots, failures, co-occurring pairs, and prose.
    """
    session_id = events[0].session_id if events else ""
    failures = tuple(e for e in events if e.type == EventType.BASH_FAILURE)
    return SessionSummary(
        session_id=session_id,
        hotspot_files=detect_hotspots(events),
        bash_failures=failures,
        co_occurring_pairs=detect_co_occurring_writes(events),
        prose_turns=extract_prose_turns(events),
    )
