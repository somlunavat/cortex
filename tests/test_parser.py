"""Tests for core/parser.py — JSONL transcript parser."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from core.parser import (
    CO_OCCURRENCE_WINDOW_SECONDS,
    HOTSPOT_WRITE_THRESHOLD,
    EventType,
    ParsedEvent,
    detect_co_occurring_writes,
    detect_hotspots,
    extract_prose_turns,
    parse_transcript,
    summarize_session,
)

SIMPLE_FIXTURE = Path("tests/fixtures/transcripts/simple.jsonl")
DECISIONS_FIXTURE = Path("tests/fixtures/transcripts/with_decisions.jsonl")

TEST_PROJECT = "/tmp/cortex_test_project"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(
    event_type: EventType,
    timestamp: int = 1000,
    data: dict | None = None,
    session_id: str = "s1",
) -> ParsedEvent:
    return ParsedEvent(
        type=event_type,
        timestamp=timestamp,
        data=data or {},
        session_id=session_id,
    )


def _make_write_event(path: str, timestamp: int = 1000) -> ParsedEvent:
    return _make_event(EventType.FILE_WRITE, timestamp=timestamp, data={"path": path})


def _write_jsonl(tmp_path: Path, lines: list[dict]) -> Path:
    p = tmp_path / "transcript.jsonl"
    p.write_text("\n".join(json.dumps(line) for line in lines))
    return p


# ---------------------------------------------------------------------------
# parse_transcript — fixture files
# ---------------------------------------------------------------------------


class TestParseTranscriptFixtures:
    def test_simple_fixture_parses_without_error(self) -> None:
        events = list(parse_transcript(SIMPLE_FIXTURE))
        assert len(events) > 0

    def test_decisions_fixture_parses_without_error(self) -> None:
        events = list(parse_transcript(DECISIONS_FIXTURE))
        assert len(events) > 0

    def test_simple_fixture_has_file_write_events(self) -> None:
        events = list(parse_transcript(SIMPLE_FIXTURE))
        writes = [e for e in events if e.type == EventType.FILE_WRITE]
        assert len(writes) > 0

    def test_simple_fixture_has_bash_failure(self) -> None:
        events = list(parse_transcript(SIMPLE_FIXTURE))
        failures = [e for e in events if e.type == EventType.BASH_FAILURE]
        assert len(failures) >= 1

    def test_user_messages_excluded_from_simple(self) -> None:
        events = list(parse_transcript(SIMPLE_FIXTURE))
        assert all(e.type != EventType.USER_MESSAGE for e in events)

    def test_decisions_fixture_has_assistant_messages(self) -> None:
        events = list(parse_transcript(DECISIONS_FIXTURE))
        prose = [e for e in events if e.type == EventType.ASSISTANT_MESSAGE]
        assert len(prose) >= 3

    def test_all_events_have_positive_timestamp(self) -> None:
        events = list(parse_transcript(SIMPLE_FIXTURE))
        assert all(e.timestamp > 0 for e in events)

    def test_all_events_have_session_id_set(self) -> None:
        events = list(parse_transcript(DECISIONS_FIXTURE))
        assert all(e.session_id for e in events)


# ---------------------------------------------------------------------------
# parse_transcript — event type coverage
# ---------------------------------------------------------------------------


class TestParseTranscriptEventTypes:
    @pytest.mark.parametrize(
        "event_type",
        [
            EventType.FILE_READ,
            EventType.FILE_WRITE,
            EventType.BASH_EXEC,
            EventType.BASH_SUCCESS,
            EventType.BASH_FAILURE,
            EventType.TOOL_USE,
            EventType.SESSION_START,
            EventType.SESSION_END,
            EventType.ASSISTANT_MESSAGE,
        ],
    )
    def test_all_known_event_types_parsed(
        self, tmp_path: Path, event_type: EventType
    ) -> None:
        p = _write_jsonl(
            tmp_path,
            [
                {
                    "type": event_type.value,
                    "timestamp": 1000,
                    "session_id": "s1",
                    "data": {},
                }
            ],
        )
        events = list(parse_transcript(p))
        assert len(events) == 1
        assert events[0].type == event_type

    def test_user_message_excluded(self, tmp_path: Path) -> None:
        p = _write_jsonl(
            tmp_path,
            [
                {
                    "type": "user_message",
                    "timestamp": 1000,
                    "session_id": "s1",
                    "data": {},
                }
            ],
        )
        events = list(parse_transcript(p))
        assert events == []

    def test_unknown_event_type_skipped(self, tmp_path: Path) -> None:
        p = _write_jsonl(
            tmp_path,
            [
                {
                    "type": "mystery_event",
                    "timestamp": 1000,
                    "session_id": "s1",
                    "data": {},
                }
            ],
        )
        events = list(parse_transcript(p))
        assert events == []


# ---------------------------------------------------------------------------
# parse_transcript — malformed lines
# ---------------------------------------------------------------------------


class TestMalformedLines:
    def test_malformed_json_skipped_with_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        p = tmp_path / "t.jsonl"
        p.write_text(
            '{"type": "session_start", "timestamp": 1, "session_id": "s1", "data": {}}\nNOT JSON\n'
        )
        with caplog.at_level(logging.WARNING):
            events = list(parse_transcript(p))
        assert len(events) == 1
        assert any("Malformed" in m for m in caplog.messages)

    def test_non_object_json_skipped(self, tmp_path: Path) -> None:
        p = tmp_path / "t.jsonl"
        p.write_text("[1, 2, 3]\n")
        events = list(parse_transcript(p))
        assert events == []

    def test_empty_lines_skipped(self, tmp_path: Path) -> None:
        p = tmp_path / "t.jsonl"
        p.write_text(
            '\n\n{"type": "session_start", "timestamp": 1, "session_id": "s1", "data": {}}\n\n'
        )
        events = list(parse_transcript(p))
        assert len(events) == 1

    def test_mixed_valid_invalid_lines(self, tmp_path: Path) -> None:
        p = tmp_path / "t.jsonl"
        lines = [
            '{"type": "session_start", "timestamp": 1, "session_id": "s1", "data": {}}',
            "BROKEN",
            '{"type": "session_end", "timestamp": 2, "session_id": "s1", "data": {}}',
        ]
        p.write_text("\n".join(lines))
        events = list(parse_transcript(p))
        assert len(events) == 2


# ---------------------------------------------------------------------------
# parse_transcript — data normalization
# ---------------------------------------------------------------------------


class TestDataNormalization:
    def test_file_write_path_normalized_to_absolute(self, tmp_path: Path) -> None:
        p = _write_jsonl(
            tmp_path,
            [
                {
                    "type": "file_write",
                    "timestamp": 1,
                    "session_id": "s1",
                    "data": {"path": "relative/path.py"},
                }
            ],
        )
        events = list(parse_transcript(p))
        assert events[0].data["path"].startswith("/")

    def test_file_read_path_normalized_to_absolute(self, tmp_path: Path) -> None:
        p = _write_jsonl(
            tmp_path,
            [
                {
                    "type": "file_read",
                    "timestamp": 1,
                    "session_id": "s1",
                    "data": {"path": "relative/file.py"},
                }
            ],
        )
        events = list(parse_transcript(p))
        assert events[0].data["path"].startswith("/")

    def test_absolute_path_unchanged(self, tmp_path: Path) -> None:
        p = _write_jsonl(
            tmp_path,
            [
                {
                    "type": "file_write",
                    "timestamp": 1,
                    "session_id": "s1",
                    "data": {"path": "/abs/path.py"},
                }
            ],
        )
        events = list(parse_transcript(p))
        assert events[0].data["path"] == "/abs/path.py"

    def test_bash_failure_exit_code_in_data(self, tmp_path: Path) -> None:
        p = _write_jsonl(
            tmp_path,
            [
                {
                    "type": "bash_failure",
                    "timestamp": 1,
                    "session_id": "s1",
                    "data": {"command": "pytest", "exit_code": 1, "output": "ERROR"},
                }
            ],
        )
        events = list(parse_transcript(p))
        assert events[0].data["exit_code"] == 1

    def test_tool_use_file_path_normalized(self, tmp_path: Path) -> None:
        p = _write_jsonl(
            tmp_path,
            [
                {
                    "type": "tool_use",
                    "timestamp": 1,
                    "session_id": "s1",
                    "data": {
                        "tool_name": "Edit",
                        "tool_input": {"file_path": "src/app.py"},
                    },
                }
            ],
        )
        events = list(parse_transcript(p))
        assert events[0].data["tool_input"]["file_path"].startswith("/")

    def test_event_session_id_preserved(self, tmp_path: Path) -> None:
        p = _write_jsonl(
            tmp_path,
            [
                {
                    "type": "file_write",
                    "timestamp": 1,
                    "session_id": "my-session-abc",
                    "data": {"path": "/some/path.py"},
                }
            ],
        )
        events = list(parse_transcript(p))
        assert events[0].session_id == "my-session-abc"

    def test_bash_failure_event_type_correct(self, tmp_path: Path) -> None:
        p = _write_jsonl(
            tmp_path,
            [
                {
                    "type": "bash_failure",
                    "timestamp": 5,
                    "session_id": "s2",
                    "data": {"command": "pytest", "exit_code": 1, "output": "err"},
                }
            ],
        )
        events = list(parse_transcript(p))
        assert events[0].type == EventType.BASH_FAILURE


# ---------------------------------------------------------------------------
# detect_hotspots
# ---------------------------------------------------------------------------


class TestDetectHotspots:
    def test_file_written_threshold_times_is_hotspot(self) -> None:
        events = [
            _make_write_event("/proj/auth.py", ts)
            for ts in range(HOTSPOT_WRITE_THRESHOLD)
        ]
        hotspots = detect_hotspots(events)
        assert "/proj/auth.py" in hotspots

    def test_file_below_threshold_not_hotspot(self) -> None:
        events = [
            _make_write_event("/proj/auth.py", ts)
            for ts in range(HOTSPOT_WRITE_THRESHOLD - 1)
        ]
        hotspots = detect_hotspots(events)
        assert "/proj/auth.py" not in hotspots

    def test_simple_fixture_has_auth_middleware_hotspot(self) -> None:
        events = list(parse_transcript(SIMPLE_FIXTURE))
        hotspots = detect_hotspots(events)
        assert any("middleware" in p for p in hotspots)

    def test_multiple_hotspots_detected(self) -> None:
        events = [
            _make_write_event("/proj/a.py", ts) for ts in range(HOTSPOT_WRITE_THRESHOLD)
        ] + [
            _make_write_event("/proj/b.py", ts + 100)
            for ts in range(HOTSPOT_WRITE_THRESHOLD)
        ]
        hotspots = detect_hotspots(events)
        assert "/proj/a.py" in hotspots
        assert "/proj/b.py" in hotspots

    def test_empty_events_returns_empty(self) -> None:
        assert detect_hotspots([]) == frozenset()

    def test_non_write_events_ignored(self) -> None:
        events = [
            _make_event(EventType.FILE_READ, data={"path": "/proj/a.py"}),
            _make_event(EventType.FILE_READ, data={"path": "/proj/a.py"}),
            _make_event(EventType.FILE_READ, data={"path": "/proj/a.py"}),
        ]
        assert detect_hotspots(events) == frozenset()

    @pytest.mark.parametrize("count", [3, 4, 10])
    def test_hotspot_threshold_parametrized(self, count: int) -> None:
        events = [_make_write_event("/proj/f.py", ts) for ts in range(count)]
        hotspots = detect_hotspots(events)
        assert "/proj/f.py" in hotspots

    def test_returns_frozenset_type(self) -> None:
        result = detect_hotspots([])
        assert isinstance(result, frozenset)

    def test_two_files_each_at_threshold_both_hotspots(self) -> None:
        events = [
            _make_write_event("/proj/x.py", ts) for ts in range(HOTSPOT_WRITE_THRESHOLD)
        ] + [
            _make_write_event("/proj/y.py", ts + 1000)
            for ts in range(HOTSPOT_WRITE_THRESHOLD)
        ]
        hotspots = detect_hotspots(events)
        assert "/proj/x.py" in hotspots
        assert "/proj/y.py" in hotspots

    def test_returns_set_of_strings(self) -> None:
        events = [
            _make_write_event("/proj/file.py", ts)
            for ts in range(HOTSPOT_WRITE_THRESHOLD)
        ]
        hotspots = detect_hotspots(events)
        assert isinstance(hotspots, (set, frozenset))
        assert all(isinstance(h, str) for h in hotspots)

    def test_many_writes_same_file_still_one_hotspot_entry(self) -> None:
        events = [
            _make_write_event("/proj/busy.py", ts)
            for ts in range(HOTSPOT_WRITE_THRESHOLD * 5)
        ]
        hotspots = detect_hotspots(events)
        assert (
            hotspots.count("/proj/busy.py") == 1
            if isinstance(hotspots, list)
            else "/proj/busy.py" in hotspots
        )

    def test_different_files_counted_separately(self) -> None:
        events = [
            _make_write_event("/proj/a.py", ts) for ts in range(HOTSPOT_WRITE_THRESHOLD)
        ] + [
            _make_write_event("/proj/b.py", ts + 1000)
            for ts in range(HOTSPOT_WRITE_THRESHOLD - 1)
        ]
        hotspots = detect_hotspots(events)
        assert "/proj/a.py" in hotspots
        assert "/proj/b.py" not in hotspots

    def test_writes_at_varied_timestamps_still_counted(self) -> None:
        timestamps = [0, 500, 1000, 5000, 10000]
        events = [_make_write_event("/proj/main.py", ts) for ts in timestamps]
        hotspots = detect_hotspots(events)
        assert "/proj/main.py" in hotspots


# ---------------------------------------------------------------------------
# detect_co_occurring_writes
# ---------------------------------------------------------------------------


class TestDetectCoOccurringWrites:
    def test_two_files_within_window_are_paired(self) -> None:
        events = [
            _make_write_event("/proj/a.py", timestamp=0),
            _make_write_event("/proj/b.py", timestamp=CO_OCCURRENCE_WINDOW_SECONDS - 1),
        ]
        pairs = detect_co_occurring_writes(events)
        assert len(pairs) == 1
        assert frozenset({"/proj/a.py", "/proj/b.py"}) in pairs

    def test_two_files_outside_window_not_paired(self) -> None:
        events = [
            _make_write_event("/proj/a.py", timestamp=0),
            _make_write_event("/proj/b.py", timestamp=CO_OCCURRENCE_WINDOW_SECONDS + 1),
        ]
        pairs = detect_co_occurring_writes(events)
        assert len(pairs) == 0

    def test_same_file_twice_not_paired(self) -> None:
        events = [
            _make_write_event("/proj/a.py", timestamp=0),
            _make_write_event("/proj/a.py", timestamp=5),
        ]
        pairs = detect_co_occurring_writes(events)
        assert len(pairs) == 0

    def test_simple_fixture_has_co_occurring_pairs(self) -> None:
        events = list(parse_transcript(SIMPLE_FIXTURE))
        pairs = detect_co_occurring_writes(events)
        assert len(pairs) >= 1

    def test_duplicate_pairs_not_repeated(self) -> None:
        events = [
            _make_write_event("/proj/a.py", timestamp=0),
            _make_write_event("/proj/b.py", timestamp=5),
            _make_write_event("/proj/a.py", timestamp=10),
        ]
        pairs = detect_co_occurring_writes(events)
        pair = frozenset({"/proj/a.py", "/proj/b.py"})
        assert pairs.count(pair) == 1

    def test_empty_events_returns_empty(self) -> None:
        assert detect_co_occurring_writes([]) == ()

    def test_three_files_within_window_produce_three_pairs(self) -> None:
        events = [
            _make_write_event("/proj/a.py", timestamp=0),
            _make_write_event("/proj/b.py", timestamp=5),
            _make_write_event("/proj/c.py", timestamp=10),
        ]
        pairs = detect_co_occurring_writes(events)
        assert len(pairs) == 3
        assert frozenset({"/proj/a.py", "/proj/b.py"}) in pairs
        assert frozenset({"/proj/a.py", "/proj/c.py"}) in pairs
        assert frozenset({"/proj/b.py", "/proj/c.py"}) in pairs

    def test_exact_boundary_is_included(self) -> None:
        events = [
            _make_write_event("/proj/x.py", timestamp=0),
            _make_write_event("/proj/y.py", timestamp=CO_OCCURRENCE_WINDOW_SECONDS),
        ]
        pairs = detect_co_occurring_writes(events)
        assert frozenset({"/proj/x.py", "/proj/y.py"}) in pairs

    def test_non_write_events_ignored_in_co_occurrence(self) -> None:
        events = [
            _make_write_event("/proj/a.py", timestamp=0),
            _make_event(EventType.BASH_EXEC, timestamp=1, data={"command": "ls"}),
            _make_event(EventType.FILE_READ, timestamp=2, data={"path": "/proj/b.py"}),
            _make_write_event("/proj/c.py", timestamp=3),
        ]
        pairs = detect_co_occurring_writes(events)
        assert frozenset({"/proj/a.py", "/proj/c.py"}) in pairs
        assert len(pairs) == 1

    def test_single_write_produces_no_pairs(self) -> None:
        events = [_make_write_event("/proj/only.py", timestamp=0)]
        pairs = detect_co_occurring_writes(events)
        assert len(pairs) == 0

    def test_pairs_are_frozensets(self) -> None:
        events = [
            _make_write_event("/proj/x.py", timestamp=0),
            _make_write_event("/proj/y.py", timestamp=1),
        ]
        pairs = detect_co_occurring_writes(events)
        assert all(isinstance(p, frozenset) for p in pairs)

    def test_four_files_within_window_produces_six_pairs(self) -> None:
        events = [
            _make_write_event(f"/proj/{c}.py", timestamp=i)
            for i, c in enumerate("abcd")
        ]
        pairs = detect_co_occurring_writes(events)
        assert len(pairs) == 6


# ---------------------------------------------------------------------------
# extract_prose_turns
# ---------------------------------------------------------------------------


class TestExtractProseTurns:
    def test_returns_assistant_message_texts(self) -> None:
        events = [
            _make_event(EventType.ASSISTANT_MESSAGE, data={"text": "I chose asyncpg"}),
            _make_event(
                EventType.ASSISTANT_MESSAGE, data={"text": "Because performance"}
            ),
        ]
        prose = extract_prose_turns(events)
        assert "I chose asyncpg" in prose
        assert "Because performance" in prose

    def test_excludes_non_assistant_events(self) -> None:
        events = [
            _make_event(EventType.FILE_WRITE, data={"path": "/f.py"}),
            _make_event(EventType.BASH_EXEC, data={"command": "ls"}),
            _make_event(EventType.ASSISTANT_MESSAGE, data={"text": "hello"}),
        ]
        prose = extract_prose_turns(events)
        assert len(prose) == 1

    def test_skips_empty_text(self) -> None:
        events = [
            _make_event(EventType.ASSISTANT_MESSAGE, data={"text": ""}),
            _make_event(EventType.ASSISTANT_MESSAGE, data={"text": "non-empty"}),
        ]
        prose = extract_prose_turns(events)
        assert len(prose) == 1

    def test_decisions_fixture_prose_contains_decision_words(self) -> None:
        events = list(parse_transcript(DECISIONS_FIXTURE))
        prose = extract_prose_turns(events)
        combined = " ".join(prose)
        assert "chose" in combined or "decided" in combined or "selected" in combined

    def test_empty_events_returns_empty(self) -> None:
        assert extract_prose_turns([]) == ()

    def test_missing_text_key_skipped(self) -> None:
        events = [
            _make_event(EventType.ASSISTANT_MESSAGE, data={}),
            _make_event(EventType.ASSISTANT_MESSAGE, data={"text": "present"}),
        ]
        prose = extract_prose_turns(events)
        assert len(prose) == 1
        assert prose[0] == "present"

    def test_returns_tuple(self) -> None:
        events = [_make_event(EventType.ASSISTANT_MESSAGE, data={"text": "hi"})]
        prose = extract_prose_turns(events)
        assert isinstance(prose, tuple)

    def test_multiple_texts_returned_in_order(self) -> None:
        events = [
            _make_event(
                EventType.ASSISTANT_MESSAGE, timestamp=i, data={"text": f"turn {i}"}
            )
            for i in range(3)
        ]
        prose = extract_prose_turns(events)
        assert list(prose) == ["turn 0", "turn 1", "turn 2"]

    def test_single_event_returns_single_turn(self) -> None:
        events = [_make_event(EventType.ASSISTANT_MESSAGE, data={"text": "only one"})]
        prose = extract_prose_turns(events)
        assert len(prose) == 1
        assert prose[0] == "only one"


# ---------------------------------------------------------------------------
# summarize_session
# ---------------------------------------------------------------------------


class TestSummarizeSession:
    def test_summary_captures_hotspots(self) -> None:
        events = [
            _make_write_event("/proj/a.py", ts) for ts in range(HOTSPOT_WRITE_THRESHOLD)
        ]
        summary = summarize_session(events)
        assert "/proj/a.py" in summary.hotspot_files

    def test_summary_captures_bash_failures(self) -> None:
        events = [
            _make_event(
                EventType.BASH_FAILURE, data={"command": "pytest", "exit_code": 1}
            ),
        ]
        summary = summarize_session(events)
        assert len(summary.bash_failures) == 1

    def test_summary_captures_prose_turns(self) -> None:
        events = [
            _make_event(EventType.ASSISTANT_MESSAGE, data={"text": "I chose asyncpg"}),
        ]
        summary = summarize_session(events)
        assert "I chose asyncpg" in summary.prose_turns

    def test_summary_session_id_from_first_event(self) -> None:
        events = [_make_event(EventType.SESSION_START, session_id="abc")]
        summary = summarize_session(events)
        assert summary.session_id == "abc"

    def test_summary_empty_events(self) -> None:
        summary = summarize_session([])
        assert summary.session_id == ""
        assert len(summary.hotspot_files) == 0
        assert len(summary.bash_failures) == 0

    def test_full_loop_simple_fixture(self) -> None:
        events = list(parse_transcript(SIMPLE_FIXTURE))
        summary = summarize_session(events)
        assert len(summary.hotspot_files) >= 1
        assert len(summary.bash_failures) >= 1
        assert len(summary.co_occurring_pairs) >= 1

    def test_co_occurring_pairs_is_tuple(self) -> None:
        events = [
            _make_write_event("/proj/a.py", timestamp=0),
            _make_write_event("/proj/b.py", timestamp=5),
        ]
        summary = summarize_session(events)
        assert isinstance(summary.co_occurring_pairs, tuple)

    def test_multiple_bash_failures_all_captured(self) -> None:
        events = [
            _make_event(
                EventType.BASH_FAILURE,
                timestamp=i,
                data={"command": f"cmd{i}", "exit_code": 1},
            )
            for i in range(3)
        ]
        summary = summarize_session(events)
        assert len(summary.bash_failures) == 3

    def test_session_id_from_first_event_not_last(self) -> None:
        events = [
            _make_event(EventType.SESSION_START, session_id="first"),
            _make_event(EventType.FILE_WRITE, session_id="other", data={"path": "/x"}),
        ]
        summary = summarize_session(events)
        assert summary.session_id == "first"

    def test_hotspot_requires_threshold_writes(self) -> None:
        events = [
            _make_write_event("/proj/rare.py", ts)
            for ts in range(HOTSPOT_WRITE_THRESHOLD - 1)
        ]
        summary = summarize_session(events)
        assert "/proj/rare.py" not in summary.hotspot_files

    def test_multiple_bash_failures_counted(self) -> None:
        events = [
            _make_event(
                EventType.BASH_FAILURE,
                timestamp=i,
                data={"command": f"cmd{i}", "exit_code": 1},
            )
            for i in range(3)
        ]
        summary = summarize_session(events)
        assert len(summary.bash_failures) == 3
