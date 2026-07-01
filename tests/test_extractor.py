"""Tests for core/extractor.py — four-channel extraction pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.extractor import (
    DURABILITY_THRESHOLD,
    NodeType,
    SourceType,
    _is_retracted,
    _score_durability,
    _split_on_conjunction,
    ast_channel,
    jsonl_channel,
    nlp_channel,
    run_extraction,
)
from core.parser import EventType, ParsedEvent, parse_transcript

TEST_PROJECT = "/tmp/cortex_test_project"
SIMPLE_FIXTURE = Path("tests/fixtures/transcripts/simple.jsonl")
DECISIONS_FIXTURE = Path("tests/fixtures/transcripts/with_decisions.jsonl")


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


def _write_event(path: str, timestamp: int = 1000) -> ParsedEvent:
    return _make_event(EventType.FILE_WRITE, timestamp=timestamp, data={"path": path})


def _failure_event(
    command: str = "pytest", exit_code: int = 1, output: str = "Error"
) -> ParsedEvent:
    return _make_event(
        EventType.BASH_FAILURE,
        data={"command": command, "exit_code": exit_code, "output": output},
    )


def _prose_event(text: str) -> ParsedEvent:
    return _make_event(EventType.ASSISTANT_MESSAGE, data={"text": text})


# ---------------------------------------------------------------------------
# Channel 1: jsonl_channel
# ---------------------------------------------------------------------------


class TestJsonlChannel:
    def test_hotspot_produces_observation(self) -> None:
        events = [_write_event(f"{TEST_PROJECT}/auth.py", ts) for ts in range(4)]
        candidates = jsonl_channel(events, TEST_PROJECT)
        obs = [c for c in candidates if c.type == NodeType.OBSERVATION]
        assert len(obs) == 1

    def test_hotspot_text_includes_filename(self) -> None:
        events = [_write_event(f"{TEST_PROJECT}/auth.py", ts) for ts in range(3)]
        candidates = jsonl_channel(events, TEST_PROJECT)
        assert any("auth.py" in c.text for c in candidates)

    def test_bash_failure_produces_error_node(self) -> None:
        events = [
            _failure_event(
                command="pytest tests/",
                exit_code=1,
                output="FAILED tests/test_auth.py::test_jwt",
            )
        ]
        candidates = jsonl_channel(events, TEST_PROJECT)
        errors = [c for c in candidates if c.type == NodeType.ERROR]
        assert len(errors) == 1

    def test_bash_failure_includes_exit_code(self) -> None:
        events = [_failure_event(exit_code=2)]
        candidates = jsonl_channel(events, TEST_PROJECT)
        assert any("exit 2" in c.text for c in candidates)

    def test_below_hotspot_threshold_no_node(self) -> None:
        events = [_write_event(f"{TEST_PROJECT}/auth.py", ts) for ts in range(2)]
        candidates = jsonl_channel(events, TEST_PROJECT)
        obs = [c for c in candidates if c.type == NodeType.OBSERVATION]
        assert len(obs) == 0

    def test_no_events_returns_empty(self) -> None:
        assert jsonl_channel([], TEST_PROJECT) == []

    def test_simple_fixture_produces_candidates(self) -> None:
        events = list(parse_transcript(SIMPLE_FIXTURE))
        candidates = jsonl_channel(events, TEST_PROJECT)
        assert len(candidates) >= 1

    def test_source_is_jsonl(self) -> None:
        events = [_write_event(f"{TEST_PROJECT}/x.py", ts) for ts in range(3)]
        candidates = jsonl_channel(events, TEST_PROJECT)
        assert all(c.source == SourceType.JSONL for c in candidates)

    def test_error_summary_extracted_from_output(self) -> None:
        events = [
            _failure_event(
                output="lots of output\nImportError: cannot import 'foo'\nmore"
            )
        ]
        candidates = jsonl_channel(events, TEST_PROJECT)
        errors = [c for c in candidates if c.type == NodeType.ERROR]
        assert any("ImportError" in c.text for c in errors)

    def test_text_truncated_to_max_length(self) -> None:
        long_cmd = "a" * 300
        events = [_failure_event(command=long_cmd)]
        candidates = jsonl_channel(events, TEST_PROJECT)
        for c in candidates:
            assert len(c.text) <= 200


# ---------------------------------------------------------------------------
# Channel 2: ast_channel
# ---------------------------------------------------------------------------


class TestAstChannel:
    def test_empty_file_list_returns_empty(self) -> None:
        assert ast_channel([], TEST_PROJECT) == []

    def test_test_files_skipped(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test_auth.py"
        test_file.write_text("def test_foo(): pass")
        result = ast_channel([test_file], TEST_PROJECT)
        assert result == []

    def test_non_git_repo_returns_empty(self, tmp_path: Path) -> None:
        py_file = tmp_path / "app.py"
        py_file.write_text("def hello(): pass")
        result = ast_channel([py_file], str(tmp_path))
        assert result == []

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        missing = tmp_path / "ghost.py"
        result = ast_channel([missing], str(tmp_path))
        assert result == []

    def test_source_is_ast(self, tmp_path: Path) -> None:
        result = ast_channel([], TEST_PROJECT)
        assert all(c.source == SourceType.AST for c in result)


# ---------------------------------------------------------------------------
# Channel 4: nlp_channel (NLP — Channel 3 git is integration-only)
# ---------------------------------------------------------------------------


class TestNlpChannel:
    def test_decision_sentence_produces_fact_node(self) -> None:
        prose = [
            "I chose asyncpg over SQLAlchemy because asyncpg has native async support."
        ]
        candidates = nlp_channel(prose, TEST_PROJECT)
        facts = [c for c in candidates if c.type == NodeType.FACT]
        assert len(facts) >= 1

    def test_convention_sentence_produces_convention_node(self) -> None:
        prose = ["We always validate input at the API boundary."]
        candidates = nlp_channel(prose, TEST_PROJECT)
        conventions = [c for c in candidates if c.type == NodeType.CONVENTION]
        assert len(conventions) >= 1

    def test_retracted_statement_not_extracted(self) -> None:
        prose = ["Actually let's not use that approach — it will cause issues."]
        candidates = nlp_channel(prose, TEST_PROJECT)
        assert len(candidates) == 0

    def test_generic_reasoning_not_extracted(self) -> None:
        prose = ["Let me think about the best approach here."]
        candidates = nlp_channel(prose, TEST_PROJECT)
        assert len(candidates) == 0

    def test_because_split_produces_rationale(self) -> None:
        prose = ["I chose asyncpg because it has better async performance."]
        candidates = nlp_channel(prose, TEST_PROJECT)
        facts = [c for c in candidates if c.type == NodeType.FACT and c.rationale]
        assert len(facts) >= 1
        assert any("performance" in (c.rationale or "") for c in facts)

    def test_source_is_nlp(self) -> None:
        prose = ["I selected JWT with RS256 signing for the auth layer."]
        candidates = nlp_channel(prose, TEST_PROJECT)
        assert all(c.source == SourceType.NLP for c in candidates)

    def test_empty_prose_returns_empty(self) -> None:
        assert nlp_channel([], TEST_PROJECT) == []

    def test_decisions_fixture_extracts_at_least_one_fact(self) -> None:
        events = list(parse_transcript(DECISIONS_FIXTURE))
        prose = [
            str(e.data.get("text", ""))
            for e in events
            if e.type == EventType.ASSISTANT_MESSAGE and e.data.get("text")
        ]
        candidates = nlp_channel(prose, TEST_PROJECT)
        facts = [c for c in candidates if c.type == NodeType.FACT]
        assert len(facts) >= 1

    def test_low_durability_dropped(self) -> None:
        prose = ["For now, maybe we can use this temporary approach."]
        candidates = nlp_channel(prose, TEST_PROJECT)
        assert all(c.durability >= DURABILITY_THRESHOLD for c in candidates)

    def test_always_boosts_convention_durability(self) -> None:
        prose = ["We always use async/await throughout the codebase."]
        candidates = nlp_channel(prose, TEST_PROJECT)
        conventions = [c for c in candidates if c.type == NodeType.CONVENTION]
        assert all(c.durability >= 0.8 for c in conventions)


# ---------------------------------------------------------------------------
# Helper function unit tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_is_retracted_detects_actually_lets_not(self) -> None:
        assert _is_retracted("Actually let's not use that approach.")

    def test_is_retracted_case_insensitive(self) -> None:
        assert _is_retracted("ACTUALLY LET'S NOT use redis.")

    def test_is_retracted_false_for_normal_text(self) -> None:
        assert not _is_retracted("I chose asyncpg for performance.")

    def test_split_on_conjunction_because(self) -> None:
        conclusion, rationale = _split_on_conjunction(
            "I chose asyncpg because it is faster."
        )
        assert "asyncpg" in conclusion
        assert rationale is not None
        assert "faster" in rationale

    def test_split_on_conjunction_no_conjunction(self) -> None:
        conclusion, rationale = _split_on_conjunction("We use asyncpg.")
        assert "asyncpg" in conclusion
        assert rationale is None

    def test_split_on_conjunction_truncates_to_max(self) -> None:
        long = "word " * 100
        conclusion, _ = _split_on_conjunction(long)
        assert len(conclusion) <= 200

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("temporary fix for now this session", False),
            ("We always use this pattern", True),
            ("maybe we could avoid this", False),
        ],
    )
    def test_score_durability_returns_above_threshold_for_stable_text(
        self, text: str, expected: bool
    ) -> None:
        import spacy

        nlp_model = spacy.load("en_core_web_sm")
        doc = nlp_model(text)
        sent = next(iter(doc.sents))
        node_type = NodeType.CONVENTION if "always" in text.lower() else NodeType.FACT
        score = _score_durability(sent, node_type)
        if expected:
            assert score >= DURABILITY_THRESHOLD
        else:
            assert score < DURABILITY_THRESHOLD + 0.3


# ---------------------------------------------------------------------------
# run_extraction — integration
# ---------------------------------------------------------------------------


class TestRunExtraction:
    def test_simple_fixture_produces_candidates(self) -> None:
        events = list(parse_transcript(SIMPLE_FIXTURE))
        candidates = run_extraction(events, TEST_PROJECT)
        assert len(candidates) >= 1

    def test_all_candidates_above_durability_threshold(self) -> None:
        events = list(parse_transcript(SIMPLE_FIXTURE))
        candidates = run_extraction(events, TEST_PROJECT)
        assert all(c.durability >= DURABILITY_THRESHOLD for c in candidates)

    def test_decisions_fixture_produces_nlp_facts(self) -> None:
        events = list(parse_transcript(DECISIONS_FIXTURE))
        candidates = run_extraction(events, TEST_PROJECT)
        nlp_facts = [c for c in candidates if c.source == SourceType.NLP]
        assert len(nlp_facts) >= 1

    def test_retracted_text_absent_from_candidates(self) -> None:
        events = list(parse_transcript(DECISIONS_FIXTURE))
        candidates = run_extraction(events, TEST_PROJECT)
        assert all("actually" not in c.text.lower() for c in candidates)

    def test_empty_events_returns_empty(self) -> None:
        assert run_extraction([], TEST_PROJECT) == []

    def test_touched_files_can_be_overridden(self) -> None:
        events = list(parse_transcript(SIMPLE_FIXTURE))
        candidates = run_extraction(events, TEST_PROJECT, touched_files=[])
        assert all(c.source != SourceType.AST for c in candidates)

    def test_candidate_volume_within_spec(self) -> None:
        events = list(parse_transcript(SIMPLE_FIXTURE))
        candidates = run_extraction(events, TEST_PROJECT)
        assert 0 <= len(candidates) <= 10


# ---------------------------------------------------------------------------
# AST channel — with a real temporary git repo
# ---------------------------------------------------------------------------


def _init_git_repo(path: Path) -> None:
    """Create a minimal git repo with one committed Python file."""
    import subprocess

    (path / "app.py").write_text("def hello():\n    pass\n")
    subprocess.run(["git", "init", str(path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@test.com"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "add", "."], check=True, capture_output=True
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        check=True,
        capture_output=True,
    )


class TestAstChannelWithGit:
    def test_added_function_detected(self, tmp_path: Path) -> None:
        _init_git_repo(tmp_path)
        app = tmp_path / "app.py"
        app.write_text("def hello():\n    pass\n\ndef new_function():\n    pass\n")
        candidates = ast_channel([app], str(tmp_path))
        assert any("new_function" in c.text for c in candidates)

    def test_removed_function_detected(self, tmp_path: Path) -> None:
        _init_git_repo(tmp_path)
        app = tmp_path / "app.py"
        app.write_text("# hello removed\n")
        candidates = ast_channel([app], str(tmp_path))
        assert any("hello" in c.text and "Removed" in c.text for c in candidates)

    def test_unchanged_file_no_candidates(self, tmp_path: Path) -> None:
        _init_git_repo(tmp_path)
        app = tmp_path / "app.py"
        candidates = ast_channel([app], str(tmp_path))
        assert candidates == []

    def test_new_file_not_in_git_skipped(self, tmp_path: Path) -> None:
        _init_git_repo(tmp_path)
        new_file = tmp_path / "new_module.py"
        new_file.write_text("def foo(): pass\n")
        candidates = ast_channel([new_file], str(tmp_path))
        assert candidates == []

    def test_source_is_ast(self, tmp_path: Path) -> None:
        _init_git_repo(tmp_path)
        app = tmp_path / "app.py"
        app.write_text("def hello():\n    pass\n\ndef added(): pass\n")
        candidates = ast_channel([app], str(tmp_path))
        assert all(c.source == SourceType.AST for c in candidates)

    def test_added_class_detected(self, tmp_path: Path) -> None:
        _init_git_repo(tmp_path)
        app = tmp_path / "app.py"
        app.write_text("def hello():\n    pass\n\nclass NewClass:\n    pass\n")
        candidates = ast_channel([app], str(tmp_path))
        assert any("NewClass" in c.text for c in candidates)

    def test_unsupported_language_returns_empty(self, tmp_path: Path) -> None:
        _init_git_repo(tmp_path)
        rb_file = tmp_path / "app.rb"
        rb_file.write_text("def hello; end\n")
        import subprocess

        subprocess.run(
            ["git", "-C", str(tmp_path), "add", "."], check=True, capture_output=True
        )
        subprocess.run(
            ["git", "-C", str(tmp_path), "commit", "-m", "add rb"],
            check=True,
            capture_output=True,
        )
        rb_file.write_text("def hello; end\ndef new_func; end\n")
        candidates = ast_channel([rb_file], str(tmp_path))
        assert candidates == []


# ---------------------------------------------------------------------------
# Git channel — with a real temporary git repo
# ---------------------------------------------------------------------------


from core.extractor import git_channel  # noqa: E402


class TestGitChannel:
    def test_non_git_dir_returns_empty(self, tmp_path: Path) -> None:
        candidates = git_channel(tmp_path, session_start=0)
        assert candidates == []

    def test_session_commit_produces_fact(self, tmp_path: Path) -> None:
        import subprocess
        import time

        _init_git_repo(tmp_path)
        before = int(time.time()) - 1
        (tmp_path / "app.py").write_text("def hello(): pass\ndef world(): pass\n")
        subprocess.run(
            ["git", "-C", str(tmp_path), "add", "."], check=True, capture_output=True
        )
        subprocess.run(
            ["git", "-C", str(tmp_path), "commit", "-m", "feat: add world function"],
            check=True,
            capture_output=True,
        )
        candidates = git_channel(tmp_path, session_start=before)
        facts = [c for c in candidates if c.source == SourceType.GIT]
        assert len(facts) >= 1
        assert any("world" in c.text for c in facts)

    def test_old_commit_not_included(self, tmp_path: Path) -> None:
        import time

        _init_git_repo(tmp_path)
        # session starts after the initial commit
        candidates = git_channel(tmp_path, session_start=int(time.time()) + 100)
        git_facts = [
            c
            for c in candidates
            if c.source == SourceType.GIT
            and c.type.value == "fact"
            and "feat" in c.text.lower()
        ]
        assert len(git_facts) == 0

    def test_source_is_git(self, tmp_path: Path) -> None:
        _init_git_repo(tmp_path)
        candidates = git_channel(tmp_path, session_start=0)
        assert all(c.source == SourceType.GIT for c in candidates)


# ---------------------------------------------------------------------------
# Helper functions — additional coverage
# ---------------------------------------------------------------------------


class TestHelpersCoverage:
    def test_extract_error_summary_no_keyword_returns_empty(self) -> None:
        from core.extractor import _extract_error_summary

        result = _extract_error_summary("no keywords here\njust normal output")
        assert result == ""

    def test_format_ast_change_unknown_action(self) -> None:
        from core.extractor import _format_ast_change

        result = _format_ast_change(
            {"action": "renamed", "kind": "function", "name": "foo"}, "app.py"
        )
        assert result == ""

    def test_format_ast_change_added(self) -> None:
        from core.extractor import _format_ast_change

        result = _format_ast_change(
            {"action": "added", "kind": "function", "name": "process"}, "api/handler.py"
        )
        assert "Added" in result and "process" in result

    def test_format_ast_change_removed(self) -> None:
        from core.extractor import _format_ast_change

        result = _format_ast_change(
            {"action": "removed", "kind": "class", "name": "OldAuth"}, "auth.py"
        )
        assert "Removed" in result and "OldAuth" in result

    def test_is_test_file_by_name(self) -> None:
        from core.extractor import _is_test_file

        assert _is_test_file(Path("test_auth.py"))

    def test_is_test_file_by_directory(self) -> None:
        from core.extractor import _is_test_file

        assert _is_test_file(Path("/project/tests/auth.py"))

    def test_is_test_file_false_for_normal(self) -> None:
        from core.extractor import _is_test_file

        assert not _is_test_file(Path("/project/core/auth.py"))

    def test_get_nlp_returns_model(self) -> None:
        from core.extractor import _get_nlp

        nlp_model = _get_nlp()
        assert nlp_model is not None

    def test_get_nlp_singleton(self) -> None:
        from core.extractor import _get_nlp

        first = _get_nlp()
        second = _get_nlp()
        assert first is second

    def test_tree_sitter_diff_added_function(self) -> None:
        from core.extractor import _tree_sitter_diff

        before = "def hello():\n    pass\n"
        after = "def hello():\n    pass\n\ndef new_func():\n    pass\n"
        changes = _tree_sitter_diff(before, after, "python")
        assert any(c["action"] == "added" and c["name"] == "new_func" for c in changes)

    def test_tree_sitter_diff_removed_function(self) -> None:
        from core.extractor import _tree_sitter_diff

        before = "def hello():\n    pass\n\ndef old_func():\n    pass\n"
        after = "def hello():\n    pass\n"
        changes = _tree_sitter_diff(before, after, "python")
        assert any(
            c["action"] == "removed" and c["name"] == "old_func" for c in changes
        )

    def test_tree_sitter_diff_unsupported_lang_returns_empty(self) -> None:
        from core.extractor import _tree_sitter_diff

        changes = _tree_sitter_diff(
            "fn main() {}", "fn main() {} fn other() {}", "rust"
        )
        assert changes == []

    def test_extract_function_names(self) -> None:
        import tree_sitter_python as tspython
        from tree_sitter import Language, Parser

        from core.extractor import _extract_function_names

        lang = Language(tspython.language())
        parser = Parser(lang)
        tree = parser.parse(b"def foo(): pass\ndef bar(): pass\n")
        names = _extract_function_names(tree.root_node)
        assert "foo" in names and "bar" in names

    def test_extract_class_names(self) -> None:
        import tree_sitter_python as tspython
        from tree_sitter import Language, Parser

        from core.extractor import _extract_class_names

        lang = Language(tspython.language())
        parser = Parser(lang)
        tree = parser.parse(b"class MyClass:\n    pass\n")
        names = _extract_class_names(tree.root_node)
        assert "MyClass" in names

    # ---------------------------------------------------------------------------
    # _split_on_conjunction — ambiguity guards
    # ---------------------------------------------------------------------------

    def test_split_since_causal(self) -> None:
        conclusion, rationale = _split_on_conjunction(
            "We chose asyncio since it avoids GIL issues."
        )
        assert "asyncio" in conclusion
        assert rationale is not None
        assert "GIL" in rationale

    def test_split_since_temporal_not_split(self) -> None:
        """'since 2021' is temporal — should not produce a rationale."""
        _conclusion, rationale = _split_on_conjunction(
            "We have used pytest since 2021."
        )
        assert rationale is None

    def test_split_as_causal_with_pronoun(self) -> None:
        _conclusion, rationale = _split_on_conjunction(
            "We use WAL mode as it allows concurrent reads."
        )
        assert rationale is not None
        assert "concurrent" in rationale

    def test_split_as_comparison_not_split(self) -> None:
        """'as fast as' is a comparison — should not split."""
        _conclusion, rationale = _split_on_conjunction(
            "SQLite is as fast as Postgres for single-file workloads."
        )
        assert rationale is None

    def test_split_because_still_works(self) -> None:
        _conclusion, rationale = _split_on_conjunction(
            "We dropped Redis because it added an infra dependency."
        )
        assert rationale is not None
        assert "infra" in rationale


# ---------------------------------------------------------------------------
# nlp_channel — spaCy unavailable path (lines 479-481, 628-629)
# ---------------------------------------------------------------------------


class TestNlpChannelFallback:
    def test_nlp_channel_returns_empty_when_nlp_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import core.extractor as ext_mod

        monkeypatch.setattr(ext_mod, "_nlp", False)
        monkeypatch.setattr(ext_mod, "_model_loaded", False, raising=False)
        result = ext_mod.nlp_channel(["we always use async"], "/tmp/proj")
        assert result == []
        monkeypatch.setattr(ext_mod, "_nlp", None)

    def test_get_nlp_logs_warning_on_import_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import core.extractor as ext_mod

        original_nlp = ext_mod._nlp
        monkeypatch.setattr(ext_mod, "_nlp", None)

        import builtins

        real_import = builtins.__import__

        def mock_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "spacy":
                raise ImportError("spacy not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        result = ext_mod._get_nlp()
        assert result is None
        monkeypatch.setattr(ext_mod, "_nlp", original_nlp)


# ---------------------------------------------------------------------------
# _process_sentence — unclassified branch (line 512) and _process_turn
# retracted turn (line 575)
# ---------------------------------------------------------------------------


class TestProcessSentenceAndTurn:
    def test_process_sentence_returns_none_for_generic_text(self) -> None:
        from core.extractor import _get_nlp, _process_sentence

        nlp = _get_nlp()
        if nlp is None:
            pytest.skip("spaCy not available")
        doc = nlp("The sky is blue and the grass is green.")
        for sent in doc.sents:
            result = _process_sentence(sent, "/tmp/proj")
            assert result is None

    def test_process_turn_returns_empty_for_retracted_turn(self) -> None:
        from core.extractor import _get_nlp, _process_turn

        nlp = _get_nlp()
        if nlp is None:
            pytest.skip("spaCy not available")
        result = _process_turn(nlp, "actually, let's not do that", "/tmp/proj")
        assert result == []

    def test_process_sentence_convention_node_returned(self) -> None:
        from core.extractor import _get_nlp, _process_sentence

        nlp = _get_nlp()
        if nlp is None:
            pytest.skip("spaCy not available")
        doc = nlp("We always use async/await for all I/O operations.")
        results = [_process_sentence(s, "/tmp/proj") for s in doc.sents]
        non_none = [r for r in results if r is not None]
        assert len(non_none) >= 1
        assert any(r.type == "convention" for r in non_none)

    def test_process_turn_retracted_whole_turn(self) -> None:
        from core.extractor import _get_nlp, _process_turn

        nlp = _get_nlp()
        if nlp is None:
            pytest.skip("spaCy not available")
        result = _process_turn(
            nlp,
            "Hmm, ignore that. Actually scratch that plan.",
            "/tmp/proj",
        )
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# _compute_file_churn — empty repo branch (line 421)
# ---------------------------------------------------------------------------


class TestComputeFileChurn:
    def test_churn_zero_commits_returns_empty_dict(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import git

        from core.extractor import _compute_file_churn

        repo = git.Repo.init(str(tmp_path))

        monkeypatch.setattr(repo.__class__, "iter_commits", lambda *a, **kw: iter([]))
        result = _compute_file_churn(repo, lookback=10, threshold=0.5)
        assert result == {}

    def test_churn_with_single_file_modified_every_commit(self, tmp_path: Path) -> None:
        import git

        from core.extractor import _compute_file_churn

        repo = git.Repo.init(str(tmp_path))
        repo.config_writer().set_value("user", "name", "Test").release()
        repo.config_writer().set_value("user", "email", "t@t.com").release()
        hot = tmp_path / "hot.py"
        for i in range(5):
            hot.write_text(f"x = {i}")
            repo.index.add(["hot.py"])
            repo.index.commit(f"change {i}")
        result = _compute_file_churn(repo, lookback=5, threshold=0.5)
        assert "hot.py" in result
        assert result["hot.py"] > 0.5


# ---------------------------------------------------------------------------
# git_channel — ImportError branch (lines 358-360) and OSError (388-389)
# ---------------------------------------------------------------------------


class TestGitChannelEdgeCases:
    def test_git_channel_returns_empty_on_import_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import builtins

        import core.extractor as ext_mod

        real_import = builtins.__import__

        def block_git(name: str, *args: object, **kwargs: object) -> object:
            if name == "git":
                raise ImportError("gitpython not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", block_git)
        result = ext_mod.git_channel(tmp_path, session_start=0)
        assert result == []

    def test_git_channel_skips_non_git_dir(self, tmp_path: Path) -> None:
        from core.extractor import git_channel

        result = git_channel(tmp_path, session_start=0)
        assert result == []

    def test_git_channel_commit_error_is_swallowed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import git

        from core.extractor import git_channel

        repo = git.Repo.init(str(tmp_path))
        repo.config_writer().set_value("user", "name", "Test").release()
        repo.config_writer().set_value("user", "email", "t@t.com").release()
        f = tmp_path / "a.py"
        f.write_text("x = 1")
        repo.index.add(["a.py"])
        repo.index.commit("init")

        original_iter = repo.__class__.iter_commits

        def bad_iter(self: object, *a: object, **kw: object) -> None:
            raise OSError("disk error")

        monkeypatch.setattr(repo.__class__, "iter_commits", bad_iter)
        result = git_channel(tmp_path, session_start=0)
        assert isinstance(result, list)
        monkeypatch.setattr(repo.__class__, "iter_commits", original_iter)
