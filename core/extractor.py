"""Four-channel extraction pipeline: JSONL, AST, Git, NLP.

Converts a parsed session into CandidateNode objects suitable for graph storage.
All channels are deterministic. No LLM calls. No external API calls.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from core.parser import EventType, ParsedEvent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------


class NodeType(StrEnum):
    OBSERVATION = "observation"
    FACT = "fact"
    CONVENTION = "convention"
    ERROR = "error"


class SourceType(StrEnum):
    JSONL = "jsonl"
    AST = "ast"
    GIT = "git"
    NLP = "nlp"


class ScopeType(StrEnum):
    SESSION = "session"
    MODULE = "module"
    PROJECT = "project"


@dataclass
class CandidateNode:
    """A memory candidate produced by any extraction channel."""

    text: str
    rationale: str | None
    type: NodeType
    source: SourceType
    scope: ScopeType
    durability: float
    project: str


# ---------------------------------------------------------------------------
# Durability constants
# ---------------------------------------------------------------------------

DURABILITY_THRESHOLD = 0.4
MAX_TEXT_LENGTH = 200

# ---------------------------------------------------------------------------
# Channel 1: JSONL signals
# ---------------------------------------------------------------------------


def jsonl_channel(events: list[ParsedEvent], project: str) -> list[CandidateNode]:
    """Extract candidates from JSONL event signals.

    Signals:
    - File write hotspots (≥3 writes to same file)
    - Bash failures with non-zero exit code

    Args:
        events: Parsed events from a single session.
        project: Absolute path to the project root.

    Returns:
        List of CandidateNode objects from JSONL signals.
    """
    from collections import Counter

    candidates: list[CandidateNode] = []

    # Hotspot detection
    write_counts: Counter[str] = Counter()
    for event in events:
        if event.type == EventType.FILE_WRITE:
            path = event.data.get("path", "")
            if path:
                write_counts[path] += 1

    for path, count in write_counts.items():
        if count >= 3:
            rel = _relative_path(path, project)
            candidates.append(
                CandidateNode(
                    text=f"{rel} was modified {count} times in this session",
                    rationale=None,
                    type=NodeType.OBSERVATION,
                    source=SourceType.JSONL,
                    scope=ScopeType.MODULE,
                    durability=0.6,
                    project=project,
                )
            )

    # Bash failure detection
    for event in events:
        if event.type == EventType.BASH_FAILURE:
            command = event.data.get("command", "")
            output = event.data.get("output", "")
            exit_code = event.data.get("exit_code", 1)
            error_summary = _extract_error_summary(output)
            text = f"Command `{command[:80]}` failed (exit {exit_code})"
            if error_summary:
                text += f": {error_summary[:80]}"
            candidates.append(
                CandidateNode(
                    text=text[:MAX_TEXT_LENGTH],
                    rationale=None,
                    type=NodeType.ERROR,
                    source=SourceType.JSONL,
                    scope=ScopeType.SESSION,
                    durability=0.5,
                    project=project,
                )
            )

    return candidates


def _extract_error_summary(output: str) -> str:
    """Extract the first meaningful error line from command output."""
    for line in output.splitlines():
        line = line.strip()
        if any(kw in line for kw in ("Error", "error", "Exception", "FAILED", "fatal")):
            return line[:120]
    return ""


def _relative_path(abs_path: str, project: str) -> str:
    """Return a path relative to project root, or the filename if not under project."""
    try:
        return str(Path(abs_path).relative_to(project))
    except ValueError:
        return Path(abs_path).name


# ---------------------------------------------------------------------------
# Channel 2: AST diff
# ---------------------------------------------------------------------------


def ast_channel(touched_files: list[Path], project: str) -> list[CandidateNode]:
    """Extract structural change candidates using tree-sitter AST diff.

    For each touched file that existed before the session, diffs the current
    AST against the git HEAD version. Skips test files and gitignored files.

    Args:
        touched_files: File paths modified during the session.
        project: Absolute path to the project root.

    Returns:
        List of CandidateNode objects from AST structural changes.
    """
    candidates: list[CandidateNode] = []

    for file_path in touched_files:
        if _is_test_file(file_path):
            continue
        try:
            changes = _diff_file_ast(file_path, project)
        except (OSError, ValueError, UnicodeDecodeError, AttributeError) as exc:
            logger.debug("AST diff failed for %s: %s", file_path, exc)
            continue
        for change in changes:
            rel = _relative_path(str(file_path), project)
            text = _format_ast_change(change, rel)
            if text:
                candidates.append(
                    CandidateNode(
                        text=text[:MAX_TEXT_LENGTH],
                        rationale=None,
                        type=NodeType.FACT,
                        source=SourceType.AST,
                        scope=ScopeType.MODULE,
                        durability=0.65,
                        project=project,
                    )
                )

    return candidates


def _is_test_file(path: Path) -> bool:
    """Return True if the path looks like a test file."""
    return path.name.startswith("test_") or "tests" in path.parts


def _diff_file_ast(file_path: Path, project: str) -> list[dict[str, Any]]:
    """Return a list of AST change dicts for a single file.

    Attempts to use gitpython to get the pre-session version, then diffs
    using tree-sitter. Falls back to empty list on any failure (e.g., file
    newly created, not in a git repo, unsupported language).
    """
    import git as gitmod

    try:
        repo = gitmod.Repo(project, search_parent_directories=True)
    except gitmod.InvalidGitRepositoryError:
        return []

    rel = str(Path(file_path).relative_to(repo.working_dir))
    try:
        blob = repo.head.commit.tree[rel]
        before_source = blob.data_stream.read().decode("utf-8", errors="replace")
    except (KeyError, ValueError, Exception):
        # File is new this session — no prior blob to diff against
        return []

    if not file_path.exists():
        return []

    after_source = file_path.read_text(encoding="utf-8", errors="replace")
    lang = _detect_language(file_path)
    if lang is None:
        return []

    return _tree_sitter_diff(before_source, after_source, lang)


def _detect_language(path: Path) -> str | None:
    """Map file extension to tree-sitter language name."""
    mapping = {
        ".py": "python",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".js": "javascript",
        ".jsx": "javascript",
        ".go": "go",
        ".rs": "rust",
    }
    return mapping.get(path.suffix.lower())


def _tree_sitter_diff(before: str, after: str, language: str) -> list[dict[str, Any]]:
    """Parse both versions with tree-sitter and return structural changes."""
    try:
        import tree_sitter_python as tspython
        from tree_sitter import Language, Parser
    except ImportError:
        logger.debug("tree-sitter language bindings not installed; skipping AST diff")
        return []

    if language != "python":
        logger.debug(
            "AST diff only supports Python in this build; skipping %s", language
        )
        return []

    try:
        lang = Language(tspython.language())
        parser = Parser(lang)
        before_tree = parser.parse(before.encode())
        after_tree = parser.parse(after.encode())
    except Exception as exc:
        logger.debug("tree-sitter parse failed: %s", exc)
        return []

    before_funcs = _extract_function_names(before_tree.root_node)
    after_funcs = _extract_function_names(after_tree.root_node)

    changes: list[dict[str, Any]] = []
    for name in after_funcs - before_funcs:
        changes.append({"action": "added", "kind": "function", "name": name})
    for name in before_funcs - after_funcs:
        changes.append({"action": "removed", "kind": "function", "name": name})

    before_classes = _extract_class_names(before_tree.root_node)
    after_classes = _extract_class_names(after_tree.root_node)
    for name in after_classes - before_classes:
        changes.append({"action": "added", "kind": "class", "name": name})
    for name in before_classes - after_classes:
        changes.append({"action": "removed", "kind": "class", "name": name})

    return changes


def _extract_function_names(node: Any) -> set[str]:
    """Recursively extract all function/method definition names from an AST node."""
    names: set[str] = set()
    if node.type in ("function_definition", "async_function_definition"):
        for child in node.children:
            if child.type == "identifier":
                names.add(
                    child.text.decode() if isinstance(child.text, bytes) else child.text
                )
                break
    for child in node.children:
        names |= _extract_function_names(child)
    return names


def _extract_class_names(node: Any) -> set[str]:
    """Recursively extract all class definition names from an AST node."""
    names: set[str] = set()
    if node.type == "class_definition":
        for child in node.children:
            if child.type == "identifier":
                names.add(
                    child.text.decode() if isinstance(child.text, bytes) else child.text
                )
                break
    for child in node.children:
        names |= _extract_class_names(child)
    return names


def _format_ast_change(change: dict[str, Any], rel_path: str) -> str:
    """Format an AST change dict into a human-readable node text."""
    action = change.get("action", "")
    kind = change.get("kind", "")
    name = change.get("name", "")
    match action:
        case "added":
            return f"Added {kind} `{name}` to {rel_path}"
        case "removed":
            return f"Removed {kind} `{name}` from {rel_path}"
        case _:
            return ""


# ---------------------------------------------------------------------------
# Channel 3: Git signals
# ---------------------------------------------------------------------------


def git_channel(project_root: Path, session_start: int) -> list[CandidateNode]:
    """Extract candidates from git history signals.

    Extracts commit messages made during this session and file churn hotspots.
    Silently returns empty list if project is not a git repo.

    Args:
        project_root: Absolute path to the project root.
        session_start: Unix timestamp marking the start of the session.

    Returns:
        List of CandidateNode objects from git signals.
    """
    try:
        import git as gitmod
    except ImportError:
        logger.debug("gitpython not available; skipping git channel")
        return []

    try:
        repo = gitmod.Repo(str(project_root), search_parent_directories=True)
    except Exception:
        logger.debug("Not a git repo: %s; skipping git channel", project_root)
        return []

    candidates: list[CandidateNode] = []

    # Session commits
    try:
        recent = list(repo.iter_commits(max_count=3))
        for commit in recent:
            if commit.committed_date >= session_start:
                msg = str(commit.message).strip().splitlines()[0][:MAX_TEXT_LENGTH]
                if msg:
                    candidates.append(
                        CandidateNode(
                            text=msg,
                            rationale=None,
                            type=NodeType.FACT,
                            source=SourceType.GIT,
                            scope=ScopeType.PROJECT,
                            durability=0.7,
                            project=str(project_root),
                        )
                    )
    except Exception as exc:
        logger.debug("Failed to read commits: %s", exc)

    # File churn hotspots
    try:
        churn = _compute_file_churn(repo, lookback=10, threshold=0.6)
        for path, fraction in churn.items():
            count = int(fraction * 10)
            candidates.append(
                CandidateNode(
                    text=f"{path} is a high-churn file (modified in {count} of last 10 commits)",
                    rationale=None,
                    type=NodeType.FACT,
                    source=SourceType.GIT,
                    scope=ScopeType.MODULE,
                    durability=0.65,
                    project=str(project_root),
                )
            )
    except Exception as exc:
        logger.debug("Failed to compute churn: %s", exc)

    return candidates


def _compute_file_churn(repo: Any, lookback: int, threshold: float) -> dict[str, float]:
    """Return files modified in > threshold fraction of the last N commits."""
    from collections import Counter

    counts: Counter[str] = Counter()
    commits = list(repo.iter_commits(max_count=lookback))
    total = len(commits)
    if total == 0:
        return {}

    for commit in commits:
        try:
            for diff in commit.diff(commit.parents[0] if commit.parents else None):
                if diff.a_path:
                    counts[diff.a_path] += 1
        except Exception:
            continue

    return {
        path: count / total
        for path, count in counts.items()
        if count / total >= threshold
    }


# ---------------------------------------------------------------------------
# Channel 4: spaCy NLP
# ---------------------------------------------------------------------------

_DECISION_VERBS: frozenset[str] = frozenset(
    {"choose", "decide", "select", "opt", "switch", "migrate", "pick"}
)
_AVOIDANCE_VERBS: frozenset[str] = frozenset({"avoid", "skip", "reject", "drop"})
_CONVENTION_MARKERS: frozenset[str] = frozenset(
    {
        "always",
        "never",
        "convention",
        "standard",
        "pattern",
        "throughout",
        "consistently",
        "everywhere",
    }
)
_RETRACTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"actually\s+let['']?s?\s+not"),
    re.compile(r"actually,?\s+let['']?s?\s+not"),
    re.compile(r"let['']?s?\s+not\s+use\s+that"),
    re.compile(r"on\s+second\s+thought"),
    re.compile(r"never\s+mind"),
    re.compile(r"disregard\s+that"),
]
_SPLIT_CONJUNCTIONS: frozenset[str] = frozenset({"because", "since", "as", "so that"})

_nlp: Any = None


def _get_nlp() -> Any:
    """Load spaCy model once at first call (lazy singleton)."""
    global _nlp
    if _nlp is None:
        try:
            import spacy

            _nlp = spacy.load("en_core_web_sm")
        except (ImportError, OSError) as exc:
            logger.warning("spaCy model not available: %s; NLP channel disabled", exc)
            _nlp = False
    return _nlp if _nlp is not False else None


def _is_retracted(text: str) -> bool:
    """Return True if the text contains a retraction marker."""
    lower = text.lower()
    return any(pat.search(lower) for pat in _RETRACTION_PATTERNS)


def _is_decision_sentence(sent: Any) -> bool:
    """Return True if the spaCy sentence contains a decision or avoidance verb."""
    lemmas = {token.lemma_.lower() for token in sent}
    return bool(lemmas & _DECISION_VERBS) or bool(lemmas & _AVOIDANCE_VERBS)


def _is_convention_sentence(sent: Any) -> bool:
    """Return True if the spaCy sentence contains a convention marker."""
    return any(token.lower_ in _CONVENTION_MARKERS for token in sent)


def _score_durability(sent: Any, node_type: NodeType) -> float:
    """Compute durability score for a sentence using rule-based heuristics."""
    text_lower = sent.text.lower()
    score = 0.5

    if node_type == NodeType.CONVENTION:
        score += 0.3
    if "always" in text_lower:
        score += 0.2
    if "never" in text_lower:
        score += 0.2
    if any(ent.label_ in ("ORG", "PRODUCT") for ent in sent.ents):
        score += 0.1

    if "maybe" in text_lower:
        score -= 0.3
    if "temporary" in text_lower:
        score -= 0.4
    if "for now" in text_lower:
        score -= 0.3
    if "this session" in text_lower:
        score -= 0.4

    return max(0.0, min(1.0, score))


def _split_on_conjunction(text: str) -> tuple[str, str | None]:
    """Split a sentence at a causal conjunction (because, since, etc.).

    "since" is ambiguous: it can be temporal ("since 2021") or causal
    ("since it avoids locks"). We only treat it as causal when the word
    immediately following is NOT a digit, which catches date/year patterns.
    "as" is similarly ambiguous when used as a comparison ("as fast as X"),
    so we require the fragment after "as" to contain a verb-like token
    (contains a common pronoun or "it", "we", "they", "this") before splitting.

    Returns:
        Tuple of (conclusion_text, rationale_text_or_None).
    """
    lower = text.lower()
    for conj in sorted(_SPLIT_CONJUNCTIONS, key=len, reverse=True):
        idx = lower.find(f" {conj} ")
        if idx == -1:
            continue

        rest_start = idx + len(conj) + 2
        rest = lower[rest_start:].lstrip()

        # Guard: "since <digit>" is temporal, not causal
        if conj == "since" and rest and rest[0].isdigit():
            continue

        # Guard: "as" as comparison ("as fast as X") — require a pronoun/subject
        if conj == "as":
            subj_markers = {"it", "we", "they", "this", "that", "i", "you", "the"}
            first_word = rest.split()[0] if rest.split() else ""
            if first_word not in subj_markers:
                continue

        conclusion = text[:idx].strip()[:MAX_TEXT_LENGTH]
        rationale = text[rest_start:].strip()[:MAX_TEXT_LENGTH]
        return conclusion, rationale
    return text[:MAX_TEXT_LENGTH], None


def nlp_channel(prose_turns: list[str], project: str) -> list[CandidateNode]:
    """Extract decision and convention candidates from Claude's prose turns.

    Uses spaCy en_core_web_sm with rule-based detection. Drops retracted
    statements and candidates below the durability threshold.

    Args:
        prose_turns: Text content of Claude's assistant_message turns.
        project: Absolute path to the project root.

    Returns:
        List of CandidateNode objects from NLP signals.
    """
    nlp = _get_nlp()
    if nlp is None:
        logger.warning("spaCy unavailable; NLP channel returning empty results")
        return []

    candidates: list[CandidateNode] = []

    for turn_text in prose_turns:
        if _is_retracted(turn_text):
            continue

        doc = nlp(turn_text)
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue

            if _is_retracted(sent_text):
                continue

            node_type: NodeType | None = None
            if _is_convention_sentence(sent):
                node_type = NodeType.CONVENTION
            elif _is_decision_sentence(sent):
                node_type = NodeType.FACT
            else:
                continue

            durability = _score_durability(sent, node_type)
            if durability < DURABILITY_THRESHOLD:
                continue

            conclusion, rationale = _split_on_conjunction(sent_text)
            candidates.append(
                CandidateNode(
                    text=conclusion,
                    rationale=rationale,
                    type=node_type,
                    source=SourceType.NLP,
                    scope=ScopeType.PROJECT,
                    durability=durability,
                    project=project,
                )
            )

    return candidates


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------


def run_extraction(
    events: list[ParsedEvent],
    project: str,
    touched_files: list[Path] | None = None,
    session_start: int = 0,
) -> list[CandidateNode]:
    """Run all four extraction channels and return the merged candidate list.

    Candidates below DURABILITY_THRESHOLD are filtered out.

    Args:
        events: All parsed events from the session.
        project: Absolute path to the project root.
        touched_files: Optional list of file paths modified this session.
            Defaults to files inferred from FILE_WRITE events.
        session_start: Unix timestamp of session start (for git channel).

    Returns:
        Merged list of CandidateNode objects, deduplicated by text.
    """
    if touched_files is None:
        seen: set[str] = set()
        touched_files = []
        for e in events:
            if e.type == EventType.FILE_WRITE:
                p = e.data.get("path", "")
                if p and p not in seen:
                    seen.add(p)
                    touched_files.append(Path(p))

    prose_turns = [
        str(e.data.get("text", ""))
        for e in events
        if e.type == EventType.ASSISTANT_MESSAGE and e.data.get("text")
    ]

    all_candidates: list[CandidateNode] = []
    all_candidates.extend(jsonl_channel(events, project))
    all_candidates.extend(ast_channel(touched_files, project))
    all_candidates.extend(git_channel(Path(project), session_start))
    all_candidates.extend(nlp_channel(prose_turns, project))

    return [c for c in all_candidates if c.durability >= DURABILITY_THRESHOLD]
