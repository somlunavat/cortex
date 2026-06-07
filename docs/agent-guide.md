# Agent Routing Guide

This document tells you which agent to invoke for each type of task, and what
context to give them. Read the relevant section before starting a Claude Code session.

## Agent Map

| Agent                  | Model  | Owns                                          |
|------------------------|--------|-----------------------------------------------|
| python-pro             | sonnet | core/, hooks/, cli/ implementation            |
| ai-engineer            | opus   | embedder.py, retrieval.py, decay math design  |
| refactoring-specialist | sonnet | Post-sprint cleanup of any module             |
| qa-expert              | sonnet | tests/, coverage, integration test design     |
| debugger               | sonnet | Hook failures, retrieval quality regressions  |
| git-workflow-manager   | haiku  | Branching, commits, tagging, releases         |
| context-manager        | sonnet | graph.py schema evolution, query optimization |

---

## python-pro

**When to use:** Writing or modifying any file in core/, hooks/, or cli/.
This is the workhorse agent for implementation sprints.

**What to tell it:**

> "Read CLAUDE.md and docs/architecture.md first. Then build [module].
> Input: [precise description]. Output: [precise description].
> Follow the hard constraints in CLAUDE.md. Write tests first in tests/test_[module].py.
> Use Google-style docstrings, strict mypy types, ruff-clean code."

**Sprint 1 prompt (graph.py):**
> "Read CLAUDE.md, schema.sql, and docs/architecture.md. Build core/graph.py.
> It wraps a sqlite3.Connection and exposes: write_node, merge_node, find_similar,
> get_all_nodes, update_decay, delete_node, write_edge, get_edges.
> All methods typed. No ORM. Write test_graph.py first using in-memory SQLite.
> Coverage must exceed 95%."

**Sprint 2 prompt (parser.py):**
> "Read CLAUDE.md and docs/extraction-pipeline.md Channel 1 spec. Build core/parser.py.
> Input: Path to a Claude Code JSONL transcript. Output: list[ParsedEvent].
> Implement all EventType variants. Handle malformed lines with logging, not exceptions.
> Write test_parser.py with fixture transcripts in tests/fixtures/transcripts/.
> Create simple.jsonl and with_failures.jsonl fixtures as part of this task."

**Sprint 3 prompt (extractor.py):**
> "Read docs/extraction-pipeline.md in full. Build core/extractor.py with four channels:
> jsonl_channel, ast_channel, git_channel, nlp_channel. Each returns list[CandidateNode].
> Load spaCy en_core_web_sm once at module import. Use tree-sitter for AST. Use gitpython.
> Implement durability scoring per the spec. Write test_extractor.py."

---

## ai-engineer

**When to use:** Designing the embedding strategy, retrieval scoring weights,
decay math, or any component where ML/AI domain knowledge determines correctness.
Use ai-engineer for design, then python-pro for implementation.

**What to tell it:**

> "Read CLAUDE.md, docs/architecture.md, and docs/extraction-pipeline.md.
> Design [component] with these constraints: local only, no API calls, must complete
> in [X]ms. Produce: a precise specification with data types, algorithm, and
> pseudocode that python-pro can implement directly."

**Embedder design prompt:**
> "Design core/embedder.py. Requirements: local embedding only using either
> nomic-embed-text via ollama or all-MiniLM-L6-v2 via sentence-transformers.
> Must preload the model once (not per call). Must serialize embeddings to numpy
> arrays in float32/int8/int2 depending on precision_bits. Must complete in <50ms
> per embedding after warm-up. Produce a spec with class interface and algorithm."

**Retrieval design prompt:**
> "Design core/retrieval.py. Three channels: vector cosine (weight 0.5), BM25 (weight 0.3),
> graph hop 1-2 edges (weight 0.2). Score fusion by weighted sum. Tier-3 nodes always
> included. Token budget 600 enforced with tiktoken. Produce a precise spec with
> data structures and algorithm that python-pro can implement directly."

**Decay math prompt:**
> "Design core/decay.py. Implement the three-tier decay system from docs/architecture.md.
> Decay rates: tier1 × 0.85, tier2 × 0.95, tier3 never. Promotion thresholds per
> the table in CLAUDE.md. Precision downcast on promotion: 32→8→2 bit using numpy dtype.
> Produce pseudocode and edge case handling for python-pro to implement."

---

## qa-expert

**When to use:** After any implementation sprint. Also use to design the fixture
transcripts and expected_nodes JSON before implementation starts.

**What to tell it:**

> "Read CLAUDE.md and docs/testing-strategy.md. [Module] was just implemented.
> Review tests/test_[module].py and core/[module].py.
> Identify: missing test cases, coverage gaps, missing edge cases, missing
> parametrize opportunities. Add tests until coverage >= [target]%.
> Also verify ruff, mypy, and bandit pass on the module."

**Pre-implementation prompt (create fixtures):**
> "Read docs/extraction-pipeline.md sections on what to extract vs drop.
> Create these test fixtures in tests/fixtures/transcripts/:
> - simple.jsonl: 20 events, one file write hotspot, one bash failure
> - with_decisions.jsonl: 10 prose turns containing decision sentences
> - with_retractions.jsonl: prose turns with retracted statements only
> - with_generic_advice.jsonl: generic Python advice, nothing project-specific
> Also create tests/fixtures/expected_nodes/ JSON files for each.
> These fixtures will be used by test_extractor.py and test_quality.py."

---

## refactoring-specialist

**When to use:** After completing a sprint (every 2–3 modules). Do not refactor
mid-sprint — finish the feature first, then clean.

**What to tell it:**

> "Read CLAUDE.md. Review core/[module].py. Run: ruff check, mypy --strict,
> and look for: long functions (>30 lines), repeated patterns, missing abstractions,
> inconsistent naming. Refactor while maintaining 100% test pass rate.
> Zero behavior changes. Update DECISIONS.md if you change any architectural pattern."

**Post-sprint-1 prompt:**
> "Read CLAUDE.md. Review core/graph.py and core/parser.py.
> Detect code smells. Refactor without changing behavior.
> Run pytest after every change. Run ruff and mypy after every change.
> If you extract a shared abstraction, document it in DECISIONS.md."

---

## debugger

**When to use:** When a hook fails silently, when extraction is producing
unexpected nodes, or when retrieval quality drops (wrong context injected).

**What to tell it:**

> "Read CLAUDE.md and docs/extraction-pipeline.md.
> Problem: [describe symptom precisely — what was expected vs what happened,
> which session, what the injected context looked like].
> Investigate: hooks/extract.py, core/extractor.py, core/graph.py.
> Produce: root cause, fix, and a regression test that would have caught this."

**Common failure modes to watch for:**

- `extract.py` runs but writes 0 nodes — usually durability threshold too high
  or JSONL parsing failing silently on a new event type
- `inject.py` exceeds 600 token budget — tier-3 nodes have grown too large
- Retrieval returns wrong project's nodes — project path normalization bug
- Decay run evicts a tier-3 node — should be impossible, check decay.py guard
- spaCy model not loaded — ollama not running, handle gracefully

---

## git-workflow-manager

**When to use:** Setting up the repo initially, creating release branches,
writing commit convention rules, setting up CI.

**Initial setup prompt:**
> "Set up Git workflow for the Cortex project. Requirements:
> - Conventional commits (feat:, fix:, chore:, docs:, test:, refactor:)
> - Main branch protected, all changes via PR
> - Semantic versioning (0.x.y during development, 1.0.0 on first stable release)
> - Pre-commit hooks: ruff, black, mypy
> - Branch naming: feat/*, fix/*, chore/*
> Create: .pre-commit-config.yaml, .github/workflows/test.yml,
> CONTRIBUTING.md with commit format guide."

---

## context-manager

**When to use:** When graph.py needs schema changes, query optimization, or
when the graph grows large enough to need index tuning.

**What to tell it:**

> "Read CLAUDE.md, schema.sql, and core/graph.py.
> Problem: [query is slow / schema needs a new field / index is missing].
> Constraints: SQLite only, no ORM, schema.sql is the source of truth.
> Produce: migration SQL (appended to schema.sql), updated graph.py methods,
> and updated test_graph.py cases. Append to DECISIONS.md."

---

## Multi-Agent Sprint Pattern

For a full sprint building one module:

1. **ai-engineer** → design spec if ML components involved
2. **qa-expert** → create test fixtures before implementation
3. **python-pro** → implement module + tests
4. **qa-expert** → review coverage, add missing cases
5. **refactoring-specialist** → clean up after tests pass
6. **git-workflow-manager** → commit with conventional message

This order prevents rework: design before build, tests before code review,
refactor after behavior is locked.
