# Changelog

All notable changes to Cortex are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added

- `cortex export` CLI command — dump memory nodes to JSON or CSV for backup and inspection
- `plugin/skills/cortex-status/SKILL.md` — `/cortex-status` slash command for inline memory summary
- `tests/test_quality.py` — memory quality eval tests and `@pytest.mark.slow` performance regression tests
- `tests/fixtures/transcripts/with_failures.jsonl` — realistic multi-failure session fixture
- `tests/fixtures/transcripts/large.jsonl` — 497-event synthetic session for performance testing
- `count_tokens()` public function in `core/retrieval.py`

### Fixed

- `graph.write_session` now persists `tokens_raw` and `tokens_injected` so the dashboard token savings chart has real data
- Dashboard `api_sessions` endpoint now includes `nodes_promoted` column
- `hooks/extract.py` computes projected token savings at session end

---

## [0.1.0] — 2026-06-07

### Added

- Three-tier SQLite knowledge graph (`core/graph.py`) — ephemeral, semantic, procedural nodes
- Four-channel deterministic extraction pipeline (`core/extractor.py`):
  - JSONL parser: hotspot files, bash failures
  - AST diff: tree-sitter structural changes (functions added/removed)
  - Git signals: commit messages, file churn rates
  - spaCy NLP: decision sentences, convention statements with retraction filtering
- Local embedding model (`core/embedder.py`) — `all-MiniLM-L6-v2` via sentence-transformers; int8/int2 precision quantization on tier promotion
- BM25 + vector cosine + graph-hop score fusion retrieval (`core/retrieval.py`) with 600-token budget enforcement via tiktoken
- Weight decay, tier promotion, and node eviction (`core/decay.py`)
- Three Claude Code hooks:
  - `hooks/inject.py` — SessionStart: retrieve and print injection block
  - `hooks/extract.py` — Stop: parse transcript, write graph, run decay
  - `hooks/compact.py` — PostCompact: re-inject after context reset
- Developer CLI (`cli/cortex.py`): `status`, `graph`, `inspect`, `prune`, `reset`, `search`, `decay`, `install`, `dashboard`
- FastAPI + WebSocket dashboard (`dashboard/server.py`) with D3 force-directed graph and Chart.js token savings chart
- Claude Code plugin manifest (`plugin/plugin.json`)
- Schema migrations via `cli/config._apply_migrations` (idempotent on every DB open)
- Embedding deserialization LRU cache in `core/graph.py`
- Batch embedding (`embed_batch`) to reduce model forward passes during extraction
- `get_edges_for_nodes` bulk edge fetch to reduce SQL round trips in graph_channel
- Full test suite: 315+ tests across all modules with ≥ 90% coverage target
- GitHub Actions CI: lint (ruff + black), type check (mypy --strict), unit tests, security scan (bandit)
