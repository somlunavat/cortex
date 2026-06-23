# Changelog

All notable changes to Cortex are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added

- `cortex list` CLI command — paginated node table with `--tier`, `--type`, `--source`, `--sort`, `--limit` filters
- `cortex doctor` CLI command — health check for spaCy model, sentence-transformers, tiktoken, SQLite schema, hook scripts, and plugin manifest
- `cortex pin <node-id>` — force a node to tier 3 (permanent memory, never decayed)
- `cortex annotate <node-id> --rationale "..."` — set or update a node's rationale for richer injection output
- `cortex bump <node-id> --by N` — manually increase a node's weight to reinforce retrieval priority
- `graph.update_node_rationale(node_id, rationale)` — set or clear rationale on any node
- `graph.set_node_weight(node_id, weight)` — set absolute weight on a node (floored at 0.0)
- `cortex clean` CLI command — remove tier-1/2 nodes inactive for N days (`--days`, `--tier`, `--dry-run`)
- `graph.get_stale_nodes(project, days, tier)` — find nodes inactive longer than threshold (excludes tier 3)
- `graph.delete_nodes_bulk(node_ids)` — batch delete in a single SQL statement

- `cortex export` CLI command — dump memory nodes to JSON or CSV for backup and inspection
- `cortex import-` CLI command — restore nodes from a JSON export, recomputing embeddings from text
- `cortex sessions` CLI command — list recent extraction sessions with node and token stats
- `cortex stats` CLI command — aggregate all-time stats: session count, node totals by tier, total tokens saved
- `cortex search --source` flag — filter BM25 search results by extraction channel (jsonl/ast/git/nlp)
- `graph.get_nodes_by_source()` — efficient source-filtered node query using new composite index
- `plugin/skills/cortex-status/SKILL.md` — `/cortex-status` slash command for inline memory summary
- `tests/test_quality.py` — memory quality eval tests and `@pytest.mark.slow` performance regression tests
- `tests/test_cli.py` — CLI integration tests using typer CliRunner (21 tests)
- `tests/fixtures/transcripts/with_failures.jsonl` — realistic multi-failure session fixture
- `tests/fixtures/transcripts/large.jsonl` — 497-event synthetic session for performance testing
- `count_tokens()` public function in `core/retrieval.py`
- Schema indexes: `idx_nodes_project_source`, `idx_nodes_project_scope` for faster filtered queries

### Fixed

- `touch_nodes` now increments `session_count`, fixing tier-1→2 promotion that requires `session_count >= 3`
- `format_injection_block` now shows rationale inline — `• <text> (<rationale>)` — so Claude sees the "why"
- `_split_on_conjunction`: "since" is no longer treated as causal when followed by a digit (temporal guard); "as" is only treated as causal when followed by a subject pronoun
- `get_all_nodes` now returns nodes `ORDER BY weight DESC` instead of arbitrary insertion order
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
