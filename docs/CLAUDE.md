# Cortex — AI Memory System for Claude Code

## What This Project Is

Cortex is a persistent, graph-based memory layer for Claude Code. It intercepts session
lifecycle events, extracts structured memory nodes from transcripts using deterministic
NLP (no LLM calls in the pipeline), stores them in a weighted knowledge graph, and
injects the most relevant compressed context at the start of every new session.

The goal: reduce redundant input tokens by 60–70% across long-running projects while
improving task accuracy by ensuring Claude always has the right project-specific
context, not everything.

## Hard Constraints — Never Violate These

- **No LLM calls in the extraction pipeline.** extract.py, parser.py, decay.py,
  retrieval.py, embedder.py must never call any external API. Use spaCy, tree-sitter,
  sentence-transformers, and rule-based logic only.
- **SQLite is the only database.** No Postgres, Redis, Chroma, or any other store.
  Use sqlite-vec for vector similarity inside SQLite.
- **Local embeddings only.** Use nomic-embed-text via ollama or all-MiniLM-L6-v2 via
  sentence-transformers. Never call OpenAI, Anthropic, or any embedding API.
- **All hook scripts must be idempotent.** Running inject.py or extract.py twice on
  the same session must produce identical results and not duplicate nodes.
- **Hook scripts must complete in under 500ms.** They run on every Claude Code turn.
  Profile before merging anything into hooks/.
- **No ORM.** Raw SQL only. schema.sql is the single source of truth for the schema.
- **No async in core/.** Hooks are synchronous shell scripts. Keep it simple.
- **Python 3.11+ only.** Use match/case, tomllib, typing improvements freely.

## Project Layout

```
cortex/
├── CLAUDE.md                  ← you are here
├── DECISIONS.md               ← append every architectural decision made
├── schema.sql                 ← authoritative DB schema, never edit inline
├── pyproject.toml             ← deps and entry points
├── .cortex/                   ← runtime data dir (gitignored)
│   ├── cortex.db              ← SQLite graph database
│   └── sessions/              ← raw JSONL session transcripts
├── core/
│   ├── parser.py              ← JSONL transcript → typed event dicts
│   ├── extractor.py           ← events → candidate memory nodes
│   ├── embedder.py            ← text → local embedding vectors
│   ├── graph.py               ← SQLite read/write/merge/query
│   ├── decay.py               ← weight updates, tier promotion, eviction
│   └── retrieval.py           ← BM25 + vector + graph-hop score fusion
├── hooks/
│   ├── inject.py              ← SessionStart: query graph → print to stdout
│   ├── extract.py             ← Stop: parse transcript → write graph
│   └── compact.py             ← PostCompact: re-inject after context reset
├── cli/
│   ├── cortex.py              ← entry point: status / inspect / reset / graph
│   └── config.py              ← decay rates, thresholds, paths
├── dashboard/
│   ├── server.py              ← FastAPI + WebSocket server
│   └── app/                   ← React frontend (token chart + D3 graph)
├── plugin/
│   ├── plugin.json            ← Claude Code plugin manifest
│   └── skills/
│       └── cortex-status/
│           └── SKILL.md
└── tests/
    ├── test_parser.py
    ├── test_extractor.py
    ├── test_graph.py
    ├── test_decay.py
    ├── test_retrieval.py
    └── conftest.py
```

## Data Model

These are the canonical field definitions. Do not add fields without updating schema.sql
and appending to DECISIONS.md.

### Node

| Field          | Type    | Notes                                              |
|----------------|---------|----------------------------------------------------|
| id             | TEXT    | UUID4, primary key                                 |
| type           | TEXT    | 'observation' \| 'fact' \| 'convention' \| 'error' |
| tier           | INTEGER | 1 (ephemeral), 2 (semantic), 3 (procedural)        |
| text           | TEXT    | One sentence. No reasoning. Just the fact.         |
| rationale      | TEXT    | One sentence why. Nullable. Decays faster.         |
| embedding      | BLOB    | Float32 numpy array, serialized                    |
| precision_bits | INTEGER | 32, 8, or 2 — downcasted as tier rises             |
| weight         | REAL    | Starts at 1.0, increments on access                |
| project        | TEXT    | Absolute path to project root                      |
| scope          | TEXT    | 'project' \| 'module' \| 'session'                 |
| source         | TEXT    | 'jsonl' \| 'ast' \| 'git' \| 'nlp'                |
| last_accessed  | INTEGER | Unix timestamp                                     |
| created_at     | INTEGER | Unix timestamp                                     |
| session_count  | INTEGER | Number of distinct sessions that accessed this     |

### Edge

| Field     | Type    | Notes                                        |
|-----------|---------|----------------------------------------------|
| source_id | TEXT    | FK → node.id                                 |
| target_id | TEXT    | FK → node.id                                 |
| strength  | REAL    | Increments when both nodes retrieved together|
| last_seen | INTEGER | Unix timestamp of last co-retrieval          |

## Decay Rules

These constants live in config.py. Do not hardcode them elsewhere.

| Tier | Decay rate (per unaccessed session) | Eviction threshold | Promotion threshold      |
|------|-------------------------------------|--------------------|--------------------------|
| 1    | × 0.85                              | weight < 0.3       | weight ≥ 8, sessions ≥ 3 |
| 2    | × 0.95                              | weight < 0.5       | weight ≥ 20, sessions ≥ 8, age ≥ 14 days |
| 3    | never                               | never              | —                        |

Precision downcast happens on promotion: tier 1 → float32, tier 2 → int8, tier 3 → int2.

## Extraction Pipeline

Four parallel channels. All deterministic. No LLM calls.

1. **JSONL parser** — typed event extraction from Claude Code transcript
2. **AST diff** — tree-sitter diff before/after session on touched files
3. **Git signals** — gitpython: commit message, diff stats, file churn
4. **NLP (spaCy)** — named entity + dependency parse on Claude's prose turns only

Each channel produces candidate nodes. Rule-based filter scores each candidate on:
- durability (would this be true in 3 months?)
- scope (session / module / project)
- source signal strength

Candidates below threshold are dropped. Survivors are embedded, deduplicated against
existing graph (cosine similarity > 0.9 → merge), then written.

## Retrieval

Three channels fused by weighted sum. Weights are configurable in config.py.

1. Vector cosine similarity (weight: 0.5)
2. BM25 keyword index (weight: 0.3)
3. Graph hop traversal 1–2 edges from matched nodes (weight: 0.2)

Top-k = 8 by default. All tier-3 nodes are always included regardless of score.
Total injected context budget: 600 tokens max (enforced by tiktoken in inject.py).

## Testing Rules

- Every module in core/ has a corresponding test_*.py
- Tests use in-memory SQLite (:memory:) — no file I/O in unit tests
- No mocking of core logic — test real behavior with real data fixtures
- Minimum coverage: 90% per module, enforced in CI
- Run with: `pytest tests/ -v --cov=core --cov-report=term-missing`

## Build Order

Do not skip steps. Do not start a step until the previous one has passing tests.

1. schema.sql + graph.py (read/write/merge) + test_graph.py
2. parser.py + test_parser.py (use fixture JSONL files)
3. extractor.py + test_extractor.py (JSONL + AST + git + NLP channels)
4. embedder.py (wrapper, test with dummy vectors)
5. decay.py + test_decay.py
6. retrieval.py + test_retrieval.py
7. hooks/extract.py + hooks/inject.py (integration test: full loop)
8. hooks/compact.py
9. cli/cortex.py
10. dashboard/server.py
11. dashboard/app/ (React)
12. plugin/plugin.json

## Code Quality Standards

This project targets enterprise/open-source grade. Every PR must pass:

- `ruff check .` — zero warnings
- `mypy --strict .` — zero errors
- `pytest tests/ --cov=core --cov-fail-under=90`
- `bandit -r core/ hooks/ cli/`
- `black --check .`

Type hints are required on every function signature and class attribute.
Docstrings are required on every public function (Google style).
No bare `except:` clauses. Always catch specific exceptions.

## DECISIONS.md Protocol

After every session where an architectural decision is made, append to DECISIONS.md:

```
## YYYY-MM-DD — <short title>
**Decision:** what was decided
**Rationale:** why
**Alternatives considered:** what else was evaluated
**Impact:** which files are affected
```

## Agent Roles

When working with specialist agents, route tasks as follows:

- **python-pro** — all core/ and hooks/ implementation
- **ai-engineer** — embedder.py, retrieval.py, decay math design
- **refactoring-specialist** — called after each sprint to clean up core/ modules
- **qa-expert** — owns tests/, coverage targets, and integration test design
- **debugger** — called when a hook fails silently or retrieval quality drops
- **git-workflow-manager** — branch strategy, commit conventions, release tagging
- **context-manager** — graph.py schema evolution and query optimization
