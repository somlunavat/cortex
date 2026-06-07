# Cortex

Persistent, graph-based memory for Claude Code. Reduces redundant context by 60–70% across long-running projects while ensuring Claude always has the right project-specific facts.

## How it works

Three Claude Code hooks form a continuous loop:

```
SessionStart → inject.py  → prints memory block to stdout → context
Stop         → extract.py → parses transcript → writes graph
PostCompact  → compact.py → re-injects after context reset
```

Memory is stored as weighted nodes in a three-tier SQLite graph:

| Tier | Name       | Decay    | Example                                     |
|------|------------|----------|---------------------------------------------|
| 1    | Ephemeral  | ×0.85/session | "auth/middleware.py modified 5× today" |
| 2    | Semantic   | ×0.95/session | "JWT RS256 chosen over HS256"          |
| 3    | Procedural | never    | "Always validate at the API boundary"       |

Extraction uses four deterministic channels — no LLM calls:
- **JSONL parser**: hotspot files, bash failures, co-occurring writes
- **AST diff**: tree-sitter structural changes (functions added/removed)
- **Git signals**: commit messages, file churn rates
- **spaCy NLP**: decision sentences, convention statements

Retrieval fuses BM25 (0.3) + vector cosine (0.5) + graph hop (0.2) within a 600-token budget.

## Installation

```bash
pip install cortex-memory
cortex install
```

Then restart Claude Code. No other configuration needed.

## Commands

```bash
cortex status              # tier counts, last session, token savings
cortex graph               # ASCII adjacency summary
cortex inspect <node-id>   # full node metadata
cortex prune <node-id>     # manually evict a node
cortex reset --project /p  # wipe all nodes for a project
cortex install             # write plugin.json to ~/.claude/plugins/
cortex dashboard           # start web UI on http://localhost:7000
```

## Dashboard

```bash
cortex dashboard
# → http://localhost:7000
```

Features:
- **Token savings chart**: line chart of tokens saved per session
- **Memory graph**: D3 force-directed graph — node size = weight, colour = tier
- **Node inspector**: click any node to see full metadata

## Requirements

- Python 3.11+
- Claude Code CLI
- No external API keys — all embeddings are local (`all-MiniLM-L6-v2`)
- No GPU required

## Architecture

```
core/
  graph.py      SQLite read/write/merge/query
  parser.py     JSONL transcript → typed events
  extractor.py  Events → candidate memory nodes (4 channels)
  embedder.py   Local embeddings + precision quantization
  decay.py      Weight decay, tier promotion, eviction
  retrieval.py  BM25 + vector + graph-hop score fusion

hooks/
  extract.py    Stop hook
  inject.py     SessionStart hook
  compact.py    PostCompact hook

cli/cortex.py   Developer CLI
dashboard/      FastAPI + WebSocket + D3 frontend
```

## Development

```bash
git clone https://github.com/somlunavat/cortex.git
cd cortex
pip install -e ".[dev]"
python -m spacy download en_core_web_sm

pytest tests/ -v --cov=core --cov-fail-under=90
ruff check .
mypy --strict core/ hooks/ cli/
```
