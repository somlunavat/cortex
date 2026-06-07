# Cortex Architecture

## Overview

Cortex is a local, zero-API-cost memory layer that runs alongside Claude Code.
It hooks into Claude Code's session lifecycle, extracts structured memory nodes
from session transcripts using deterministic NLP, stores them in a weighted
knowledge graph, and injects compressed context at session start.

```
┌─────────────────────────────────────────────────────────────┐
│                     Claude Code CLI                         │
│                                                             │
│  SessionStart hook ──→ inject.py ──→ stdout ──→ context     │
│  PostCompact hook  ──→ compact.py ──→ stdout ──→ context     │
│  Stop hook         ──→ extract.py ──→ SQLite graph          │
└─────────────────────────────────────────────────────────────┘
           │                              │
           ▼                              ▼
   ┌───────────────┐            ┌──────────────────┐
   │  Retrieval    │            │  Extraction      │
   │  Engine       │            │  Pipeline        │
   │               │            │                  │
   │  BM25         │            │  JSONL parser    │
   │  + vector     │            │  + AST diff      │
   │  + graph hop  │            │  + git signals   │
   │               │            │  + spaCy NLP     │
   └───────┬───────┘            └────────┬─────────┘
           │                             │
           └──────────┬──────────────────┘
                      ▼
            ┌──────────────────┐
            │  SQLite Graph    │
            │  (cortex.db)     │
            │                  │
            │  nodes table     │
            │  edges table     │
            │  sessions table  │
            └──────────────────┘
```

## Memory Hierarchy

Cortex models memory as a three-tier hierarchy inspired by RAPTOR (recursive
abstraction) and SuperLocalMemory (precision quantization).

### Tier 1 — Ephemeral (Observations)

Raw signals extracted from a single session. High volume, short-lived.

- Source: JSONL events, failed command exits, first-time file touches
- Precision: float32 (full embedding fidelity)
- Decay: weight × 0.85 per unaccessed session
- Eviction: weight < 0.3
- Promotion: weight ≥ 8 AND session_count ≥ 3 → Tier 2
- Example: "Modified auth/middleware.py in this session"

### Tier 2 — Semantic (Facts)

Consolidated patterns that have recurred across sessions. Medium volume, moderate
lifespan. Access-weighted: used facts survive, stale facts fade.

- Source: Promoted from Tier 1, git commit messages, spaCy NLP decisions
- Precision: int8 (8-bit quantized embedding)
- Decay: weight × 0.95 per unaccessed session
- Eviction: weight < 0.5
- Promotion: weight ≥ 20 AND session_count ≥ 8 AND age ≥ 14 days → Tier 3
- Example: "auth/middleware.py handles JWT validation and is frequently modified"

### Tier 3 — Procedural (Conventions)

Project-wide conventions that are permanently stable. Low volume, never evicted.
Always injected regardless of retrieval score.

- Source: Promoted from Tier 2 only (never written directly)
- Precision: int2 (2-bit quantized embedding)
- Decay: none
- Eviction: never (manual pruning via CLI only)
- Example: "Project uses async/await throughout, no callbacks"

## Extraction Pipeline

Four parallel channels produce candidate nodes after every session ends.
All channels are deterministic. No LLM calls.

### Channel 1: JSONL Parser

Reads the Claude Code session transcript (JSONL). Extracts typed events:

```python
# Event types produced by parser.py
class EventType(str, Enum):
    FILE_READ      = "file_read"
    FILE_WRITE     = "file_write"
    BASH_EXEC      = "bash_exec"
    BASH_SUCCESS   = "bash_success"
    BASH_FAILURE   = "bash_failure"
    TOOL_USE       = "tool_use"
    SESSION_START  = "session_start"
    SESSION_END    = "session_end"
```

High-value signals: bash_failure (error pattern), repeated file_write on same file
(hotspot), co-occurring file_write pairs (coupling signal).

### Channel 2: AST Diff

Runs tree-sitter on files touched during the session. Diffs AST before vs after.

Extracts:
- Functions added / removed / renamed
- Classes added / removed
- Import additions (new dependency)
- Cyclomatic complexity delta per function

Produces node type `fact` with source `ast`.

### Channel 3: Git Signals

Runs gitpython after session end (if project is a git repo).

Extracts:
- Commit message (if commit was made) → convention candidate
- Files changed + line delta → churn metric
- Branch name → task context
- File churn rate (files changed > 3 sessions in a row → hotspot fact)

### Channel 4: spaCy NLP

Runs only on Claude's prose output turns (not tool outputs, not user messages).
Applies spaCy `en_core_web_sm` pipeline with custom rules.

Extracts:
- Decision sentences: contains "chose", "decided", "use X because", "avoid Y"
- Convention statements: contains "always", "never", "convention", "standard"
- Error resolutions: contains "fixed", "resolved", "the issue was"

Each extraction is scored:
- `durability`: would this be true in 3 months? (rule-based heuristic)
- `scope`: session / module / project (based on noun phrases present)

Candidates below durability threshold 0.4 are dropped before embedding.

## Retrieval Engine

Three channels fused by weighted sum. Produces top-k nodes for injection.

```
query = current task description (first user message of session)

channel_1 = vector_cosine_similarity(embed(query), all_node_embeddings)
channel_2 = bm25_score(query, all_node_texts)
channel_3 = graph_hop(top_3_vector_results, hops=2)

score = 0.5 * channel_1 + 0.3 * channel_2 + 0.2 * channel_3

# Tier-3 nodes always included regardless of score
result = tier3_nodes + top_k(remaining_nodes_by_score, k=8)
```

### Deduplication on Write

Before any candidate node is written, retrieval runs cosine similarity against
existing nodes for the same project. If similarity > 0.9, the existing node is
updated (weight incremented, text merged if different) rather than a new node
being created. This prevents near-duplicate accumulation.

## Injection Format

inject.py prints to stdout. Claude Code captures this as session context.
Format is a compact structured block, not JSON, to read naturally.

```
=== CORTEX MEMORY (PROJECT: /path/to/project) ===

[CONVENTIONS — always apply]
• Async/await throughout, no callbacks or raw threads
• PostgreSQL via asyncpg, no ORM, raw SQL in db/queries.py
• All API responses use CortexResponse dataclass from core/types.py

[RELEVANT CONTEXT — this session]
• auth/middleware.py handles JWT validation; modified in 4 of last 6 sessions
• SessionExpiredError was the root cause of the auth bug fixed in session 14
• test_auth.py has flaky timing tests; known issue, not a regression

=== END CORTEX MEMORY (tokens: 312 / budget: 600) ===
```

## Token Budget

inject.py enforces a hard 600-token budget using tiktoken with the cl100k_base
encoding. If retrieved nodes exceed the budget:

1. Tier-3 nodes are always kept (they're small by design)
2. Tier-2 nodes are sorted by score and truncated from the bottom
3. Tier-1 nodes are dropped first

The token count is logged to the sessions table for savings calculation.

## Dashboard

A separate process (not a hook) that reads cortex.db via SQLite WAL and serves
a local web UI on port 7000.

Components:
- **Token savings chart**: line chart of (tokens_raw - tokens_injected) per session
- **Memory graph**: D3 force-directed graph. Node size = weight. Opacity = tier.
  Fading nodes are visually decaying. Click a node to see full metadata.
- **Node inspector**: tier, weight, decay rate, last accessed, source, text, rationale

The dashboard is read-only. Node pruning is done via `cortex graph prune <id>` CLI.

## Plugin Integration

Cortex ships as a Claude Code plugin. The plugin.json wires all hooks automatically.
After `cortex install`, no manual configuration is needed.

```json
{
  "name": "cortex",
  "version": "0.1.0",
  "hooks": {
    "SessionStart": [{"command": "python ~/.cortex/hooks/inject.py"}],
    "Stop":         [{"command": "python ~/.cortex/hooks/extract.py"}],
    "PostCompact":  [{"command": "python ~/.cortex/hooks/compact.py"}]
  }
}
```
