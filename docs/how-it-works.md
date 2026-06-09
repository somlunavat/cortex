# Cortex — How It Works

Cortex is a persistent, self-managing memory layer for Claude Code. It runs as a set of hooks that silently observe every session, extract durable facts into a knowledge graph, and inject relevant context at the start of each new session.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [The Three-Tier Memory Model](#2-the-three-tier-memory-model)
3. [Session Lifecycle](#3-session-lifecycle)
4. [Extraction Pipeline (four channels)](#4-extraction-pipeline)
5. [The Knowledge Graph](#5-the-knowledge-graph)
6. [Example Graph Maps](#6-example-graph-maps)
7. [Retrieval Pipeline](#7-retrieval-pipeline)
8. [What Gets Injected](#8-what-gets-injected)
9. [Decay, Promotion, and Eviction](#9-decay-promotion-and-eviction)
10. [Hook Wiring](#10-hook-wiring)

---

## 1. System Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    Claude Code Session                        │
│                                                              │
│  SessionStart hook          PostCompact hook                 │
│  ┌──────────────┐           ┌──────────────┐                 │
│  │  inject.py   │           │  compact.py  │                 │
│  │  (retrieval) │           │  (re-inject) │                 │
│  └──────┬───────┘           └──────┬───────┘                 │
│         │ stdout                   │ stdout                   │
│         ▼                          ▼                          │
│  ┌─────────────────────────────────────────────────────┐     │
│  │               Claude's context window               │     │
│  │  === CORTEX MEMORY (PROJECT: /myproject) ===        │     │
│  │  [CONVENTIONS] • always use async/await             │     │
│  │  [CONTEXT]     • db/auth.py modified 4× this week   │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                              │
│  Stop hook                                                   │
│  ┌──────────────┐                                            │
│  │  extract.py  │──── parses transcript ──► knowledge graph  │
│  └──────────────┘                                            │
└──────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  cortex.db        │
                    │  (SQLite + WAL)   │
                    │  nodes / edges /  │
                    │  sessions tables  │
                    └───────────────────┘
```

---

## 2. The Three-Tier Memory Model

Every memory node lives in exactly one tier. Tier determines how long a node
survives, how its embedding is stored, and whether it is ever auto-evicted.

```
Tier 1 — Ephemeral          Tier 2 — Semantic           Tier 3 — Procedural
────────────────────        ────────────────────        ────────────────────
New observation from        Fact confirmed across       Convention/rule that
a single session.           multiple sessions.          applies everywhere.

Decay rate : ×0.85/run      Decay rate : ×0.95/run      No decay. Ever.
Evict at   : weight < 0.3   Evict at   : weight < 0.5   Never evicted.
Embedding  : float32 (32b)  Embedding  : int8   (8b)    Embedding : int2 (2b)

Promote to Tier 2 when:     Promote to Tier 3 when:
  weight ≥ 8.0 AND            weight ≥ 20.0 AND
  session_count ≥ 3           session_count ≥ 8 AND
                              age ≥ 14 days
```

Precision is downcast on promotion to reduce storage while preserving
cosine-similarity accuracy (cosine similarity is scale-invariant).

---

## 3. Session Lifecycle

```
                    ┌───────────────────────────────────┐
                    │  SessionStart (new chat begins)   │
                    └─────────────┬─────────────────────┘
                                  │
                    ┌─────────────▼─────────────────────┐
                    │  inject.py                        │
                    │  • open cortex.db                 │
                    │  • retrieve(CLAUDE_INITIAL_MSG)   │
                    │  • print injection block → stdout │
                    └─────────────┬─────────────────────┘
                                  │
                    ┌─────────────▼─────────────────────┐
                    │  … conversation in progress …    │
                    └─────────────┬─────────────────────┘
                                  │
                    ┌─────────────▼─────────────────────┐  (if context compacted)
                    │  PostCompact — compact.py          │◄──────────────────────┐
                    │  (same as inject.py — re-injects)  │                       │
                    └─────────────┬─────────────────────┘                       │
                                  │
                    ┌─────────────▼─────────────────────┐
                    │  Stop (session ends)               │
                    └─────────────┬─────────────────────┘
                                  │
                    ┌─────────────▼─────────────────────┐
                    │  extract.py                       │
                    │  • parse JSONL transcript         │
                    │  • run 4-channel extraction       │
                    │  • embed + merge into graph       │
                    │  • run decay / eviction cycle     │
                    │  • log session row                │
                    └───────────────────────────────────┘
```

---

## 4. Extraction Pipeline

At session end, `extract.py` runs four independent extraction channels over
the session transcript. Each channel produces `CandidateNode` objects.
Candidates with `durability < 0.4` are discarded before writing.

```
JSONL transcript
      │
      ├──[Channel 1: JSONL]──────────────────────────────────────────────┐
      │   • File write hotspots (≥3 writes to same file) → OBSERVATION   │
      │   • Bash failures with non-zero exit code       → ERROR          │
      │                                                                   │
      ├──[Channel 2: AST diff]───────────────────────────────────────────┤
      │   • tree-sitter parses touched files vs HEAD                     │
      │   • Added/removed functions or classes          → FACT           │
      │                                                                   │
      ├──[Channel 3: Git]────────────────────────────────────────────────┤
      │   • Commits made during this session            → FACT           │
      │   • Files modified in >60% of last 10 commits  → FACT           │
      │                                                                   │
      └──[Channel 4: NLP (spaCy)]────────────────────────────────────────┤
          • Sentences with decision verbs               → FACT           │
            (choose, decide, select, opt, migrate…)                      │
          • Sentences with convention markers           → CONVENTION     │
            (always, never, convention, standard…)                       │
          • Retraction patterns stripped automatically                   │
                                                                         │
                        ┌────────────────────────────────────────────────┘
                        │  all candidates merged
                        ▼
               ┌─────────────────┐
               │ durability ≥ 0.4 │──NO──► discard
               └────────┬────────┘
                        │YES
                        ▼
               embed text (all-MiniLM-L6-v2, 384-dim)
                        │
               find_similar(threshold=0.9)
                        │
             ┌──────────┴──────────┐
             │ match found?        │
             │YES                  │NO
             ▼                     ▼
        merge_node             write_node
        (weight +0.5,          (tier=1, weight=1.0)
         session_count+1)
```

---

## 5. The Knowledge Graph

The graph is stored in a single SQLite database (`.cortex/cortex.db`).

```
nodes table                          edges table
─────────────────────────────────    ─────────────────────────────
id             UUID4                 source_id   → nodes.id
type           observation|fact|     target_id   → nodes.id
               convention|error      strength    co-retrieval count
tier           1 | 2 | 3             last_seen   unix timestamp
text           one sentence
rationale      why (nullable)        Edges grow stronger each time
embedding      BLOB (float32/int8/   both endpoints are retrieved
               int2 numpy bytes)     in the same query.
precision_bits 32 | 8 | 2
weight         access-weighted score
project        /absolute/path
scope          session|module|project
source         jsonl|ast|git|nlp
last_accessed  unix timestamp
created_at     unix timestamp
session_count  distinct sessions
```

Nodes in the same session that are co-retrieved get an edge between them.
The edge `strength` increments by 1.0 each time they co-occur, which the
graph-hop channel uses to promote related nodes in future retrievals.

---

## 6. Example Graph Maps

### 6a. Small project — early sessions

After 2-3 sessions on a new backend API project:

```
[T1-OBSERVATION]  api/auth.py modified 3× this session
     │ strength=1
     ▼
[T1-FACT]         Added function `verify_jwt` to api/auth.py
     │ strength=1
     ▼
[T1-FACT]         Removed function `legacy_token_check` from api/auth.py

[T1-ERROR]        Command `pytest tests/test_auth.py` failed (exit 1): AssertionError

[T1-FACT]         feat: add JWT verification middleware
```

All nodes are Tier 1 (new, unconfirmed). Weights are around 1.0–2.0.
Edges connect nodes that were retrieved together in the same query.

---

### 6b. Same project — after 6 sessions

High-value nodes have accumulated weight and session count. Some have
promoted to Tier 2:

```
                         ┌─────────────────────────────────────┐
[T3-CONVENTION]  ◄───────│  always use async/await for DB I/O  │
                         │  weight=∞  (never decays)           │
                         └─────────────────────────────────────┘
                                           │
                              strength=4.0 │
                                           ▼
                         ┌─────────────────────────────────────┐
[T2-FACT]                │  asyncpg used for all DB connections │
                         │  weight=12.4  session_count=6       │
                         │  precision_bits=8 (promoted from T1) │
                         └──────────┬──────────────────────────┘
                                    │
                         strength=2.0│           strength=1.5
                   ┌────────────────┘                   │
                   ▼                                     ▼
[T1-OBSERVATION]  db/pool.py modified 5×       [T1-FACT]  Added function
                  this session                            `get_connection`
                  weight=2.1                             weight=1.6
```

The Tier 3 convention node was promoted because it appeared in 8+ sessions
with weight ≥ 20. It is now permanent and always injected regardless of query.

---

### 6c. After a long-running project (weeks of daily use)

The graph develops deep structure. High-churn files become Tier 2 facts.
Architectural decisions become Tier 3 conventions.

```
[T3-CONVENTION]  always use async/await for DB I/O           weight=∞
[T3-CONVENTION]  use Pydantic v2 models for all API schemas  weight=∞
[T3-CONVENTION]  never store secrets in environment files    weight=∞
       │                    │                    │
       │ strength=6         │ strength=9         │ strength=4
       ▼                    ▼                    ▼
[T2-FACT]         [T2-FACT]              [T2-FACT]
api/models.py is  asyncpg connection     secrets loaded
high-churn        pool configured in     from Vault at startup
(9 of last 10     db/config.py           weight=8.1
commits)          weight=15.3            session_count=6
weight=11.2
session_count=7
       │
       │ strength=3
       ▼
[T1-OBSERVATION]  api/models.py modified 4× this session
                  weight=1.0  (freshly written, may decay away)
```

Tier 1 leaves are session-specific observations. Most will evict after
1-2 sessions of non-access (weight decays by 15% per run; eviction at 0.3).

---

## 7. Retrieval Pipeline

At session start, `inject.py` runs a three-channel retrieval against the
graph scoped to the current project. The query is `CLAUDE_INITIAL_MESSAGE`
(the user's first message), or empty string (still returns Tier 3 nodes).

```
query = "add rate limiting to the auth middleware"
      │
      ├─[Vector channel]───────────────────────────────── weight: 0.5
      │   embed(query) → 384-dim float32
      │   cosine_similarity vs every node with an embedding
      │   → scored list
      │
      ├─[BM25 channel]─────────────────────────────────── weight: 0.3
      │   tokenize(query), BM25Okapi over node texts
      │   scores normalized to [0, 1]
      │   → scored list
      │
      └─[Graph-hop channel]────────────────────────────── weight: 0.2
          seed_nodes = top-3 by vector score
          1-hop neighbours → score 0.5
          2-hop neighbours → score 0.25
          → scored list

fused_score = 0.5 * vector + 0.3 * BM25 + 0.2 * graph_hop

top-8 candidates selected by fused score
      │
      ▼
Token budget enforcement (default 600 tokens, via tiktoken cl100k_base)
      │
      ▼
Tier-3 nodes ALWAYS included first (conventions)
Remaining budget filled by scored candidates in descending order
```

---

## 8. What Gets Injected

The injected block is printed to stdout by the SessionStart hook. Claude Code
prepends it to the system prompt.

```
=== CORTEX MEMORY (PROJECT: /home/user/myproject) ===

[CONVENTIONS — always apply]
• always use async/await for DB I/O
• use Pydantic v2 models for all API schemas

[RELEVANT CONTEXT — this session]
• asyncpg connection pool configured in db/config.py
• api/models.py is a high-churn file (modified in 9 of last 10 commits)
• Added function `verify_jwt` to api/auth.py
• api/auth.py modified 4× last session

=== END CORTEX MEMORY (tokens: 124 / budget: 600) ===
```

Tier-3 nodes appear in `[CONVENTIONS]` — they carry the strongest signal and
should be applied unconditionally. Tier 1/2 nodes appear in `[RELEVANT CONTEXT]`
to orient Claude toward the most active parts of the codebase.

---

## 9. Decay, Promotion, and Eviction

`run_decay` is called at the end of every session (inside `extract.py`).

```
For each non-Tier-3 node in the project:

  new_weight = node.weight × DECAY_RATE
                                            Tier 1: DECAY_RATE = 0.85
                                            Tier 2: DECAY_RATE = 0.95

  ┌───────────────────┐
  │ weight < threshold│──YES──► delete_node (CASCADE removes edges)
  └─────────┬─────────┘          Tier 1: threshold = 0.3
            │NO                  Tier 2: threshold = 0.5
            ▼
  ┌──────────────────────────────────────────┐
  │ promotion conditions met?                │
  │                                          │
  │ Tier1→2: weight≥8 AND sessions≥3        │──YES──► update_node_tier
  │ Tier2→3: weight≥20 AND sessions≥8       │         embedding re-serialized
  │          AND age≥14 days                │         at lower precision
  └──────────────────────────────────────────┘

Result: DecayResult(project, nodes_decayed, nodes_evicted, nodes_promoted)
        logged to sessions table
```

**Why multiplicative decay?** A node that was relevant 5 sessions ago still
holds residual value — it decays toward zero rather than being hard-deleted
on a fixed schedule. Each new access (merge or retrieval) resets `weight`
upward, keeping genuinely useful nodes alive indefinitely.

---

## 10. Hook Wiring

After running `cortex install`, the following hooks are registered in
`~/.claude/plugins/cortex.json`:

```json
{
  "hooks": {
    "SessionStart" : ["python3 hooks/inject.py"],
    "Stop"         : ["python3 hooks/extract.py"],
    "PostCompact"  : ["python3 hooks/compact.py"]
  }
}
```

Environment variables read by the hooks:

| Variable               | Used by       | Description                              |
|------------------------|---------------|------------------------------------------|
| `CLAUDE_TRANSCRIPT`    | extract.py    | Path to the session JSONL file           |
| `CLAUDE_PROJECT_PATH`  | all           | Absolute path to the project root        |
| `CLAUDE_INITIAL_MESSAGE` | inject.py   | First user message (used as query)       |
| `CORTEX_DB_PATH`       | all           | Override default `.cortex/cortex.db`     |

The database defaults to `<project>/.cortex/cortex.db`. Each project gets
its own database; there is no cross-project memory leakage.
