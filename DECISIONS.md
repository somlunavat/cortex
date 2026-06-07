# Cortex — Architectural Decision Log

Append every decision here after the session it was made in.
Format: date, title, decision, rationale, alternatives, impact.
Never edit past entries.

---

## 2026-06-07 — No LLM in extraction pipeline

**Decision:** The extraction pipeline (parser, extractor, embedder, decay, retrieval)
must never call any LLM API.

**Rationale:** Cortex exists to save tokens. Using Claude or any API to extract memory
defeats the purpose — the extraction cost would approach or exceed the savings. A
coding-specific session has structured, parseable signals (JSONL events, AST diffs, git
output) that yield 70% of useful memory without language understanding. The remaining
30% from prose is handled by a local spaCy NLP pipeline at zero API cost.

**Alternatives considered:**
- Haiku for lightweight extraction: rejected because any API call creates a cost floor
  and a network dependency that breaks offline use.
- Local Qwen2.5-3B: available as fallback for prose extraction if spaCy NLP quality
  proves insufficient, but not the default path.

**Impact:** core/extractor.py, core/embedder.py, hooks/extract.py

---

## 2026-06-07 — SQLite as sole database

**Decision:** SQLite with sqlite-vec extension is the only persistence layer.

**Rationale:** Cortex runs as a local developer tool. Zero-dependency setup is a hard
requirement for open-source adoption. SQLite with WAL mode handles the write patterns
(one extraction per session end) and read patterns (one retrieval per session start)
trivially. Vector similarity via sqlite-vec keeps everything in one file.

**Alternatives considered:**
- Chroma: adds a server process dependency, overkill for local use.
- pgvector + Postgres: requires a running Postgres instance, ruled out.
- FAISS: file-based but no graph support, would require a separate adjacency store.

**Impact:** schema.sql, core/graph.py, all tests

---

## 2026-06-07 — Three-tier memory hierarchy

**Decision:** Nodes exist in one of three tiers: ephemeral (1), semantic (2),
procedural (3). Decay rate, precision, and eviction threshold differ per tier.
Nodes promote upward but never demote.

**Rationale:** Mirrors the RAPTOR abstraction tree and SuperLocalMemory precision
quantization research. Prevents graph bloat — raw observations decay fast, conventions
survive indefinitely. Precision downcast (32→8→2 bit) reduces embedding storage cost
as nodes age into permanence.

**Alternatives considered:**
- Flat single-tier with global decay: simpler but causes bloat over months.
- Four tiers: unnecessary complexity, three maps cleanly to observation/fact/convention.

**Impact:** core/decay.py, schema.sql, core/graph.py

---

## 2026-06-07 — Synchronous hooks, 500ms budget

**Decision:** All hook scripts (inject.py, extract.py, compact.py) are synchronous
Python with a hard 500ms completion budget.

**Rationale:** Claude Code hooks block the session lifecycle. Async complexity is not
worth it for scripts that run once per session turn. The embedding model must be
preloaded as a persistent sidecar process (ollama serve) to avoid cold-start latency.

**Alternatives considered:**
- Async hooks: ruled out, no benefit for single-execution scripts.
- Background extraction after hook returns: harder to guarantee write completion.

**Impact:** hooks/inject.py, hooks/extract.py, hooks/compact.py

---

## 2026-06-07 — Agent routing

**Decision:** Specialist agents are assigned fixed domains and should not cross lanes
without a note in DECISIONS.md.

| Agent                  | Domain                                      |
|------------------------|---------------------------------------------|
| python-pro             | core/, hooks/ implementation                |
| ai-engineer            | embedder.py, retrieval.py, decay math       |
| refactoring-specialist | post-sprint cleanup of core/ modules        |
| qa-expert              | tests/, coverage, integration test design   |
| debugger               | silent hook failures, retrieval regressions |
| git-workflow-manager   | branch strategy, commits, release tagging   |
| context-manager        | graph.py schema evolution, query tuning     |

**Impact:** All files — sets expectations for every session.

---

## 2026-06-07 — Embedding backend: sentence-transformers (ollama not available)

**Decision:** Use `sentence-transformers` with `all-MiniLM-L6-v2` as the local
embedding backend. Ollama is not installed in this environment.

**Rationale:** The spec requires either `nomic-embed-text` via ollama or
`all-MiniLM-L6-v2` via sentence-transformers. Since ollama is unavailable,
sentence-transformers is the correct fallback. Both produce 384-dim vectors
compatible with the graph schema.

**Alternatives considered:**
- Ollama with nomic-embed-text: not available; can be added later by updating
  `core/embedder.py` to detect `ollama serve` on startup.

**Impact:** core/embedder.py (Sprint 4)
