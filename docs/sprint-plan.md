# Sprint Plan

Build order with acceptance criteria. Do not start a sprint until the previous
one's acceptance criteria are fully met and tests are green.

## Sprint 0 — Foundations (Day 1)

**Goal:** Repo scaffolding, schema, CI, git workflow. No feature code.

**Tasks:**
- [ ] Init git repo, create all directories from CLAUDE.md layout
- [ ] Write schema.sql (use the canonical version — do not alter)
- [ ] Write pyproject.toml with all dependencies pinned
- [ ] Set up pre-commit hooks (ruff, black, mypy)
- [ ] Set up .github/workflows/test.yml
- [ ] Create CONTRIBUTING.md with commit conventions
- [ ] Install spaCy model: `python -m spacy download en_core_web_sm`
- [ ] Verify ollama is available or sentence-transformers fallback works
- [ ] Create tests/conftest.py with db and graph fixtures
- [ ] Create fixture transcript files (agent: qa-expert)

**Acceptance criteria:**
- `git commit` triggers pre-commit hooks successfully
- `pytest tests/` runs and produces 0 errors (no tests yet, 0 collected is fine)
- schema.sql creates all three tables in SQLite without errors
- `mypy --strict core/` passes on empty __init__.py files

**Agents:** git-workflow-manager, qa-expert

---

## Sprint 1 — Graph Layer (Days 2–4)

**Goal:** SQLite graph read/write/merge fully working and tested.

**Files:** core/graph.py, tests/test_graph.py

**Tasks:**
- [ ] Implement Graph class with connection injection
- [ ] write_node(candidate: CandidateNode) → str (node id)
- [ ] merge_node(existing_id: str, candidate: CandidateNode) → None
- [ ] find_similar(embedding, project, threshold, limit) → list[Node]
- [ ] get_all_nodes(project, tier=None) → list[Node]
- [ ] delete_node(node_id) → None
- [ ] write_edge(source_id, target_id) → None (increments strength if exists)
- [ ] get_edges(node_id) → list[Edge]
- [ ] update_weight(node_id, delta) → None
- [ ] Write test_graph.py with all cases from testing-strategy.md

**Acceptance criteria:**
- `pytest tests/test_graph.py -v` — all pass
- Coverage ≥ 95% on graph.py
- mypy --strict passes
- ruff passes
- No file I/O in tests (all :memory:)

**Agents:** python-pro → qa-expert review → refactoring-specialist cleanup

---

## Sprint 2 — Parser (Days 4–5)

**Goal:** JSONL transcript → typed ParsedEvent list.

**Files:** core/parser.py, tests/test_parser.py, tests/fixtures/transcripts/

**Tasks:**
- [ ] Define ParsedEvent dataclass and EventType enum
- [ ] parse_transcript(path: Path) → Iterator[ParsedEvent]
- [ ] Detect all EventType variants from JSONL
- [ ] Detect file write hotspots (≥3 writes same file)
- [ ] Detect bash failures with exit code
- [ ] Detect co-occurring file writes (60s window)
- [ ] Create fixture: simple.jsonl (20 events)
- [ ] Create fixture: with_failures.jsonl (bash failures)
- [ ] Write test_parser.py

**Acceptance criteria:**
- All fixture transcripts parse without error
- Hotspots correctly identified in simple.jsonl
- Malformed lines produce log warning, not exception
- Coverage ≥ 95%

**Agents:** python-pro → qa-expert

---

## Sprint 3 — Extractor (Days 5–8)

**Goal:** Four-channel extraction producing CandidateNode list.

**Files:** core/extractor.py, tests/test_extractor.py, tests/fixtures/

**Tasks:**
- [ ] CandidateNode dataclass
- [ ] jsonl_channel(events: list[ParsedEvent]) → list[CandidateNode]
- [ ] ast_channel(touched_files: list[Path]) → list[CandidateNode]
- [ ] git_channel(project_root: Path) → list[CandidateNode]
- [ ] nlp_channel(prose_turns: list[str]) → list[CandidateNode]
- [ ] Durability scoring per docs/extraction-pipeline.md
- [ ] run_extraction(transcript_path, project) → list[CandidateNode]
- [ ] Create fixture: with_decisions.jsonl
- [ ] Create fixture: with_retractions.jsonl
- [ ] Create expected_nodes JSON files
- [ ] Write test_extractor.py + test_quality.py eval cases

**Acceptance criteria:**
- Decision sentences extracted correctly from with_decisions.jsonl
- Retracted statements produce zero nodes from with_retractions.jsonl
- Generic advice produces zero nodes
- Extraction volume: 2–8 nodes per fixture transcript
- Coverage ≥ 90%
- Extraction of simple.jsonl completes in < 400ms (excluding model load)

**Agents:** ai-engineer (NLP design) → python-pro → qa-expert

---

## Sprint 4 — Embedder + Dedup (Days 8–10)

**Goal:** Local embedding, deduplication on write.

**Files:** core/embedder.py, tests/test_embedder.py

**Tasks:**
- [ ] Embedder class with lazy model loading
- [ ] embed(text: str) → np.ndarray
- [ ] serialize(embedding: np.ndarray, precision_bits: int) → bytes
- [ ] deserialize(blob: bytes, precision_bits: int) → np.ndarray
- [ ] cosine_similarity(a, b) → float
- [ ] Integrate dedup into graph.write_node
- [ ] Write test_embedder.py (use dummy vectors for unit tests)

**Acceptance criteria:**
- Embedding model loads once, not per call
- serialize/deserialize round-trips correctly at all precision levels
- Dedup: two identical candidates produce one node, not two
- Coverage ≥ 85%

**Agents:** ai-engineer (precision quantization design) → python-pro → qa-expert

---

## Sprint 5 — Decay (Days 10–12)

**Goal:** Weight decay, tier promotion, eviction.

**Files:** core/decay.py, tests/test_decay.py

**Tasks:**
- [ ] run_decay(graph, project) → DecayResult
- [ ] Tier-1 decay: × 0.85 per unaccessed session
- [ ] Tier-2 decay: × 0.95 per unaccessed session
- [ ] Tier-3: no decay
- [ ] Eviction: delete nodes below threshold
- [ ] Promotion: tier-1 → tier-2 on threshold conditions
- [ ] Promotion: tier-2 → tier-3 on threshold conditions
- [ ] Precision downcast on promotion
- [ ] DecayResult dataclass: nodes_decayed, nodes_evicted, nodes_promoted

**Acceptance criteria:**
- Tier-3 nodes never decayed in any test
- Promotion requires all conditions (weight AND session_count AND age)
- Eviction removes correct nodes
- Coverage ≥ 95%

**Agents:** python-pro → qa-expert → refactoring-specialist

---

## Sprint 6 — Retrieval (Days 12–15)

**Goal:** Three-channel retrieval with token budget enforcement.

**Files:** core/retrieval.py, tests/test_retrieval.py

**Tasks:**
- [ ] build_bm25_index(nodes: list[Node]) → BM25Index
- [ ] vector_channel(query_embedding, nodes) → list[ScoredNode]
- [ ] bm25_channel(query: str, index) → list[ScoredNode]
- [ ] graph_channel(seed_nodes, graph) → list[ScoredNode]
- [ ] fuse_scores(v, b, g) → list[ScoredNode]
- [ ] retrieve(query, project, graph, budget_tokens=600) → list[Node]
- [ ] format_injection_block(nodes, project) → str

**Acceptance criteria:**
- Tier-3 nodes always in result regardless of score
- Token budget ≤ 600 enforced
- Graph hop traversal reaches 1–2 edge hops only
- Coverage ≥ 90%

**Agents:** ai-engineer (scoring weight design) → python-pro → qa-expert

---

## Sprint 7 — Hooks (Days 15–18)

**Goal:** End-to-end loop working. This is the first integration milestone.

**Files:** hooks/inject.py, hooks/extract.py, hooks/compact.py, tests/test_hooks.py

**Tasks:**
- [ ] extract.py: reads $CLAUDE_TRANSCRIPT env var, runs pipeline, writes graph
- [ ] inject.py: reads project path, retrieves context, prints injection block
- [ ] compact.py: re-runs inject.py logic on PostCompact
- [ ] Integration test: extract → verify nodes → inject → verify output
- [ ] Measure: end-to-end loop < 500ms on simple.jsonl

**Acceptance criteria:**
- Full loop test passes: fixture transcript → extract → inject → correct output
- Idempotency: running extract.py twice produces same node count
- Injection block always within token budget
- Hook scripts exit 0 on success, non-zero on unrecoverable error

**Agents:** python-pro → qa-expert (integration test) → debugger review

---

## Sprint 8 — CLI (Days 18–20)

**Goal:** Developer-facing CLI for visibility and control.

**Files:** cli/cortex.py, cli/config.py

**Commands:**
- `cortex status` — node count by tier, token savings this week, last session
- `cortex graph` — print ASCII adjacency summary (not the dashboard)
- `cortex inspect <node-id>` — full node metadata
- `cortex prune <node-id>` — manually evict a node
- `cortex reset --project <path>` — wipe all nodes for a project
- `cortex install` — write plugin.json to Claude Code plugins directory
- `cortex dashboard` — start dashboard server on port 7000

**Agents:** python-pro

---

## Sprint 9 — Dashboard (Days 20–26)

**Goal:** Live web UI showing token savings and memory graph.

**Files:** dashboard/server.py, dashboard/app/

**Tasks:**
- [ ] FastAPI server with WebSocket endpoint
- [ ] Reads cortex.db via SQLite WAL (read-only, no locks)
- [ ] React app: token savings line chart (Recharts)
- [ ] React app: D3 force-directed graph (node opacity = tier)
- [ ] React app: node inspector panel (click node → metadata)
- [ ] `cortex dashboard` starts server + opens browser

**Agents:** python-pro (server) → frontend for React

---

## Sprint 10 — Plugin + Release (Days 26–30)

**Goal:** One-command install, open-source release ready.

**Files:** plugin/plugin.json, plugin/skills/cortex-status/SKILL.md, README.md

**Tasks:**
- [ ] plugin.json with all three hooks wired
- [ ] cortex-status SKILL.md for /cortex-status slash command
- [ ] README.md: installation, quick start, architecture overview
- [ ] CHANGELOG.md
- [ ] `cortex install` command working
- [ ] GitHub release workflow
- [ ] PyPI publish workflow

**Agents:** git-workflow-manager → python-pro (packaging)

---

## Definition of Done (all sprints)

A sprint is done when ALL of the following are true:

- [ ] `pytest tests/ -v --cov=core --cov-fail-under=90` passes
- [ ] `ruff check .` — zero warnings
- [ ] `mypy --strict core/ hooks/ cli/` — zero errors
- [ ] `bandit -r core/ hooks/ cli/ -ll` — zero medium+ issues
- [ ] `black --check .` — no formatting differences
- [ ] DECISIONS.md updated if any architectural choice was made
- [ ] PR merged to main with conventional commit message
