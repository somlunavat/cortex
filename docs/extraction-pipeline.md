# Extraction Pipeline

This document specifies the extraction pipeline in enough detail to implement
without ambiguity. Read this before touching extractor.py or parser.py.

## Principle

The extraction pipeline converts a raw Claude Code session into 2–8 structured
memory nodes. It runs after every session ends (Stop hook). It must:

- Complete in under 400ms (leaving 100ms budget for graph write)
- Never call any external API
- Be idempotent: running twice on the same session produces identical output
- Produce nodes that are *durable* (still true in 3 months) and *specific*
  (project-specific, not generic advice)

## What Gets Extracted vs Dropped

### EXTRACT

| Signal | Example | Node type |
|--------|---------|-----------|
| File modified 3+ times in session | auth.py touched 5 times | observation |
| Bash command failed with non-zero exit | pytest failed with ImportError | error |
| Function added/removed (AST) | added `validate_token()` to auth.py | fact |
| Commit message (git) | "fix: resolve JWT expiry race condition" | fact |
| Decision sentence in Claude prose | "I chose asyncpg over SQLAlchemy because..." | fact |
| Convention statement in Claude prose | "We always validate at the boundary" | convention candidate |
| Import added (AST) | added `import redis` to cache.py | observation |
| File churn hotspot | auth.py modified in 5 of last 7 sessions | fact |

### DROP (do not extract)

| Signal | Reason |
|--------|--------|
| Claude's internal reasoning steps | Not durable — process, not decision |
| Retracted statements ("actually let's not...") | Negated before session end |
| Generic advice ("you should use types") | Not project-specific |
| Repetition of existing graph nodes (similarity > 0.9) | Dedup handles this |
| Session-scoped one-offs (temp debug code) | scope='session', evicts fast |
| User messages | User context, not project memory |
| Tool output / file contents | Too large, raw data not memory |

## Channel Specifications

### Channel 1: JSONL Parser (parser.py)

Input: path to a Claude Code JSONL transcript file
Output: `list[ParsedEvent]`

```python
@dataclass(frozen=True)
class ParsedEvent:
    type: EventType           # see EventType enum in architecture.md
    timestamp: int            # unix timestamp
    data: dict[str, Any]      # event-specific payload
    session_id: str
```

**Implementation notes:**
- Each JSONL line is one event. Parse with `json.loads()` per line.
- Skip malformed lines (log warning, don't raise).
- The `tool_use` event contains `tool_name` and `tool_input` in data.
- File paths in events should be normalized to absolute paths.
- For `bash_exec` events, capture both the command and exit code.

**High-value event patterns to detect:**
```python
# Repeated file write → hotspot signal
file_write_counts: Counter[str] = Counter()
for event in events:
    if event.type == EventType.FILE_WRITE:
        file_write_counts[event.data["path"]] += 1

hotspots = {path for path, count in file_write_counts.items() if count >= 3}

# Bash failure → error signal
failures = [e for e in events if e.type == EventType.BASH_FAILURE]

# Co-occurring writes → coupling signal (files written in same 60s window)
```

### Channel 2: AST Diff (extractor.py, ast_channel)

Input: list of file paths touched in session
Output: `list[CandidateNode]`

Uses tree-sitter. Supported languages: Python, TypeScript, JavaScript, Go, Rust.
Falls back to line-diff for unsupported languages.

**Procedure:**
1. For each touched file, get the git blob before session start (`git show HEAD:file`)
2. Parse both versions with tree-sitter
3. Diff the AST: functions added, removed, renamed; classes added/removed
4. Each structural change becomes one candidate node

**Node text format:**
```
# Addition
"Added function `validate_jwt_expiry` to auth/middleware.py"

# Removal
"Removed function `legacy_auth_check` from auth/middleware.py"

# Complexity increase
"Function `process_request` in api/handler.py: cyclomatic complexity increased 4→11"
```

**Skip if:** file was created this session (no prior blob to diff against).
**Skip if:** file is in .gitignore.
**Skip if:** file is a test file (path contains `test_` or `/tests/`).

### Channel 3: Git Signals (extractor.py, git_channel)

Input: project root path
Output: `list[CandidateNode]`

Uses gitpython. Skips gracefully if project is not a git repo.

**Signals to extract:**

```python
repo = git.Repo(project_root)

# 1. Commit message if a commit was made this session
recent_commits = list(repo.iter_commits(max_count=3))
session_commits = [c for c in recent_commits if c.committed_date >= session_start]

# 2. File churn: files changed in > 60% of last 10 commits
file_churn = compute_churn(repo, lookback=10, threshold=0.6)

# 3. Branch name as task context (weak signal, tier 1 only)
branch = repo.active_branch.name
```

Commit messages produce `fact` nodes with source `git`.
Churn hotspots produce `fact` nodes: "auth/middleware.py is a high-churn file
(modified in 7 of last 10 commits)".

### Channel 4: spaCy NLP (extractor.py, nlp_channel)

Input: list of Claude's prose output turns (text only, not tool outputs)
Output: `list[CandidateNode]`

Model: `en_core_web_sm` (lightweight, no GPU required).
Load once at module import time — do not reload per session.

**Decision sentence detection:**
```python
DECISION_VERBS = {"chose", "decided", "selected", "opted", "switched", "migrated"}
AVOIDANCE_VERBS = {"avoid", "skip", "not use", "reject", "drop"}

def is_decision_sentence(sent: spacy.tokens.Span) -> bool:
    lemmas = {token.lemma_.lower() for token in sent}
    return bool(lemmas & DECISION_VERBS) or bool(lemmas & AVOIDANCE_VERBS)
```

**Convention sentence detection:**
```python
CONVENTION_MARKERS = {"always", "never", "convention", "standard", "pattern",
                      "throughout", "consistently", "everywhere"}

def is_convention_sentence(sent: spacy.tokens.Span) -> bool:
    return any(token.lower_ in CONVENTION_MARKERS for token in sent)
```

**Splitting conclusion from rationale:**

When a decision sentence contains "because", "since", "as", or "so that",
split at the conjunction:
- Text (conclusion): content before the conjunction → stored in `node.text`
- Rationale: content after the conjunction → stored in `node.rationale`

Both are one sentence max. Truncate at 200 characters.

**Durability scoring (rule-based):**

```python
def score_durability(sent: spacy.tokens.Span, node_type: str) -> float:
    score = 0.5  # baseline

    # Boosts
    if node_type == "convention":       score += 0.3
    if "always" in sent.text.lower():   score += 0.2
    if "never" in sent.text.lower():    score += 0.2
    if any(ent.label_ in ("ORG", "PRODUCT") for ent in sent.ents):
        score += 0.1  # named technology = more specific

    # Penalties
    if "maybe" in sent.text.lower():    score -= 0.3
    if "temporary" in sent.text.lower(): score -= 0.4
    if "for now" in sent.text.lower():  score -= 0.3
    if "this session" in sent.text.lower(): score -= 0.4

    return max(0.0, min(1.0, score))
```

Candidates with durability < 0.4 are dropped before embedding.

## Candidate Node Schema

All four channels produce candidates in this format:

```python
@dataclass
class CandidateNode:
    text: str                    # one sentence, conclusion only
    rationale: str | None        # one sentence why, nullable
    type: NodeType               # observation | fact | convention | error
    source: SourceType           # jsonl | ast | git | nlp
    scope: ScopeType             # session | module | project
    durability: float            # 0.0–1.0, computed per channel
    project: str                 # absolute project path
```

## Deduplication

After candidates are produced, embedder.py embeds each one.
graph.py then runs deduplication:

```python
for candidate in candidates:
    similar = graph.find_similar(
        embedding=candidate.embedding,
        project=candidate.project,
        threshold=0.9,
        limit=5,
    )

    if similar:
        # Update existing node — do not create new
        graph.merge_node(existing_id=similar[0].id, candidate=candidate)
    else:
        # Create new node
        graph.write_node(candidate)
```

Merging rules:
- weight += 0.5 (partial credit, not full access increment)
- last_accessed = now()
- session_count += 1
- text: keep existing (more established) unless new text is significantly shorter
- rationale: keep whichever is non-null; if both non-null, keep existing

## Output Volume

A well-tuned extraction run produces:
- 0–2 nodes from JSONL channel (only genuine signals)
- 0–3 nodes from AST channel (structural changes only)
- 0–2 nodes from git channel (commit messages + churn)
- 0–3 nodes from NLP channel (decisions + conventions only)

Total: 2–8 nodes per session. If your implementation is producing more than 10
nodes per session consistently, the durability thresholds are too low.
