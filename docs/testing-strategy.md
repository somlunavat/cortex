# Testing Strategy

This document is the qa-expert's reference. Every decision about test structure,
coverage targets, and validation methodology lives here.

## Philosophy

Tests in Cortex serve two purposes:
1. Catch regressions in memory quality (the hard problem)
2. Catch regressions in pipeline correctness (the easy problem)

Most test frameworks only solve problem 2. Cortex needs both.

## Test Structure

```
tests/
├── conftest.py              ← shared fixtures (in-memory DB, fixture transcripts)
├── fixtures/
│   ├── transcripts/
│   │   ├── simple.jsonl     ← minimal valid transcript
│   │   ├── with_failures.jsonl  ← bash failures, error signals
│   │   ├── with_decisions.jsonl ← Claude prose with decision sentences
│   │   └── large.jsonl      ← 500+ events, performance test
│   ├── repos/
│   │   └── sample_project/  ← minimal git repo for git_channel tests
│   └── expected_nodes/
│       └── simple.json      ← expected extraction output for simple.jsonl
├── test_parser.py
├── test_extractor.py
├── test_embedder.py
├── test_graph.py
├── test_decay.py
├── test_retrieval.py
├── test_hooks.py            ← integration tests for inject + extract loop
└── test_quality.py          ← memory quality evals (see below)
```

## Coverage Targets

| Module          | Minimum coverage | Notes                          |
|-----------------|------------------|--------------------------------|
| core/parser.py  | 95%              | All event types must be tested |
| core/extractor.py | 90%            | All four channels              |
| core/graph.py   | 95%              | All CRUD + merge paths         |
| core/decay.py   | 95%              | All tier transitions           |
| core/retrieval.py | 90%            | All three channels             |
| core/embedder.py | 85%            | Wrapper, harder to test        |
| hooks/          | 80%             | Integration tests cover rest   |

Enforced via: `pytest --cov=core --cov-fail-under=90`

## Unit Test Rules

- **No file I/O in unit tests.** Use in-memory SQLite (`:memory:`).
- **No network calls in unit tests.** Mock the embedding model with
  `numpy.random.rand(384)` as a dummy vector.
- **No real git repos in unit tests.** Use the fixture sample_project.
- **No mocking of core logic.** Test real behavior. Only mock I/O boundaries.
- Use `pytest.mark.parametrize` aggressively — each event type, each tier
  transition, each dedup case should be a separate parametrized case.

## conftest.py Fixtures

```python
@pytest.fixture
def db() -> Generator[sqlite3.Connection, None, None]:
    """In-memory SQLite connection with schema applied."""
    conn = sqlite3.connect(":memory:")
    schema = Path("schema.sql").read_text()
    conn.executescript(schema)
    yield conn
    conn.close()

@pytest.fixture
def graph(db) -> Graph:
    """Graph instance backed by in-memory DB."""
    return Graph(connection=db)

@pytest.fixture
def simple_transcript() -> list[ParsedEvent]:
    """Pre-parsed events from fixtures/transcripts/simple.jsonl."""
    path = Path("tests/fixtures/transcripts/simple.jsonl")
    return list(parse_transcript(path))

@pytest.fixture
def dummy_embedding() -> np.ndarray:
    """Deterministic dummy 384-dim embedding for tests."""
    rng = np.random.default_rng(seed=42)
    return rng.random(384).astype(np.float32)
```

## Key Test Cases per Module

### test_parser.py

```python
def test_parse_file_write_event(simple_transcript):
    """FILE_WRITE events are extracted with correct path and timestamp."""

def test_parse_bash_failure_captures_exit_code(with_failures_transcript):
    """BASH_FAILURE events include exit_code in data payload."""

def test_repeated_file_writes_detected_as_hotspot(simple_transcript):
    """Files written ≥3 times in one session are flagged as hotspots."""

def test_malformed_jsonl_lines_are_skipped(tmp_path):
    """Malformed JSONL lines produce a warning log, not an exception."""

def test_user_messages_are_excluded(simple_transcript):
    """ParsedEvent list contains no events from user message turns."""
```

### test_graph.py

```python
def test_write_node_creates_correct_schema_fields(graph, dummy_embedding):
    """Written node has all required fields populated correctly."""

def test_merge_node_updates_weight_not_duplicate(graph, dummy_embedding):
    """Merging a node increments weight; does not create a second row."""

def test_find_similar_returns_above_threshold_only(graph, dummy_embedding):
    """find_similar with threshold=0.9 excludes nodes at 0.85 similarity."""

def test_edge_strength_increments_on_co_retrieval(graph, dummy_embedding):
    """Edge strength increments when both nodes are retrieved together."""

def test_cascade_delete_removes_edges(graph, dummy_embedding):
    """Deleting a node removes all its edges via CASCADE."""
```

### test_decay.py

```python
def test_tier1_weight_decays_by_correct_rate(graph):
    """Unaccessed tier-1 node weight = initial * 0.85 after one decay run."""

def test_tier2_promotion_requires_both_conditions(graph):
    """Node promotes only when weight ≥ 20 AND session_count ≥ 8 AND age ≥ 14d."""

def test_tier1_eviction_at_threshold(graph):
    """Tier-1 node with weight < 0.3 is deleted on decay run."""

def test_tier3_nodes_never_decay(graph):
    """Tier-3 node weight is unchanged after 100 simulated decay runs."""

def test_precision_downcasts_on_promotion(graph):
    """Tier-2 node has precision_bits=8 after promotion from tier-1."""
```

### test_retrieval.py

```python
def test_tier3_nodes_always_included(graph):
    """All tier-3 project nodes appear in retrieval result regardless of score."""

def test_vector_channel_uses_cosine_similarity(graph):
    """Vector channel ranks nodes by cosine similarity to query embedding."""

def test_token_budget_enforced(graph):
    """Result set never exceeds 600 tokens as counted by tiktoken."""

def test_graph_hop_expands_from_vector_results(graph):
    """Graph hop channel returns nodes 1–2 edges from top vector results."""

def test_bm25_channel_handles_empty_index(graph):
    """BM25 channel returns empty list gracefully when no nodes exist."""
```

## Memory Quality Evals (test_quality.py)

This is the hard problem. These tests verify that extraction produces *useful*
memory, not just syntactically correct nodes.

### Eval set

`tests/fixtures/expected_nodes/` contains ground-truth JSON files mapping
fixture transcripts to expected extraction outputs.

```json
{
  "transcript": "with_decisions.jsonl",
  "expected_nodes": [
    {
      "type": "fact",
      "source": "nlp",
      "text_contains": ["asyncpg", "chose"],
      "rationale_contains": ["performance"],
      "scope": "project"
    }
  ],
  "should_not_extract": [
    "Let me think about",
    "actually let's not",
    "for now"
  ]
}
```

### Eval tests

```python
def test_decision_sentences_extracted(with_decisions_transcript, graph):
    """Decision sentences produce fact nodes with correct conclusion/rationale split."""
    run_extraction(with_decisions_transcript, graph)
    nodes = graph.get_all_nodes(project=TEST_PROJECT)
    decision_nodes = [n for n in nodes if n.source == "nlp" and n.type == "fact"]
    assert len(decision_nodes) >= 1

def test_retracted_statements_not_extracted(with_retractions_transcript, graph):
    """Statements followed by retractions produce zero nodes."""
    run_extraction(with_retractions_transcript, graph)
    nodes = graph.get_all_nodes(project=TEST_PROJECT)
    assert all("actually" not in n.text for n in nodes)

def test_generic_advice_not_extracted(with_generic_advice_transcript, graph):
    """Generic Python advice not specific to the project is dropped."""
    run_extraction(with_generic_advice_transcript, graph)
    nodes = graph.get_all_nodes(project=TEST_PROJECT)
    assert len(nodes) == 0

def test_injection_volume_within_budget(graph, populated_graph_fixture):
    """inject.py output is always under 600 tokens."""
    injected_text = run_injection(project=TEST_PROJECT)
    token_count = count_tokens(injected_text)
    assert token_count <= 600
```

## Performance Tests

```python
@pytest.mark.slow
def test_extraction_completes_within_400ms(large_transcript, tmp_path):
    """Full extraction pipeline on a 500-event transcript under 400ms."""
    start = time.perf_counter()
    run_extraction(large_transcript, tmp_path / "cortex.db")
    elapsed = time.perf_counter() - start
    assert elapsed < 0.4

@pytest.mark.slow
def test_injection_completes_within_100ms(populated_graph):
    """inject.py completes within 100ms on a graph with 200 nodes."""
    start = time.perf_counter()
    run_injection(project=TEST_PROJECT)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.1
```

Run slow tests with: `pytest -m slow tests/`

## Integration Test: Full Loop

```python
def test_full_extract_inject_loop(tmp_path):
    """
    Simulates a complete session cycle:
    1. Run extract.py on a fixture transcript
    2. Verify nodes written to graph
    3. Run inject.py for the same project
    4. Verify injected text contains expected conventions
    5. Verify token budget respected
    """
```

## CI Pipeline

```yaml
# .github/workflows/test.yml
- name: Lint
  run: ruff check . && black --check .

- name: Type check
  run: mypy --strict core/ hooks/ cli/

- name: Unit tests
  run: pytest tests/ -v --cov=core --cov-fail-under=90 -m "not slow"

- name: Security scan
  run: bandit -r core/ hooks/ cli/ -ll

- name: Slow tests (scheduled only)
  run: pytest tests/ -m slow
  if: github.event_name == 'schedule'
```
