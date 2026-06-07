# Contributing to Cortex

## Commit Conventions

All commits must follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]
```

### Types

| Type       | Use when                                      |
|------------|-----------------------------------------------|
| `feat`     | Adding a new feature                          |
| `fix`      | Fixing a bug                                  |
| `chore`    | Build, deps, tooling (no production change)   |
| `docs`     | Documentation only                            |
| `test`     | Adding or updating tests                      |
| `refactor` | Code change that neither fixes nor adds       |
| `perf`     | Performance improvement                       |

### Examples

```
feat(core): add graph.py with SQLite read/write/merge layer
fix(hooks): handle missing transcript path gracefully
test(extractor): add parametrized NLP channel cases
chore(deps): pin sentence-transformers to 2.7.x
```

## Branch Naming

- `feat/<short-description>` — new features
- `fix/<short-description>` — bug fixes
- `chore/<short-description>` — tooling / housekeeping

## Development Flow

1. Fork and clone
2. `pip install -e ".[dev]"`
3. `python -m spacy download en_core_web_sm`
4. `pre-commit install`
5. Create a branch: `git checkout -b feat/your-feature`
6. Write tests first, then implementation
7. Ensure all checks pass: `ruff check . && black --check . && mypy --strict core/ && pytest tests/`
8. Open a PR against `main`

## Code Standards

- Type hints on every function and class attribute
- Google-style docstrings on every public function
- No bare `except:` — always catch specific exceptions
- No ORM — raw SQL only in `core/graph.py`
- No LLM calls in the extraction pipeline
- All hooks must complete in under 500ms
