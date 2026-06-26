.PHONY: check fmt test lint typecheck security install

# Run the full CI suite locally — matches .github/workflows/test.yml exactly
check: lint typecheck test security

lint:
	ruff check .
	black --check --diff .

fmt:
	ruff check --fix .
	black .

typecheck:
	mypy --strict --ignore-missing-imports core/ hooks/ cli/

test:
	pytest tests/ -v --cov=core --cov-fail-under=90 -m "not slow"

security:
	bandit -r core/ hooks/ cli/ -ll

install:
	pip install -e ".[dev]"
	python -m spacy download en_core_web_sm
	pre-commit install
