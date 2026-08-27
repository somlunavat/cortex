#!/usr/bin/env python3
"""Daily automated test-coverage commits for somlunavat/cortex.

Run via GitHub Actions on a daily schedule. Uses the Anthropic API to generate
new test classes, validates them (syntax + lint + pytest), commits each class,
pushes a branch, opens a PR, and enables auto-merge.

Requires:
    ANTHROPIC_API_KEY  — Anthropic API key (GitHub secret)
    GITHUB_TOKEN       — GitHub token (automatic in Actions, write access)
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO = "somlunavat/cortex"
GH_TOKEN = os.environ["GITHUB_TOKEN"]
NOW = _dt.datetime.now(_dt.UTC)
DATE_TAG = NOW.strftime("%Y%m%d")

# Six test areas; we rotate through them so each day hits a fresh set.
TARGETS = [
    {
        "test_file": "tests/test_parser.py",
        "source_file": "core/parser.py",
        "module": "parser",
    },
    {
        "test_file": "tests/test_extractor.py",
        "source_file": "core/extractor.py",
        "module": "extractor",
    },
    {
        "test_file": "tests/test_retrieval.py",
        "source_file": "core/retrieval.py",
        "module": "retrieval",
    },
    {
        "test_file": "tests/test_graph.py",
        "source_file": "core/graph.py",
        "module": "graph",
    },
    {
        "test_file": "tests/test_decay.py",
        "source_file": "core/decay.py",
        "module": "decay",
    },
    {
        "test_file": "tests/test_hooks.py",
        "source_file": "hooks/extract.py",
        "module": "hooks",
    },
]

# ---------------------------------------------------------------------------
# Shell helpers
# ---------------------------------------------------------------------------


def run(cmd: str, cwd: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)


def run_check(cmd: str, cwd: str | None = None) -> subprocess.CompletedProcess[str]:
    result = run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {cmd}\nstdout: {result.stdout[:400]}\nstderr: {result.stderr[:400]}"
        )
    return result


# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------


def _gh(method: str, path: str, body: dict | None = None) -> dict:
    data_flag = f"-d '{json.dumps(body)}'" if body else ""
    result = run(
        f"curl -sf -X {method} "
        f'-H "Authorization: token {GH_TOKEN}" '
        f'-H "Content-Type: application/json" '
        f"{data_flag} "
        f'"https://api.github.com/repos/{REPO}/{path}"'
    )
    if not result.stdout.strip():
        return {}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}


def create_pr(branch: str, title: str, body: str) -> int | None:
    resp = _gh(
        "POST", "pulls", {"title": title, "body": body, "head": branch, "base": "main"}
    )
    return resp.get("number")


def enable_automerge(pr_number: int) -> None:
    """Enable squash auto-merge via GraphQL (best-effort)."""
    resp = _gh("GET", f"pulls/{pr_number}")
    node_id = resp.get("node_id", "")
    if not node_id:
        return
    mutation = (
        "mutation($id:ID!){enablePullRequestAutoMerge"
        "(input:{pullRequestId:$id,mergeMethod:SQUASH})"
        "{pullRequest{number}}}"
    )
    run(
        f"curl -sf -X POST "
        f'-H "Authorization: token {GH_TOKEN}" '
        f'-H "Content-Type: application/json" '
        f'-d \'{json.dumps({"query": mutation, "variables": {"id": node_id}})}\' '
        f'"https://api.github.com/graphql"'
    )


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------


def extract_class_code(raw: str) -> str:
    """Strip markdown fences and return just the class definition."""
    raw = re.sub(r"```python\s*", "", raw)
    raw = re.sub(r"```\s*", "", raw)
    raw = raw.strip()
    m = re.search(r"^class \w+", raw, re.MULTILINE)
    return raw[m.start() :] if m else raw


def generate_test_class(target: dict, class_suffix: str) -> str:
    client = anthropic.Anthropic()
    test_content = Path(target["test_file"]).read_text()
    source_content = Path(target["source_file"]).read_text()
    module = target["module"]
    class_name = f"Test{module.title()}{class_suffix}"

    # Provide imports and last 60 lines as style context
    header_lines = [
        ln
        for ln in test_content.splitlines()
        if ln.startswith(("import ", "from ", "def _"))
    ]
    header = "\n".join(header_lines[:40])
    tail = "\n".join(test_content.splitlines()[-60:])

    prompt = f"""Add a new pytest test class to an existing Python test file.
Return ONLY the raw Python class code — no explanation, no markdown fences.

Rules:
- Class name: {class_name}
- Exactly 6 test methods, each with a clear assert
- Use ONLY imports already in the file (shown below)
- Use ONLY helper functions already defined (e.g. _make_event, _make_node, _write_jsonl)
- No new top-level imports
- Match style exactly: type annotations, spacing, naming
- Target edge cases not covered by existing tests in the file

Source file {target["source_file"]} (first 2500 chars):
{source_content[:2500]}

Existing imports / helpers:
{header}

Last 60 lines of test file (style reference):
{tail}

Return the class definition starting with `class {class_name}:` and nothing else."""

    msg = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    return extract_class_code(msg.content[0].text)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_and_commit(
    test_file: str,
    original_content: str,
    new_code: str,
    module: str,
    class_name: str,
) -> None:
    """Append new_code to test_file, lint, run tests, commit. Raises on failure."""
    Path(test_file).write_text(original_content + "\n\n" + new_code + "\n")

    # Syntax check
    r = run(f"python3 -m py_compile {test_file}")
    if r.returncode != 0:
        Path(test_file).write_text(original_content)
        raise ValueError(f"Syntax error: {r.stderr[:300]}")

    # Format
    run(f"python3 -m black {test_file} -q")
    r = run(f"python3 -m ruff check {test_file} --fix -q")
    if r.returncode != 0:
        Path(test_file).write_text(original_content)
        raise ValueError(f"Ruff: {r.stdout[:200]}")

    # Run just the new class
    r = run(f"python3 -m pytest {test_file}::{class_name} -q --tb=short -x")
    if r.returncode != 0:
        Path(test_file).write_text(original_content)
        raise ValueError(f"Tests failed:\n{r.stdout[:400]}")

    # Commit
    run_check(f"git add {test_file}")
    run_check(f'git commit -m "tests({module}): add {class_name} coverage"')


# ---------------------------------------------------------------------------
# Per-target workflow
# ---------------------------------------------------------------------------


def process_target(target: dict, pr_index: int) -> bool:
    """Create branch, add 3 test-class commits, push, open PR. Return True on success."""
    module = target["module"]
    branch = f"test/{module}-auto-{DATE_TAG}-{pr_index}"
    test_file = target["test_file"]

    # Branch off latest main
    try:
        run_check("git checkout main")
        run_check("git pull origin main")
        run_check(f"git checkout -b {branch}")
    except RuntimeError as e:
        print(f"[{module}] Branch setup failed: {e}", file=sys.stderr)
        return False

    commits_made = 0
    for i in range(3):
        suffix = f"{DATE_TAG}{chr(65 + i)}"  # e.g. 20260826A / B / C
        class_name = f"Test{module.title()}{suffix}"
        current_content = Path(test_file).read_text()
        try:
            raw = generate_test_class(target, suffix)
            validate_and_commit(test_file, current_content, raw, module, class_name)
            commits_made += 1
            print(f"[{module}] commit {i+1}/3 ok ({class_name})")
        except Exception as e:
            print(f"[{module}] commit {i+1}/3 skipped: {e}", file=sys.stderr)
            # Restore to last committed state
            run("git reset HEAD " + test_file + " 2>/dev/null || true")
            run("git checkout -- " + test_file + " 2>/dev/null || true")

    if commits_made == 0:
        run("git checkout main")
        run(f"git branch -D {branch}")
        return False

    # Push
    r = run(f"git push -u origin {branch}")
    if r.returncode != 0:
        print(f"[{module}] Push failed: {r.stderr[:200]}", file=sys.stderr)
        return False

    # Open PR
    pr_number = create_pr(
        branch=branch,
        title=f"tests({module}): automated coverage additions {DATE_TAG}",
        body=(
            f"Automated daily test additions for `{module}`.\n\n"
            f"- {commits_made} new test class(es), each with 6 assertions\n"
            f"- Validated: syntax + black + ruff + pytest before each commit\n\n"
            f"## Test plan\n- [ ] CI passes (auto-merge enabled)"
        ),
    )
    if pr_number:
        enable_automerge(pr_number)
        print(f"[{module}] PR #{pr_number} opened, auto-merge enabled")
    else:
        print(f"[{module}] PR create returned no number", file=sys.stderr)

    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    # Pick 4 targets, rotating by day-of-year so we cover all areas over time
    day_of_year = NOW.timetuple().tm_yday
    start = day_of_year % len(TARGETS)
    selected = [TARGETS[(start + i) % len(TARGETS)] for i in range(4)]

    prs_opened = 0
    for i, target in enumerate(selected):
        print(f"\n=== {target['module']} (PR {i+1}/4) ===")
        if process_target(target, i):
            prs_opened += 1

    print(f"\nDone: {prs_opened}/4 PRs opened.")
    if prs_opened == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
