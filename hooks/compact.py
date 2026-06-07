#!/usr/bin/env python3
"""PostCompact hook — re-inject memory context after Claude Code compacts context."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hooks.inject import main as inject_main

if __name__ == "__main__":
    sys.exit(inject_main())
