# /cortex-status

Show a summary of the current Cortex memory state for this project.

## What this skill does

Runs `cortex status` and displays:
- Node counts by tier (Ephemeral / Semantic / Procedural)
- Last session metadata (nodes written, evicted, tokens saved)
- Database location

## Usage

Type `/cortex-status` in any Claude Code session.

## Example output

```
Cortex — /path/to/project
┌──────┬────────────┬───────┐
│ Tier │ Label      │ Nodes │
├──────┼────────────┼───────┤
│ 1    │ Ephemeral  │    12 │
│ 2    │ Semantic   │     4 │
│ 3    │ Procedural │     2 │
└──────┴────────────┴───────┘

Last session: 1717800000
  Nodes written: 3
  Nodes evicted: 1
  Tokens saved:  284
```

## Implementation

```bash
cortex status
```
