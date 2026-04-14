# Agent Instructions

## Claude Code Specific

1. Project lives on `/media/lin/disk2/brawlstar-agent/` — avoid writing to /home (low space).
2. Use `uv` for Python package management, not pip/conda.
3. If `sudo apt install` is needed, flag it for the user to run manually.

## Python Environment

```bash
cd /media/lin/disk2/brawlstar-agent
uv sync                          # create/update venv
uv run python <script>           # run with venv
PYTHONPATH=src uv run python ...  # if importing from src/
```

<!-- BEGIN CURSOR RULES (auto-synced — do not edit below) -->

<!-- source: .cursor/rules/memory-bank.mdc -->

# Memory Bank Protocol

This project uses a memory-bank system for cross-session continuity.
Memory files live in `memory-bank/` at the project root.

## On Every New Chat / Agent Launch

1. **Read these files first**, before doing any work:
   - `memory-bank/memory-bank.md` (master index and constraints)
   - `memory-bank/progress.md` (what's done, what's next)
   - `memory-bank/activeContext.md` (current focus)
2. Read these if the task touches technical setup or decisions:
   - `memory-bank/techContext.md`
   - `memory-bank/decisions.md`
   - `memory-bank/architecture.md`

## When to Update Memory

Update memory files when any of these happen:
- A milestone is completed → update `progress.md`
- Current focus changes → update `activeContext.md`
- A technical fact is discovered (new package, version, compatibility) → update `techContext.md`
- A significant decision is made → append to `decisions.md`
- Architecture or directory layout changes → update `architecture.md`

## Update Rules

- Keep updates **concise** — append, don't rewrite history
- Use dates in progress entries (YYYY-MM-DD)
- Mark completed items with `[x]` in progress.md
- When a phase transitions, move items between sections in progress.md
- Never delete past decisions — mark superseded ones with `**Superseded by DEC-XXX**`

## Hard Constraints (always enforce)

These are repeated from memory-bank.md for visibility:
1. Everything local on Linux
2. Android environment, not iOS
3. Isolated from personal accounts
4. No live-match botting — research/offline analysis only
5. No cloud streaming unless explicitly approved

<!-- END CURSOR RULES -->
