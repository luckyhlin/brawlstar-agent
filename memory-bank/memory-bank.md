# Brawl Stars AI Research — Memory Bank

> Master index. Read this file first in every new session.

## Project Identity

- **Name**: Brawl Stars AI Research Sandbox
- **Location**: `/media/lin/disk2/brawlstar-agent/`
- **Purpose**: Offline CV/ML research on Brawl Stars gameplay — perception, strategy analysis, dataset generation.
- **Data source**: YouTube gameplay recordings (emulator path failed due to anti-cheat).
- **Scope**: Offline analysis only. NO live-match botting.

## Memory Bank Files

| File | Purpose |
|------|---------|
| `memory-bank.md` | This file — project overview and index |
| `progress.md` | What's done, what's next, session log |
| `activeContext.md` | Current focus and immediate actions |
| `techContext.md` | Machine specs and installed software |
| `decisions.md` | Key decisions with rationale |
| `architecture.md` | Directory layout, pipeline design, module inventory |

## Hard Constraints

1. Everything local on Linux.
2. No personal accounts or personal phone.
3. All data stays local.
4. No live multiplayer automation — research/offline only.

## Project Phases

1. ~~Environment Setup~~ — emulator failed, pivoted to YouTube capture
2. **Data Pipeline** — DONE: download → extract → review → crop → 308 gameplay frames
3. **Perception Baseline** — DONE (partial): OCR works on timer, brawler detection is weak
4. **Brawler Identification** — NEXT: need dedicated pipeline, not heuristics
5. Strategy Analysis — future
6. Coach Overlay — stretch goal
