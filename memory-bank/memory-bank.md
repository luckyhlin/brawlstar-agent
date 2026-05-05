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
2. ~~Data Pipeline~~ — DONE: download → extract → review → crop → 308 gameplay frames
3. ~~Perception Baseline~~ — DONE (partial): OCR works on timer, brawler detection is weak
4. ~~API Battle Analytics Pipeline~~ — DONE (DEC-006): SQLite + collector + analytics queries; ~200k battles, 553k tags
5. ~~Production Deploy~~ — DONE (DEC-007/008/009): always-on DigitalOcean droplet with 3 systemd timers (bulk crawl + pinned tags + analytics precompute), local-primary git workflow, dashboard reads precomputed cache
6. **Brawler-pick recommendation model** — NEXT: ML on collected battle data. Start with `docs/analytics-notes.md`. **Read the team-result bug caveat before training.**
7. Brawler Identification (CV) — deferred; API gives structured data, no need for visual ID right now
8. Strategy Analysis / Coach Overlay — stretch goal

## For agents starting a new session

If you're picking up the project, read in this order:
1. `memory-bank/memory-bank.md` (this file) — overview
2. `memory-bank/progress.md` — what's done, session log, current phase
3. `memory-bank/activeContext.md` — current focus + next steps
4. `memory-bank/decisions.md` — DEC-001..009, why we chose what we chose
5. If touching infrastructure: `memory-bank/techContext.md` + `docs/deployment.md`
6. **If doing analytics or model training: `docs/analytics-notes.md` is required reading** — there are non-obvious data caveats (especially the team-result bug fix not being backfilled) that will silently corrupt a trained model.
