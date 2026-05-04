# Decisions

## DEC-001: Emulator — All Paths Failed
- AVD: anti-cheat self-kills Brawl Stars (x86 CPU detected, obfuscated integrity check)
- Genymotion: ARM translation removed in v3.9, game can't install
- Waydroid: needs Wayland + binder/ashmem modules, not viable on X11
- **Outcome**: pivoted to YouTube gameplay recordings

## DEC-002: Project Location
- `/media/lin/disk2/brawlstar-agent/` — 295GB free NVMe
- Avoid /home (27GB) and / (8GB)

## DEC-003: Data Source — YouTube Gameplay
- Download with yt-dlp, extract frames with ffmpeg
- Human review via browser-based hub (auto-label + click to correct)
- Works well. 308 frames from 24 clips.
- Future: physical Android device for live capture if needed

## DEC-004: Brawler Identification — Needs Real Approach
- Color histogram matching against portrait catalog does not work
- Blob detection by saturation is too noisy
- **Next decision needed**: what approach for brawler ID?
  - Option A: train a small classifier (ResNet/MobileNet) on labeled in-game crops
  - Option B: YOLO-style object detector fine-tuned on Brawl Stars frames
  - Option C: contrastive embeddings (portrait → in-game matching)
  - All require labeled training data (brawler crops with identity labels)

## DEC-006: API-Based Battle Analytics Pipeline
- CV-based brawler identification is hard (needs labeled data, training, etc.)
- Official API gives structured battle data directly: who played what brawler, win/loss, mode, map
- Pipeline: rankings → player tags → battlelogs → SQLite → analytics queries
- Storage: SQLite at `data/brawlstars.db` (16MB after 4,628 battles from 200 top players)
- Rate limiting: 1-2 req/s with exponential backoff, conservative to avoid bans
- Snowball discovery: each battlelog yields ~6 new player tags from opponents/teammates
- **First run**: 200 global top players → 4,628 battles → 19,845 discovered players
- Analytics: brawler win rates, team comp win rates, matchup matrix, synergy matrix
- **Outcome**: much faster path to useful insights than CV pipeline

## DEC-005: Dataset Backup Strategy — Deferred
- `datasets/` is gitignored for now (images too large for normal git)
- When a high-quality dataset is ready, we need a proper backup strategy
- Options to evaluate later:
  - Git LFS (track large files in git via pointers)
  - External storage + download script (e.g. tarball on local NAS or cloud, `scripts/fetch-dataset.sh`)
  - DVC (data version control — git-like tracking for large files)
- **No action now** — revisit when dataset quality justifies preservation

## DEC-008: Local-Primary Workflow, Droplet Deploys via Git Pull (2026-05-03)
- **Local machine = source of truth for code**. All edits happen in Cursor on local.
- **Droplet = deploy target**. Receives updates via `git pull`. Never edit code on the droplet.
- Workflow: edit on local → commit → push to GitHub → SSH droplet → `git pull` → restart services if needed.
- Rationale:
  - Droplet has only 1 GB RAM; Cursor remote-SSH would compete with the crawler for memory.
  - Heavy assets (`datasets/`, `capture/`, `emulator/`) live only on local; deliberately NOT synced.
  - Single source of truth eliminates merge/divergence risk between two parallel workspaces.
  - Standard CI/CD-style pattern; future-us understands it instantly.
- Per-machine state stays per-machine and gitignored: `api.env`, `data/brawlstars.db`, `~/.bashrc` env vars.
- Droplet authenticates to GitHub via its own SSH key registered as a deploy key (read-only) on the repo. Generated on-droplet, never copied between machines.
- Migration path documented in `docs/deployment.md` for future VPS moves.

## DEC-007: Hosting — DigitalOcean Droplet, Phase-1 All-in-One (2026-05-03)
- Need: always-on machine for periodic crawling. No always-on home machine available.
- Binding constraint: Brawl Stars API key requires IPv4 whitelist (specific IPs only, no CIDR). Rules out serverless cron (GitHub Actions, Cloudflare Workers, Lambda, Neon `pg_cron`) — all use ephemeral IPs.
- Options compared (May 2026):
  - Oracle Always-Free Ampere — $0/mo, 24 GB RAM, free IPv4 — rejected for now (signup tax, ARM provisioning lottery)
  - GCP e2-micro "free" — actually $3.65/mo because Feb 2024 GCP charges for all in-use external IPv4 ($0.005/hr)
  - AWS Lightsail $5 — viable, predictable bill
  - Hetzner US CX22 $5.60 — best specs/$ (4 GB RAM) but unfamiliar brand
  - **DigitalOcean Basic $6/mo** — 1 GB RAM, 25 GB SSD, included IPv4, US-based, best ecosystem/docs
  - Neon free tier rejected: 0.5 GB cap (DB already 617 MB), `pg_cron` is SQL-only (can't run Python crawler)
- **Decision**: DigitalOcean Basic Droplet, $6/mo, Ubuntu 24.04 LTS, US region.
- **Architecture**: Phase-1 all-in-one. SQLite stays (no migration to Postgres yet — single writer + few readers fits SQLite perfectly). Crawler via systemd timer. Cloudflare R2 for nightly `pg_dump`-equivalent backups (free 10 GB tier). Reserved IP attached to droplet (free while attached) for stable BS API whitelist.
- **Phase-2** (only if/when needed): expose read-only API via Cloudflare Tunnel + Pages frontend. No public Postgres exposure.
- **Phase-3** (only at >50 GB DB or read-heavy public dashboard): partition by month, cold storage to R2 as Parquet, or migrate to Postgres + Neon read replica.
- Lock-in: near zero. Stock Ubuntu + SQLite/Postgres + standard backup formats. Migration to any other VPS is `rsync` + edit one systemd unit.
