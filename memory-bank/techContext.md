# Technical Context

## Host Machine

| Property | Value |
|----------|-------|
| OS | Ubuntu 22.04.5 LTS |
| Kernel | 6.8.0-52-generic |
| CPU | Intel i7-12700H (14C/20T) |
| RAM | 62 GB |
| GPU | NVIDIA RTX 3060 Mobile, driver 535.230.02, CUDA 12.2 |
| Disk | /media/lin/disk2 — 456GB NVMe, project lives here |
| Home disk | /home — only 27GB free, avoid writing here |

## Software

| Tool | Version |
|------|---------|
| Python (uv venv) | 3.12.13 |
| uv | 0.10.9 |
| OpenCV | 4.13.0 |
| numpy | 2.4.4 |
| yt-dlp | latest |
| tesseract | installed (apt) |
| ffmpeg | 4.4.2 |
| Docker | 28.0.1 |

## Brawl Stars API

| Property | Value |
|----------|-------|
| Base URL | `https://api.brawlstars.com/v1` |
| Auth | Bearer token (JWT, IP-locked, developer/silver tier) |
| Credentials | `api.env` (git-ignored): `BRAWL_STAR_API`, `MAJOR_ACCOUNT_TAG=#RYY9LJVL` |
| Docs | `docs/brawlstars-api.md` (full reference) |
| Example data | `docs/api-examples/` (live responses from 2026-04-13) |
| Brawler count | 101 (IDs 16000000–16000103) |
| Rate limit | Undocumented, throttles aggressive scanning |

Endpoints in use: player profile, battlelog, global rankings, brawler list, brawler detail, game modes, event rotation.

## Emulator Software (installed but not useful for Brawl Stars)

| Tool | Status |
|------|--------|
| Android SDK | cmdline-tools 12.0, platform-tools 37.0.0, emulator 36.4.10 |
| Genymotion | 3.9.0 (no ARM translation) |
| JDK | Temurin 17.0.18+8 (on disk2) |
